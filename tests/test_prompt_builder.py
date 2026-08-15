"""Tests for llm_kit/prompt_builder.py.

resolvers/filters are looked up in whatever resolver_registry/
filter_registry the caller passes into PromptBuilder's constructor
(default: empty dicts) - the module itself carries no registry and no
sibling-module imports, so it stays usable as a standalone file.
"""
from __future__ import annotations

import ast
import inspect

import pytest
from jinja2.exceptions import TemplateAssertionError, UndefinedError

from llm_kit.prompt_builder import BlockSpec, PromptBuilder, PromptingConfig


class _FakeTokenizer:
    def tokenize(self, text):
        return text.split()


def _write_block(blocks_dir, name, version, content):
    block_dir = blocks_dir / name
    block_dir.mkdir(parents=True, exist_ok=True)
    (block_dir / f"{version}.j2").write_text(content)


def test_prompt_builder_module_has_no_sibling_module_imports():
    """Regression guard: prompt_builder.py must not import any other
    llm_kit module - it's meant to be usable as a single standalone file,
    with resolvers/filters injected rather than looked up in a shared
    registry module."""
    import llm_kit.prompt_builder as module

    tree = ast.parse(inspect.getsource(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    sibling_imports = {m for m in imported if m.startswith("llm_kit")}
    assert not sibling_imports, f"prompt_builder.py imports other llm_kit modules: {sibling_imports}"


def test_builds_without_any_registry_when_config_uses_no_resolvers_or_filters(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello, {{ name }}!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100,
                              join_format="plain")

    builder = PromptBuilder(config, _FakeTokenizer())
    result = builder.build(task=None, context={"name": "world"})

    assert result == "Hello, world!"


def test_resolver_registry_is_used_when_provided(tmp_path):
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["dynamic"], token_limit=100,
                              resolvers=["dynamic"], join_format="plain")

    def my_resolver(task, remaining_tokens, context, builder):
        return f"resolved: {task}"

    builder = PromptBuilder(config, _FakeTokenizer(), resolver_registry={"dynamic": my_resolver})
    result = builder.build(task="my-task", context={})

    assert result == "resolved: my-task"


def test_filter_registry_is_used_when_provided(tmp_path):
    _write_block(tmp_path, "shout", "v1", "{{ text | shout }}")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["shout"], token_limit=100,
                              filters=["shout"], join_format="plain")

    builder = PromptBuilder(config, _FakeTokenizer(), filter_registry={"shout": str.upper})
    result = builder.build(task=None, context={"text": "hi"})

    assert result == "HI"


def test_unregistered_resolver_name_raises_keyerror(tmp_path):
    """No registry passed -> empty default -> a resolver name not found in
    it should fail loudly (KeyError), not silently skip or crash later
    with a confusing error deep inside build()."""
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["missing"], token_limit=100,
                              resolvers=["missing"])

    with pytest.raises(KeyError):
        PromptBuilder(config, _FakeTokenizer())


def test_required_context_keys_lists_variables_a_template_block_reads(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello, {{ name }}! Today is {{ day }}.")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100)

    builder = PromptBuilder(config, _FakeTokenizer())

    assert builder.required_context_keys() == {"greeting": ["day", "name"]}


def test_required_context_keys_skips_resolver_driven_blocks(tmp_path):
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["dynamic"], token_limit=100,
                              resolvers=["dynamic"])
    builder = PromptBuilder(config, _FakeTokenizer(), resolver_registry={"dynamic": lambda *a: "x"})

    assert builder.required_context_keys() == {}


def test_render_block_missing_context_raises_undefinederror_listing_missing_keys(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello, {{ person }}! Today is {{ day }}.")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100)
    builder = PromptBuilder(config, _FakeTokenizer())

    with pytest.raises(UndefinedError) as exc_info:
        builder.render_block("greeting", person="world")

    message = str(exc_info.value)
    assert "day" in message
    assert "required_context_keys()" in message


def test_build_missing_context_raises_undefinederror_listing_missing_keys(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello, {{ name }}!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100)
    builder = PromptBuilder(config, _FakeTokenizer())

    with pytest.raises(UndefinedError) as exc_info:
        builder.build(task=None, context={})

    assert "name" in str(exc_info.value)


def test_build_missing_filter_raises_templateassertionerror_pointing_at_config_filters(tmp_path):
    _write_block(tmp_path, "shout", "v1", "{{ text | shout }}")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["shout"], token_limit=100)
    builder = PromptBuilder(config, _FakeTokenizer(), filter_registry={"shout": str.upper})

    with pytest.raises(TemplateAssertionError) as exc_info:
        builder.build(task=None, context={"text": "hi"})

    assert "config.filters" in str(exc_info.value)


def test_build_resolver_undefinederror_is_annotated_with_the_resolver(tmp_path):
    def broken_resolver(task, remaining_tokens, context, builder):
        raise UndefinedError("'input_grid' is undefined")

    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["dynamic"], token_limit=100,
                              resolvers=["dynamic"])
    builder = PromptBuilder(config, _FakeTokenizer(), resolver_registry={"dynamic": broken_resolver})

    with pytest.raises(UndefinedError) as exc_info:
        builder.build(task=None, context={})

    message = str(exc_info.value)
    assert "'dynamic' resolver" in message
    assert "broken_resolver" in message


def test_block_overrides_from_config_is_used_by_default(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello, {{ name }}!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100,
                              block_overrides={"greeting": "Hi there."}, join_format="plain")
    builder = PromptBuilder(config, _FakeTokenizer())

    result = builder.build(task=None, context={})

    assert result == "Hi there."


def test_explicit_overrides_param_wins_over_config_block_overrides(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello, {{ name }}!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100,
                              block_overrides={"greeting": "from config"}, join_format="plain")
    builder = PromptBuilder(config, _FakeTokenizer())

    result = builder.build(task=None, context={}, overrides={"greeting": "from call"})

    assert result == "from call"


class _FakeChatTokenizer(_FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.last_kwargs = kwargs
        return "|".join(m["content"] for m in messages)


def test_chat_template_kwargs_reach_apply_chat_template(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100,
                              chat_template="whatever", chat_template_kwargs={"enable_thinking": False})
    tokenizer = _FakeChatTokenizer()
    builder = PromptBuilder(config, tokenizer)

    builder.build(task=None, context={})

    assert tokenizer.last_kwargs == {"enable_thinking": False}


def test_xml_join_format_wraps_every_block_even_without_an_explicit_tag(tmp_path):
    """Regression test: _join used to only wrap a block when its BlockSpec
    had an explicit .tag set - but blocks are configured as plain name
    strings almost everywhere (block_overrides, block names as raw str in
    `blocks`), which BlockSpec.parse leaves with tag=None. join_format
    therefore had no visible effect for the common case: config asked for
    XML and got plain concatenated text back."""
    _write_block(tmp_path, "greeting", "v1", "Hello!")
    _write_block(tmp_path, "farewell", "v1", "Bye!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting", "farewell"],
                              token_limit=100, join_format="xml")
    builder = PromptBuilder(config, _FakeTokenizer())

    result = builder.build(task=None, context={})

    assert result == "<GREETING>\nHello!\n</GREETING>\n<FAREWELL>\nBye!\n</FAREWELL>"


def test_md_join_format_wraps_every_block_even_without_an_explicit_tag(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"],
                              token_limit=100, join_format="md")
    builder = PromptBuilder(config, _FakeTokenizer())

    result = builder.build(task=None, context={})

    assert result == "## GREETING\n\nHello!\n\n---"


def test_plain_join_format_leaves_blocks_unwrapped(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello!")
    _write_block(tmp_path, "farewell", "v1", "Bye!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting", "farewell"],
                              token_limit=100, join_format="plain")
    builder = PromptBuilder(config, _FakeTokenizer())

    result = builder.build(task=None, context={})

    assert result == "Hello!\nBye!"


def test_explicit_block_tag_overrides_the_uppercased_name(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=[BlockSpec(name="greeting", tag="HELLO")],
                              token_limit=100, join_format="xml")
    builder = PromptBuilder(config, _FakeTokenizer())

    result = builder.build(task=None, context={})

    assert result == "<HELLO>\nHello!\n</HELLO>"


def test_xml_join_format_wraps_blocks_under_a_chat_template_too(tmp_path):
    """Same _join code path exists twice - once for the plain-string
    return, once for building chat messages before apply_chat_template -
    the tag-wrapping bug affected both identically."""
    _write_block(tmp_path, "greeting", "v1", "Hello!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100,
                              join_format="xml", chat_template="whatever")
    tokenizer = _FakeChatTokenizer()
    builder = PromptBuilder(config, tokenizer)

    result = builder.build(task=None, context={})

    assert result == "<GREETING>\nHello!\n</GREETING>"


def test_assistant_prefix_appends_to_a_plain_joined_prompt(tmp_path):
    """Regression test: assistant_prefix used to be silently dropped
    whenever chat_template wasn't set - a plain completion model can
    still be primed the same way: ending the prompt with the desired
    continuation's start."""
    _write_block(tmp_path, "greeting", "v1", "Hello!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100,
                              join_format="plain", assistant_prefix="1,1:\n")
    builder = PromptBuilder(config, _FakeTokenizer())

    result = builder.build(task=None, context={})

    assert result == "Hello!\n1,1:\n"


def test_build_assistant_prefix_param_overrides_config(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100,
                              join_format="plain", assistant_prefix="from-config")
    builder = PromptBuilder(config, _FakeTokenizer())

    result = builder.build(task=None, context={}, assistant_prefix="from-call")

    assert result == "Hello!\nfrom-call"


def test_assistant_prefix_reaches_the_assistant_turn_under_chat_template(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100,
                              join_format="plain", chat_template="whatever", assistant_prefix="1,1:\n")
    tokenizer = _FakeChatTokenizer()
    builder = PromptBuilder(config, tokenizer)

    result = builder.build(task=None, context={})

    # _FakeChatTokenizer.apply_chat_template joins each message's content
    # with "|" - the assistant turn carrying assistant_prefix has to be
    # the last message.
    assert result == "Hello!|1,1:\n"
