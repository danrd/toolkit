"""Putting blocks into a module's prompt: where they go, and that nothing else about the builder changes."""
from types import SimpleNamespace

from agent_kit.blocks import install_blocks, with_blocks
from llm_kit.prompt_builder import ApproxTokenizer, PromptBuilder, PromptingConfig


def config(blocks):
    return PromptingConfig(blocks=blocks, token_limit=1000)


def test_missing_blocks_go_before_output_format_and_present_ones_are_not_doubled():
    wider = with_blocks(config(["task", "output_format"]), ["memory", "task"])
    assert wider.blocks == ["task", "memory", "output_format"]
    unchanged = config(["task", "memory"])
    assert with_blocks(unchanged, ["memory"]) is unchanged


def test_without_an_output_format_they_go_last():
    assert with_blocks(config(["task"]), ["memory"]).blocks == ["task", "memory"]


def test_installing_rebuilds_the_builder_keeping_its_resolvers_and_filters():
    resolvers = {"memory": lambda task, budget, context, builder: "remembered"}
    filters = {"shout": lambda text: text.upper()}
    builder = PromptBuilder(config(["task", "output_format"]), ApproxTokenizer(),
                            resolver_registry=resolvers, filter_registry=filters)
    module = SimpleNamespace(builder=builder)

    install_blocks(module, ["memory"])

    assert module.builder is not builder
    assert module.builder.config.blocks == ["task", "memory", "output_format"]
    assert module.builder.resolver_registry is resolvers and module.builder.filter_registry is filters


def test_a_module_that_has_the_blocks_keeps_its_builder():
    builder = PromptBuilder(config(["task", "memory"]), ApproxTokenizer())
    module = SimpleNamespace(builder=builder)
    install_blocks(module, ["memory"])
    assert module.builder is builder
