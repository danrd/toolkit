"""Tests for llm_kit/llm_runtime.py's GenerationConfig and ServerRunner."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from llm_kit.llm_runtime import (AllModelsFailedError, CometApiRunner, GenerationConfig, HostedRunner, OpenRouterRunner,
                                 ServerRunner, build_hosted_runner, build_openrouter_runner)


def test_to_chat_completions_omits_extra_body_by_default():
    params = GenerationConfig().to_chat_completions(seed=42)

    assert "extra_body" not in params


def test_to_chat_completions_forwards_chat_template_kwargs_as_extra_body():
    config = GenerationConfig(chat_template_kwargs={"enable_thinking": False})

    params = config.to_chat_completions(seed=42)

    assert params["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_to_chat_completions_sends_repetition_penalty_and_top_k_via_extra_body():
    """repetition_penalty and top_k aren't part of the official Chat
    Completions schema - the openai client's chat.completions.create()
    raises TypeError on unrecognized top-level kwargs before a request is
    even sent, so vLLM's vendor extensions have to travel via extra_body
    instead, same as chat_template_kwargs."""
    config = GenerationConfig(repetition_penalty=1.2, top_k=40)

    params = config.to_chat_completions(seed=42)

    assert "repetition_penalty" not in params
    assert "top_k" not in params
    assert params["extra_body"] == {"repetition_penalty": 1.2, "top_k": 40}


def test_to_chat_completions_forwards_grammar_as_extra_body():
    """Raw GBNF text, unlike to_llama_cpp - only llama-cpp-python's server
    accepts this as a vendor extension field; there's no client-side
    compiling to do. Defaults to the llama_cpp key name."""
    config = GenerationConfig(grammar='root ::= "a"')

    params = config.to_chat_completions(seed=42)

    assert params["extra_body"]["grammar"] == 'root ::= "a"'


def test_to_chat_completions_uses_guided_grammar_key_for_vllm_backend():
    """llama.cpp-server and vllm-serve accept a raw grammar under different
    extra_body key names - the caller (llm_setup, which just started one or
    the other) says which, rather than this method guessing or sending
    both keys speculatively."""
    config = GenerationConfig(grammar='root ::= "a"')

    params = config.to_chat_completions(seed=42, grammar_backend="vllm")

    assert params["extra_body"]["guided_grammar"] == 'root ::= "a"'
    assert "grammar" not in params["extra_body"]


def test_to_chat_completions_rejects_an_unknown_grammar_backend():
    config = GenerationConfig(grammar='root ::= "a"')

    with pytest.raises(ValueError):
        config.to_chat_completions(seed=42, grammar_backend="bogus")


def test_to_chat_completions_ignores_grammar_backend_when_no_grammar_is_set():
    """An unrecognized grammar_backend only matters once there's an actual
    grammar to route - no grammar set means nothing to raise about."""
    config = GenerationConfig()

    params = config.to_chat_completions(seed=42, grammar_backend="bogus")

    assert "extra_body" not in params


def test_to_llama_cpp_omits_grammar_by_default():
    params = GenerationConfig().to_llama_cpp(seed=42)

    assert "grammar" not in params


def test_to_llama_cpp_compiles_grammar_to_a_llama_grammar_instance():
    from llama_cpp import LlamaGrammar

    config = GenerationConfig(grammar='root ::= "a"')

    params = config.to_llama_cpp(seed=42)

    assert isinstance(params["grammar"], LlamaGrammar)


class _FakeOpenAI:
    """Captures the kwargs openai.OpenAI(...) was constructed with, without
    requiring the openai package to actually be installed in this test env."""
    last_kwargs: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        _FakeOpenAI.last_kwargs = kwargs


def test_server_runner_client_uses_configured_request_timeout(monkeypatch):
    """CPU inference of a large model can run well past the openai client's
    own default timeout (600s) on a single request, before anything is
    actually wrong - this needs to be a config-level knob, not stuck at
    the library default."""
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    runner = ServerRunner(process=None, port=8001, model_name="fake", generation_kwargs={},
                           request_timeout=120.0)

    _ = runner.client

    assert _FakeOpenAI.last_kwargs["timeout"] == 120.0


def test_server_runner_client_defaults_request_timeout_to_600s(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    runner = ServerRunner(process=None, port=8001, model_name="fake", generation_kwargs={})

    _ = runner.client

    assert _FakeOpenAI.last_kwargs["timeout"] == 600.0


def test_build_openrouter_runner_reads_models_and_settings_from_llm_config(monkeypatch):
    """The hosted counterpart to llm_setup.build_runner() - constructs an
    OpenRouterRunner from config.llm's OpenRouter fields rather than a
    separate config object, so a hosted and a local module can share one
    ExperimentConfig without mutating it (see SubsymbolicModule's
    runner= injection in llm_kit's consuming projects)."""
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    llm = SimpleNamespace(openrouter_models=["model/a", "model/b"],
                           openrouter_max_retries=5, openrouter_request_timeout=12.0)
    config = SimpleNamespace(llm=llm, to_chat_completions=lambda: {"temperature": 0.0})

    runner = build_openrouter_runner(config)

    assert isinstance(runner, OpenRouterRunner)
    assert runner.models == ["model/a", "model/b"]
    assert runner.generation_kwargs == {"temperature": 0.0}
    assert _FakeOpenAI.last_kwargs["max_retries"] == 5
    assert _FakeOpenAI.last_kwargs["timeout"] == 12.0


class _DeadProcess:
    """A subprocess.Popen stand-in that has already exited."""
    returncode = 137

    def poll(self):
        return self.returncode


class _LiveProcess:
    def poll(self):
        return None  # still running


class _RefusingClient:
    """Fails the way the openai client does when nothing is listening on
    the port."""

    def __init__(self):
        def refuse(**kwargs):
            raise ConnectionError("[Errno 111] Connection refused")

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=refuse))


def test_generate_explains_a_server_that_exited(monkeypatch):
    """Regression test for the notebook restart case: interrupting the
    cell sends the interrupt to the whole process group, killing the
    server while this runner object survives. Re-running then failed deep
    inside httpx with a bare "Connection refused" - naming neither the
    dead server nor the way out."""
    runner = ServerRunner(process=_DeadProcess(), port=8001, model_name="fake",
                           generation_kwargs={}, client=_RefusingClient())

    with pytest.raises(RuntimeError) as excinfo:
        runner.generate("hello")

    message = str(excinfo.value)
    assert "8001" in message
    assert "137" in message                    # the exit code it actually died with
    assert "build_runner" in message           # what to do about it
    assert isinstance(excinfo.value.__cause__, ConnectionError)  # original kept


def test_generate_reraises_untouched_when_the_server_is_alive():
    """A live server refusing one request is a different problem - it must
    not be relabelled as "the server is gone"."""
    runner = ServerRunner(process=_LiveProcess(), port=8001, model_name="fake",
                           generation_kwargs={}, client=_RefusingClient())

    with pytest.raises(ConnectionError):
        runner.generate("hello")


def test_generate_reraises_untouched_for_an_externally_managed_server():
    """process=None means the server isn't ours to diagnose - there's no
    exit code to report, so the original error stands."""
    runner = ServerRunner(process=None, port=8001, model_name="fake",
                           generation_kwargs={}, client=_RefusingClient())

    with pytest.raises(ConnectionError):
        runner.generate("hello")


# ---------------------------------------------------------------------------
# HostedRunner: one fallback loop, the service is a base_url and a key variable
# ---------------------------------------------------------------------------

class _RateLimit(Exception):
    pass


class _ApiError(Exception):
    pass


def _fake_openai_with_errors(monkeypatch):
    module = SimpleNamespace(OpenAI=_FakeOpenAI, RateLimitError=_RateLimit, APIError=_ApiError,
                              APITimeoutError=TimeoutError, APIConnectionError=ConnectionError)
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setattr("llm_kit.llm_runtime.time.sleep", lambda seconds: None)


class _ScriptedClient:
    """chat.completions.create answers per model: a string is a reply, an exception is raised."""

    def __init__(self, script):
        self.script, self.asked = script, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kwargs):
        self.asked.append(model)
        outcome = self.script[model]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


def test_the_next_model_is_tried_when_one_fails(monkeypatch):
    _fake_openai_with_errors(monkeypatch)
    client = _ScriptedClient({"a": _RateLimit("slow down"), "b": _ApiError("boom"), "c": "the reply"})
    runner = HostedRunner(["a", "b", "c"], {}, client=client, base_url="https://x/v1")

    assert runner.generate("p") == "the reply"
    assert client.asked == ["a", "b", "c"]


def test_every_model_failing_raises_and_names_the_service(monkeypatch):
    _fake_openai_with_errors(monkeypatch)
    client = _ScriptedClient({"a": _ApiError("boom"), "b": _ApiError("bang")})

    with pytest.raises(AllModelsFailedError, match=r"All 2 CometAPI models failed.*bang"):
        CometApiRunner(["a", "b"], {}, client=client).generate("p")
    with pytest.raises(AllModelsFailedError, match="All 2 OpenRouter models failed"):
        OpenRouterRunner(["a", "b"], {}, client=client).generate("p")


def test_each_service_has_its_own_address_and_key_variable(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k1")
    monkeypatch.setenv("COMETAPI_KEY", "k2")

    OpenRouterRunner(["m"], {})
    assert _FakeOpenAI.last_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert _FakeOpenAI.last_kwargs["api_key"] == "k1"
    CometApiRunner(["m"], {})
    assert _FakeOpenAI.last_kwargs["base_url"] == "https://api.cometapi.com/v1"
    assert _FakeOpenAI.last_kwargs["api_key"] == "k2"


def test_a_missing_key_is_reported_by_the_variable_that_should_hold_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.delenv("COMETAPI_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="COMETAPI_KEY is not set"):
        CometApiRunner(["m"], {})
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
        OpenRouterRunner(["m"], {})


def test_a_runner_without_an_address_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    with pytest.raises(ValueError, match="needs a base_url"):
        HostedRunner(["m"], {}, api_key="k")


def test_any_openai_compatible_service_is_reached_by_its_address_and_variable(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("MY_SERVICE_KEY", "k3")

    HostedRunner(["m"], {}, base_url="https://my.service/v1", api_key_env="MY_SERVICE_KEY")

    assert _FakeOpenAI.last_kwargs["base_url"] == "https://my.service/v1"
    assert _FakeOpenAI.last_kwargs["api_key"] == "k3"


def _hosted_config(**llm_fields):
    llm = SimpleNamespace(openrouter_models=["m/a"], openrouter_max_retries=3, openrouter_request_timeout=9.0,
                           **llm_fields)
    return SimpleNamespace(llm=llm, to_chat_completions=lambda: {"temperature": 0.0})


def test_build_hosted_runner_picks_the_service_the_config_names(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k1")
    monkeypatch.setenv("COMETAPI_KEY", "k2")

    assert type(build_hosted_runner(_hosted_config())) is OpenRouterRunner          # no field: as before
    assert type(build_hosted_runner(_hosted_config(hosted_provider="openrouter"))) is OpenRouterRunner
    comet = build_hosted_runner(_hosted_config(hosted_provider="CometAPI"))
    assert type(comet) is CometApiRunner and comet.models == ["m/a"]
    assert _FakeOpenAI.last_kwargs["max_retries"] == 3 and _FakeOpenAI.last_kwargs["timeout"] == 9.0


def test_build_hosted_runner_lets_the_config_replace_the_address_and_variable(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OTHER_KEY", "k4")

    build_hosted_runner(_hosted_config(hosted_provider="cometapi", hosted_base_url="https://proxy/v1",
                                        hosted_api_key_env="OTHER_KEY"))

    assert _FakeOpenAI.last_kwargs["base_url"] == "https://proxy/v1" and _FakeOpenAI.last_kwargs["api_key"] == "k4"


def test_an_unknown_provider_is_refused_with_the_known_ones_listed():
    with pytest.raises(ValueError, match="unknown hosted_provider 'nope'; known: cometapi, openrouter"):
        build_hosted_runner(_hosted_config(hosted_provider="nope"))
