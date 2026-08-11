"""Tests for llm_kit/llm_runtime.py's GenerationConfig and ServerRunner."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict

from llm_kit.llm_runtime import GenerationConfig, ServerRunner


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
