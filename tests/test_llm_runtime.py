"""Tests for llm_kit/llm_runtime.py's GenerationConfig.chat_template_kwargs."""
from __future__ import annotations

from llm_kit.llm_runtime import GenerationConfig


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
