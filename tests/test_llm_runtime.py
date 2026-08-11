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
