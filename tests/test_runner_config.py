"""Tests for llm_kit/runner_config.py's RunnerConfig."""
from __future__ import annotations

from llm_kit.llm_runtime import GenerationConfig
from llm_kit.llm_setup import BaseConfig
from llm_kit.prompt_builder import PromptingConfig
from llm_kit.runner_config import RunnerConfig


def test_chat_template_kwargs_syncs_from_generation_to_prompt():
    config = RunnerConfig(generation=GenerationConfig(chat_template_kwargs={"enable_thinking": False}))

    assert config.prompt.chat_template_kwargs == {"enable_thinking": False}


def test_chat_template_kwargs_syncs_from_prompt_to_generation():
    config = RunnerConfig(prompt=PromptingConfig(chat_template_kwargs={"enable_thinking": False}))

    assert config.generation.chat_template_kwargs == {"enable_thinking": False}


def test_chat_template_kwargs_untouched_when_neither_side_set():
    config = RunnerConfig()

    assert config.prompt.chat_template_kwargs == {}
    assert config.generation.chat_template_kwargs == {}


def test_chat_template_kwargs_left_alone_when_both_sides_set_independently():
    config = RunnerConfig(
        generation=GenerationConfig(chat_template_kwargs={"a": 1}),
        prompt=PromptingConfig(chat_template_kwargs={"b": 2}),
    )

    assert config.generation.chat_template_kwargs == {"a": 1}
    assert config.prompt.chat_template_kwargs == {"b": 2}


def test_chat_template_kwargs_sync_does_not_alias_the_dict():
    config = RunnerConfig(generation=GenerationConfig(chat_template_kwargs={"enable_thinking": False}))

    config.prompt.chat_template_kwargs["extra"] = True

    assert "extra" not in config.generation.chat_template_kwargs


def test_to_chat_completions_delegates_to_generation_seeded_from_base():
    config = RunnerConfig(base=BaseConfig(seed=7), generation=GenerationConfig(temperature=0.5))

    params = config.to_chat_completions()

    assert params["seed"] == 7
    assert params["temperature"] == 0.5
