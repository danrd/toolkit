"""Small aggregating config for the common case of building both a runner
(via llm_kit.llm_setup.build_runner) and a PromptBuilder from one object,
instead of wiring LlmConfig/GenerationConfig/PromptingConfig separately.
build_runner() already expects its `config` argument to expose `.base`,
`.generation`, and `.to_llama_cpp()`/`.to_vllm()`/`.to_hf()`/
`.to_chat_completions()` (each delegating to `generation`'s own method,
seeded from `base.seed`) - RunnerConfig is that object, so callers don't
have to hand-roll one.

Not a requirement - build_runner()/PromptBuilder() still take their own
configs directly if that's more convenient for a given caller. RunnerConfig
exists for the specific case where a setting has to be known to BOTH:
generation.chat_template_kwargs (e.g. Qwen3's enable_thinking=False) needs
to reach two structurally different places depending on which runner tier
ends up active - GenerationConfig.chat_template_kwargs for server-backed
tiers (extra_body), PromptingConfig.chat_template_kwargs for local
in-process tiers (baked into the prompt string via apply_chat_template).
Building both through RunnerConfig auto-syncs the two so setting either
one is enough.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from llm_kit.llm_runtime import GenerationConfig
from llm_kit.llm_setup import BaseConfig, LlmConfig
from llm_kit.prompt_builder import PromptingConfig


class RunnerConfig(BaseModel):
    """Bundles the configs build_runner()/PromptBuilder() need, with
    cross-config defaults synced where a setting has to reach both."""
    base: BaseConfig = Field(default_factory=BaseConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    prompt: PromptingConfig = Field(default_factory=PromptingConfig)

    @model_validator(mode="after")
    def _sync_chat_template_kwargs(self) -> "RunnerConfig":
        """chat_template_kwargs (e.g. Qwen3's enable_thinking) has to live
        on both generation (server-backed tiers - sent as extra_body, or
        as the llama.cpp server's own --chat_template_kwargs startup flag)
        and prompt (local in-process tiers - baked into the prompt string
        via apply_chat_template), since those are two structurally
        different delivery points - see PromptingConfig.chat_template_kwargs
        / GenerationConfig.chat_template_kwargs. Callers shouldn't have to
        know that split: if only one side was set, mirror it onto the
        other so setting it once is enough. Leaves both alone if either
        both or neither were set explicitly."""
        if self.generation.chat_template_kwargs and not self.prompt.chat_template_kwargs:
            self.prompt.chat_template_kwargs = dict(self.generation.chat_template_kwargs)
        elif self.prompt.chat_template_kwargs and not self.generation.chat_template_kwargs:
            self.generation.chat_template_kwargs = dict(self.prompt.chat_template_kwargs)
        return self

    def to_llama_cpp(self) -> dict:
        return self.generation.to_llama_cpp(seed=self.base.seed)

    def to_vllm(self):
        return self.generation.to_vllm(seed=self.base.seed)

    def to_hf(self) -> dict:
        return self.generation.to_hf(seed=self.base.seed)

    def to_chat_completions(self) -> dict:
        return self.generation.to_chat_completions(seed=self.base.seed)
