"""Universal LLM inference runner: the "how to generate" layer only.

One interface regardless of backend: `Runner.generate(prompt: str) -> str`.
`prompt` is always a plain string (whatever PromptBuilder already produces,
with or without a chat template) — the string-to-chat-messages translation
happens only at the one boundary that actually needs it (ServerRunner /
OpenRouterRunner talking to an OpenAI-compatible endpoint), not in every
caller.

Hosted/proprietary models (OpenRouter, and in principle OpenAI/Anthropic/
Gemini) are a deliberately SEPARATE, explicit path (`OpenRouterRunner`) —
not merged into llm_kit.llm_setup.build_runner. Whether to use local
inference or a hosted model is the caller's decision, not something to
infer from config.

Everything about GETTING a Runner into existence (starting a local server,
constructing an in-process model, the local-inference fallback chain via
`build_runner`) lives in llm_kit.llm_setup instead — that's "what to
load", this module is "how to sample from it once loaded".

Heavy dependencies (torch, transformers, openai) are imported lazily
inside whichever class/function actually needs them, so importing this
module never requires every backend's library to be installed.
"""
from __future__ import annotations

import gc
import os
import random
import subprocess
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GenerationConfig(BaseModel):
    """Base framework-agnostic config specifying key llm generation parameters."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid", frozen=False)

    temperature: float = Field(default=0.0, ge=0.0, le=2.0,
                               description="Scales logit distribution before softmax. 0.0 = greedy (argmax). < 1.0 = sharper, > 1.0 = flatter.")

    max_tokens:  int   = Field(default=256,  ge=1,
                               description="Maximum number of tokens to generate.")

    top_p:       float = Field(default=1.0,  ge=0.0, le=1.0,
                               description="Nucleus sampling: keep smallest token set whose cumulative probability ≥ top_p. 1.0 = disabled.")

    top_k:       int   = Field(default=-1,   ge=-1,
                               description="Sample from top-k most probable tokens. -1 = disabled.")

    stop:        List[str] = Field(default_factory=list,
                                   description="Stop generation immediately when any of these strings is produced.")

    repetition_penalty: float = Field(default=1.0, ge=0.0,
                                      description="Multiplicative penalty on previously generated tokens. 1.0 = no penalty, > 1.0 = penalise repetition.")

    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0,
                                     description="Additive logit penalty scaled by token frequency in context. Positive = reduce repetition.")

    use_beam_search: bool = Field(default=False,
                                   description="Use beam search instead of sampling. Requires temperature=0.0.")

    best_of: int = Field(default=1, ge=1,
                           description="Generate best_of candidates, return the best n. Must be ≥ n. Required for beam search.")

    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict,
                                   description="Extra kwargs forwarded to the backend's own chat-template "
                                               "application, for server-backed tiers only (e.g. "
                                               "{'enable_thinking': False} for Qwen3's reasoning toggle). "
                                               "Sent via to_chat_completions()'s extra_body - vLLM reads "
                                               "chat_template_kwargs from there; llama-cpp-python's server "
                                               "does NOT (it's a model-load-time setting there instead - see "
                                               "llm_setup._start_llama_cpp_server's --chat_template_kwargs "
                                               "flag). Ignored by to_llama_cpp/to_vllm/to_hf: those tiers "
                                               "consume an already-built prompt string, with any "
                                               "chat-template kwargs already baked in by "
                                               "PromptingConfig.chat_template_kwargs instead - set both by "
                                               "hand if you don't know ahead of time which tier will end up "
                                               "active.")

    def to_dict(self, exclude_none: bool = True, exclude_unset: bool = False) -> Dict[str, Any]:
        """Plain dict for **kwargs unpacking."""
        return self.model_dump(exclude_none=exclude_none, exclude_unset=exclude_unset)

    def merge(self, overrides: Dict[str, Any]) -> "GenerationConfig":
        """Return a new config with override values applied (non-mutating)."""
        return self.model_copy(update=overrides)

    def to_llama_cpp(self, seed: int) -> dict:
        """Prepare generation config for llama_cpp framework using a set of defaults parameters."""
        return {
            "temperature":        self.temperature,
            "max_tokens":         self.max_tokens,
            "top_p":              self.top_p,
            "top_k":              self.top_k if self.top_k != -1 else 0,
            "seed":               seed,
            "stop":               self.stop,
            "repeat_penalty":     self.repetition_penalty,
        }

    def to_vllm(self, seed: int):
        """Prepare generation config for vllm framework using a set of defaults parameters."""
        from vllm import SamplingParams
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=seed,
            stop=self.stop,
            repetition_penalty=self.repetition_penalty,
        )

    def to_hf(self, seed: int) -> dict:
        """Prepare generation config for HuggingFace Transformers."""
        from transformers import set_seed
        set_seed(seed)

        if self.use_beam_search:
            if self.temperature != 0.0:
                raise ValueError(
                    "HF beam search should be used with temperature=0.0 / do_sample=False."
                )

            params = {
                "max_new_tokens": self.max_tokens,
                "do_sample": False,
                "num_beams": self.best_of,
                "num_return_sequences": 1,
                "early_stopping": True,
                "repetition_penalty": self.repetition_penalty,
                "stop_strings": self.stop
            }
            return params

        do_sample = self.temperature > 0.0

        params = {
            "max_new_tokens": self.max_tokens,
            "do_sample": do_sample,
            "repetition_penalty": self.repetition_penalty,
        }

        if do_sample:
            params.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k if self.top_k != -1 else 0,
                }
            )

        if self.stop:
            params["stop_strings"] = self.stop

        return params

    def to_chat_completions(self, seed: int) -> dict:
        """Prepare generation config for any OpenAI-compatible Chat
        Completions endpoint - OpenRouter (OpenRouterRunner) as well as a
        local llama.cpp-server/vllm-serve instance (ServerRunner talks to
        the exact same request shape)."""
        if self.use_beam_search:
            raise ValueError(
                "Chat Completions API does not support beam search via `use_beam_search`."
            )
        if self.best_of != 1:
            raise ValueError(
                "Chat Completions API does not support `best_of` in the same way as vLLM. "
                "Use best_of=1."
            )

        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop if self.stop else None,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        if self.top_k != -1:
            params["top_k"] = self.top_k

        if seed is not None:
            params["seed"] = seed

        if self.chat_template_kwargs:
            params["extra_body"] = {"chat_template_kwargs": self.chat_template_kwargs}

        return {k: v for k, v in params.items() if v is not None}


class AllModelsFailedError(Exception):
    """Raised when every model in a resilience chain (e.g. OpenRouter's
    model list) failed — never silently swallowed into a fake "sorry"
    string that could be mistaken for a real answer downstream."""


class BaseRunner:
    """Common interface for every backend. Use as a context manager to
    guarantee server processes / GPU memory are cleaned up:
        with build_runner(config) as runner:
            text = runner.generate(prompt)
    """

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseRunner":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Server-backed runner (llama.cpp-server or `vllm serve`, OpenAI-compatible)
# ---------------------------------------------------------------------------

class ServerRunner(BaseRunner):
    """Wraps a local OpenAI-compatible HTTP server. This is the one place a
    plain prompt string gets wrapped into a single-turn chat message — every
    other backend just consumes the string directly."""

    def __init__(self, process: Optional[subprocess.Popen], port: int,
                 model_name: str, generation_kwargs: Dict[str, Any], client=None):
        self.process = process
        self.port = port
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs
        self._client = client  # allows injecting a fake client for testing

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="not-needed")
        return self._client

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **self.generation_kwargs,
        )
        return response.choices[0].message.content

    def close(self) -> None:
        if self.process is not None:
            _terminate_process(self.process)
            self.process = None


# ---------------------------------------------------------------------------
# In-process backends (no HTTP server)
# ---------------------------------------------------------------------------

class LlamaCppRunner(BaseRunner):
    """Wraps an in-process llama_cpp.Llama instance."""

    def __init__(self, model, generation_kwargs: Dict[str, Any]):
        self.model = model
        self.generation_kwargs = generation_kwargs

    def generate(self, prompt: str) -> str:
        return self.model(prompt, **self.generation_kwargs)["choices"][0]["text"]

    def close(self) -> None:
        self.model = None
        gc.collect()


class VLLMRunner(BaseRunner):
    """Wraps an in-process vllm.LLM instance."""

    def __init__(self, llm, sampling_params):
        self.llm = llm
        self.sampling_params = sampling_params

    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def close(self) -> None:
        self.llm = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFRunner(BaseRunner):
    """Wraps a plain transformers model + tokenizer (e.g. 4-bit via
    bitsandbytes) — the last-resort tier on GPU."""

    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def generate(self, prompt: str) -> str:
        import torch
        from transformers import GenerationConfig

        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            if isinstance(self.generation_config, GenerationConfig):
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            else:
                outputs = self.model.generate(**inputs, **self.generation_config)

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text

    def close(self) -> None:
        self.model = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Hosted / proprietary models — separate, explicit path (not part of
# build_runner's local cpu/gpu selection; the caller opts into this).
# ---------------------------------------------------------------------------

class OpenRouterRunner(BaseRunner):
    """Tries a list of OpenRouter models in order, with backoff on rate
    limits. Raises AllModelsFailedError if every model failed — no silent
    "service unavailable" string standing in for a real answer."""

    def __init__(self, models: List[str], generation_kwargs: Dict[str, Any],
                 api_key: Optional[str] = None, max_retries: int = 2,
                 timeout: float = 30.0, client=None):
        self.models = models
        self.generation_kwargs = generation_kwargs
        self._client = client
        if self._client is None:
            api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set")
            from openai import OpenAI
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key,
                                   max_retries=max_retries, timeout=timeout)

    def generate(self, prompt: str) -> str:
        from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

        messages = [{"role": "user", "content": prompt}]
        last_error: Optional[BaseException] = None

        for index, model in enumerate(self.models):
            try:
                response = self._client.chat.completions.create(
                    model=model, messages=messages, **self.generation_kwargs,
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                wait_time = 2 ** (index + 1) + random.uniform(0, 1)
                time.sleep(wait_time)
                last_error = e

            except (APIError, APITimeoutError, APIConnectionError, ConnectionError) as e:
                time.sleep(1)
                last_error = e

            except Exception as e:  # noqa: BLE001 - deliberately broad: keep trying remaining models
                last_error = e

        raise AllModelsFailedError(f"All {len(self.models)} OpenRouter models failed: {last_error}")


# ---------------------------------------------------------------------------
# Process cleanup — used by ServerRunner.close() above, and reused by
# llm_kit.llm_setup's fallback chain when a server fails its health
# check right after being spawned.
# ---------------------------------------------------------------------------

def _terminate_process(process: subprocess.Popen, timeout: float = 10.0) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    log_file = getattr(process, "log_file", None)
    if log_file is not None:
        log_file.close()
