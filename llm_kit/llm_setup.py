"""Everything involved in getting a runnable LLM backend into memory:
config describing WHICH backend to load (BaseConfig - technical run
parameters that aren't specific to being an LLM: seed, device, serving
timeouts; LlmConfig - model identity, quantization, framework; generation-
time sampling params live in llm_kit.llm_runtime.GenerationConfig instead,
since that's "how to sample", not "what to load"), plus the actual
loading/starting logic (spawning a local server and waiting for it to
come up, constructing an in-process model + tokenizer) and
`build_runner(config)`, the public factory that ties it together with a
fallback chain per config.base.device:
    CPU:  llama.cpp server -> llama.cpp in-process
    GPU:  vLLM server -> vLLM in-process -> HF in-process (4-bit)
Every tier's error is collected; if all tiers fail, RuntimeError chains them.

Hosted/proprietary models (OpenRouter, and in principle OpenAI/Anthropic/
Gemini) are NOT part of this factory - llm_kit.llm_runtime.OpenRouterRunner
is a deliberately separate, explicit path the caller opts into directly.

Heavy dependencies (torch, transformers, llama_cpp, vllm) are imported
lazily inside whichever function actually needs them, so importing this
module - or building a runner for one backend - never requires every other
backend's library to be installed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from llm_kit.llm_runtime import (
    BaseRunner,
    HFRunner,
    LlamaCppRunner,
    ServerRunner,
    VLLMRunner,
    _terminate_process,
)


class BaseConfig(BaseModel):
    """Technical run parameters that aren't specific to running an LLM -
    seed, which compute device/serving knobs to use, timeouts. Kept
    separate from LlmConfig (model identity/loading) so the two can vary
    independently."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid", frozen=False)
    seed: int = 42
    checkpoint_interval: int = 1  # number of examples to process before printing relevant info
    device: str = 'cpu'
    port: int = 8001
    server_ready_timeout: float = 60.0
    verbose: bool = False


class LlmConfig(BaseModel):
    """Which model to load and how - identity, quantization, and the
    llama.cpp serving knobs specific to running it locally."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid", frozen=False)
    framework: str = 'llama_cpp'  # llama_cpp | vllm | hf
    model: str = 'unsloth/Qwen3.6-27B-GGUF'
    # A GGUF repo (the default above) embeds its own tokenizer for
    # llama.cpp's use, but has none of the files transformers.AutoTokenizer
    # needs - trying to load one from `model` directly fails with
    # "Couldn't instantiate the backend tokenizer...". Set this to the
    # equivalent plain HF repo when you need a real tokenizer (e.g. for
    # PromptBuilder's token counting); callers should use
    # `tokenizer_model or model`. None when `model` is already a plain HF
    # repo and doubles fine as its own tokenizer source.
    tokenizer_model: Optional[str] = None
    quant_file: str = 'Qwen3.6-27B-Q4_K_M.gguf'
    # Where quant_file GGUFs live once downloaded - see resolve_local_model_path().
    pretrained_models_dir: str = '/data/pretrained_models'
    max_context: int = 9000  # llm token limit for computational resources to control
    openrouter_models: List[str] = ["google/gemma-4-26b-a4b-it",
                                    "nvidia/nemotron-3-ultra-550b-a55b"]
    n_ctx: Optional[int] = None  # falls back to generation.max_tokens when unset
    n_tokens_batch: int = 512
    use_mlock: bool = True
    n_gpu_layers: int = 0
    tensor_parallel_size: int = 1  # vLLM: shard the model across this many GPUs


# ---------------------------------------------------------------------------
# Server startup + health check + cleanup
# ---------------------------------------------------------------------------

def _wait_for_server_ready(process: subprocess.Popen, port: int,
                            timeout: float = 60.0, interval: float = 1.0) -> bool:
    """Poll the OpenAI-compatible /v1/models endpoint until it answers, the
    server process dies, or timeout is hit."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if process.poll() is not None:
            return False  # process already exited — no point polling further
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(interval)
    return False


def resolve_local_model_path(config) -> str:
    """Resolve config.llm.model into a real local file llama.cpp can load.
    A GGUF-only repo id (e.g. 'unsloth/Qwen3.6-27B-GGUF') is not itself a
    loadable path - the weights are one file within that repo, named by
    quant_file, and need to be downloaded first. Uses huggingface_hub
    directly (the same caching/resume logic as the `hf download` CLI, but
    idempotent in-process - a no-op if the file is already present) rather
    than shelling out. If `model` already names an existing local file
    (e.g. a test fixture pointing straight at a tiny GGUF) it's returned
    as-is - quant_file is irrelevant to that case. When quant_file isn't
    set either, `model` is assumed to already be a loadable path or a
    plain (non-GGUF) HF repo id, and is returned as-is."""
    model = config.llm.model
    if os.path.isfile(model):
        return model
    quant_file = getattr(config.llm, "quant_file", None)
    if not quant_file:
        return model
    from huggingface_hub import hf_hub_download
    local_dir = getattr(config.llm, "pretrained_models_dir", "/data/pretrained_models")
    return hf_hub_download(repo_id=model, filename=quant_file, local_dir=local_dir)


def _start_llama_cpp_server(config) -> subprocess.Popen:
    # llama-cpp-python[server] is a declared dependency (pyproject.toml's
    # `llama-cpp` extra) - unlike vllm below, its wheels aren't
    # CUDA-version-sensitive, so there's no reason to keep installing it at
    # runtime instead of just requiring it ahead of time.
    port = getattr(config.base, "port", 8001)
    n_ctx = str(getattr(config.llm, "n_ctx", None) or getattr(config.generation, "max_tokens", 2048))
    log_file = open("llama_cpp.log", "w", encoding="utf-8")

    args = [sys.executable, "-m", "llama_cpp.server", "--model", resolve_local_model_path(config),
            "--port", str(port), "--use_mlock", "True", "--n_ctx", n_ctx]
    # llama-cpp-python's server applies chat_template_kwargs at model-load
    # time, not per request - unlike a generic OpenAI-compatible server,
    # so it has to travel as a CLI flag here rather than through
    # generation_kwargs / to_chat_completions()'s extra_body.
    chat_template_kwargs = getattr(config.generation, "chat_template_kwargs", None)
    if chat_template_kwargs:
        args += ["--chat_template_kwargs", json.dumps(chat_template_kwargs)]

    process = subprocess.Popen(
        args,
        stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy(),
    )
    process.log_file = log_file
    return process


def _start_vllm_server(config) -> subprocess.Popen:
    # vLLM's wheels are CUDA-version-sensitive, so there's no one version
    # to declare as a dependency ahead of time - but forcing an upgrade on
    # every call risks silently pulling a release whose CUDA build doesn't
    # match the GPU/driver stack actually present. Only install when vllm
    # isn't already importable, and capture the install's output so a
    # routine run doesn't flood the notebook - surfaced in full only if
    # the install fails.
    try:
        import vllm  # noqa: F401
    except ImportError:
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "vllm"],
            capture_output=True, text=True,
        )
        if install.returncode != 0:
            raise RuntimeError(
                f"pip install vllm failed (exit {install.returncode}):\n"
                f"{install.stdout}\n{install.stderr}"
            )

    port = getattr(config.base, "port", 8001)
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    log_file = open("vllm_server.log", "w", encoding="utf-8")

    args = ["vllm", "serve", config.llm.model, "--port", str(port)]
    tensor_parallel_size = getattr(config.llm, "tensor_parallel_size", 1)
    if tensor_parallel_size and tensor_parallel_size != 1:
        args += ["--tensor-parallel-size", str(tensor_parallel_size)]

    process = subprocess.Popen(
        args,
        stdout=log_file, stderr=subprocess.STDOUT, env=env,
    )
    process.log_file = log_file
    return process


# ---------------------------------------------------------------------------
# In-process backend construction
# ---------------------------------------------------------------------------

def setup_hf_model(model_id: str):
    """Initialize an HF causal LM in 4-bit (bitsandbytes) + its tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, torch_dtype=compute_dtype,
        use_cache=True, device_map="auto", trust_remote_code=True,
    )
    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
    return model, tokenizer


def setup_llama_cpp_model(model_path: str, config=None, tokenizer_id: Optional[str] = None):
    """In-process llama.cpp, using the same GGUF file the server tier would
    have used — the CPU fallback tier when the server fails to come up."""
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python") from e

    tokenizer = None
    if tokenizer_id is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    llm_cfg = getattr(config, "llm", config) if config is not None else object()
    base_cfg = getattr(config, "base", config) if config is not None else object()
    gen_cfg = getattr(config, "generation", config) if config is not None else object()

    model = Llama(
        model_path=model_path,
        n_ctx=getattr(llm_cfg, "n_ctx", None) or getattr(gen_cfg, "max_tokens", 2048),
        n_batch=getattr(llm_cfg, "n_tokens_batch", 512),
        use_mlock=getattr(llm_cfg, "use_mlock", True),
        n_gpu_layers=getattr(llm_cfg, "n_gpu_layers", 0),
        verbose=getattr(base_cfg, "verbose", False),
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Factory: local inference, with fallback chain
#
# Each tier passes `config` straight to ExperimentConfig.to_llama_cpp() /
# .to_vllm() / .to_hf() / .to_chat_completions() - the config already knows
# how to translate itself into that backend's kwargs shape (seed included),
# so there's no separate per-tier kwargs-building step to keep in sync here.
# ---------------------------------------------------------------------------

def build_runner(config) -> BaseRunner:
    """Build a local inference runner per config.base.device, falling back
    through progressively simpler backends if a tier fails to start:
        CPU:  llama.cpp server -> llama.cpp in-process
        GPU:  vLLM server -> vLLM in-process -> HF in-process (4-bit)
    Raises RuntimeError (chaining every tier's error) if all tiers fail.
    """
    device = config.base.device.lower()
    if device == "cpu":
        return _build_cpu_runner(config)
    if device == "gpu":
        return _build_gpu_runner(config)
    raise ValueError(f"Unsupported device: {config.base.device}")


def _build_cpu_runner(config) -> BaseRunner:
    errors = []
    port = getattr(config.base, "port", 8001)
    server_ready_timeout = getattr(config.base, "server_ready_timeout", 60.0)

    try:
        process = _start_llama_cpp_server(config)
        if _wait_for_server_ready(process, port, timeout=server_ready_timeout):
            return ServerRunner(process, port, config.llm.model, config.to_chat_completions())
        _terminate_process(process)
        errors.append(
            f"llama.cpp server: failed health check within {server_ready_timeout}s "
            f"(set config.base.server_ready_timeout to wait longer for a large model) "
            f"- see {process.log_file.name} for what the server actually logged"
        )
    except Exception as e:
        errors.append(f"llama.cpp server: {type(e).__name__}: {e}")

    try:
        model, _ = setup_llama_cpp_model(resolve_local_model_path(config), config=config)
        return LlamaCppRunner(model, config.to_llama_cpp())
    except Exception as e:
        errors.append(f"llama.cpp in-process: {type(e).__name__}: {e}")

    raise RuntimeError("All CPU backends failed:\n" + "\n".join(errors))


def _build_gpu_runner(config) -> BaseRunner:
    errors = []
    port = getattr(config.base, "port", 8001)
    server_ready_timeout = getattr(config.base, "server_ready_timeout", 60.0)

    try:
        process = _start_vllm_server(config)
        if _wait_for_server_ready(process, port, timeout=server_ready_timeout):
            return ServerRunner(process, port, config.llm.model, config.to_chat_completions())
        _terminate_process(process)
        errors.append(
            f"vLLM server: failed health check within {server_ready_timeout}s "
            f"(set config.base.server_ready_timeout to wait longer for a large model) "
            f"- see {process.log_file.name} for what the server actually logged"
        )
    except Exception as e:
        errors.append(f"vLLM server: {type(e).__name__}: {e}")

    try:
        from vllm import LLM
        llm = LLM(model=config.llm.model, tensor_parallel_size=getattr(config.llm, "tensor_parallel_size", 1))
        return VLLMRunner(llm, config.to_vllm())
    except Exception as e:
        errors.append(f"vLLM in-process: {type(e).__name__}: {e}")

    try:
        model, tokenizer = setup_hf_model(config.llm.model)
        return HFRunner(model, tokenizer, config.to_hf())
    except Exception as e:
        errors.append(f"HF in-process: {type(e).__name__}: {e}")

    raise RuntimeError("All GPU backends failed:\n" + "\n".join(errors))
