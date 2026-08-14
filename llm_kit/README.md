# llm_kit

Reusable LLM pipeline toolkit, originally extracted from the
`subsymbolic/` module of the ARC-AGI project, now the canonical source -
`lector`/`maestro` depend on this directly; `arc-agi`'s own
`subsymbolic/llm_setup.py` etc. stay a physical copy, kept in sync via a
script rather than a live dependency (see arc-agi's own
`scripts/sync_llm_kit.py`). Three independent pieces, meant to be used
together or separately, plus one small optional aggregator:

- **`prompt_builder.py`** — `PromptBuilder`/`PromptingConfig`: compose a
  prompt from ordered Jinja2 "blocks" (`<name>/<version>.j2` template
  files), token-budgeted and joined as XML/Markdown/plain text or via
  `tokenizer.apply_chat_template`. Project-specific `resolvers`/`filters`
  are injected via constructor params (`resolver_registry`/
  `filter_registry`), not hardcoded — this module has zero project-specific
  imports.
- **`llm_runtime.py` + `llm_setup.py`** — `GenerationConfig` and a family
  of `Runner` classes (`ServerRunner`, `LlamaCppRunner`, `VLLMRunner`,
  `HFRunner`, `OpenRouterRunner`) behind one interface:
  `runner.generate(prompt: str) -> str`. `llm_setup.build_runner(config)`
  picks a local backend with a fallback chain:
  ```
  CPU:  llama.cpp server -> llama.cpp in-process
  GPU:  vLLM server -> vLLM in-process -> HF in-process (4-bit)
  ```
  Hosted models (OpenRouter, in principle OpenAI/Anthropic/Gemini) are a
  separate, explicit path (`OpenRouterRunner`) - never inferred from
  config. Heavy dependencies (torch, transformers, llama_cpp, vllm, openai)
  are imported lazily, so importing this module never requires every
  backend's library installed.
- **`llm_run.py` + `logging.py`** — `run_llm_over_tasks(...)`: a generic,
  resumable loop over tasks (each exposing a JSON-serializable `.id`) -
  build a prompt, generate, score with a pluggable
  `evaluator(task, generated_text) -> EvalResult`, log + checkpoint to
  wandb as it goes. Safe to interrupt and re-run with the same `run_id`
  - already-processed tasks are skipped.
- **`runner_config.py`** — `RunnerConfig`: bundles `base`/`generation`/
  `prompt` into the one object `build_runner(config)` already expects
  (`config.base`, `config.generation`, `config.to_llama_cpp()`/`.to_vllm()`/
  `.to_hf()`/`.to_chat_completions()`), so callers don't have to hand-roll
  it. Also auto-syncs `chat_template_kwargs` (e.g. Qwen3's
  `enable_thinking=False`) between `generation` and `prompt` when only one
  side is set - see PromptingConfig.chat_template_kwargs's docstring for
  why that setting has to reach two separate places. Entirely optional:
  `build_runner()`/`PromptBuilder()` still take their own configs
  directly if you'd rather wire them yourself.

## Install

Lives in the `toolkit` repo (this one), installed by other projects as a
git dependency - not published to PyPI:

```bash
uv add "toolkit[llm-kit] @ git+https://github.com/danrd/toolkit.git"                 # base: PromptBuilder + llm_run's loop
uv add "toolkit[llm-kit-llama-cpp] @ git+https://github.com/danrd/toolkit.git"       # + local CPU inference
uv add "toolkit[llm-kit-vllm] @ git+https://github.com/danrd/toolkit.git"            # + local GPU inference
uv add "toolkit[llm-kit-openrouter] @ git+https://github.com/danrd/toolkit.git"      # + hosted models via OpenRouter
uv add "toolkit[llm-kit-logging] @ git+https://github.com/danrd/toolkit.git"         # + wandb logging (run_llm_over_tasks needs this)
```

`lector`/`maestro` re-export these as their own `llama-cpp`/`vllm`/
`openrouter`/`hf`/`logging` extras, so `uv add "lector[llama-cpp]"` still
works unchanged - it just resolves through `toolkit` now instead of
installing the backend packages directly.

## Minimal usage

```python
from llm_kit.prompt_builder import PromptBuilder
from llm_kit.llm_setup import build_runner
from llm_kit.llm_run import EvalResult, run_llm_over_tasks
from llm_kit.runner_config import RunnerConfig

config = RunnerConfig()
config.prompt.blocks_dir = "prompts/"
config.prompt.blocks = ["instructions", "task"]
builder = PromptBuilder(config.prompt, tokenizer, resolver_registry={}, filter_registry={})

class LlmModule:
    def __init__(self, builder, runner):
        self.builder = builder
        self.runner = runner

runner = build_runner(config)  # build_runner reads config.base/.generation/.to_*()
module = LlmModule(builder, runner)

def my_evaluator(task, generated_text) -> EvalResult:
    ...

run_llm_over_tasks(tasks=[...], llm_module=module, evaluator=my_evaluator)
```

## Tests

```bash
uv run pytest
```
