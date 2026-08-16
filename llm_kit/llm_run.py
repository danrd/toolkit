"""Generic LLM-based task-processing loop: build a prompt for each task via
an `llm_module` (anything with `.builder.build(task, context)` and
`.runner.generate(prompt)` - e.g. PromptBuilder + a llm_setup.build_runner()
Runner glued together by the caller), score it with a pluggable evaluator,
and log + checkpoint the run to wandb along the way.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from llm_kit.logging import load_checkpoint_from_wandb, prepare_prompt_artifact, save_checkpoint_to_wandb


@dataclass
class EvalResult:
    """What one evaluator call returns for one task.

    metrics: arbitrary named scores, all logged to wandb as-is (e.g.
        {"exact_match": 1.0, "lev_sim": 0.83}).
    primary_score: the one number used for the running average and for
        picking a per-task "best" result - callers decide what it means.
    solved: whether this task counts toward the run's solved-task tally.
    """
    metrics: Dict[str, float] = field(default_factory=dict)
    primary_score: float = 0.0
    solved: bool = False


EvaluatorFn = Callable[[Any, str], EvalResult]
ContextBuilderFn = Callable[[Any], Dict[str, Any]]
AssistantPrefixBuilderFn = Callable[[Any], Optional[str]]
ResultPlotterFn = Callable[[Any, str, EvalResult], Any]


_ARTIFACT_NAME_UNSAFE_RE = re.compile(r"[^a-zA-Z0-9_\-.]")


def _sanitize_artifact_name_part(text: str) -> str:
    """wandb artifact names reject most punctuation outright (and a "/"
    is parsed as an entity/project/artifact path separator, not a literal
    character) - run_name is caller-supplied free text (e.g. a model repo
    id like "unsloth/Qwen3-30B-...")."""
    return _ARTIFACT_NAME_UNSAFE_RE.sub("-", text)


def _artifact_task_label(task) -> str:
    """A wandb-artifact-name-safe per-task label. Zero-pads the index
    (when the task carries one) so wandb's Artifacts list - which sorts
    names as plain strings, not numbers - doesn't put "task_1919_..."
    before "task_443_..."; falls back to the bare id when there's no
    index to lead with."""
    task_index = getattr(task, "index", None)
    if isinstance(task_index, int):
        return f"{task_index:04d}_{task.id}"
    return str(task.id)


def _artifact_run_label(run, run_name: Optional[str]) -> str:
    """Disambiguates one run's artifacts from another's. wandb versions
    an artifact by NAME within a project - without this, two unrelated
    runs (different models, different configs) that happen to process
    the same task would silently share ONE artifact identity, stacking
    up as versions v1, v2... of "the same" prompt when they're actually
    unrelated. run.id is always unique and always available; run_name
    (when the caller set one) makes the artifact list human-readable too."""
    label = run_name or run.id
    return _sanitize_artifact_name_part(label)


class WandbLogConfig(BaseModel):
    """How much detail a run logs/checkpoints to wandb. The per-task result
    plot is by far the most expensive part (renders a figure per task) -
    off by default; everything else is cheap enough to leave on."""
    model_config = ConfigDict(extra="forbid")

    project: str = "llm-run"
    group: Optional[str] = None
    log_per_task_metrics: bool = True
    log_prompt_artifacts: bool = True
    log_result_plot: bool = False
    checkpoint_interval: int = Field(default=1, ge=1)


def _print_debug_task_info(task, prompt: str, generation: str, eval_result: EvalResult,
                            processing_time_min: float) -> None:
    """debug=True's per-task dump - the full prompt and full raw generation,
    not just the score, since the point is to see exactly what went in and
    what came back while iterating on a prompt/parsing problem, without
    waiting on the wandb dashboard."""
    task_index = getattr(task, "index", None)
    task_label = f"task {task_index} ({task.id})" if task_index is not None else f"task {task.id}"
    separator = "=" * 80
    print(separator)
    print(f"{task_label}: score={eval_result.primary_score:.3f} solved={eval_result.solved} "
          f"metrics={eval_result.metrics} time={processing_time_min:.2f}min")
    print(f"--- prompt ({len(prompt)} chars) ---")
    print(prompt)
    print(f"--- generation ({len(generation)} chars) ---")
    print(generation)
    print(separator)


def run_llm_over_tasks(
    tasks: List[Any],
    llm_module: Any,
    evaluator: EvaluatorFn,
    context_builder: Optional[ContextBuilderFn] = None,
    assistant_prefix_builder: Optional[AssistantPrefixBuilderFn] = None,
    result_plotter: Optional[ResultPlotterFn] = None,
    log_config: Optional[WandbLogConfig] = None,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
    run_description: str = "",
    extra_config: Optional[Dict[str, Any]] = None,
    show_progress: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run `llm_module` over `tasks`, logging + checkpointing to
    wandb as it goes.

    Args:
        tasks: task objects, each exposing a JSON-serializable `.id`
            (str/int) - it's the checkpoint's resume key.
        llm_module: anything with `.builder.build(task, context)`
            and `.runner.generate(prompt)` - a llm_kit.prompt_builder.PromptBuilder
            and a llm_kit.llm_setup.build_runner() Runner is the usual pairing.
        evaluator(task, generated_text) -> EvalResult: scores one
            generation. The only place project-specific (dataset-specific)
            scoring logic belongs.
        context_builder(task) -> dict: builds the PromptBuilder context for
            one task (e.g. a per-task role instruction). Defaults to {}.
        assistant_prefix_builder(task) -> str | None: builds a per-task
            PromptBuilder.build(assistant_prefix=...) override - e.g. a
            known partial-answer header, so the model only has to continue
            the answer instead of producing its own header from scratch.
            Defaults to None (no override). Whatever prefix is used here
            has to be threaded into `evaluator`/`result_plotter` separately
            (they only ever see the model's own completion, not the prefix
            it continues from) - wrap them in closures sharing the same
            per-task prefix logic.
        result_plotter(task, generated_text, eval_result) -> a
            wandb-loggable object (e.g. a matplotlib figure): built once
            per task whenever either log_config.log_result_plot or
            show_progress is on, and reused for both - not recomputed per
            destination.
        run_id: pass the same id as a previous call to resume it - already
            processed task ids are skipped.
        show_progress: print each task's score as it's scored, and (if
            result_plotter is set) display its figure inline via
            plt.show() - live feedback for a notebook-driven run, instead
            of only being visible later on the wandb dashboard. The
            printed line leads with `task.index` instead of the bare id
            when a task happens to carry one - purely an optional,
            generic attribute as far as this loop is concerned, not tied
            to any particular dataset's own numbering.
        debug: skip wandb entirely - no wandb.init/log/artifacts/
            checkpointing, and wandb doesn't even need to be installed -
            and print each task's full prompt and full raw generation
            (not just the score) instead of show_progress's one-line
            summary, plus still show its plot when result_plotter is set.
            For iterating on a prompt/parsing problem locally without
            touching the dashboard. run_id resume is ignored (there is no
            wandb checkpoint to resume from); log_config's wandb-specific
            fields are unused.

    Wandb specifics (skipped entirely when debug=True):
        - Per-task prompt artifacts (log_config.log_prompt_artifacts) are
          named f"{run_name or run.id}_task_{...}_prompt" - the run label
          disambiguates one run's artifacts from another's, since wandb
          versions an artifact by name *within the whole project*: without
          it, two unrelated runs that happen to process the same task
          would silently share one artifact identity instead of each
          getting their own. Each artifact also carries `metadata`
          (run_id, run_name, task_id, task_index, primary_score, solved)
          so it's self-describing without tracing back to its run.
        - tasks_summary (task_id/primary_score/prompt_len so far) is
          logged as a live, sortable Table panel on the same cadence as
          checkpointing - the intended way to browse "which tasks, what
          score" for one run, rather than scrolling the project's whole
          Artifacts list.
        - "summary/solved_tasks" carries the actual list of solved task
          ids (not just "summary/total_solved"'s count) as a run summary
          value.

    Returns {"results": [...], "solved_tasks": [...], "avg_score": float}.
    """
    log_config = log_config or WandbLogConfig()
    context_builder = context_builder or (lambda task: {})
    assistant_prefix_builder = assistant_prefix_builder or (lambda task: None)

    processed_ids = set()
    all_results: List[Dict[str, Any]] = []
    solved_tasks: List[Any] = []
    wandb = None
    run = None
    tasks_summary = None

    if not debug:
        import wandb  # lazy: debug runs (and just constructing WandbLogConfig/EvalResult) shouldn't require wandb installed
        config_dict = {"run_description": run_description, **(extra_config or {})}
        run = wandb.init(project=log_config.project, name=run_name, group=log_config.group,
                          config=config_dict, resume="allow", id=run_id)
        # log_mode="MUTABLE": this table is logged repeatedly (once per checkpoint)
        # after further add_data() calls in between - wandb's default IMMUTABLE mode
        # would silently drop everything past the first log() for the same table.
        tasks_summary = wandb.Table(columns=["task_id", "primary_score", "prompt_len"], log_mode="MUTABLE")

        if run_id:
            checkpoint = load_checkpoint_from_wandb(run) or {}
            processed_ids = set(checkpoint.get("processed_tasks", []))
            all_results = checkpoint.get("prompts_data", [])
            solved_tasks = checkpoint.get("solved_tasks", [])
            summary_data = checkpoint.get("tasks_summary")
            if summary_data:
                tasks_summary = wandb.Table(dataframe=pd.DataFrame(summary_data), log_mode="MUTABLE")
            print(f"Resuming wandb run {run_id}: {len(processed_ids)} tasks already processed")

    tasks_since_checkpoint = 0
    for task in tasks:
        task_id = task.id
        if task_id in processed_ids:
            continue

        start = time.time()
        context = context_builder(task)
        assistant_prefix = assistant_prefix_builder(task)
        prompt = llm_module.builder.build(task, context=context, assistant_prefix=assistant_prefix)
        if prompt is None:
            print(f"task {task_id}: skipped, prompt didn't fit token_limit")
            processed_ids.add(task_id)
            continue

        generation = llm_module.runner.generate(prompt)
        eval_result = evaluator(task, generation)
        processing_time_min = (time.time() - start) / 60

        if eval_result.solved:
            solved_tasks.append(task_id)

        all_results.append({
            "task_id": task_id, "prompt_text": prompt, "generation_result": generation,
            "prompt_length": len(prompt), "run_description": run_description,
            "metrics": eval_result.metrics, "primary_score": eval_result.primary_score,
            "processing_time_min": processing_time_min,
        })
        if not debug:
            tasks_summary.add_data(str(task_id), eval_result.primary_score, len(prompt))

        fig = None
        if result_plotter is not None and (show_progress or debug or log_config.log_result_plot):
            fig = result_plotter(task, generation, eval_result)

        if debug:
            _print_debug_task_info(task, prompt, generation, eval_result, processing_time_min)
        elif show_progress:
            task_index = getattr(task, "index", None)
            task_label = f"task {task_index} ({task_id})" if task_index is not None else f"task {task_id}"
            print(f"{task_label}: score={eval_result.primary_score:.3f} solved={eval_result.solved}")

        if (debug or show_progress) and fig is not None:
            import matplotlib.pyplot as plt  # lazy: opt-in, plotting shouldn't be a hard dependency otherwise
            plt.show()

        if not debug and log_config.log_per_task_metrics:
            log_payload = {f"task_{task_id}_{name}": value for name, value in eval_result.metrics.items()}
            log_payload[f"task_{task_id}_processing_time_min"] = processing_time_min
            log_payload[f"task_{task_id}_prompt_len"] = len(prompt)
            log_payload["summary/total_tasks"] = len(all_results)
            log_payload["summary/avg_primary_score"] = tasks_summary.get_dataframe()["primary_score"].mean()
            log_payload["summary/total_solved"] = len(solved_tasks)
            log_payload["summary/solved_tasks"] = solved_tasks
            if log_config.log_result_plot and fig is not None:
                log_payload[f"task_{task_id}_result_plot"] = fig
            wandb.log(log_payload)

        if not debug and log_config.log_prompt_artifacts:
            artifact_name = f"{_artifact_run_label(run, run_name)}_task_{_artifact_task_label(task)}_prompt"
            artifact = wandb.Artifact(artifact_name, type="dataset", metadata={
                "run_id": run.id, "run_name": run_name, "task_id": task_id,
                "task_index": getattr(task, "index", None),
                "primary_score": eval_result.primary_score, "solved": eval_result.solved,
            })
            artifact = prepare_prompt_artifact(artifact, task_id, all_results[-1])
            run.log_artifact(artifact)

        processed_ids.add(task_id)
        if not debug:
            tasks_since_checkpoint += 1
            if tasks_since_checkpoint >= log_config.checkpoint_interval:
                save_checkpoint_to_wandb(run, tasks_summary, all_results, processed_ids, solved_tasks)
                wandb.log({"tasks_summary": tasks_summary})
                tasks_since_checkpoint = 0

    if not debug:
        wandb.finish()
    avg_score = float(sum(r["primary_score"] for r in all_results) / len(all_results)) if all_results else 0.0
    return {"results": all_results, "solved_tasks": solved_tasks, "avg_score": avg_score}
