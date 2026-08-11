"""Generic LLM-based task-processing loop: build a prompt for each task via
an `llm_module` (anything with `.builder.build(task, context)` and
`.runner.generate(prompt)` - e.g. PromptBuilder + a llm_setup.build_runner()
Runner glued together by the caller), score it with a pluggable evaluator,
and log + checkpoint the run to wandb along the way.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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
ResultPlotterFn = Callable[[Any, str, EvalResult], Any]


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


def run_llm_over_tasks(
    tasks: List[Tuple[Any, Any]],
    llm_module: Any,
    evaluator: EvaluatorFn,
    context_builder: Optional[ContextBuilderFn] = None,
    result_plotter: Optional[ResultPlotterFn] = None,
    log_config: Optional[WandbLogConfig] = None,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
    prompt_description: str = "",
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run `llm_module` over `tasks`, logging + checkpointing to
    wandb as it goes.

    Args:
        tasks: (task_id, task) pairs. task_id must be JSON-serializable
            (str/int) - it's the checkpoint's resume key.
        llm_module: anything with `.builder.build(task, context)`
            and `.runner.generate(prompt)` - a llm_kit.prompt_builder.PromptBuilder
            and a llm_kit.llm_setup.build_runner() Runner is the usual pairing.
        evaluator(task, generated_text) -> EvalResult: scores one
            generation. The only place project-specific (dataset-specific)
            scoring logic belongs.
        context_builder(task) -> dict: builds the PromptBuilder context for
            one task (e.g. a per-task role instruction). Defaults to {}.
        result_plotter(task, generated_text, eval_result) -> a
            wandb-loggable object (e.g. a matplotlib figure): only called
            if log_config.log_result_plot is True.
        run_id: pass the same id as a previous call to resume it - already
            processed task ids are skipped.

    Returns {"results": [...], "solved_tasks": [...], "avg_score": float}.
    """
    import wandb  # lazy: constructing WandbLogConfig/EvalResult shouldn't require wandb installed

    log_config = log_config or WandbLogConfig()
    context_builder = context_builder or (lambda task: {})
    config_dict = {"prompt_description": prompt_description, **(extra_config or {})}

    run = wandb.init(project=log_config.project, name=run_name, group=log_config.group,
                      config=config_dict, resume="allow", id=run_id)

    processed_ids = set()
    all_results: List[Dict[str, Any]] = []
    solved_tasks: List[Any] = []
    tasks_summary = wandb.Table(columns=["task_id", "primary_score", "prompt_len"])

    if run_id:
        checkpoint = load_checkpoint_from_wandb(run) or {}
        processed_ids = set(checkpoint.get("processed_tasks", []))
        all_results = checkpoint.get("prompts_data", [])
        solved_tasks = checkpoint.get("solved_tasks", [])
        summary_data = checkpoint.get("tasks_summary")
        if summary_data:
            tasks_summary = wandb.Table(dataframe=pd.DataFrame(summary_data))
        print(f"Resuming wandb run {run_id}: {len(processed_ids)} tasks already processed")

    tasks_since_checkpoint = 0
    for task_id, task in tasks:
        if task_id in processed_ids:
            continue

        start = time.time()
        context = context_builder(task)
        prompt = llm_module.builder.build(task, context=context)
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
            "prompt_length": len(prompt), "prompt_description": prompt_description,
            "metrics": eval_result.metrics, "primary_score": eval_result.primary_score,
            "processing_time_min": processing_time_min,
        })
        tasks_summary.add_data(str(task_id), eval_result.primary_score, len(prompt))

        if log_config.log_per_task_metrics:
            log_payload = {f"task_{task_id}_{name}": value for name, value in eval_result.metrics.items()}
            log_payload[f"task_{task_id}_processing_time_min"] = processing_time_min
            log_payload[f"task_{task_id}_prompt_len"] = len(prompt)
            log_payload["summary/total_tasks"] = len(all_results)
            log_payload["summary/avg_primary_score"] = tasks_summary.get_dataframe()["primary_score"].mean()
            log_payload["summary/total_solved"] = len(solved_tasks)
            if log_config.log_result_plot and result_plotter is not None:
                log_payload[f"task_{task_id}_result_plot"] = result_plotter(task, generation, eval_result)
            wandb.log(log_payload)

        if log_config.log_prompt_artifacts:
            artifact = wandb.Artifact(f"task_{task_id}_prompt", type="dataset")
            artifact = prepare_prompt_artifact(artifact, task_id, all_results[-1])
            run.log_artifact(artifact)

        processed_ids.add(task_id)
        tasks_since_checkpoint += 1
        if tasks_since_checkpoint >= log_config.checkpoint_interval:
            save_checkpoint_to_wandb(run, tasks_summary, all_results, processed_ids, solved_tasks)
            tasks_since_checkpoint = 0

    wandb.finish()
    avg_score = float(tasks_summary.get_dataframe()["primary_score"].mean()) if all_results else 0.0
    return {"results": all_results, "solved_tasks": solved_tasks, "avg_score": avg_score}
