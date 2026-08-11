"""Tests for llm_kit/llm_run.py, the generic task-processing loop.

wandb needs a real account/network for its normal path, so these tests run
against a FakeWandb stand-in (persists "artifacts" to a local tmp dir, no
network) installed into sys.modules - both llm_run.py and llm_kit/logging.py
`import wandb` lazily inside the functions that use it, so patching
sys.modules["wandb"] is what actually takes effect for either of them, not
patching a module-level attribute.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from llm_kit.llm_run import EvalResult, WandbLogConfig, run_llm_over_tasks


# -- FakeWandb: no network, artifacts persisted to a local tmp dir -----------

class _FakeFile:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self._f = open(self.path, "w")
        return self._f

    def __exit__(self, *exc):
        self._f.close()


class _FakeArtifact:
    def __init__(self, name, type, storage_dir):
        self.name = name
        self.type = type
        self._dir = storage_dir
        os.makedirs(self._dir, exist_ok=True)

    def new_file(self, filename):
        return _FakeFile(os.path.join(self._dir, filename))

    def download(self):
        return self._dir


class _FakeRun:
    def __init__(self, run_id, artifact_root):
        self.id = run_id or "fake-run-id"
        self._artifact_root = artifact_root
        self.logged_artifacts: List[_FakeArtifact] = []

    def use_artifact(self, name, type):
        base_name = name.split(":")[0]
        storage_dir = os.path.join(self._artifact_root, base_name)
        if not os.path.isdir(storage_dir):
            raise FileNotFoundError(f"no such artifact: {name}")
        return _FakeArtifact(base_name, type, storage_dir)

    def log_artifact(self, artifact):
        self.logged_artifacts.append(artifact)


class _FakeTable:
    def __init__(self, columns=None, dataframe=None):
        if dataframe is not None:
            self._columns = list(dataframe.columns)
            self.rows = dataframe.values.tolist()
        else:
            self._columns = columns or []
            self.rows = []

    def add_data(self, *args):
        self.rows.append(list(args))

    def get_dataframe(self):
        return pd.DataFrame(self.rows, columns=self._columns)


class FakeWandb:
    """Stand-in for the wandb module - artifacts persist to a local tmp dir
    so checkpoint save/load actually round-trips, not just gets stubbed out."""

    def __init__(self, artifact_root):
        self._artifact_root = str(artifact_root)
        self.run: Optional[_FakeRun] = None
        self.logged: List[Dict[str, Any]] = []
        self.finished = False

    def init(self, **kwargs):
        self.run = _FakeRun(kwargs.get("id"), self._artifact_root)
        return self.run

    def Table(self, columns=None, dataframe=None):
        return _FakeTable(columns=columns, dataframe=dataframe)

    def Artifact(self, name, type):
        return _FakeArtifact(name, type, os.path.join(self._artifact_root, name))

    def log(self, payload):
        self.logged.append(payload)

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(tmp_path, monkeypatch):
    fake = FakeWandb(tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


# -- fake llm_module ---------------------------------------------------

class _FakeBuilder:
    def __init__(self, prompts: Dict[str, Optional[str]]):
        self._prompts = prompts

    def build(self, task, context=None):
        return self._prompts[task]


class _FakeRunner:
    def __init__(self, generations: Dict[str, str]):
        self._generations = generations

    def generate(self, prompt):
        return self._generations[prompt]


def _fake_module(prompts, generations):
    class _Module:
        pass
    module = _Module()
    module.builder = _FakeBuilder(prompts)
    module.runner = _FakeRunner(generations)
    return module


def _exact_match_evaluator(task, generated_text: str) -> EvalResult:
    solved = generated_text == "CORRECT"
    return EvalResult(metrics={"exact_match": float(solved)}, primary_score=float(solved), solved=solved)


# -- run_llm_over_tasks --------------------------------------------------------

def test_run_llm_over_tasks_basic(fake_wandb):
    prompts = {"t1": "prompt-1", "t2": "prompt-2"}
    generations = {"prompt-1": "CORRECT", "prompt-2": "WRONG"}
    module = _fake_module(prompts, generations)

    summary = run_llm_over_tasks(
        tasks=[("t1", "t1"), ("t2", "t2")],
        llm_module=module,
        evaluator=_exact_match_evaluator,
        log_config=WandbLogConfig(project="test-proj"),
    )

    assert summary["solved_tasks"] == ["t1"]
    assert summary["avg_score"] == 0.5
    assert len(summary["results"]) == 2
    assert fake_wandb.finished is True
    # per-task metrics were logged for both tasks
    logged_keys = {k for payload in fake_wandb.logged for k in payload}
    assert "task_t1_exact_match" in logged_keys
    assert "task_t2_exact_match" in logged_keys


def test_run_llm_over_tasks_skips_prompt_that_doesnt_fit(fake_wandb):
    prompts = {"t1": None}  # PromptBuilder.build() returns None when it doesn't fit token_limit
    module = _fake_module(prompts, generations={})

    calls = []

    def evaluator(task, text):
        calls.append(task)
        return EvalResult()

    summary = run_llm_over_tasks(
        tasks=[("t1", "t1")], llm_module=module, evaluator=evaluator,
        log_config=WandbLogConfig(project="test-proj"),
    )

    assert summary["results"] == []
    assert calls == []  # evaluator never called - there was nothing to evaluate


def test_run_llm_over_tasks_resume_skips_already_processed_tasks(fake_wandb):
    """Regression scenario for the original script's main requirement: a
    second run with the same run_id must not redo work the first run
    already checkpointed - including surviving a fully fresh FakeWandb
    instance (like a real interrupted-and-restarted process would see)."""
    prompts = {"t1": "prompt-1", "t2": "prompt-2"}
    generations = {"prompt-1": "CORRECT", "prompt-2": "WRONG"}
    module = _fake_module(prompts, generations)

    run_llm_over_tasks(
        tasks=[("t1", "t1")], llm_module=module, evaluator=_exact_match_evaluator,
        log_config=WandbLogConfig(project="test-proj", checkpoint_interval=1),
        run_id="resume-me",
    )

    calls = []

    def counting_evaluator(task, text):
        calls.append(task)
        return _exact_match_evaluator(task, text)

    summary = run_llm_over_tasks(
        tasks=[("t1", "t1"), ("t2", "t2")], llm_module=module, evaluator=counting_evaluator,
        log_config=WandbLogConfig(project="test-proj", checkpoint_interval=1),
        run_id="resume-me",
    )

    assert calls == ["t2"]  # t1 was skipped - already in the checkpoint
    assert {r["task_id"] for r in summary["results"]} == {"t1", "t2"}
    assert summary["solved_tasks"] == ["t1"]


def test_run_llm_over_tasks_resume_with_no_prior_checkpoint_does_not_crash(fake_wandb):
    """Regression test: load_checkpoint_from_wandb returns None when no
    checkpoint artifact exists yet - the original script's `checkpoint.get(...)`
    on that None would have raised AttributeError."""
    module = _fake_module({"t1": "prompt-1"}, {"prompt-1": "CORRECT"})

    summary = run_llm_over_tasks(
        tasks=[("t1", "t1")], llm_module=module, evaluator=_exact_match_evaluator,
        log_config=WandbLogConfig(project="test-proj"),
        run_id="never-run-before",
    )

    assert summary["solved_tasks"] == ["t1"]

