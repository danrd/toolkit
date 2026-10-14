"""The refinement loop: ask the model, check the answer, show it what was wrong, ask again.

A model answers; the answer is read into a value (`parse`) and checked (`review`); if something is
wrong the model is called again with its own answer and the reasons in front of it (the `memory`
block). It stops at the first answer nothing is found wrong with, or after `rounds` calls, and
returns the attempt with the fewest complaints. This is the loop that looks like self-correction
and is not quite: the corrections come from outside the model, from checks that do not depend on it.

What a task is, how an answer is read and what is wrong with it are the project's, given as
functions:

    parse(text) -> value or None              # None: the reply could not be read
    review(task, value) -> (notes, strong, weak)   # notes empty: nothing wrong was found

    refined = RefiningModule(module, review, parse, rounds=3)
    refined.solve(task, context)               # the same shape as an llm_kit module's solve

`loop` is the same thing with the calls passed in, for anything that is not a module with a builder.
Whether it helps is a measurement, not a given: asking again with the previous answer alone did not
work, and what is new here is only that the reasons are concrete. `RefineResult.attempts` keeps every
attempt and its reasons, so a run can be read afterwards for whether the second answer was ever
better than the first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent_kit.blocks import install_blocks
from agent_kit.trace import Tracer

#: review(task, value) -> (notes, strong complaints, weak complaints)
Review = Callable[[Any, Any], Tuple[List[str], int, int]]


@dataclass
class Attempt:
    number: int
    text: str
    value: Any = None
    notes: List[str] = field(default_factory=list)
    strong: int = 0
    weak: int = 0

    @property
    def clean(self) -> bool:
        return not self.notes and self.value is not None


def render_history(attempts: Sequence[Attempt]) -> str:
    """The attempts for the `memory` block: the answer as written, and why it was not accepted."""
    parts = []
    for attempt in attempts:
        if attempt.clean:
            continue
        shown = attempt.text.strip() if isinstance(attempt.text, str) and attempt.text.strip() else "(no answer)"
        reasons = "\n".join(f"- {note}" for note in attempt.notes) or "- (no reason recorded)"
        parts.append(f"Attempt {attempt.number}. Your answer:\n{shown}\nIt was not accepted:\n{reasons}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\nDo not repeat these answers; correct what they got wrong."


def best_attempt(attempts: Sequence[Attempt]) -> Optional[Attempt]:
    """The best attempt: one that reads over one that does not, then the fewest strong complaints,
    then the fewest weak; the later of equals."""
    if not attempts:
        return None
    return min(reversed(list(attempts)), key=lambda a: (a.value is None, a.strong, a.weak))


def memoize_verdicts(verify: Callable[[Any, Any], bool],
                     key: Callable[[Any, Any], Any] = lambda task, value: (id(task), repr(value))
                     ) -> Callable[[Any, Any], bool]:
    """`verify` that answers a (task, value) it has been asked before from memory. A decision and the
    feedback may both want a second model's word on the same answer; it is asked once. `key` says
    when two questions are the same (by default: the same task object and the same value's repr)."""
    verdicts: Dict[Any, bool] = {}

    def verdict(task, value) -> bool:
        k = key(task, value)
        if k not in verdicts:
            verdicts[k] = bool(verify(task, value))
        return verdicts[k]

    return verdict


@dataclass
class RefineResult:
    attempts: List[Attempt] = field(default_factory=list)
    best: Optional[Attempt] = None

    @property
    def accepted(self) -> bool:
        return self.best is not None and self.best.clean

    @property
    def solution(self) -> str:
        return self.best.text if self.best is not None else ""


def loop(task, ask: Callable[[Dict[str, Any]], str], review: Review, parse: Callable[[str], Any],
         rounds: int = 3, context: Optional[Dict[str, Any]] = None) -> RefineResult:
    """`ask(context) -> text` up to `rounds` times; each call after the first has the attempts so far
    in `context["history_text"]`. Stops at the first attempt nothing is found wrong with."""
    attempts: List[Attempt] = []
    for _ in range(max(1, rounds)):
        call = dict(context or {})
        history = render_history(attempts)
        if history:
            call["history_text"] = history
        text = ask(call)
        text = text if isinstance(text, str) else ""
        value = parse(text) if text.strip() else None
        notes, strong, weak = review(task, value)
        attempts.append(Attempt(len(attempts) + 1, text, value, notes, strong, weak))
        if attempts[-1].clean:
            break
    return RefineResult(attempts=attempts, best=best_attempt(attempts))


class RefiningModule:
    """A module whose `solve` is `loop` around the wrapped one (anything with `.solve(task, context)`
    and a `.builder` whose prompt can hold a `memory` block)."""

    def __init__(self, module, review: Review, parse: Callable[[str], Any], rounds: int = 3,
                 tracer: Optional[Tracer] = None):
        self.module = module
        self.review, self.parse, self.rounds, self.tracer = review, parse, rounds, tracer
        self.last: Optional[RefineResult] = None
        self.runs: List[RefineResult] = []        # every call's loop, for reading a run afterwards
        install_blocks(module, ["memory"])

    @property
    def builder(self):
        return self.module.builder

    @property
    def runner(self):
        return self.module.runner

    def close(self) -> None:
        self.module.close()

    def solve(self, task, context: Optional[dict] = None) -> Dict[str, Any]:
        errors: List[str] = []

        def ask(call: Dict[str, Any]) -> str:
            result = self.module.solve(task, context=call)
            if self.tracer is not None:
                self.tracer.record_module(result.get("module_results"), "refine")
            if "error" in (result.get("module_results") or {}):
                errors.append(str(result["module_results"]["error"]))
            return result.get("solution", "")

        self.last = loop(task, ask, self.review, self.parse, self.rounds, context)
        self.runs.append(self.last)
        if not self.last.attempts or (not self.last.solution and errors):
            return {"solution": "", "module_results": {"error": errors[-1] if errors else "no answer"}}
        return {"solution": self.last.solution,
                "module_results": {"rounds": len(self.last.attempts), "accepted": self.last.accepted,
                                   "notes": [a.notes for a in self.last.attempts]}}
