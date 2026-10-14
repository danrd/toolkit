"""Where a run's time goes.

A run through a pipeline is a handful of very different costs - cheap checks
in milliseconds, a search in minutes, the model's generation, a second model's
verdict, a background job waited on once - and which of them dominates is not
something to guess. The tracer is what the pieces report to: each wraps the
work it does in a span, a span records how long it took and, for a call to a
model, how many tokens went in and came out, and `summary` reads the spans
back as a table by phase.

    tracer = Tracer()
    with tracer.span("llm", "attempt 1", tokens_in=900) as span:
        reply = runner.generate(prompt)
        span.tokens_out = 120

A span that raises is still recorded, with the error, and the error goes on.
Nothing here is needed for a run: every piece that takes a tracer takes
`None` and then records nothing.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional

#: The phases a table lists, in order; a phase not named here comes after them, in the order it was first used.
#: A project with more phases of its own replaces this line.
PHASES = ("prompt", "llm", "verify", "decide", "other")


@dataclass
class Span:
    phase: str
    name: str
    seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Tracer:
    """The spans of one run, in the order they ended."""

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self.spans: List[Span] = []

    @contextmanager
    def span(self, phase: str, name: str = "", tokens_in: int = 0, **extra) -> Iterator[Span]:
        span = Span(phase=phase, name=name, tokens_in=tokens_in, extra=dict(extra))
        started = self._clock()
        try:
            yield span
        except BaseException as error:
            span.error = f"{type(error).__name__}: {error}"[:300]
            raise
        finally:
            span.seconds = self._clock() - started
            self.spans.append(span)

    def record(self, phase: str, name: str = "", seconds: float = 0.0, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """A span whose time was measured by someone else (a module's own report of what a call cost)."""
        self.spans.append(Span(phase=phase, name=name, seconds=seconds, tokens_in=tokens_in, tokens_out=tokens_out))

    def record_module(self, results, name: str = "") -> None:
        """The spans a model module's `module_results` report: building the prompt, then generating (the keys
        prompt_seconds, generate_seconds, prompt_tokens, reply_tokens that llm_kit's modules write)."""
        if not results or "generate_seconds" not in results:
            return
        self.record("prompt", name, results.get("prompt_seconds", 0.0))
        self.record("llm", name, results["generate_seconds"], results.get("prompt_tokens", 0),
                    results.get("reply_tokens", 0))

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Per phase: calls, seconds in all, the longest single call, tokens in and out, errors."""
        rows: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "calls": 0, "seconds": 0.0, "longest": 0.0, "tokens_in": 0, "tokens_out": 0, "errors": 0})
        for span in self.spans:
            row = rows[span.phase]
            row["calls"] += 1
            row["seconds"] += span.seconds
            row["longest"] = max(row["longest"], span.seconds)
            row["tokens_in"] += span.tokens_in
            row["tokens_out"] += span.tokens_out
            row["errors"] += span.error is not None
        ordered = sorted(rows, key=lambda p: PHASES.index(p) if p in PHASES else len(PHASES))
        return {phase: dict(rows[phase]) for phase in ordered}

    def records(self) -> List[Dict[str, Any]]:
        return [asdict(span) for span in self.spans]


def traced(tracer: Optional[Tracer], phase: str, name: str = "", **kwargs):
    """`tracer.span(...)` when there is a tracer, a span that records nothing when there is not."""
    if tracer is None:
        return _NoSpan()
    return tracer.span(phase, name, **kwargs)


class _NoSpan:
    def __enter__(self) -> Span:
        return Span(phase="", name="")

    def __exit__(self, *exc) -> bool:
        return False


def render(summary: Dict[str, Dict[str, float]]) -> str:
    """The summary as a table, the share of the total in the last column."""
    total = sum(row["seconds"] for row in summary.values()) or 1.0
    lines = [f"{'phase':<10}{'calls':>6}{'seconds':>10}{'longest':>9}{'tok in':>9}{'tok out':>9}{'errors':>7}{'share':>7}"]
    for phase, row in summary.items():
        lines.append(f"{phase:<10}{row['calls']:>6.0f}{row['seconds']:>10.1f}{row['longest']:>9.1f}"
                     f"{row['tokens_in']:>9.0f}{row['tokens_out']:>9.0f}{row['errors']:>7.0f}"
                     f"{100 * row['seconds'] / total:>6.0f}%")
    return "\n".join(lines)
