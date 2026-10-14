"""The tracer: what a span records, and what a summary adds up."""
import pytest

from agent_kit.trace import Tracer, render, traced


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_a_span_records_how_long_its_block_took_and_the_tokens():
    clock = Clock()
    tracer = Tracer(clock)
    with tracer.span("llm", "attempt 1", tokens_in=900) as span:
        clock.now += 2.5
        span.tokens_out = 120
    (recorded,) = tracer.spans
    assert (recorded.phase, recorded.name, recorded.seconds) == ("llm", "attempt 1", 2.5)
    assert (recorded.tokens_in, recorded.tokens_out, recorded.error) == (900, 120, None)


def test_a_span_that_raises_is_recorded_with_the_error_and_the_error_goes_on():
    clock = Clock()
    tracer = Tracer(clock)
    with pytest.raises(ValueError):
        with tracer.span("verify"):
            clock.now += 1.0
            raise ValueError("no")
    assert tracer.spans[0].seconds == 1.0 and "ValueError: no" in tracer.spans[0].error
    assert tracer.summary()["verify"]["errors"] == 1


def test_the_summary_adds_a_phase_up_and_lists_phases_in_the_usual_order():
    clock = Clock()
    tracer = Tracer(clock)
    for phase, seconds, tin, tout in [("verify", 1.0, 10, 1), ("llm", 4.0, 100, 20), ("llm", 2.0, 50, 10)]:
        with tracer.span(phase, tokens_in=tin) as span:
            clock.now += seconds
            span.tokens_out = tout
    summary = tracer.summary()
    assert list(summary) == ["llm", "verify"]
    assert summary["llm"] == {"calls": 2, "seconds": 6.0, "longest": 4.0, "tokens_in": 150,
                              "tokens_out": 30, "errors": 0}
    assert tracer.records()[0]["phase"] == "verify"


def test_without_a_tracer_nothing_is_recorded_and_the_block_runs():
    ran = []
    with traced(None, "llm", "x") as span:
        ran.append(span)
    assert ran and traced(Tracer(), "llm").__class__.__name__ != "_NoSpan"


def test_the_table_gives_each_phase_its_share_of_the_time():
    clock = Clock()
    tracer = Tracer(clock)
    for phase, seconds in [("llm", 3.0), ("verify", 1.0)]:
        with tracer.span(phase):
            clock.now += seconds
    text = render(tracer.summary())
    assert "75%" in text and "25%" in text


def test_what_a_module_reports_about_a_call_becomes_a_prompt_span_and_a_generation_span():
    tracer = Tracer()
    tracer.record_module({"prompt_seconds": 0.5, "generate_seconds": 4.0, "prompt_tokens": 900, "reply_tokens": 30}, "m")
    assert [(s.phase, s.seconds) for s in tracer.spans] == [("prompt", 0.5), ("llm", 4.0)]
    assert (tracer.spans[1].tokens_in, tracer.spans[1].tokens_out) == (900, 30)
    tracer.record_module({"error": "x"}, "m")
    tracer.record_module(None)
    assert len(tracer.spans) == 2 and list(tracer.summary()) == ["prompt", "llm"]
