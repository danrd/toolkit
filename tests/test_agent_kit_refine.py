"""The refinement loop with the project's parts given as functions: here the 'answer' is an integer."""
from types import SimpleNamespace

from agent_kit.refine import (Attempt, RefiningModule, best_attempt, loop, memoize_verdicts, render_history)
from llm_kit.prompt_builder import ApproxTokenizer, PromptBuilder, PromptingConfig


def parse(text):
    try:
        return int(text.strip())
    except ValueError:
        return None


def review_against(target):
    def review(task, value):
        if value is None:
            return ["not a number"], 1, 0
        return ([], 0, 0) if value == target else ([f"{value} is not {target}"], 1, abs(value - target))
    return review


class TestHistory:
    def test_each_refused_answer_is_shown_with_its_reasons_and_clean_ones_are_skipped(self):
        attempts = [Attempt(1, "3", 3, ["too small"], 1, 0), Attempt(2, "5", 5, [], 0, 0), Attempt(3, "", None, ["no"], 1, 0)]
        text = render_history(attempts)
        assert "Attempt 1. Your answer:\n3" in text and "- too small" in text
        assert "Attempt 2" not in text and "(no answer)" in text and "Attempt 3" in text
        assert render_history([]) == "" and render_history([attempts[1]]) == ""

    def test_the_best_attempt_reads_over_not_reading_then_fewest_strong_then_weak_the_later_of_equals(self):
        unread = Attempt(1, "x", None, ["no"], 1, 0)
        worse = Attempt(2, "3", 3, ["a"], 2, 0)
        better = Attempt(3, "4", 4, ["a"], 1, 5)
        tie = Attempt(4, "6", 6, ["a"], 1, 5)
        assert best_attempt([unread, worse]) is worse
        assert best_attempt([unread, worse, better]) is better
        assert best_attempt([better, tie]) is tie
        assert best_attempt([]) is None


def test_a_verdict_is_asked_once_for_the_same_question():
    asked = []
    verdict = memoize_verdicts(lambda task, value: asked.append(value) or True)
    task = object()
    assert verdict(task, 3) and verdict(task, 3) and verdict(task, 4)
    assert asked == [3, 4]
    by_value = memoize_verdicts(lambda t, v: asked.append(("k", v)) or True, key=lambda t, v: v)
    by_value(object(), 1), by_value(object(), 1)
    assert asked.count(("k", 1)) == 1


class TestLoop:
    def test_it_stops_at_the_first_answer_nothing_is_wrong_with_and_the_next_call_sees_the_reasons(self):
        replies, seen = iter(["1", "2", "7", "9"]), []

        def ask(context):
            seen.append(context.get("history_text", ""))
            return next(replies)

        result = loop(object(), ask, review_against(7), parse, rounds=5)

        assert [a.text for a in result.attempts] == ["1", "2", "7"]
        assert result.accepted and result.solution == "7"
        assert seen[0] == "" and "1 is not 7" in seen[1] and "2 is not 7" in seen[2]

    def test_without_an_accepted_answer_the_one_with_the_fewest_complaints_wins(self):
        replies = iter(["1", "6", "2"])
        result = loop(object(), lambda c: next(replies), review_against(7), parse, rounds=3)
        assert not result.accepted and result.solution == "6"

    def test_a_reply_that_does_not_read_is_an_attempt_with_a_reason(self):
        result = loop(object(), lambda c: "hmm", review_against(7), parse, rounds=1)
        assert result.attempts[0].value is None and result.attempts[0].notes == ["not a number"]

    def test_the_callers_context_reaches_every_call_and_is_not_modified(self):
        context, seen = {"role": "x"}, []
        loop(object(), lambda c: seen.append(dict(c)) or "1", review_against(7), parse, rounds=2, context=context)
        assert all(c["role"] == "x" for c in seen) and context == {"role": "x"}

    def test_rounds_below_one_still_ask_once(self):
        asked = []
        loop(object(), lambda c: asked.append(1) or "1", review_against(7), parse, rounds=0)
        assert asked == [1]


class FakeModule:
    def __init__(self, replies):
        self.replies, self.contexts = iter(replies), []
        self.builder = PromptBuilder(PromptingConfig(blocks=["task", "output_format"], token_limit=1000),
                                     ApproxTokenizer())

    def solve(self, task, context=None):
        self.contexts.append(dict(context or {}))
        return {"solution": next(self.replies), "module_results": {}}

    def close(self):
        self.closed = True


class TestRefiningModule:
    def test_it_installs_the_memory_block_and_returns_the_accepted_answer(self):
        module = FakeModule(["1", "7"])
        refined = RefiningModule(module, review_against(7), parse, rounds=3)

        result = refined.solve(object(), {"role": "r"})

        assert "memory" in module.builder.config.blocks
        assert result["solution"] == "7" and result["module_results"]["accepted"] is True
        assert result["module_results"]["rounds"] == 2 and "1 is not 7" in module.contexts[1]["history_text"]
        assert refined.last is refined.runs[-1]

    def test_a_module_that_only_reports_errors_gives_the_last_error(self):
        class Failing(FakeModule):
            def solve(self, task, context=None):
                return {"solution": "", "module_results": {"error": "prompt didn't fit"}}

        result = RefiningModule(Failing([]), review_against(7), parse, rounds=2).solve(object())
        assert result == {"solution": "", "module_results": {"error": "prompt didn't fit"}}

    def test_it_forwards_the_builder_the_runner_and_close(self):
        module = FakeModule(["7"])
        module.runner = "runner"
        refined = RefiningModule(module, review_against(7), parse)
        assert refined.builder is module.builder and refined.runner == "runner"
        refined.close()
        assert module.closed


def test_attempts_are_made_as_the_class_the_project_names():
    from dataclasses import dataclass

    @dataclass
    class NumberAttempt(Attempt):
        @property
        def number_value(self):
            return self.value

    result = loop(object(), lambda c: "7", review_against(7), parse, attempt_class=NumberAttempt)
    assert isinstance(result.attempts[0], NumberAttempt) and result.best.number_value == 7
    module = FakeModule(["7"])
    refined = RefiningModule(module, review_against(7), parse, attempt_class=NumberAttempt)
    refined.solve(object())
    assert isinstance(refined.last.attempts[0], NumberAttempt)
