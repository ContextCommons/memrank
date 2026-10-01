"""How many judge calls a run needs, known before the first one is spent.

This began as a CAP: a run whose `--max-judge-calls` was too small was refused. The ceiling is gone,
because capping the JUDGE bounds the measurement rather than the system under test -- which queries
got graded would depend on where a counter ran out. The arithmetic survives as a REPORT, printed
before a judged run starts, the way HELM prints estimated token usage.

The estimate is a restatement of what `judge_query` does, so these tests run BOTH against a counting
completer and assert they agree. Adding a judging step without updating the estimate fails here
rather than costing someone a paid run.
"""
from __future__ import annotations

import pytest

from memrank.judging.judge import (
    JudgeConfig,
    calls_per_query,
    judge_query,
    no_context_calls,
    required_calls,
    with_context_calls,
)
from memrank.judging.shape import GENERIC_BINARY_CATEGORIES, BinaryJudgeShape
from tests.fakes import JudgeFakeBenchmark

_VERDICT = '{"passed": true, "rationale": "ok"}'


def _counter():
    """A completer that records how many times it was called and always parses."""
    calls: list[tuple[str, str]] = []

    def complete(model: str, system: str, user: str) -> str:
        calls.append((model, system))
        return _VERDICT
    return complete, calls


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("samples", [1, 3])
@pytest.mark.parametrize("no_context_control", [True, False])
def test_the_estimate_equals_the_actual_calls(negative, samples, no_context_control):
    """The drift guard. Any new judging step must be reflected in calls_per_query."""
    cfg = JudgeConfig(samples=samples, no_context_control=no_context_control)
    complete, calls = _counter()

    judge_query(complete, question="q", context="c", gold="g", cfg=cfg, negative=negative)

    assert len(calls) == calls_per_query(cfg, negative=negative), (
        f"calls_per_query says {calls_per_query(cfg, negative=negative)} but judge_query made "
        f"{len(calls)}. The estimate is a restatement of judge_query; update it in the same commit.")


def test_a_negative_query_is_cheaper_than_a_positive_one():
    """Sufficiency is only asked about queries that have something to be sufficient for."""
    cfg = JudgeConfig()
    assert calls_per_query(cfg, negative=True) < calls_per_query(cfg, negative=False)


def test_dropping_the_no_context_control_is_cheaper():
    """It is a real methodology control, so this is a trade, not a free saving -- but the
    preflight should be able to quote the cheaper number."""
    with_control = calls_per_query(JudgeConfig(no_context_control=True), negative=False)
    without = calls_per_query(JudgeConfig(no_context_control=False), negative=False)
    assert without < with_control


def test_the_demo_smoke_slice_needs_twenty_four():
    """The exact run that failed: 4 positive + 1 negative, defaults, cap of 20."""
    from memrank.benchmarks import get_benchmark

    units = get_benchmark("demo", k=10, slice="smoke").load()
    assert required_calls(units, JudgeConfig()) == 24


def test_required_calls_counts_every_unit():
    """A multi-unit benchmark must not be costed as though it were one."""
    from memrank.benchmarks import get_benchmark

    units = get_benchmark("demo", k=10, slice="smoke").load()
    cfg = JudgeConfig()
    doubled = required_calls([*units, *units], cfg)
    assert doubled == 2 * required_calls(units, cfg)


# --- what the estimate must NOT count, and what it must count twice --------------------------- #

@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("no_context_control", [True, False])
def test_the_two_halves_are_the_whole(negative, no_context_control):
    """The split underneath the estimate must not lose or double a call.

    `calls_per_query` is the drift guard's contract with `judge_query`; the halves exist only to
    say which of those calls a second engine can reuse. If they stop summing to it, the sweep
    estimate is wrong in a way the drift guard cannot see."""
    cfg = JudgeConfig(no_context_control=no_context_control)

    assert (no_context_calls(cfg) + with_context_calls(cfg, negative=negative)
            == calls_per_query(cfg, negative=negative))


def test_unjudgeable_queries_are_not_costed():
    """The over-count, in the smallest fixture that has one.

    `JudgeFakeBenchmark` has 3 queries and 2 the judge can grade -- the third ships no gold
    answer, so the binary shape skips it and `_apply_judge` never grades it. Pricing all 3 is
    the class of over-count that once had a smoke slice quoted at more calls than it could
    spend, a rule that then got published in a repro report."""
    units = JudgeFakeBenchmark().load()
    cfg = JudgeConfig()
    shape = BinaryJudgeShape(GENERIC_BINARY_CATEGORIES)
    judgeable = [q for u in units for q in u.queries if shape.is_judgeable(q)]

    assert len(judgeable) == 2 and len(units[0].queries) == 3
    assert required_calls(units, cfg) == sum(
        calls_per_query(cfg, negative=q.get("kind") == "negative") for q in judgeable)


def test_a_sweep_pays_for_the_no_context_control_once():
    """The failure tech-debt.md called "worse than not having it".

    The cap and its counter are built once and shared across a fan-out, so N engines cost more
    than one -- but not N times more. The no-context half is the same question, gold and model for
    every engine, so from the second onward it is a cache hit, and cache hits never reach the cap.
    demo/smoke over 4 engines is 66 calls: not 24 (what a per-target estimate quoted, before the
    run died on engine 2) and not 96 (what multiplying the whole thing would demand)."""
    from memrank.benchmarks import get_benchmark

    units = get_benchmark("demo", k=10, slice="smoke").load()
    cfg = JudgeConfig()
    one = required_calls(units, cfg)
    shared = no_context_calls(cfg) * 5      # 5 judgeable queries in demo/smoke

    assert one == 24
    assert required_calls(units, cfg, targets=4) == shared + (one - shared) * 4 == 66
    assert required_calls(units, cfg, targets=4) < 4 * one


def test_the_sweep_discount_is_conditional_on_the_cache():
    """`--no-judge-cache` removes the mechanism the discount is made of.

    The no-context half is only free for engine 2 because engine 1's identical request was
    written to the cache -- and the cache wraps the cap, so the hit is never billed. Take the
    cache away and every engine pays it again. Claiming the discount anyway would under-cost a
    sweep by the whole control, which is the failure this gate exists to prevent."""
    from memrank.benchmarks import get_benchmark

    units = get_benchmark("demo", k=10, slice="smoke").load()
    cached, uncached = JudgeConfig(), JudgeConfig(cache=False)

    assert required_calls(units, cached) == required_calls(units, uncached), "one target: same"
    assert required_calls(units, uncached, targets=4) == 4 * required_calls(units, uncached)
    assert required_calls(units, uncached, targets=4) > required_calls(units, cached, targets=4)


def test_a_sweep_costs_nothing_extra_when_the_control_is_off():
    """With no shared half left, N engines really is N times one -- the formula must not invent
    a discount it cannot justify."""
    from memrank.benchmarks import get_benchmark

    units = get_benchmark("demo", k=10, slice="smoke").load()
    cfg = JudgeConfig(no_context_control=False)

    assert required_calls(units, cfg, targets=3) == 3 * required_calls(units, cfg)


# --- the estimate, at the CLI level ------------------------------------------------------------ #

def _run(*args, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from memrank.runner import app
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    return CliRunner().invoke(app, ["submit", "word-overlap", "demo",
                                    "--judge", *args])


#: Typer draws output inside an 80-column box, so a sentence is broken across lines with a
#: `│` at each edge. Both have to go before a phrase can be matched.
_BOX = str.maketrans("", "", "│╭╮╯╰─")


def _unwrapped(text: str) -> str:
    """Terminal output with its box and line wrapping collapsed.

    A phrase this test is about can be split across two lines by nothing more than an added word.
    Asserting on the wrapped form makes the test fail for the wrapping rather than for the
    arithmetic, which is the thing it is here to check.
    """
    return " ".join(text.translate(_BOX).split())


def test_a_judged_run_says_what_it_will_spend(tmp_path, monkeypatch):
    """Reported before the first call, the way HELM prints estimated token usage.

    This used to be a REFUSAL: `--max-judge-calls 20` was rejected because demo needs 24. The cap
    is gone -- bounding the judge would bound the measurement, since which queries got graded would
    depend on where a counter ran out -- and the arithmetic that powered the refusal is now simply
    said out loud, so a caller can choose a smaller slice before starting.
    """
    output = _unwrapped(_run(tmp_path=tmp_path, monkeypatch=monkeypatch).output)

    assert "~24 judge calls" in output
    assert "5 judgeable queries (4 positive, 1 negative)" in output
    assert "5 calls per positive and 4 per negative" in output


def test_the_estimate_names_the_retry_ceiling_too(tmp_path, monkeypatch):
    """The upper figure is what re-asking every unparseable verdict could cost -- the one way a run
    exceeds its plan. Loose on purpose: answer calls are never re-asked."""
    output = _unwrapped(_run(tmp_path=tmp_path, monkeypatch=monkeypatch).output)

    assert "Up to 72 if every verdict has to be re-asked." in output


def test_nothing_is_refused_for_being_expensive(tmp_path, monkeypatch):
    """The estimate informs; it never gates. The run proceeds to fail on the fake key instead."""
    output = _unwrapped(_run(tmp_path=tmp_path, monkeypatch=monkeypatch).output)

    assert "cannot finish this run" not in output
    assert "--max-judge-calls" not in output, "the flag is retired; nothing should suggest it"


def test_a_two_target_sweep_is_costed_for_both(tmp_path, monkeypatch):
    """One local sweep shares one counter across every engine, so the estimate must too -- quoting
    one target's number would understate a sweep by its fan-out width."""
    from typer.testing import CliRunner

    from memrank.runner import app
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    result = CliRunner().invoke(app, ["submit", "word-overlap,no-context", "demo",
                                      "--judge"])
    output = _unwrapped(result.output)

    assert "~38 judge calls" in output
    assert "over 2 engines sharing one counter" in output
