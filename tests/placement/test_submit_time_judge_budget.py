"""A judged cloud submission settles what it can settle here, not inside the Fargate task.

On 2026-08-13 four tasks died on the same arithmetic -- `beam:100k-smoke` needs 288 judge calls and
the default cap was 200 -- each after an image pull and a 274 MB dataset download. The check itself
was correct and had been there for months; it ran on the wrong side of the wire, because `--on
cloud` returned from the command before `_sweep_gates`.

The cap that produced those failures is gone: a ceiling on the JUDGE bounds the measurement rather
than the system under test, since which queries got graded would depend on where a counter ran out.
What survives is the arithmetic, reported, and the coverage refusal -- a judged run that can score
nothing still costs a task to discover.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from memrank.judging.judge import MAX_VERDICT_ATTEMPTS, JudgeConfig, required_calls
from memrank.placement import run_api_client
from memrank.runner import app, get_benchmark

runner = CliRunner()


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    """A cloud submit path that records payloads and launches nothing."""
    from memrank import runner as runner_mod  # noqa: F401 - some stubs still land here
    from memrank.orchestration import cloud as cloud_mod

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    state: dict = {"submits": []}

    class _FakeHttp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _submit(http, org, payload):
        state["submits"].append(payload)
        rid = f"20260813-120000__demo__{len(state['submits'])}"
        return {"id": rid, "state": "submitted", "taskdef_arn": "arn:td",
                "task_arn": f"arn:aws:ecs:::task/c/{rid}", "cluster": "c",
                "region": "us-east-1", "log_group": "/ecs/c",
                "artifact": {"bucket": "b", "prefix": f"cloud-runs/{rid}"}}

    monkeypatch.setattr(cloud_mod, "_api_client", lambda: _FakeHttp())
    monkeypatch.setattr(run_api_client, "submit_run", _submit)
    return state


def _submit(*args):
    return runner.invoke(app, ["submit", "word-overlap", "demo", "--on", "cloud",
                               "--org", "acme", "--judge", *args])


def _flat(output: str) -> str:
    """Output with Typer's box borders and line wrapping collapsed, so a phrase stays a phrase."""
    return " ".join(output.translate(str.maketrans("", "", "│╭╮╯╰─")).split())


def test_a_submission_says_what_the_judging_will_cost(cloud):
    """Before the task exists, so the answer can still change the command."""
    bench = get_benchmark("demo", k=10)
    needed = required_calls(bench.load(), JudgeConfig(), targets=1, shape=bench.judge_shape())

    result = _submit()

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert f"~{needed} judge calls" in flat
    assert f"Up to {needed * MAX_VERDICT_ATTEMPTS}" in flat


def test_the_estimate_names_the_arithmetic_not_just_the_number(cloud):
    """The next person is choosing a slice for a different benchmark; a bare number teaches
    nothing about why it is that number."""
    result = _submit()

    flat = _flat(result.output)
    assert "judgeable queries" in flat
    assert "calls per positive" in flat and "per negative" in flat


def test_no_cap_reaches_the_task(cloud):
    """The flag is retired, so the command the server renders must not carry it -- an image that
    no longer parses it would die on its own command.

    A described submission cannot state one at all: there is no such field, so what used to be a
    property of the renderer's diligence is now a property of the model. That the command the
    server renders IS the one this CLI would have rendered is the separate guarantee in
    `tests/internal/accounts/test_cli_config_equivalence.py`."""
    result = _submit()

    assert result.exit_code == 0, result.output
    config = cloud["submits"][0]["config"]
    assert not [key for key in config if "judge_calls" in key]


def test_nothing_is_refused_for_being_expensive(cloud):
    """The estimate informs; it never gates. A costly slice submits exactly like a cheap one."""
    result = _submit()

    assert result.exit_code == 0, result.output
    assert len(cloud["submits"]) == 1
    assert "cannot finish" not in _flat(result.output)


def test_each_submitted_target_is_costed_as_its_own_task(cloud):
    """A LOCAL sweep shares one counter across the fan-out; each submitted target is a separate
    task with a separate counter, so quoting the multiplied figure would misstate a submission."""
    bench = get_benchmark("demo", k=10)
    one = required_calls(bench.load(), JudgeConfig(), targets=1, shape=bench.judge_shape())

    result = runner.invoke(app, ["submit", "word-overlap,no-context,fixed-context", "demo", "--on", "cloud",
                                 "--org", "acme", "--judge"])

    assert result.exit_code == 0, result.output
    assert len(cloud["submits"]) == 3
    assert f"~{one} judge calls" in _flat(result.output), "costed per task, not per sweep"
