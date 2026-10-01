"""Building the command the ECS task runs.

The load-bearing case is the first test: if ``--on cloud`` survives into the remote argv, the task
submits another task, which submits another. That is the one bug in this milestone that costs real
money on every occurrence, so it is asserted directly rather than inferred from a happy path.
"""
from __future__ import annotations

import pytest

from memrank.placement.cloud_submit import remote_command

#: A slice rides the eval ref, never a flag -- `test_remote_argv.py::test_tier_and_slice_ride_the_
#: ref_not_flags` asserts the submit path cannot produce the `--slice mini` this fixture used to
#: carry, so building the remote command from one was modelling an argv that never arrives.
ARGV = ["submit", "mem0", "beam:100k-mini"]


def _cmd(argv=None, **kw):
    kw.setdefault("bucket", "memrank-artifacts")
    kw.setdefault("run_id", "20260730-120000__beam__abc123")
    return remote_command(argv if argv is not None else ARGV, **kw)


def test_on_cloud_is_stripped_or_the_task_resubmits_itself():
    got = _cmd([*ARGV, "--on", "cloud"])
    assert "cloud" not in got.split("&&")[0].split()


@pytest.mark.parametrize("form", [["--on", "cloud"], ["--on=cloud"]])
def test_both_flag_forms_are_stripped(form):
    """`--on=cloud` must not survive just because the spaced form was the one anyone tested."""
    assert "cloud" not in _cmd([*ARGV, *form]).split("&&")[0].split()


def test_placement_is_pinned_by_the_renderer_not_inherited():
    """The task is told where to run, exactly once, whatever the submitter passed.

    Stripping alone stopped being enough when placement gained a config default: there is no
    flag to strip then, and the shipped default is cloud -- a task that read it would submit
    another task.
    """
    evaluate = _cmd([*ARGV, "--on", "cloud"]).split("&&")[0]
    assert evaluate.split().count("--on") == 1
    assert "--on none" in evaluate


def test_run_id_is_stripped_then_reissued():
    """The remote run must carry the submitter's id, not a second one, so S3 and runs/ agree."""
    got = _cmd([*ARGV, "--run-id", "someone-elses-id"])
    assert "someone-elses-id" not in got
    assert "--run-id 20260730-120000__beam__abc123" in got


def test_output_dir_is_rewritten_to_the_container_path():
    """A local --output-dir is meaningless in the task; /work/results is what upload reads."""
    got = _cmd([*ARGV, "--output-dir", "/home/me/results"])
    assert "/home/me/results" not in got
    assert "--output-dir /work/results" in got


def test_the_eval_arguments_survive_intact():
    got = _cmd()
    assert "memrank submit mem0 beam:100k-mini" in got


def test_upload_is_chained_with_and_not_semicolon():
    """`&&` is load-bearing: a failed evaluation must never publish a partial artifact.

    This used to assert `";" not in got`, which was a PROXY for that rule -- `;` would have run the
    upload regardless of the outcome. The rule is now stated directly, because the failure path
    legitimately contains one: a task that dies in the judge stage has already paid for every
    ingest and retrieve, and publishing its retrieval checkpoint is what makes that recoverable.
    """
    got = _cmd()
    assert "&&" in got
    eval_part, rest = got.split("&&", 1)
    assert "memrank submit" in eval_part
    assert "upload_results.py" in rest
    # The unrestricted upload -- the one that would publish a half-written artifact -- must be
    # reachable ONLY after a successful evaluation.
    success_branch = rest.split("||", 1)[0]
    assert "upload_results.py" in success_branch
    assert "--only" not in success_branch


def test_a_failed_evaluation_publishes_the_checkpoint_and_nothing_else():
    """The rescue path exists so a judge-stage failure costs the judging, not the whole run --
    4h50m of completed ingest and retrieval were discarded that way on 2026-08-12."""
    got = _cmd()
    assert "||" in got, "no failure branch: a dead task still leaves its checkpoint on ephemeral disk"
    failure_branch = got.split("||", 1)[1]
    assert "--only retrieval.json" in failure_branch, (
        "the failure path must name ONE file; uploading the directory could publish a "
        "half-written artifact, which is what the && guards against")
    assert "exit 1" in failure_branch, "rescuing the checkpoint must not mark a failed task green"


def test_upload_targets_the_run_id_prefix_under_cloud_runs():
    got = _cmd()
    assert "--bucket memrank-artifacts" in got
    assert "--prefix cloud-runs/20260730-120000__beam__abc123" in got


def test_upload_never_targets_the_gc_owned_prefix():
    """`runs/` in S3 belongs to leaderboard-gc, which deletes anything without a DB row."""
    assert "--prefix runs/" not in _cmd()


def test_a_run_id_that_could_reshape_the_command_is_refused():
    """The command is a shell string; an id carrying shell syntax must not be interpolated."""
    with pytest.raises(ValueError, match="run id"):
        _cmd(run_id="ok && rm -rf /")
