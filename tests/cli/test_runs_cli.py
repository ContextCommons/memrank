# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""``memrank runs ls`` / ``show`` over the local registry.

Everything writes into a tmp ``MEMRANK_RUNS_DIR`` (the tests/runs/test_run_registry.py pattern), so
the suite never reads a developer's accumulated runs -- and never depends on what they ran.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from memrank.runner import app
from memrank.runs import registry
from memrank.runs.status import UNKNOWN_STATE

runner = CliRunner()


@pytest.fixture
def runs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    return tmp_path


def _cell(run_dir, target, eval_name="demo", *, composite=0.5, started_at="2026-01-01T00:00:00"):
    payload = {"adapter": target, "benchmark": eval_name, "composite": composite,
               "receipt": {"adapter_name": target, "benchmark_name": eval_name,
                           "config_hash": "abcdef123456", "started_at": started_at}}
    (run_dir / f"{target}__{eval_name}.json").write_text(json.dumps(payload))


def _heartbeat(run_dir, *, target, eval_name="demo", state="done", started="2026-01-01T00:00:00",
               beating=False, pid=None):
    """Write a heartbeat. ``beating`` stamps it now, which is what `classify` reads as running --
    a non-terminal state with an old timestamp is `stale`, and rightly so."""
    updated = datetime.now(timezone.utc).isoformat() if beating else started
    payload = {"run_id": run_dir.name, "pid": pid, "target": target, "benchmark": eval_name,
               "slice": None, "state": state, "progress": {}, "message": "",
               "started_at": started, "updated_at": updated, "error": None}
    (run_dir / "status.json").write_text(json.dumps(payload))


def test_each_target_in_a_sweep_is_its_own_row(runs_dir):
    """One run per target: a two-target sweep is two runs, each an id that names one row."""
    first = registry.new_run_dir("demo")
    _cell(first, "word-overlap")
    _heartbeat(first, target="word-overlap")
    second = registry.new_run_dir("demo")
    _cell(second, "none")
    _heartbeat(second, target="none")

    output = runner.invoke(app, ["runs", "ls"]).stdout
    assert first.name in output and second.name in output
    assert "word-overlap,no-context" not in output


def test_live_includes_a_queued_sibling(runs_dir):
    """Mid-sweep, a not-yet-started target is live -- someone can still attach to or stop it."""
    import os

    queued = registry.new_run_dir("demo")
    _heartbeat(queued, target="none", state="queued", pid=os.getpid())

    output = runner.invoke(app, ["runs", "ls", "--live"]).stdout
    assert queued.name in output


def test_listing_is_newest_first(runs_dir):
    older = registry.new_run_dir("demo")
    _cell(older, "word-overlap", started_at="2026-01-01T00:00:00")
    _heartbeat(older, target="word-overlap", started="2026-01-01T00:00:00")
    newer = registry.new_run_dir("demo")
    _cell(newer, "word-overlap", started_at="2026-06-01T00:00:00")
    _heartbeat(newer, target="word-overlap", started="2026-06-01T00:00:00")

    output = runner.invoke(app, ["runs", "ls"]).stdout
    assert output.index(newer.name) < output.index(older.name)


def test_a_run_without_a_heartbeat_is_unknown_not_done(runs_dir):
    """Results on disk do not prove a run finished -- say so rather than inventing a state."""
    run_dir = registry.new_run_dir("demo")
    _cell(run_dir, "word-overlap")

    output = runner.invoke(app, ["runs", "ls"]).stdout
    assert UNKNOWN_STATE in output
    assert "done" not in output


def test_live_shows_only_what_is_still_going(runs_dir):
    finished = registry.new_run_dir("demo")
    _heartbeat(finished, target="word-overlap", state="done")
    live = registry.new_run_dir("demo")
    _heartbeat(live, target="word-overlap", state="retrieving", beating=True)

    output = runner.invoke(app, ["runs", "ls", "--live"]).stdout
    assert live.name in output
    assert finished.name not in output


def test_filters_narrow_by_target_and_eval(runs_dir):
    kept = registry.new_run_dir("demo")
    _cell(kept, "word-overlap", "demo")
    _heartbeat(kept, target="word-overlap", eval_name="demo")
    other = registry.new_run_dir("locomo")
    _cell(other, "mem0", "locomo")
    _heartbeat(other, target="mem0", eval_name="locomo")

    by_target = runner.invoke(app, ["runs", "ls", "--target", "word-overlap"]).stdout
    assert kept.name in by_target and other.name not in by_target
    by_eval = runner.invoke(app, ["runs", "ls", "--eval", "locomo"]).stdout
    assert other.name in by_eval and kept.name not in by_eval


def test_json_carries_the_cells(runs_dir):
    run_dir = registry.new_run_dir("demo")
    _cell(run_dir, "word-overlap", composite=0.75)
    _heartbeat(run_dir, target="word-overlap")

    payload = json.loads(runner.invoke(app, ["runs", "ls", "--json"]).stdout)
    assert [r["run_id"] for r in payload] == [run_dir.name]
    assert payload[0]["cells"][0]["composite"] == 0.75


def test_show_prints_state_and_results(runs_dir):
    run_dir = registry.new_run_dir("demo")
    _cell(run_dir, "word-overlap", composite=0.75)
    _heartbeat(run_dir, target="word-overlap")

    result = runner.invoke(app, ["runs", "show", run_dir.name])
    assert result.exit_code == 0
    assert "state " in result.output          # the heartbeat block
    assert "0.7500" in result.output          # the record -- what `status` does not show


def test_show_works_without_a_heartbeat(runs_dir):
    """A run with results but no heartbeat is still a run; `status` refuses these."""
    run_dir = registry.new_run_dir("demo")
    _cell(run_dir, "word-overlap")

    result = runner.invoke(app, ["runs", "show", run_dir.name])
    assert result.exit_code == 0
    assert UNKNOWN_STATE in result.output


def test_show_refuses_an_unknown_id(runs_dir):
    result = runner.invoke(app, ["runs", "show", "no-such-run"])
    assert result.exit_code != 0
    assert "no-such-run" in result.output


def test_empty_registry_says_so(runs_dir):
    result = runner.invoke(app, ["runs", "ls"])
    assert result.exit_code == 0
    assert "no runs" in result.output


def test_age_is_time_since_start_not_how_long_it_took(runs_dir):
    """A quick run from long ago is old, not brief -- the column says AGE and must mean it."""
    run_dir = registry.new_run_dir("demo")
    _cell(run_dir, "word-overlap")
    # Started long ago, finished a second later: duration is ~1s, age is years.
    _heartbeat(run_dir, target="word-overlap", started="2020-01-01T00:00:00+00:00")

    row = json.loads(runner.invoke(app, ["runs", "ls", "--json"]).stdout)[0]
    assert row["age"].endswith("d")
    assert row["age"] != "0s"


def test_age_of_an_unparseable_timestamp_is_no_answer(runs_dir):
    """A receipt without a start time leaves the run id in its place; that is not a date."""
    run_dir = registry.new_run_dir("demo")
    _cell(run_dir, "word-overlap", started_at=run_dir.name)

    row = json.loads(runner.invoke(app, ["runs", "ls", "--json"]).stdout)[0]
    assert row["age"] == "—"


def test_columns_stay_aligned_for_a_legacy_id(runs_dir):
    """Legacy ids are wider than today's; the column follows the data, not a constant."""
    legacy = runs_dir / "supermemory-locomo-20260727-064017-44505"
    legacy.mkdir()
    _cell(legacy, "supermemory", "locomo")
    _heartbeat(legacy, target="supermemory", eval_name="locomo")

    lines = [ln for ln in runner.invoke(app, ["runs", "ls"]).stdout.splitlines() if ln.strip()]
    header, row = lines[0], lines[1]
    assert legacy.name in row
    # Each column's value must begin exactly where its header does -- the id is 40 chars, so a
    # 38-wide column would shift every one of these.
    for label, value in (("TARGET", "supermemory"), ("EVAL", "locomo"), ("STATE", "done")):
        assert row[header.index(label):].startswith(value), f"{label} column is misaligned"


def test_empty_after_filtering_names_the_filter(runs_dir):
    """Filtering to nothing is not an empty registry -- say which constraint to drop."""
    run_dir = registry.new_run_dir("demo")
    _heartbeat(run_dir, target="word-overlap", state="done")

    result = runner.invoke(app, ["runs", "ls", "--live"])
    assert result.exit_code == 0
    assert "--live" in result.output
    assert "no runs recorded here" not in result.output


# --- the org universe: scoping, merging, and what a skewed deployment may claim ------------- #

@pytest.fixture
def signed_in(monkeypatch, tmp_path):
    """A machine with a session and a default org, and a stubbed platform listing."""
    from memrank import settings
    from memrank.cli import runs as runs_cli
    from memrank.cli import runs_show as runs_show_cli

    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    for spec in settings.SETTINGS:
        monkeypatch.delenv(spec.env, raising=False)
    settings.put("defaults.org", "acme")

    state: dict = {"body": {"runs": [], "scope": "mine"}, "calls": []}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _list(http, org, *, mine=True, limit=50):
        state["calls"].append({"org": org, "mine": mine, "limit": limit})
        if isinstance(state["body"], Exception):
            raise state["body"]
        return state["body"]

    monkeypatch.setattr(runs_cli.run_api_client, "authenticated_client", lambda: _Client())
    monkeypatch.setattr(runs_cli.run_api_client, "list_runs", _list)
    monkeypatch.setattr(runs_show_cli, "platform_record", lambda run_id: None)
    return state


def _platform_run(run_id, *, state="running", target="mem0:voyage", created="2026-07-01T00:00:00+00:00"):
    return {"id": run_id, "state": state, "target_ref": target, "benchmark": "locomo",
            "created_at": created, "cluster": "bench", "region": "us-east-1",
            "artifact": {"bucket": "b", "prefix": f"cloud-runs/{run_id}"}}


def test_the_org_universe_joins_the_listing(signed_in, runs_dir):
    signed_in["body"] = {"runs": [_platform_run("cloud-1")], "scope": "mine"}
    local = registry.new_run_dir("demo")
    _heartbeat(local, target="word-overlap")

    output = runner.invoke(app, ["runs", "ls"]).stdout
    assert "cloud-1" in output and local.name in output
    assert [(c["org"], c["mine"]) for c in signed_in["calls"]] == [("acme", True)]


def test_org_widens_the_scope(signed_in, runs_dir):
    signed_in["body"] = {"runs": [], "scope": "org"}
    runner.invoke(app, ["runs", "ls", "--org"])
    assert [(c["org"], c["mine"]) for c in signed_in["calls"]] == [("acme", False)]


def test_a_cloud_run_is_one_row_not_two(signed_in, runs_dir):
    """Submission writes a local record under the SERVER's id, so both stores hold this run.

    The platform is authoritative for state -- the laptop only remembers submitting -- while the
    local record keeps the cells that exist once artifacts were fetched.
    """
    shared = "20260801-120000__locomo__abc123"
    run_dir = runs_dir / shared
    run_dir.mkdir()
    _cell(run_dir, "mem0", "locomo", composite=0.42)
    _heartbeat(run_dir, target="mem0", eval_name="locomo", state="done")
    signed_in["body"] = {"runs": [_platform_run(shared, state="running")], "scope": "mine"}

    payload = json.loads(runner.invoke(app, ["runs", "ls", "--json"]).stdout)
    rows = [r for r in payload if r["run_id"] == shared]
    assert len(rows) == 1, "the same run must not appear once per store"
    assert rows[0]["state"] == "running"                    # the platform's answer
    assert rows[0]["cells"][0]["composite"] == 0.42         # the local record's
    assert rows[0]["place"] == "cloud"


def test_platform_states_are_shown_in_the_local_vocabulary(signed_in, runs_dir):
    """`stopped-success` and `done` are one condition; spelling both in one column misleads."""
    signed_in["body"] = {"runs": [_platform_run("c1", state="stopped-success")], "scope": "mine"}

    payload = json.loads(runner.invoke(app, ["runs", "ls", "--json"]).stdout)
    assert payload[0]["state"] == "done"


def test_live_includes_submitted_but_not_unknown(signed_in, runs_dir):
    signed_in["body"] = {"runs": [_platform_run("queued", state="submitted"),
                                  _platform_run("lost", state="unknown")], "scope": "mine"}

    output = runner.invoke(app, ["runs", "ls", "--live"]).stdout
    assert "queued" in output
    assert "lost" not in output


def test_a_server_that_cannot_scope_is_not_captioned_as_mine(signed_in, runs_dir):
    """FastAPI ignores unknown query parameters, so an older deployment answers `mine=true`
    with everyone's runs. Saying nothing would attribute other people's work to the caller."""
    signed_in["body"] = {"runs": [_platform_run("someone-elses")]}      # no `scope` key

    result = runner.invoke(app, ["runs", "ls"])
    assert result.exit_code == 0
    assert "cannot scope" in result.stderr
    assert "someone-elses" in result.stdout


def test_an_unreachable_api_degrades_the_listing_but_never_fails_it(signed_in, runs_dir):
    """Bare `runs ls` is not an explicitly remote request; the model forbids it failing."""
    from memrank.placement.run_api_client import RunApiError

    signed_in["body"] = RunApiError("boom")
    local = registry.new_run_dir("demo")
    _heartbeat(local, target="word-overlap")

    result = runner.invoke(app, ["runs", "ls"])
    assert result.exit_code == 0
    assert local.name in result.stdout
    assert "showing this machine's runs" in result.stderr


def test_org_is_an_explicitly_remote_ask_and_fails_loudly(signed_in, runs_dir):
    """The opposite rule, deliberately: answering `--org` from local records is fabrication."""
    from memrank.placement.run_api_client import RunApiError

    signed_in["body"] = RunApiError("boom")
    result = runner.invoke(app, ["runs", "ls", "--org"])
    assert result.exit_code != 0


def test_logged_out_lists_local_and_says_nothing_about_it(runs_dir, monkeypatch):
    """A local question, answered locally: there is no hosted side to have failed. The full
    rule, in both directions, is tests/cli/test_hosted_notes.py's."""
    local = registry.new_run_dir("demo")
    _heartbeat(local, target="word-overlap")

    result = runner.invoke(app, ["runs", "ls"])
    assert result.exit_code == 0
    assert local.name in result.stdout
    assert "not signed in" not in result.stderr


def test_logged_out_org_errors_and_names_the_fix(runs_dir):
    result = runner.invoke(app, ["runs", "ls", "--org"])
    assert result.exit_code != 0
    assert "auth login" in result.output


# --- a working set, not an archive ----------------------------------------------------------- #

def _many(runs_dir, count, *, target="word-overlap", eval_name="demo", month=1):
    """`count` recorded runs, oldest first, each a minute apart so ordering is unambiguous."""
    made = []
    for index in range(count):
        run_dir = runs_dir / f"2026{month:02d}01-1200{index:02d}__{eval_name}__x{index:04d}"
        run_dir.mkdir()
        started = f"2026-{month:02d}-01T12:{index:02d}:00+00:00"
        _cell(run_dir, target, eval_name, started_at=started)
        _heartbeat(run_dir, target=target, eval_name=eval_name, started=started)
        made.append(run_dir.name)
    return made


def test_a_bare_listing_shows_a_working_set(runs_dir):
    from memrank.cli.runs import DEFAULT_LIMIT

    _many(runs_dir, DEFAULT_LIMIT + 5)
    result = runner.invoke(app, ["runs", "ls"])

    rows = [ln for ln in result.stdout.splitlines() if ln.strip()][1:]   # drop the header
    assert len(rows) == DEFAULT_LIMIT


def test_a_truncated_listing_says_it_is_truncated(runs_dir):
    """A partial answer that looks complete is the defect; the cap must be visible."""
    _many(runs_dir, 25)
    result = runner.invoke(app, ["runs", "ls"])

    assert "20 most recent" in result.stderr
    assert "--limit 0" in result.stderr


def test_nothing_is_said_when_nothing_was_dropped(runs_dir):
    _many(runs_dir, 3)
    assert "most recent" not in runner.invoke(app, ["runs", "ls"]).stderr


def test_limit_zero_shows_everything(runs_dir):
    _many(runs_dir, 25)
    rows = [ln for ln in runner.invoke(app, ["runs", "ls", "--limit", "0"]).stdout.splitlines()
            if ln.strip()][1:]
    assert len(rows) == 25


def test_filters_apply_before_the_cap(runs_dir):
    """Otherwise `--target mem0` answers "none" whenever the newest 20 are all baseline --
    a filter silently returning the wrong answer."""
    old = _many(runs_dir, 1, target="mem0", month=1)[0]
    _many(runs_dir, 25, target="word-overlap", month=6)

    result = runner.invoke(app, ["runs", "ls", "--target", "mem0"])
    assert old in result.stdout


def test_json_is_capped_the_same_way(runs_dir):
    """A script and a human must see the same set, or one of them is reasoning about a
    different listing than the other."""
    from memrank.cli.runs import DEFAULT_LIMIT

    _many(runs_dir, 25)
    payload = json.loads(runner.invoke(app, ["runs", "ls", "--json"]).stdout)
    assert len(payload) == DEFAULT_LIMIT


def test_the_limit_bounds_what_the_platform_is_asked_for(signed_in, runs_dir):
    """The API pages, so an unbounded ask would truncate invisibly at its own default.

    Bounded *and* larger than the display cap, deliberately: platform rows that duplicate local
    ones collapse in the merge, so fetching exactly `limit` would leave the listing short by
    however many collapsed.
    """
    signed_in["body"] = {"runs": [], "scope": "mine"}
    runner.invoke(app, ["runs", "ls", "--limit", "7"])

    asked = signed_in["calls"][0]["limit"]
    assert 7 <= asked <= 100, f"asked the platform for {asked}"


# --- the credential is read once, and a store that refuses says so ------------------------- #

def test_the_credential_is_read_at_most_once_per_command(runs_dir, monkeypatch):
    """It used to be read twice: a `_signed_in()` probe answering a boolean, then the call that
    actually spends it. On macOS that is two Keychain prompts, one of them for nothing."""
    from memrank.accounts import credentials

    reads: list[int] = []

    class Counting(credentials.CredentialStore):
        def load(self):
            reads.append(1)
            return None

    monkeypatch.setattr(credentials, "CredentialStore", Counting)
    runner.invoke(app, ["runs", "ls"])

    assert len(reads) <= 1, f"read the credential {len(reads)} times"


def test_a_refused_credential_store_is_not_reported_as_logged_out(runs_dir, monkeypatch):
    """`gh` ships this bug (cli/cli#13317): a failed keychain read becomes a silent
    unauthenticated request. A denied prompt is a local permission problem -- saying "not signed
    in" sends the user to re-authenticate for nothing, and "API unreachable" blames the network."""
    from memrank.accounts import credentials

    class Refusing(credentials.CredentialStore):
        def load(self):
            raise credentials.CredentialError("the credential store refused to answer (denied)")

    monkeypatch.setattr(credentials, "CredentialStore", Refusing)
    local = registry.new_run_dir("demo")
    _heartbeat(local, target="word-overlap")

    result = runner.invoke(app, ["runs", "ls"])
    assert result.exit_code == 0                      # a local listing still works
    assert "credential store refused" in result.stderr
    assert "not signed in" not in result.stderr
    assert "unreachable" not in result.stderr


def test_a_refused_credential_store_still_fails_org_loudly(runs_dir, monkeypatch):
    """--org is an explicitly remote ask; it must not degrade whatever the reason."""
    from memrank.accounts import credentials

    class Refusing(credentials.CredentialStore):
        def load(self):
            raise credentials.CredentialError("the credential store refused to answer (denied)")

    monkeypatch.setattr(credentials, "CredentialStore", Refusing)
    assert runner.invoke(app, ["runs", "ls", "--org"]).exit_code != 0


# --- the SYNCED column: what the org has, and what it lacks ------------------------------------ #

def test_a_finished_local_run_reads_pending_until_it_is_synced(runs_dir):
    """The column is how a reader SEES what the org lacks without having to know."""
    run_dir = registry.new_run_dir("demo")
    _cell(run_dir, "word-overlap")
    _heartbeat(run_dir, target="word-overlap")

    output = runner.invoke(app, ["runs", "ls"]).stdout
    assert "SYNCED" in output
    assert "pending" in output


def test_a_synced_local_run_reads_yes(runs_dir):
    run_dir = registry.new_run_dir("demo")
    _cell(run_dir, "word-overlap")
    _heartbeat(run_dir, target="word-overlap")
    data = json.loads((run_dir / "status.json").read_text())
    data["synced_org"] = "acme"
    (run_dir / "status.json").write_text(json.dumps(data))

    output = runner.invoke(app, ["runs", "ls"]).stdout
    assert "pending" not in output
    assert "yes" in output


def test_a_running_run_has_nothing_to_sync_yet(runs_dir):
    import os

    run_dir = registry.new_run_dir("demo")
    _heartbeat(run_dir, target="word-overlap", state="running", beating=True, pid=os.getpid())

    output = runner.invoke(app, ["runs", "ls"]).stdout
    assert "pending" not in output


def test_the_json_form_carries_synced(runs_dir):
    run_dir = registry.new_run_dir("demo")
    _heartbeat(run_dir, target="word-overlap")

    payload = json.loads(runner.invoke(app, ["runs", "ls", "--json"]).stdout)
    assert payload[0]["synced"] == "pending"


def test_a_platform_row_reports_where_it_actually_ran(signed_in, runs_dir):
    """A synced local run listed by the org is still a local run -- place is a fact about it."""
    signed_in["body"] = {"runs": [_platform_run("r-local") | {"place": "local"},
                                  _platform_run("r-cloud")],
                         "scope": "mine"}

    rows = {r["run_id"]: r for r in
            json.loads(runner.invoke(app, ["runs", "ls", "--json"]).stdout)}

    assert rows["r-local"]["place"] == "local"
    assert rows["r-cloud"]["place"] == "cloud"      # no `place` key: a pre-sync deployment
    assert {r["synced"] for r in rows.values()} == {"yes"}


def test_show_of_a_synced_local_run_does_not_invent_a_cluster(runs_dir, monkeypatch):
    """`place` is what keeps a number honest about where it came from -- including in `show`."""
    from memrank.cli import runs_show as runs_show_cli

    monkeypatch.setattr(runs_show_cli, "platform_record", lambda run_id: {
        "id": run_id, "state": "stopped-success", "target_ref": "word-overlap",
        "benchmark": "demo", "place": "local", "created_at": "2026-08-03T09:00:00+00:00",
        "artifact": {"bucket": "", "prefix": ""}})

    output = runner.invoke(app, ["runs", "show", "20260803-090000__demo__x"]).output

    assert "place    local" in output
    assert "cloud" not in output and "cluster" not in output


def test_an_unreachable_platform_does_not_label_cloud_runs_stale(runs_dir, monkeypatch):
    """The 2026-08-03 report: a 500 from the org listing turned every cloud run `stale`.

    Degrading to local records is correct and the note says so -- but a local heartbeat is not
    evidence about a task running elsewhere, and `stale` is the word `kill` uses for a corpse.
    """
    from memrank.cli import runs as runs_cli

    run_dir = registry.new_run_dir("locomo")
    _heartbeat(run_dir, target="hindsight", eval_name="locomo", state="running")
    data = json.loads((run_dir / "status.json").read_text())
    data["placement"] = "cloud"
    # Aware and hours old: submitted, never watched -- exactly the fleet in the report, and
    # not the separate naive-timestamp path.
    data["updated_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    (run_dir / "status.json").write_text(json.dumps(data))

    def unreachable(*args, **kwargs):
        raise runs_cli.run_api_client.RunApiError("Internal Server Error (500)")

    monkeypatch.setattr(runs_cli.run_api_client, "authenticated_client", unreachable)
    result = runner.invoke(app, ["runs", "ls"])

    assert result.exit_code == 0, "a broken platform must never break a local listing"
    assert "stale" not in result.output
    assert UNKNOWN_STATE in result.output


# --- one question, one answer ------------------------------------------------------------------ #

def test_ps_and_runs_ls_live_are_the_same_command(monkeypatch, tmp_path):
    """The defect this closes: they could disagree about what was running.

    `ps` read local status.json files only, so a cloud sweep submitted from another directory was
    invisible to it while `runs ls --live` showed it 89% complete. Same implementation now, so the
    two cannot drift apart again -- which is what the interface model means by a flat ALIAS.
    """
    from typer.testing import CliRunner

    from memrank.runner import app

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    invoke = CliRunner().invoke

    assert invoke(app, ["ps", "--json"]).stdout == invoke(app, ["runs", "ls", "--live", "--json"]).stdout


def test_ps_all_matches_an_unfiltered_listing(monkeypatch, tmp_path):
    """`--all` is `--live` off, not a second notion of what counts as a run."""
    from typer.testing import CliRunner

    from memrank.runner import app

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    invoke = CliRunner().invoke

    assert invoke(app, ["ps", "--all", "--json"]).stdout == invoke(app, ["runs", "ls", "--json"]).stdout


#: The hot-path aliases the interface model names, and the `runs` verb each one aliases. Exactly
#: these: a flat name the model does not list spends a word the noun-spaces own (`status` meant
#: one run here, the session there, and machine readiness in `doctor`), and an alias whose
#: canonical form is unregistered is the only form of the command it aliases (`logs`, until
#: `runs logs` existed).
FLAT_ALIASES = {"ps": "ls", "watch": "watch", "logs": "logs", "kill": "kill"}


def test_every_flat_alias_has_its_runs_twin_and_no_others_exist():
    """Enumerated over the live Typer tree, so a new flat command cannot be added silently."""
    # Hidden ones count: the engine-hosting path is hidden, not unregistered (ATO-2343), and
    # its aliases must stay paired for as long as its code stays.
    from memrank.cli.retired import RETIRED
    from memrank.cli.runs import runs_app
    from memrank.runner import app

    flat = {c.name for c in app.registered_commands} - set(RETIRED)
    twins = {c.name for c in runs_app.registered_commands}

    # The flat surface is the aliases plus the commands that are nobody's alias: `run` and
    # `serve` (evaluating an agent, decision 0034), `submit`, and `version`. `preflight` used to
    # sit here; it retired into `targets show` when `doctor` was dropped from the model.
    assert flat - {"run", "serve", "submit", "version"} == set(FLAT_ALIASES)
    assert set(FLAT_ALIASES.values()) <= twins
