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
"""Organizations, sign-in and syncing are mentioned only where the hosted side is in play.

Both directions, at both call sites that reach for the org universe while serving a request that
may be entirely local -- ``runs ls`` and the hook after each finished run. Observed in a
usability session on 2026-09-16: a local evaluation printed a note about a default org and one
about not being signed in, directly where the person was reading a score.

The rule these pin is :mod:`memrank.placement.hosted`'s: absent is silent, broken is not.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memrank import settings
from memrank.placement import hosted, run_api_client
from memrank.runner import app

runner = CliRunner()

#: Every word a purely local run must never print. `auth login` and `defaults.org` are the two
#: commands the old notes sent people to; "sync" catches the post-run hook's phrasing.
CLOUD_WORDS = ("org", "signed in", "auth login", "sync")


def _assert_says_nothing_hosted(output: str) -> None:
    for word in CLOUD_WORDS:
        assert word not in output.lower(), f"a local run mentioned {word!r}: {output!r}"


@pytest.fixture
def runs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


@pytest.fixture
def config(monkeypatch, tmp_path):
    """A config dir of this test's own, with every setting's env override cleared."""
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    for spec in settings.SETTINGS:
        monkeypatch.delenv(spec.env, raising=False)


def _local_run(runs_dir, run_id="20260916-200342__demo__aff956"):
    """A finished local run as the machine leaves it: heartbeat plus one cell."""
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps(
        {"run_id": run_id, "pid": None, "target": "word-overlap", "benchmark": "demo",
         "slice": None, "state": "done", "progress": {}, "message": "",
         "started_at": "2026-09-16T20:03:42+00:00", "updated_at": "2026-09-16T20:04:02+00:00",
         "error": None}))
    (run_dir / "word-overlap__demo.json").write_text(json.dumps(
        {"adapter": "word-overlap", "benchmark": "demo", "composite": 0.8,
         "receipt": {"adapter_name": "word-overlap", "benchmark_name": "demo",
                     "config_hash": "abc123", "started_at": "2026-09-16T20:03:42+00:00"}}))
    return run_dir


def _session(monkeypatch, *, org: str | None, fail=None):
    """Stand in for a machine that has a session. ``org`` is its ``defaults.org``, if any."""
    if org is not None:
        settings.put("defaults.org", org)

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _list(http, org_slug, *, mine=True, limit=50):
        if fail is not None:
            raise fail
        return {"runs": [], "scope": "mine"}

    def _sync(http, org_slug, run_id, payload):
        if fail is not None:
            raise fail
        return {"id": run_id, "place": "local"}

    monkeypatch.setattr(run_api_client, "authenticated_client", lambda: _Client())
    monkeypatch.setattr(run_api_client, "list_runs", _list)
    monkeypatch.setattr(run_api_client, "sync_run", _sync)


def _signed_out(monkeypatch):
    def refuse():
        raise run_api_client.RunApiError("not signed in -- run `memrank auth login`",
                                         code="no_session")

    monkeypatch.setattr(run_api_client, "authenticated_client", refuse)


# --- the listing ----------------------------------------------------------------------------- #

def test_a_signed_out_listing_says_nothing_about_the_hosted_side(config, runs_dir, monkeypatch):
    """The note quoted in the ticket. Nothing hosted was asked for and nothing hosted exists."""
    _signed_out(monkeypatch)
    local = _local_run(runs_dir)

    result = runner.invoke(app, ["runs", "ls"])

    assert result.exit_code == 0
    assert local.name in result.stdout
    _assert_says_nothing_hosted(result.stderr)


def test_a_listing_with_no_configured_org_says_nothing_either(config, runs_dir, monkeypatch):
    """The other half of the same silence: a session exists but names no universe to read."""
    _session(monkeypatch, org=None)
    local = _local_run(runs_dir)

    result = runner.invoke(app, ["runs", "ls"])

    assert result.exit_code == 0
    assert local.name in result.stdout
    _assert_says_nothing_hosted(result.stderr)


def test_a_configured_listing_still_reports_an_unreachable_api(config, runs_dir, monkeypatch):
    """The other direction. This machine IS set up for an org and did not get what it expects,
    so the listing it shows is incomplete and saying so is the whole point."""
    _session(monkeypatch, org="acme", fail=run_api_client.RunApiError("boom"))
    local = _local_run(runs_dir)

    result = runner.invoke(app, ["runs", "ls"])

    assert result.exit_code == 0
    assert local.name in result.stdout
    assert "showing this machine's runs" in result.stderr


def test_org_remains_an_explicitly_hosted_ask_and_still_refuses(config, runs_dir, monkeypatch):
    """Silence is about a LOCAL question. `--org` asks a remote one, so absence is the answer."""
    _signed_out(monkeypatch)
    _local_run(runs_dir)

    result = runner.invoke(app, ["runs", "ls", "--org"])

    assert result.exit_code != 0
    assert "auth login" in result.output


# --- the hook after a finished run ----------------------------------------------------------- #

def test_a_finished_local_run_is_not_told_it_was_not_synced(config, runs_dir, monkeypatch, capsys):
    """Signed in, no org -- the second note the 2026-09-16 sessions printed under a score."""
    from memrank.runs import push

    _session(monkeypatch, org=None)
    run_dir = _local_run(runs_dir)

    push.auto_sync(run_dir)

    _assert_says_nothing_hosted(capsys.readouterr().err)


def test_a_signed_out_finished_run_is_silent_too(config, runs_dir, monkeypatch, capsys):
    from memrank.runs import push

    _signed_out(monkeypatch)
    run_dir = _local_run(runs_dir)

    push.auto_sync(run_dir)

    _assert_says_nothing_hosted(capsys.readouterr().err)


def test_a_run_that_had_an_org_and_missed_it_is_reported(config, runs_dir, monkeypatch, capsys):
    """The other direction: an org was configured, the push failed, and a retry exists."""
    from memrank.runs import push

    _session(monkeypatch, org="acme", fail=run_api_client.RunApiError("boom"))
    run_dir = _local_run(runs_dir)

    push.auto_sync(run_dir)

    captured = capsys.readouterr().err
    assert "not synced to the org" in captured
    assert "memrank runs sync" in captured


# --- the gate itself ------------------------------------------------------------------------- #

@pytest.mark.parametrize("problem", [hosted.no_org(),
                                     hosted.refused(run_api_client.RunApiError(
                                         "not signed in", code="no_session"))])
def test_an_absent_hosted_side_is_not_worth_saying(problem):
    assert problem.absent
    assert not hosted.worth_saying(problem)
    assert hosted.worth_saying(problem, asked_for_hosted=True)


@pytest.mark.parametrize("problem", [
    hosted.unreachable(RuntimeError("no route to host")),
    hosted.refused(run_api_client.RunApiError("the credential store refused to answer",
                                              code="credential_store")),
    hosted.refused(run_api_client.RunApiError("you are not a member of this org")),
])
def test_a_broken_hosted_side_is_always_worth_saying(problem):
    """A refused credential store is the one that must not be mistaken for absence: it means
    this machine could not READ a session, not that there is none (cli/cli#13317)."""
    assert not problem.absent
    assert hosted.worth_saying(problem)
