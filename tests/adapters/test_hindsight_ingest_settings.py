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
"""Hindsight states its bank-creation settings, and proves the engine took them (audit F6, F7).

`enable_observations` decides which of hindsight's four memory networks are ever written, and it is
fixed when the bank is created -- no retrieval setting can reach it. AMB disables it to produce the
published numbers while the shipped product leaves it on, so the faithful `hindsight` was
reproducing a
configuration it did not have.

Two failure modes are asserted against, because both produce a NUMBER rather than an error:

  * the engine silently ignoring the setting (the bank PUT accepts unknown fields and echoes none
    of them back), and
  * retain being queued rather than settled, which recalls from a half-filled bank and scores as
    poor recall.

Request shapes here were observed against ghcr.io/vectorize-io/hindsight, not assumed.
"""

from __future__ import annotations

from typing import Any

import pytest

from memrank.adapters import errors as adapter_errors
from memrank.adapters.hindsight import Hindsight
from memrank.core import Document

SETTLED = {"success": True, "bank_id": "b", "items_count": 1, "async": False,
           "usage": {"total_tokens": 524}}


class _Response:
    # A real response carries a status, and the adapters now read it instead of calling a
    # no-op `raise_for_status`. Stating 200 here is what makes this double honest: the old
    # fake could not have represented a refusal at all.
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


class _Client:
    """Records requests; answers GET /config with whatever the bank is said to have taken."""

    def __init__(self, *, effective: dict[str, Any] | None = None,
                 retain: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._effective = effective if effective is not None else {}
        self._retain = retain if retain is not None else SETTLED

    def put(self, url: str, json: dict[str, Any]) -> _Response:  # noqa: A002
        self.calls.append(("PUT", url, json))
        return _Response({"bank_id": "b", "name": json.get("name", "")})

    def get(self, url: str) -> _Response:
        self.calls.append(("GET", url, {}))
        return _Response({"bank_id": "b", "config": self._effective})

    def post(self, url: str, json: dict[str, Any]) -> _Response:  # noqa: A002
        self.calls.append(("POST", url, json))
        return _Response(self._retain)

    def delete(self, url: str) -> _Response:
        self.calls.append(("DELETE", url, {}))
        return _Response({})


def _adapter(client: _Client, **kwargs: Any) -> Hindsight:
    adapter = Hindsight(base_url="http://stub", api_key="k", **kwargs)
    adapter._client = client  # type: ignore[assignment]
    return adapter


def _bank_put(client: _Client) -> dict[str, Any]:
    return [call for call in client.calls if call[0] == "PUT"][-1][2]


# ------------------------------------------------------------------ #
# F6 -- the setting is stated, and only when a target states it
# ------------------------------------------------------------------ #

def test_a_target_that_disables_observations_says_so_at_bank_creation():
    client = _Client(effective={"enable_observations": False})
    _adapter(client, ingest={"enable_observations": False}).prepare("u1")
    assert _bank_put(client)["enable_observations"] is False


def test_a_target_that_states_nothing_sends_nothing():
    """Unset means the engine's own default. Sending an invented value would misreport the run."""
    client = _Client()
    _adapter(client).prepare("u1")
    assert "enable_observations" not in _bank_put(client)


def test_stating_nothing_does_not_even_ask_the_engine_what_it_took():
    """No claim, nothing to verify -- matched mode must not pay for a round trip it does not need."""
    client = _Client()
    _adapter(client).prepare("u1")
    assert not [call for call in client.calls if call[0] == "GET"]


def test_a_setting_this_adapter_cannot_send_is_refused_before_the_run_starts():
    """The hole the live check found: an unknown key was dropped in silence.

    The manifest does not police an `ingest:` block's keys by design, and the adapter forwards only
    the ones it knows -- so a misspelled setting sent nothing, the bank kept its default, and the
    verification below passed because nothing had been claimed. A complete, plausible, wrong run.
    """
    with pytest.raises(ValueError, match="cannot send ingest settings"):
        Hindsight(base_url="http://stub", ingest={"enable_obserrvations": False})


# ------------------------------------------------------------------ #
# F6 -- a 200 is not evidence
# ------------------------------------------------------------------ #

def test_a_bank_that_ignored_the_setting_stops_the_run():
    """The PUT accepts unknown fields and echoes none back, so 200 proves only that it parsed.

    `ConfigurationNotTaken` rather than a bare `RuntimeError`: this is a live engine disagreeing
    with the target that declared it, and a builtin exception reached the CLI's "this is a bug in
    memrank" branch -- sending the reader to the harness for an engine's answer.
    """
    client = _Client(effective={"enable_observations": True})   # asked false, engine kept true
    adapter = _adapter(client, ingest={"enable_observations": False})
    with pytest.raises(adapter_errors.ConfigurationNotTaken, match="did not take the configuration"):
        adapter.prepare("u1")


def test_the_engine_is_asked_what_it_actually_took():
    client = _Client(effective={"enable_observations": False})
    _adapter(client, ingest={"enable_observations": False}).prepare("u1")
    reads = [call for call in client.calls if call[0] == "GET"]
    assert reads and reads[-1][1].endswith("/config")


# ------------------------------------------------------------------ #
# F7 -- retain is settled before anything recalls
# ------------------------------------------------------------------ #

def _ingest_one(retain: dict[str, Any]) -> None:
    adapter = _adapter(_Client(retain=retain))
    adapter.prepare("u1")
    adapter.ingest([Document(id="d1", content="Caroline studies counseling.", user_id="u1")])


def test_a_settled_retain_is_accepted():
    _ingest_one(SETTLED)        # the observed shape of a synchronous retain


def test_a_queued_retain_stops_the_run_instead_of_scoring_as_poor_recall():
    """Observed: an async retain answers with an operation_id and no usage."""
    with pytest.raises(RuntimeError, match="queued retain"):
        _ingest_one({"success": True, "bank_id": "b", "items_count": 1, "async": True,
                     "operation_id": "60dc58e9"})


def test_a_retain_that_did_not_report_success_stops_the_run():
    with pytest.raises(RuntimeError, match="did not report success"):
        _ingest_one({"success": False, "bank_id": "b", "items_count": 0, "async": False})


def test_a_partial_retain_stops_the_run():
    """Fewer items stored than sent is a smaller corpus, which scores as a worse engine."""
    with pytest.raises(RuntimeError, match="retained 0 of 1"):
        _ingest_one({"success": True, "bank_id": "b", "items_count": 0, "async": False})
