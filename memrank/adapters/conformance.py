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
"""Check a running translator against the system contract (``docs/system-contract.md``).

This exists because memrank ships as an isolated ``uv tool`` install, so somebody writing a
translator cannot run memrank's own pytest suite against it. Without this, their first signal that
the contract is violated would be a benchmark number -- and the failures that matter most here are
the ones that produce a plausible number rather than an error: state leaking between isolation
units, or an empty result standing in for a failure.

Deliberately narrow. It asserts the published contract and nothing else -- not whether the machine
is healthy, not whether credentials exist, not whether the engine is any good.
memrank has no general ``doctor`` command, deliberately, and the reason that decision was made
applies here too: every check in this module is about a third party's implementation of a spec,
which nothing else in the tool examines.
"""

from __future__ import annotations

from memrank.core import REQUIRED_TOKEN_KEYS, Document, Memory
from memrank.docs import doc_url
from memrank.placement.base import Requirement

_DOC = doc_url("system-contract.md")
_UNIT_A = "memrank-verify-unit-a"
_UNIT_B = "memrank-verify-unit-b"
_USER = "memrank-verify-user"
_MATCH_ID = "memrank-verify-match"
_QUERY = "where did we agree to meet at dusk"


def _probe_documents() -> list[Document]:
    """One document that answers the probe query, and one that must not outrank it."""
    return [
        Document(id=_MATCH_ID, user_id=_USER,
                 content="we agreed to meet at dusk by the harbour lighthouse"),
        Document(id="memrank-verify-distractor", user_id=_USER,
                 content="an unrelated note about migrating a database schema"),
    ]


def _describe(adapter: Memory) -> Requirement:
    """The handshake: version, identity, components and capabilities all stated."""
    try:
        reported = adapter.describe_engine()
    except Exception as exc:  # noqa: BLE001 - every failure is a finding, not a crash
        return Requirement("describe", False, str(exc), f"see {_DOC} section 4")
    if reported is None:
        return Requirement("describe", False, "reported nothing",
                           f"GET /memrank/v1/describe must answer; see {_DOC} section 4")
    return Requirement("describe", True,
                       f"engine {adapter.engine_version}, graph_snapshot={adapter.graph_capable}",
                       "")


def _ingest(adapter: Memory) -> Requirement:
    """Open a unit and load the probe documents."""
    try:
        adapter.prepare(_UNIT_A)
        adapter.ingest(_probe_documents())
    except Exception as exc:  # noqa: BLE001
        return Requirement("ingest", False, str(exc), f"see {_DOC} section 4")
    return Requirement("ingest", True, "accepted 2 documents", "")


def _ranking(adapter: Memory) -> Requirement:
    """Retrieve must return documents ranked best-first -- that ordering IS the measurement."""
    try:
        documents = adapter.retrieve(_QUERY, k=5, user_id=_USER).documents
    except Exception as exc:  # noqa: BLE001
        return Requirement("retrieve", False, str(exc), f"see {_DOC} section 4")
    if not documents:
        return Requirement(
            "retrieve", False, "returned nothing for a query whose answer was just ingested",
            "an empty list means 'nothing matched', so a failure disguised as one scores like the "
            f"no-memory control arm; raise instead (see {_DOC} section 8)")
    if documents[0].id != _MATCH_ID:
        return Requirement(
            "retrieve", False, f"ranked {documents[0].id!r} above the matching document",
            f"documents must be ordered best-first; recall@k reads the order (see {_DOC})")
    return Requirement("retrieve", True,
                       f"ranked the matching document first of {len(documents)}", "")


def _isolation(adapter: Memory) -> Requirement:
    """A fresh unit must not see the previous one's documents.

    The most consequential check here. Leaked state does not raise -- it inflates every score after
    the first, because the engine has already been shown the answers.
    """
    try:
        adapter.cleanup()
        adapter.prepare(_UNIT_B)
        leaked = adapter.retrieve(_QUERY, k=5, user_id=_USER).documents
        adapter.cleanup()
    except Exception as exc:  # noqa: BLE001
        return Requirement("isolation", False, str(exc), f"see {_DOC} section 4")
    if leaked:
        return Requirement(
            "isolation", False,
            f"a fresh unit returned {len(leaked)} document(s) ingested under the previous one",
            "prepare() must give each isolation_unit its own namespace, collection or tenant; "
            "leaked state inflates every score after the first")
    return Requirement("isolation", True, "a fresh unit saw nothing from the previous one", "")


def _metrics(adapter: Memory) -> Requirement:
    """The engine's usage DECLARATION must be well shaped, where it makes one.

    Latency is no longer asked of the engine: memrank times every ingest and retrieve at its own
    call boundary, so there is nothing here for an engine to get wrong or to flatter (ATO-2136).
    What remains is the one measurement only the engine can make -- what a provider billed it --
    and this checks the shape of that declaration, not whether one was made.
    """
    try:
        tokens = adapter.token_metrics()
    except Exception as exc:  # noqa: BLE001
        return Requirement("metrics", False, str(exc), f"see {_DOC} section 5")
    missing = sorted(REQUIRED_TOKEN_KEYS - set(tokens))
    if missing:
        return Requirement("metrics", False, f"missing {', '.join(missing)}",
                           "these keys are what the leaderboard reads")
    counted = [key for key, value in tokens.items() if value is not None]
    detail = f"token usage reported ({len(counted)} keys)" if counted else (
        "token usage not reported, recorded as unmeasured rather than zero")
    return Requirement("metrics", True, detail, "")


def contract_checks(adapter: Memory) -> list[Requirement]:
    """Exercise the contract against a live translator and report every finding.

    Args:
        adapter: A constructed adapter pointed at the running translator.

    Returns:
        One :class:`Requirement` per check, in the order they were exercised. Reported rather than
        raised so one pass names everything wrong, matching how placements report preconditions.
    """
    describe = _describe(adapter)
    if not describe.ok:
        # Nothing downstream is meaningful: without a valid handshake memrank does not know what
        # it is talking to, and every later failure would be a consequence of this one.
        return [describe]
    ingest = _ingest(adapter)
    if not ingest.ok:
        return [describe, ingest]
    return [describe, ingest, _ranking(adapter), _isolation(adapter), _metrics(adapter)]
