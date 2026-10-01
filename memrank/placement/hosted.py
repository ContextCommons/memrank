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
#
# Used by the current path (cli/runs, runs/push): it sits in a deprecated package but is not
# deprecated itself, and it must not warn.
"""When the hosted side could not be consulted, and whether that is worth saying.

Two paths reach for the org universe while serving a request that may be entirely local:
``runs ls`` merges the org's rows into this machine's own, and the hook after each finished run
pushes it to the org. Both come back empty-handed on a machine that was never set up for the
hosted side, and both used to say so -- which put a line about organizations and ``auth login``
under every local evaluation, at the moment the person was trying to read a score.

The distinction that decides it is **absent** versus **broken**:

* *Absent* -- not signed in, or no ``defaults.org`` -- is not a fault. Nothing hosted is in play,
  the question asked was local, and the answer given was local. There is nothing to report.
* *Broken* -- a credential store that refused, a 403, an unreachable API -- is a fault worth
  reporting, because this machine **is** configured for the hosted side and did not get what it
  is configured to get.

An explicitly hosted ask (``runs ls --org``, ``runs sync``) overrides the distinction: there,
absence is the whole answer and staying quiet would leave the person with nothing.

Decided here rather than at each call site so the paths cannot drift apart, and so a third one
inherits the rule instead of re-deriving it.
"""
from __future__ import annotations

from dataclasses import dataclass

#: No ``defaults.org``. Not one of ``RunApiError``'s codes -- the API never refuses for this,
#: the CLI declines to guess at an org -- so it is spelled once here and shared.
NO_ORG = "no_org"

#: Why the hosted side is absent rather than broken. ``no_session`` is
#: :mod:`memrank.placement.run_api_client`'s code for "nothing is stored"; a credential store
#: that *refused* carries ``credential_store`` instead and is deliberately not in this set.
ABSENT_CODES = frozenset({NO_ORG, "no_session"})


@dataclass(frozen=True)
class Unavailable:
    """Why the hosted side could not be consulted.

    ``message`` states the PROBLEM only, never its consequence: the same problem means "showing
    this machine's runs" to a bare listing and "cannot answer at all" to ``--org``, and a message
    that assumed one told the other user something untrue.
    """

    message: str
    code: str = ""

    @property
    def absent(self) -> bool:
        """Whether this machine simply has no hosted side, as opposed to a failing one."""
        return self.code in ABSENT_CODES

    def __str__(self) -> str:
        return self.message


def no_org() -> Unavailable:
    """No org is configured, so there is no universe to consult."""
    return Unavailable("no default org is configured -- "
                       "`memrank config set defaults.org <slug>`", code=NO_ORG)


def refused(exc) -> Unavailable:
    """A :class:`~memrank.placement.run_api_client.RunApiError`, keeping its code.

    Its message is already phrased for a person -- not signed in, the credential store would not
    answer, a 403 -- so it is passed through rather than wrapped in "could not be listed", which
    would bury the reason under a restatement of the symptom.
    """
    return Unavailable(str(exc), code=getattr(exc, "code", "") or "")


def unreachable(exc) -> Unavailable:
    """The API could not be reached at all. Broken, never absent: something is wired up."""
    return Unavailable(f"the memrank API is unreachable: {exc}")


def worth_saying(unavailable: Unavailable, *, asked_for_hosted: bool = False) -> bool:
    """Whether the person should be told. The one gate; see the module docstring."""
    return asked_for_hosted or not unavailable.absent
