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
"""Putting a run in the organisation's run history, with the organisation's own keys.

Decision 0037, the W&B pattern: the run folder is the working record, and the finished run --
judged in this process -- is uploaded from it to the ordinary runs API
(``PUT /orgs/{org}/runs/{id}``). The upload is idempotent, so a failed one is finished by
sending the same record again -- ``memrank run --resume``. The same route registers the run as
it starts and records where it stopped (:func:`register`, :mod:`memrank.loop.live`).

A login is required before anything runs (:func:`connect`), with the stored session and the
organisation ``memrank auth login`` chose. There is no offline mode. A judged run also needs the
organisation's saved ``ANTHROPIC_API_KEY``, fetched the way ``submit --org`` fetches an org's
credentials; without it the run asks for one at a terminal, and is refused before it starts
anywhere else (:mod:`memrank.loop.judge_key`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memrank import settings
from memrank.accounts.run_secrets import SIGN_IN, session_rejected
from memrank.errors import ActionRequired
from memrank.loop.judge_key import KeyPrompt, Rerun, judge_key
from memrank.outcome import Step
from memrank.placement import run_api_client
from memrank.placement.run_api_client import RunApiError
from memrank.service.judgings import judging_of
from memrank.service.protocol import ANSWERS_FILE, RunResult, split_answers
from memrank.term.style import identity, prefixed

#: What a user runs to choose the organisation runs are recorded in.
_SET_ORG = "memrank config set defaults.org <slug>"


class LoginRequired(ActionRequired):
    """``memrank run`` needs a login and an organisation to record the run in."""


@dataclass(frozen=True)
class Account:
    """Who is signed in, and the display name of the organisation runs are recorded in --
    what a run's opening lines say, so nobody records to one org believing it is another."""

    login: str
    email: str
    org_name: str


@dataclass
class Hosted:
    """Where runs are recorded: the signed-in API client, the organisation, its judge key."""

    http: Any
    org: str
    account: Account
    #: The organisation's saved ``ANTHROPIC_API_KEY``; ``None`` when judging is off.
    judge_key: str | None = None


def _signed_in() -> Any:
    try:
        return run_api_client.authenticated_client()
    except RunApiError as exc:
        if exc.code != "no_session":
            raise  # this machine could not read the session: a failure, not a missing step
        raise LoginRequired("You are not signed in. memrank run records every run in your "
                            "organisation's run history, so it needs a login.",
                            steps=(Step("Sign in:", (SIGN_IN,)),)) from exc


def _check_membership(http: Any, org: str) -> None:
    """One cheap call that proves the session is live and ``org`` is the caller's."""
    try:
        run_api_client.list_runs(http, org, limit=1)
    except RunApiError as exc:
        if exc.code == "unauthorized":
            raise session_rejected() from exc
        if exc.code == "forbidden":
            raise _not_a_member(org) from exc
        raise


def _not_a_member(org: str) -> LoginRequired:
    return LoginRequired(
        f"You are not a member of {org}, the organisation memrank run records runs in.",
        steps=(Step("Set one you belong to (`memrank auth status` lists them):", (_SET_ORG,)),))


def _account(http: Any, org: str) -> Account:
    """Who the session belongs to (``/whoami``, as ``memrank auth status`` asks it)."""
    try:
        me = run_api_client.whoami(http)
    except RunApiError as exc:
        if exc.code == "unauthorized":
            raise session_rejected() from exc
        raise
    names = {member["slug"]: member["name"] for member in me["orgs"]}
    if org not in names:
        raise _not_a_member(org)
    return Account(login=me["login"], email=me["email"], org_name=names[org])


def connect(*, judge: bool, ask: KeyPrompt | None = None, rerun: Rerun | None = None) -> Hosted:
    """The signed-in client and organisation, and its judge key when ``judge``; refuse without.

    Args:
        judge: Whether the run is judged, and so needs the organisation's Anthropic key.
        ask: How to ask for a missing key; ``None`` where nobody is at a terminal to answer.
        rerun: The user's own command with ``--no-judge``, offered when judging cannot go ahead.
    """
    http = _signed_in()
    org = settings.get("defaults.org")
    if not org:
        raise LoginRequired("No organisation is set to record the run in.",
                            steps=(Step("Sign in, which sets your default:", (SIGN_IN,)),
                                   Step("Or name one you belong to:", (_SET_ORG,))))
    _check_membership(http, org)
    account = _account(http, org)
    key = judge_key(http, org, ask, rerun) if judge else None
    return Hosted(http=http, org=org, account=account, judge_key=key)


def _started_at(run_id: str) -> str:
    """When the run began, from its id (``<UTC YYYYMMDD-HHMMSS>__...``, minted at creation)."""
    stamp = datetime.strptime(run_id.split("__")[0], "%Y%m%d-%H%M%S")
    return stamp.replace(tzinfo=timezone.utc).isoformat()


def _payload(run_id: str, agent: str, evaluation: str, state: str,
             record: dict[str, Any]) -> dict[str, Any]:
    finished = {} if state == "running" else {"finished_at": datetime.now(timezone.utc).isoformat()}
    return {"kind": "agent", "target_ref": agent, "benchmark": evaluation, "state": state,
            "started_at": _started_at(run_id), "record": {"kind": "agent", **record},
            **finished}


def upload(hosted: Hosted, result: RunResult, echo: Callable[[str], None]) -> str | None:
    """Record the finished run in the org; return its page, when the API names one.

    Two requests, answers first: every question goes up as :data:`ANSWERS_FILE`, then the record
    -- the run's summary -- which points at it. The record is bounded; a run of any size is not.
    """
    summary, cases = split_answers(result)
    run_api_client.put_answers(hosted.http, hosted.org, result.run_id,
                               {"cases": [case.model_dump(mode="json") for case in cases]})
    payload = _payload(result.run_id, result.agent.name, result.evaluation, "done",
                       {"result": summary.model_dump(mode="json"), "answers": ANSWERS_FILE})
    stored = run_api_client.sync_run(hosted.http, hosted.org, result.run_id, payload)
    echo(prefixed(f"Recorded in {identity(hosted.org)}'s run history"))
    return stored.get("url")


def add_judging(hosted: Hosted, judged: RunResult, echo: Callable[[str], None]) -> dict[str, Any]:
    """Add ``judged`` -- a recorded run graded again -- to that run as a judging (decision 0040).

    The earlier judgings stay as they are. Returns the stored judging's summary as the API
    lists it in the run's record, and whether this call added it (``False`` when it was there).
    """
    judging = judging_of(judged)
    stored = run_api_client.add_judging(hosted.http, hosted.org, judged.run_id,
                                        judging.model_dump(mode="json"))
    verb = "Added" if stored["added"] else "Already had"
    echo(prefixed(f"{verb} the judging by {judging.judge_model} "
                  f"(prompts {judging.judge_prompt_version}) in {identity(hosted.org)}"))
    return stored


def register(hosted: Hosted, run_id: str, agent: str, evaluation: str, *, state: str,
             progress: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
    """Record a run that is ``running`` or has ``stopped`` short of its end, with its progress.

    Returns the API's answer: the run's page (``url``) and, while running, the progress channel
    (``progress``: ``{url, token}``, or ``{}`` where the deployment offers none).
    """
    payload = _payload(run_id, agent, evaluation, state, {"progress": progress})
    if reason is not None:
        payload["error"] = reason
    return run_api_client.sync_run(hosted.http, hosted.org, run_id, payload)
