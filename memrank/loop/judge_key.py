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
"""The organisation's Anthropic key for a judged run, and asking for it when there is none.

Decision 0037: ``memrank run`` judges with the organisation's own saved ``ANTHROPIC_API_KEY``.
An organisation without one is a step the user has not taken yet, not a failure, so at a
terminal the run asks for the key, checks it with one small Anthropic call, saves it the way
``memrank secrets set --org`` does and carries on. Where nobody can answer -- a script, CI, a
coding agent -- it says what to run instead. The key is never echoed, logged or written
anywhere but the organisation's secrets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from memrank.accounts.run_secrets import (
    OrgSecretsError,
    load_org_secrets,
    save_org_secret,
    session_rejected,
)
from memrank.errors import ActionRequired
from memrank.judging.client import JUDGE_SECRET
from memrank.outcome import Step

#: What a user runs to save the key themselves (or asks an owner to).
_SAVE = "memrank secrets set " + JUDGE_SECRET + " --org {org}"
_NO_JUDGE = "memrank run ... --no-judge"

#: The user's own command with ``--no-judge``, built only when a refusal shows it: reading the
#: command line back is the CLI's business and cannot fail a run that never needs it.
Rerun = Callable[[], str]


class JudgeKeyMissing(ActionRequired):
    """A judged run was asked for and the organisation has no Anthropic key saved."""


@dataclass
class KeyPrompt:
    """How a run at a terminal asks for the key: every part injected, so a test decides what
    is typed and what Anthropic answers with no terminal and no network.

    Attributes:
        read: Ask for the key with the input hidden; ``""`` when the user cancels.
        verify: ``None`` when Anthropic accepts the key, else why it did not.
        echo: Narration to the user.
    """

    read: Callable[[str], str]
    verify: Callable[[str], str | None]
    echo: Callable[[str], None]


def _rerun_line(rerun: Rerun | None) -> str:
    return rerun() if rerun is not None else _NO_JUDGE


def _unscored(rerun: Rerun | None) -> Step:
    return Step("Or run without a quality score:", (_rerun_line(rerun),))


def missing_key(org: str, rerun: Rerun | None = None) -> JudgeKeyMissing:
    """The refusal of a judged run when ``org`` has no judge key and nobody can be asked."""
    return JudgeKeyMissing(
        f"{org} has no Anthropic key saved, and memrank run judges answers with your "
        "organisation's own key.",
        steps=(Step("Save one:", (_SAVE.format(org=org),)), _unscored(rerun)))


def judge_key(http: Any, org: str, ask: KeyPrompt | None, rerun: Rerun | None) -> str:
    """``org``'s saved key, asking for one through ``ask`` when there is none."""
    try:
        key = load_org_secrets(http, org).get(JUDGE_SECRET)
    except OrgSecretsError as exc:
        if exc.status == 401:
            raise session_rejected() from exc
        if exc.status == 403:  # membership is already proven: this is the role
            raise ActionRequired(
                f"Judged runs use {org}'s own Anthropic key, and only an owner of {org} can "
                "read or save it.",
                steps=(Step(f"Ask an owner of {org} to save one:", (_SAVE.format(org=org),)),
                       _unscored(rerun))) from exc
        raise
    if key:
        return key
    if ask is None:
        raise missing_key(org, rerun)
    return _obtain(http, org, ask, rerun)


def _cancelled(rerun: Rerun | None) -> ActionRequired:
    return ActionRequired("No key was saved, so the run did not start.",
                          steps=(Step("To run without a quality score:",
                                      (_rerun_line(rerun),)),))


def _accepted(ask: KeyPrompt, rerun: Rerun | None) -> str:
    """Read keys until Anthropic accepts one; an empty answer or Ctrl-C cancels."""
    try:
        while key := ask.read(JUDGE_SECRET).strip():
            refusal = ask.verify(key)
            if refusal is None:
                return key
            ask.echo(f"Anthropic did not accept that key: {refusal}")
            ask.echo("Paste another, or press Enter to cancel.")
    except KeyboardInterrupt:
        # Ctrl-C at the prompt (or during the check) is a cancel, answered as an empty line is
        raise _cancelled(rerun) from None
    raise _cancelled(rerun)


def _obtain(http: Any, org: str, ask: KeyPrompt, rerun: Rerun | None) -> str:
    ask.echo(f"{org} has no Anthropic key yet. Judged runs use your organisation's own key: "
             f"paste one and memrank checks it with Anthropic and saves it to {org}.")
    ask.echo("The input is hidden. Press Enter on an empty line to cancel.")
    key = _accepted(ask, rerun)
    save_org_secret(http, org, JUDGE_SECRET, key)
    ask.echo(f"Saved {JUDGE_SECRET} to {org}.")
    return key
