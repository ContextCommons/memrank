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
"""A failure on the agent's or the judge's side, said in plain words: whose side, and what to do.

The raw payload says exactly what happened, and it stays -- whole, after the lead. What it does
not say is who has to act: an Anthropic 400 about a credit balance, relayed through a shipped
agent's 502, reads like memrank breaking when it is the account behind the key this machine
gave the agent. The lead sentence says that, and what to check; the detail follows unchanged.

The judge is the other model account a run calls. Its refusals arrive as the Anthropic SDK's own
exceptions, which are not memrank's, so without :func:`judge_failure` a run that answered every
question ended on "a bug in memrank" when Anthropic declined to judge it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memrank.agents import SHIPPED_DIR
from memrank.connect.base import AgentError
from memrank.connect.spec import AgentSpec
from memrank.errors import MemrankError, optional_import
from memrank.outcome import Details, Step
from memrank.reference import KEY_NAME, PROVIDER_REFUSED

#: How Anthropic says an account has no credit left. It answers 400 with this in the message --
#: the same status as a malformed request -- so the words are the only signal there is.
_NO_CREDIT = "credit balance"
#: The reason :func:`reason` gives for it, which an ending matches to say what to do.
NO_CREDIT_REASON = "credit balance too low"

#: A provider's refusal by status, in a few plain words.
_REASONS = {401: "the key was not accepted", 403: "the key is not allowed to do this",
            429: "too many requests, even after waiting", 529: "the provider is overloaded"}


def is_shipped(agent: AgentSpec) -> bool:
    """Whether the agent is one memrank ships, rather than the user's own."""
    path = agent.ref.spec_path
    return path is not None and Path(path).resolve().parent == SHIPPED_DIR.resolve()


def key_source(name: str) -> str:
    """Where this machine's ``name`` is read from, in words; never the key itself."""
    from memrank.secrets import wallet

    if os.environ.get(name):
        return "the environment"
    if wallet.get(name) is not None:
        return f"memrank's wallet ({wallet.store_path()})"
    return "nowhere: it is not set on this machine"


def _provider(body: Any) -> dict[str, Any] | None:
    """The reference agent's structured account of a failed model call, when that is what this is."""
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, dict) and "provider" in detail else None


def _provider_lead(provider: dict[str, Any], agent: AgentSpec) -> str:
    who = f"{provider['provider']}" + (f", {provider['status']}" if provider.get("status") else "")
    what = ("was refused by its provider" if provider.get("code") == PROVIDER_REFUSED
            else "failed before its provider answered")
    lead = f"The agent's model call {what} ({who}) -- that is the model account, not memrank."
    if is_shipped(agent):
        name = str(provider.get("key_name") or KEY_NAME)
        return (f"{lead} Shipped agents call the model with the {name} on this machine, read "
                f"from {key_source(name)}: check that key's account (credit, permissions, "
                "limits).")
    return f"{lead} Check the key and account your agent calls its model with."


def lead(exc: AgentError, agent: AgentSpec) -> str:
    """One sentence: what happened, whose side it is on, and what to check."""
    provider = _provider(exc.body)
    if provider is not None:
        return _provider_lead(provider, agent)
    if exc.status is not None:
        return (f"The agent answered with an error ({exc.status}) -- that is the agent's side, "
                "not memrank's. Check the agent's own output for why.")
    return ("The agent's call failed -- that is the agent's side, not memrank's. Check the "
            "agent's own output for why.")


def explained(exc: AgentError, agent: AgentSpec) -> str:
    """The lead sentence, then the raw detail exactly as the agent gave it."""
    return f"{lead(exc, agent)}\n  what it said: {exc}"


def reason(status: int | None, message: str) -> str:
    """Why a model provider refused a call, in a few plain words."""
    if _NO_CREDIT in message.lower():
        return NO_CREDIT_REASON
    if status in _REASONS:
        return _REASONS[status]
    if status is not None and status >= 500:
        return f"it failed on its side ({status})"
    return f"it rejected the request ({status})" if status is not None else "it did not answer"


def cause(exc: AgentError) -> str:
    """Why one step failed, in a few plain words: what failed steps are counted by."""
    provider = _provider(exc.body)
    if provider is None:
        return (f"your agent answered with an error ({exc.status})" if exc.status is not None
                else "your agent's call failed")
    if provider.get("code") != PROVIDER_REFUSED:
        return "the agent's model provider did not answer"
    why = reason(provider.get("status"), str(provider.get("message") or ""))
    return f"the agent's model provider refused the calls ({why})"


@dataclass(frozen=True)
class Account:
    """An agent failure as an ending shows it: what happened, what to check, the raw words."""

    happened: str
    check: Step
    details: Details


def account(exc: AgentError, agent: AgentSpec) -> Account:
    """The agent's failure as what happened, what to check, and what the other side said."""
    provider = _provider(exc.body)
    if provider is None:
        what = (f"answered with an error ({exc.status})" if exc.status is not None
                else "call failed")
        return Account(f"Your agent's {what}. That is on your agent's side, not memrank's.",
                       Step("Check your agent's own output for why."),
                       Details("your agent", (str(exc),)))
    name = str(provider["provider"]).capitalize()
    status = provider.get("status")
    said = str(provider.get("message") or exc)
    if provider.get("code") == PROVIDER_REFUSED:
        happened = f"{name} refused your agent's model call: {reason(status, said)}."
    else:
        happened = f"Your agent's model call never got an answer from {name}."
    if is_shipped(agent):
        key = str(provider.get("key_name") or KEY_NAME)
        check = Step(f"Check the {name} account behind the {key} on this machine (read from "
                     f"{key_source(key)}): its credit, permissions and limits.")
    else:
        check = Step("Check the key and account your agent calls its model with.")
    return Account(happened, check, Details(f"{name} (through your agent)", (said,)))


class JudgeFailed(MemrankError):
    """Judging stopped on Anthropic's side: it refused the calls, or never answered them.

    Attributes:
        status: Anthropic's HTTP status; ``None`` when no answer came back at all.
        reason: Why, in a few plain words (:func:`reason`).
        details: What Anthropic, or the connection to it, said -- verbatim.
    """

    def __init__(self, message: str, *, status: int | None, why: str,
                 details: tuple[str, ...]) -> None:
        super().__init__(message)
        self.status = status
        self.reason = why
        self.details = details


def _chain(exc: BaseException) -> tuple[str, ...]:
    rows = [f"{type(exc).__name__}: {exc}"]
    inner = exc.__cause__ or exc.__context__
    while inner is not None and len(rows) < 4:
        rows.append(f"caused by {type(inner).__name__}: {inner}")
        inner = inner.__cause__ or inner.__context__
    return tuple(rows)


def _refused(exc: Any) -> JudgeFailed:
    body = exc.body if isinstance(exc.body, dict) else {}
    said = body.get("error")
    error: dict[str, Any] = said if isinstance(said, dict) else {}
    message = str(error.get("message") or exc.message)
    kind = f" {error['type']}" if error.get("type") else ""
    request = f" ({exc.request_id})" if getattr(exc, "request_id", None) else ""
    why = reason(exc.status_code, message)
    return JudgeFailed(f"Anthropic refused to judge the answers: {why}", status=exc.status_code,
                       why=why, details=(f"{exc.status_code}{kind}: {message}{request}",))


def judge_failure(exc: BaseException) -> JudgeFailed | None:
    """``exc`` as a judge failure when it is Anthropic's refusal or silence; else ``None``.

    A connection error covers more than the network: the SDK wraps anything that goes wrong
    while sending, its own faults included, so the ending says no verdict came back and keeps
    the whole chain underneath rather than blaming the reader's network.
    """
    anthropic = optional_import("anthropic", None)
    if isinstance(exc, anthropic.APIStatusError):
        return _refused(exc)
    if isinstance(exc, anthropic.APIConnectionError):
        return JudgeFailed("No verdict came back from Anthropic", status=None,
                           why="it did not answer", details=_chain(exc))
    return None
