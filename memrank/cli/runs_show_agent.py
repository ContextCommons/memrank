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
"""``memrank runs show`` for a run made by ``memrank run`` -- read from its record.

An engine run's results are cells, fetched from the org's artifacts into this machine's run
registry. A ``memrank run`` run has no cells: its record is ``{"kind": "agent", "result": ...}``
(:mod:`memrank.api.agent_runs`), the run's summary, and its every question is either inline in
that result or, since ATO-2343, the ``answers.json`` artifact the record points at. Reconciling
such a run as cells asked the artifacts route for results it never had -- refused with 409 for a
record with inline answers, and fetched as an unreadable cell otherwise -- so ``show`` reported
the run as producing nothing. This module reads the record instead (ATO-2365).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, TypeGuard

from memrank import settings
from memrank.loop.report import result_facts
from memrank.placement import run_api_client
from memrank.service.protocol import ANSWERS_FILE, CaseOutcome, RunResult
from memrank.term import detail, style

#: The record ``kind`` of a run made by ``memrank run``.
AGENT_KIND = "agent"

#: How much of a question or an answer one line of ``--full`` shows.
_EXCERPT_CHARS = 80


def is_agent(platform: dict | None) -> TypeGuard[dict]:
    """Whether the org's record is a ``memrank run`` run rather than an engine run."""
    return platform is not None and platform.get("kind") == AGENT_KIND


def result_of(platform: dict) -> RunResult | None:
    """The run's result as recorded; ``None`` for a run that has not finished."""
    raw = (platform.get("record") or {}).get("result")
    return None if raw is None else RunResult.model_validate(raw)


def answers(run_id: str, platform: dict, result: RunResult) -> list[CaseOutcome]:
    """Every case with its questions: the ``answers.json`` artifact, or the record's own.

    Read through the org's artifacts route, the one the run's page reads it through. A refusal
    propagates: the result is worded from the questions, and showing none would say it asked
    none.
    """
    if (platform.get("record") or {}).get("answers") != ANSWERS_FILE:
        return result.cases  # a record from before ATO-2343 carries its questions inline
    org = settings.get("defaults.org")
    assert org is not None  # the record itself was read from this org
    with run_api_client.authenticated_client() as http, \
            tempfile.TemporaryDirectory() as scratch:
        path = run_api_client.download_artifact(http, org, run_id, ANSWERS_FILE,
                                                Path(scratch) / ANSWERS_FILE)
        body = json.loads(path.read_text(encoding="utf-8"))
    return [CaseOutcome.model_validate(case) for case in body["cases"]]


def describe(run_id: str, platform: dict, *, full: bool) -> dict[str, Any] | None:
    """The run's result for ``--json``: the recorded summary, and every case's questions
    when ``full``. ``None`` for a run that has not finished."""
    result = result_of(platform)
    if result is None:
        return None
    if full:
        result = result.model_copy(update={"cases": answers(run_id, platform, result)})
    return result.model_dump(mode="json")


def render(run_id: str, platform: dict, *, full: bool) -> None:
    """The result block under the state panel: what ``memrank run`` said when it finished."""
    result = result_of(platform)
    if result is None:
        style.out(f"  {style.pad('results', 9, style.label)}  "
                  f"{style.unit('(none recorded yet)')}")
        return
    style.out(f"  {style.label('results')}")
    # Every question, even without `--full`: how the run was graded and why answers are missing
    # are read from them, so the summary words the run as its ending did (ATO-2378).
    cases = answers(run_id, platform, result)
    facts = [(name.lower(), style.value(text)) for name, text in
             result_facts(result, [q for case in cases for q in case.questions])]
    # The run's page, where the API names one: older deployments and a deployment with no web
    # UI send no `url`, and a guessed address would be a link to nothing.
    if platform.get("url"):
        facts.append(("online", style.link(platform["url"])))
    detail.emit(detail.fields(facts, indent=4))
    if full:
        _render_questions(cases)


def _render_questions(cases: list[CaseOutcome]) -> None:
    """One line per question with its verdict, then its answer or its error."""
    for case in cases:
        detail.emit(detail.rule(case.case_id, indent=4, width=56))
        if case.error:
            style.out(f"      {style.bad('case failed:')} {case.error}")
        for question in case.questions:
            verdict = _verdict(question.status, question.passed)
            category = style.unit(f"[{question.category}]") if question.category else ""
            style.out(f"      {verdict} {style.accent(question.question_id)} {category} "
                      f"{_excerpt(question.question)}".rstrip())
            if question.error:
                style.out(f"          {style.label('error')}  {question.error}")
            else:
                style.out(f"          {style.label('answer')}  {_excerpt(question.answer or '')}")


def _verdict(status: str, passed: bool | None) -> str:
    """The verdict as one word: judged ones pass or fail, the rest name their status."""
    if status == "judged":
        return style.good("pass") if passed else style.bad("fail")
    return style.caution(status) if status == "failed" else style.unit(status)


def _excerpt(text: str) -> str:
    """``text`` on one line, cut to :data:`_EXCERPT_CHARS`."""
    flat = " ".join(text.split())
    return flat if len(flat) <= _EXCERPT_CHARS else flat[:_EXCERPT_CHARS - 3] + "..."
