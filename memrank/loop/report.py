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
"""A finished run as its ending shows it: done, the score in plain words, and what failed.

Also the two sentences every other ending borrows when the run got far enough to have them:
what the agent got through (:func:`achieved`) and why answers failed (:func:`failures`).
"""

from __future__ import annotations

from memrank.loop import readout
from memrank.loop.trace import RunTrace
from memrank.outcome import Kind, Link, Outcome
from memrank.service.protocol import QuestionOutcome, RunResult


def failures(result: RunResult, trace: RunTrace) -> str | None:
    """Why answers failed, counted by cause: ``221 answers failed: <cause>``; ``None`` if none.

    The causes are the ones this session saw. A resumed run's earlier failures are counted
    without one, and the report has each failure's own words.
    """
    if not result.failed:
        return None
    seen = sorted(trace.failures.items(), key=lambda item: (-item[1], item[0]))
    parts = [why for why, _ in seen] if len(seen) == 1 else [f"{n} {why}" for why, n in seen]
    earlier = result.failed - sum(trace.failures.values())
    if earlier > 0:
        parts.append(f"{earlier} in an earlier session (report.html has each one's error)")
    noun = "answer" if result.failed == 1 else "answers"
    return f"{result.failed} {noun} failed: " + "; ".join(parts)


def achieved(result: RunResult, judged: bool = False) -> str:
    """What the agent got through, when the run ends short of done after the steps."""
    failed = f" ({result.failed} of them failed, see below)" if result.failed else ""
    tail = " and they were judged" if judged else ""
    return f"Your agent answered all {result.questions} questions{failed}{tail}"


def _title(result: RunResult) -> str:
    total = result.questions
    answered = f"{total - result.failed} of {total}" if result.failed else f"{total}"
    judged = "answered and judged" if result.judged else "answered, not judged"
    return f"Done: {answered} questions {judged}"


def _facts(result: RunResult) -> list[tuple[str, str]]:
    return result_facts(result, [q for case in result.cases for q in case.questions])


def _rows(label: str, lines: list[str]) -> list[tuple[str, str]]:
    """``lines`` under one label: the first beside it, the rest under the first."""
    return [(label if index == 0 else "", line) for index, line in enumerate(lines)]


def result_facts(result: RunResult,
                 questions: list[QuestionOutcome] | None) -> list[tuple[str, str]]:
    """A finished run's facts in plain words (:mod:`memrank.loop.readout`).

    ``questions`` are the run's every question, which say how it was graded and why answers
    are missing; None when they are not at hand, and the words then claim neither. Shared with
    ``memrank runs show``, so the two cannot describe one result two ways.
    """
    kind = readout.scoring(questions or [])
    if result.judged:
        facts = [("Score", readout.headline(result, kind)),
                 *_rows("Likely range", readout.range_lines(result)),
                 *_rows("By kind", readout.by_kind_lines(result, kind))]
    else:
        facts = [("Score", "none: this run was not judged (--no-judge)")]
    groups = readout.failure_groups(questions) if questions is not None else None
    facts.append(("Not answered", readout.not_answered(result, groups)))
    if result.unjudged:
        facts.append(("Unjudged", f"{result.unjudged} answers got no readable verdict from "
                                  "the judge and are not scored"))
    if result.notice and result.judged:
        facts.append(("Notice", result.notice))
    facts.append(("Speed", readout.speed(result)))
    if result.identity.judge_model:
        facts.append(("Judged by", result.identity.judge_model))
    return facts


def done(result: RunResult, trace: RunTrace, url: str | None, org: str) -> Outcome:
    """The ending of a run that finished: everything asked for was done and recorded in
    ``org``, which it names beside the run's id."""
    folder = f"{trace.folder}/" if trace.folder else None
    return Outcome(kind=Kind.DONE, title=_title(result),
                   identity=(("Org", org), ("Run", result.run_id)),
                   facts=tuple(_facts(result)), saved=folder,
                   link=Link("View results", url) if url else None)
