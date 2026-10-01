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
"""A finished run's numbers in plain words (ATO-2378): what the ending and ``runs show`` say.

"95% CI 36%-43%, n=400" is exact and unreadable. This says the same thing as a person would:
the average score or the correct count, how many conversations it came from, the likely range
and when to read it as rough, why the agent failed to answer, and times in minutes and seconds.
Nothing here computes a score -- :mod:`memrank.service.scoring` does that once -- it only words
what the result already holds. ``web-gui/lib/agent-run.ts`` words the run page the same way.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from memrank.service.protocol import Interval, QuestionOutcome, RunResult

#: Every answer is right or wrong (a gold-answer judge, exact, choice, numeric).
RIGHT_OR_WRONG = "right_or_wrong"
#: An answer can earn part of its point (a rubric, or a grader program's 0..1 score).
PARTIAL_CREDIT = "partial_credit"

#: Below this many conversations the likely range is called rough: a bootstrap over a handful
#: of clusters understates how far the score could move.
ROUGH_BELOW = 20

#: How long one quoted error may be.
_EXCERPT_CHARS = 60

#: Why an answer is missing, in the order a message is tested against them. A status or exit
#: code comes first: the agent answered, so its own words say why, even about a timeout.
_CAUSES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("agent_error", re.compile(r"returned \d{3}\b|exited -?\d+")),
    ("timed_out", re.compile(r"time ?out|timed out|did not finish in", re.IGNORECASE)),
    ("unreachable", re.compile(r"could not connect|connect(ion)? ?error|connection (refused|"
                               r"reset|closed)|remoteprotocolerror|disconnected",
                               re.IGNORECASE)),
    ("unreadable", re.compile(r"not JSON|in the ask response")),
)
#: Each cause as a count reads: ``29 timed out``, ``6 agent errors``.
CAUSE_WORDS = {"timed_out": ("timed out", "timed out"),
               "agent_error": ("agent error", "agent errors"),
               "unreachable": ("couldn't reach the agent", "couldn't reach the agent"),
               "unreadable": ("unreadable answer", "unreadable answers"),
               "other": ("other error", "other errors")}


@dataclass(frozen=True)
class FailureGroup:
    """Failed answers that failed for one reason: which, how many, and one of their messages."""

    cause: str
    count: int
    excerpt: str

    def words(self) -> str:
        one, many = CAUSE_WORDS[self.cause]
        return f"{self.count} {one if self.count == 1 else many}"


def cause_of(error: str) -> str:
    """Which of :data:`CAUSE_WORDS` one recorded error message belongs to."""
    for cause, pattern in _CAUSES:
        if pattern.search(error):
            return cause
    return "other"


#: The request a connector names before what went wrong with it: ``POST /v1/chat/completions ``.
_REQUEST = re.compile(r"^[A-Z]+ /\S* ")
_STATUS = re.compile(r"^returned (\d{3}): (.*)$", re.S)
_SAID = re.compile(r"""['"]message['"]:\s*['"]([^'"]+)""")
_CALL = re.compile(r"""^(\w+)\((['"])(.*)\2\)$""", re.S)


def short_error(error: str) -> str:
    """The part of a recorded error that says what happened, cut short.

    ``POST /ask returned 500: {"error": {"message": "context length exceeded"}}`` reads as
    ``HTTP 500: context length exceeded``, and ``... failed after sending: ReadTimeout('timed
    out')`` as ``ReadTimeout: timed out``; a command's path is cut to its name.
    """
    text = _REQUEST.sub("", " ".join(error.split()), count=1)
    text = text.removeprefix("failed after sending: ")
    status = _STATUS.match(text)
    if status:
        said = _SAID.search(status.group(2))
        text = f"HTTP {status.group(1)}: {said.group(1) if said else status.group(2)}"
    call = _CALL.match(text)
    if call:
        text = f"{call.group(1)}: {call.group(3)}"
    first, _, rest = text.partition(" ")
    if "/" in first and rest:
        text = f"{first.rsplit('/', 1)[-1]} {rest}"
    return text if len(text) <= _EXCERPT_CHARS else text[:_EXCERPT_CHARS - 3] + "..."


def failure_groups(questions: Iterable[QuestionOutcome]) -> list[FailureGroup]:
    """The failed questions grouped by cause, most first, each quoting its commonest message."""
    messages: dict[str, Counter[str]] = {}
    for question in questions:
        if question.status == "failed":
            error = question.error or "no reason recorded"
            messages.setdefault(cause_of(error), Counter())[error] += 1
    groups = [FailureGroup(cause, sum(seen.values()), short_error(seen.most_common(1)[0][0]))
              for cause, seen in messages.items()]
    return sorted(groups, key=lambda group: (-group.count, group.cause))


def scoring(questions: Iterable[QuestionOutcome]) -> str | None:
    """How the run's answers were graded, read from their grading data; None with none to read.

    Partial credit when any question carries a rubric or a grader that can give it (a rubric
    or a program), or any judged score falls between 0 and 1; right or wrong otherwise.
    """
    rows = list(questions)
    if not rows:
        return None
    for question in rows:
        grader = question.grading.get("grader")
        kind = grader.get("kind") if isinstance(grader, dict) else None
        if question.grading.get("rubric") or kind == "rubric" or \
                (isinstance(grader, dict) and grader.get("command")):
            return PARTIAL_CREDIT
        if question.score is not None and 0.0 < question.score < 1.0:
            return PARTIAL_CREDIT
    return RIGHT_OR_WRONG


def conversations(result: RunResult) -> int:
    """How many conversations the run asked about: its distinct case ids."""
    return len({case.case_id for case in result.cases} or set(result.identity.case_ids))


def duration(ms: float | None) -> str:
    """A time in the unit a person reads it in: ``850 ms``, ``7.3 s``, ``3 min 29 s``."""
    if ms is None:
        return "-"
    if ms < 1000:
        return f"{round(ms)} ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    whole = round(seconds)
    if whole < 3600:
        return f"{whole // 60} min {whole % 60} s"
    return f"{whole // 3600} h {whole % 3600 // 60} min"


def percent(value: float | None, digits: int = 0) -> str:
    """``value`` as a percentage, rounded half up as the run page's ``toFixed`` rounds it.

    Python's own formatting rounds half to even, which would print 22.5 as 22% here and 23%
    on the page; ``Decimal(float)`` is the float's exact value, as ``toFixed`` reads it.
    """
    if value is None:
        return "-"
    exact = Decimal(value * 100).quantize(Decimal(1).scaleb(-digits), ROUND_HALF_UP)
    return f"{exact}%"


def likely_range(value: Interval) -> str | None:
    """``36%-43%``, or None when the interval could not be computed."""
    low, high = value.ci95
    return None if low is None or high is None else f"{percent(low)}-{percent(high)}"


def correct_count(value: Interval) -> int:
    """How many questions a right-or-wrong score counts as correct: every score is 0 or 1."""
    return round((value.mean or 0.0) * value.n)


def headline(result: RunResult, kind: str | None) -> str:
    """``39.5% average score across 400 questions, from 20 conversations`` -- or, judged right
    or wrong, ``77.5% correct: 31 of 40 questions, from 2 conversations``."""
    value, count = result.score, conversations(result)
    source = f"from {count} conversation{'' if count == 1 else 's'}"
    if value.mean is None:
        return f"none: no answer could be scored ({value.n} questions)"
    if kind == RIGHT_OR_WRONG:
        return (f"{percent(value.mean, 1)} correct: {correct_count(value)} of {value.n} "
                f"questions, {source}")
    return f"{percent(value.mean, 1)} average score across {value.n} questions, {source}"


def range_lines(result: RunResult) -> list[str]:
    """What the likely range is, then the warning when too few conversations make it rough."""
    band = likely_range(result.score)
    if band is None:
        return []
    lines = [f"{band}: on other conversations like these, the score would usually land here"]
    count = conversations(result)
    if count < ROUGH_BELOW:
        lines.append(f"Only {count} conversation{'' if count == 1 else 's'}: treat this range "
                     "as rough.")
    return lines


def by_kind_lines(result: RunResult, kind: str | None) -> list[str]:
    """One line per kind of question, highest score first, names and numbers aligned."""
    rows = sorted(((name.replace("_", " "), value) for name, value in result.per_category.items()
                   if value.mean is not None),
                  key=lambda row: (-(row[1].mean or 0.0), row[0]))
    width = max((len(name) for name, _ in rows), default=0)
    lines = []
    for name, value in rows:
        count = (f"  {correct_count(value)} of {value.n} correct" if kind == RIGHT_OR_WRONG
                 else f"  {value.n} questions")
        band = likely_range(value)
        likely = f", likely {band}" if band else ""
        lines.append(f"{name:<{width}}  {percent(value.mean):>4}{count}{likely}")
    return lines


def not_answered(result: RunResult, groups: list[FailureGroup] | None) -> str:
    """``Agent didn't answer 37 of 400 (29 timed out, 6 agent errors, ...)``, or ``none``.

    ``groups`` is None when the questions are not at hand to say why.
    """
    if not result.failed:
        return "none"
    why = f" ({', '.join(group.words() for group in groups)})" if groups else ""
    return f"Agent didn't answer {result.failed} of {result.questions}{why}"


def speed(result: RunResult) -> str:
    """Typical and slow answer times, and the typical time to load one conversation's history."""
    latency = result.latency_ms
    parts = []
    if latency.get("ask_p50") is not None:
        parts.append(f"answers {duration(latency.get('ask_p50'))} typical, "
                     f"{duration(latency.get('ask_p95'))} slow (1 in 20)")
    if latency.get("feed_p50") is not None:
        parts.append(f"loading the history {duration(latency.get('feed_p50'))} typical")
    return "; ".join(parts) or "not measured: no step was timed"
