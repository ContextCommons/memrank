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
"""What ``memrank evals check`` says about a valid evaluation file: what a run would ask."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from memrank.definitions.base import Case
from memrank.definitions.file import FileDefinition
from memrank.definitions.schema import GraderSpec

#: Shown in place of the full digest; the full one is on every run.
FINGERPRINT_CHARS = 23


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _counted(counter: Counter[str]) -> str:
    return ", ".join(f"{name} {count}" for name, count in counter.most_common()) or "none"


def _source(definition: FileDefinition, seed: int) -> str:
    spec = definition.cases_source
    if spec is None:
        return f"written in {definition.path.name}"
    if spec.file is not None:
        return f"{spec.file}, one per line"
    argv = [*(spec.command or []), "--seed", str(seed)]
    return "printed by `" + " ".join(argv) + "`"


def _size(definition: FileDefinition, cases: list[Case]) -> str:
    spec = definition.cases_source
    files = [definition.path] + ([definition.folder / spec.file] if spec and spec.file else [])
    kilobytes = sum(path.stat().st_size for path in files) / 1024
    history = sum(len(turn.text) for case in cases for session in case.sessions
                  for turn in session.turns)
    return f"{kilobytes:.1f} KB on disk; {history:,} characters of history to feed"


def _cases(cases: list[Case]) -> str:
    fed = [case for case in cases if case.sessions]
    sessions = sum(len(case.sessions) for case in fed)
    turns = sum(len(s.turns) for case in fed for s in case.sessions)
    return (f"{len(cases)} ({len(fed)} with history: {_plural(sessions, 'session')}, "
            f"{_plural(turns, 'turn')}; {len(cases) - len(fed)} question-only)")


def summarise(path: Path, seed: int) -> tuple[str, list[tuple[str, str]]]:
    """A title and label/value rows for the evaluation at ``path``; refuses a broken one.

    Runs the case program when the file has one (with ``--seed``), since its cases are only
    known once printed; runs no agent and no grader.
    """
    definition = FileDefinition(path)
    cases = definition.load_cases(seed)
    queries = [q for case in cases for q in case.queries]
    graders = [GraderSpec.model_validate(q["grader"]) for q in queries]
    judged = sum(1 for grader in graders if grader.uses_model)
    rows = [
        ("cases", _cases(cases)),
        ("from", _source(definition, seed)),
        ("questions", str(len(queries))),
        ("categories", _counted(Counter(q.get("category") or "uncategorized"
                                        for q in queries))),
        ("grading", _counted(Counter(grader.name for grader in graders))),
        ("judge model", f"grades {_plural(judged, 'question')}, with your organisation's "
                        "Anthropic key"
                        if judged else "not used; no key needed"),
        ("size", _size(definition, cases)),
        ("fingerprint", definition.fingerprint[:FINGERPRINT_CHARS] + "..."),
    ]
    if definition.description:
        rows.insert(0, ("about", definition.description))
    return f"{definition.evaluation}  ({definition.given})", rows
