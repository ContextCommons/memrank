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
"""Plans: which cases and questions of an evaluation a run asks, in which order.

Every evaluation -- a shipped benchmark or an evaluation file -- is a
:class:`~memrank.definitions.base.Definition`, and a run request is planned from it on this one
path: the same seed samples the same cases and questions, and the plan carries the evaluation's
identity (name, version, fingerprint) for the result. A case's queries keep their references and
graders here, server-side; the agent only ever receives
:class:`~memrank.service.protocol.Session` and :class:`~memrank.service.protocol.Question`.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache

from memrank.definitions import resolve
from memrank.definitions.base import Case
from memrank.definitions.shipped import session_of
from memrank.errors import ActionRequired, MemrankError
from memrank.outcome import Step
from memrank.service.protocol import RunCreate

__all__ = ["Case", "EvaluationChanged", "Plan", "PlanError", "build_plan", "session_of"]


class PlanError(MemrankError):
    """A run request the service cannot turn into cases."""


class EvaluationChanged(ActionRequired):
    """A run is resumed on cases other than the ones it started with."""


@dataclass(frozen=True)
class Plan:
    """Everything a run needs from its evaluation, derived deterministically from the request."""

    evaluation: str
    dataset_version: str
    task_version: int
    cases: tuple[Case, ...]
    version: str | None = None
    fingerprint: str | None = None
    source: str | None = None

    def case(self, case_id: str) -> Case:
        return next(case for case in self.cases if case.id == case_id)

    @cached_property
    def cases_digest(self) -> str:
        """``sha256:...`` over exactly what the run asks and grades against, in order."""
        rows = [[case.id, [s.model_dump(mode="json") for s in case.sessions],
                 list(case.queries)] for case in self.cases]
        blob = json.dumps(rows, sort_keys=True, default=str).encode()
        return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def _sample(count: int, wanted: int | None, rng: random.Random) -> list[int]:
    """Indices to keep, in their original order: all of them, or a seeded sample."""
    if wanted is None or wanted >= count:
        return list(range(count))
    return sorted(rng.sample(range(count), wanted))


def build_plan(request: RunCreate) -> Plan:
    """The cases a run request selects. Same request, same cases, same questions, same order."""
    definition = resolve(request.evaluation)
    return _cached_plan(definition.source, definition.fingerprint, request.cases,
                        request.questions, request.seed)


def check_unchanged(plan: Plan, started_with: str | None, run_id: str) -> None:
    """Refuse to go on with a run whose cases are not the ones it started with."""
    if started_with is None or started_with == plan.cases_digest:
        return
    raise EvaluationChanged(
        f"Run {run_id} started on other cases than {plan.source or plan.evaluation} gives "
        "now: the evaluation (or the program that prints its cases) changed since. A run "
        "continues only on the cases it started with, so every answer is to one evaluation.",
        steps=(Step("Put the evaluation back as it was to continue this run, or start a new "
                    "run on the changed one."),))


@lru_cache(maxsize=8)
def _cached_plan(source: str, fingerprint: str, cases: int | None, questions: int | None,
                 seed: int) -> Plan:
    definition = resolve(source)
    units = definition.load_cases(seed)
    if cases is not None and cases > len(units):
        raise PlanError(f"{definition.evaluation} has {len(units)} cases; asked for {cases}")
    rng = random.Random(seed)
    chosen = [units[i] for i in _sample(len(units), cases, rng)]
    sampled = tuple(
        Case(id=unit.id, sessions=unit.sessions,
             queries=tuple(unit.queries[i] for i in _sample(len(unit.queries), questions, rng)))
        for unit in chosen)
    plan = Plan(evaluation=definition.evaluation, dataset_version="",
                task_version=definition.task_version, cases=sampled,
                version=definition.version, fingerprint=fingerprint, source=source)
    # A file's cases have no dataset release; the digest of what is asked stands in for one.
    return replace(plan, dataset_version=definition.dataset_version
                   or f"cases {plan.cases_digest[:19]}")
