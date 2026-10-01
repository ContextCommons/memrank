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
"""Why a run refuses, worked out before the system is touched.

A refusal is a result with no traces and a stated reason. The three things checked here are
the three the model names: the system must be a kind memrank knows and must supply that kind's
required verbs; every measure must read trace fields and value names something produces; and
the material an evaluation carries must have a verb to be given through.

The fourth is the answer writer (H1): when the system only recalls and a measure reads
``answered``, something has to write the answer, and memrank ships no reader that runs
in-process without a key -- so it refuses and names what to pass rather than inventing one.
"""

from __future__ import annotations

from memrank.instrument.evaluation import Evaluation
from memrank.instrument.measure import Measure
from memrank.instrument.system import KIND_NAMES, System, kind_of, missing_verbs
from memrank.instrument.trace import TRACE_FIELDS

#: The kinds that answer for themselves. The others recall, and need a writer to be judged.
ANSWERING_KINDS = ("model", "assistant")
#: The kinds that can be told things before the prompt.
FED_KINDS = ("memory",)


def _kind_refusal(system: object) -> str | None:
    if not isinstance(system, System):
        return (
            f"{type(system).__name__} is not a memrank System: subclass one of "
            f"{', '.join(sorted(set(KIND_NAMES.values())))} "
            f"(memrank.Memory, memrank.Model, memrank.Retriever, memrank.Assistant)")
    if kind_of(system) is None:
        return (
            f"{type(system).__name__} subclasses System but none of its kinds, so there is no "
            f"verb to put a task to it with; subclass memrank.Memory, memrank.Model, "
            f"memrank.Retriever or memrank.Assistant")
    absent = missing_verbs(system)
    if absent:
        return (
            f"{type(system).__name__} is a {kind_of(system)} system and does not supply "
            f"{', '.join(absent)}, which that kind requires")
    return None


def _measure_refusal(measures: tuple[Measure, ...], produces: frozenset[str]) -> str | None:
    for m in measures:
        unknown = [name for name in m.reads if name not in produces]
        if unknown:
            return (
                f"measure {m.name!r} reads {', '.join(sorted(unknown))}, which nothing in this "
                f"run produces; the trace fields are {', '.join(sorted(TRACE_FIELDS))} and the "
                f"value names are {', '.join(sorted(produces - TRACE_FIELDS)) or '(none)'}")
    return None


def refusal(system: object, evaluation: Evaluation, answerer: object | None) -> str | None:
    """The reason this run cannot be set up, or `None` when it can."""
    reason = _kind_refusal(system)
    if reason is not None:
        return reason
    assert isinstance(system, System)
    kind = kind_of(system)
    measures = tuple(evaluation.measures)
    produces = TRACE_FIELDS | {m.name for m in measures}
    reason = _measure_refusal(measures, produces)
    if reason is not None:
        return reason
    if any(task.context for task in evaluation.tasks) and kind not in FED_KINDS:
        return (
            f"{evaluation.name} carries documents to give before the prompt and "
            f"{type(system).__name__} is a {kind} system, which has no verb to be told things "
            f"with; material for a system that cannot be fed belongs to that system when it "
            f"is built")
    wants_answer = any("answered" in m.reads for m in measures)
    if wants_answer and kind not in ANSWERING_KINDS and answerer is None:
        return (
            f"{', '.join(m.name for m in measures if 'answered' in m.reads)} reads `answered`, "
            f"a {kind} system only recalls, and no answer writer was given; pass "
            f"answerer=<your writer> to evaluation.run (memrank ships no reader that runs "
            f"in-process without a model key)")
    return None
