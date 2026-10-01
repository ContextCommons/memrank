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
"""The measures memrank ships. Each is an ordinary `Measure`; none is a special case.

- :class:`WordMatch` -- the deterministic span proxy (`memrank.SpanRecall`'s arithmetic),
  decided by a rule. It marks whether the gold spans appear in a recalled document, and it is
  never a claim that an answer would be correct.
- :class:`Judge` -- the same grading `memrank.judging` does, decided by a model.
- :class:`Latency` -- p50 and p95 per step, from the timings memrank took itself.
- :class:`FailureRate` -- how much of the run broke, decided by memrank's own bookkeeping.
- :class:`BenchmarkScore` -- an in-tree benchmark's own `score()`, applied over the traces.

A measure that produces more than one number names each value under its own name with a dotted
suffix (``latency.retrieve.p50``), so no two values of one measure are told apart by position.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memrank.core import AdapterResponse, Benchmark, BenchmarkUnit, Document
from memrank.instrument.measure import Decider, Measure, Scope, Value
from memrank.instrument.trace import Trace


def _documents(trace: Trace) -> list[Document]:
    """What this trace recalled, as the shipped scorers read it: documents in ranked order."""
    return list(trace.recalled.documents) if trace.recalled is not None else []


def _percentile(samples: list[float], fraction: float) -> float:
    """The nearest-rank percentile of a sorted sample. Deterministic, no interpolation."""
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


class WordMatch(Measure):
    """Do the gold spans appear verbatim in something the system recalled?

    A retrieval proxy decided by a fixed rule -- the arithmetic of `memrank.SpanRecall`, per
    task instead of per unit. Never answer correctness, which is what `why` says on every value.
    """

    name = "word-match"
    scope = Scope.TASK
    reads = ("recalled",)
    decider = Decider.RULE
    #: Carried on every value, so the caveat travels with the number.
    LABEL = "span match in a recalled document; a retrieval proxy, not answer correctness"

    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        from memrank.metrics.scoring import EvidenceSpec, score_query

        produced = []
        for trace in traces:
            expected = trace.task.expected
            spec = EvidenceSpec(
                required_spans=list(expected.required_spans),
                forbidden_spans=list(expected.forbidden_spans),
                evidence_doc_ids=list(expected.evidence_doc_ids),
                kind=expected.polarity)
            if trace.error is not None:
                produced.append(Value(measure=self.name, decider=self.decider,
                                      task_id=trace.task_id, value=None,
                                      why=f"the task broke at {trace.error.step}"))
                continue
            marked = score_query(spec, _documents(trace))
            produced.append(Value(
                measure=self.name, decider=self.decider, task_id=trace.task_id,
                value=1.0 if marked.hit else 0.0,
                why=f"{self.LABEL}; matched {marked.matched_span!r} in "
                    f"{marked.matched_doc_id!r}" if marked.hit else self.LABEL))
        return produced


class FailureRate(Measure):
    """How much of the run broke. memrank's own bookkeeping, not anybody's claim."""

    name = "failure-rate"
    scope = Scope.RUN
    reads = ("error",)
    decider = Decider.MEMRANK

    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        if not traces:
            return [Value(measure=self.name, decider=self.decider, value=None,
                          why="no traces")]
        broken = [t for t in traces if t.error is not None]
        steps = sorted({t.error.step for t in broken if t.error})
        return [Value(
            measure=self.name, decider=self.decider, value=len(broken) / len(traces),
            why=f"{len(broken)} of {len(traces)} traces carry an error"
                + (f" (at {', '.join(steps)})" if steps else ""))]


class Latency(Measure):
    """p50 and p95 per step, over the timings memrank took at its own call boundary.

    Never asserted on and never ranked here: it is reported, per step, with the sample count
    it was taken over, because a percentile over three samples is a different object from one
    over three hundred.
    """

    name = "latency"
    scope = Scope.RUN
    reads = ("timings_ms",)
    decider = Decider.MEMRANK

    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        samples: dict[str, list[float]] = {}
        for trace in traces:
            for step, ms in trace.timings_ms.items():
                samples.setdefault(step, []).append(ms)
        produced = []
        for step in sorted(samples):
            taken = samples[step]
            for label, fraction in (("p50", 0.5), ("p95", 0.95)):
                produced.append(Value(
                    measure=f"{self.name}.{step}.{label}", decider=self.decider,
                    value=_percentile(taken, fraction),
                    why=f"milliseconds, memrank's own clock, over {len(taken)} sample(s)"))
        return produced


class BenchmarkScore(Measure):
    """An in-tree benchmark's own `score()`, applied over the traces it produced.

    The benchmark's unit IS the group, so this produces one value per group and nothing that
    combines them: no run-level number is invented on a benchmark's behalf.
    """

    scope = Scope.RUN
    decider = Decider.RULE
    reads = ("recalled", "declared")

    def __init__(self, benchmark: Benchmark, units: Sequence[BenchmarkUnit]) -> None:
        self.benchmark = benchmark
        self.units = {unit.isolation_id: unit for unit in units}
        self.name = f"{benchmark.name}-score"
        #: What the benchmark says its number is, in `quality_metric`'s vocabulary.
        self.metric = getattr(benchmark, "quality_metric", "")

    def _responses(self, traces: Sequence[Trace]) -> list[AdapterResponse]:
        return [AdapterResponse(query_id=t.task_id, documents=_documents(t),
                                raw=t.recalled.declared if t.recalled else None,
                                latency_ms=t.timings_ms.get("retrieve"))
                for t in traces if t.error is None]

    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        produced = []
        for group, unit in self.units.items():
            mine = [t for t in traces if (t.group or t.task_id) == group]
            if not mine:
                continue
            produced.append(self._one_group(group, unit, mine))
        return produced

    def _one_group(self, group: str, unit: BenchmarkUnit, traces: Sequence[Trace]) -> Value:
        broken = [t for t in traces if t.error is not None]
        if len(broken) == len(traces):
            # Every task in the group failed, so the benchmark would be scoring an empty
            # response list -- and a benchmark with a negative query scores that as a hit.
            # A number here reads as a measurement of the system; there was none.
            steps = sorted({t.error.step for t in broken if t.error})
            return Value(
                measure=self.name, decider=self.decider, group=group, value=None,
                why=f"every one of the {len(traces)} task(s) in this group failed "
                    f"(at {', '.join(steps)}), so there was nothing to score")
        try:
            scored: dict[str, Any] = self.benchmark.score(unit, self._responses(traces))
        except Exception as failure:
            # The benchmark's own refusal -- a graph benchmark handed no snapshot says so --
            # is recorded as a value that is None with the reason, never as a zero.
            return Value(measure=self.name, decider=self.decider, group=group, value=None,
                         why=f"{type(failure).__name__}: {failure}")
        composite = scored.get("composite")
        label = str(scored.get("metric") or self.metric or self.benchmark.name)
        if broken:
            label += f"; {len(broken)} of {len(traces)} tasks failed and were not scored"
        return Value(
            measure=self.name, decider=self.decider, group=group,
            value=float(composite) if isinstance(composite, (int, float)) else None,
            why=label)


class Judge(Measure):
    """A model adjudicates the answer against what was expected.

    Constructible without a key -- it is a declaration, and a run refuses on the declaration
    before anything is spent. Running it without one raises, naming how to set it: there is no
    mode in which this measure decides something with nobody having judged it.
    """

    name = "judge"
    scope = Scope.TASK
    reads = ("answered",)
    decider = Decider.MODEL

    def __init__(self, config: Any | None = None) -> None:
        from memrank.judging.judge import JudgeConfig

        self.config = config or JudgeConfig(no_context_control=False)

    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        from memrank.judging.client import build_completer
        from memrank.judging.judge import judge_answer

        complete, _ = build_completer(self.config)
        produced = []
        for trace in traces:
            gold = (trace.task.expected.answers or ("",))[0]
            if trace.answered is None:
                produced.append(Value(measure=self.name, decider=self.decider,
                                      task_id=trace.task_id, value=None,
                                      why="nothing answered this task"))
                continue
            verdict = judge_answer(
                complete, question=trace.task.prompt, answer=trace.answered.text, gold=gold,
                model=self.config.judge_model, samples=self.config.samples,
                negative=trace.task.expected.polarity == "negative")
            produced.append(Value(
                measure=self.name, decider=self.decider, task_id=trace.task_id,
                value=verdict.passed, why=verdict.rationale))
        return produced
