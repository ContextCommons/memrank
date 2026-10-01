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
"""Composing an evaluation from questions and scoring supplied independently (ATO-2149).

`Benchmark` -- the catalog's loader-and-scorer, not the public `memrank.Evaluation` -- fuses the
two: `load()` and `score()` sit on one class, so a person with their own questions still had to
write a scorer, and a person with their own notion of correct still had to write a loader.
:class:`ComposedEvaluation` is a `Benchmark` built from the two halves, so three runs are
writable and none of them asks for the half that was not brought:

    ComposedEvaluation(name="mine", questions=[unit, ...])          # my questions, memrank's scorer
    ComposedEvaluation(name="demo+mine", questions=DemoBenchmark(), scorer=MyScorer())
    ComposedEvaluation(name="mine", questions=[unit, ...], scorer=MyScorer())

The class stays what the registry, the `evals` catalog and `config_for_receipt()` key on; this
adds a way to build one, and takes nothing away.

**What separating the halves must not lose.** `judge_shape()` is the only place a benchmark's
grading rules and its loader's labels are held together, and `BinaryJudgeShape._prompt_for`
raises on a query whose declared prompt key the shape does not define because a silent skip
"is how 12% of this benchmark was graded against the wrong object for months"
(`memrank/judging/shape.py:170`). Pulling the halves apart removes the class that held them
together, so both declarations are compared HERE, at composition, before a single item runs --
and a disagreement is refused with both lists named, never averaged and never skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from memrank.core import AdapterResponse, Benchmark, BenchmarkUnit, EvalInfo
from memrank.errors import MemrankError

if TYPE_CHECKING:  # the judge shape is reached lazily; keep the judge out of this import graph.
    from memrank.judging.shape import JudgeShape

#: What a composed evaluation declares about its size when nobody said. Declared, never counted:
#: `EvalInfo` is a catalog view and rendering it must not call `load()`.
_SUPPLIED_IN_PYTHON = "supplied in Python, not from a dataset"


class CriteriaMismatch(MemrankError):
    """Two halves of a composition declare things that disagree.

    Raised at composition time, naming both declarations, rather than late in a `KeyError` or --
    worse -- silently, by reducing over whichever criterion happened to be there.
    """


class Scorer:
    """The rule that decides whether what came back is right. One half of an evaluation.

    Subclass it and implement :meth:`score`, which is `Benchmark.score`'s contract unchanged: one
    unit's responses in, a dict with at least ``composite`` out.

    :attr:`criterion_names` is the declaration the composition checks. It names the scored keys
    an aggregation may be written over -- ``("composite",)`` for memrank's own scorer. Declaring
    nothing means "whatever the questions expect", which is what keeps somebody else's criteria
    acceptable over memrank's questions.
    """

    #: The scored keys this scorer promises to return. Empty declares nothing.
    criterion_names: tuple[str, ...] = ()
    #: What this scorer calls itself in the receipt. Derived from the class name when unset.
    identity: str | None = None
    #: The kind of number this scorer produces, in `Benchmark.quality_metric`'s vocabulary.
    #: ``None`` leaves the question source's declaration standing.
    quality_metric: str | None = None
    # What a number this scorer decided is a number OF, and what it must not be read as, is
    # NOT declared here. `memrank/quality.py` owns that prose, keyed by `quality_metric`, and
    # `memrank.evaluation.score.Score` is what carries it onto the result; a second copy on the
    # scorer would be the one that goes stale. A kind `quality.declaration_for` does not know
    # already answers honestly -- it names the kind and claims nothing about it -- so a scorer
    # bringing a novel metric is readable today and better-declared once a person's scorer can
    # register its own `MetricDeclaration`, which is not this step's to build.

    def score(self, unit: BenchmarkUnit,
              responses: Sequence[AdapterResponse]) -> dict[str, Any]:
        """Mark one unit. Same contract as :meth:`memrank.core.Benchmark.score`."""
        raise NotImplementedError

    def judge_shape(self) -> JudgeShape | None:
        """How this scorer grades under ``--judge``, or ``None`` when it does not grade.

        ``None`` means "this scorer has no opinion", and the question source's own shape stands.
        It is not a claim that the questions are ungradeable.
        """
        return None

    def name(self) -> str:
        """The one string that names who decided, for the receipt and for a reader."""
        return self.identity or type(self).__name__


def _kind(scorer: Scorer) -> str:
    """The scorer's declared quality metric, refusing the empty string as an answer."""
    declared = scorer.quality_metric
    if declared is not None and not declared:
        raise ValueError(
            f"{scorer.name()} declares quality_metric='' -- an empty kind is not a kind. "
            f"Name what the number is (see QUALITY_METRIC_LABELS), or leave it None to keep "
            f"the question source's declaration.")
    return declared or ""


class _Questions:
    """The question half, normalised: how to load, and what it declares about itself."""

    def __init__(self, questions: Benchmark | Sequence[BenchmarkUnit] |
                 Callable[[], list[BenchmarkUnit]],
                 *, criterion_names: Sequence[str], prompt_keys: Sequence[str]) -> None:
        self.source = questions if isinstance(questions, Benchmark) else None
        self._supplied = questions
        self.criterion_names = tuple(criterion_names) or (
            tuple(self.source.criterion_names) if self.source else ())
        self.prompt_keys = tuple(prompt_keys) or self._declared_prompt_keys()

    def load(self) -> list[BenchmarkUnit]:
        if self.source is not None:
            return self.source.load()
        if callable(self._supplied):
            return list(self._supplied())
        return list(self._supplied)  # type: ignore[arg-type]

    def _declared_prompt_keys(self) -> tuple[str, ...]:
        """The per-type grading keys this source's own shape says its queries carry.

        Asked of the source's `judge_shape()` rather than of a second declaration, because that
        shape IS where a benchmark's labels and its grading rules are held together. A source
        that is not a `Benchmark`, or whose shape defines no per-type prompts, carries none.
        """
        if self.source is None:
            return ()
        return tuple(getattr(self.source.judge_shape(), "prompts", None) or ())

    def count(self) -> str:
        if self.source is not None:
            return self.source.info.units_declared
        if callable(self._supplied):
            return f"units {_SUPPLIED_IN_PYTHON} by a callable"
        return f"{len(self._supplied)} unit(s) {_SUPPLIED_IN_PYTHON}"  # type: ignore[arg-type]


#: The flags a `Benchmark` question source declares about its own data and protocol. Copied onto
#: the composition so "memrank's questions, my scorer" keeps the questions' declarations -- their
#: egress safety, their graph requirement, their reader-context policy -- rather than silently
#: taking this class's defaults. `quality_metric` is deliberately absent: the scorer decides what
#: kind of number a run produces, so it is set separately and by the scorer where it declares one.
_SOURCE_FLAGS = (
    "dataset_version", "substring_recall_supported", "composite_rankable", "is_synthetic",
    "question_text_public", "requires_graph", "context_policy", "criterion_names", "VERSION",
)


class ComposedEvaluation(Benchmark):
    """An evaluation built from a question half and a scoring half, supplied independently.

    Args:
        name: What this evaluation is called, in artifacts and run rows.
        questions: An `Benchmark` whose `load()` supplies the questions, a list of
            `BenchmarkUnit`s, or a zero-argument callable returning them.
        scorer: The rule that marks an answer. Defaults to `memrank.SpanRecall`, which is the
            point: bringing questions does not mean writing a scorer.
        dataset_version, info, report_template: Declared where the question half cannot say.
        criterion_names: What the questions expect the scorer to produce. Only needed when the
            question half is not a `Benchmark` that declares it.
        prompt_keys: The per-type grading keys the questions carry, same exception.
    """

    def __init__(self, *, name: str,
                 questions: Benchmark | Sequence[BenchmarkUnit] |
                 Callable[[], list[BenchmarkUnit]],
                 scorer: Scorer | None = None,
                 dataset_version: str | None = None,
                 info: EvalInfo | None = None,
                 report_template: str | None = None,
                 criterion_names: Sequence[str] = (),
                 prompt_keys: Sequence[str] = ()) -> None:
        from memrank.metrics.scoring import SpanRecall

        self.name = name
        self.scorer = scorer if scorer is not None else SpanRecall()
        self._questions = _Questions(questions, criterion_names=criterion_names,
                                     prompt_keys=prompt_keys)
        self._adopt_source_flags()
        if dataset_version is not None:
            self.dataset_version = dataset_version
        if (kind := _kind(self.scorer)):
            self.quality_metric = kind
        self.info = info if info is not None else EvalInfo(
            unit="unit", units_declared=self._questions.count(), slices=())
        self._report_template = report_template
        self.check_declarations()

    def _adopt_source_flags(self) -> None:
        """Take the question half's own declarations about its data and protocol, where it is one."""
        source = self._questions.source
        if source is None:
            return
        for flag in _SOURCE_FLAGS:
            setattr(self, flag, getattr(source, flag))

    # ---- the check that keeps the two halves from drifting apart -----------------------

    def check_declarations(self) -> None:
        """Compare both halves' declarations and refuse a disagreement. Never fails quietly.

        Called at construction AND from :meth:`load`, which is the first thing a run does, so a
        scorer swapped onto an already-built composition is checked before any item runs.
        """
        self._check_criterion_names()
        self._check_prompt_keys()

    def _check_criterion_names(self) -> None:
        mine = tuple(self._questions.criterion_names)
        theirs = tuple(self.scorer.criterion_names)
        # Silence on either side is not a disagreement: questions that name no criteria accept
        # whatever scorer is brought, and a scorer that names none is taken at the questions'.
        if not mine or not theirs or set(mine) == set(theirs):
            return
        raise CriteriaMismatch(
            f"{self.name!r} composes questions declaring criteria {sorted(mine)} with scorer "
            f"{self.scorer.name()} declaring {sorted(theirs)}: only in the questions "
            f"{sorted(set(mine) - set(theirs))}, only in the scorer "
            f"{sorted(set(theirs) - set(mine))}. Refused before any item ran, because an "
            f"aggregation written over one set of criterion names cannot be taken over another.")

    def _check_prompt_keys(self) -> None:
        """The judge half of the same check -- see this module's docstring for why it is here."""
        shape = self.scorer.judge_shape()
        theirs = tuple(getattr(shape, "prompts", None) or ())
        mine = tuple(self._questions.prompt_keys)
        if not mine or not theirs or set(mine) == set(theirs):
            return
        raise CriteriaMismatch(
            f"{self.name!r} composes questions carrying judge prompt keys {sorted(mine)} with "
            f"scorer {self.scorer.name()} defining {sorted(theirs)}: only in the questions "
            f"{sorted(set(mine) - set(theirs))}, only in the scorer "
            f"{sorted(set(theirs) - set(mine))}. Refused before any item ran: grading a query "
            f"under a prompt written for a different question type is how 12% of a benchmark "
            f"was graded against the wrong object for months "
            f"(memrank/judging/shape.py, BinaryJudgeShape._prompt_for).")

    # ---- the Benchmark contract, served from the two halves ----------------------------

    def load(self) -> list[BenchmarkUnit]:
        self.check_declarations()
        return self._questions.load()

    def score(self, unit: BenchmarkUnit,
              responses: list[AdapterResponse]) -> dict[str, Any]:
        scored = self.scorer.score(unit, responses)
        missing = [key for key in self.scorer.criterion_names if key not in scored]
        if missing:
            raise CriteriaMismatch(
                f"scorer {self.scorer.name()} declares criteria "
                f"{sorted(self.scorer.criterion_names)} and returned "
                f"{sorted(scored)} for unit {unit.unit_id!r}, missing {sorted(missing)}. A "
                f"declared criterion that never arrives is a scorer bug, not a missing value.")
        return scored

    def report_template(self) -> str:
        if self._report_template is not None:
            return self._report_template
        if self._questions.source is not None:
            return self._questions.source.report_template()
        return (f"# {self.name} report -- {{adapter}}\n\n"
                f"- Composite: **{{composite}}**\n- Dataset: {{dataset_version}}\n")

    def judge_shape(self) -> JudgeShape:
        """The scorer's shape where it has one, else the questions' own."""
        if (mine := self.scorer.judge_shape()) is not None:
            return mine
        if self._questions.source is not None:
            return self._questions.source.judge_shape()
        return super().judge_shape()

    def rollup(self, per_unit_scores: list[dict[str, Any]], *,
               ranked: bool = True) -> dict[str, Any]:
        """The question half's run-level reduction, which belongs to its protocol, not to this."""
        if self._questions.source is not None:
            return self._questions.source.rollup(per_unit_scores, ranked=ranked)
        return {}

    def config_for_receipt(self) -> dict[str, Any]:
        """The knobs, plus WHICH scorer decided -- two scorers are not one reproducible run."""
        source = self._questions.source
        knobs = source.config_for_receipt() if source is not None else super().config_for_receipt()
        return {**knobs, "scorer": self.scorer.name(),
                "questions": source.name if source is not None else self.name}
