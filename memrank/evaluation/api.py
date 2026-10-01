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
"""``memrank.run`` -- evaluate a memory engine from a Python script, and nothing else.

The supported library entry point: resolve the engine and the evaluation the way the CLI
does, run the cell, return a typed :class:`EvalResult`. No run directory, no
``status.json``, no sync, no cloud -- the only disk touched is the dataset/tokenizer/judge
caches, and the only network egress is the engine under test (plus Anthropic when judged).

The two parameters were once ``target`` and ``eval`` -- the command line's own nouns, on a
Python function. Both are still accepted as keywords and warn; see
:data:`DEPRECATED_PARAMETERS`, which is what ``tests/repo/test_python_vocabulary.py``
enumerates so a third spelling cannot appear without a deliberate edit.
"""
from __future__ import annotations

import uuid
import warnings
from dataclasses import replace
from pathlib import Path
from typing import cast

from memrank import deprecation
from memrank.core import Benchmark, MemoryAdapter
from memrank.evaluation.cell import _close_adapter, run_cell
from memrank.evaluation.constants import DEFAULT_JUDGE_WORKERS
from memrank.evaluation.observer import NULL_OBSERVER, EvalObserver
from memrank.evaluation.result import EvalResult
from memrank.judging.judge import JudgeConfig

#: ``{deprecated spelling: the name it was replaced by}``. Both were the command line's
#: nouns: ``target`` means the expected answer in machine learning and a deployment
#: destination in operations, and ``eval`` shadows a builtin. Removed at plan step 22.
DEPRECATED_PARAMETERS = {"target": "engine", "eval": "evaluation"}


def _settled(supplied: object | None, deprecated: object | None, old_name: str) -> object:
    """The value of a parameter that has two spellings, one of them deprecated.

    Giving both raises rather than choosing: a caller who passes ``engine=`` and ``target=``
    has two intentions and a silent winner would measure whichever one this function picked.

    Returns ``object`` because mypy cannot carry a type variable through two optional unions
    -- ``pick(x: T | None, y: T | None) -> T`` called with ``str | Adapter | None`` infers
    ``object``, verified against the pinned mypy. The two call sites cast, which is where the
    real type is known and readable.
    """
    current = DEPRECATED_PARAMETERS[old_name]
    if deprecated is None:
        if supplied is None:
            raise TypeError(
                f"run() needs {current}: the {current} to measure, as a catalog name or an "
                f"object you built")
        return supplied
    if supplied is not None:
        raise TypeError(
            f"run() got both {current}= and {old_name}=; {old_name} is the deprecated spelling "
            f"of {current}, so pass one of them")
    warnings.warn(
        f"run({old_name}=...) is deprecated; pass {current}= instead",
        DeprecationWarning, stacklevel=3)
    return deprecated


def run(
    engine: str | MemoryAdapter | None = None,
    evaluation: str | Benchmark | None = None,
    *,
    k: int = 10,
    repeats: int = 3,
    seed: int = 42,
    token_budget: int = 5000,
    model: str = "gpt-4o-mini",
    judge: bool | JudgeConfig | None = None,
    units: list | None = None,
    workers: int = 1,
    judge_workers: int = DEFAULT_JUDGE_WORKERS,
    fail_fast: bool = False,
    observer: EvalObserver | None = None,
    checkpoint_path: Path | None = None,
    target: str | MemoryAdapter | None = None,
    eval: str | Benchmark | None = None,  # noqa: A002 - the deprecated spelling of `evaluation`
) -> EvalResult:
    """Run one evaluation cell and return its result. Writes nothing, prints nothing.

    Args:
        engine: A catalog name (``"mem0"``, ``"hindsight:matched"``, ``"word-overlap"``)
            or a ready :class:`MemoryAdapter` instance. A name is resolved exactly as
            ``memrank submit`` resolves it; a container-backed engine must already be
            reachable -- this function provisions nothing.
        evaluation: A catalog name (``"demo"``, ``"locomo:smoke"``, ``"beam:100k-smoke"``)
            or a :class:`Benchmark` instance. ``load()`` may download the dataset into the
            cache on first use.
        judge: ``None`` asks the benchmark (`judge_required`) -- the evals whose
            protocol IS the judge get one by default; ``True``/``False`` force it; a
            :class:`JudgeConfig` is used as given. Judging calls Anthropic and keeps a
            local verdict cache (``JudgeConfig(cache=False)`` opts out).
        fail_fast: Stop at the first unit that raises. The default attempts every unit and
            records what happened to each on ``result.unit_outcomes``, with the counts under
            ``units_total``/``units_failed``.
        observer: Where progress narration goes. ``None`` is silent -- pass an
            :class:`EvalObserver` subclass to hear about units, items and warnings.
        checkpoint_path: ``None`` writes no checkpoint. Give a path to make a judged
            run resumable (``memrank ops rejudge``) at the cost of one file.
        target: Deprecated spelling of ``engine``. Accepted, warns, removed at step 22.
        eval: Deprecated spelling of ``evaluation``. Accepted, warns, removed at step 22.

    Defaults mirror ``memrank submit``'s (k=10, repeats=3, seed=42, token_budget=5000,
    priced at gpt-4o-mini), so a script and a sweep of the same cell agree.

    Note: ``seed`` seeds the process-global ``random`` module, and the tokenizer
    downloads (~4 MB, cached) on first cost accounting -- both inherited from the
    measurement loop and documented rather than hidden.
    """
    deprecation.warn("memrank.evaluation.api.run()")
    chosen_engine = cast("str | MemoryAdapter", _settled(engine, target, "target"))
    chosen_evaluation = cast("str | Benchmark", _settled(evaluation, eval, "eval"))

    if isinstance(chosen_evaluation, Benchmark):
        benchmark = chosen_evaluation
    else:
        from memrank.benchmarks import from_ref

        benchmark = from_ref(chosen_evaluation)

    if judge is None:
        # The single definition of "does this eval need judging" -- an unjudged run of a
        # judge-protocol benchmark measures latency and cost and nothing else.
        judge_cfg = JudgeConfig() if not benchmark.composite_rankable else None
    elif judge is True:
        judge_cfg = JudgeConfig()
    elif judge is False:
        judge_cfg = None
    else:
        judge_cfg = judge

    ref: str | None = None
    make_adapter = None
    owns_adapter = False
    if isinstance(chosen_engine, MemoryAdapter):
        adapter = chosen_engine
    else:
        from memrank.targets import resolve_target
        from memrank.targets.factory import build_adapter

        ref = chosen_engine

        def make_adapter() -> MemoryAdapter:
            return build_adapter(resolve_target(ref))

        adapter = make_adapter()
        owns_adapter = True

    try:
        result = run_cell(
            adapter, benchmark, k=k, repeats=repeats,
            run_id_prefix=f"{adapter.name}-{uuid.uuid4().hex[:8]}",
            model=model, token_budget=token_budget, seed=seed, judge=judge_cfg,
            units=units, workers=workers, judge_workers=judge_workers, fail_fast=fail_fast,
            checkpoint_path=checkpoint_path, make_adapter=make_adapter,
            observer=observer if observer is not None else NULL_OBSERVER,
            )
    finally:
        if owns_adapter:
            _close_adapter(adapter)
    if ref is not None:
        result = replace(result, target=ref)
    return result
