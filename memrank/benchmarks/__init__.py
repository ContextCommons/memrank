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
"""Benchmark registry.

Public benchmark classes register themselves here for the CLI and the
conformance suite. Each benchmark wraps both dataset loading and scoring.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from memrank.benchmarks.beam import BEAMBenchmark
from memrank.benchmarks.demo import DemoBenchmark
from memrank.benchmarks.locomo import LoCoMoBenchmark
from memrank.benchmarks.longmemeval import LongMemEvalBenchmark
from memrank.benchmarks.relation_graph import RelationGraphBenchmark
from memrank.core import Benchmark

REGISTRY: dict[str, type[Benchmark]] = {
    "demo": DemoBenchmark,
    "locomo": LoCoMoBenchmark,
    "beam": BEAMBenchmark,
    "longmemeval": LongMemEvalBenchmark,
    "relation_graph": RelationGraphBenchmark,
}


def get_benchmark(name: str, **kwargs) -> Benchmark:
    """Construct the benchmark registered under ``name``."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown benchmark: {name!r}. Available: {available}")
    return REGISTRY[name](**kwargs)


def judge_required(name: str) -> bool:
    """Whether ``name`` produces no quality number at all without a judge.

    True for the benchmarks whose published protocol IS the judge -- locomo, longmemeval and beam
    all grade a generated answer and report no self-contained composite, so an unjudged run of one
    measures latency and cost and nothing else. False where the benchmark has a valid metric of
    its own (``demo``'s substring recall over verbatim spans, ``relation_graph``'s graph score),
    and judging adds a second measurement rather than the only one.

    The single definition of "does this eval need judging". The CLI derives the default from it,
    `evals show` states it, and the browser seeds its switch from it; three places asking the
    benchmark separately is how the answer starts to differ between doors.

    Constructing a benchmark is I/O-free by contract -- only ``load()`` may download -- so this is
    safe on a request path.
    """
    return not get_benchmark(name).composite_rankable


def resolve_eval(ref: str, **extra: Any) -> tuple[Benchmark, str]:
    """Construct the evaluation named by ``ref``, and return it with its canonical ref.

    The canonical ref is what artifacts and run rows are keyed on, so `beam` and `beam:100k`
    cannot produce two differently-named records of the same evaluation. `extra` carries the
    knobs that are genuinely run-level rather than part of the evaluation's identity (`k`).
    """
    from memrank.benchmarks.refs import parse_eval_ref

    name, kwargs, canonical = parse_eval_ref(ref)
    return REGISTRY[name](**{**kwargs, **extra}), canonical


def from_ref(ref: str, **overrides: Any) -> Benchmark:
    """The benchmark an eval ref names, constructed -- ``from_ref("beam:100k-smoke")``.

    The library-door twin of the CLI's EVAL argument (and of ``memrank.run``'s ``eval``
    parameter -- exported at the top level as ``memrank.benchmark``): one string in the
    same grammar, one configured :class:`Benchmark` out. ``overrides`` are constructor
    keywords laid over what the ref declares (``from_ref("demo", k=5)``), so a knob the
    ref grammar does not spell is still reachable without abandoning the ref.

    Construction is I/O-free by contract -- only ``load()``/``raw()`` may download.
    Unknown refs raise :class:`memrank.targets.resolve.RefError` naming the known
    variants. The thin face of :func:`resolve_eval` for callers that don't need the
    canonical ref back.
    """
    bench, _canonical = resolve_eval(ref, **overrides)
    return bench


def load_raw(name: str, **kwargs: Any) -> list[dict[str, Any]]:
    """The dataset's records exactly as upstream ships them -- see :meth:`Benchmark.raw`.

    The instance method is the primary form (``memrank.benchmark("beam:100k").raw()``
    answers for its own tier); this module-level form stays for callers that start from
    a registry name.

    ``demo`` returns its one bundled scenario as a single record, and ``relation_graph``
    its in-repo fixtures; neither has an upstream, and both answer here so a caller can
    ask every benchmark the same question.
    """
    return get_benchmark(name, **kwargs).raw()


def list_benchmarks() -> list[str]:
    """Return the registered benchmark names in deterministic order."""
    return sorted(REGISTRY.keys())


def cache_root() -> Path:
    """Return the on-disk dataset cache root.

    Honors ``MEMRANK_CACHE_DIR`` for users who want a custom location.
    Defaults to ``~/.memrank/datasets``.
    """
    base = os.environ.get("MEMRANK_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".memrank" / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def dataset_download_notice(name: str, source: str, dest: Path) -> Iterator[None]:
    """Narration bracket around a one-time dataset download (may take minutes).

    Spoken as ``logging`` INFO rather than printed: a library caller hears it through
    ordinary logging config (or not at all), and the CLI's `term.style.install_log_bridge`
    renders it to stderr exactly as before.
    """
    log = logging.getLogger(__name__)
    log.info("downloading %s dataset from %s", name, source)
    log.info("  -> %s (one-time, cached afterwards) ...", dest)
    yield
    log.info("%s dataset ready", name)


__all__ = [
    "BEAMBenchmark",
    "DemoBenchmark",
    "LoCoMoBenchmark",
    "LongMemEvalBenchmark",
    "RelationGraphBenchmark",
    "REGISTRY",
    "cache_root",
    "dataset_download_notice",
    "from_ref",
    "get_benchmark",
    "list_benchmarks",
    "load_raw",
]
