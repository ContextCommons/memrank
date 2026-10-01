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
"""Local run registry -- accumulate every ``memrank submit`` for later comparison.

``memrank submit`` keeps writing its "latest" per-cell file to ``--output-dir`` (overwritten),
AND additionally snapshots each run into ``runs/<run-id>/`` here, so history piles up instead
of being clobbered. ``list-runs`` browses it and ``compare-versions`` reads from it.

Root is ``runs/`` (repo-relative), overridable via ``MEMRANK_RUNS_DIR`` -- the test suite points
that at a tmp dir so it never writes into the repo. This is a LOCAL filesystem registry; it is
unrelated to the object-storage ``runs/<sha>.json`` leaderboard prefix.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memrank.evaluation import result as result_schema
from memrank.metrics import headline as headline_lib
from memrank.term import fmt

#: Where a checkout's runs used to land. Kept only so the CLI can NOTICE one and say where its
#: results went; nothing reads or writes it any more.
LEGACY_ROOT = "runs"


def runs_root() -> Path:
    """Where this machine keeps its runs.

    ``$MEMRANK_RUNS_DIR``, else ``$XDG_DATA_HOME/memrank/runs``, else
    ``~/.local/share/memrank/runs``.

    Per-user and ABSOLUTE. It used to be the relative ``runs/``, so the registry was whichever
    directory you happened to be standing in: `memrank runs show <id>` answered with a score in one
    checkout and "none recorded yet" in another, and a run id stopped being a handle. That was
    survivable while memrank was a repo you `cd`'d into, and stopped being survivable the moment it
    became a tool on PATH that people run from anywhere.

    Data rather than configuration, so it takes the XDG data location instead of joining the wallet
    and settings in ``~/.config/memrank`` (:func:`memrank.secrets.wallet.config_dir`) -- run
    artifacts grow without bound and are not something to carry in a dotfiles repo.
    """
    configured = os.environ.get("MEMRANK_RUNS_DIR")
    if configured:
        return Path(configured)
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / "memrank" / "runs"


def legacy_runs(cwd: Path | None = None) -> Path | None:
    """A checkout-local ``runs/`` holding results the registry root does not, or ``None``.

    Reported, never touched. Two checkouts hold overlapping id space, so merging them silently is
    exactly where a receipt could end up attached to the wrong run --
    ``scripts/internal/one-offs/migrate-runs.py`` does it deliberately and reversibly instead.

    Args:
        cwd: Directory to look in; the process's own by default.

    Returns:
        The legacy directory when it holds run ids absent from the registry root.
    """
    legacy = (cwd or Path.cwd()) / LEGACY_ROOT
    if not legacy.is_dir() or legacy.resolve() == runs_root().resolve():
        return None
    here = {d.name for d in legacy.iterdir() if d.is_dir()}
    if not here:
        return None
    root = runs_root()
    known = {d.name for d in root.iterdir() if d.is_dir()} if root.is_dir() else set()
    return legacy if here - known else None


def mint_run_id(benchmark: str, slice_: str | None = None, tier: str | None = None) -> str:
    """A fresh run id: ``<UTC-YYYYMMDD-HHMMSS>__<benchmark>[__slice][__tier]__<short-uuid>``.

    Sortable and readable; the short uuid keeps back-to-back runs distinct within the same
    second. Pure naming, no filesystem side effect -- the accounts API mints server-side run
    ids with this too, so the one shape stays defined once. The alphabet is shell- and
    S3-safe by construction (``[A-Za-z0-9._-]``), which ``remote_command`` re-checks.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    parts = [stamp, benchmark, *(p for p in (slice_, tier) if p), uuid.uuid4().hex[:6]]
    return "__".join(parts)


def new_run_dir(benchmark: str, slice_: str | None = None, tier: str | None = None,
                run_id: str | None = None) -> Path:
    """Create and return a fresh, unique run folder for one ``run`` invocation.

    When ``run_id`` is given (e.g. a ``--detach`` parent that already picked the id), that
    exact folder name is reused instead of generating a new one.
    """
    if run_id is None:
        run_id = mint_run_id(benchmark, slice_, tier)
    run_dir = runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def cell_files(run_dir: Path) -> list[Path]:
    """The per-cell result files in a run folder.

    Matched POSITIVELY on the cell naming convention -- ``<target>__<benchmark>.json`` -- rather than
    by excluding known non-cells. The blocklist version excluded only ``summary__`` and therefore
    swallowed ``status.json`` the moment the run-status heartbeat started writing one into every
    run directory (16e358d). That made every run list twice, once as a phantom
    ``?xdemo composite=-``, and broke `compare-versions <run-id>` outright: a single-cell run looked
    like two cells, so it raised "holds multiple cells; pass --adapter to disambiguate" for
    every run that existed.

    A blocklist fails open -- the next metadata file lands in the same trap. This fails closed.
    """
    return sorted(p for p in run_dir.glob("*__*.json") if not p.name.startswith("summary__"))


@dataclass
class RunInfo:
    """One recorded cell (an adapter x benchmark result) within a run folder."""

    run_id: str
    path: Path
    adapter: str
    benchmark: str
    config_hash: str | None
    composite: float | None
    #: False only when the artifact says so. A composite the methodology refuses to rank is
    #: not a score, and showing it as one puts it into comparisons it was excluded from.
    composite_rankable: bool
    timestamp: str
    #: The cell's headline score -- judged when the run was judged, else the rankable composite,
    #: else nothing. This is what a listing shows; ``composite`` above stays because comparison
    #: and JSON consumers read the raw number. Projected by ``memrank.metrics.headline``, the
    #: one place that decides, so this row and the same run in the GUI cannot read differently.
    headline: float | None = None
    headline_kind: str = headline_lib.WITHHELD
    headline_coverage: float | None = None


def _read_info(run_id: str, cell_path: Path) -> RunInfo | None:
    """One cell, or ``None`` when its artifact cannot be parsed.

    A truncated or half-written result file used to raise out of every listing that touched the
    registry -- so ONE bad artifact made `runs ls`, `runs show` and `ps` fail entirely, for every
    run. A partial upload or an interrupted write is a normal thing to find on disk; losing the
    other 235 runs to it is not.

    The cell is skipped rather than shown as empty: a row with no adapter and no score is not a
    result, and pretending otherwise would put it in comparisons.
    """
    try:
        data = json.loads(cell_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    receipt = data.get("receipt", {})
    head = headline_lib.cell_headline(data)
    return RunInfo(
        run_id=run_id, path=cell_path,
        adapter=result_schema.adapter_if_present(data) or receipt.get("adapter_name", "?"),
        benchmark=(result_schema.benchmark_if_present(data)
                   or receipt.get("benchmark_name", "?")),
        config_hash=receipt.get("config_hash"),
        composite=data.get("composite"),
        composite_rankable=data.get("composite_rankable") is not False,
        timestamp=receipt.get("started_at") or run_id,
        headline=head.value, headline_kind=head.kind, headline_coverage=head.coverage,
    )


def list_runs(adapter: str | None = None, benchmark: str | None = None) -> list[RunInfo]:
    """Every recorded cell, oldest first, optionally filtered by adapter/benchmark."""
    root = runs_root()
    if not root.exists():
        return []
    infos: list[RunInfo] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cell in cell_files(run_dir):
            info = _read_info(run_dir.name, cell)
            if info is None:
                continue
            if adapter and info.adapter != adapter:
                continue
            if benchmark and info.benchmark != benchmark:
                continue
            infos.append(info)
    infos.sort(key=lambda i: (i.timestamp, i.run_id))
    return infos


def cells_by_run() -> dict[str, list[RunInfo]]:
    """Every recorded cell, grouped under the run that produced it."""
    grouped: dict[str, list[RunInfo]] = defaultdict(list)
    for info in list_runs():
        grouped[info.run_id].append(info)
    return grouped


def resolve(ref: str, *, adapter: str | None = None, benchmark: str | None = None) -> Path:
    """Resolve a comparison reference to a cell file: a path passes through; a run-id
    resolves to that folder's cell (disambiguated by adapter/benchmark when it holds several)."""
    as_path = Path(ref)
    if as_path.is_file():
        return as_path
    run_dir = runs_root() / ref
    if not run_dir.is_dir():
        raise FileNotFoundError(f"not a result file or run-id: {ref!r}")
    cells = cell_files(run_dir)
    if adapter and benchmark:
        cells = [c for c in cells if c.name == f"{adapter}__{benchmark}.json"]
    elif adapter:
        cells = [c for c in cells if c.name.startswith(f"{adapter}__")]
    if len(cells) == 1:
        return cells[0]
    if not cells:
        raise FileNotFoundError(f"no matching cell in run {ref!r}")
    raise ValueError(f"run {ref!r} holds multiple cells; pass --adapter to disambiguate")


def latest(adapter: str, benchmark: str, n: int = 2) -> list[Path]:
    """The ``n`` most recent recorded cells for ``adapter``x``benchmark`` (oldest->newest)."""
    return [info.path for info in list_runs(adapter=adapter, benchmark=benchmark)[-n:]]


def _unjudged_reasons(judged: dict[str, Any]) -> str:
    """Why the judge skipped what it skipped, as one line -- or empty when it skipped nothing.

    Named reasons rather than a bare count because the two that occur are fixed in different
    places: a missing gold answer is a loader defect, an unsupported category is a judge one.
    """
    categories = ", ".join(judged.get("unsupported_categories") or [])
    counted = (
        (judged.get("n_unjudged_no_gold"), "no gold answer"),
        (judged.get("n_unjudged_unsupported_category"),
         f"unsupported category ({categories})" if categories else "unsupported category"),
        (judged.get("n_unjudged_unparseable_verdict"), "unparseable verdict"),
    )
    return ", ".join(f"{n} {reason}" for n, reason in counted if n)


def _judged_quality(judged: dict[str, Any]) -> dict[str, tuple[Any, str]]:
    """What the LLM judge measured, and over how much of the benchmark.

    Absent from the view until now, which is how a judged BEAM run came to be read as its 0.065
    substring recall -- the figure the same block labels "benchmark ranks unjudged: no" -- while
    its 0.275 judged correctness sat unrendered in the same artifact.

    COVERAGE RIDES ON THE NUMBER rather than sitting on its own line below. 0.275 over half the
    queries is not 0.275 over the benchmark, and a separate `judged_coverage: 0.5` two lines down
    is an invitation to quote the first without the second.
    """
    if not judged:
        return {}
    total = (judged.get("n_judged") or 0) + (judged.get("n_unjudged") or 0)
    correctness = judged.get("answer_correctness")
    coverage = f"  ({judged.get('n_judged')}/{total} queries judged)" if total else ""
    return {
        "answer correctness": (
            f"{fmt.measurement(correctness)}{coverage}" if correctness is not None else None,
            "text"),
        # The headline minus the questions a model answers from its own weights. Where memory had
        # to do the work, and so the number that says whether the engine helped.
        "answer correctness (context-dependent)": (
            judged.get("answer_correctness_context_dependent"), "score"),
        "retrieval sufficiency": (judged.get("retrieval_sufficiency"), "score"),
        # LongMemEval publishes three numbers and its ADR makes the micro-mean above the
        # headline, on condition these two always appear beside it. Absent (None) for every
        # benchmark that declares no extra aggregates, which is all of the others.
        "answer correctness (task-averaged)": (
            judged.get("answer_correctness_task_averaged"), "score"),
        "abstention accuracy": (judged.get("abstention_accuracy"), "score"),
        "unjudged": (_unjudged_reasons(judged) or None, "text"),
    }


def _rate_limit(raw: dict[str, Any], field: str) -> Any:
    """A rate-limit stat from the receipt, or ``None`` for the runs that predate it.

    ``None`` rather than ``0`` for absence, on the same rule as everything else here: zero pauses
    is a measurement, and an artifact that never recorded one must not appear to claim it.
    """
    return (((raw.get("receipt") or {}).get("extra") or {}).get("rate_limit") or {}).get(field)


#: The unit a group's rates are counted in, for :func:`memrank.term.fmt.rate`. Beside the groups
#: rather than in a renderer: ingest throughput is documents per second in every surface that shows
#: it, and a second renderer that guessed "item" would quietly relabel the same measurement.
RATE_NOUNS = {"latency": "doc"}


def cell_metrics(path: Path) -> dict[str, Any]:
    """Everything one result artifact recorded, grouped for reading.

    Args:
        path: A result artifact, i.e. ``RunInfo.path``.

    Returns:
        :func:`metrics_from` of the artifact's contents, or ``{"error": ...}`` when it cannot be
        read -- a corrupt third cell must not hide a sweep's fourth.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": {"artifact": f"{path} could not be read: {exc}"}}
    return metrics_from(raw)


def metrics_from(raw: dict[str, Any]) -> dict[str, Any]:
    """Everything one cell recorded, grouped for reading.

    Takes the cell as a DICT rather than a path because a cell is read from two places now: the
    CLI opens the artifact on disk, and the API reads the synced record out of Postgres. Splitting
    the grouping from the file read is what lets both of them mean the same thing -- the alternative
    was a second implementation in TypeScript, and the browser and the terminal would have started
    disagreeing about a run at the first renamed field.

    The listing carries only ``composite`` because that is all a listing needs. A run also measures
    latency percentiles, token counts, an estimated cost per query and the provenance of the engine
    that produced them -- and none of it was reachable without opening the JSON by hand.

    Absent fields are OMITTED rather than defaulted. A supermemory run genuinely records no tokens,
    and printing ``0.0`` would assert something the artifact does not say.

    Args:
        raw: One cell, as written by ``runner._aggregate_cell`` (with or without the bulk keys
            a synced record strips).

    Each value carries its KIND (``seconds``, ``bytes``, ``count``, ``rate``, ``score``,
    ``money``, ``text``) so the renderer formats rather than guesses. Declared here, beside the
    value, because the alternative -- matching a formatter to a label string -- puts one fact in
    two places, and a renamed label then starts printing seconds as bytes with nothing failing.

    Returns:
        ``{group: {label: (value, kind)}}`` in reading order.
    """

    def present(mapping: dict[str, tuple[Any, str]]) -> dict[str, tuple[Any, str]]:
        return {k: v for k, v in mapping.items() if v[0] is not None}

    latency = raw.get("latency_metrics") or {}
    tokens = raw.get("token_metrics") or {}
    throughput = raw.get("ingest_throughput") or {}
    basis = raw.get("cost_basis") or {}
    provenance = (raw.get("receipt") or {}).get("engine_provenance") or {}
    properties = provenance.get("properties") or {}
    env = (raw.get("receipt") or {}).get("environment") or {}
    groups = {
        "quality": present({
            # `score`, so it renders at full precision. A composite is the run's PRODUCT; two runs
            # differing in the fifth decimal are two results, and a tidier rendering hides that.
            raw.get("quality_metric", "composite"): (raw.get("composite"), "score"),
            # A property of the BENCHMARK, not of this run: whether its composite ranks without a
            # judge. Labelled precisely because "rankable" alone reads as a claim about the number
            # beside it -- which is a different question, and one this field does not answer.
            "benchmark ranks unjudged": (raw.get("composite_rankable"), "text"),
            **_judged_quality(raw.get("judged_metrics") or {}),
            "k": (raw.get("k"), "count"), "repeats": (raw.get("repeats"), "count"),
            "units": (raw.get("n_units"), "count"),
            # Beside the unit count, because the composite above is a mean over the units that
            # RAN: a cell that lost twelve of fifty is not a score over fifty. Omitted for the
            # runs that predate per-unit outcomes -- `present` drops None, and 0 there would
            # claim an all-units-attempted run the artifact never recorded.
            "units failed": (raw.get("units_failed"), "count"),
        }),
        "latency": present({
            # HOW LONG THE RUN TOOK -- the first thing anyone asks, and until now the one thing this
            # never said. `finalize_receipt` has always stamped it; nothing rendered it.
            #
            # Deliberately the line ABOVE the summed latency rather than a header field. The two
            # are both time-shaped and their ratio is roughly the worker count, so adjacent and
            # differently labelled they explain each other; separated, the sum was the only
            # duration on screen and got read as the run's length.
            "wall clock": ((raw.get("receipt") or {}).get("duration_seconds"), "seconds"),
            "retrieve p50/p95/p99 ms": (_percentiles(latency, "retrieve"), "text"),
            "ingest p50/p95/p99 ms": (_percentiles(latency, "ingest"), "text"),
            # Percentiles cannot say what ingest COST: p50=12,750 ms is the same number whether
            # seventy documents paid it or one did. Engines differ here by four orders of
            # magnitude, which the quality column never shows.
            "ingest rate": (throughput.get("documents_per_second"), "rate"),
            # NOT elapsed time, and labelled so nobody reads it as one. It sums every document's
            # latency, so a concurrent run overstates the clock by roughly its worker count: an
            # 11m 30s run rendered here as "ingest total 50m 49s" and was reported as taking most
            # of an hour. Wall clock lives in the run's own timestamps, not in a latency summary.
            "ingest summed latency": (throughput.get("latency_total_seconds")
                                      or throughput.get("total_seconds"), "seconds"),
            # Contention makes a latency number incomparable. Reporting the percentile without it
            # is the same defect one level down from reporting a score without its config.
            "contended": (raw.get("latency_contended"), "text"),
            # Time the run spent deliberately idle, waiting out a provider rate limit. It inflates
            # WALL CLOCK without touching the percentiles above, so a run that paused for six
            # minutes and one that never paused are not the same measurement -- and every other
            # number here would look identical.
            "rate-limit pauses": (_rate_limit(raw, "pauses"), "count"),
            "paused waiting on quota": (_rate_limit(raw, "paused_seconds"), "seconds"),
        }),
        "corpus": present({
            "documents": (raw.get("corpus_documents"), "count"),
            "bytes": (raw.get("corpus_bytes"), "bytes"),
            # Same pinned encoding as context tokens, so the two divide into each other.
            "tokens": (raw.get("corpus_tokens"), "count"),
        }),
        "cost (hypothetical prompt cost)": present({
            # `money`, not rounded: 6.615e-05 rounded to cents is zero, and the magnitude IS the
            # information here.
            "$/query (est)": (raw.get("est_dollars_per_query"), "money"),
            "$/cell (est)": (raw.get("est_dollars_per_cell"), "money"),
            # What the estimate assumes. It prices the retrieved context against a named model at
            # a pinned table -- NOT what the run spent, and not necessarily a model the engine used.
            "priced as": (basis.get("model"), "text"),
            "excludes": (", ".join(basis.get("excludes") or ()) or None, "text"),
            # `score`, not `count`: these are MEANS, and 440.553 rounded to 440 discards the
            # part that distinguishes two engines' context budgets. An exact corpus token count
            # (above) is a count; an average of one is a measurement.
            "context tokens (mean)": (raw.get("context_tokens_mean"), "score"),
            # `or None` deliberately, and only here. Artifacts written before the collector
            # distinguished them store "the engine reported no usage" as 0.0 -- the same value a
            # genuine zero would take -- and nothing can tell them apart after the fact. New
            # artifacts record None, so the ambiguity stops being created; this keeps the older
            # ones from reading as a measured zero.
            "tokens/ingest (mean)": (tokens.get("tokens_per_ingest_mean") or None, "score"),
            "tokens/query (mean)": (tokens.get("tokens_per_query_mean") or None, "score"),
        }),
        "engine": present({
            "purl": (provenance.get("purl"), "text"),
            "release (index)": (properties.get("memrank:index_digest"), "text"),
            "platform": (properties.get("memrank:platform"), "text"),
        }),
        # The conditions the numbers above were measured under. Beside them deliberately: a
        # latency percentile without its environment is the same defect as a score without its
        # config, one level out. Absent for artifacts written before schema 2.
        "environment": present({
            "place": (env.get("place"), "text"),
            "machine": (_machine(env), "text"),
            "cpus": (env.get("cpu_count"), "count"),
            "memory": (env.get("memory_bytes"), "bytes"),
        }),
    }
    return {name: fields for name, fields in groups.items() if fields}


def cell_environment(path: Path) -> dict[str, Any]:
    """The conditions one artifact was measured under, or ``{}`` for one written before schema 2.

    Read straight from the receipt rather than recomputed: this must describe the machine that
    produced the numbers, which for a cloud run is a Fargate task and never the laptop reading it.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return (raw.get("receipt") or {}).get("environment") or {}


def _machine(env: dict[str, Any]) -> str | None:
    """``arm64/darwin 25.5.0`` -- one cell, because the three are read together or not at all."""
    if not env.get("arch"):
        return None
    return f"{env['arch']}/{env.get('os', '?')} {env.get('os_version', '')}".strip()


def _percentiles(latency: dict[str, Any], phase: str) -> str | None:
    """``703 / 1040 / 1441`` for one phase, or ``None`` when it measured nothing."""
    values = [latency.get(f"{phase}_p{p}_ms") for p in (50, 95, 99)]
    if not any(v is not None for v in values):
        return None
    return " / ".join("—" if v is None else f"{v:.0f}" for v in values)
