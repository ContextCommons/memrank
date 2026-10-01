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
"""Reproducibility receipt format.

A receipt is the structured metadata block published alongside every
Memrank result. The vendor-neutral charter requires that no number is
published without one. The receipt makes the run reproducible: third
parties can re-execute with the same adapter/engine/dataset versions and
should get the same composite within published noise.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memrank.provenance import environment, install
from memrank.quality import is_reproducible
from memrank.secrets.names import is_secret_key

#: The receipt layout THIS memrank writes. Version 3 adds a centrally-derived evidence class so a
#: native development observation cannot be mistaken for artifact-backed published evidence.
SCHEMA_VERSION = 3


@dataclass
class Receipt:
    """Reproducibility receipt for a single (adapter, benchmark) run.

    Persist via :meth:`to_json`; load via :meth:`from_dict`. The
    ``config_hash`` field is a sha256 of the structured config so two runs
    with identical configs but different timestamps still match.
    """

    memrank_version: str
    adapter_name: str
    adapter_version: str
    engine_version: str
    benchmark_name: str
    dataset_version: str
    seed: int
    started_at: str
    # Bumped when the receipt's own field layout changes in a breaking way, so a
    # comparator can flag cross-schema comparisons. Config-level schema growth
    # (new keys inside ``config``) is handled by the open-dict design + tolerant
    # ``from_dict``, not this field.
    #
    # The DEFAULT stays 1 while :data:`SCHEMA_VERSION` moves: this value is what a record that
    # never carried the field turns out to be, and 1 is the layout that predates it. Stamping
    # the current version here instead would have every legacy artifact claim the newest schema
    # -- the same reasoning as ``config_hash_version``'s 0.
    schema_version: int = 1
    # Which revision of the config-hashing SCHEME produced ``config_hash``. Distinct from
    # ``schema_version`` (the receipt's field layout): this versions HOW the hash is computed and
    # what feeds it, so a change to either cannot silently make two runs look different when only
    # the recording changed. Defaults to 0 -- meaning "written before this field existed" -- so a
    # legacy artifact read via ``from_dict`` is never mistaken for a current one.
    config_hash_version: int = 0
    finished_at: str | None = None
    duration_seconds: float | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    config_hash: str | None = None
    # The exact (non-secret) config, persisted in full so every result ships
    # with what produced it -- not just config_hash. Secrets never enter config.
    config: dict[str, Any] = field(default_factory=dict)
    # The conditions the run was measured under (memrank/provenance/environment.py). Recorded, NEVER
    # hashed: it must not reach ``config_hash``, which defines the run's identity -- the same
    # question asked in two places has to keep one identity, or the two can never be compared.
    # Replaced the ``host`` blob (hostname + an unparseable platform string) in schema 2.
    environment: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    # Engine provenance (purl identity + CycloneDX pedigree). Kept OUT of ``config``
    # and ``config_hash``: it carries the volatile image digest / source SHA / dirty
    # flag, which change on every rebuild even when the logical config is unchanged --
    # so it lives beside ``host``/``env_vars`` as forensic pinning, not a grouping axis.
    engine_provenance: dict[str, Any] = field(default_factory=dict)
    # What claims this run is eligible to support. Derived from execution facts, never selected by
    # the operator: a workspace process is useful development signal but is not a published build.
    evidence: dict[str, Any] = field(default_factory=dict)
    # Whether a reproducibility claim can honestly be made for this run at all. Derived from
    # ``dataset_version``, never asserted: a benchmark whose material CANNOT be frozen declares
    # ``memrank.quality.UNFREEZABLE`` and this goes False, so the receipt states the limit rather
    # than reporting a version it cannot support. Distinct from ``evidence``, which grades what
    # was OBSERVED about the engine; this grades what the MATERIAL allows.
    #
    # Defaults True: every record written before this field existed was over frozen material, and
    # a legacy artifact must not be read as making a disclaimer nobody made.
    reproducible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Receipt:
        """Build from a dict, tolerant of schema drift.

        Unknown keys (from a NEWER record than this code knows) are ignored, and
        missing keys fall back to field defaults -- so records written by different
        memrank versions all load. ``config`` itself is an open dict, so config
        schema growth never requires changing this class.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


def _git_sha() -> str | None:
    """HEAD of the WORKING DIRECTORY's repository, or ``None`` when there is none.

    Not the memrank build's commit, which is what :func:`install.build_identity` answers and
    what receipts record. This asks about wherever the process is standing, so it is only
    correct for a caller that knows it runs inside this repository -- ``memrank/arena/build.py``,
    which builds Arena artifacts from a source checkout and legitimately wants that checkout.
    """
    return install.head_commit()


# Revision of the config-hashing scheme. BUMP THIS whenever what feeds ``config_hash`` changes --
# a new field in ``config``, a change to how a value is derived, or a change to the hash itself.
# Bumping deliberately invalidates cross-version comparison instead of letting a recording change
# masquerade as a change in the system under test.
#
#   1  (2026-07-30) first versioned scheme. Same run recorded before vs after this session's target
#      work hashes differently: ``config.components`` moved from ambient environment to target
#      manifests, gained ``embedder.provider``, and gained ``verified``. Anything written earlier
#      reads back as version 0 and is NOT comparable to a version-1 record.
#   2  bumped without a note. Left as found rather than back-filled with a guess -- inventing the
#      reason would be worse than recording that it is unknown.
#   3  (2026-08-10) ``config.target`` is now recorded for EVERY target, where it used to appear
#      only for source-bound ones (runner.py:_build_receipt). Container targets -- mem0, hindsight,
#      supermemory -- previously hashed without their manifest, so two runs of the same engine at
#      different retrieval depths were indistinguishable (audit F16). A version-2 record and a
#      version-3 record of the same run therefore differ by construction.
#   4  (2026-08-12) BEAM records ``config.beam_protocol`` (benchmarks/beam.py:config_for_receipt).
#      BEAM's reference harness contradicts BEAM's paper in three places, so "a BEAM score" names
#      two different metrics; without this field two runs under different protocols collide on one
#      hash. Only BEAM carries the key, but the SCHEME changed for everything, which is what a
#      version is for.
#   5  (2026-08-13) Judged configs record ``config.answer_temperature`` (reader pinned at 0;
#      runner.py:_judge_receipt_config), and BEAM records ``config.time_anchors`` /
#      ``config.context_policy`` (benchmarks/beam.py:config_for_receipt) -- the run-protocol facts
#      behind decision-beam-runs-its-own-protocol.md. A version-4 BEAM record describes a run with
#      rendered anchors and a capped reader; a version-5 record describes role+content, uncapped.
#   6  (2026-08-13) LoCoMo records ``config.locomo_dataset_sha256``
#      (benchmarks/locomo.py:config_for_receipt): the digest of the file actually scored, so two
#      artifacts share a config hash only on identical data. A version-5 LoCoMo record predates
#      the corrected category mapping (LoCoMoBenchmark.VERSION = 1) -- its per-category labels
#      are void and its judged denominator was 603 of 1,540.
#   7  (2026-08-14) LongMemEval records ``config.longmemeval_dataset_sha256``
#      (benchmarks/longmemeval.py:config_for_receipt). It matters more here than for LoCoMo,
#      which has one upstream commit ever: this dataset has shipped three named revisions, the
#      original now carries a deprecation banner, and a third-party re-review found 15.4% of the
#      cleaned file's records wrong. A version-6 LongMemEval record cannot say which artifact it
#      scored. Its slices also differ -- version-6 smoke/mini were `raw[:N]` over a type-grouped
#      file, so both covered one of six question types and no abstention item.
CONFIG_HASH_VERSION = 7

def _sanitize_value(value: Any) -> Any:
    """Recurse into nested dicts AND lists/tuples so secrets can't hide inside a
    list of provider configs."""
    if isinstance(value, dict):
        return _redact_secrets(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return value


def _redact_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``config`` with secret-looking values masked.

    Enforced (not advisory): the receipt is persisted in full, so a caller that
    accidentally passes an API key in config never leaks it into a result file --
    including keys buried inside nested dicts or lists.
    """
    return {key: ("[redacted]" if is_secret_key(key) else _sanitize_value(value))
            for key, value in config.items()}


def _hash_config(config: dict[str, Any]) -> str:
    """Stable sha256 of a config dict (sorted-key JSON)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _filtered_env(allowed_prefixes: tuple[str, ...]) -> dict[str, str]:
    """Capture env vars with names starting with any prefix.

    Values are SHA-256-fingerprinted rather than stored verbatim so secret
    keys never leak into a receipt.
    """
    out: dict[str, str] = {}
    for key, raw in os.environ.items():
        if not any(key.startswith(prefix) for prefix in allowed_prefixes):
            continue
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        out[key] = f"sha256:{digest}"
    return out


def build_receipt(
    *,
    adapter_name: str,
    adapter_version: str,
    engine_version: str,
    benchmark_name: str,
    dataset_version: str,
    seed: int,
    config: dict[str, Any] | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    extra: dict[str, Any] | None = None,
    engine_provenance: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    env_var_prefixes: tuple[str, ...] = ("MEMRANK_", "ATOMICMEMORY_", "MEM0_", "HINDSIGHT_",
                                         "SUPERMEMORY_"),
) -> Receipt:
    """Build a fresh ``Receipt`` capturing the host + environment context.

    Caller should call :meth:`Receipt.write` after the run completes to flush
    ``finished_at`` and ``duration_seconds``.
    """
    started = datetime.now(timezone.utc).isoformat()
    config = _redact_secrets(config) if config else config
    cfg_hash = _hash_config(config) if config else None
    return Receipt(
        schema_version=SCHEMA_VERSION,
        memrank_version=install.build_identity(),
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        engine_version=engine_version,
        benchmark_name=benchmark_name,
        dataset_version=dataset_version,
        seed=seed,
        started_at=started,
        config_hash=cfg_hash,
        config_hash_version=CONFIG_HASH_VERSION,
        config=config or {},
        llm_provider=llm_provider,
        llm_model=llm_model,
        environment=environment.detect().as_dict(),
        env_vars=_filtered_env(env_var_prefixes),
        extra=extra or {},
        engine_provenance=engine_provenance or {},
        evidence=evidence or {"class": "reproducible_evidence", "publishable": True},
        reproducible=is_reproducible(dataset_version),
    )


def finalize_receipt(receipt: Receipt) -> Receipt:
    """Stamp ``finished_at`` and ``duration_seconds`` on ``receipt``."""
    finished = datetime.now(timezone.utc)
    receipt.finished_at = finished.isoformat()
    started = datetime.fromisoformat(receipt.started_at)
    receipt.duration_seconds = max(0.0, (finished - started).total_seconds())
    return receipt
