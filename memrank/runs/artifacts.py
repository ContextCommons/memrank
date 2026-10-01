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
"""Bring a cloud run's artifacts down into its local run directory.

The mirror image of ``deploy/upload_results.py``, which the task runs on success. Once the files
land in ``runs/<id>/``, a cloud run is indistinguishable from a local one to ``list-runs``,
``compare``, ``report`` and ``leaderboard-ingest`` -- no ``aws s3 cp`` step, no second code path.

Deliberately NOT part of :class:`~memrank.leaderboard.storage.S3Storage`. That class implements the
leaderboard's content-addressed ``runs/<sha256>.json`` contract and scopes every listing to it;
teaching it about ``cloud-runs/`` would widen a class whose narrowness is the point. This mirrors
the uploader instead, and inherits its one hard rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memrank.errors import MemrankError

# Raw cloud-run artifacts live here. NEVER "runs/": that prefix belongs to leaderboard-gc, which
# deletes any object without a matching lb.run_artifacts row, and a raw cloud upload never has one
# (deploy/upload_results.py:28-32). Stated here too because this module reads what that one wrote,
# and a disagreement between them would look like a missing result rather than a wrong prefix.
ARTIFACT_PREFIX = "cloud-runs"

#: Directory each question's detail object is written under, relative to the run's artifact
#: prefix: ``detail/<target>__<benchmark>/<query_id>.json``. Per cell, because two engines in one
#: run answer the same ``query_id`` -- a flat ``detail/q1.json`` would let whichever cell was
#: written second silently claim the other's evidence.
DETAIL_DIRNAME = "detail"

#: Where a cell's corpus is written: ``corpus/<target>__<benchmark>.json``. Beside the details
#: rather than inside them, because it is the one object every question in the cell shares -- a
#: reader fetches it once and then pays only per question.
CORPUS_DIRNAME = "corpus"

# Both live HERE, with the prefix they hang off, rather than in `memrank.analysis.question_detail`
# where they were written. They are a storage LAYOUT fact, and the layout has two readers on
# opposite sides of the public boundary: `runs/reconcile.py` skips these directories on sync (the
# CLI never wants them), and `api/results_repository.py` serves out of them. Keeping them in
# `analysis/` made a published module import an unpublished package for two strings.


class ArtifactFetchError(MemrankError):
    """A cloud run's artifacts could not be retrieved."""


def _client() -> Any:
    """An S3 client. Lazy import, no explicit session -- botocore resolves region and credentials
    from the standard chain and fails loudly, matching storage_client.py."""
    import boto3

    return boto3.client("s3")


def list_run_ids(*, bucket: str, client: Any = None) -> list[str]:
    """Every cloud run id present in ``bucket``, sorted.

    The run id is the S3 prefix, so listing with ``Delimiter="/"`` returns one ``CommonPrefixes``
    entry per run however many objects it holds -- the alternative, listing every key and deduping
    the first path segment, downloads a full key listing to learn the same thing.

    This is what makes a cloud run visible from a machine that did not submit it: the ids live in
    the bucket, not in the local registry.

    Args:
        bucket: The artifact bucket.
        client: Injected S3 client (tests supply a stub).

    Returns:
        The run ids, sorted. Empty when no cloud run has been submitted yet -- unlike
        :func:`fetch_run`, that is a legitimate state and not an error.
    """
    s3 = client or _client()
    prefix = f"{ARTIFACT_PREFIX}/"

    run_ids: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for folder in page.get("CommonPrefixes") or []:
            run_ids.append(folder["Prefix"][len(prefix):].rstrip("/"))
    return sorted(run_ids)


def fetch_run(*, bucket: str, run_id: str, dest: Path, client: Any = None) -> list[Path]:
    """Download every artifact of ``run_id`` into ``dest``.

    Args:
        bucket: The artifact bucket.
        run_id: The run id, which is also the S3 prefix -- local ``runs/<id>/`` and
            ``cloud-runs/<id>/`` in S3 name the same run deliberately.
        dest: Local run directory; created if absent.
        client: Injected S3 client (tests supply a stub).

    Returns:
        The written paths, sorted.

    Raises:
        ArtifactFetchError: When the prefix holds no objects. A successful task always uploads at
            least a summary, so an empty prefix means the upload never ran -- reporting success
            with nothing fetched would leave a run that looks recorded but has no results.
    """
    s3 = client or _client()
    prefix = f"{ARTIFACT_PREFIX}/{run_id}/"
    dest.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            relative = key[len(prefix):]
            if not relative or relative.endswith("/"):
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(target))
            written.append(target)

    if not written:
        raise ArtifactFetchError(
            f"no artifacts under s3://{bucket}/{prefix} -- the task reported success but uploaded "
            f"nothing. Check the upload step in the task's logs before trusting this run.")
    return sorted(written)


#: Where a running task publishes how far along it is. Sits beside the run's results under the
#: same prefix, so the one IAM grant the task already holds covers it and no Terraform changes.
PROGRESS_FILE = "progress.json"


def put_progress(*, bucket: str, run_id: str, record: dict[str, Any], client: Any = None) -> None:
    """Publish one progress record for a running cloud task.

    Overwrites: only the latest matters, and keeping a history would turn a progress bar into a
    write-amplified log nobody reads. Callers treat failure as non-fatal -- see
    ``run_status._publish_remote`` for why that swallow is the correct one.
    """
    (client or _client()).put_object(
        Bucket=bucket, Key=f"{ARTIFACT_PREFIX}/{run_id}/{PROGRESS_FILE}",
        Body=json.dumps(record).encode("utf-8"), ContentType="application/json")
