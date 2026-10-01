"""Bringing a cloud run's artifacts down into its local run directory.

Once these land, a cloud run must be indistinguishable from a local one to ``list-runs``,
``compare`` and ``leaderboard-ingest`` -- that is the whole point of fetching rather than leaving
them in S3 behind an `aws s3 cp` step.
"""
from __future__ import annotations

import json

import pytest

from memrank.runs import artifacts
from memrank.runs.artifacts import ArtifactFetchError, fetch_run

RUN_ID = "20260730-120000__demo__abc123"


class _FakeS3:
    """Just enough S3 to exercise the paginator + download path."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.downloaded: list[str] = []

    def get_paginator(self, _name):
        outer = self

        class _Pager:
            def paginate(self, *, Bucket, Prefix, Delimiter=None):  # noqa: N803 - boto3's casing
                if Delimiter:
                    # The listing form: S3 collapses everything below the delimiter into
                    # CommonPrefixes, so a run with ten cells still yields one entry.
                    names = sorted({k[len(Prefix):].split(Delimiter)[0]
                                    for k in outer.objects if k.startswith(Prefix)})
                    folders = [{"Prefix": f"{Prefix}{n}{Delimiter}"} for n in names]
                    yield {"CommonPrefixes": folders[:1]}
                    yield {"CommonPrefixes": folders[1:]}
                    return
                contents = [{"Key": k} for k in sorted(outer.objects) if k.startswith(Prefix)]
                # Two pages, to prove pagination is actually consumed rather than assumed single.
                yield {"Contents": contents[:1]}
                yield {"Contents": contents[1:]}
        return _Pager()

    def download_file(self, bucket, key, path):
        self.downloaded.append(key)
        with open(path, "wb") as fh:
            fh.write(self.objects[key])


def _s3(**files):
    return _FakeS3({f"cloud-runs/{RUN_ID}/{name}": body for name, body in files.items()})


def test_artifacts_land_in_the_run_dir_under_their_own_names(tmp_path):
    s3 = _s3(**{"baseline__demo.json": b'{"recall": 0.8}', "summary__demo.json": b"{}"})
    written = fetch_run(bucket="b", run_id=RUN_ID, dest=tmp_path, client=s3)

    assert {p.name for p in written} == {"baseline__demo.json", "summary__demo.json"}
    assert json.loads((tmp_path / "baseline__demo.json").read_text())["recall"] == 0.8


def test_every_page_is_consumed(tmp_path):
    s3 = _s3(**{f"cell{i}__demo.json": b"{}" for i in range(4)})
    assert len(fetch_run(bucket="b", run_id=RUN_ID, dest=tmp_path, client=s3)) == 4


def test_nested_keys_keep_their_shape(tmp_path):
    s3 = _s3(**{"raw/unit-1.json": b"{}"})
    written = fetch_run(bucket="b", run_id=RUN_ID, dest=tmp_path, client=s3)
    assert written[0] == tmp_path / "raw" / "unit-1.json"


def test_an_empty_prefix_is_an_error_not_an_empty_success(tmp_path):
    """A successful task always uploads a summary. Nothing there means the upload never ran, and
    reporting success would leave a run that looks recorded but has no results."""
    with pytest.raises(ArtifactFetchError, match="uploaded nothing"):
        fetch_run(bucket="b", run_id=RUN_ID, dest=tmp_path, client=_FakeS3({}))


def test_directory_placeholder_keys_are_skipped(tmp_path):
    s3 = _FakeS3({f"cloud-runs/{RUN_ID}/": b"", f"cloud-runs/{RUN_ID}/summary__demo.json": b"{}"})
    written = fetch_run(bucket="b", run_id=RUN_ID, dest=tmp_path, client=s3)
    assert [p.name for p in written] == ["summary__demo.json"]


def test_the_dest_directory_is_created(tmp_path):
    dest = tmp_path / "runs" / RUN_ID
    fetch_run(bucket="b", run_id=RUN_ID, dest=dest, client=_s3(**{"summary__demo.json": b"{}"}))
    assert dest.is_dir()


def test_every_cloud_run_is_listed_once_whatever_it_holds():
    """One entry per run, not per object -- a run with several cells must not list several times."""
    s3 = _FakeS3({"cloud-runs/run-a/baseline__demo.json": b"{}",
                  "cloud-runs/run-a/summary__demo.json": b"{}",
                  "cloud-runs/run-b/raw/unit-1.json": b"{}"})
    assert artifacts.list_run_ids(bucket="b", client=s3) == ["run-a", "run-b"]


def test_listing_consumes_every_page():
    s3 = _FakeS3({f"cloud-runs/run-{i}/summary__demo.json": b"{}" for i in range(4)})
    assert len(artifacts.list_run_ids(bucket="b", client=s3)) == 4


def test_no_cloud_runs_is_an_empty_list_not_an_error():
    """Unlike fetch_run, nothing to list is a legitimate state -- no cloud run has been submitted."""
    assert artifacts.list_run_ids(bucket="b", client=_FakeS3({})) == []


def test_the_gc_owned_prefix_is_never_read():
    """`runs/` in S3 belongs to leaderboard-gc; reading a run from there would mean the uploader
    had written somewhere its own guard forbids."""
    assert artifacts.ARTIFACT_PREFIX == "cloud-runs"
