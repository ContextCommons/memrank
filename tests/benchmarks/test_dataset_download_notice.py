"""The dataset-download notice: a log record in the library, a stderr line in the CLI.

Cache-miss only, silent on hit -- and spoken as ``logging`` INFO rather than printed, so a
script embedding memrank hears it through ordinary logging config (or not at all) while the
CLI's `term.style.install_log_bridge` renders it to stderr exactly as before.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pytest

from memrank.benchmarks import dataset_download_notice
from memrank.benchmarks.locomo import LoCoMoBenchmark
from memrank.term import style


@pytest.fixture
def unbridged():
    """Strip any bridge a CLI-invoking test left on the process-wide logger, and restore."""
    logger = logging.getLogger("memrank")
    before = list(logger.handlers)
    logger.handlers = [h for h in before if not isinstance(h, style._NarrationHandler)]
    try:
        yield logger
    finally:
        logger.handlers = before


def test_notice_brackets_the_block(caplog):
    with caplog.at_level(logging.INFO, logger="memrank.benchmarks"):
        with dataset_download_notice("Demo", "https://example.test/x.json", Path("/tmp/x.json")):
            pass
    messages = [record.getMessage() for record in caplog.records]
    assert "downloading Demo dataset from https://example.test/x.json" in messages
    assert any("(one-time, cached afterwards)" in m for m in messages)
    assert "Demo dataset ready" in messages


def test_a_library_caller_hears_nothing_on_a_terminal(capsys, unbridged):
    """The library contract: INFO narration reaches no stream unless someone wired one."""
    with dataset_download_notice("Demo", "https://example.test/x.json", Path("/tmp/x.json")):
        pass
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_the_cli_bridge_renders_the_notice_to_stderr(capsys, unbridged):
    """The CLI contract: `_root` installs the bridge, and the lines read as they always have."""
    style.install_log_bridge()
    with dataset_download_notice("Demo", "https://example.test/x.json", Path("/tmp/x.json")):
        pass
    err = capsys.readouterr().err
    assert "downloading Demo dataset from https://example.test/x.json" in err
    assert "(one-time, cached afterwards)" in err
    assert "Demo dataset ready" in err


def test_the_bridge_is_idempotent(unbridged):
    """Commands re-enter the root callback; a second install must not double every line."""
    before = len(unbridged.handlers)
    style.install_log_bridge()
    style.install_log_bridge()
    assert len(unbridged.handlers) - before == 1


def test_locomo_cache_miss_announces_download(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    calls: list[tuple[str, Path]] = []

    def fake_retrieve(url: str, dest: Path) -> None:
        calls.append((url, dest))
        Path(dest).write_text("[]", encoding="utf-8")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    # The faked download cannot match the pinned sha256; the digest path has its own tests
    # (test_locomo_methodology.py) and is not what this notice test measures.
    monkeypatch.setattr(LoCoMoBenchmark, "_verify_digest", classmethod(lambda cls, path: None))
    with caplog.at_level(logging.INFO, logger="memrank.benchmarks"):
        path = LoCoMoBenchmark()._data_path()
    messages = [record.getMessage() for record in caplog.records]
    assert calls and path.exists()
    assert any("downloading LoCoMo dataset" in m for m in messages)
    assert any("LoCoMo dataset ready" in m for m in messages)


def test_locomo_cache_hit_is_silent(tmp_path, monkeypatch, caplog, capsys):
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    cached = tmp_path / "locomo" / "locomo10.json"
    cached.parent.mkdir(parents=True)
    cached.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda *a: (_ for _ in ()).throw(
        AssertionError("must not download on cache hit")))
    # Same isolation as the cache-miss test: the digest pin is covered elsewhere.
    monkeypatch.setattr(LoCoMoBenchmark, "_verify_digest", classmethod(lambda cls, path: None))
    with caplog.at_level(logging.INFO, logger="memrank.benchmarks"):
        assert LoCoMoBenchmark()._data_path() == cached
    assert caplog.records == []
    assert capsys.readouterr().err == ""
