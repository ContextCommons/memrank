"""A large artifact is downloaded from storage, not proxied through the API.

Reading a 312 MB cell into the API and pushing it back through the load balancer failed in
practice: the client received 216 MB of 327 MB and the connection closed, while the service
logged 200. Anything big is now redirected to a presigned URL, which takes both the API process
and the balancer out of the data path and gives the client Range-based resume.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from memrank.placement import run_api_client

_BIG = b"x" * 5000


class _Stream:
    """Context-manager stand-in for httpx's streaming response."""

    def __init__(self, status: int, *, headers: dict | None = None, body: bytes = b"",
                 chunks: list[bytes] | None = None, fail_after: int | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self._chunks = chunks if chunks is not None else [body]
        self._fail_after = fail_after

    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def read(self): return self._body

    def iter_bytes(self):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise httpx.ReadError("peer closed connection")
            yield chunk


class _ApiStub:
    """The memrank API: answers the artifact route with whatever it was given."""

    def __init__(self, response: _Stream) -> None:
        self._response = response

    def stream(self, method: str, url: str, **kw):
        return self._response


def _direct_client(monkeypatch, responses: list[_Stream]) -> list[dict]:
    """Replace the bare client used for storage; record what it was asked for."""
    seen: list[dict] = []

    class _Direct:
        def __init__(self, **kw): self.kw = kw
        def __enter__(self): return self
        def __exit__(self, *exc): return False

        def stream(self, method, url, headers=None, **kw):
            seen.append({"url": url, "headers": headers or {}})
            return responses.pop(0)

    monkeypatch.setattr(httpx, "Client", _Direct)
    return seen


def test_a_small_artifact_still_comes_through_the_api(tmp_path: Path):
    """One request, no expiring URL, for a file that was never going to fail."""
    api = _ApiStub(_Stream(200, chunks=[b"small"]))
    dest = tmp_path / "summary.json"
    run_api_client.download_artifact(api, "org", "run", "summary.json", dest)
    assert dest.read_bytes() == b"small"


def test_a_redirect_is_followed_to_storage(tmp_path: Path, monkeypatch):
    api = _ApiStub(_Stream(307, headers={"location": "https://s3.example/presigned?sig=abc"}))
    seen = _direct_client(monkeypatch, [_Stream(200, chunks=[_BIG])])
    dest = tmp_path / "cell.json"

    run_api_client.download_artifact(api, "org", "run", "cell.json", dest)

    assert dest.read_bytes() == _BIG
    assert seen[0]["url"] == "https://s3.example/presigned?sig=abc"


def test_no_authorization_header_reaches_storage(tmp_path: Path, monkeypatch):
    """A presigned URL carries its own auth; adding ours is "only one auth mechanism allowed"."""
    api = _ApiStub(_Stream(307, headers={"location": "https://s3.example/presigned"}))
    seen = _direct_client(monkeypatch, [_Stream(200, chunks=[_BIG])])

    run_api_client.download_artifact(api, "org", "run", "cell.json", tmp_path / "cell.json")

    assert "Authorization" not in seen[0]["headers"]
    assert "authorization" not in {k.lower() for k in seen[0]["headers"]}


def test_a_dropped_transfer_resumes_instead_of_restarting(tmp_path: Path, monkeypatch):
    """The failure this exists for: a single mid-transfer close two-thirds through."""
    api = _ApiStub(_Stream(307, headers={"location": "https://s3.example/presigned"}))
    seen = _direct_client(monkeypatch, [
        _Stream(200, chunks=[b"aaaa", b"bbbb"], fail_after=1),   # dies after the first chunk
        _Stream(206, chunks=[b"bbbb"]),                          # resumes with the remainder
    ])
    dest = tmp_path / "cell.json"

    run_api_client.download_artifact(api, "org", "run", "cell.json", dest)

    assert dest.read_bytes() == b"aaaabbbb"
    assert seen[1]["headers"]["Range"] == "bytes=4-"    # asked to continue, not start over


def test_a_range_the_server_ignores_restarts_the_file(tmp_path: Path, monkeypatch):
    """200 to a ranged request means the whole body again; appending would corrupt it."""
    api = _ApiStub(_Stream(307, headers={"location": "https://s3.example/presigned"}))
    _direct_client(monkeypatch, [
        _Stream(200, chunks=[b"aaaa", b"bbbb"], fail_after=1),
        _Stream(200, chunks=[b"aaaabbbb"]),            # ignored the Range, sent everything
    ])
    dest = tmp_path / "cell.json"

    run_api_client.download_artifact(api, "org", "run", "cell.json", dest)

    assert dest.read_bytes() == b"aaaabbbb"            # not "aaaaaaaabbbb"


def test_a_refusal_from_storage_is_not_silent(tmp_path: Path, monkeypatch):
    api = _ApiStub(_Stream(307, headers={"location": "https://s3.example/expired"}))
    _direct_client(monkeypatch, [_Stream(403, body=b"expired")])

    with pytest.raises(run_api_client.RunApiError, match="storage refused"):
        run_api_client.download_artifact(api, "org", "run", "cell.json", tmp_path / "cell.json")
