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
"""A reference translator for the memrank system contract (docs/system-contract.md).

This is a TEMPLATE, not a memory system. It stores documents in a dict and ranks them by word
overlap, so it can run anywhere with no dependencies and no engine -- which is what lets the
conformance suite verify the contract itself rather than somebody's backend.

To wrap a real engine, keep the five handlers and replace their bodies: `_prepare` creates or
resets your namespace, `_ingest` writes documents, `_retrieve` searches, `_cleanup` drops the
namespace, and `_describe` reports what your engine is actually configured with.

Run it:

    python examples/native-adapter/translator.py --port 8099

Standard library only, deliberately: a translator is a process memrank launches, so the fewer
things that must be installed before it starts, the fewer ways a benchmark can fail for reasons
that have nothing to do with the engine.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

CONTRACT_VERSION = "v1"
_WORD = re.compile(r"[a-z0-9]+")

#: isolation_unit -> the documents ingested under it. Replaced by your engine's own storage.
_STORE: dict[str, list[dict[str, Any]]] = {}
#: The unit memrank most recently opened with prepare().
_CURRENT: dict[str, str | None] = {"unit": None}


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _describe() -> dict[str, Any]:
    """Report identity and configuration.

    `components` is where a real translator states the LLM and embedder its engine is running
    with. memrank records what you say here and marks the run verified against the engine, so
    report the truth rather than an aspiration. An explicit `null` means "this engine has no such
    part" -- which is honest for this one: it does word overlap, with no model anywhere.
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "adapter": {"name": "reference-translator", "version": "0.1.0"},
        "engine": {"name": "reference-wordmatch", "version": "0.1.0"},
        "components": {"llm": None, "embedder": None},
        "capabilities": {"graph_snapshot": False, "context_budget": "matched"},
    }


def _prepare(payload: dict[str, Any]) -> dict[str, Any]:
    """Open a fresh isolation unit.

    Nothing ingested under a previous unit may be visible to this one. Here that is a new list;
    for a real engine it is a new namespace, collection or tenant. Get this wrong and every score
    after the first measures an engine that has already seen the answers.
    """
    unit = payload["isolation_unit"]
    _STORE[unit] = []
    _CURRENT["unit"] = unit
    return {}


def _ingest(payload: dict[str, Any]) -> dict[str, Any]:
    """Store the unit's documents.

    Turning a memrank Document into whatever shape your engine wants happens here -- rendering
    `messages` into a transcript, chunking oversized content, passing `timestamp` through so the
    engine dates memories to when they happened rather than to the benchmark run.

    No `usage` key is returned: this translator counts no tokens, and reporting zero would claim
    the engine spent nothing rather than admitting nobody measured.
    """
    started = time.perf_counter()
    unit = _CURRENT["unit"]
    if unit is None:
        raise ValueError("ingest called before prepare")
    _STORE[unit].extend(payload["documents"])
    return {"engine_ms": (time.perf_counter() - started) * 1000.0}


def _retrieve(payload: dict[str, Any]) -> dict[str, Any]:
    """Rank the unit's documents against the query, best first.

    The ordering is the measurement: recall@k reads this list in order. Scores go in each
    document's `metadata.score`. Returning fewer than `k` is fine -- never pad the list.
    """
    started = time.perf_counter()
    unit = _CURRENT["unit"]
    if unit is None:
        raise ValueError("retrieve called before prepare")
    query = _words(payload["query"])
    user_id = payload.get("user_id")
    scored = []
    for document in _STORE[unit]:
        if user_id and document.get("user_id") and document["user_id"] != user_id:
            continue
        overlap = len(query & _words(document.get("content") or ""))
        if overlap:
            scored.append((overlap, document))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = [_with_score(document, score) for score, document in scored[: payload["k"]]]
    return {
        "documents": top,
        "raw": {"candidates": len(_STORE[unit]), "matched": len(scored)},
        "engine_ms": (time.perf_counter() - started) * 1000.0,
    }


def _with_score(document: dict[str, Any], score: int) -> dict[str, Any]:
    """Copy a stored document with its relevance score attached."""
    out = dict(document)
    out["metadata"] = {**(document.get("metadata") or {}), "score": score}
    return out


def _cleanup(_: dict[str, Any]) -> dict[str, Any]:
    """Release the current unit. Safe to call twice, and safe before any prepare."""
    unit = _CURRENT["unit"]
    if unit is not None:
        _STORE.pop(unit, None)
        _CURRENT["unit"] = None
    return {}


_OPERATIONS = {"prepare": _prepare, "ingest": _ingest, "retrieve": _retrieve, "cleanup": _cleanup}


class Handler(BaseHTTPRequestHandler):
    """Routes /memrank/v1/* to the handlers above."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        """Serve describe, which memrank also polls as the readiness probe.

        Answer only once the engine behind you can really serve traffic: answering early turns a
        start-up failure into a benchmark result.
        """
        if self.path.rstrip("/") == "/memrank/v1/describe":
            self._send(200, _describe())
            return
        self._send(404, {"error": f"unknown path {self.path!r}"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        operation = self.path.rstrip("/").rsplit("/", 1)[-1]
        handler = _OPERATIONS.get(operation)
        if handler is None or not self.path.startswith("/memrank/v1/"):
            self._send(404, {"error": f"unknown operation {self.path!r}"})
            return
        try:
            payload = self._read_json()
            self._send(200, handler(payload))
        except Exception as exc:  # noqa: BLE001 - every failure must reach memrank as an error
            # Never swallow this and return an empty result: an empty document list is a valid
            # answer meaning "nothing matched", so a failure disguised as one scores like the
            # no-memory control arm and reads as a real finding.
            self._send(500, {"error": f"{operation} failed: {exc}"})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _send(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_: Any) -> None:
        """Silence per-request logging; memrank captures stdout into the run's engine log."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True, help="port to bind on 127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"reference translator listening on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
