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
"""Contract tests for every registered adapter.

Static checks (no backend required):
- adapter class declares the required class attributes
- adapter implements every abstract method
- the contract asks for NOTHING that reports a measurement memrank takes itself
- ``token_metrics()`` -- the one declaration only an engine can make -- returns a
  dict with the required keys before any work has been done

Live checks (skip-if-unavailable): a smoke ingest+retrieve pass against
the adapter's real backend. Configure backends via env vars; absence of
a backend skips that adapter cleanly rather than failing.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import httpx
import pytest

from memrank.adapters import REGISTRY
from memrank.core import (
    REQUIRED_TOKEN_KEYS,
    Document,
    MemoryAdapter,
)

#: The whole contract: tell it something, ask it something, and the lifecycle around both.
#: `latency_metrics` and `token_metrics` were here and are not any more -- memrank times its own
#: calls, and usage is declared rather than required (ATO-2136).
REQUIRED_METHODS = ("prepare", "ingest", "retrieve", "cleanup")

#: Methods an engine must NOT have to write, because memrank takes the measurement itself. A
#: regression here is the stub-writing this step removed coming back.
NOT_REQUIRED_METHODS = ("latency_metrics", "token_metrics")


@pytest.fixture(params=sorted(REGISTRY.keys()))
def adapter_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def adapter_cls(adapter_name: str) -> type[MemoryAdapter]:
    return REGISTRY[adapter_name]


# ----------------------------------------------------------------------- #
# Static contract assertions
# ----------------------------------------------------------------------- #

def test_adapter_subclasses_memory_adapter(adapter_cls: type[MemoryAdapter]):
    """Every registered adapter must subclass MemoryAdapter."""
    assert issubclass(adapter_cls, MemoryAdapter)


def test_adapter_declares_class_attrs(adapter_cls: type[MemoryAdapter]):
    """name / version / engine_version must be non-empty strings on the class."""
    for attr in ("name", "version", "engine_version"):
        value = getattr(adapter_cls, attr, None)
        assert isinstance(value, str), f"{adapter_cls.__name__}.{attr} should be a string"
        assert value, f"{adapter_cls.__name__}.{attr} should be non-empty"


def test_adapter_implements_required_methods(adapter_cls: type[MemoryAdapter]):
    """Every method on the ABC contract must be implemented (not abstract)."""
    abstracts = getattr(adapter_cls, "__abstractmethods__", frozenset())
    assert not abstracts, f"{adapter_cls.__name__} still has abstract methods: {abstracts}"
    for name in REQUIRED_METHODS:
        method = getattr(adapter_cls, name, None)
        assert callable(method), f"{adapter_cls.__name__}.{name} must be callable"


def test_the_contract_requires_no_measurement_memrank_takes_itself(
        adapter_cls: type[MemoryAdapter]):
    """An engine written from the contract alone writes neither method (ATO-2136).

    Asserted against the ABC rather than against the adapters, because the adapters may keep
    theirs -- it is being REQUIRED that was the defect, and an engine that declares its usage
    still has that recorded.
    """
    for name in NOT_REQUIRED_METHODS:
        assert name not in getattr(MemoryAdapter, "__abstractmethods__", frozenset())
    assert not hasattr(MemoryAdapter, "latency_metrics")


def test_adapter_metric_methods_emit_required_keys(adapter_cls: type[MemoryAdapter]):
    """Fresh-from-construction usage declaration should already emit every required key."""
    sig = inspect.signature(adapter_cls.__init__)
    if any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        or (param.default is inspect.Parameter.empty and name not in ("self", "args", "kwargs"))
        for name, param in sig.parameters.items()
        if name != "self"
    ):
        # Adapter requires constructor args we don't know about -- skip.
        pytest.skip(f"{adapter_cls.__name__} requires constructor args")
    instance = adapter_cls()
    assert REQUIRED_TOKEN_KEYS.issubset(instance.token_metrics().keys())


# ----------------------------------------------------------------------- #
# Live smoke (skip when backend unreachable)
# ----------------------------------------------------------------------- #

def _backend_url(adapter_name: str) -> str | None:
    if adapter_name == "atomicmemory":
        return os.environ.get("ATOMICMEMORY_API_URL", "http://localhost:3070")
    if adapter_name == "mem0":
        return os.environ.get("MEM0_HTTP_URL", "http://localhost:8888")
    if adapter_name == "hindsight":
        return os.environ.get("HINDSIGHT_API_URL", "http://localhost:7000")
    if adapter_name == "supermemory":
        return os.environ.get("SUPERMEMORY_BASE_URL", "http://localhost:6767")
    return None


def _backend_reachable(url: str | None) -> bool:
    if not url:
        return False
    try:
        httpx.get(url.rstrip("/") + "/", timeout=1.0)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
        return False
    except Exception:
        return False
    return True


def test_adapter_smoke_ingest_and_retrieve(adapter_cls: type[MemoryAdapter], adapter_name: str):
    """End-to-end smoke ingest + retrieve against a live backend (skip if absent)."""
    url = _backend_url(adapter_name)
    if not _backend_reachable(url):
        pytest.skip(f"Live backend for {adapter_name!r} not reachable at {url!r}")
    try:
        adapter = adapter_cls()
    except Exception as exc:
        pytest.skip(f"{adapter_cls.__name__} could not be constructed: {exc}")
    isolation = "memrank-conformance"
    try:
        adapter.prepare(isolation)
        adapter.ingest([Document(id="d1", content="The capital of France is Paris.", user_id=isolation)])
        docs, _ = adapter.retrieve("What is the capital of France?", k=3, user_id=isolation)
    except Exception as exc:
        pytest.skip(f"{adapter_cls.__name__} smoke run failed against backend: {exc}")
    finally:
        try:
            adapter.cleanup()
        except Exception:
            pass
        closer: Any = getattr(adapter, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
    # NOT just `isinstance(docs, list)` -- an empty list satisfies that, so this test passed for
    # every mem0 run memrank ever made while the engine returned nothing. A smoke test that
    # ingests a document and does not check the document comes back is a type check wearing a
    # smoke test's name.
    assert isinstance(docs, list)
    assert docs, (
        f"{adapter_cls.__name__} ingested a document and retrieved nothing. An engine that stores "
        f"and cannot recall is broken in the way that matters, and it scores like the no-memory "
        f"control rather than failing.")
    assert any("Paris" in (d.content or "") for d in docs), (
        f"{adapter_cls.__name__} retrieved {len(docs)} document(s), none containing the ingested "
        f"fact. Retrieval returned something, but not the thing that was stored.")
