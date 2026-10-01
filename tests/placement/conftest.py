"""Machine-independent rendering for every placement test.

Every image a target runs is named in full by its own graph, so rendering consults no ambient
configuration at all. This fixture removes what a developer's machine might supply anyway: a test
that passes only because ``MEMRANK_ENGINES_*`` happens to be exported is exactly the failure it
exists to prevent -- rendering worked everywhere it was tried and failed in the one place it had to
work, the API container, which has no such variable and no AWS context file.
"""
from __future__ import annotations

import os

import pytest
import yaml

# ``TEST_REGISTRY`` used to live here and now lives in ``tests/internal/engine_images.py``. It is
# an ECR host, so it necessarily carries the account id, and THIS FILE SHIPS. The targets it
# describes do not ship either, pending ATO-1831; the three tests that key on it are classified
# internal in ``publish.toml`` alongside them.


@pytest.fixture(autouse=True)
def _no_ambient_image_configuration(monkeypatch):
    """No rendering path may consult the environment for an image address or tag."""
    for name in [k for k in os.environ if k.startswith("MEMRANK_ENGINES")]:
        monkeypatch.delenv(name)


@pytest.fixture
def declare_target(monkeypatch, tmp_path):
    """Write a manifest and its graph into an operator target directory, and resolve it.

    Adding a target is meant to be data -- a manifest plus a Compose file, no change to memrank --
    so a test that needs an unusual graph writes one rather than mutating a shipped file.
    ``catalog.graph_file`` already searches the operator directory before the package's, which this
    exercises as a side effect; nothing else covers that path.

    Function-scoped on purpose. The session's config dir is shared, and a synthetic target left in
    it would appear in ``list_targets()`` for every module that enumerates the catalog at import.
    """
    def declare(name: str, services: dict, *, adapter: str = "hindsight", service: str = "engine",
                components: dict | None = None):
        from memrank.targets import resolve_target

        targets = tmp_path / "targets"
        targets.mkdir(parents=True, exist_ok=True)
        (targets / f"{name}.compose.yaml").write_text(
            yaml.safe_dump({"services": services}), encoding="utf-8")
        # Every role the adapter exposes is declared: an unstated one is refused, because the
        # engine would fill it and no receipt would record what it chose. `components` overrides
        # the default when a test needs an embedder too -- the sidecar graphs that used to be
        # covered by `mem0:bge-tei` are written here since that target moved to the research lane.
        (targets / f"{name}.yaml").write_text(yaml.safe_dump({
            "name": name, "kind": "stack", "adapter": adapter, "transport": "http",
            "compose": f"{name}.compose.yaml", "service": service,
            "engine": {"artifact": name, "port": 8888},
            "components": components or {
                "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}},
        }), encoding="utf-8")
        monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))
        return resolve_target(name)
    return declare


@pytest.fixture
def sidecar_stack(declare_target):
    """The canonical multi-container target: an engine, a datastore, and a self-hosted embedder.

    Both renderers must translate a three-service graph -- healthcheck shapes, dependency
    conditions, per-container log prefixes, an interpolated `${MEMRANK_EMBEDDER_MODEL}`. That was
    covered by `mem0:bge-tei` until 2026-08-19, when it moved to the research lane; a shipped
    target is no longer the right carrier for a capability the renderer has regardless of what the
    catalog happens to contain.

    The values that ARE load-bearing are kept: `retries: 10` because ECS caps retries there and a
    5s/20-retry equivalent renders locally and is refused by RegisterTaskDefinition, and
    `start_period` because a model-loading sidecar needs grace rather than failures.
    """
    return declare_target("sidecar-stack", {
        "engine": {
            "image": "example.registry/engine:1",
            "ports": ["127.0.0.1::8000"],
            # A constant the image needs, not a credential: mem0 reaches TEI through the OpenAI
            # SDK, which refuses to construct without an `api_key` that TEI then ignores.
            "environment": {"POSTGRES_HOST": "datastore",
                            "MEM0_EMBEDDER_ENDPOINT": "http://embedder/v1",
                            "OPENAI_API_KEY": "unused-by-tei"},
            "depends_on": {"datastore": {"condition": "service_healthy"},
                           "embedder": {"condition": "service_healthy"}},
        },
        "datastore": {
            "image": "example.registry/pg:1",
            "healthcheck": {"test": ["CMD-SHELL", "pg_isready -q -U postgres -d postgres"],
                            "interval": "10s", "timeout": "5s", "retries": 10},
        },
        "embedder": {
            "image": "ghcr.io/huggingface/text-embeddings-inference:cpu-1.6",
            "command": ["--model-id", "${MEMRANK_EMBEDDER_MODEL}", "--auto-truncate"],
            "environment": {"NO_COLOR": "1"},
            "healthcheck": {"test": ["CMD-SHELL", "curl -fsS http://localhost/health || exit 1"],
                            "interval": "10s", "timeout": "5s", "retries": 10,
                            "start_period": "200s"},
        },
    }, adapter="mem0", components={
        "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        "embedder": {"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5", "dims": 384},
    })
