"""Nothing the retired ECS templates configured may vanish from the rendered task definition.

M6 replaced four hand-authored `deploy/ecs/taskdef.*.json.tpl` with a renderer. Four separate facts
were lost in that move -- the TEI mirror, per-engine image tags, per-engine health checks, and a
shared bearer token -- and every one was found by a question or a paid failure rather than a test.
Only the last announced itself; the rest were silent.

So this compares names, engine by engine, against the frozen originals in
`tests/fixtures/retired_taskdefs/`. **Values are deliberately not compared** -- most are derived now,
which is the entire point of M6. A variable *disappearing* is the defect this catches.

It is necessary, not sufficient: a variable set to a wrong value still passes here. Component values
are the drift gate's business (test_no_drift.py); nothing yet covers the rest.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from memrank.placement.cloud import AwsContext, render_taskdef
from memrank.targets import resolve_target
from tests import withheld

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "retired_taskdefs"

# ref -> the fixture whose shape it should still satisfy.
ENGINES = {"mem0": "mem0", "hindsight": "hindsight",
           "atomicmemory": "atomicmemory", "supermemory": "supermemory"}

# The templates named each container after what it was; the graph names it after the ROLE it
# plays, so every target has an `engine` and mem0's sidecars are `datastore` and `embedder`.
# Mapped here rather than in the renderer: the fixtures are frozen evidence of what the retired
# files configured, and translating them at the point of comparison keeps them frozen.
TEMPLATE_TO_SERVICE = {"postgres": "datastore", "tei": "embedder"}


def _service_name(container: str, ref: str) -> str:
    """The rendered container that corresponds to a template's container."""
    if container == "memrank":
        return container
    return TEMPLATE_TO_SERVICE.get(container, resolve_target(ref).service)

# Names the renderer deliberately no longer sets. EVERY entry needs its reason here -- an
# undocumented addition is exactly how the sixth loss would get through.
ALLOWED_DROPS: dict[str, str] = {
    # The engine listens on manifest `engine.port`, which the renderer writes as PORT only for
    # images that read it. mem0 does; the others take their port from their own config.
    # (mem0 still gets PORT, so this covers nobody today and is kept as documentation.)

    # Both belong to the TEI sidecar, and no shipped target runs one any more: the retired mem0
    # template froze the self-hosted-embedder stack, which became `mem0:bge-tei` and moved to the
    # research lane on 2026-08-19. The renderer has NOT lost the ability to set them -- the
    # sidecar_stack fixture in conftest.py asserts both on a three-service graph. What is gone is
    # a shipped target that needs them.
    "MEM0_EMBEDDER_ENDPOINT": "TEI sidecar only; no built-in target self-hosts an embedder",
    "NO_COLOR": "TEI sidecar only; no built-in target self-hosts an embedder",
}

#: Containers a frozen template declares that no shipped target renders any more, for the same
#: reason as the entries above. Separate from ALLOWED_DROPS because losing a whole container is a
#: bigger claim than losing a variable, and should have to be written down as one.
ALLOWED_MISSING_CONTAINERS: dict[str, str] = {
    "embedder": "the TEI sidecar left with `mem0:bge-tei` (research lane, 2026-08-19)",
}

ARNS = {"ANTHROPIC_API_KEY": "arn:a", "OPENAI_API_KEY": "arn:o", "VOYAGE_API_KEY": "arn:v"}
AWS = AwsContext(
    family="memrank-bench", region="us-east-1", log_group="/ecs/x",
    execution_role_arn="arn:e", task_role_arn="arn:t",
    memrank_image="registry.example/bench:v1", command="memrank submit ...", secret_arns=ARNS)


def _template_env(engine: str) -> dict[str, set[str]]:
    """{container name: env var names} from a frozen template, placeholders stubbed out."""
    raw = re.sub(r"__[A-Z0-9_]+__", "x", (FIXTURES / f"{engine}.json.tpl").read_text())
    doc = json.loads(raw)
    return {c["name"]: {p["name"] for p in c.get("environment", [])}
            for c in doc["containerDefinitions"]}


def _rendered_env(ref: str) -> dict[str, set[str]]:
    doc = render_taskdef(resolve_target(ref), aws=AWS)
    return {c["name"]: {p["name"] for p in c.get("environment", [])}
            for c in doc["containerDefinitions"]}


@pytest.mark.parametrize("ref,engine", ENGINES.items())
def test_no_environment_variable_was_dropped(ref, engine):
    """Per container, every name the template set must still be rendered."""
    withheld.require(ref)
    template, rendered = _template_env(engine), _rendered_env(ref)
    missing: dict[str, set[str]] = {}
    for container, names in template.items():
        gone = names - rendered.get(_service_name(container, ref), set()) - set(ALLOWED_DROPS)
        if gone:
            missing[container] = gone
    assert not missing, (
        f"{ref}: the retired template set these and the renderer does not: {missing}. "
        f"Either render them, or add each to ALLOWED_DROPS with the reason it is obsolete. "
        f"Four such losses reached production before this test existed; one cost a paid 401 and "
        f"one would have timed out the first real benchmark mid-ingest.")


@pytest.mark.parametrize("ref,engine", ENGINES.items())
def test_no_container_was_dropped(ref, engine):
    """A missing sidecar is the same class of loss, one level up."""
    withheld.require(ref)
    expected = {_service_name(c, ref) for c in _template_env(engine)}
    assert expected - set(ALLOWED_MISSING_CONTAINERS) <= set(_rendered_env(ref))


def test_every_allowed_drop_has_a_reason():
    """An entry without a documented reason is indistinguishable from an accident."""
    assert all(reason.strip() for reason in ALLOWED_DROPS.values())
    assert all(reason.strip() for reason in ALLOWED_MISSING_CONTAINERS.values())


def test_the_allowlist_names_only_things_actually_dropped():
    """A stale entry silently widens the gate for a name that has since come back.

    All four engines or none. Narrowing the population to the refs a tree happens to carry would
    make the gate quieter exactly where it is asked to be loud, so a tree missing one skips."""
    withheld.require(*ENGINES)
    still_rendered = {name for ref in ENGINES for names in _rendered_env(ref).values()
                      for name in names}
    assert not (set(ALLOWED_DROPS) & still_rendered), (
        "these are in ALLOWED_DROPS but ARE rendered; remove them so the list stays meaningful")


def test_the_fixtures_are_the_real_retired_templates():
    """Guards against the fixture being quietly edited to make the test pass."""
    mem0 = (FIXTURES / "mem0.json.tpl").read_text()
    assert "__MEM0_ENGINE_IMAGE_DIGEST__" in mem0
    assert '"family": "__FAMILY__"' in mem0
    assert len(list(FIXTURES.glob("*.json.tpl"))) == 4
