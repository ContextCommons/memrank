"""Catalog discovery, precedence, and end-to-end resolution."""
from __future__ import annotations

import pytest

from memrank.targets import catalog as c
from memrank.targets.manifest import ManifestError
from tests import withheld


def _write(tmp_path, monkeypatch, name, body):
    targets = tmp_path / "targets"
    targets.mkdir(exist_ok=True)
    (targets / f"{name}.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))


def test_builtin_targets_are_discovered():
    assert "word-overlap" in c.list_targets()
    withheld.require("mem0")
    assert "mem0" in c.list_targets()


def test_user_dir_overrides_builtin(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, "mem0",
           "name: mem0\nkind: stack\nadapter: mem0\ncomponents:\n"
           "  llm: {provider: openai, model: gpt-4o}\n")
    assert c.resolve_target("mem0").components["llm"].provider == "openai"


def test_resolve_applies_from_inheritance(tmp_path, monkeypatch):
    """A variant takes what it does not restate, and restates what it means to change.

    Synthetic rather than shipped: the matched mem0 variants that used to prove this moved to the
    research lane on 2026-08-19, and a discovery test should not depend on an operator having that
    lane configured.
    """
    withheld.require("mem0")
    _write(tmp_path, monkeypatch, "mem0-swapped",
           "from: mem0\nname: mem0:swapped\ncomponents:\n"
           "  embedder: {provider: voyage, model: voyage-4-large, dims: 1024}\n")
    got = c.resolve_target("mem0:swapped")
    assert got.adapter == "mem0"                       # inherited
    assert got.components["embedder"].model == "voyage-4-large"
    assert got.components["embedder"].dims == 1024


def test_resolve_applies_overrides():
    withheld.require("mem0")
    got = c.resolve_target("mem0", ["embedder=voyage/voyage-4-large", "embedder.dims=1024"])
    assert got.components["embedder"].dims == 1024


def test_override_without_dims_fails_loudly():
    withheld.require("mem0")
    with pytest.raises(ManifestError, match=r"components\.embedder\.dims is required"):
        c.resolve_target("mem0", ["embedder=voyage/voyage-4-large"])


def test_unknown_target_lists_known_ones():
    with pytest.raises(c.TargetNotFound, match="known:"):
        c.resolve_target("nope")


def test_inheritance_cycle_is_detected(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, "loop", "from: loop\nname: loop\nkind: stack\nadapter: mem0\n")
    with pytest.raises(ManifestError, match="inheritance cycle"):
        c.resolve_target("loop")


EXPECTED = {"word-overlap",
            "atomicmemory", "hindsight", "supermemory",
            # Vendor-configuration targets -- vendor depth, never on the leaderboard. Since
            # 2026-08-19 this is what a BARE vendor ref means, for every vendor engine: `mem0`
            # is mem0 as mem0 ships it, `hindsight` is hindsight at the depth its own AMB
            # harness publishes. memrank's
            # comparison arm carries the suffix -- `hindsight:matched` below.
            #
            # `mem0:voyage` and `mem0:bge-tei` used to sit here. They moved to the research lane on
            # 2026-08-19 because their components are OURS -- Anthropic extraction and an embedder
            # mem0 neither ships nor publishes -- so the shipped catalog now carries no matched mem0
            # at all. A matched one arriving later belongs in this set; one arriving through
            # `targets.path` does not, because this set is what the PACKAGE guarantees.
            "mem0", "hindsight:matched"}


# One case per seed rather than one loop over all of them. `mem0` and `supermemory` are not in
# every tree (`tests/withheld.py`), and a loop makes the first missing one retire the check for the
# seeds that ARE there. Parametrised, each seed answers for itself and the assertions are the same.
seed = pytest.mark.parametrize("name", sorted(EXPECTED))


@seed
def test_every_seed_ships(name):
    withheld.require(name)
    assert name in c.list_targets()


@seed
def test_every_seed_resolves_and_validates(name):
    withheld.require(name)
    assert c.resolve_target(name).name == name


@seed
def test_every_seed_names_a_registered_adapter(name):
    """A manifest pointing at a nonexistent adapter would only fail at run time."""
    from memrank.adapters import REGISTRY

    withheld.require(name)
    assert c.resolve_target(name).adapter in REGISTRY


@seed
def test_every_seed_declares_a_transport(name):
    """Declared, never inferred -- hindsight's adapter class sets none, so it read 'unknown'."""
    withheld.require(name)
    assert c.resolve_target(name).transport is not None


def test_mem0_pins_http_transport():
    """Methodology constraint: the public mem0ai SDK cannot replicate the running server."""
    withheld.require("mem0")
    assert c.resolve_target("mem0").transport == "http"


def test_baseline_is_in_process():
    got = c.resolve_target("word-overlap")
    assert got.kind == "in-process"
    assert got.depends == ()


def test_a_hosted_embedder_adds_its_own_key(tmp_path, monkeypatch):
    """Each component contributes the credential its provider needs, extraction included.

    Synthetic since `mem0:voyage` moved to the research lane; what is under test is the derivation,
    not that one shipped manifest happens to name Voyage.
    """
    withheld.require("mem0")
    _write(tmp_path, monkeypatch, "mem0-hosted",
           "from: mem0\nname: mem0:hosted\ncomponents:\n"
           "  llm: {provider: anthropic, model: claude-sonnet-4-5-20250929}\n"
           "  embedder: {provider: voyage, model: voyage-4-large, dims: 1024}\n")
    got = c.required_secrets_for(c.resolve_target("mem0:hosted"))
    assert "VOYAGE_API_KEY" in got
    assert "ANTHROPIC_API_KEY" in got


def test_a_self_hosted_embedder_needs_no_embedder_key(tmp_path, monkeypatch):
    """huggingface/TEI serves the model locally, so only extraction spends a credential."""
    withheld.require("mem0")
    _write(tmp_path, monkeypatch, "mem0-selfhosted",
           "from: mem0\nname: mem0:selfhosted\ncomponents:\n"
           "  llm: {provider: anthropic, model: claude-sonnet-4-5-20250929}\n"
           "  embedder: {provider: huggingface, model: BAAI/bge-small-en-v1.5, dims: 384}\n")
    assert c.required_secrets_for(c.resolve_target("mem0:selfhosted")) == ["ANTHROPIC_API_KEY"]


def test_baseline_requires_nothing():
    assert c.required_secrets_for(c.resolve_target("word-overlap")) == []


def test_supermemory_demands_no_credential():
    """It spends nothing: a local keyless embedder, and no LLM call on memrank's ingest path.

    It used to demand a real OPENAI_API_KEY, which made it the one target a sweep needed a second
    provider account for. The key its binary wants at BOOT is a placeholder in its compose file --
    a property of the image, not a credential this target spends. Verified by booting with an
    invalid key and watching ingest and retrieve return the right document
    (tests/live/conformance/test_supermemory_llm_unused.py).
    """
    withheld.require("supermemory")
    assert c.required_secrets_for(c.resolve_target("supermemory")) == []


def test_secret_status_reports_resolvability(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))   # empty wallet
    withheld.require("mem0")
    _write(tmp_path, monkeypatch, "mem0-twokey",
           "from: mem0\nname: mem0:twokey\ncomponents:\n"
           "  llm: {provider: anthropic, model: claude-sonnet-4-5-20250929}\n"
           "  embedder: {provider: voyage, model: voyage-4-large, dims: 1024}\n")
    status = dict(c.secret_status(c.resolve_target("mem0:twokey")))
    assert status["ANTHROPIC_API_KEY"] is True
    assert status["VOYAGE_API_KEY"] is False


# --- a base is a declaration, not a system ----------------------------------------------------- #

def test_a_base_is_marked_and_its_variants_are_not(tmp_path, monkeypatch):
    """`abstract` describes the FILE it appears in. Inheriting it would make every variant of a
    base unrunnable, which is the opposite of what a base is for.

    Written against an operator manifest because no BUILTIN target is abstract any more: `mem0` was
    the worked example until it became runnable (2026-08-18). The rule still governs every manifest
    reached through `targets.path` or `$MEMRANK_CONFIG_DIR/targets`, which share this namespace."""
    _write(tmp_path, monkeypatch, "houseblend",
           "name: houseblend\nkind: stack\nabstract: true\nadapter: mem0\ntransport: http\n"
           "components:\n  llm: {provider: anthropic, model: claude-sonnet-4-5-20250929}\n")
    (tmp_path / "targets" / "brew.yaml").write_text(
        "from: houseblend\nname: brew\ncomponents:\n"
        "  embedder: {provider: voyage, model: voyage-4-large, dims: 1024}\n", encoding="utf-8")

    assert c.resolve_target("houseblend").abstract is True
    assert c.resolve_target("brew").abstract is False


def test_a_variant_inherits_the_graph_but_pins_its_own_configuration(tmp_path, monkeypatch):
    """A matched variant takes the adapter, engine and container graph from `mem0` and nothing else.

    The LLM assertion is inverted on purpose. It used to read `variant.llm == base.llm`, which held
    while the base was an abstract declaration of memrank's held-constant extraction model. The base
    is now mem0's OWN configuration, so a variant that inherited its LLM would silently stop being
    budget-comparable. This is the regression gate for that.

    The variant is written here rather than resolved from the catalog because the two that used to
    prove it -- `mem0:voyage` and `mem0:bge-tei` -- moved to the research lane on 2026-08-19. What
    the gate protects is unchanged and now matters MORE: those manifests live in another repo, so
    nothing in this one fails when the base changes underneath them."""
    withheld.require("mem0")
    _write(tmp_path, monkeypatch, "mem0-matched",
           "from: mem0\nname: mem0:matched\ncontext_budget: matched\ncomponents:\n"
           "  llm: {provider: anthropic, model: claude-sonnet-4-5-20250929}\n"
           "  embedder: {provider: voyage, model: voyage-4-large, dims: 1024}\n"
           "partitioning:\n")
    base, variant = c.resolve_target("mem0"), c.resolve_target("mem0:matched")

    assert variant.adapter == base.adapter
    assert variant.compose == base.compose
    assert variant.engine.port == base.engine.port

    assert base.components["llm"].model == "gpt-4o-mini"
    assert variant.components["llm"].model == "claude-sonnet-4-5-20250929"
    assert base.context_budget == "uncapped" and variant.context_budget == "matched"
    assert base.partitioning == {"by": "speaker"} and variant.partitioning == {}


def test_exactly_the_manifests_that_leave_a_knob_blank_are_marked_abstract():
    """The rule, enumerated from the catalog: a target either declares every component its adapter
    exposes, or says it is a base for variants to fill. A new target failing here is the point."""
    from memrank.targets.engine_env import unstated_components

    for ref in c.list_targets():
        target = c.resolve_target(ref)
        assert bool(unstated_components(target)) == target.abstract, (
            f"{ref}: unstated={unstated_components(target)} abstract={target.abstract} -- declare "
            f"the component, or mark the manifest abstract")


def test_submitting_a_base_refuses_before_anything_is_launched(tmp_path, monkeypatch):
    """`memrank submit mem0` once produced a Fargate task that died on a missing OPENAI_API_KEY.

    Refused here, not in a task: no image is pulled and nothing is paid for, and the message names
    the variants to run instead. `mem0` is concrete now (it declares mem0's own embedder rather
    than inheriting one silently), so the refusal is exercised against an operator base -- the shape
    it still applies to."""
    from typer.testing import CliRunner

    from memrank.runner import app

    _write(tmp_path, monkeypatch, "houseblend",
           "name: houseblend\nkind: stack\nabstract: true\nadapter: mem0\ntransport: http\n"
           "components:\n  llm: {provider: anthropic, model: claude-sonnet-4-5-20250929}\n")
    (tmp_path / "targets" / "espresso.yaml").write_text(
        "from: houseblend\nname: houseblend:espresso\ncomponents:\n"
        "  embedder: {provider: voyage, model: voyage-4-large, dims: 1024}\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["submit", "houseblend", "demo", "--on", "local"])

    assert result.exit_code == 1
    assert "houseblend:espresso" in result.output      # named by the `base:variant` convention
    assert "base" in result.output
