from memrank.provenance.receipt import build_receipt
from tests import withheld


def test_config_hash_differs_for_different_models():
    base = {"adapter_name": "a", "adapter_version": "1", "engine_version": "1",
            "benchmark_name": "b", "dataset_version": "1", "seed": 42}
    r1 = build_receipt(**base, config={"model": "gpt-4o-mini", "k": 10},
                       llm_model="gpt-4o-mini")
    r2 = build_receipt(**base, config={"model": "gpt-4o", "k": 10},
                       llm_model="gpt-4o")
    assert r1.config_hash != r2.config_hash
    assert r1.llm_model == "gpt-4o-mini"


def test_config_hash_differs_for_endpoint_class():
    base = {"adapter_name": "a", "adapter_version": "1", "engine_version": "1",
            "benchmark_name": "b", "dataset_version": "1", "seed": 42}
    r1 = build_receipt(**base, config={"model": "m", "endpoint_class": "http"})
    r2 = build_receipt(**base, config={"model": "m", "endpoint_class": "in-process"})
    assert r1.config_hash != r2.config_hash


def test_config_hash_stable_for_same_config():
    base = {"adapter_name": "a", "adapter_version": "1", "engine_version": "1",
            "benchmark_name": "b", "dataset_version": "1", "seed": 42}
    cfg = {"model": "gpt-4o-mini", "k": 10, "repeats": 3}
    assert build_receipt(**base, config=cfg).config_hash == \
        build_receipt(**base, config=cfg).config_hash


def test_engine_provenance_roundtrips_and_stays_out_of_config_hash():
    # A rebuild changes the image digest but not the logical config, so config_hash
    # (the run-grouping key) must NOT move -- provenance is forensic pinning, not a group axis.
    base = {"adapter_name": "mem0", "adapter_version": "1", "engine_version": "1",
            "benchmark_name": "b", "dataset_version": "1", "seed": 42}
    cfg = {"model": "m", "k": 10}
    r1 = build_receipt(**base, config=cfg,
                       engine_provenance={"purl": "pkg:oci/mem0-server@sha256:aaaa"})
    r2 = build_receipt(**base, config=cfg,
                       engine_provenance={"purl": "pkg:oci/mem0-server@sha256:bbbb"})
    assert r1.config_hash == r2.config_hash
    assert r1.engine_provenance["purl"].endswith("aaaa")
    assert "engine_provenance" not in r1.config  # never folded into config


class _FakeAdapter:
    """The surface `_build_receipt` reads, plus the target it was built from."""

    name = "hindsight"
    version = "1"
    engine_version = "1"
    transport = "http"

    def __init__(self, target):
        self._memrank_target = target

    def effective_config(self):
        return {}


class _FakeBenchmark:
    name = "locomo"
    dataset_version = "snap-research/locomo10@v1"

    def config_for_receipt(self):
        return {}


def _receipt_for(target):
    """`target` is a catalog ref, or an already-resolved manifest a test wrote itself."""
    from memrank.runner import _build_receipt
    from memrank.targets import resolve_target

    adapter = _FakeAdapter(resolve_target(target) if isinstance(target, str) else target)
    return _build_receipt(adapter, _FakeBenchmark(), k=10, repeats=3, model="m",
                          token_budget=5000, seed=42, judge=None)


def _matched_mem0(tmp_path, monkeypatch):
    """A matched mem0 variant, written here rather than resolved from the catalog.

    `mem0:voyage` and `mem0:bge-tei` were the shipped pair until 2026-08-19, when both moved to the
    research lane. They still inherit `from: mem0` ACROSS repositories, which is exactly why this
    test matters more than it did: nothing in this repo fails if the base changes underneath them,
    so the receipt fields that tell the two modes apart are the only remaining guard.
    """
    from memrank.targets import resolve_target

    targets = tmp_path / "targets"
    targets.mkdir(exist_ok=True)
    (targets / "mem0-matched.yaml").write_text(
        "from: mem0\nname: mem0:matched\ncontext_budget: matched\ncomponents:\n"
        "  llm: {provider: anthropic, model: claude-sonnet-4-5-20250929}\n"
        "  embedder: {provider: voyage, model: voyage-4-large, dims: 1024}\n"
        "partitioning:\n", encoding="utf-8")
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))
    return resolve_target("mem0:matched")


def test_a_container_target_records_its_manifest_in_the_receipt():
    # Used to require `binding is not None`, so only source-bound targets were recorded and every
    # containerised engine hashed without its manifest (audit F16).
    #
    # Asserts on `retrieval` rather than the adapter name because to_dict emits two shapes: a
    # source-bound target nests `interface.adapter`, a container target is a flat asdict. Both put
    # `retrieval` at the top level, which is what F16 is about. The shape split is real and
    # recorded in tech-debt.md.
    recorded = _receipt_for("hindsight").config["target"]
    assert "retrieval" in recorded and "components" in recorded


def test_faithful_and_matched_variants_are_distinguishable():
    """Since 2026-08-19 the BARE `hindsight` is the vendor's configuration and the suffixed one is
    memrank's, matching how mem0 is arranged. The receipt has to tell them apart either way."""
    matched, faithful = _receipt_for("hindsight:matched"), _receipt_for("hindsight")
    assert faithful.config["target"]["retrieval"]["budget"] == "high"
    assert matched.config["target"]["retrieval"] == {}   # engine defaults, stated not absent
    assert matched.config_hash != faithful.config_hash


def test_the_mem0_vendor_target_is_distinguishable_from_its_matched_variants(tmp_path, monkeypatch):
    """For hindsight the `:amb` suffix says which mode a number came from. mem0 has no such suffix
    -- the vendor configuration IS the bare ref -- so the receipt is the only carrier, and these
    three fields are what carry it."""
    withheld.require("mem0")
    vendor = _receipt_for("mem0")
    matched = _receipt_for(_matched_mem0(tmp_path, monkeypatch))

    assert vendor.config["target"]["context_budget"] == "uncapped"
    assert vendor.config["target"]["partitioning"] == {"by": "speaker"}
    assert matched.config["target"]["context_budget"] == "matched"
    assert "partitioning" not in matched.config["target"]   # cleared, and omitted when empty
    assert vendor.config_hash != matched.config_hash


def test_a_receipt_never_carries_a_developers_checkout_path():
    assert "/Users/" not in repr(_receipt_for("hindsight").config)
