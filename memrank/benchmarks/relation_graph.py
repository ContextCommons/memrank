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
"""Relation-graph benchmark.

Synthetic fixture pack for scoring memory graph behavior separately from answer
quality. The first pack mirrors the Supermemory RE fixture set:

- targeted ambiguous update;
- same-name entity collision;
- distractor derivation;
- two-chunk mixed update/extension/inference.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from memrank.benchmarks.relation_graph_fixtures import FIXTURES
from memrank.core import AdapterResponse, Benchmark, BenchmarkUnit, Document, EvalInfo
from memrank.judging.prompts import MEMRANK_READER

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()




class RelationGraphBenchmark(Benchmark):
    """Synthetic relation-graph fixture benchmark."""

    name = "relation_graph"
    dataset_version = "relation-graph-v0"
    # Slices include the fixture-addressing forms the loader honors beyond smoke/mini.
    info = EvalInfo(unit="fixture",
                    units_declared="4 synthetic relation-graph fixtures",
                    slices=("smoke", "mini", "<group>", "<fixture-id>"))
    substring_recall_supported = False
    # The graph composite IS a self-contained, rankable quality score (no judge
    # needed); substring recall is simply N/A for a relation-graph benchmark.
    composite_rankable = True
    quality_metric = "graph_score"
    is_synthetic = True
    question_text_public = True
    # No official reader prompt exists for this synthetic evaluation: memrank's own, declared.
    answer_prompt = MEMRANK_READER
    requires_graph = True

    def __init__(self, slice: str | None = None, k: int = 10) -> None:
        self.slice = slice
        self.k = k

    def load(self) -> list[BenchmarkUnit]:
        units: list[BenchmarkUnit] = []
        for fixture in self._fixtures():
            units.append(BenchmarkUnit(
                unit_id=fixture["id"],
                isolation_id=fixture["id"].replace("/", "-"),
                documents=_fixture_documents(fixture),
                queries=_fixture_queries(fixture),
                metadata={"relation_graph_fixture": fixture},
            ))
        return units

    def score(self, unit: BenchmarkUnit, responses: list[AdapterResponse]) -> dict[str, Any]:
        fixture = unit.metadata["relation_graph_fixture"]
        raw = self._raw_graph(responses)
        memories = raw.get("memories") or []
        search_results = self._search_results(responses)
        result = _score_fixture(fixture, memories, search_results)
        result["unit_id"] = unit.unit_id
        result["fixture_id"] = fixture["id"]
        result["n_queries"] = len(unit.queries)
        return {
            "composite": result["weighted_graph_score"],
            "per_category": result["scores"],
            "n_queries": len(unit.queries),
            "metric": "relation_graph_score",
            "failures": result["failures"],
            "fixture_id": fixture["id"],
        }

    def report_template(self) -> str:
        return (
            "# Relation Graph report - {adapter}\n\n"
            "- Composite graph score: **{composite}**\n"
            "- Dataset: {dataset_version}\n"
            "\n## Per-category graph scores\n{per_category_md}\n"
        )

    def _fixtures(self) -> list[dict[str, Any]]:
        """Select fixtures for the configured slice; raise on an unknown slice.

        Valid slices: None/"" (all), "smoke" (1 fixture), "mini" (2), a fixture
        group (e.g. "ambiguous-update"), or a full fixture id. An unknown slice
        fails loud rather than silently selecting nothing (which would publish a
        misleading composite=0.0 / n_units=0 result).
        """
        if not self.slice:
            return FIXTURES
        if self.slice == "smoke":
            return FIXTURES[:1]
        if self.slice == "mini":
            return FIXTURES[:2]
        matched = [f for f in FIXTURES if f["id"].split("/")[-2] == self.slice or f["id"] == self.slice]
        if not matched:
            groups = sorted({f["id"].split("/")[-2] for f in FIXTURES})
            raise ValueError(
                f"relation_graph: unknown slice {self.slice!r} "
                f"(use 'smoke', 'mini', a full fixture id, or a group: {groups})"
            )
        return matched

    @staticmethod
    def _raw_graph(responses: list[AdapterResponse]) -> dict[str, Any]:
        """Return the adapter graph snapshot, failing loud if it is missing or errored.

        relation_graph is a graph-capability benchmark: a missing snapshot or one
        carrying an adapter ``error`` is a capability FAILURE, not a valid empty
        graph that would silently score zero.
        """
        for response in responses:
            graph = (response.raw or {}).get("graph_snapshot")
            if isinstance(graph, dict):
                if graph.get("error"):
                    raise RuntimeError(f"relation_graph: adapter graph snapshot errored: {graph['error']}")
                return graph
        raise RuntimeError(
            "relation_graph: no graph_snapshot in adapter responses -- this benchmark "
            "requires a graph-capable adapter (raw['graph_snapshot'])"
        )

    @staticmethod
    def _search_results(responses: list[AdapterResponse]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for response in responses:
            for rank, doc in enumerate(response.documents, start=1):
                out.append({
                    "query_id": response.query_id,
                    "provider_memory_id": doc.id,
                    "text": doc.content,
                    "rank": rank,
                    "relations_in_context": (doc.metadata or {}).get("relation_context", []),
                })
        return out


def _fixture_documents(fixture: dict[str, Any]) -> list[Document]:
    """Build the seed + input documents for a relation-graph fixture."""
    docs: list[Document] = []
    for seed in fixture["seed_memories"]:
        docs.append(Document(
            id=seed["id"],
            content=seed["text"],
            metadata={
                "relation_graph_role": "seed",
                "fixture_id": fixture["id"],
                "fixture_seed_id": seed["id"],
            },
        ))
    for doc in fixture["input_documents"]:
        docs.append(Document(
            id=doc["id"],
            content=doc["text"],
            metadata={
                "relation_graph_role": "document",
                "fixture_id": fixture["id"],
                "fixture_document_id": doc["id"],
                "task_type": "memory",
            },
        ))
    return docs


def _fixture_queries(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    """Copy each fixture query, applying the relation-graph query defaults."""
    queries = []
    for query in fixture["queries"]:
        q = dict(query)
        q.setdefault("category", "relation_graph")
        q.setdefault("kind", "positive")
        q.setdefault("required_spans", [])
        q.setdefault("forbidden_spans", [])
        queries.append(q)
    return queries


def _fact_catalog(fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for seed in fixture.get("seed_memories", []):
        facts[seed["id"]] = {"id": seed["id"], "canonical_fact": seed["text"], "aliases": [], "type": "seed"}
    for memory in fixture.get("expected_memories", []):
        facts[memory["id"]] = {**memory, "type": "expected_memory"}
    for absence in fixture.get("expected_absences", []):
        facts[absence["id"]] = {
            "id": absence["id"],
            "canonical_fact": absence["fact"],
            "aliases": [],
            "type": "expected_absence",
        }
    return facts


def _match_text(text: str, facts: dict[str, dict[str, Any]]) -> str | None:
    """Resolve a memory text to a fact id by normalized exact-or-alias match,
    falling back to containment ONLY when token overlap is high enough to be
    unambiguous. Fixtures supply ``aliases`` for expected paraphrases; this is a
    near-verbatim matcher, not a semantic one."""
    n = _norm(text)
    if not n:
        return None
    exact = [fid for fid, f in facts.items() if any(c and n == c for c in _candidates(f))]
    if len(exact) == 1:
        return exact[0]
    subset = [fid for fid, f in facts.items() if any(_contains(n, c) for c in _candidates(f))]
    return subset[0] if len(subset) == 1 else None


def _candidates(fact: dict[str, Any]) -> list[str]:
    return [_norm(fact.get("canonical_fact", "")), *[_norm(a) for a in fact.get("aliases", [])]]


def _contains(n: str, c: str) -> bool:
    """One normalized string contains the other AND shares >=60% of the longer's tokens."""
    if not c or not (n in c or c in n):
        return False
    short, long = (n, c) if len(n) <= len(c) else (c, n)
    long_tokens = set(long.split())
    return bool(long_tokens) and len(set(short.split()) & long_tokens) / len(long_tokens) >= 0.6


def _build_mappings(fixture: dict[str, Any], memories: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
    facts = _fact_catalog(fixture)
    provider_to_fact: dict[str, str] = {}
    fact_to_providers: dict[str, set[str]] = defaultdict(set)       # ALL (incl history) -- parent resolution
    live_fact_to_providers: dict[str, set[str]] = defaultdict(set)  # LIVE only -- presence/absence
    for memory in memories:
        live = bool(memory.get("is_latest", True)) and not bool(memory.get("is_forgotten", False))
        rows = [(memory.get("provider_memory_id"), memory.get("text", ""), live)]
        rows.extend((h.get("provider_memory_id"), h.get("text", ""), False) for h in memory.get("history", []))
        for provider_id, text, is_live in rows:
            if not provider_id:
                continue
            fact_id = _match_text(text, facts)
            if fact_id:
                provider_to_fact[str(provider_id)] = fact_id
                fact_to_providers[fact_id].add(str(provider_id))
                if is_live:
                    live_fact_to_providers[fact_id].add(str(provider_id))
    return facts, provider_to_fact, fact_to_providers, live_fact_to_providers


def _relation_matches(memory: dict[str, Any], expected: dict[str, Any], provider_to_fact: dict[str, str], fact_to_providers: dict[str, set[str]]) -> bool:
    """True iff a single edge matches BOTH the expected relation type AND parent.

    The strict full-match used for inference/derives provenance, where a derivation
    edge must point at the right parent with the right relation.
    """
    acceptable = set(expected.get("acceptable_relations") or [expected["relation"]])
    parents = fact_to_providers.get(expected["parent"], set())
    for relation in memory.get("relations", []):
        if relation.get("relation") not in acceptable:
            continue
        parent = str(relation.get("parent_provider_memory_id"))
        if provider_to_fact.get(parent) == expected["parent"] or parent in parents:
            return True
    return False


def _relation_type_matches(memory: dict[str, Any], expected: dict[str, Any]) -> bool:
    """True iff any edge has the expected relation TYPE (independent of parent)."""
    acceptable = set(expected.get("acceptable_relations") or [expected["relation"]])
    return any(r.get("relation") in acceptable for r in memory.get("relations", []))


def _relation_parent_matches(memory: dict[str, Any], expected: dict[str, Any], provider_to_fact: dict[str, str], fact_to_providers: dict[str, set[str]]) -> bool:
    """True iff any edge points at the expected PARENT (independent of relation type)."""
    parents = fact_to_providers.get(expected["parent"], set())
    for relation in memory.get("relations", []):
        parent = str(relation.get("parent_provider_memory_id"))
        if provider_to_fact.get(parent) == expected["parent"] or parent in parents:
            return True
    return False


_DIM_WEIGHTS = {
    "entity_disambiguation": 0.20,
    "edge_type_correctness": 0.20,
    "parent_set_correctness": 0.20,
    "inference_provenance_completeness": 0.15,
    "retrieval_exposure": 0.15,
}


def _mark(dims: dict[str, list[int]], dim: str, passed: bool) -> None:
    dims[dim][1] += 1
    dims[dim][0] += 1 if passed else 0


def _score_inference(
    memory: dict[str, Any],
    provider_memories: list[dict[str, Any]],
    provider_to_fact: dict[str, str],
    fact_to_providers: dict[str, set[str]],
    dims: dict[str, list[int]],
    failures: list[dict[str, Any]],
) -> None:
    """Score the inference flag + derives-provenance for an expected-inference memory."""
    if not memory.get("expected_inference"):
        return
    flag_ok = any(m.get("is_inference") for m in provider_memories)
    derives = [r for r in memory.get("relation_expectations", []) if r.get("required") and r.get("relation") == "derives"]
    derives_ok = bool(derives) and any(
        all(_relation_matches(m, r, provider_to_fact, fact_to_providers) for r in derives)
        for m in provider_memories
    )
    _mark(dims, "inference_provenance_completeness", flag_ok)
    _mark(dims, "inference_provenance_completeness", derives_ok)
    if not flag_ok:
        failures.append({"label": "expected_inference_flag_missing", "fact": memory["id"]})
    if not derives_ok:
        failures.append({"label": "expected_derives_provenance_missing", "fact": memory["id"]})


def _score_expected_memories(
    memories: list[dict[str, Any]],
    fact_to_providers: dict[str, set[str]],
    live_fact_to_providers: dict[str, set[str]],
    provider_to_fact: dict[str, str],
    expected: list[dict[str, Any]],
    dims: dict[str, list[int]],
    failures: list[dict[str, Any]],
) -> None:
    """Score entity-disambiguation, edge/parent-set, and inference dims over expected memories."""
    for memory in expected:
        if memory.get("required"):
            present = bool(live_fact_to_providers.get(memory["id"]))
            _mark(dims, "entity_disambiguation", present)
            if not present:
                failures.append({"label": "missing_required_fact", "fact": memory["id"]})
        provider_memories = [
            m for m in memories if str(m.get("provider_memory_id")) in fact_to_providers.get(memory["id"], set())
        ]
        for rel in memory.get("relation_expectations", []):
            if not rel.get("required"):
                continue
            type_ok = any(_relation_type_matches(m, rel) for m in provider_memories)
            parent_ok = any(_relation_parent_matches(m, rel, provider_to_fact, fact_to_providers) for m in provider_memories)
            _mark(dims, "edge_type_correctness", type_ok)
            _mark(dims, "parent_set_correctness", parent_ok)
            if not (type_ok and parent_ok):
                failures.append({
                    "label": "missing_or_wrong_relation",
                    "fact": memory["id"],
                    "expected_parent": rel["parent"],
                    "expected_relation": rel["relation"],
                })
        _score_inference(memory, provider_memories, provider_to_fact, fact_to_providers, dims, failures)


def _score_absences(
    absences: list[dict[str, Any]],
    live_fact_to_providers: dict[str, set[str]],
    dims: dict[str, list[int]],
    failures: list[dict[str, Any]],
) -> None:
    """Score that forbidden facts are absent from the LIVE graph (history is allowed)."""
    for absence in absences:
        absent = not bool(live_fact_to_providers.get(absence["id"]))
        _mark(dims, "entity_disambiguation", absent)
        if not absent:
            failures.append({"label": "forbidden_fact_present", "fact": absence["id"]})


def _score_retrieval(
    queries: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    facts: dict[str, dict[str, Any]],
    provider_to_fact: dict[str, str],
    dims: dict[str, list[int]],
    failures: list[dict[str, Any]],
) -> None:
    """Score retrieval exposure: required facts surfaced, forbidden suppressed, context relations exposed."""
    for query in queries:
        results = [r for r in search_results if r.get("query_id") == query["id"]]
        retrieval = query.get("expected_retrieval", {})
        for fact_id in retrieval.get("required_facts", []):
            found = any(_result_fact(r, facts, provider_to_fact) == fact_id for r in results)
            _mark(dims, "retrieval_exposure", found)
            if not found:
                failures.append({"label": "required_fact_not_retrieved", "query": query["id"], "fact": fact_id})
        for fact_id in retrieval.get("forbidden_facts", []):
            found = any(_result_fact(r, facts, provider_to_fact) == fact_id for r in results)
            _mark(dims, "retrieval_exposure", not found)
            if found:
                failures.append({"label": "forbidden_fact_retrieved", "query": query["id"], "fact": fact_id})
        for rel in retrieval.get("required_context_relations", []):
            found = _context_relation_found(results, rel, facts, provider_to_fact)
            _mark(dims, "retrieval_exposure", found)
            if not found:
                failures.append({"label": "required_context_relation_not_exposed", "query": query["id"], **rel})


def _score_fixture(fixture: dict[str, Any], memories: list[dict[str, Any]], search_results: list[dict[str, Any]]) -> dict[str, Any]:
    facts, provider_to_fact, fact_to_providers, live_fact_to_providers = _build_mappings(fixture, memories)
    failures: list[dict[str, Any]] = []
    dims: dict[str, list[int]] = {
        "entity_disambiguation": [0, 0],
        "edge_type_correctness": [0, 0],
        "parent_set_correctness": [0, 0],
        "inference_provenance_completeness": [0, 0],
        "retrieval_exposure": [0, 0],
    }
    _score_expected_memories(memories, fact_to_providers, live_fact_to_providers, provider_to_fact,
                             fixture.get("expected_memories", []), dims, failures)
    _score_absences(fixture.get("expected_absences", []), live_fact_to_providers, dims, failures)
    _score_retrieval(fixture.get("queries", []), search_results, facts, provider_to_fact, dims, failures)
    # Only dimensions with at least one applicable check are scored; untested
    # dimensions are EXCLUDED (not awarded a free 1.0) and the weights renormalize
    # over the tested set, so a fixture's composite reflects only what it measured.
    scores = {name: passed / total for name, (passed, total) in dims.items() if total}
    active_weight = sum(_DIM_WEIGHTS[name] for name in scores)
    weighted = (
        sum(scores[name] * _DIM_WEIGHTS[name] for name in scores) / active_weight
        if active_weight else 0.0
    )
    return {"scores": scores, "weighted_graph_score": weighted, "failures": failures}


def _result_fact(result: dict[str, Any], facts: dict[str, dict[str, Any]], provider_to_fact: dict[str, str]) -> str | None:
    provider_id = result.get("provider_memory_id")
    if provider_id and str(provider_id) in provider_to_fact:
        return provider_to_fact[str(provider_id)]
    return _match_text(str(result.get("text", "")), facts)


def _context_relation_found(
    results: list[dict[str, Any]],
    expected: dict[str, Any],
    facts: dict[str, dict[str, Any]],
    provider_to_fact: dict[str, str],
) -> bool:
    for result in results:
        if _result_fact(result, facts, provider_to_fact) != expected["fact"]:
            continue
        for ctx in result.get("relations_in_context", []):
            ctx_fact = _result_fact(ctx, facts, provider_to_fact)
            if ctx_fact == expected["parent"] and ctx.get("relation") == expected["relation"]:
                return True
    return False
