"""The agent's reported token usage: kept per answer and per feed, added up and priced per run."""

from __future__ import annotations

import pytest

from memrank.metrics.prices import PRICE_TABLE_VERSION
from tests.service.conftest import create, post, step

HAIKU = {"provider": "anthropic", "model": "claude-haiku-4-5"}
EMBED = {"component": "embedder", "provider": "openai", "model": "text-embedding-3-small"}


def _reader(tokens: int, out: int = 10) -> dict:
    return {"component": "reader", **HAIKU, "input_tokens": tokens, "output_tokens": out}


def _run(client, feeds: dict, asks: dict) -> dict:
    """Every step answered: ``feeds[case]`` and ``asks[case]`` are the fields posted."""
    run_id = create(client)
    while (current := step(client, run_id))["op"] != "done":
        if current["op"] == "ask":
            fields = asks[current["case_id"]]
        elif current["op"] == "feed":
            fields = feeds.get(current["case_id"], {"ok": True})
        else:
            fields = {"ok": True}
        assert post(client, run_id, current["step_id"], **fields).status_code == 200
    return client.get(f"/v1/runs/{run_id}/result").json()


def test_usage_is_kept_per_answer_and_feed_and_priced_per_run(client):
    result = _run(client, {"c1": {"ok": True, "usage": [{**EMBED, "input_tokens": 500_000}]},
                           "c2": {"ok": True, "usage": []}},
                  {"c1": {"answer": "blue", "usage": [_reader(1000)]},
                   "c2": {"answer": "blue", "usage": [_reader(3000)]}})
    usage = result["usage"]
    assert usage["price_table_version"] == PRICE_TABLE_VERSION
    assert usage["context_tokens"]["mean"] == pytest.approx(5000 / 3)  # c1 has two questions
    # ask: (1000*1 + 10*5)/1e6 twice, (3000*1 + 10*5)/1e6 once; x 1e6 per million queries
    assert usage["cost_per_million_queries_usd"] == pytest.approx((1050 * 2 + 3050) / 3)
    # feed: 500k embedding tokens at $0.02/M on c1, nothing paid on c2
    assert usage["ingest_cost_per_conversation_usd"] == pytest.approx(0.01 / 2)
    embedder = next(c for c in usage["components"] if c["component"] == "embedder")
    assert (embedder["step"], embedder["output_tokens"]) == ("feed", None)
    assert result["cases"][0]["feed_usage"][0]["input_tokens"] == 500_000
    assert result["cases"][1]["questions"][0]["usage"] == [_reader(3000)]


def test_steps_that_report_nothing_are_left_out_never_zero(client):
    result = _run(client, {}, {"c1": {"answer": "blue", "usage": [_reader(2000)]},
                               "c2": {"answer": "blue"}})
    usage = result["usage"]
    assert (usage["answers"], usage["answers_reported"]) == (3, 2)
    assert usage["context_tokens"]["mean"] == 2000
    assert usage["ingest_cost_per_conversation_usd"] is None  # no feed reported
    assert result["cases"][1]["questions"][0]["usage"] is None


def test_without_a_reader_entry_there_are_no_context_tokens(client):
    result = _run(client, {}, {c: {"answer": "x", "usage": [{**EMBED, "input_tokens": 7}]}
                               for c in ("c1", "c2")})
    assert result["usage"]["context_tokens"] is None


def test_a_run_whose_agent_reports_nothing_has_no_usage(client):
    assert _run(client, {}, {"c1": {"answer": "x"}, "c2": {"error": "timeout"}})["usage"] is None


def test_an_unpriced_model_keeps_the_tokens_and_refuses_the_cost_by_name(client):
    unknown = {"component": "reader", "provider": "acme", "model": "brain-9",
               "input_tokens": 4000, "output_tokens": 3}
    result = _run(client, {}, {"c1": {"answer": "x", "usage": [unknown]},
                               "c2": {"answer": "x", "usage": [_reader(1000)]}})
    usage = result["usage"]
    assert "acme/brain-9" in usage["cost_refused"]
    assert usage["cost_per_million_queries_usd"] is None
    assert all(c["cost_usd"] is None for c in usage["components"])
    assert usage["context_tokens"]["n"] == 3
    assert result["cases"][0]["questions"][0]["usage"] == [unknown]


@pytest.mark.parametrize("fields", [{"error": "boom", "usage": []},
                                    {"answer": "x", "usage": [{"component": "reader"}]}])
def test_usage_on_a_failure_or_malformed_is_refused(client, fields):
    run_id = create(client)
    current = step(client, run_id)
    assert post(client, run_id, current["step_id"], **fields).status_code == 422
