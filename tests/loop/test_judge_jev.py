"""``memrank run --judge typesafe:...``: the run judges through Jev and records it truthfully."""

from __future__ import annotations

import json

import httpx
import pytest

from memrank.judging.client import build_completer
from memrank.judging.jev import QUESTION_SET, JevClient
from memrank.judging.jev.check import jev_refusal
from memrank.judging.judge import JudgeConfig
from memrank.judging.providers import resolve
from memrank.judging.providers.check import model_refusal
from memrank.loop import judge as loop_judge
from memrank.service.judgings import judging_of
from tests.service.test_scoring import recorded

JEV = "typesafe:jev-1.14.0"


def _systemone(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    answers = {key: {"type": "noul", "noul": 0.1 if key != "states_reference" else 0.9}
               for key in body["questions"]}
    return httpx.Response(200, json={"model": body["model"], "answers": answers, "usage": {}})


def test_a_jev_judged_run_records_jev_its_question_set_and_every_probability(monkeypatch,
                                                                             tmp_path):
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    keys: list[str] = []

    def client(target, api_key, *, cache_dir):
        keys.append(api_key)
        return JevClient(target, api_key, cache_dir=cache_dir,
                         transport=httpx.MockTransport(_systemone))

    monkeypatch.setattr(loop_judge, "JevClient", client)
    scored = loop_judge.judge(recorded(), "org-typesafe-key", JudgeConfig(judge=JEV))
    identity = scored.identity
    assert keys == ["org-typesafe-key"]
    assert (identity.judge, identity.judge_provider, identity.judge_model) == (
        JEV, "typesafe", "jev-1.14.0")
    assert identity.judge_settings == {"adapter": "typesafe", "question_set": QUESTION_SET,
                                       "threshold": 0.5}
    assert identity.judge_prompt_version == QUESTION_SET
    judged = [q for c in scored.cases for q in c.questions if q.status == "judged"]
    assert len(judged) == 3 and all(q.passed for q in judged)
    assert all(q.probability == pytest.approx(0.9) and q.decision["model"] == "jev-1.14.0"
               for q in judged)
    # A judging added afterwards keeps the same record (decision 0040).
    verdicts = judging_of(scored).verdicts
    assert all(v.decision is not None for v in verdicts if v.status == "judged")


def test_jev_writes_no_answers_and_judges_only_through_its_own_grader():
    with pytest.raises(ValueError, match="cannot answer"):
        JudgeConfig(reader=JEV)
    with pytest.raises(ValueError, match="writes no text"):
        build_completer(JudgeConfig(judge=JEV, cache=False))
    with pytest.raises(ValueError, match="samples must be 1"):
        loop_judge.judge(recorded(), "k", JudgeConfig(judge=JEV, samples=3, cache=False))


def _models(*replies: httpx.Response):
    sent: list[httpx.Request] = []
    queue = list(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return queue.pop(0)

    return httpx.MockTransport(handle), sent


LISTING = {"models": [{"name": "jev-latest"}, {"name": "jev-preview"}]}


def test_the_key_is_checked_free_and_a_versioned_model_with_one_tiny_request():
    transport, sent = _models(httpx.Response(200, json=LISTING),
                              httpx.Response(200, json={"model": "jev-1.14.0", "answers": {}}))
    assert model_refusal(resolve(JEV), "k", transport=transport) is None
    assert [r.url.path for r in sent] == ["/v1/models", "/v1/systemone"]
    assert json.loads(sent[1].content)["model"] == "jev-1.14.0"


@pytest.mark.parametrize(("replies", "said"), [
    ((httpx.Response(401),), "the key was not accepted (401)"),
    ((httpx.Response(200, json=LISTING),
      httpx.Response(400, json={"detail": {"message": "Unknown model: jev-9.9.9"}})),
     "it has no model 'jev-9.9.9' (Unknown model: jev-9.9.9)"),
])
def test_a_refused_key_or_unknown_model_stops_the_run_before_it_starts(replies, said):
    transport, _ = _models(*replies)
    assert jev_refusal(resolve("typesafe:jev-9.9.9"), "k", transport=transport) == said



def test_the_readout_flags_an_ability_jev_scored_experimentally():
    from memrank.loop.report import result_facts

    result = recorded()
    question = result.cases[0].questions[0].model_copy(update={
        "category": "event_ordering", "decision": {"experimental": "experimental: ..."}})
    facts = dict(result_facts(result, [question, *result.cases[1].questions]))
    assert facts["Experimental"].startswith("event ordering: scored by Jev without extracting")
    assert "Experimental" not in dict(result_facts(result, list(result.cases[1].questions)))
