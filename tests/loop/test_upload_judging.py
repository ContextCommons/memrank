"""A run graded again is added to it as one more judging, named by its judge (decision 0040)."""

from __future__ import annotations

import json

import httpx
import pytest

from memrank.loop import upload as up
from memrank.loop.upload import Account, Hosted
from memrank.service.judgings import judging_of
from memrank.service.scoring import apply_verdicts
from tests.service.test_scoring import PASS, recorded


def _graded(**judge):
    return apply_verdicts(recorded(), {"q1": PASS, "q2": PASS, "q3": PASS}, judge_samples=1,
                          **judge)


def test_the_judging_carries_every_verdict_and_its_judge_name():
    judging = judging_of(_graded(judge_model="m", judge_prompt_version="v"))
    assert judging.key() == ("m", "v") and judging.score.mean == 0.75
    assert [(v.question_id, v.status) for v in judging.verdicts] == [
        ("q1", "judged"), ("q2", "judged"), ("q3", "judged"), ("q4", "failed")]


@pytest.mark.parametrize("judge", [{"judge_model": None}, {"judge_model": "m"}],
                         ids=["no-model", "no-prompt-version"])
def test_a_result_no_judge_model_graded_is_not_a_judging(judge):
    with pytest.raises(ValueError, match="judge model and prompt version"):
        judging_of(_graded(**judge))


def test_add_judging_posts_it_to_the_run(monkeypatch):
    sent: list[tuple[str, dict]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append((f"{request.method} {request.url.path}", body))
        return httpx.Response(200, json={"judging": {"judge_model": "m"}, "added": True})

    http = httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle))
    hosted = Hosted(http=http, org="acme", account=Account("ada", "a@x", "Acme"))
    graded = _graded(judge_model="m", judge_prompt_version="v")
    echoed: list[str] = []
    stored = up.add_judging(hosted, graded, echoed.append)
    assert stored["added"] is True
    [(call, body)] = sent
    assert call == f"POST /orgs/acme/runs/{graded.run_id}/judgings"
    assert body == judging_of(graded).model_dump(mode="json")
    assert "Added the judging by m" in echoed[0]
