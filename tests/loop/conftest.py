"""The runner tests drive the same fake-planned service the service tests do, and a fake org.

The org is the runs API as the runner sees it: ``PUT /orgs/acme/runs/{id}`` (register, stop,
upload), ``PUT /orgs/acme/runs/{id}/answers`` (every question, before the summary record that
points at them) and ``POST /runs/{id}/progress`` (the run-token channel a running run is handed).
"""

from __future__ import annotations

import json

import httpx

from memrank.loop.upload import Account, Hosted
from memrank.service.protocol import RunResult
from tests.service.conftest import client, engine, judge  # noqa: F401 - pytest fixtures

PAGE = "https://memrank.test/acme/runs/{run_id}"
#: The signed-in account, as ``/whoami`` answers it and as the run's opening lines name it.
WHOAMI = {"login": "ada", "email": "ada@example.test",
          "orgs": [{"slug": "acme", "name": "Acme Labs", "role": "owner"}]}
ACCOUNT = Account(login="ada", email="ada@example.test", org_name="Acme Labs")
CHANNEL = "http://api/runs/{run_id}/progress"


class OrgApi:
    """Every sync and progress sample the runner sent, in order; ``down`` refuses progress."""

    def __init__(self) -> None:
        self.syncs: list[dict] = []
        self.samples: list[dict] = []
        #: Each run's uploaded answers, by run id.
        self.answers: dict[str, dict] = {}
        self.progress_down = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content)
        run_id = path.split("/")[-2 if path.endswith(("/progress", "/answers")) else -1]
        if path.endswith("/answers"):
            self.answers[run_id] = body
            return httpx.Response(200, json={"answers": "answers.json"})
        if path.endswith("/progress"):
            if self.progress_down:
                return httpx.Response(503, text="no bus")
            assert request.headers["authorization"] == "Bearer mrr_token"
            self.samples.append(body)
            return httpx.Response(200, json={"state": "running", "interval_s": 10.0})
        self.syncs.append(body)
        running = body["state"] == "running"
        channel = {"url": CHANNEL.format(run_id=run_id), "token": "mrr_token"}
        return httpx.Response(200, json={"url": PAGE.format(run_id=run_id),
                                         "progress": channel if running else None})

    def hosted(self, judge_key: str | None = "org-key") -> Hosted:
        http = httpx.Client(base_url="http://api", transport=httpx.MockTransport(self.handle))
        return Hosted(http=http, org="acme", account=ACCOUNT, judge_key=judge_key)

    @property
    def states(self) -> list[str]:
        return [sync["state"] for sync in self.syncs]

    @property
    def uploads(self) -> list[RunResult]:
        """Each finished run as uploaded: its summary record joined with its answers."""
        return [self._joined(sync["record"]) for sync in self.syncs if sync["state"] == "done"]

    def _joined(self, record: dict) -> RunResult:
        summary = RunResult.model_validate(record["result"])
        assert record["answers"] == "answers.json"
        assert all(not case.questions for case in summary.cases)
        cases = self.answers[summary.run_id]["cases"]
        return summary.model_copy(update={
            "cases": [type(c).model_validate(stored) for c, stored in
                      zip(summary.cases, cases, strict=True)]})


def run_id_of(echoed: list[str]) -> str:
    """The id the run's opening line names: ``memrank: Run <id> . <evaluation> ...``."""
    return next(line.split()[2] for line in echoed if line.startswith("memrank: Run "))
