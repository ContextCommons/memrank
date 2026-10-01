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
"""The evaluation service over HTTP, for a shared or hosted service (``memrank serve``).

Six routes, each one call on :class:`~memrank.service.engine.EvaluationService` -- the same
object ``memrank run`` uses in process. Handlers are synchronous on purpose: FastAPI runs them
on its thread pool, and the store's transaction makes requests for one run take turns.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from memrank.definitions.commands import CommandFailed
from memrank.definitions.problems import EvaluationInvalid
from memrank.definitions.shipped import UnknownEvaluation
from memrank.errors import MemrankError
from memrank.service.cases import EvaluationChanged, PlanError
from memrank.service.engine import EvaluationService, RunNotFinished
from memrank.service.machine import ProtocolViolation, UnknownStep
from memrank.service.protocol import (
    Ack,
    Reopened,
    RunCreate,
    RunCreated,
    RunResult,
    RunStatus,
    Step,
    StepResult,
)
from memrank.service.store import RunNotFound, RunUnreadable

#: Exception type -> HTTP status. Checked in order, so subclasses come first.
_STATUS: tuple[tuple[type[Exception], int], ...] = (
    (RunNotFound, 404), (UnknownStep, 404), (ProtocolViolation, 409), (RunNotFinished, 409),
    (PlanError, 422), (RunUnreadable, 409), (UnknownEvaluation, 422), (EvaluationInvalid, 422),
    (CommandFailed, 422), (EvaluationChanged, 409),
)


def _routes(app: FastAPI, service: EvaluationService) -> None:
    @app.post("/v1/runs")
    def create_run(request: RunCreate) -> RunCreated:
        return service.create(request)

    @app.post("/v1/runs/{run_id}/next")
    def next_step(run_id: str, lane: int = 0, lanes: int = 1) -> Step:
        return service.next(run_id, lane, lanes)

    @app.post("/v1/runs/{run_id}/steps/{step_id}")
    def post_step(run_id: str, step_id: str, result: StepResult) -> Ack:
        return service.post(run_id, step_id, result)

    @app.post("/v1/runs/{run_id}/retry-failed")
    def retry_failed(run_id: str) -> Reopened:
        return service.retry_failed(run_id)

    @app.get("/v1/runs/{run_id}/status")
    def status(run_id: str) -> RunStatus:
        return service.status(run_id)

    @app.get("/v1/runs/{run_id}/result")
    def result(run_id: str) -> RunResult:
        return service.result(run_id)


def create_app(service: EvaluationService) -> FastAPI:
    """The service as an ASGI app. Errors the rules raise become 4xx with their message."""
    app = FastAPI(title="memrank evaluation service", version="1")

    @app.exception_handler(MemrankError)
    def _refused(_: Request, exc: MemrankError) -> JSONResponse:
        status = next((code for kind, code in _STATUS if isinstance(exc, kind)), 500)
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    _routes(app, service)
    return app
