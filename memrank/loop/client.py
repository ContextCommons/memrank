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
"""The runner's client for the evaluation service.

Only a connection that was never made is retried. Every other call either succeeded or was
refused with a reason the service states; that reason is raised verbatim.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx

from memrank.errors import MemrankError
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

T = TypeVar("T")

#: Attempts at a call that never connected, and the pause before each retry.
ATTEMPTS = 4
BACKOFF_S = (1.0, 2.0, 4.0)
#: Steps are quick; a result may judge every answer before it returns.
STEP_TIMEOUT_S = 60.0
RESULT_TIMEOUT_S = 3600.0


class ServiceError(MemrankError):
    """The service refused a call, or could not be reached."""


class Unreached(MemrankError):
    """The call was never delivered, so repeating it is safe."""


def retry_unreached(call: Callable[[], T], *, unreached: type[Exception],
                    sleep: Callable[[float], None] = time.sleep) -> T:
    """``call()``, repeated only while it fails with ``unreached``; the last failure propagates."""
    for pause in BACKOFF_S[:ATTEMPTS - 1]:
        try:
            return call()
        except unreached:
            sleep(pause)
    return call()


class ServiceClient:
    """The protocol calls, as methods."""

    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=STEP_TIMEOUT_S,
                                  transport=transport)
        self._sleep = sleep

    def close(self) -> None:
        self._http.close()

    def _call(self, method: str, path: str, body: Any = None,
              timeout: float = STEP_TIMEOUT_S) -> Any:
        def once() -> httpx.Response:
            try:
                return self._http.request(method, path, json=body, timeout=timeout)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                raise Unreached(str(exc)) from exc

        try:
            response = retry_unreached(once, unreached=Unreached, sleep=self._sleep)
        except Unreached as exc:
            raise ServiceError(f"cannot reach the evaluation service at {self.base_url}: {exc}. "
                               "Start it with `memrank serve` or pass --service.") from exc
        if response.is_error:
            raise ServiceError(f"service refused {method} {path} ({response.status_code}): "
                               f"{_detail(response)}")
        return response.json()

    def create(self, request: RunCreate) -> RunCreated:
        return RunCreated.model_validate(
            self._call("POST", "/v1/runs", request.model_dump(mode="json")))

    def next(self, run_id: str, lane: int = 0, lanes: int = 1) -> Step:
        return Step.model_validate(
            self._call("POST", f"/v1/runs/{run_id}/next?lane={lane}&lanes={lanes}"))

    def post(self, run_id: str, step_id: str, result: StepResult) -> Ack:
        return Ack.model_validate(self._call("POST", f"/v1/runs/{run_id}/steps/{step_id}",
                                             result.model_dump(mode="json")))

    def retry_failed(self, run_id: str) -> Reopened:
        return Reopened.model_validate(self._call("POST", f"/v1/runs/{run_id}/retry-failed"))

    def status(self, run_id: str) -> RunStatus:
        return RunStatus.model_validate(self._call("GET", f"/v1/runs/{run_id}/status"))

    def result(self, run_id: str) -> RunResult:
        return RunResult.model_validate(
            self._call("GET", f"/v1/runs/{run_id}/result", timeout=RESULT_TIMEOUT_S))


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text
