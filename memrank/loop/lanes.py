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
"""``--concurrency N``: N lanes, each carrying one case's steps at a time, in that case's order.

The service decides which case a lane works and what its next step is
(:mod:`memrank.service.machine`); a lane only asks, carries the step to the agent, and reports.
Cases therefore run side by side while every case stays strictly reset, feed, ask, ask... One
lane is the sequential run.

Nothing about the result depends on which lane finished first: the service records every step
against its case and builds the result in plan order. The progress lines printed here are the
one thing in completion order, because they report steps as they complete.

**Stopping.** A lane that cannot reach the agent stops the run (see :mod:`memrank.loop.run`),
and so does Ctrl-C. Either way no lane asks for another step, the steps already in flight are
waited for -- a thread cannot be killed -- but their results are not recorded, and the first
failure, in lane order, is raised. Not recording them is deliberate: Ctrl-C at a terminal reaches
the agent's own processes too, so a step in flight may have failed *because* of the interrupt,
and a failure recorded is final. Unrecorded, the service treats the step as lost (rule 4 of
:mod:`memrank.service.machine`) and a resume restarts that case cleanly.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

from memrank.connect.base import AgentError
from memrank.service.protocol import Op, Step, StepResult

#: How many lanes ``memrank run`` opens when ``--concurrency`` is not given. One lane is the
#: sequential run.
DEFAULT_CONCURRENCY = 8


@dataclass
class Lanes:
    """What every lane shares: how to get a step, carry it, and report it."""

    next: Callable[[int, int], Step]
    carry: Callable[[Step, Callable[[AgentError], None]], StepResult]
    post: Callable[[Step, StepResult], None]
    failed: Callable[[AgentError], None]


def _lane(lanes: Lanes, index: int, count: int, stop: threading.Event) -> None:
    while not stop.is_set():
        step = lanes.next(index, count)
        if step.op is Op.DONE:
            return
        result = lanes.carry(step, lanes.failed)
        if stop.is_set():
            return  # the run is stopping: this step stays lost, and a resume redoes its case
        lanes.post(step, result)


def drive(lanes: Lanes, count: int) -> None:
    """Run ``count`` lanes until the service has no case left for any of them."""
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="memrank-lane") as pool:
        futures = [pool.submit(_lane, lanes, index, count, stop) for index in range(count)]
        try:
            wait(futures, return_when=FIRST_EXCEPTION)
        finally:
            # Also on Ctrl-C: no lane records another step, and leaving the block waits for them.
            stop.set()
    _raise_first(futures)


def _raise_first(futures: list[Future[None]]) -> None:
    for future in futures:
        error = future.exception()
        if error is not None:
            raise error
