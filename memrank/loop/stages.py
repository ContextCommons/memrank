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
"""How many agent calls ``memrank run`` makes at once: one ceiling per stage.

Three pools, each a hard ceiling and none depending on another:

- ``--ingest-concurrency`` -- a case's reset and then its feed (ingesting one conversation);
- ``--answer-concurrency`` -- questions, across every fed case;
- ``--judge-concurrency`` -- judge calls, one answer graded each (:mod:`memrank.loop.grading_pool`).

``--concurrency N`` sets all three; an option for one stage overrides it for that stage.

The service decides what each pool may carry now (:mod:`memrank.service.machine`): a case's
questions only once its feed succeeded, and with ``--questions-in-order`` one question of a
case at a time. Each stage runs as many workers as its ceiling, each on its own thread, and each
worker loops on its own: ask the service for its pool's next step, carry it to the agent, post
the result. So service calls run side by side, as the agent calls do, and a stage is never
wider than its ceiling.

A worker offered nothing waits, without polling, until a step comes back that could have made
work for it, and asks again. Only an ingest step makes work for another worker -- a feed
acknowledged gives the answer pool its case's questions -- so only an ingest step wakes the
others; a worker that posts an answer asks again itself, which is all an answer can make work
for (its case's next question, in order). The run is finished when every worker has asked
since the last such step and been offered nothing: no step is held, so nothing more can come.

Nothing about the result depends on which call finished first: the service records every step
against its case and question and builds the result in plan order. The progress lines are the
one thing in completion order, because they report steps as they complete.

**Stopping.** A step that cannot reach the agent stops the run (see :mod:`memrank.loop.steps`),
and so does Ctrl-C. Either way no worker asks for another step, the steps already in flight are
waited for -- a thread cannot be killed -- but their results are not recorded, and the failure
is raised. Not recording them is deliberate: Ctrl-C at a terminal reaches the agent's own
processes too, so a step in flight may have failed *because* of the interrupt, and a failure
recorded is final. Unrecorded, the next runner to start treats them as lost (rule 4 of
:mod:`memrank.service.machine`) and restarts their cases cleanly.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from memrank.service.protocol import Op, Pool, Step, StepResult

#: Each stage's ceiling when neither its own option nor ``--concurrency`` is given.
DEFAULT_CONCURRENCY = 8


@dataclass(frozen=True)
class Ceilings:
    """At most how many of each stage run at once, and whether a case's questions go in order."""

    ingest: int = DEFAULT_CONCURRENCY
    answer: int = DEFAULT_CONCURRENCY
    judge: int = DEFAULT_CONCURRENCY
    #: One question of a case in flight at a time, in order -- for an agent that stores what
    #: it is asked. ``answer`` still caps the questions in flight across cases.
    questions_in_order: bool = False

    def __post_init__(self) -> None:
        for name in ("ingest", "answer", "judge"):
            if getattr(self, name) < 1:
                raise ValueError(f"the {name} concurrency is at least 1, not {getattr(self, name)}")

    @classmethod
    def resolve(cls, every: int | None = None, *, ingest: int | None = None,
                answer: int | None = None, judge: int | None = None,
                questions_in_order: bool = False) -> Ceilings:
        """``--concurrency`` (``every``) for each stage not given its own, else the default."""
        base = DEFAULT_CONCURRENCY if every is None else every
        return cls(ingest=base if ingest is None else ingest,
                   answer=base if answer is None else answer,
                   judge=base if judge is None else judge, questions_in_order=questions_in_order)

    def settings(self) -> dict[str, int | bool]:
        """How a session ran, as its ``sessions.jsonl`` start line records it."""
        return {"ingest_concurrency": self.ingest, "answer_concurrency": self.answer,
                "judge_concurrency": self.judge, "questions_in_order": self.questions_in_order}


@dataclass
class Stages:
    """What every worker shares: how to get a step from a pool, carry it, and report it.

    All three are called from many workers at once, and must be safe to.
    """

    next: Callable[[Pool], Step]
    carry: Callable[[Step], StepResult]
    post: Callable[[Step, StepResult], None]


@dataclass
class _Shared:
    """What the workers know of one another, guarded by ``changed``'s lock."""

    workers: int
    changed: threading.Condition = field(default_factory=threading.Condition)
    #: Moves on whenever an ingest step is posted: work another worker may now be offered.
    generation: int = 0
    #: The workers offered nothing since ``generation`` last moved on.
    idle: set[int] = field(default_factory=set)
    finished: bool = False
    stopped: bool = False

    def made_work(self) -> None:
        with self.changed:
            self.generation += 1
            self.idle.clear()
            self.changed.notify_all()

    def halt(self) -> None:
        """No worker asks for another step or posts one it carries."""
        with self.changed:
            self.stopped = True
            self.changed.notify_all()

    def over(self) -> bool:
        with self.changed:
            return self.finished or self.stopped

    def asked_at(self) -> int:
        with self.changed:
            return self.generation

    def offered_nothing(self, worker: int, asked_at: int) -> bool:
        """Wait until there may be work again; False when the run is finished or stopping.

        ``asked_at`` is the generation the worker asked at: if it has moved on since, the
        answer is already stale and the worker asks again at once.
        """
        with self.changed:
            if self.generation != asked_at:
                return not self.stopped
            self.idle.add(worker)
            if len(self.idle) == self.workers:
                self.finished = True  # every worker offered nothing, and none holds a step
                self.changed.notify_all()
                return False
            while self.generation == asked_at and not (self.finished or self.stopped):
                self.changed.wait()
            return not (self.finished or self.stopped)


def _work(stages: Stages, pool: Pool, worker: int, shared: _Shared) -> None:
    try:
        while not shared.over():
            asked_at = shared.asked_at()
            step = stages.next(pool)
            if step.op is Op.DONE:
                if not shared.offered_nothing(worker, asked_at):
                    return
                continue
            result = stages.carry(step)
            if shared.over():
                return  # the run is stopping: this step stays lost, and a resume redoes it
            stages.post(step, result)
            if pool is Pool.INGEST:
                shared.made_work()
    except BaseException:
        shared.halt()  # every other worker stops asking, and posts nothing it still carries
        raise


def drive(stages: Stages, ceilings: Ceilings) -> None:
    """Carry every step the service hands out, within ``ceilings``, until none is left.

    Ctrl-C or a failure in any worker stops every worker; the steps still in flight are waited
    for, never posted, and the first failure, in worker order, is raised.
    """
    pools = [Pool.INGEST] * ceilings.ingest + [Pool.ANSWER] * ceilings.answer
    shared = _Shared(workers=len(pools))
    with ThreadPoolExecutor(max_workers=len(pools), thread_name_prefix="memrank-stage") as run:
        futures = [run.submit(_work, stages, pool, worker, shared)
                   for worker, pool in enumerate(pools)]
        try:
            wait(futures, return_when=FIRST_EXCEPTION)
        finally:
            shared.halt()  # also on Ctrl-C; leaving the block waits for the workers
    for future in futures:
        error = future.exception()
        if error is not None:
            raise error
