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
"""How ``memrank run`` sets up its judging: the judge it uses, and the pool that grades.

Settled before the first step, from the run's status: whether it is judged, and whether its
plan's graders call the judge model -- in which case the organisation's key must already be
there (:func:`memrank.loop.judge_key.missing_key` says how to save it).
"""

from __future__ import annotations

from pathlib import Path

from memrank.definitions import resolve as resolve_evaluation
from memrank.judging.judge import JudgeConfig
from memrank.judging.providers import resolve as resolve_judge
from memrank.loop.event_log import EventLog
from memrank.loop.explain import judge_failure
from memrank.loop.grading_pool import GradingPool
from memrank.loop.judge import DEFAULT_JUDGE, run_grading
from memrank.loop.judge_key import missing_key
from memrank.loop.live import LiveRun
from memrank.loop.output import RESULT_FILE
from memrank.loop.upload import Hosted
from memrank.service.protocol import RunResult, RunStatus


def saved_result(folder: Path) -> RunResult | None:
    """The folder's result when nothing is left to do to it: judged, or never to be judged."""
    path = folder / RESULT_FILE
    if not path.exists():
        return None
    saved = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    return saved if saved.judged or not saved.identity.judge_requested else None


def judge_config(spec: str | None, hosted: Hosted) -> JudgeConfig | None:
    """The configuration the run judges with: its spec, and the endpoint it resolved to."""
    if spec is None or hosted.judge is None:
        return None  # no grader calls the model, so no judge is configured
    endpoint = hosted.judge.endpoint
    return JudgeConfig(judge=spec, endpoints=(endpoint,) if endpoint else ())


def start_grading(status: RunStatus, hosted: Hosted, spec: str | None, log: EventLog,
          live: LiveRun, resume: str, *, workers: int) -> GradingPool | None:
    """The run's grading pool of ``workers`` judge calls at once, started; None when the run is
    not judged or already is.

    Whether its judging calls a model is the plan's to say (``status.uses_model``), so a key it
    needs and does not have is refused here, before the first step rather than after the last.
    """
    if not status.judge or saved_result(log.path.parent) is not None:
        return None
    cfg = judge_config(spec, hosted)
    if status.uses_model and (hosted.judge_key is None or cfg is None):
        raise missing_key(hosted.org, hosted.judge or resolve_judge(spec or DEFAULT_JUDGE))
    grading = run_grading(resolve_evaluation(status.evaluation_source), status.uses_model,
                          hosted.judge_key, cfg)
    target = hosted.judge

    def explain(exc: Exception) -> Exception:
        return (judge_failure(exc, target) if target else None) or exc

    pool = GradingPool(log, grading, resume=resume, workers=workers, graded=live.one_judged,
                       explain=explain)
    pool.start()
    return pool
