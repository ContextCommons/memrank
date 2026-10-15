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
"""Judging a finished run in the runner's own process, with the organisation's key.

Decision 0037: ``memrank run`` judges the way ``submit`` does -- the same provider-backed completer
(cached, counted, rate-limit gated), keyed by the organisation's saved key for the judge it names
(``ANTHROPIC_API_KEY`` for ``anthropic:``, an endpoint's ``key_ref`` for ``<endpoint>:``) -- and
the verdicts then travel in the run's own record. Each answer is graded by its evaluation's
own grader (:meth:`~memrank.definitions.base.Definition.grader_for`): a shipped benchmark's
judge shape, or an evaluation file's built-in or command grader. Only a grader that calls the
judge model needs the key, so an evaluation graded without one runs without one. The judge
cache makes judging again after an interruption free for every answer already graded.

``typesafe:`` (TypeSafe's Jev, ``TYPESAFE_API_KEY``) decides typed questions rather than
completing text, so its graders go through :class:`~memrank.judging.jev.JevJudge` and name its
question set where a text judge names its prompts' version.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from memrank.definitions import resolve
from memrank.definitions.base import Definition, ModelJudge, Verdict
from memrank.evaluation.constants import DEFAULT_JUDGE_WORKERS
from memrank.judging.client import build_completer, cache_dir
from memrank.judging.jev import QUESTION_SET, JevClient, JevJudge
from memrank.judging.judge import JUDGE, Completer, JudgeConfig
from memrank.judging.prompts import JUDGE_PROMPT_VERSION
from memrank.judging.providers import ResolvedJudge
from memrank.judging.providers import resolve as resolve_judge
from memrank.judging.record import role_record
from memrank.service.protocol import JudgeIdentity, QuestionOutcome, RunResult
from memrank.service.scoring import apply_verdicts

#: The judge ``memrank run`` uses unless told otherwise: Haiku, the cheap one (ATO-2343). Every
#: result names the judge, so a score is never read without knowing whose it is.
DEFAULT_JUDGE = "anthropic:claude-haiku-4-5"


def definition_of(result: RunResult) -> Definition:
    """The evaluation a result was recorded on, found again by its source."""
    return resolve(result.identity.evaluation_source or result.identity.evaluation)


def needs_key(result: RunResult) -> bool:
    """Whether judging ``result`` calls the judge model, and so needs the organisation's key."""
    definition = definition_of(result)
    return any(definition.grader_for(q).uses_model for case in result.cases
               for q in case.questions if q.status == "not_judged")


def _model(cfg: JudgeConfig, api_key: str | None) -> ModelJudge:
    keys: dict[str, str] = {}
    if cfg.completer is None:
        if api_key is None:
            raise ValueError("a grader needs the judge model and no key was given")
        # The key is the one the judge names: its provider's, or its endpoint's key_ref.
        target = resolve_judge(cfg.judge, cfg.endpoints)
        if target.decides:
            return _decider(cfg, target, api_key)
        keys[target.key_name] = api_key
    complete, _ = build_completer(cfg, keys=keys)
    return ModelJudge(complete=complete, cfg=cfg)


def _decider(cfg: JudgeConfig, target: ResolvedJudge, api_key: str) -> ModelJudge:
    """A judge that decides typed questions (Jev): one request per answer, never a completion."""
    if cfg.samples != 1:
        raise ValueError(f"{cfg.judge} answers one request per verdict; samples must be 1, "
                         f"got {cfg.samples}")
    client = JevClient(target, api_key, cache_dir=cache_dir() if cfg.cache else None)
    return ModelJudge(complete=_writes_no_text(cfg.judge), cfg=cfg, decider=JevJudge(client))


def _writes_no_text(spec: str) -> Completer:
    def complete(model: str, system: str, user: str) -> str:
        raise ValueError(f"{spec} decides typed questions and writes no text")

    return complete


def prompt_version(model: ModelJudge) -> str:
    """What names the judge's questions: the text judge's prompts, or a decider's question set."""
    return QUESTION_SET if model.decider is not None else JUDGE_PROMPT_VERSION


def _identity(model: ModelJudge) -> JudgeIdentity:
    """The judge ``model`` is, as the record names it: the spec, what it resolved to, and the
    settings sent (:func:`memrank.judging.record.role_record`)."""
    record = role_record(model.cfg, JUDGE)
    return JudgeIdentity(judge=record["spec"], judge_provider=record["provider"],
                         judge_model=record["model"], judge_endpoint=record["endpoint"],
                         judge_settings=record["settings"])


def judge(result: RunResult, api_key: str | None, cfg: JudgeConfig | None = None,
          workers: int = DEFAULT_JUDGE_WORKERS,
          judged: Callable[[], None] = lambda: None) -> RunResult:
    """``result`` with every recorded answer graded and its scores computed.

    Args:
        result: The run as recorded, every answer ``not_judged``.
        api_key: The organisation's key for the judge ``cfg`` names; None when no grader calls
            the model, or a ``completer`` judges.
        cfg: The judge configuration, :data:`DEFAULT_JUDGE` when none is given; a
            ``completer`` on it (with its ``completer_name``) replaces the providers.
        workers: How many answers are judged at once -- the old path's judge-worker default.
        judged: Called once per answer as its verdict lands, from whichever worker judged it.
    """
    cfg = cfg or JudgeConfig(judge=DEFAULT_JUDGE)
    definition = definition_of(result)
    answered = [q for case in result.cases for q in case.questions if q.status == "not_judged"]
    graders = [definition.grader_for(q) for q in answered]
    uses_model = any(grader.uses_model for grader in graders)
    model = _model(cfg, api_key) if uses_model else None

    def one(pair: tuple[QuestionOutcome, int]) -> Verdict:
        question, index = pair
        verdict = graders[index].grade(question, model)
        judged()
        return verdict

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        # in question order, however they finish
        verdicts = list(pool.map(one, [(q, i) for i, q in enumerate(answered)]))
    by_question = {q.question_id: v for q, v in zip(answered, verdicts, strict=True)}
    return apply_verdicts(result, by_question,
                          judge=_identity(model) if model else None,
                          judge_samples=model.cfg.samples if model else 0,
                          judge_prompt_version=prompt_version(model) if model else None)
