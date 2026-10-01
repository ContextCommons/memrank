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
"""Pure, injectable LLM-judge scoring.

A ``Completer`` is ``(model, system, user) -> text``; all functions take one so
the judge is fully testable without the network. The Anthropic-backed completer
(with caching + a hard call cap) lives in ``judge_client``. Verdicts are strict JSON, bare or
inside a single markdown code fence; anything else raises (no silent fallback).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memrank.errors import MemrankError
from memrank.judging import prompts as jp

if TYPE_CHECKING:  # `judge_shape` imports this module; the annotation must not close the loop.
    from memrank.judging.shape import JudgeShape

Completer = Callable[[str, str, str], str]

# The global JUDGE_VALID_CATEGORIES allowlist that used to sit here is gone (2026-08-13).
# Judgeability is declared by each benchmark's `judge_shape()` -- `BinaryJudgeShape(categories=...)`
# in memrank/judging/shape.py -- so a loader's category vocabulary and its judge gate are one
# object, and an unknown category raises instead of silently shrinking the denominator. The
# allowlist was keyed on bare strings across three benchmarks' vocabularies at once, which is
# how LoCoMo judged 39.2% of itself under mislabeled categories without any signal.


def gold_answer(query: dict[str, Any]) -> str:
    """The gold answer a query is graded against, or ``""`` when it has none."""
    return (query.get("gold_answers") or [""])[0]


@dataclass
class JudgeVerdict:
    passed: bool
    rationale: str
    samples: int = 1


@dataclass
class JudgedQuery:
    generated_answer: str
    sufficiency: JudgeVerdict | None
    correctness: JudgeVerdict
    answered_without_context: bool
    #: What this query scored, in [0, 1]. For a binary shape it is `correctness.passed` as 1.0 or
    #: 0.0, so averaging it is arithmetically the same as counting Trues. It exists as a float
    #: because BEAM's metric is a mean over rubric nuggets and lands here without the runner
    #: needing to know which shape produced it.
    score: float = 0.0
    #: Per-criterion detail when a rubric shape produced this, else None. Carried so a human can
    #: check WHICH criteria a response missed -- an aggregate nobody can audit is how an
    #: unvalidated judge becomes a confident wrong number. Reaches the raw per-cell artifact only;
    #: the published allowlists drop it, which is correct since rationales quote the response.
    nuggets: list[NuggetScore] | None = None
    #: Component scores and the event alignment, when a sequence shape produced this. The product
    #: alone is unauditable -- `0.31` says nothing about whether the model missed events or listed
    #: them out of order, and those are different failures with different fixes.
    ordering: dict[str, Any] | None = None


@dataclass
class JudgeConfig:
    answer_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-opus-4-8"
    samples: int = 1
    token_budget: int = 5000
    no_context_control: bool = True
    cache: bool = True
    allow_empty_coverage: bool = False
    # Sampling temperature for ANSWER-model calls (the reader), pinned at 0 for
    # reproducibility -- the protocol BEAM itself pins and the piece we can pin: the judge
    # model (claude-opus-4-8) rejects the parameter, so judge calls stay provider-default
    # and the receipt records both truthfully.
    answer_temperature: float = 0.0
    completer: Completer | None = None
    # The reader prompt answers are generated with. `run_cell` sets it from the benchmark it runs
    # (Benchmark.answer_prompt), and the receipt records it; memrank's own reader is only the
    # default for a configuration built without a benchmark.
    answer_prompt: jp.AnswerPrompt = jp.MEMRANK_READER

    def __post_init__(self) -> None:
        # Methodology-critical knobs are validated at construction (no silent
        # max(1, n) coercion that records an invalid config but runs a valid one).
        for name in ("samples", "token_budget"):
            if getattr(self, name) < 1:
                raise ValueError(f"JudgeConfig.{name} must be >= 1, got {getattr(self, name)}")


#: The one delimiter a judge reply may carry around its JSON object. Whether a JSON answer comes
#: back bare or inside a markdown fence is a habit of the model, not a property of the grading:
#: `claude-opus-4-8`, `claude-sonnet-5` and `claude-fable-5` return it bare, `claude-haiku-4-5`
#: fences every one, in both judge roles, deterministically (ATO-1885). Reading only the incumbents'
#: habit made a whole judge unusable at three billed attempts per grading.
_FENCE = "```"


def _json_payload(text: str) -> str:
    """The JSON text of a judge reply: the reply itself, or the whole content of one code fence.

    Two accepted shapes, chosen by the reply's own delimiters and not by parsing -- so this is a
    grammar rather than a fallback chain. Nothing is salvaged from a reply that is neither: a fence
    left open, a fence tagged as something other than JSON, or an object with prose around it comes
    back unchanged and fails at :func:`json.loads` as the malformed reply it is. A fence carries no
    information of its own, so removing it cannot change what the judge said -- which is what
    separates this from the degraded modes the repository forbids.
    """
    if not isinstance(text, str):
        return text
    body = text.strip()
    if not body.startswith(_FENCE):
        return text
    tag, newline, rest = body[len(_FENCE):].partition("\n")
    if not newline or tag.strip().lower() not in ("", "json"):
        return text
    rest = rest.rstrip()
    if not rest.endswith(_FENCE):
        return text
    return rest[: -len(_FENCE)]


def parse_verdict(text: str) -> JudgeVerdict:
    """Parse strict JSON ``{passed, rationale}``; raise ValueError on anything else."""
    try:
        data = json.loads(_json_payload(text))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Judge did not return JSON: {text!r}") from exc
    if not isinstance(data, dict) or "passed" not in data or "rationale" not in data:
        raise ValueError(f"Judge JSON missing fields: {data!r}")
    if not isinstance(data["passed"], bool):
        raise ValueError(f"Judge 'passed' must be a JSON boolean, got {data['passed']!r}")
    if not isinstance(data["rationale"], str):
        raise ValueError(f"Judge 'rationale' must be a string, got {data['rationale']!r}")
    return JudgeVerdict(passed=data["passed"], rationale=data["rationale"])


def _vote(verdicts: list[JudgeVerdict]) -> JudgeVerdict:
    """Strict-majority vote over ``passed`` (ties resolve to False, the
    conservative choice for a correctness metric); rationale from the first
    verdict on the winning side."""
    passes = sum(1 for v in verdicts if v.passed)
    won = passes * 2 > len(verdicts)
    rationale = next((v.rationale for v in verdicts if v.passed == won), verdicts[0].rationale)
    return JudgeVerdict(passed=won, rationale=rationale, samples=len(verdicts))


def generate_answer(complete: Completer, *, question: str, context: str, model: str,
                    prompt: jp.AnswerPrompt, query_date: str | None = None) -> str:
    """The reader's answer under ``prompt`` -- the evaluation's own, never a silent default."""
    system, user = prompt.render(question=question, context=context, query_date=query_date)
    return complete(model, system, user).strip()


def _sample_user(user: str, i: int) -> str:
    """Distinct, deterministic prompt per sample so the response cache stores N
    independent gradings instead of replaying sample 0 (cache keys on the prompt).
    Sample 0 is the plain prompt; reruns reproduce each sample by index."""
    return user if i == 0 else f"{user}\n\n[Independent grading pass #{i + 1}; grade afresh.]"


# How many times to ask for one grading before giving up on it. A judge that quotes the candidate
# verbatim can emit unescaped quotes inside its own JSON string; re-asking almost always fixes it.
#
# Public because it is the multiplier a derived judge cap is built from: `required_calls` counts the
# gradings a run PLANS, and this is the only documented way it can spend more than that (see
# `_one_parsed` -- "retries are real egress and count against the cap by construction"). The ceiling
# and the retry policy have to move together, so they are one number rather than two that agree.
MAX_VERDICT_ATTEMPTS = 3
# Appended on a retry. Does double duty: it tells the model what went wrong, AND it changes the
# prompt -- cached_completer keys on the prompt, so re-asking the identical text would replay the
# malformed response from cache forever.
_REPARSE_NUDGE = ("[Your previous reply was not valid JSON. Reply with ONLY the JSON object. Do "
                  "not quote the candidate; paraphrase, so no unescaped quotes appear.]")


class UnparseableVerdict(MemrankError):
    """The judge never returned parseable JSON for one grading, after retries.

    Raised rather than returned so a caller cannot mistake it for a verdict. The runner catches it
    per query and records that query as unjudged -- one bad response must not discard a cell.
    """


def _one_parsed(complete: Completer, model: str, system: str, user: str,
                parse: Callable[[str], Any]) -> Any:
    """One parseable grading, re-asking a malformed response.

    Retrying is not verdict-shopping: it fires only where the response could not be *parsed*, never
    where it parsed to something unwelcome. Retries are real egress and count against the cap by
    construction, since the counter wraps the completer beneath this.

    ``parse`` is a parameter because a nugget grading returns a score on a three-point scale rather
    than a boolean; the retry policy is identical and must not be forked.
    """
    last: Exception | None = None
    for attempt in range(MAX_VERDICT_ATTEMPTS):
        prompt = user if attempt == 0 else f"{user}\n\n{_REPARSE_NUDGE}"
        try:
            return parse(complete(model, system, prompt))
        except ValueError as exc:
            last = exc
    raise UnparseableVerdict(
        f"judge returned unparseable JSON {MAX_VERDICT_ATTEMPTS} times; last error: {last}"
    ) from last


def _one_verdict(complete: Completer, model: str, system: str, user: str) -> JudgeVerdict:
    """One parseable boolean verdict."""
    verdict: JudgeVerdict = _one_parsed(complete, model, system, user, parse_verdict)
    return verdict


def _graded(complete: Completer, model: str, system: str, user: str, samples: int) -> JudgeVerdict:
    """Majority verdict over ``samples`` independent gradings.

    A sample that never parses raises rather than being dropped: silently grading on fewer samples
    than the config records would make the receipt describe a methodology the run did not use.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    return _vote([_one_verdict(complete, model, system, _sample_user(user, i))
                  for i in range(samples)])


#: BEAM's scale. A grading outside it is a malformed response, not a value to clamp -- clamping
#: would let a judge that ignored the instruction still contribute a number.
_NUGGET_SCALE = (0.0, 0.5, 1.0)


@dataclass
class NuggetScore:
    """One rubric criterion, graded."""
    nugget: str
    score: float
    rationale: str
    samples: int = 1


def parse_nugget_score(text: str) -> tuple[float, str]:
    """Parse strict JSON ``{score, rationale}`` on the {0, 0.5, 1} scale; raise otherwise."""
    try:
        data = json.loads(_json_payload(text))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Nugget judge did not return JSON: {text!r}") from exc
    if not isinstance(data, dict) or "score" not in data or "rationale" not in data:
        raise ValueError(f"Nugget JSON missing fields: {data!r}")
    score = data["score"]
    # `bool` is a subclass of `int`, so True would otherwise sail through as 1.0 -- a judge
    # answering the wrong question in the wrong shape must not be read as a full score.
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"Nugget 'score' must be a number, got {score!r}")
    if float(score) not in _NUGGET_SCALE:
        raise ValueError(f"Nugget 'score' must be one of {_NUGGET_SCALE}, got {score!r}")
    if not isinstance(data["rationale"], str):
        raise ValueError(f"Nugget 'rationale' must be a string, got {data['rationale']!r}")
    return float(score), data["rationale"]


def grade_nugget(complete: Completer, *, question: str, answer: str, nugget: str,
                 model: str, samples: int) -> NuggetScore:
    """Grade one rubric criterion, averaging over ``samples`` independent gradings.

    The mean, not a majority: the scale is ordinal, so two samples of 1 and 0.5 are honestly 0.75
    rather than whichever won a vote. This is the one place a value off the three-point scale can
    legitimately appear, and only when ``samples > 1``.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    user = jp.nugget_user(question=question, answer=answer, nugget=nugget)
    graded = [_one_parsed(complete, model, jp.NUGGET_SYSTEM, _sample_user(user, i),
                          parse_nugget_score) for i in range(samples)]
    return NuggetScore(nugget=nugget, score=sum(s for s, _ in graded) / len(graded),
                       rationale=graded[0][1], samples=samples)


def parse_events(text: str) -> list[str]:
    """Parse strict JSON ``{"events": [...]}``; raise ValueError on anything else.

    Empty is a legitimate answer -- a response that lists no events extracts to nothing and scores
    zero, which is the correct outcome rather than an error.
    """
    try:
        data = json.loads(_json_payload(text))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Event extractor did not return JSON: {text!r}") from exc
    if not isinstance(data, dict) or "events" not in data:
        raise ValueError(f"Event JSON missing 'events': {data!r}")
    events = data["events"]
    if not isinstance(events, list) or not all(isinstance(e, str) for e in events):
        raise ValueError(f"'events' must be a list of strings, got {events!r}")
    return [e.strip() for e in events if e.strip()]


def extract_events(complete: Completer, *, question: str, answer: str, model: str) -> list[str]:
    """The events an answer lists, in the order it lists them."""
    user = jp.event_extraction_user(question=question, answer=answer)
    events: list[str] = _one_parsed(complete, model, jp.EVENT_EXTRACTION_SYSTEM, user, parse_events)
    return events


def judge_equivalent(complete: Completer, *, reference: str, candidate: str,
                     model: str, samples: int) -> JudgeVerdict:
    """Whether two short event descriptions name the same event."""
    user = jp.equivalence_user(reference=reference, candidate=candidate)
    return _graded(complete, model, jp.EQUIVALENCE_SYSTEM, user, samples)


def judge_rubric_sufficiency(complete: Completer, *, question: str, context: str,
                             rubric: list[str], model: str, samples: int) -> JudgeVerdict:
    """Whether the retrieved context could satisfy the criteria the answer is graded on."""
    user = jp.rubric_sufficiency_user(question=question, context=context, rubric=rubric)
    return _graded(complete, model, jp.SUFFICIENCY_SYSTEM, user, samples)


def judge_sufficiency(complete: Completer, *, question: str, context: str, gold: str,
                      model: str, samples: int) -> JudgeVerdict:
    user = jp.sufficiency_user(question=question, context=context, gold=gold)
    return _graded(complete, model, jp.SUFFICIENCY_SYSTEM, user, samples)


def judge_answer(complete: Completer, *, question: str, answer: str, gold: str,
                 model: str, samples: int, negative: bool = False,
                 prompt: tuple[str, str] | None = None) -> JudgeVerdict:
    """Grade one answer. ``prompt`` is an optional ``(system, gold_label)`` override.

    A benchmark supplies it when the protocol it implements defines per-question-type grading
    rules -- LongMemEval ships six such prompts and they are the only ones in this repo with a
    published human-agreement figure. When absent, the generic pair is used and the bytes are
    identical to what every benchmark sent before the override existed.
    """
    if prompt is not None:
        system, gold_label = prompt
    else:
        system = jp.CORRECTNESS_NEGATIVE_SYSTEM if negative else jp.CORRECTNESS_SYSTEM
        gold_label = "Reference answer"
    user = jp.correctness_user(question=question, answer=answer, gold=gold, gold_label=gold_label)
    return _graded(complete, model, system, user, samples)


def no_context_calls(cfg: JudgeConfig) -> int:
    """The no-context control's calls for one query: answer without context, then judge it.

    Counted apart from the rest because it is ENGINE-INDEPENDENT -- both prompts are built from the
    question, the gold answer and the model, and never from anything retrieved. Two engines judged
    in one comparison therefore send byte-identical requests here, so the second one is a cache hit;
    and since :func:`memrank.judging.client.build_completer` composes ``cache(cap(base))``, a hit
    never reaches the cap or the counter. A sweep pays this half once, not once per engine. """
    return 1 + cfg.samples if cfg.no_context_control else 0


def with_context_calls(cfg: JudgeConfig, *, negative: bool) -> int:
    """The calls that depend on what an engine actually retrieved, for one query.

    Sufficiency (positives only -- a negative has nothing to find), the answer generated from the
    retrieved context, and the grading of that answer. Every one of them contains the engine's own
    context, so no other engine's run can satisfy it from cache: this half multiplies by targets.
    """
    calls = cfg.samples if not negative else 0   # sufficiency
    return calls + 1 + cfg.samples               # answer with context, then judge it


def calls_per_query(cfg: JudgeConfig, *, negative: bool) -> int:
    """How many completer calls :func:`judge_query` makes for one query.

    A restatement of the function directly below, and deliberately adjacent to it: the two must
    change together or a preflight built on this quotes a number the run does not honour.
    ``tests/judging/test_judge_budget.py`` runs both against a counting completer and fails if they drift.

    Args:
        cfg: The judge configuration; ``samples`` and ``no_context_control`` both change the cost.
        negative: Whether the query is a negative (no expected answer), which skips sufficiency.

    Returns:
        The exact call count for one query.
    """
    return no_context_calls(cfg) + with_context_calls(cfg, negative=negative)


def required_calls(units: Sequence[Any], cfg: JudgeConfig, *, targets: int = 1,
                   shape: JudgeShape | None = None) -> int:
    """Billable judge calls a run needs, knowable before the first one is spent.

    Two things this counts that a naive per-query sum gets wrong, each measured on a real
    benchmark (locomo/smoke, default config):

    - **Only judgeable queries.** The shape's ``is_judgeable`` decides what ``_apply_judge``
      grades, and
      the ungraded ones cost nothing. All 152 of locomo/smoke's non-adversarial queries are
      judgeable since the category-map fix (LoCoMoBenchmark.VERSION 1): 760 calls. Before that
      fix the mislabeled map admitted 69 of them (345 calls) -- a quote that documented the bug,
      not the benchmark.
    - **The no-context half is paid once for the whole sweep** *when the cache is on*, because the
      cap counter is built once and shared across the fan-out while that half is
      engine-independent (see :func:`no_context_calls`). Four targets on locomo/smoke is 2,128 --
      not 760, which is what a per-target estimate quoted for a sweep that then died mid-run, and
      not 3,040 either. Under ``--no-judge-cache`` there is no sharing to claim and every engine
      pays it again, so the discount is conditional on ``cfg.cache``: assuming it unconditionally
      under-costs an uncapped-cache run by exactly the control's share.

    An UPPER bound otherwise, deliberately. A cache warmed by an EARLIER run makes the real
    billable count lower and never higher, so the error is toward refusing a run that would have
    fitted rather than admitting one that dies part-way -- and a capped run produces nothing at all.

    Args:
        units: The loaded benchmark units, already filtered to what will run.
        cfg: The judge configuration.
        targets: How many engines share this cap; a sweep's fan-out width.

    Returns:
        The most calls the run can be billed for.
    """
    if shape is None:
        from memrank.judging.shape import GENERIC_BINARY_CATEGORIES, BinaryJudgeShape
        shape = BinaryJudgeShape(GENERIC_BINARY_CATEGORIES)
    judgeable = [q for unit in units for q in unit.queries if shape.is_judgeable(q)]
    control = sum(shape.control_calls(cfg, q) for q in judgeable)
    per_target = sum(shape.context_calls(cfg, q) for q in judgeable)
    if not cfg.cache:
        return (control + per_target) * targets
    return control + per_target * targets


def judge_query(complete: Completer, *, question: str, context: str, gold: str,
                cfg: JudgeConfig, negative: bool,
                query_date: str | None = None,
                prompt: tuple[str, str] | None = None) -> JudgedQuery:
    answered_without_context = False
    if cfg.no_context_control:
        # The control gets the date too: it must differ from the real answer by CONTEXT only,
        # or a date-driven improvement would be misread as context dependence. It gets the same
        # per-type prompt for the same reason.
        nc_answer = generate_answer(complete, question=question, context="",
                                    model=cfg.answer_model, prompt=cfg.answer_prompt,
                                    query_date=query_date)
        answered_without_context = judge_answer(
            complete, question=question, answer=nc_answer, gold=gold,
            model=cfg.judge_model, samples=cfg.samples, negative=negative,
            prompt=prompt).passed
    sufficiency = None
    if not negative:
        sufficiency = judge_sufficiency(complete, question=question, context=context,
                                        gold=gold, model=cfg.judge_model, samples=cfg.samples)
    answer = generate_answer(complete, question=question, context=context,
                             model=cfg.answer_model, prompt=cfg.answer_prompt,
                             query_date=query_date)
    correctness = judge_answer(complete, question=question, answer=answer, gold=gold,
                               model=cfg.judge_model, samples=cfg.samples, negative=negative,
                               prompt=prompt)
    return JudgedQuery(answer, sufficiency, correctness, answered_without_context,
                       score=1.0 if correctness.passed else 0.0)
