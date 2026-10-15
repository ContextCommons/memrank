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
"""The typed questions Jev is asked, versioned as one question set.

These play the part :mod:`memrank.judging.prompts` plays for a text judge, and are a methodology
artifact the same way: changing a word of them is a methodology change and a bump of
:data:`QUESTION_SET`, which the cache key and every record carry.

TypeSafe's guidance is one condition per Noul, combined in code, so a binary verdict is three
independent Nouls over shared state rather than one "is it correct" question; the rule that
combines them is :mod:`memrank.judging.jev.rules`. Each LongMemEval question type's own grading
rule (the six official prompts in :data:`memrank.judging.prompts.LME_PROMPTS`) is carried into the
criteria of the Noul it bears on, and the reference is named in the state the way the official
prompts name it: an answer, a rubric, or an explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The version of every question below. Bumped with any change to their wording or shape.
QUESTION_SET = "jev-2026-10-15.1"

#: The three binary conditions, as the rule and the record name them.
STATES = "states_reference"
CONTRADICTS = "contradicts_reference"
DECLINES = "declines"

_EQUIVALENT = ("Judge by meaning: paraphrases, synonyms, and equivalent forms of dates, times and "
               "numbers count (7 May 2023 and May 7, 2023; two weeks and 14 days).")

_DECLINES = {
    "type": "noul",
    "instructions": "Does `candidate_answer` decline to answer `question`?",
    "criteria": {
        "true": ("It does not commit to an answer: it says it does not know or cannot tell, that "
                 "the information is missing or incomplete, or that related information is "
                 "available but not the information asked for -- even if it also offers other "
                 "content."),
        "false": "It commits to an answer to the question.",
    },
}


@dataclass(frozen=True)
class BinaryRules:
    """How one kind of question is graded: what the reference is, and each Noul's criteria.

    ``abstention`` flips the verdict: the question is unanswerable, so only declining passes and
    the other two conditions are not asked.
    """

    reference_field: str
    states_true: str
    states_false: str
    conflict_false: str
    abstention: bool = False


_STATES_FALSE = ("It omits the reference information, gives only part of it, or gives different "
                 "information.")
_CONFLICT_FALSE = ("Nothing in it is incompatible with the reference; compatible extra detail is "
                   "not a conflict.")

#: The generic rule, for LoCoMo and an evaluation file's ``judge`` grader.
GENERIC = BinaryRules(
    reference_field="reference_answer",
    states_true=f"It gives the information in the reference answer, in any wording. {_EQUIVALENT} "
                "Extra correct detail is fine.",
    states_false=_STATES_FALSE, conflict_false=_CONFLICT_FALSE)

# LongMemEval's default prompt: the answer, an equivalent, or every step to it; a subset is not.
_LME_STATES = (f"It contains the correct answer, is equivalent to it, or contains all the "
               f"intermediate steps to get it. {_EQUIVALENT}")
_LME_SUBSET = ("It does not contain the correct answer, or contains only a subset of the "
               "information the answer requires.")
_LME_DEFAULT = BinaryRules("reference_answer", _LME_STATES, _LME_SUBSET, _CONFLICT_FALSE)
_OFF_BY_ONE = ("An off-by-one error in a number of days, weeks, months or years (19 days when the "
               "answer is 18) still counts.")

#: LongMemEval's six official rules, keyed by ``judge_prompt_key`` exactly as its prompts are.
LONGMEMEVAL: dict[str, BinaryRules] = {
    "single-session-user": _LME_DEFAULT,
    "single-session-assistant": _LME_DEFAULT,
    "multi-session": _LME_DEFAULT,
    "temporal-reasoning": BinaryRules(
        "reference_answer", f"{_LME_STATES} {_OFF_BY_ONE}", _LME_SUBSET,
        f"{_CONFLICT_FALSE} {_OFF_BY_ONE.replace('still counts', 'is not a conflict')}"),
    "knowledge-update": BinaryRules(
        "reference_answer",
        f"{_LME_STATES} It may also mention earlier information, as long as the updated answer "
        f"is the one it gives as the answer.",
        _LME_SUBSET,
        f"{_CONFLICT_FALSE} Earlier, superseded information mentioned beside the updated answer "
        f"is not a conflict."),
    "single-session-preference": BinaryRules(
        "rubric",
        "It satisfies the desired personalized response the rubric describes: it recalls and uses "
        "the user's personal information correctly. It need not reflect every point in the "
        "rubric.",
        "It does not recall or use the user's personal information as the rubric describes.",
        "Nothing it says about the user is incompatible with the rubric."),
    "abstention": BinaryRules("explanation", "", "", "", abstention=True),
}


def binary_state(rules: BinaryRules, *, question: str, reference: str,
                 answer: str) -> dict[str, str]:
    """The shared state the binary Nouls read: the question, its reference, the answer."""
    return {"question": question, rules.reference_field: reference, "candidate_answer": answer}


def binary_questions(rules: BinaryRules) -> dict[str, Any]:
    """The Nouls one binary verdict is decided from, one condition each."""
    if rules.abstention:
        return {DECLINES: _DECLINES}
    reference = f"`{rules.reference_field}`"
    return {
        STATES: {
            "type": "noul",
            "instructions": f"Does `candidate_answer` give the information in {reference} in "
                            f"answer to `question`?",
            "criteria": {"true": rules.states_true, "false": rules.states_false},
        },
        CONTRADICTS: {
            "type": "noul",
            "instructions": f"Does `candidate_answer` assert anything that conflicts with "
                            f"{reference}?",
            "criteria": {
                "true": ("It asserts a fact, date, number or name incompatible with the "
                         "reference, anywhere in the answer -- including alongside the correct "
                         "information."),
                "false": rules.conflict_false,
            },
        },
        DECLINES: _DECLINES,
    }


#: The three levels a rubric criterion is scored on, in order: 0, 0.5 and 1 once mapped.
RUBRIC_LEVELS = (
    "Not met: the response does not satisfy the criterion, or does not address the question.",
    "Partly met: the criterion is addressed but incompletely or partly inaccurately.",
    "Fully met: the response fully satisfies the criterion.",
)

#: BEAM's own grading rules, as :data:`memrank.judging.prompts.NUGGET_SYSTEM` states them.
_RUBRIC_RULES = (
    f"{_EQUIVALENT} Ignore tone, length and formatting unless the criterion requires a format. A "
    "criterion phrased as a prohibition (should not, avoid, should abstain) is met when the "
    "response honours it AND still addresses the question.")


def rubric_key(index: int) -> str:
    return f"criterion_{index}"


def rubric_questions(rubric: list[str]) -> dict[str, Any]:
    """One Score per criterion, all over the shared ``{question, response}`` state."""
    return {rubric_key(i): {
        "type": "score",
        "instructions": {"criterion": item, "rules": _RUBRIC_RULES,
                         "question": "How fully does `response` satisfy `criterion`?"},
        "criteria": list(RUBRIC_LEVELS),
    } for i, item in enumerate(rubric)}


#: The options a reference pair's order question answers with, as :mod:`rules` reads them.
A_FIRST, B_FIRST, UNCLEAR = "a_first", "b_first", "unclear"
EXTRAS = "lists_other_events"


def present_key(index: int) -> str:
    return f"mentions_{index}"


def pair_key(first: int, second: int) -> str:
    return f"order_{first}_{second}"


def pair_events(first: int, second: int) -> tuple[int, int]:
    """Which reference event is shown as ``event_a``: alternated, so ``a_first`` is not always
    the reference order and a lean toward the first-named option cannot pass as agreement."""
    return (first, second) if (first + second) % 2 == 0 else (second, first)


def ordering_questions(events: list[str]) -> dict[str, Any]:
    """A Noul per reference event, a Choice per reference pair, and a Noul for other events."""
    questions: dict[str, Any] = {present_key(i): {
        "type": "noul",
        "instructions": {"event": event,
                         "question": "Does `answer` state that `event` happened?"},
        "criteria": {
            "true": "It asserts this event, in any wording, as something that happened.",
            "false": "It does not mention it, or mentions it only negated, hedged, or as a "
                     "possibility.",
        },
    } for i, event in enumerate(events)}
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            a, b = pair_events(i, j)
            questions[pair_key(i, j)] = {
                "type": "choice",
                "instructions": {"event_a": events[a], "event_b": events[b],
                                 "question": "In the order `answer` gives, which of `event_a` "
                                             "and `event_b` comes first?"},
                "criteria": {
                    A_FIRST: "It mentions both and puts event_a before event_b.",
                    B_FIRST: "It mentions both and puts event_b before event_a.",
                    UNCLEAR: "It does not mention both, or does not make their order clear.",
                },
            }
    questions[EXTRAS] = {
        "type": "noul",
        "instructions": {"reference_events": events,
                         "question": "Does `answer` list any event that is not one of "
                                     "`reference_events`?"},
        "criteria": {"true": "It lists at least one event matching none of the reference events.",
                     "false": "Every event it lists is one of the reference events, in any "
                              "wording."},
    }
    return questions
