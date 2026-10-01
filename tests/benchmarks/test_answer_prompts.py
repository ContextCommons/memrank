"""Each evaluation answers with its benchmark's official reader prompt, pinned to its source.

The texts are pinned by hash of the verbatim constants: a change here is a methodology change
(JUDGE_PROMPT_VERSION), never an edit. The hashes were taken from the authors' files at the
commits the prompts' ``source`` URLs name.
"""

from __future__ import annotations

import hashlib

import pytest

from memrank.benchmarks import REGISTRY
from memrank.benchmarks import answer_prompts as ap
from memrank.judging.prompts import ANSWER_SYSTEM, MEMRANK_READER

PINNED = {
    "BEAM_ANSWER_GENERATION_FOR_RAG":
        "ed89b98d1024432a46bd297345961cf89d2429746daec37befb50db46e0c914a",
    "LOCOMO_QA_PROMPT": "a30fe18a6c9b58c5d22b6188860e5a99cdfef33fcc8a36141b571d954b9b0266",
    "LONGMEMEVAL_DIRECT_READER":
        "e427ff913456e51a132ec865b1b5038d562bdc36890976943ad421cc9b365c9d",
}

OFFICIAL = {"beam": ap.BEAM_READER, "locomo": ap.LOCOMO_READER,
            "longmemeval": ap.LONGMEMEVAL_READER}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_each_official_prompt_is_the_pinned_verbatim_text(name: str) -> None:
    text = getattr(ap, name)
    assert hashlib.sha256(text.encode()).hexdigest() == PINNED[name]


def test_beam_keeps_its_two_trailing_spaces() -> None:
    assert "in the context below. \n" in ap.BEAM_ANSWER_GENERATION_FOR_RAG
    assert "without any explanation \n" in ap.BEAM_ANSWER_GENERATION_FOR_RAG


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_benchmark_declares_its_answer_prompt(name: str) -> None:
    declared = REGISTRY[name].__dict__.get("answer_prompt")
    assert declared is not None, f"{name} must declare answer_prompt, not inherit it silently"
    assert declared is OFFICIAL.get(name, MEMRANK_READER)


@pytest.mark.parametrize("name", sorted(OFFICIAL))
def test_official_prompts_name_a_pinned_source_and_impose_no_abstention(name: str) -> None:
    prompt = OFFICIAL[name]
    assert prompt.official and "/blob/" in prompt.source
    system, user = prompt.render(question="Q", context="C", query_date="2023/05/30")
    assert system == ""  # every official reader is one user message
    assert "I don't know" not in user and ANSWER_SYSTEM not in user


def test_memrank_reader_is_the_declared_non_official_fallback() -> None:
    assert not MEMRANK_READER.official
    system, _ = MEMRANK_READER.render(question="Q", context="C")
    assert system == ANSWER_SYSTEM


def test_locomo_renders_as_its_harness_does() -> None:
    _, user = ap.LOCOMO_READER.render(question="Where?", context="ctx")
    assert user == "ctx\n\n" + ap.LOCOMO_QA_PROMPT.format("Where?")


def test_beam_renders_context_then_question_without_a_date() -> None:
    _, user = ap.BEAM_READER.render(question="Q?", context="C {not a field}",
                                    query_date="2024-01-01")
    expected = (ap.BEAM_ANSWER_GENERATION_FOR_RAG.replace("<context>", "C {not a field}")
                .replace("<question>", "Q?"))
    assert user == expected and "2024-01-01" not in user


def test_longmemeval_states_the_date_and_refuses_a_question_without_one() -> None:
    _, user = ap.LONGMEMEVAL_READER.render(question="Q?", context="C", query_date="2023/05/30")
    assert user.endswith("History Chats:\n\nC\n\nCurrent Date: 2023/05/30\nQuestion: Q?\nAnswer:")
    with pytest.raises(ValueError, match="states the current date"):
        ap.LONGMEMEVAL_READER.render(question="Q?", context="C")


@pytest.mark.parametrize(("ref", "expected"), [
    ("beam:100k", ap.BEAM_READER), ("locomo", ap.LOCOMO_READER),
    ("longmemeval", ap.LONGMEMEVAL_READER), ("demo", MEMRANK_READER),
    ("evals/support.yaml", MEMRANK_READER)])
def test_an_eval_ref_names_its_prompt(ref: str, expected) -> None:
    assert ap.answer_prompt_for(ref) is expected


def test_an_unknown_eval_ref_is_refused_not_given_the_fallback() -> None:
    with pytest.raises(Exception, match="unknown eval"):
        ap.answer_prompt_for("nosuch")


def test_identity_changes_with_the_bytes() -> None:
    identities = {p.identity for p in (*OFFICIAL.values(), MEMRANK_READER)}
    assert len(identities) == 4
    assert ap.BEAM_READER.identity.startswith("beam-answer-generation-for-rag@")
