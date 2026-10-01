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
"""Each benchmark's OFFICIAL reader prompt, verbatim, pinned to the commit it was copied from.

The reader turns a question and a context into the answer the judge grades, so its wording is
part of the benchmark's protocol: a reader told to reply "I don't know" when unsure scores a
different benchmark from one that is not. Each prompt below is the authors' own, copied byte for
byte (tests/benchmarks/test_answer_prompts.py pins the text), and sent the way their code sends
it: one user message, no system prompt. Two things memrank's own reader adds are deliberately
absent -- the untrusted-data sentinel wrapping, and (for BEAM and LoCoMo) a current-date line --
because neither official reader has them and the dataset text is trusted benchmark data, not user
input. Fidelity to each benchmark's reader is what keeps a score comparable with published ones.

An evaluation whose benchmark publishes no reader keeps memrank's own,
:data:`memrank.judging.prompts.MEMRANK_READER`, declared as such.
"""

from __future__ import annotations

from memrank.judging.prompts import MEMRANK_READER, AnswerPrompt

_BEAM_COMMIT = "b2da22eac88bb0874c64665f13457eb99835774a"
_LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
_LONGMEMEVAL_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"

#: ``answer_generation_for_rag`` from BEAM's src/prompts.py, filled by
#: src/answer_probing_questions/long_term_memory_methods.py with ``.replace("<context>", ...)``
#: then ``.replace("<question>", ...)`` and sent as the whole prompt. Spelled with escapes so
#: its two trailing spaces survive editors.
BEAM_ANSWER_GENERATION_FOR_RAG = (
    "\nYou are an assistant that MUST answer questions using ONLY the information provided in "
    "the context below. \n\nSTRICT INSTRUCTIONS:\n1. Answer ONLY based on the provided context\n"
    "2. Do NOT use your internal knowledge\n\nCONTEXT:\n<context>\n\nQUESTION:\n<question>\n\n"
    "ANSWER REQUIREMENTS:\n- Be direct and concise\n"
    "- Only output the answer to the question without any explanation \n\nRESPONSE:\n")

#: ``QA_PROMPT`` from LoCoMo's task_eval/gpt_utils.py. The official call is
#: ``context + '\n\n' + QA_PROMPT.format(question)``. Category 5 (adversarial) has its own
#: ``QA_PROMPT_CAT_5``, a multiple choice built from the gold answer; memrank excludes category 5
#: (444 of its 446 items ship no ``answer`` to build the choice from), so it is not used here.
LOCOMO_QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

#: The direct (no chain-of-thought) reader from LongMemEval's src/generation/run_generation.py,
#: ``prepare_prompt`` with the default ``merge_key_expansion_into_value='none'``: the variant used
#: both for retrieved chunks and for the full history, so it fits every agent's context. The
#: ``replace`` variant ("several facts extracted from history chats") presumes extracted facts,
#: which only some engines return, and would misdescribe a raw-history context.
LONGMEMEVAL_DIRECT_READER = (
    "I will give you several history chats between you and a user. Please answer the question "
    "based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n"
    "Question: {}\nAnswer:")

BEAM_READER = AnswerPrompt(
    name="beam-answer-generation-for-rag",
    source=f"https://github.com/mohammadtavakoli78/BEAM/blob/{_BEAM_COMMIT}/src/prompts.py",
    system="",
    template=BEAM_ANSWER_GENERATION_FOR_RAG.replace("<context>", "{}").replace("<question>", "{}"),
    fields=("context", "question"))

LOCOMO_READER = AnswerPrompt(
    name="locomo-qa-prompt",
    source=f"https://github.com/snap-research/locomo/blob/{_LOCOMO_COMMIT}/task_eval/gpt_utils.py",
    system="",
    template="{}\n\n" + LOCOMO_QA_PROMPT,
    fields=("context", "question"))

LONGMEMEVAL_READER = AnswerPrompt(
    name="longmemeval-direct-reader",
    source=(f"https://github.com/xiaowu0162/LongMemEval/blob/{_LONGMEMEVAL_COMMIT}"
            "/src/generation/run_generation.py"),
    system="",
    template=LONGMEMEVAL_DIRECT_READER,
    fields=("context", "date", "question"))


def answer_prompt_for(evaluation: str) -> AnswerPrompt:
    """The reader prompt for an eval ref, or :data:`MEMRANK_READER` for an evaluation file.

    Asks the benchmark class rather than constructing it, so nothing is loaded.
    """
    from memrank.benchmarks import REGISTRY
    from memrank.benchmarks.refs import parse_eval_ref
    from memrank.definitions.file import is_file_ref

    if is_file_ref(evaluation):
        return MEMRANK_READER
    name, _, _ = parse_eval_ref(evaluation)
    return REGISTRY[name].answer_prompt
