"""Evaluation files: what they load into, and every way a broken one is refused up front."""

from __future__ import annotations

import shutil

import pytest

from memrank.definitions import is_file_ref, resolve
from memrank.definitions.commands import CommandFailed
from memrank.definitions.file import FileDefinition
from memrank.definitions.problems import EvaluationInvalid
from tests.definitions.conftest import FIXTURES


def test_a_hand_written_file_loads_its_cases_questions_and_identity():
    definition = resolve(str(FIXTURES / "travel.yaml"))
    cases = definition.load_cases(0)
    assert definition.evaluation == "travel@2" and definition.version == "2"
    assert definition.fingerprint.startswith("sha256:")
    assert [c.id for c in cases] == ["trip", "capital"]
    trip, capital = cases
    assert [t.text for t in trip.sessions[0].turns][0].startswith("I am flying to Lisbon")
    assert trip.sessions[0].timestamp == "2026-03-02"
    assert capital.sessions == () and capital.queries[0]["gold_answers"] == ["Canberra"]
    assert capital.question(0).text.endswith("A. Sydney\nB. Canberra\nC. Melbourne")
    assert "Canberra" not in capital.question(0).model_dump_json().split("B. ")[0]
    assert definition.uses_model(0)  # the judged question


@pytest.mark.parametrize("ref, is_file", [
    ("locomo", False), ("beam:100k", False), ("my-eval.yaml", True), ("cases.jsonl", True),
    ("./evals/x", True), ("x.yml", True),
])
def test_a_path_or_a_yaml_or_jsonl_name_is_a_file(ref, is_file):
    assert is_file_ref(ref) is is_file


PROBLEMS = [
    ("name: Bad Name\ncases: [{question: q, answer: a}]", "`name`: must be lowercase"),
    ("name: locomo\ncases: [{question: q, answer: a}]", "shipped evaluation's name"),
    ("name: x\n", "`cases`: give a list of cases"),
    ("name: x\ncases: [{questions: []}]", "case 1, `questions`"),
    ("name: x\ncases: [{id: c, questions: [{question: q, answr: a}]}]",
     "case 1 (c), question 1, `answr`: is not a field"),
    ("name: x\ncases: [{question: q}]", "case 1, question 1, `answer`: is required by the "
                                        "judge grader"),
    ("name: x\ncases: [{question: q, grade: exact}]", "`answer`: is required by the exact"),
    ("name: x\ncases: [{question: q, answer: many, grade: numeric}]", "'many' is not a number"),
    ("name: x\ncases: [{question: q, choices: [a, b], answer: c}]", "neither one of the choices"),
    ("name: x\ncases: [{question: q, choices: [a, A], answer: a}]", "two choices read the same"),
    ("name: x\ncases: [{question: q, rubric: []}]", "`rubric`"),
    ("name: x\ncases: [{question: q, answer: a, grade: fuzzy}]", "`grade.kind`"),
    ("name: x\ncases: [{question: q, answer: a, grade: {kind: exact, tolerance: 1}}]",
     "`tolerance` is for the numeric grader"),
    ("name: x\ncases: [{id: a, question: q, answer: a}, {id: a, question: r, answer: b}]",
     "case 2 (a): `id` 'a' is used by another case"),
    ("name: x\ncases: [{id: 'a b', question: q, answer: a}]", "`id`: must be letters"),
    ("name: x\ncases: [{history: [{turns: [{role: robot, text: hi}]}], "
     "questions: [{question: q, answer: a}]}]", "history session 1, turn 1, `role`"),
    ("name: x\ncases: {file: missing.jsonl}", "`cases.file`"),
    ("name: x\ncases: {file: a.jsonl, command: [b]}", "`cases`: give the cases a `file` or"),
    ("name: [x", "not valid YAML"),
]


@pytest.mark.parametrize("text, expected", PROBLEMS)
def test_a_malformed_file_is_refused_naming_the_case_and_field(write, text, expected):
    with pytest.raises(EvaluationInvalid) as refused:
        FileDefinition(write(text))
    assert any(expected in problem for problem in refused.value.problems), \
        refused.value.problems
    assert "memrank evals check" in str(refused.value)


def test_every_problem_is_reported_at_once(write):
    path = write("""\
        name: Bad
        cases:
          - {id: c, questions: [{question: q, grade: exact}, {question: r, answer: x, grade: numeric}]}
        """)
    with pytest.raises(EvaluationInvalid) as refused:
        FileDefinition(path)
    assert len(refused.value.problems) == 3


def test_a_missing_file_is_refused_as_a_step_not_a_crash(tmp_path):
    with pytest.raises(EvaluationInvalid, match="cannot be read"):
        resolve(str(tmp_path / "nope.yaml"))


def test_cases_can_live_in_a_jsonl_file_and_a_bare_jsonl_file_is_an_evaluation(write):
    lines = '{"id": "a", "question": "q1", "answer": "x"}\n\n{"question": "q2", "answer": "y"}\n'
    write(lines, "cases.jsonl")
    wrapped = FileDefinition(write("name: wrapped\ngrade: exact\ncases: {file: cases.jsonl}\n"))
    bare = FileDefinition(write(lines, "cases.jsonl"))
    assert [c.id for c in wrapped.load_cases(0)] == ["a", "case-2"]
    assert bare.evaluation == "cases" and not wrapped.uses_model(0) and bare.uses_model(0)


def test_a_bad_jsonl_line_is_named_by_its_line_number(write):
    with pytest.raises(EvaluationInvalid) as refused:
        FileDefinition(write('{"question": "q", "answer": "a"}\nnot json\n', "cases.jsonl"))
    assert refused.value.problems[0].startswith("cases.jsonl line 2: not JSON")


def test_a_case_command_is_given_the_seed_and_its_cases_are_checked(tmp_path):
    for name in ("counting.yaml", "make_cases.py", "grade_contains.py"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    definition = FileDefinition(tmp_path / "counting.yaml")
    first, other = definition.load_cases(1), definition.load_cases(2)
    assert [c.queries[0]["gold_answers"] for c in first] == [["100"], ["101"], ["102"]]
    assert other[0].queries[0]["gold_answers"] == ["200"]
    assert not definition.uses_model(1)


def test_a_failing_case_command_says_what_it_printed(write):
    path = write("name: x\ncases: {command: ['{python}', '-c', 'import sys; sys.exit(\"boom\")']}\n")
    with pytest.raises(CommandFailed, match="The case command exited 1: .*\n.*boom"):
        FileDefinition(path).load_cases(0)


def test_the_fingerprint_covers_the_files_a_command_names(tmp_path):
    for name in ("counting.yaml", "make_cases.py", "grade_contains.py"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    before = FileDefinition(tmp_path / "counting.yaml").fingerprint
    grader = tmp_path / "grade_contains.py"
    grader.write_text(grader.read_text() + "\n# changed\n")
    assert FileDefinition(tmp_path / "counting.yaml").fingerprint != before
