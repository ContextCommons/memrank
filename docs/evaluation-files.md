# Evaluation files

An evaluation file is your own evaluation, run exactly as a shipped one is:

```bash
memrank evals new my-eval.yaml       # write a commented starter
memrank evals check my-eval.yaml     # validate it and summarise it; runs no agent
memrank run my-eval.yaml --agent ./my-agent.yaml
```

`memrank run` treats an argument that contains `/` or ends in `.yaml`, `.yml` or `.jsonl` as an
evaluation file, and anything else as a shipped evaluation (`memrank evals ls`). Everything else
about the run is the same: `--cases`, `--questions`, `--seed`, `--concurrency`, `--no-judge`,
`--resume`, the live page, the upload to your organisation's run history and the ending.

A file that is malformed is refused before anything runs, with every problem named by its case,
question and field:

```text
! The run didn't start

What happened
  The evaluation my-eval.yaml has 2 problems, so nothing ran:
    - case 1 (trip), question 2 (trip-nights), `answer`: 'seven' is not a number
    - case 2 (capital), question 1 (capital), `answr`: is not a field memrank knows (check its spelling)
```

## The file

```yaml
name: support-faq          # lowercase letters, digits, '.', '_' or '-'
version: "2"               # optional; bump it when you change a case or its grading
description: One sentence. # optional
grade: judge               # optional default grader (see below)
cases:
  - id: refunds            # optional; letters, digits, '.', '_' or '-'
    history:               # optional: past conversations, fed before the questions
      - timestamp: "2026-03-02"
        turns:
          - {role: user, text: "I bought the blue kettle on the 2nd."}
          - {role: assistant, text: "Thanks, noted."}
    questions:
      - id: refund-window  # optional; unique in the file
        question: How long do I have to return the kettle?
        answer: 30 days    # the reference; a list gives several acceptable answers
        category: policy   # optional; scores are also reported per category
        grade: exact
  - question: What is 2 + 2?   # a case that is one question on its own, with no history
    answer: 4
    grade: numeric
```

A turn's `role` is `user` or `assistant`; `speaker` (a name) and `timestamp` are optional. A
question may carry a `timestamp`, which the agent receives as the question's date. With
`choices`, the agent is asked the question followed by the choices, lettered `A.`, `B.`, and so
on.

Cases can also be kept outside the YAML file:

- `cases: {file: cases.jsonl}` reads one case per line, each written as in the list above.
- `cases: {command: [python3, make_cases.py]}` runs your program from the file's folder, with
  `--seed N` added (the run's `--seed`), and reads one JSON case per line from its output. It
  must print the same cases for the same seed; a run continues after an interruption only on
  the cases it started with.
- A bare `.jsonl` file is an evaluation of its own, named after the file, with no version.

`{python}` in a command is the Python that runs memrank. Commands run without a shell.

## Grading

A question is graded by its own `grade`; else by `rubric` when it has a `rubric` and by
`choice` when it has `choices`; else by the file's `grade`; else by `judge`.

| Grader | Passes when | Needs |
|---|---|---|
| `exact` | the answer equals a reference once both are lowercased, with punctuation, the articles a/an/the and extra spaces removed | `answer` |
| `choice` | the answer picks the right choice: by its letter (`B`, `(B)`, `B.`), its text, or naming only that choice | `choices`, `answer` (the right choice's text or letter) |
| `numeric` | the last number in the answer is within `tolerance` (default 0) of a reference: `grade: {kind: numeric, tolerance: 0.5}` | `answer`, a number |
| `judge` | the judge model decides the answer matches the reference | `answer` |
| `rubric` | scored by the judge model per criterion (0, 0.5 or 1) and averaged; passes when every criterion is met | `rubric`, a list of criteria |
| a command | your program's score is 1: `grade: {command: [python3, grade.py]}` | nothing |

A command grader runs from the file's folder, reads one JSON object on stdin -- `case_id`,
`question_id`, `question`, `answer`, `reference` (the first reference), `references`,
`category`, and `choices` or `rubric` when the question has them -- and prints one JSON object:
`{"score": <0 to 1>, "why": "..."}`. A grader that fails or prints anything else stops judging;
the answers are kept, and `memrank run --resume <run-id>` judges them again once it is fixed.

Built-in graders score 1 or 0; `rubric` and a command can give partial credit. A question the
agent failed on scores 0, as for a shipped evaluation, and scores come with 95% intervals that
resample whole cases.

`judge` and `rubric` use the judge model -- Claude Haiku unless `memrank run --judge-model` names
another -- with your organisation's saved Anthropic key. An
evaluation none of whose questions use them needs no key, and `memrank evals check` says which
case you are in.

## What a run records

A run on an evaluation file is recorded under `<name>@<version>` (or `<name>` without a
version), so runs on the same evaluation are compared together. Its result also records the
file's path, its version, and a fingerprint: a SHA-256 over the file and every file its
commands name (a case program, a grader script, a JSONL case list). Two runs with the same
fingerprint ran the same evaluation. The dataset version shown for the run is a digest of the
exact cases and questions it asked.
