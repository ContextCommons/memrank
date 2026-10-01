# 03 -- your own evaluation

`uv run python examples/03-your-own-evaluation/run.py`

Define an evaluation with four context notes, three tasks, scoring measures and a clearing
rule. It uses the same `Evaluation` type as Memrank's included evaluations.

The script prints `WordMatch` values for the tasks and one task's trace. Inspect the trace to
see the supplied context IDs and returned documents behind the score.
