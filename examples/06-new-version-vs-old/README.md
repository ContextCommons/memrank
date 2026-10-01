# 06 -- new version vs old

`uv run python examples/06-new-version-vs-old/run.py`

Compare `TinyMemory` with `TinyMemoryV2`, which removes words shared by every question before
counting overlap. Both run on the same three tasks with a retrieval limit of two documents.

The script prints a paired comparison: mean difference (`gap`), changed tasks (`flips`) and a
caution about limited evidence. Inspect individual changes as well as the mean, since gains and
regressions can cancel out.
