# 01 -- a first result

`uv run python examples/01-first-result/run.py`

The local installation check runs `TFIDF` on the bundled `Demo` scenario. After installation it
needs no service, network access or API key.

The script prints the system, evaluation, measured values and trace count. `demo-score` is
evidence recall, a retrieval proxy, not answer correctness.

Use `memrank.catalog()` to list other systems and evaluations with their requirements.
