"""Live judge smoke test -- makes a real Anthropic call, so it is explicit opt-in.

Two conditions, both required.

**An explicit opt-in**, ``MEMRANK_LIVE_TESTS=1``, matching the house convention for anything with a
cost or a side effect (``ARENA_BETA_MODE``, ``MEMRANK_MLFLOW_ENABLED``): default OFF, never
inferred. Keying only on "is a credential available" would make an ordinary ``pytest`` run start
spending money the moment someone stored a key.

**A resolvable credential**, through ``config.secret()`` rather than ``os.environ``, so a key held
in memrank's own wallet counts. Reading the environment alone made this test's execution depend on
whether something had happened to export the key -- a live test that silently stops running is its
own kind of masking.
"""
import os

import pytest

from memrank import config
from memrank.judging.client import JUDGE_SECRET

_OPTED_IN = os.environ.get("MEMRANK_LIVE_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not _OPTED_IN or config.secret(JUDGE_SECRET) is None,
    reason=("MEMRANK_LIVE_TESTS=1 not set" if not _OPTED_IN else
            f"{JUDGE_SECRET} resolves from neither the environment nor the wallet "
            f"(`memrank secrets set {JUDGE_SECRET}`)") + "; live judge test skipped",
)


def test_live_judge_correctness_smoke():
    pytest.importorskip("anthropic")
    from memrank.judging.client import build_completer
    from memrank.judging.judge import JudgeConfig, judge_answer
    complete, _ = build_completer(JudgeConfig())
    v = judge_answer(complete, question="What is 2+2?", answer="4", gold="4",
                     model="claude-opus-4-8", samples=1)
    assert v.passed is True
