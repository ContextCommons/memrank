"""A retrieving engine that retrieves nothing must fail, not score.

The mem0 adapter sent a search parameter its engine does not honour, so it ingested perfectly and
retrieved nothing on every run memrank ever made. It scored 0.2 on `demo` -- identical to the
no-memory control, for the same reason (the single negative query) -- and that number was reported
as a result repeatedly.

A plausible number is worse than an error: it is indistinguishable from a real result. This guard
makes a whole-run empty retrieval refuse.
"""
from __future__ import annotations

import pytest

from memrank.runner import EmptyRetrieval, _assert_retrieved_something


class _Adapter:
    def __init__(self, name="mem0", context_budget="matched"):
        self.name = name
        self.context_budget = context_budget


def _rows(*retrieved_counts):
    return [{"retrieved": [{"id": str(i)}] * n} for i, n in enumerate(retrieved_counts)]


def test_a_run_that_retrieved_nothing_at_all_fails():
    with pytest.raises(EmptyRetrieval, match="retrieved 0 documents across all 5 queries"):
        _assert_retrieved_something(_Adapter(), _rows(0, 0, 0, 0, 0))


def test_one_non_empty_query_is_enough():
    """A per-query miss is ordinary; the guard is about a whole run returning nothing."""
    _assert_retrieved_something(_Adapter(), _rows(0, 0, 3, 0, 0))


def test_the_no_memory_arm_is_exempt():
    """`none` retrieves nothing by design -- that is the control it exists to provide."""
    _assert_retrieved_something(_Adapter("none", context_budget="none"), _rows(0, 0, 0))


def test_an_adapter_without_a_context_budget_is_still_guarded():
    """Defaulting to "matched" means a new adapter is protected before anyone remembers to opt in."""
    class _Bare:
        name = "bare"

    with pytest.raises(EmptyRetrieval):
        _assert_retrieved_something(_Bare(), _rows(0, 0))


def test_an_empty_query_set_is_not_a_failure():
    """Nothing was asked, so nothing retrieved proves nothing."""
    _assert_retrieved_something(_Adapter(), [])


def test_the_message_names_the_likely_cause():
    """The real cause was a silently-ignored search parameter; the next one probably is too."""
    with pytest.raises(EmptyRetrieval, match="search contract"):
        _assert_retrieved_something(_Adapter(), _rows(0))
