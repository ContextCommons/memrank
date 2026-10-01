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
"""Builtin targets a tree may not carry, and the one way a test says so.

`mem0` and `supermemory` name engine images no outsider can pull -- one is a private fork, the
other wraps a closed vendor binary we may not redistribute (`docs/misc/engine-images.md`). The
published package therefore ships neither manifest, and a test written against one has nothing to
resolve.

Such a test SKIPS, with the reason stated, exactly the way `tests/live/` skips when a backend is
not running: the condition is a probe of what is actually present, the assertions are untouched,
and the same test run against a tree that DOES carry the manifest still enforces every one of them.
What is not acceptable here is the alternative that was considered and rejected -- weakening an
assertion so it holds either way, which would retire the check in both trees at once.

Two calls, and the difference between them is when the answer is needed:

- :func:`require` inside a test body, for the usual case.
- :func:`skip_without` as a `pytest.mark.skipif` argument, for a module or a parametrised case
  whose ref is decided at collection time.

Neither takes a hardcoded list. The condition is "is this ref resolvable here", so a target that
later becomes publishable stops being skipped by the act of shipping its manifest, with no edit
here -- and a target that vanishes by accident is reported as skipped rather than passing silently,
which is why `tests/targets/test_catalog.py` still asserts the catalog's contents outright.
"""
from __future__ import annotations

import pytest

from memrank.targets import catalog

#: What a reader sees, and it names the file that explains the whole of it.
REASON = ("target {ref!r} is not in this tree's catalog: its engine image is not publicly "
          "obtainable, so the manifest is withheld and the backend is self-provisioned "
          "(docs/misc/engine-images.md)")


def present(ref: str) -> bool:
    """True when `ref`'s manifest is resolvable in this tree.

    The BASE is what is tested: `mem0:matched` is written by a test into its own config dir and
    inherits `from: mem0`, so it resolves only where the builtin it derives from does.
    """
    return ref.split(":", 1)[0] in set(catalog.list_targets())


def require(*refs: str) -> None:
    """Skip the calling test unless every `ref` resolves here."""
    for ref in refs:
        if not present(ref):
            pytest.skip(REASON.format(ref=ref))


def skip_without(*refs: str) -> bool:
    """`True` when any `ref` is missing -- the condition for a `skipif` mark."""
    return not all(present(ref) for ref in refs)
