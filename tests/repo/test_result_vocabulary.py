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
"""A result's identity keys are named in one module, and this enumerates every other one.

``adapter`` and ``benchmark`` are the stored artifact's two identity keys. Nine modules read
them by literal subscript -- the arena curator, the compare renderer, the leaderboard
normalizer, the run registry, the sweep's summary row, the ops report, the version diff, the
question drilldown and the API projection -- so the artifact vocabulary was interpreted in
nine places and renaming it was nine edits with no list of which nine. Two of those modules
translated the pair into the outward ``target``/``eval`` vocabulary independently, which is
two places a rename had to find and one of them undocumented.

So this is an ENUMERATION rather than a set of fixes: it walks every module under
``memrank/`` and fails when one names either key, which is what makes a NEW surface reaching
for the raw key fail here rather than silently becoming the tenth. Reads go through
:mod:`memrank.evaluation.result`'s four accessors, and the one translation into the outward
vocabulary is :func:`memrank.api.run_projection.identity_of`.

**The allowlist below is not a list of exemptions.** Every entry is a dict that is NOT a
result: a run heartbeat, an org API payload, a compare artifact's own metadata, a leaderboard
board, a target manifest. Those carry a key spelled the same and mean something else, and a
static scan cannot tell them apart -- so each is named with the expression it reads, which is
what makes the claim checkable by someone reading this file.
"""
from __future__ import annotations

import ast
from pathlib import Path

#: The two keys this guards. Spelled here rather than imported from the private constants in
#: `memrank.evaluation.result`, so that renaming the artifact's keys does not also rename what
#: this test looks for -- after a rename this must still prove the OLD names are gone.
_IDENTITY_KEYS = ("adapter", "benchmark")

#: The one module allowed to name them: the artifact schema's owner, where the accessors live.
_SCHEMA_OWNER = "memrank/evaluation/result.py"

#: ``(module, expression)`` pairs that read a key spelled like a result's from a dict that is
#: not a result. The expression is the receiver as it appears in the source, so a reader can
#: check each claim, and so a new read of a genuine result dict is not covered by an entry
#: written for something else.
_NOT_A_RESULT = {
    # The compare ARTIFACT's own metadata -- one benchmark for the whole artifact, beside
    # `rows`/`model`/`run_id`. Its rows are results; it is not one.
    ("memrank/analysis/compare.py", "result"),
    ("memrank/leaderboard/aggregate.py", "m"),
    ("memrank/arena/curation_policy.py", "meta"),
    # `comparability_warnings`'s own return value, built a few lines above.
    ("memrank/analysis/compare_versions.py", "comparison"),
    # A leaderboard board: a published grouping that names the benchmark it ranks.
    ("memrank/leaderboard/history.py", "board"),
    # A target manifest's declared interface and its top-level adapter name -- what to build,
    # not what was measured.
    ("memrank/targets/manifest.py", "interface"),
    ("memrank/targets/manifest.py", "data"),
    # The run heartbeat (`status.json`) and the org API's run payload. Both name the run's
    # benchmark; neither is a cell's result, and the push reads them side by side with one.
    ("memrank/cli/monitor.py", "data"),
    ("memrank/cli/runs.py", "data"),
    ("memrank/cli/runs.py", "payload"),
    ("memrank/cli/runs_show.py", "payload"),
    ("memrank/runs/push.py", "data"),
    ("memrank/cli/watch.py", "data"),
    ("memrank/cli/watch.py", "record"),
}


def _reads_in(path: Path) -> list[tuple[int, str, str]]:
    """Every read of an identity key in one module, as ``(line, receiver, key)``.

    Both spellings a reader can use: ``cell["adapter"]`` and ``cell.get("adapter")``. Only
    loads -- a dict LITERAL whose key is ``"adapter"`` is the module writing its own output
    shape, which is a different question from interpreting a result's vocabulary.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    reads: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in _IDENTITY_KEYS):
            reads.append((node.lineno, ast.unparse(node.value), node.slice.value))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in _IDENTITY_KEYS):
            reads.append((node.lineno, ast.unparse(node.func.value), node.args[0].value))
    return reads


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2] / "memrank"


def test_only_the_schema_owner_reads_a_result_identity_key_by_literal():
    root = _package_root().parent
    offenders = []
    for path in sorted(_package_root().rglob("*.py")):
        module = path.relative_to(root).as_posix()
        if module == _SCHEMA_OWNER:
            continue
        for line, receiver, key in _reads_in(path):
            if (module, receiver) in _NOT_A_RESULT:
                continue
            offenders.append(f"{module}:{line} reads {receiver}[{key!r}]")
    assert offenders == [], (
        "these read a result's identity key by literal, which puts the artifact vocabulary "
        "back in more than one place:\n  " + "\n  ".join(offenders) + "\n\n"
        "Read it through `memrank.evaluation.result`'s accessors instead -- adapter_of / "
        "benchmark_of when a missing key is a failure, adapter_if_present / "
        "benchmark_if_present when it is not. To translate into the outward target/eval "
        "vocabulary, call `memrank.api.run_projection.identity_of`. If the dict is NOT a "
        "result -- a run heartbeat, an org payload, artifact metadata, a board, a manifest "
        "-- add it to _NOT_A_RESULT in this file with a comment saying which.")


def test_the_allowlist_has_no_entry_that_nothing_reads():
    """An allowlist that outlives the read it excused starts excusing the next one.

    Each entry claims a specific module reads a specific non-result dict. When that read goes
    away the entry must go with it, or it silently covers whatever is next given that
    variable name in that module.
    """
    root = _package_root().parent
    live = set()
    for path in sorted(_package_root().rglob("*.py")):
        module = path.relative_to(root).as_posix()
        for _, receiver, _key in _reads_in(path):
            live.add((module, receiver))
    # Only entries whose module is present in the tree this runs in. The published tree is a
    # PROJECTION of this repository (`tools/project.py`), and it drops the internal-only
    # modules several entries name -- analysis/, leaderboard/, arena/curation_policy.py. An
    # entry whose file is absent is neither stale nor excusing a read: there is no module for
    # it to cover. Staleness is only decidable where the module exists.
    stale = sorted(entry for entry in _NOT_A_RESULT
                   if entry not in live and (root / entry[0]).exists())
    assert stale == [], (
        f"_NOT_A_RESULT entries that no longer match any read: {stale}. Remove them -- an "
        f"entry kept past its read excuses the next read that happens to share the name.")
