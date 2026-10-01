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
"""The public/private boundary, checked rather than described.

`publish.toml` says which paths ship. Until this file existed, that was a claim in a document,
and the failure mode of a claim in a document is specific and known: someone adds an internal
module, nobody classifies it, and the mistake surfaces as a leak in a public repository rather
than as a red test here.

So there are three assertions, and they correspond to the three ways the boundary can be wrong:

1. a tracked path matches no rule, or matches two that disagree -- the manifest has gone stale
   against the tree;
2. a public module imports an internal one -- the projection would not import;
3. the closure of the published entry points reaches outside the public set -- the projection
   would not run.

There is a fourth, and it is the one none of those three can see: material that is not tracked at
all. `git ls-files` does not list it, so every assertion above steps over it, and the projector
reads committed objects -- which makes the exposure window exactly one moment wide, the commit that
first tracks the file. `test_no_untracked_path_would_be_published` closes that window by asking for
the decision one step earlier, while the material is still undecided.

The manifest is deliberately a PREFIX list with an exceptions table, and the exceptions table is
deliberately written out in full. `publish.toml` says why.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from tools import manifest as mf

# `tomllib` is stdlib only from 3.11, and `pyproject.toml` declares `requires-python = ">=3.10"`.
# Without this shim both of the tests that hold the public/private boundary raise at IMPORT on the
# declared floor -- a collection error in exactly the two files that are supposed to fail loudly.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "publish.toml"

#: The three published entry points. `memrank.runner` is the `memrank` command, `mcp_server` the
#: `memrank-mcp` command, and `evaluation.api` is the previous run -- the Python abstraction.
ENTRY_POINTS = ("memrank.runner", "memrank.mcp_server", "memrank.evaluation.api")

PUBLIC, INTERNAL = mf.PUBLIC, mf.INTERNAL


@pytest.fixture(scope="module")
def manifest() -> dict:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tracked() -> tuple[str, ...]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return tuple(out.stdout.splitlines())


@pytest.fixture(scope="module")
def undecided() -> tuple[str, ...]:
    """What `tracked` cannot see: present on disk, not ignored, and not known to git."""
    return mf.untracked(ROOT)


def _classify(path: str, manifest: dict) -> tuple[str | None, str]:
    """Thin alias. The rule itself lives in `tools/manifest.py`, which `tools/project.py` also
    reads -- one implementation, so the test and the tool cannot disagree about what ships."""
    return mf.classify(path, manifest)


def _public_paths(manifest: dict, tracked: tuple[str, ...]) -> frozenset[str]:
    return frozenset(mf.public_paths(manifest, tracked))


# --------------------------------------------------------------------------- assertion 1

def test_every_tracked_path_is_classified_exactly_once(manifest, tracked):
    """No unclassified file. A new internal module that nobody sorted fails here, not later."""
    unresolved = {p: why for p in tracked if (r := _classify(p, manifest))[0] is None
                  for why in (r[1],)}

    assert unresolved == {}, (
        f"{len(unresolved)} tracked path(s) publish.toml cannot classify. Add a prefix rule, or "
        f"an entry to [exceptions]:\n"
        + "\n".join(f"  {p}: {why}" for p, why in sorted(unresolved.items())[:20]))


def test_no_exception_names_a_path_that_is_gone(manifest, tracked):
    """The other direction of rot. An exception left behind after its file moved is a lie about
    the size of the remaining debt, and the phases are measured by that count.

    Skipped in a projected tree, where an INTERNAL exception is absent by design rather than by
    rot. The tell is whether any internal path is tracked at all: in this repository thousands
    are, in the published tree none is. Without this the guard fires on `examples/
    compare-am-mem0.py` the moment the projection is checked out -- which would make the boundary
    test the reason a clean public checkout goes red.
    """
    known = set(tracked)
    if not any(_classify(path, manifest)[0] == INTERNAL for path in tracked):
        pytest.skip("projected tree: internal exceptions are absent by design")
    stale = sorted(p for side in (INTERNAL, PUBLIC)
                   for p in manifest["exceptions"].get(side, []) if p not in known)

    assert stale == [], (
        f"publish.toml [exceptions] names {len(stale)} path(s) that are not tracked -- they were "
        f"moved or deleted without updating the manifest:\n  " + "\n  ".join(stale[:20]))


def test_no_untracked_path_would_be_published(manifest, undecided):
    """Undecided material under a public prefix, caught before the commit that publishes it.

    This was found once, before the split: `docs/pitch/` and `docs/research/` sat untracked under
    a now-public prefix, and would have classified PUBLIC the instant anyone staged them.
    Nothing would have objected -- a prefix rule matches a whole directory, so internal material
    lands inside a public tree by inheriting a classification that was never made about it.

    So the guard is deliberately about the DECISION, not about the content: no rule can tell an
    internal pitch deck from a public methodology note once both sit under `docs/`. What it can
    tell is that nobody has said which this is. Three answers satisfy it, and each records the
    decision somewhere durable:

    * it ships -- stage it, and the change is then visible in a diff someone reviews;
    * it never ships -- add it to `[exceptions] internal` in `publish.toml`, or move it under an
      internal prefix (`docs-internal/`, `tests/internal/`, `scripts/internal/`);
    * it is not repository material at all -- add it to `.gitignore`.

    A path the manifest cannot classify is left to
    `test_every_tracked_path_is_classified_exactly_once`, which catches it the moment it is
    staged; it cannot reach the published tree in the meantime, because the projector copies the
    public set rather than removing the internal one.
    """
    exposed = sorted(p for p in undecided if _classify(p, manifest)[0] == PUBLIC)

    assert exposed == [], (
        f"{len(exposed)} untracked path(s) would be published the moment they are staged, having "
        f"been classified by a prefix rule nobody applied to them deliberately. Stage them, or "
        f"classify them internal in publish.toml, or add them to .gitignore:\n  "
        + "\n  ".join(exposed[:20]))


# --------------------------------------------------------------------------- the import graph

def _module_of(path: str) -> str | None:
    """`memrank/cli/runs.py` -> `memrank.cli.runs`; a package `__init__` -> the package."""
    if not path.startswith("memrank/") or not path.endswith(".py"):
        return None
    dotted = path[:-3].replace("/", ".")
    return dotted[: -len(".__init__")] if dotted.endswith(".__init__") else dotted


@pytest.fixture(scope="module", autouse=True)
def _resolve_against_the_index(tracked):
    """Resolution consults `git ls-files`, not the filesystem.

    On a case-insensitive filesystem `Path.is_file()` answers yes for
    `memrank/targets/Manifest.py`, so `from memrank.targets import Manifest` -- a CLASS -- would
    resolve to `manifest.py` and be reported as a module. The index is case-exact, and it is also
    the only set the projection can actually copy.
    """
    _TRACKED.clear()
    _TRACKED.update(tracked)


#: Populated by the fixture above; the module-level helpers read it.
_TRACKED: set[str] = set()


def _path_of(module: str) -> str | None:
    """The tracked file a dotted `memrank.*` name resolves to, module before package."""
    stem = module.replace(".", "/")
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if candidate in _TRACKED:
            return candidate
    return None


def _imports_of(path: str) -> set[str]:
    """Every `memrank.*` file `path` imports, at any nesting -- lazy imports inside a function
    are still imports the projection has to satisfy.

    `from memrank.analysis import question_detail` resolves to the SUBMODULE when one exists and
    to the package `__init__` otherwise, which is the distinction the whole boundary turns on:
    `memrank/analysis/` is internal and one file inside it is not.
    """
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("memrank"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if not node.module.startswith("memrank"):
                continue
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return {resolved for name in found if (resolved := _path_of(name))}


def _closure(seeds: tuple[str, ...]) -> set[str]:
    """Every file reachable from `seeds` by following imports. The audit computed this once by
    hand; here it is the test."""
    reached, queue = set(), [p for m in seeds if (p := _path_of(m))]
    while queue:
        current = queue.pop()
        if current in reached:
            continue
        reached.add(current)
        queue.extend(_imports_of(current))
    return reached


# --------------------------------------------------------------------------- assertion 2

def test_no_public_module_imports_an_internal_one(manifest, tracked):
    """The audit's AST closure, turned into a guard.

    A public module that imports an internal one is a public tree that raises ImportError on
    first use, and nothing else in this repository would notice: the internal module is right
    there on the developer's path.
    """
    public = _public_paths(manifest, tracked)
    leaks = sorted((src, dst) for src in public if src.startswith("memrank/")
                   and src.endswith(".py")
                   for dst in _imports_of(src) if dst not in public)

    assert leaks == [], (
        f"{len(leaks)} public module(s) import internal code -- the public tree would not "
        f"import:\n  " + "\n  ".join(f"{s} -> {d}" for s, d in leaks[:20]))


# --------------------------------------------------------------------------- assertion 3

def test_the_published_closure_is_a_subset_of_the_public_set(manifest, tracked):
    """The projection must be able to RUN, not merely import.

    Assertion 2 asks whether any public file reaches internal code. This asks the complementary
    question from the other end: everything the three entry points actually need must be
    published. A module dropped from the manifest by mistake fails here even if no public file
    was left behind pointing at it.
    """
    public = _public_paths(manifest, tracked)
    missing = sorted(_closure(ENTRY_POINTS) - public)

    assert missing == [], (
        f"{len(missing)} file(s) reachable from {', '.join(ENTRY_POINTS)} are classified "
        f"internal -- `memrank --help` would not run in the public tree:\n  "
        + "\n  ".join(missing[:20]))


# --------------------------------------------------------------------------- assertion 4

#: Public test modules importing internal code. **Empty, and the assertion below keeps it empty.**
#: It held five entries when Phase 3 wrote it down: two importing `memrank.analysis.compare` at
#: module scope -- collection errors in a public checkout, which take unrelated tests down with them
#: -- and three reaching for `memrank.api` or `analysis.compare` from inside a single test. All five
#: were split along the boundary rather than deleted: the RULE each asserts is public and stayed,
#: the assertion that an internal surface applies it moved to `tests/internal/`.
#:
#: An entry added here is a decision to ship a public test suite that cannot run in the public
#: tree. There is no such thing as a temporary one.
KNOWN_TEST_LEAKS: set[str] = set()


def test_no_new_public_test_imports_internal_code(manifest, tracked):
    """A ratchet, not a clean bill of health.

    The public suite runs green in THIS repository because the internal code is still on the
    path. In a public checkout each of these is a collection error, and collection errors are not
    confined to the file that causes them. Pinning the set means the number can fall to zero
    without anyone re-deriving it, and cannot quietly rise in the meantime.
    """
    public = _public_paths(manifest, tracked)
    leaking = {src for src in public
               if src.startswith("tests/") and src.endswith(".py")
               and any(dst not in public for dst in _imports_of(src))}

    assert leaking <= KNOWN_TEST_LEAKS, (
        f"new public test(s) importing internal code: {sorted(leaking - KNOWN_TEST_LEAKS)}. "
        f"Split the test or move the file under tests/internal/.")
    assert (fixed := KNOWN_TEST_LEAKS - leaking) == set(), (
        f"{sorted(fixed)} no longer import internal code -- remove them from KNOWN_TEST_LEAKS so "
        f"the ratchet keeps its grip.")
