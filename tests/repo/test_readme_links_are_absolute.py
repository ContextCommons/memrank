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
"""The README is read somewhere its relative links do not resolve.

`tests/repo/test_doc_links.py` asks whether a link names something this repository has. That is the
right question for every page but one. `README.md` is also the project description on PyPI, which
renders the markdown and resolves `docs/install.md` against `pypi.org` -- where no such page exists.
Every link on the 0.4.2 page was dead for that reason (ATO-2258), and a release's description is
immutable, so the fix can only ship with the next version.

So the README's links are absolute URLs into the public repository, and the other pages' stay
relative because they are only ever read on GitHub. Three things have to hold for that to keep
working, and each is one assertion below:

1. no link in the README is relative -- the defect itself, which is what a reader meets on PyPI;
2. a link into the public repository uses the `blob/main/` or `tree/main/` form, so it is a URL
   GitHub actually serves rather than a plausible-looking one;
3. the path it names is in the PROJECTION -- the published tree the URL resolves against. A path
   that exists here and is classified internal would be a live link to a 404, which no check
   confined to this repository can see.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.repo.test_doc_links import LINK, _strip_fenced_blocks
from tools import manifest as mf

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

#: The public repository, and the two forms GitHub serves a path in. A file takes `blob/`, a
#: directory `tree/`; GitHub redirects between them, but a URL that has to be redirected is one
#: this check would rather catch than follow.
REPO = "https://github.com/ContextCommons/memrank/"
BLOB, TREE = f"{REPO}blob/main/", f"{REPO}tree/main/"

#: A link into the repository must be one of those two. Anything else under `REPO` -- a `master`
#: branch, a `raw.githubusercontent.com` host, a bare `.../memrank` -- is out of contract.
FORMS = (BLOB, TREE)

#: Repository pages that are not files in the tree -- the issue tracker the README's Help section
#: names, and the public CI workflow its status badge links to. GitHub serves both as absolute URLs,
#: so they resolve on PyPI, and the projection has nothing to check them against.
PAGES = (f"{REPO}issues", f"{REPO}actions/workflows/")

#: Guard the guard. The README carried 20 links when this landed; a floor well under that catches
#: an extraction that matched nothing without pinning the count.
LINK_FLOOR = 12


def _links(body: str) -> list[str]:
    """Every inline link target in the README, fenced blocks excluded.

    The quick start quotes a URL inside a ```text``` block for a reader to paste to their agent.
    It is prose in that block rather than a link of the README's own, and `_strip_fenced_blocks`
    is what keeps the two apart -- the same rule `test_doc_links.py` applies everywhere else.
    """
    return [match.group(1) for match in LINK.finditer(_strip_fenced_blocks(body))]


def _defects(body: str, published: frozenset[str]) -> list[str]:
    """Every link in `body` that would not resolve for a reader on PyPI, with the reason."""
    found = []
    for raw in _links(body):
        if not raw.startswith(("http://", "https://")):
            found.append(f"{raw}: relative -- PyPI resolves it against pypi.org")
            continue
        if not raw.startswith(REPO):
            continue                      # an external site: nothing here to check
        if raw.startswith(PAGES):
            continue
        if not raw.startswith(FORMS):
            found.append(f"{raw}: not a {BLOB} or {TREE} URL")
            continue
        blob = raw.startswith(BLOB)
        path = raw[len(BLOB if blob else TREE):].split("#")[0]
        if blob and path not in published:
            found.append(f"{raw}: {path} is not a file in the published tree")
        elif not blob and not any(p.startswith(f"{path}/") for p in published):
            found.append(f"{raw}: {path} is not a folder in the published tree")
    return found


@pytest.fixture(scope="module")
def published() -> frozenset[str]:
    """The projected tree: what the URLs in the README actually resolve against.

    `tools/manifest.py` is the one implementation of the rule, the same one `tools/project.py`
    projects with and `test_public_boundary.py` checks -- so this test and the published tree
    cannot disagree about which paths a reader can reach.
    """
    return frozenset(mf.public_paths(mf.load(ROOT), mf.tracked(ROOT)))


def test_every_readme_link_is_absolute_into_the_public_repository(published):
    """The assertion the PyPI page needed and did not have."""
    body = README.read_text(encoding="utf-8")
    assert len(_links(body)) >= LINK_FLOOR, "the link extraction found almost nothing"

    defects = _defects(body, published)
    assert defects == [], (
        f"{len(defects)} link(s) in README.md would not resolve on PyPI:\n  "
        + "\n  ".join(defects))


def test_a_relative_link_injected_into_the_readme_is_caught(published):
    """Guards the guard, at the point of consequence rather than on a synthetic string.

    The check above passes on a README with no links at all, and would pass on one whose links it
    silently failed to read. This injects the exact defect ATO-2258 records -- into the real
    document, so an extraction that stopped reaching the README fails here too.
    """
    body = README.read_text(encoding="utf-8")
    injected = body + "\n[install](docs/install.md)\n"

    assert _defects(injected, published) == _defects(body, published) + [
        "docs/install.md: relative -- PyPI resolves it against pypi.org"]


def test_a_link_at_a_path_the_projection_drops_is_caught(published):
    """The defect only the projection can see: a live URL to a 404.

    `docs-internal/` is never published, so a README link into it resolves here and nowhere a
    reader can reach. Pinning it as a case keeps the published-tree test from degrading into a
    check that the file exists on this disk.
    """
    internal = f"[notes]({BLOB}docs-internal/VISION.md)"

    assert _defects(internal, published) == [
        f"{BLOB}docs-internal/VISION.md: docs-internal/VISION.md is not a file in the "
        f"published tree"]
