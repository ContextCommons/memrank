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
"""A link into a page lands on a heading that page has, and "section N" means section N.

`test_doc_links.py` strips `#fragment` before resolving a link, so an anchor that names no heading
passes it, and so does a section number in link text that names the wrong section: the README
sent the governance reader to "[SPEC.md section 5]" -- the evaluation contract -- when the
charter is section 7, on the front page and on PyPI, where a release's description cannot be
edited afterwards.

Three rules, over the public documents only (they must hold in the published tree too):

1. a fragment on a link to a markdown page -- relative, same-page, or the README's absolute
   `blob/main` form -- names a heading of that page, slugged the way GitHub slugs one;
2. a link whose text names "section N" carries a fragment, and the heading it lands on is
   numbered N -- so a renumbering, or a wrong number, fails rather than reading as a link;
3. an unlinked "section N" in a numbered specification (`SPEC.md`, `system-contract.md`) is a
   reference to its own section N, which must exist.
"""
from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

from tests.repo.prose import ROOT
from tests.repo.test_doc_links import FENCE, _strip_fenced_blocks
from tests.repo.test_runnable_blocks import DOCUMENTS

LINK = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]+)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
BLOB = "https://github.com/ContextCommons/memrank/blob/main/"
SECTION = re.compile(r"\bsection (\d+(?:\.\d+)*)\b", re.I)
NUMBERED_SPECS = ("docs/SPEC.md", "docs/system-contract.md")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercased, punctuation dropped, each space a hyphen."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def headings(path: str) -> list[str]:
    """The headings of one page, outside fenced blocks, in order."""
    text = _strip_fenced_blocks((ROOT / path).read_text(encoding="utf-8"))
    return [found.group(1) for line in text.splitlines()
            if not FENCE.match(line) and (found := HEADING.match(line))]


def anchors(path: str) -> dict[str, str]:
    """slug -> heading, with GitHub's `-1`, `-2` suffixes for a repeated heading."""
    seen: dict[str, str] = {}
    for heading in headings(path):
        base = candidate = slug(heading)
        suffix = 0
        while candidate in seen:
            suffix += 1
            candidate = f"{base}-{suffix}"
        seen[candidate] = heading
    return seen


def _target(source: str, raw: str) -> tuple[str, str] | None:
    """(repository path, fragment) for a link into a markdown page, or None."""
    if raw.startswith(BLOB):
        raw_path, _, fragment = raw[len(BLOB):].partition("#")
        return (raw_path, fragment) if raw_path.endswith(".md") else None
    if raw.startswith(("http://", "https://", "mailto:", "<")):
        return None
    raw_path, _, fragment = raw.partition("#")
    if not raw_path:
        return source, fragment
    resolved = (ROOT / Path(source).parent / urllib.parse.unquote(raw_path)).resolve()
    if resolved.suffix != ".md" or not resolved.is_file():
        return None
    return resolved.relative_to(ROOT).as_posix(), fragment


def link_offences(source: str, text: str) -> list[str]:
    found = []
    for label, raw in LINK.findall(_strip_fenced_blocks(text)):
        target = _target(source, raw)
        if target is None or not (ROOT / target[0]).is_file():
            continue
        page, fragment = target
        named = SECTION.search(label)
        if named and not fragment:
            found.append(f"{source}: [{label}] names a section but links no anchor")
            continue
        if not fragment:
            continue
        heading = anchors(page).get(urllib.parse.unquote(fragment))
        if heading is None:
            found.append(f"{source}: [{label}]({raw}) -- {page} has no heading #{fragment}")
        elif named and not heading.startswith(named.group(1)):
            found.append(f"{source}: [{label}] lands on '{heading}', not section {named.group(1)}")
    return found


def self_reference_offences(path: str) -> list[str]:
    numbers = {h.split()[0].rstrip(".") for h in headings(path) if h[:1].isdigit()}
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", _strip_fenced_blocks(
        (ROOT / path).read_text(encoding="utf-8")))
    return [f"{path} refers to its own section {n}, which it does not have"
            for n in SECTION.findall(text) if n not in numbers]


def test_every_anchor_lands_on_a_heading():
    found = [o for path in DOCUMENTS
             for o in link_offences(path, (ROOT / path).read_text(encoding="utf-8"))]
    assert not found, "\n".join(found)


def test_every_numbered_self_reference_exists():
    found = [o for path in NUMBERED_SPECS if path in DOCUMENTS
             for o in self_reference_offences(path)]
    assert not found, "\n".join(found)


def test_the_slug_is_githubs():
    assert slug("7. Governance -- the vendor-neutral charter") == (
        "7-governance----the-vendor-neutral-charter")
    assert slug("3. The `Document` shape") == "3-the-document-shape"


def test_the_scan_catches_the_reference_that_shipped_wrong():
    wrong = f"The commitments are in [SPEC.md section 5]({BLOB}docs/SPEC.md)."
    renumbered = (f"[SPEC.md section 5]({BLOB}docs/SPEC.md"
                  "#7-governance----the-vendor-neutral-charter).")
    right = (f"[SPEC.md section 7]({BLOB}docs/SPEC.md"
             "#7-governance----the-vendor-neutral-charter).")
    assert len(link_offences("README.md", wrong)) == 1
    assert len(link_offences("README.md", renumbered)) == 1
    assert link_offences("README.md", right) == []
    assert len(link_offences("README.md", f"[x]({BLOB}docs/SPEC.md#no-such-heading)")) == 1
