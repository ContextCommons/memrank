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
"""Comments and docstrings that cite a document the reader will never have.

`tests/repo/test_doc_links.py` asks this of prose links and `tests/repo/test_public_boundary.py`
asks it of imports. Neither can see the third way the boundary leaks, which is the commonest:
a comment in a shipping file that credits its reasoning to an internal audit, direction brief or
plan. The link checker does not look inside source files, and the import graph does not care what
a comment says, so 129 of them accumulated unopposed.

None is a secret and none is a URL, so the publication gate passes every one. What they are is
dangling pointers. An outsider reading the projected tree is told the reason exists and is given a
filename that resolves to nothing in the repository they have -- which costs them the trip and
teaches them that the citations in this codebase are not for them.

So the rule this file holds is not "cite less". It is that a shipping comment states its reason
rather than delegating it: where an outsider needs the WHY, the WHY is in the comment; where they
do not, the pointer goes. The record is not lost -- it is in the commit that made the change, in
the internal tree, and in the ticket.

THE COUNT IS ZERO AND IT STAYS ZERO. This is not a ratchet with a pinned list, because a list of
tolerated citations is how the previous 129 were tolerated. Scope is exactly what ships: paths
`publish.toml` classifies public, which is why the check runs unchanged on the projection.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

from tools import manifest as mf

ROOT = Path(__file__).resolve().parents[2]

#: The three shapes a citation takes, assembled from fragments so that this file -- which ships --
#: does not match its own patterns and report itself.
_INTERNAL_TREE = "docs" + "-internal/"
_DATED = r"\d{4}-\d{2}-\d{2}"

#: A path into the internal documentation tree. The trailing character class is what distinguishes
#: naming a DOCUMENT from naming the directory: the prefix on its own is a fact about the
#: repository layout that `publish.toml` states outright, and the manifest has to be able to say it.
CITATIONS = (
    re.compile(re.escape(_INTERNAL_TREE) + r"[A-Za-z0-9_]"),
    re.compile(r"\b" + _DATED + r"-audit-"),
    re.compile(r"\b(?:directions|plans)/" + _DATED),
)

#: Extensions whose comments this reads. Markdown is absent deliberately: it has no comment syntax,
#: and a link in a published document to one that does not ship is already
#: `tests/repo/test_doc_links.py`'s assertion, pinned there in both directions.
COMMENTED = (".py", ".yaml", ".yml", ".toml", ".sh", ".cfg", ".ini")


def _python_prose(text: str) -> list[tuple[int, str]]:
    """Every comment and docstring line in one Python source, as `(line number, that line)`.

    Two passes because the two live in different places. `tokenize` has the comments and does not
    distinguish a docstring from any other string; `ast` knows which strings are docstrings and
    has discarded the comments by the time it runs.

    A docstring is returned line by line rather than whole, so that what is reported is the line
    carrying the citation and not the paragraph around it.
    """
    found = [(token.start[0], token.string)
             for token in tokenize.generate_tokens(io.StringIO(text).readline)
             if token.type == tokenize.COMMENT]

    tree = ast.parse(text)
    holders = [tree] + [node for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for node in holders:
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        found += [(node.body[0].lineno + offset, line)   # body[0] is the docstring expression
                  for offset, line in enumerate(doc.splitlines())]
    return found


def _hash_comments(text: str) -> list[tuple[int, str]]:
    """Whole-line `#` comments, for the formats that have nothing richer.

    Whole-line only, and not the trailing comment after a value: a `#` inside a quoted string is
    a `#` inside a quoted string, and no parser cheap enough to be worth having here can tell the
    two apart across YAML, TOML and shell alike. The trailing case is not where prose goes.
    """
    return [(number, line) for number, line in enumerate(text.splitlines(), 1)
            if line.lstrip().startswith("#")]


def prose_citations(path: str) -> list[str]:
    """`path:line: excerpt` for every internal-document citation in its comments or docstrings."""
    source = ROOT / path
    if not source.is_file():
        return []
    text = source.read_text(encoding="utf-8", errors="replace")
    reader = _python_prose if path.endswith(".py") else _hash_comments
    try:
        prose = reader(text)
    except (SyntaxError, tokenize.TokenError) as exc:            # unparseable is not this test's
        return [f"{path}: cannot read: {exc}"]                   # business to diagnose

    return [f"{path}:{number}: {line.strip()[:110]}" for number, line in prose
            if any(pattern.search(line) for pattern in CITATIONS)]


def test_no_public_comment_cites_an_internal_document():
    """Zero, over exactly the set of files that ships.

    The path set comes from git and the classification from the manifest, both of which the
    published tree carries -- so the projection checks this property on its own bytes rather than
    inheriting a claim made upstream. There it resolves to every tracked path, since everything
    there classifies public.
    """
    shipping = mf.public_paths(mf.load(ROOT), mf.tracked(ROOT))
    public = [path for path in shipping if path.endswith(COMMENTED)]
    assert public, "no public commentable file found -- the scan matched nothing"

    citations = [citation for path in public for citation in prose_citations(path)]

    assert citations == [], (
        f"{len(citations)} comment(s) or docstring(s) in shipping files cite a document the "
        f"reader does not have. State the reason in the comment, or drop the pointer:\n  "
        + "\n  ".join(citations[:40]))


def test_the_scan_reads_prose_and_only_prose():
    """The extractor itself, pinned: a comment and a docstring count, a data literal does not.

    The distinction is load-bearing rather than fussy. `tests/repo/test_doc_links.py` holds a
    ratchet of known-broken links keyed by the document each lives in, and most of those documents
    are internal -- that set is the test's data, it must name them to do its job, and no reader is
    being sent anywhere by it.
    """
    cited = _INTERNAL_TREE + "plans/2026-01-01-a-plan.md"
    assert _python_prose(f"# see {cited}\n") == [(1, f"# see {cited}")]

    docstring = _python_prose(f'"""See {cited}."""\n')
    assert docstring == [(1, f"See {cited}.")]

    assert _python_prose(f'KNOWN = ("{cited}",)\n') == []
    assert _hash_comments(f"  # see {cited}\n") == [(1, f"  # see {cited}")]
    assert _hash_comments(f'known = "{cited}"  # a value\n') == []


def test_the_directory_name_alone_is_not_a_citation():
    """`publish.toml` has to be able to state the prefix it classifies, and so does prose about
    the boundary. What is refused is naming a document inside it."""
    assert not any(p.search(f'"{_INTERNAL_TREE}",') for p in CITATIONS)
    assert any(p.search(_INTERNAL_TREE + "VISION.md") for p in CITATIONS)
