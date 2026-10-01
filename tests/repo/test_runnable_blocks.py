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
"""Every Python block in every public document, executed rather than trusted.

`tests/repo/test_doc_links.py` proves a document's links resolve and
`tests/repo/test_docs_teach_the_current_cli.py` proves its shell blocks name commands that still
exist. Neither runs a line of the Python a reader is invited to copy. Nothing did: the README's
blocks were executed by hand when it was written, and the 2026-09-16 and 2026-09-21 usability
sessions both produced the same failure -- a block copied from the front page that raised.

So each block runs here, in a fresh subprocess, against a temporary working directory, and a
non-zero exit fails. That is the whole contract, and there is exactly one way out of it.

**A declared marker.** A block that is not meant to execute says so in the document, on the
nearest non-blank line above its opening fence::

    <!-- runnable: no -- an interface sketch; the ABC is the subject, not a script -->

The comment is invisible in rendered markdown and the reason is free text, because the reasons
are not one kind. Two kinds are expected and both are declared the same way: a FRAGMENT, shown
inside a method body or naming a class the reader is about to write, which could never run; and a
LIVE DEPENDENCY -- a backend, a key, the network -- which could run somewhere but not here. The
second is the case `tests/withheld.py` already answers for an absent engine image, and this marker
is the same move for a document: the block SKIPS with its reason stated, rather than an assertion
being weakened until it holds either way.

There is deliberately no ratchet of tolerated failures beside it, the way `test_doc_links.py`
keeps one for dangling links. A link that dangles is inert; a block that raises is a degraded mode
this repository refuses, and pinning one would let a document go on inviting a reader to copy a
traceback. A block either runs or says why it does not, and a marker is a claim about what the
block IS -- so declaring one on a block that was written as a script and merely broke is a lie the
reviewer of that document is being asked to catch, not a mechanism this file offers.

**What is deliberately not here.** No block's stdout is compared to the `console` fence beside it.
That check is worth having and is a separate deliverable, and it cannot be built on top of this
one as written: `README.md`'s `print(result)` renders memrank's own measured latency, so its
output is legitimately different on every run. Nothing in this file asserts on a duration, and
nothing here is given a timeout -- a wall-clock bound is the timing-dependent logic this
repository refuses, and the blocks it covers are offline and finish in tenths of a second.

Shell blocks are not executed. They stay `test_docs_teach_the_current_cli.py`'s.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

# `tomllib` is stdlib only from 3.11 and `pyproject.toml` declares a 3.10 floor. The shim lives in
# `test_public_boundary.py`, beside the manifest reading it exists for; importing it is how this
# file stays a collection error on neither interpreter without a second copy of the try/except.
from tests.repo.test_public_boundary import PUBLIC, tomllib
from tools import manifest as mf

ROOT = Path(__file__).resolve().parents[2]

#: A fenced block's delimiter and its info string. Same rule as `test_doc_links.py`: the run
#: length and the character are both captured, because a closing fence must use the same character
#: and be at least as long -- which is what lets a ``` sit inside a ```` block without ending it.
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)")

#: The info strings this file treats as Python. `py` appears nowhere in the tracked corpus today;
#: it is accepted so that writing it does not silently exempt a block from the whole check.
PYTHON = ("python", "py", "python3")

#: The opt-out, and the reason is required: a marker with an empty reason is not a declaration.
MARKER = re.compile(r"^\s*<!--\s*runnable:\s*no\s*--\s*(?P<reason>.*?)\s*-->\s*$")

@dataclass(frozen=True)
class Block:
    """One fenced Python block, and what the document said about it."""

    path: str
    line: int          #: 1-based line of the OPENING fence, which is what an editor jumps to
    source: str
    reason: str | None  #: the declared marker's reason, or None when the block must run

    @property
    def id(self) -> str:
        return f"{self.path}:{self.line}"


def _public_markdown() -> tuple[str, ...]:
    """Tracked markdown that the published tree actually carries.

    Read at import, because the blocks are parameters and pytest wants those at collection. The
    classification is `tools/manifest.py`'s, the same one `test_public_boundary.py` and the
    projector use, so this corpus is identical here and in a projected tree.
    """
    listed = subprocess.run(["git", "ls-files"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    rules = tomllib.loads((ROOT / "publish.toml").read_text(encoding="utf-8"))
    return tuple(p for p in listed
                 if p.endswith(".md") and mf.classify(p, rules)[0] == PUBLIC
                 and (ROOT / p).is_file())


def _declared_reason(lines: list[str], fence_index: int) -> str | None:
    """The marker on the nearest non-blank line above the fence, if there is one.

    Nearest rather than immediately-above because a blank line between a comment and a fence is
    ordinary markdown and a reader will write one without meaning anything by it.
    """
    for candidate in reversed(lines[:fence_index]):
        if candidate.strip():
            found = MARKER.match(candidate)
            return found.group("reason") if found else None
    return None


def blocks_in(path: str, text: str) -> list[Block]:
    """Every fenced Python block in one document, with its declared reason."""
    lines = text.splitlines()
    found: list[Block] = []
    closing: str | None = None
    language, body, opened = "", [], 0
    for index, line in enumerate(lines):
        fence = FENCE.match(line)
        if closing is None:
            if fence:
                closing, language, body, opened = fence.group(1), fence.group(2), [], index
            continue
        if fence and fence.group(1)[0] == closing[0] and len(fence.group(1)) >= len(closing):
            if language in PYTHON:
                found.append(Block(path, opened + 1, "\n".join(body) + "\n",
                                   _declared_reason(lines, opened)))
            closing = None
        else:
            body.append(line)
    return found


def _collect() -> list[Block]:
    return [block
            for path in _public_markdown()
            for block in blocks_in(path, (ROOT / path).read_text(encoding="utf-8"))]


#: Collected once at import. `DOCUMENTS` is what the scan reached, which the guard below asserts on.
DOCUMENTS = _public_markdown()
BLOCKS = _collect()


def execute(block: Block) -> subprocess.CompletedProcess[str]:
    """Run one block as its own program, in its own empty directory.

    A fresh interpreter per block is the point: a block in a document is read on its own, so it is
    run on its own, and no block may be made to pass by something an earlier one left behind. The
    working directory is temporary for the same reason -- a block that writes a file writes it
    where nothing else will find it, and the repository is not touched.
    """
    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "block.py"
        script.write_text(block.source, encoding="utf-8")
        return subprocess.run([sys.executable, str(script)], cwd=workdir,
                              capture_output=True, text=True)


# --------------------------------------------------------------------------- the blocks

@pytest.mark.parametrize("block", BLOCKS, ids=lambda b: b.id)
def test_the_block_runs(block: Block):
    """Exit 0, or a declared skip. There is no third outcome."""
    if block.reason is not None:
        pytest.skip(f"{block.id} declares it is not run: {block.reason}")

    completed = execute(block)
    assert completed.returncode == 0, (
        f"{block.id} exits {completed.returncode}. A reader who copies it gets this:\n"
        + "\n".join(completed.stderr.strip().splitlines()[-12:]))


def test_every_declared_marker_states_a_reason():
    """A marker is a declaration, and a declaration with nothing in it is an exemption.

    The regex already requires the `--`, so this catches the remaining shape: `<!-- runnable: no --
    -->`, which parses and says nothing. It is asserted separately from the run so that the failure
    names the empty declaration rather than a block that mysteriously skipped.
    """
    empty = [block.id for block in BLOCKS if block.reason is not None and not block.reason]
    assert empty == [], f"marker(s) with no reason: {empty}"


# --------------------------------------------------------------------------- guards the guard

def test_the_scan_reaches_the_documents_that_matter():
    """An extraction that silently matched nothing would make every test above pass.

    The floor is on Python blocks rather than documents, because `docs/install.md` and
    `docs/local-development.md` are wholly shell and a count of documents would not notice the
    extractor losing every fence it was built for. Forty-three blocks stand at the anchor; a
    floor of twenty-five catches a broken extractor without pinning a number the plan is moving.
    """
    assert DOCUMENTS, "the scan found no public markdown at all"
    for must_cover in ("README.md", "docs/systems.md", "docs/evaluations.md"):
        assert must_cover in DOCUMENTS, f"{must_cover} fell out of the corpus"
    assert len(BLOCKS) >= 25, f"only {len(BLOCKS)} Python block(s) found -- extraction is broken"


def test_the_corpus_still_executes_something():
    """The markers' own guard. Every block acquiring a marker would empty the check in silence."""
    executed = [block.id for block in BLOCKS if block.reason is None]
    assert len(executed) >= 20, (
        f"only {len(executed)} block(s) are actually executed -- the rest declare a marker, which "
        f"turns this file into a no-op:\n  " + "\n  ".join(sorted(executed)))


def test_the_readme_leads_with_a_block_that_runs():
    """The finding this test exists for, asserted where it landed.

    F1 of the 2026-09-21 session is that the README's FIRST Python block did not run after the
    README's own install command. A floor over the whole corpus would pass with that block marked
    and the rest green, which is the one outcome that would reproduce the session exactly.

    The README's quick start is now shell -- the one-command install and `memrank run` -- so it
    may carry no Python block at all. When it carries one, the first must still run.
    """
    first = next((b for b in BLOCKS if b.path == "README.md"), None)
    if first is None:
        return
    assert first.reason is None, (
        f"the README's first Python block declares a marker ({first.reason}) -- the front page's "
        f"way in is not allowed to be a block nobody runs")


def test_two_consecutive_runs_agree():
    """Determinism, stated as the property this file actually claims.

    It claims the VERDICT is stable, not the output: `README.md`'s `print(result)` renders
    memrank's own measured latency, so identical stdout is not a property a correct block has and
    asserting it would be the wall-clock dependence this repository refuses. One executed block is
    run twice; running the whole corpus twice would double the suite to witness the same thing.
    """
    block = next(b for b in BLOCKS if b.reason is None)
    assert execute(block).returncode == execute(block).returncode == 0, (
        f"{block.id} does not agree with itself across two runs")


# --------------------------------------------------------------------------- the extractor

def test_the_extractor_reads_fences_the_way_markdown_does():
    """Pins the fence rule, including the three cases that fail OPEN when got wrong.

    An unclosed fence, a fence closed by the wrong character, and a nested shorter fence each end
    with the extractor resuming inside quoted material -- which is how a block from a code sample
    inside a tutorial gets executed as if it were the tutorial's own.
    """
    def ids(text: str) -> list[int]:
        return [b.line for b in blocks_in("d.md", text)]

    assert ids("```python\nx = 1\n```") == [1]
    assert ids("~~~python\nx = 1\n~~~") == [1]
    assert ids("   ```python\nx = 1\n   ```") == [1]
    assert ids("```bash\nls\n```") == []
    assert ids("````markdown\n```python\nx = 1\n```\n````") == []
    assert ids("```python\nx = 1") == [], "an unclosed fence must not yield a block"
    assert ids("```python\n~~~\nx = 1\n```") == [1]
    assert blocks_in("d.md", "```python\nx = 1\n```")[0].source == "x = 1\n"


def test_the_extractor_reads_the_marker():
    """Pins the opt-out, which is the one thing here that can turn the check off."""
    def reason(text: str) -> str | None:
        return blocks_in("d.md", text)[0].reason

    assert reason("```python\nx = 1\n```") is None
    assert reason("<!-- runnable: no -- a sketch -->\n```python\nx = 1\n```") == "a sketch"
    assert reason("<!-- runnable: no -- a sketch -->\n\n```python\nx = 1\n```") == "a sketch"
    assert reason("<!-- runnable: no -- a sketch -->\ntext\n```python\nx = 1\n```") is None
    assert reason("<!-- runnable: no -->\n```python\nx = 1\n```") is None
    assert reason("<!-- runnable: no --  -->\n```python\nx = 1\n```") == ""
    assert reason("<!-- runnable: yes -->\n```python\nx = 1\n```") is None
