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
"""No live doc or script may teach a command the CLI refuses.

The cutover to the interface model was done in three passes and each one missed what the last
had already touched: `019d19e` repointed "every surviving script hint" at `memrank run --on
cloud`, `3ee60ce` renamed `run` to `submit` without revisiting those hints, and the `--slice`
retirement revisited neither. Sixteen call sites were still printing refused commands months
later, including the six `build-push*.sh` scripts that echo one to the terminal on success --
copy, paste, exit 2.

Per-file vigilance is what produced that, so the control is enumerated instead: this walks the
live corpus against :mod:`memrank.cli.retired`'s own tables, and a flag cannot be retired
without every instruction that teaches it failing here. The historical corpus is excluded by an
explicit list rather than by a pattern, so bringing a new directory under the rule -- or
declaring one a record of what was typed at the time -- is a deliberate edit to this file.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from memrank.cli.retired import RETIRED, RETIRED_FLAGS

ROOT = Path(__file__).resolve().parents[2]

#: Where instructions live. A reader is expected to be able to run what these say.
#: `.ipynb` is deliberately absent: the dataset notebooks load and inspect data and
#: invoke no CLI command, so there is nothing here for them to get wrong. Their prose
#: lives in notebooks/README.md, which IS scanned.
LIVE_GLOBS = ("*.md", "docs/**/*.md", "docs-internal/**/*.md", "deploy/**/*.md",
              "examples/**/*.md", "notebooks/**/*.md", "scripts/**/*.sh")

#: Excluded because they record what WAS typed, not what to type. Rewriting a command inside a
#: dated audit or a repro's narration would falsify the record it exists to preserve -- the point
#: of a 2026-08-12 evidence file is that this is the line that produced that number.
#:
#: `docs/SPEC.md` was here as "a frozen copy of the 2026-05-09 public spec" and is not exempt any
#: more. It is the specification a public reader is pointed at, and a specification that may
#: teach `memrank run --adapter X --benchmark Y` is a specification of a CLI that does not exist.
HISTORICAL: tuple[str, ...] = (
    "tech-debt.md",               # narrates past runs; only its forward-looking lines are fixed
    "docs-internal/open-issues.md",   # ditto
    "docs-internal/superpowers/",     # evidence / plans / specs, each dated
    "docs-internal/experimental-prds/",
)

#: A dated audit is a record of a moment, same as the above, but there are ten of them and more
#: arrive; matching the naming convention beats listing each.
DATED_AUDIT = re.compile(r"/\d{4}-\d{2}-\d{2}-")

#: `docs/repro/` carries the same date prefix and is deliberately NOT exempt. A repro doc's whole
#: claim is that typing this reproduces that number -- a command in one is a standing invitation,
#: not a record, so it is the one dated corpus that has to keep working.
DATED_BUT_STILL_EXECUTABLE = ("docs-internal/repro/",)

#: The commands whose flags these are. `memrank-ops.py compare` still declares `--slice`,
#: `--tier` and `--ack-egress` (memrank/ops/analysis_cli.py) and is a DIFFERENT product -- the
#: platform team curating a leaderboard, not a user evaluating a memory system. Keying on the
#: invoked command rather than the bare flag string is what keeps this test off its back.
INVOCATION = re.compile(r"\bmemrank(?:\.runner)?\s+(?:submit|run)\b")

#: A markdown inline-code span, which is typeable even though its surrounding prose is not.
INLINE_CODE = re.compile(r"`([^`]+)`")

#: Retired *command names* are only wrong when someone is told to type them. `memrank run` in
#: prose about the past is not an instruction, but inside a shell block it is one, so the same
#: INVOCATION anchor decides both.
RETIRED_COMMANDS = tuple(f"memrank {old}" for old in RETIRED)


def _live_files() -> list[Path]:
    """Every file whose commands a reader is entitled to run."""
    found: set[Path] = set()
    for glob in LIVE_GLOBS:
        found.update(p for p in ROOT.glob(glob) if p.is_file())
    live = []
    for path in found:
        relative = path.relative_to(ROOT).as_posix()
        if any(relative.startswith(h) for h in HISTORICAL):
            continue
        if DATED_AUDIT.search("/" + relative) and not relative.startswith(DATED_BUT_STILL_EXECUTABLE):
            continue
        live.append(path)
    return sorted(live)


def _typeable_lines(path: Path) -> list[tuple[int, str]]:
    """Lines a reader could plausibly type, with shell continuations joined.

    In markdown that means fenced code only. Prose says things like "it deliberately records no
    memrank run results" and "the judge half of a memrank run is not bit-reproducible", which are
    English sentences about the tool, not instructions to it -- scanning prose flags four such
    files and teaches the next person to silence this test rather than trust it.

    A multi-line invocation also hides its flags from a line-at-a-time scan -- `docs/partner-demo-
    script.md` puts `--tier` four lines below the command it belongs to -- so a trailing backslash
    pulls the next line up before anything is matched.
    """
    markdown = path.suffix == ".md"
    inside_fence = not markdown
    lines: list[tuple[int, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if markdown and line.startswith("```"):
            inside_fence = not inside_fence
            continue
        if not inside_fence:
            # Inline code in prose is still a command -- `docs/repro/`'s Method section states the
            # reproducing command in backticks mid-sentence, which is exactly the line that has to
            # keep working -- so the spans are scanned even though the surrounding prose is not.
            lines.extend((number, span) for span in INLINE_CODE.findall(line))
            continue
        if lines and lines[-1][1].endswith("\\"):
            first, previous = lines[-1]
            lines[-1] = (first, previous.rstrip("\\") + " " + line)
        else:
            lines.append((number, line))
    return lines


def _instructions(path: Path) -> list[tuple[int, str]]:
    """The typeable lines that actually invoke the CLI."""
    return [(n, text) for n, text in _typeable_lines(path) if INVOCATION.search(text)]


def _offences(path: Path) -> list[str]:
    """Every retired form this file tells a reader to type."""
    found = []
    for number, text in _instructions(path):
        for flag, instead in RETIRED_FLAGS.items():
            if re.search(rf"{re.escape(flag)}(?=[\s=]|$)", text):
                found.append(f"{path.relative_to(ROOT)}:{number} teaches `{flag}` -- {instead}")
        for command in RETIRED_COMMANDS:
            if re.search(rf"{re.escape(command)}(?=[\s]|$)", text):
                found.append(f"{path.relative_to(ROOT)}:{number} teaches `{command}`, which exits 2")
    return found


@pytest.mark.parametrize("path", _live_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_live_instruction_is_refused_by_the_cli(path: Path):
    """A doc or script that teaches a retired form sends its reader straight into exit 2."""
    offences = _offences(path)
    assert not offences, "\n".join(offences)


#: Files the scan must reach or it is vacuous. Split by side, because this test runs in the
#: published projection too and most of these do not exist there. Asserting an internal path
#: unconditionally would turn the anti-vacuity guard into the thing that fails a clean public
#: checkout -- the same defect this guard exists to prevent, in its own body.
PUBLIC_MUST_COVER = ("docs/system-contract.md", "docs/local-development.md", "docs/SPEC.md")
INTERNAL_MUST_COVER = ("scripts/local-eval.sh", "scripts/internal/build-push.sh", "AGENTS.md",
                       "deploy/README.md",
                       "docs-internal/repro/2026-08-04-judged-matched-baseline.md")


def test_the_scan_reaches_the_files_that_actually_broke():
    """Guards the guard: an over-eager exclusion would make every case above vacuously pass."""
    scanned = {p.relative_to(ROOT).as_posix() for p in _live_files()}
    for must_cover in PUBLIC_MUST_COVER:
        assert must_cover in scanned, f"{must_cover} fell out of the live corpus"
    for must_cover in INTERNAL_MUST_COVER:
        if (ROOT / must_cover).exists():
            assert must_cover in scanned, f"{must_cover} fell out of the live corpus"


def test_the_ops_cli_keeps_its_own_slice_flag():
    """`memrank-ops.py compare --slice` is a different product's flag and must not be caught.

    Without this, the obvious tightening -- match the bare flag anywhere -- silently deletes a
    working command from `docs/partner-demo-script.md` on the next person's cleanup pass.
    """
    ops = "uv run python scripts/internal/memrank-ops.py compare --benchmark beam --tier 100k --slice smoke"
    assert not INVOCATION.search(ops)
