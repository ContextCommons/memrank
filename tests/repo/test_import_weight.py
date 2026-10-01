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
"""What a real import actually pulls in: the CLI's stack, and the library run path's.

Two boundaries, both checked in a subprocess against a real import rather than by reading
pyproject, because the question is what the module graph does.

`pip install memrank` gets the command; fastapi, uvicorn, psycopg and the Fernet backend live
in the `api` extra because only the service needs them. Nothing stops a future top-level import
from putting one back on the CLI's path -- at which point the extra still *resolves* (a developer
install has everything) and the breakage appears only in a user's `ModuleNotFoundError`.

So this is checked in a subprocess against a real import, not by reading pyproject: the question
is what the module graph actually pulls in.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# `tomllib` is stdlib only from 3.11, and `pyproject.toml` declares `requires-python = ">=3.10"`.
# Without this shim both of the tests that hold the public/private boundary raise at IMPORT on the
# declared floor -- a collection error in exactly the two files that are supposed to fail loudly.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on 3.10
    import tomli as tomllib

#: Everything the CLI reaches for on the way to running a command.
CLI_MODULES = (
    "memrank.runner",
    "memrank.cli.auth",
    "memrank.cli.config",
    "memrank.cli.runs",
    "memrank.cli.evals",
    "memrank.cli.targets",
    "memrank.cli.secrets",
    "memrank.settings",
    "memrank.accounts.credentials",
    "memrank.placement.run_api_client",
)

#: Never published. Kept in step with `publish.toml`'s `internal` prefixes under `memrank/`.
INTERNAL_PACKAGES = ("memrank.api", "memrank.arena", "memrank.events", "memrank.leaderboard",
                     "memrank.analysis", "memrank.ops")


def _published_anyway() -> set[str]:
    """The `[exceptions] public` entries, read from the manifest rather than copied.

    A module under an internal prefix that publish.toml nevertheless publishes is not a leak, and
    the two lists must not be able to drift: `tests/repo/test_public_boundary.py` checks the manifest
    statically, this checks it against a real import, and they have to agree on the same file.
    """
    root = Path(__file__).resolve().parents[2]
    manifest = tomllib.loads((root / "publish.toml").read_text(encoding="utf-8"))
    return {p[: -len(".py")].replace("/", ".").removesuffix(".__init__")
            for p in manifest["exceptions"].get("public", []) if p.endswith(".py")}

#: The command-line framework. A library caller who ran an evaluation asked for an
#: evaluation, not for a terminal -- `memrank.config` imported `memrank.term.style` (and so
#: `typer`) at module level to serve ONE line, in a secrets-prompting path a library run cannot
#: reach, and that put the whole terminal stack into every such caller's process.
#
#: `click` -- typer's own dependency -- is deliberately NOT here. It reaches the run path from
#: `httpx.__init__`, which imports its own `_main` CLI module, and httpx is the HTTP library
#: every adapter is required to use. That import is not memrank's to make local, so naming it
#: here would be a permanently red assertion about somebody else's package.
TERMINAL_ONLY = ("typer",)

#: Declared in the `api` extra. A CLI that imports one of these is a CLI a user cannot install.
SERVICE_ONLY = ("fastapi", "uvicorn", "psycopg", "psycopg_pool", "pydantic_settings",
                "cryptography", "redis", "websockets")

#: What `import memrank` must not pull in, now that the seven are plain imports rather than a
#: lazy table.
#:
#: The budget is named modules rather than a number of milliseconds, because a wall-clock
#: assertion is not deterministic and would fail on a loaded machine rather than on a
#: regression. These are the trees that actually cost: `tiktoken` (the cell run's token
#: accounting, reached through `memrank.metrics.cost`) and `anthropic` (the judge). Measured
#: with `-X importtime` while this was written, they are the difference between a ~66ms import
#: and a ~340ms one -- and none of it is on the path of a caller who asked for `memrank.run`.
#:
#: `pydantic` is deliberately NOT here. The seven ARE pydantic models; there is no import of
#: `memrank.Task` that does not load it, and naming it would be a permanently red assertion
#: about a dependency the surface is made of. Nor is `httpx`: `memrank.systems` holds the
#: shipped system CLASSES, an editor can only follow a name that is really there, and every
#: HTTP client memrank ships is required to use httpx. Measured with `-X importtime` when
#: `memrank.systems` and `memrank.evaluations` were added, `import memrank` went from ~68ms to
#: ~98ms, all of it httpx and none of it the trees below.
PREVIOUS_SURFACE_ONLY = ("tiktoken", "anthropic")

#: The names `import memrank` must resolve WITHOUT a module-level `__getattr__`, because an
#: editor cannot follow one: command-click on `memrank.run` went nowhere and hover showed
#: nothing while these were lazy. Checked as real module globals, which is the same thing
#: Pyright and mypy read. `systems` and `evaluations` are here for the same reason one step
#: further out: a string name is not navigable either, and these are the modules that give the
#: shipped systems and evaluations Python names an editor can follow.
STATIC_NAMES = ("measure", "paired", "system", "evaluation", "Result", "Task", "Trace",
                "Measure", "Value", "System", "Memory", "Evaluation", "WordMatch",
                "catalog", "systems", "evaluations")


def _imported_by(modules: tuple[str, ...]) -> set[str]:
    """Every module name in sys.modules after importing `modules`, in a fresh process.

    Full dotted names, not just top-level packages: the boundary check below is about
    `memrank.api` versus `memrank.cli`, and a top-level projection cannot tell those apart.
    Top-level distribution names are still present in the set, so `SERVICE_ONLY` matches.
    """
    script = (
        "import sys\n"
        f"for name in {list(modules)!r}:\n"
        "    __import__(name)\n"
        "print(' '.join(sorted(sys.modules)))\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return set(done.stdout.split())


def _modules_after(script: str, tmp_path: Path) -> set[str]:
    """Every module name in sys.modules after running `script`, in a fresh process.

    State dirs point into `tmp_path`: a run resolves the target catalog, which reads whatever
    `targets.path` this machine happens to name, and a developer with one configured would
    otherwise see this fail on their own settings rather than on the import graph.
    """
    env = {**os.environ,
           "MEMRANK_CONFIG_DIR": str(tmp_path / "cfg"),
           "MEMRANK_RUNS_DIR": str(tmp_path / "runs")}
    body = script + "\nimport sys\nprint(' '.join(sorted(sys.modules)))\n"
    done = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    return set(done.stdout.split())


def test_importing_memrank_resolves_the_surface_without_a_lazy_table(tmp_path):
    """Every name in `memrank.__all__` is a real global on the module, not a `__getattr__` hit.

    This is the inspectability contract, checked where an editor checks it: a name in
    `module.__dict__` is what "go to definition", hover and autocomplete follow, and a name a
    module-level `__getattr__` invents at runtime is invisible to all three.
    """
    loaded = _modules_after(
        "import memrank\n"
        "missing = [n for n in memrank.__all__ if n not in vars(memrank)]\n"
        "assert not missing, f'resolved lazily, invisible to an editor: {missing}'\n"
        f"for name in {list(STATIC_NAMES)!r}:\n"
        "    assert name in vars(memrank), f'{name} is not a static global'", tmp_path)

    assert "memrank" in loaded


def test_importing_memrank_does_not_pull_the_previous_surface_in(tmp_path):
    """Static names, and still a reasonable import: the cost stayed where it was being paid.

    The seven became plain imports; the previous surface -- `EvalResult`, `JudgeConfig`,
    `SpanRecall` and the rest -- did not, and this is why. Each is reachable by its own full
    module path, which is where an editor finds it anyway.
    """
    loaded = _modules_after("import memrank", tmp_path)

    leaked = sorted(set(PREVIOUS_SURFACE_ONLY) & loaded)
    assert leaked == [], (
        f"`import memrank` pulled {leaked} in. Something at the top of `memrank/__init__.py` "
        f"now reaches the cell run or the judge -- keep that name in `_PREVIOUS_SURFACE`, or "
        f"make the heavy dependency local to the function that uses it.")


def test_a_completed_library_run_imports_no_terminal_machinery(tmp_path):
    """`evaluation.run(system=...)` end to end: is the command-line framework in the process?

    The run is the typed one over the seven, which is what a library caller now reaches for,
    written the way the README writes it: `memrank.system("word-overlap")` rather than an
    import of the class, so the budget covers the whole adapter registry the name resolves
    through and not just the one module a hand-written import would have reached.

    The whole run rather than `import memrank`, because the import was already clean -- the
    eval surface resolves lazily (PEP 562) and `memrank.config` only loads once something
    actually runs. A budget that stopped at the import would have passed throughout.
    """
    loaded = _modules_after(
        'import memrank\n'
        'memrank.evaluation("demo").run(system=memrank.system("word-overlap"))', tmp_path)

    leaked = sorted(set(TERMINAL_ONLY) & loaded)
    assert leaked == [], (
        f"a completed library run imported {leaked}. Something on the run path imports "
        f"`memrank.term.style` (or typer directly) at module level. Make the import local to "
        f"the call site that needs it -- a library caller has no terminal.")


def test_a_completed_library_run_imports_no_command_line_module(tmp_path):
    """The other half of the same boundary: no `memrank.cli` submodule either.

    `memrank.term` is presentation and `memrank.cli` is the command line itself. Checked here
    beside the framework because the two leak by the same mechanism -- one module-level import
    on a shared path -- and the run path is where it matters.
    """
    loaded = _modules_after(
        'import memrank\n'
        'memrank.evaluation("demo").run(system=memrank.system("word-overlap"))', tmp_path)

    leaked = sorted(m for m in loaded if m == "memrank.cli" or m.startswith("memrank.cli."))
    assert leaked == [], (
        f"a completed library run imported {leaked} -- the library must not depend on the "
        f"command line it is meant to replace as the primary surface.")


def test_the_cli_does_not_import_the_api_stack():
    loaded = _imported_by(CLI_MODULES)
    leaked = sorted(set(SERVICE_ONLY) & loaded)
    assert leaked == [], (
        f"the CLI imports {leaked}, which live in the `api` extra -- a user who ran "
        f"`pip install memrank` would get ModuleNotFoundError. Import it lazily, inside the "
        f"function that needs it.")


def test_the_cli_does_not_import_an_internal_package():
    """The public closure stays closed.

    `publish.toml` says these packages never ship. The CLI reaching one is not a packaging
    inconvenience like the check above -- it is a public tree that cannot import its own entry
    point, and it would be found by the first outsider to `pip install memrank` rather than
    here.
    """
    loaded = _imported_by(CLI_MODULES)
    published = _published_anyway()
    leaked = sorted(m for m in loaded if m not in published
                    and any(m == pkg or m.startswith(f"{pkg}.") for pkg in INTERNAL_PACKAGES))
    assert leaked == [], (
        f"the CLI imports {leaked}, which `publish.toml` classifies internal -- the public tree "
        f"would not contain them. Import it lazily, or move the code the CLI needs out.")
