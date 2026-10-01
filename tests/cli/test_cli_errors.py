"""The CLI's error boundary: one actionable line for mistakes, no stack for anything.

These drive ``runner.app`` and ``runner.main`` rather than ``CliRunner.invoke(app, ...)``,
because ``CliRunner`` calls the underlying click command and never enters
``Typer.__call__`` -- where the boundary lives.

Driving ``app()`` matters more than it looks. Console scripts are generated at install time,
so an editable install keeps whatever ``sys.exit(app())`` line it was created with no matter
how ``[project.scripts]`` changes. A boundary reachable only through ``main`` was, for every
already-installed CLI, no boundary at all; the first version of this file tested ``main``
alone and let exactly that ship.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from memrank import runner
from memrank.errors import MemrankError

PACKAGE = pathlib.Path(runner.__file__).parent

#: Classes permitted to skip ``MemrankError``. Add only with a reason recorded here.
INTERNAL_ONLY: frozenset[str] = frozenset({
    # The base itself; it is what everything else must reach.
    "MemrankError",
    # Deliberately a BaseException so no adapter's `except Exception` can swallow a
    # `memrank kill` on its way up. Making it a MemrankError would reintroduce exactly the
    # interception its docstring exists to prevent. Caught explicitly in the sweep.
    "SweepKilled",
    # An internal carrier: it tags an arbitrary cause with the stage it came from and never
    # escapes the eval loop, which either turns it into a UnitOutcome or re-raises the cause.
    # Its message is the cause's type and text, which is a bug report rather than the actionable
    # sentence MemrankError promises the boundary it may print verbatim.
    "UnitStageError",
})

#: Builtin exception names that, used as a direct base, mean the class bypassed the boundary.
_BUILTIN_BASES = {"Exception", "BaseException", "RuntimeError", "ValueError", "OSError",
                  "TypeError", "KeyError", "IOError", "LookupError", "ArithmeticError"}


#: The exact override slot that spilled a traceback: a stray shell token, no `=`.
BAD_OVERRIDE = ["submit", "myengine", "demo", "--on", "cloud", "pull"]


def _run(argv: list[str], monkeypatch, *, entry=None) -> None:
    """Invoke the CLI through ``entry``, defaulting to the current entry point."""
    monkeypatch.setattr("sys.argv", ["memrank", *argv])
    (entry or runner.main)()


def _bounded_app_that_crashes():
    """A real ``_BoundedTyper`` whose command fails the way a bug fails: an unexpected type.

    Built here rather than by patching ``runner.app``, because the boundary now lives on
    ``__call__`` -- replacing the app object would delete the thing under test. Two commands,
    so Typer dispatches by name instead of collapsing to a single implicit callback.
    """
    app = runner._BoundedTyper(pretty_exceptions_enable=False)

    @app.command()
    def crash() -> None:
        raise RuntimeError("kaboom")

    @app.command()
    def fine() -> None:
        return None

    return app


def test_a_malformed_override_is_one_line_not_a_traceback(monkeypatch, capsys):
    """The crash that prompted the boundary: a stray shell token in the override slot."""
    with pytest.raises(SystemExit) as exit_info:
        _run(BAD_OVERRIDE, monkeypatch)
    out = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "error: override 'pull' is not valid; expected key=value" in out.err
    assert "Traceback" not in out.err + out.out


def test_the_boundary_holds_when_a_stale_console_script_calls_app(monkeypatch, capsys):
    """`sys.exit(app())` -- what every console script generated before `main` existed does.

    The regression this file exists for: `pyproject.toml` can name any entry point it likes,
    but an installed script keeps the one it was generated with, so `app` must be bounded.
    """
    with pytest.raises(SystemExit) as exit_info:
        _run(BAD_OVERRIDE, monkeypatch, entry=runner.app)
    out = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "error: override 'pull' is not valid; expected key=value" in out.err
    assert "Traceback" not in out.err + out.out


def test_a_user_error_stays_one_line_even_under_MEMRANK_DEBUG(monkeypatch, capsys):
    """MEMRANK_DEBUG surfaces bugs, not refusals -- a user mistake reads the same either way."""
    monkeypatch.setenv("MEMRANK_DEBUG", "1")
    with pytest.raises(SystemExit):
        _run(BAD_OVERRIDE, monkeypatch)
    assert "Traceback" not in capsys.readouterr().err


def test_an_unexpected_exception_names_itself_a_bug(monkeypatch, capsys):
    """An internal fault must not masquerade as something the user did wrong."""
    monkeypatch.delenv("MEMRANK_DEBUG", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        _run(["crash"], monkeypatch, entry=_bounded_app_that_crashes())
    out = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "error: internal error: RuntimeError: kaboom" in out.err
    assert "MEMRANK_DEBUG=1" in out.err
    assert "Traceback" not in out.err + out.out


def test_MEMRANK_DEBUG_re_raises_an_unexpected_exception(monkeypatch):
    """The escape hatch hands back the real stack for debugging."""
    monkeypatch.setenv("MEMRANK_DEBUG", "1")
    with pytest.raises(RuntimeError, match="kaboom"):
        _run(["crash"], monkeypatch, entry=_bounded_app_that_crashes())


def _error_classes_bypassing_the_base() -> list[str]:
    """Every class under memrank/ rooted directly at a builtin exception.

    Static, so it needs no imports and cannot be defeated by an optional dependency being
    absent. Inheriting from MemrankError or from another memrank error is fine by
    transitivity; inheriting straight from RuntimeError is the bypass this looks for.
    """
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name in INTERNAL_ONLY:
                continue
            names = {b.id for b in node.bases if isinstance(b, ast.Name)}
            if names & _BUILTIN_BASES and "MemrankError" not in names:
                found.append(f"{path.relative_to(PACKAGE.parent)}::{node.name}")
    return found


def test_every_error_class_routes_through_the_boundary():
    """The bypass test: a new error class that skips MemrankError fails here, not in the wild.

    Per-call-site error handling is how the override crash escaped. This is the enumeration
    that keeps the chokepoint from quietly growing a hole.
    """
    assert _error_classes_bypassing_the_base() == []


def test_the_base_is_reachable_from_a_leaf_error():
    """Sanity: the enumeration would be vacuous if nothing actually inherited the base."""
    from memrank.targets.resolve import RefError
    assert issubclass(RefError, MemrankError)


def _import_guards_raising_builtins() -> list[str]:
    """Every `except ImportError` handler under memrank/ that raises a non-memrank error.

    The sibling enumeration above walks exception CLASSES, and structurally cannot see this: a
    lazy import raises a BUILTIN, defining nothing. Four sites did exactly that -- the judge, the
    MLflow mirror, the MCP server and BEAM's downloader -- so a missing optional package, the most
    user-fixable failure there is, reached the terminal as "this is a bug in memrank; re-run with
    MEMRANK_DEBUG=1 for a traceback".

    Static like its sibling, so an absent optional dependency cannot make it vacuous.
    """
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
            caught = handler.type
            names = {caught.id} if isinstance(caught, ast.Name) else set()
            if "ImportError" not in names:
                continue
            for raised in (n for n in ast.walk(handler) if isinstance(n, ast.Raise)):
                exc = raised.exc
                func = exc.func if isinstance(exc, ast.Call) else exc
                name = func.id if isinstance(func, ast.Name) else None
                if name in _BUILTIN_BASES or name == "ImportError":
                    found.append(f"{path.relative_to(PACKAGE.parent)}:{raised.lineno} -> {name}")
    return found


def test_a_missing_optional_package_is_never_reported_as_a_bug():
    """Use memrank.errors.optional_import instead of a hand-rolled try/except ImportError.

    Fails when a fifth optional dependency is added without the helper -- the hole the
    class-based enumeration cannot cover.
    """
    assert _import_guards_raising_builtins() == []


def test_the_helper_names_an_install_command_that_fits_the_reader(monkeypatch):
    """One shape, the reader's own, read from PEP 610 metadata.

    It used to name both -- `uv sync` for a checkout and a `uv tool install ... <the git URL
    you installed from>` for everyone else -- because the helper could not tell them apart. It
    can: a released install has no git URL to substitute, and telling it to fetch one replaces
    the release with a branch checkout.
    """
    from memrank.errors import MemrankError, optional_import

    monkeypatch.setattr("memrank.provenance.install._direct_url", lambda distribution: None)
    monkeypatch.setattr("memrank.provenance.install._installed_as_uv_tool",
                        lambda distribution: False)
    with pytest.raises(MemrankError) as excinfo:
        optional_import("a_package_that_does_not_exist", "judge")
    message = str(excinfo.value)
    assert "'judge'" in message
    assert "pip install --upgrade 'memrank[judge]'" in message
    assert "git" not in message


def test_a_checkout_is_told_to_sync_rather_than_to_install(monkeypatch):
    """The contributor's shape is still served -- it is just no longer served to everyone."""
    from memrank.errors import MemrankError, optional_import

    monkeypatch.setattr("memrank.provenance.install._direct_url",
                        lambda distribution: {"url": "file:///home/dev/memrank",
                                              "dir_info": {"editable": True}})
    with pytest.raises(MemrankError) as excinfo:
        optional_import("a_package_that_does_not_exist", "judge")

    assert "uv sync --extra judge" in str(excinfo.value)


def test_a_base_dependency_is_not_reported_as_a_missing_extra(monkeypatch):
    """`tiktoken` is a base dependency; advising `--extra tiktoken` would send someone the wrong
    way. A failed import there means the install is broken, not incomplete."""
    from memrank.errors import MemrankError, optional_import

    monkeypatch.setattr("memrank.provenance.install._direct_url",
                        lambda distribution: {"url": "file:///home/dev/memrank",
                                              "dir_info": {"editable": True}})
    with pytest.raises(MemrankError) as excinfo:
        optional_import("another_package_that_does_not_exist", None)
    message = str(excinfo.value)
    assert "--extra" not in message and "uv sync" in message
