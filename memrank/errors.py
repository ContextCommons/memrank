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
"""The root of every error a user is meant to read -- the type the CLI boundary trusts.

Before this existed, whether a mistake reached the terminal as one actionable line or as a
Rich traceback depended on whether that particular call site remembered to wrap itself. It
usually did; ``style.error(...) + typer.Exit(1)`` appears two dozen times. But a defense
applied per surface is leaky by construction, and the surface that got missed was
``_split_overrides`` -- which runs a few lines before the ``except ManifestError`` that would
have caught it -- so a stray shell token spilled a stack trace.

``MemrankError`` closes that by inverting the default. Raising it is a promise that
``str(exc)`` is a complete, actionable sentence, and ``runner.main`` prints exactly that with
no traceback. Anything still deriving from bare ``Exception`` is thereby declaring itself an
internal bug, which is the distinction the codebase previously had no way to express.

The promise is enforced, not merely documented: ``tests/cli/test_cli_errors.py`` walks every
exception class defined under ``memrank/`` and fails when a new one skips this base.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from memrank.outcome import Outcome, Step


class MemrankError(Exception):
    """An error the operator can act on: bad input, bad config, a refused request.

    Subclass this for anything a user could plausibly cause and could plausibly fix. The
    message is the whole user interface -- write it as a sentence that names what was wrong
    and, where there is one, the move that fixes it. It is printed verbatim.
    """


class ActionRequired(MemrankError):
    """Nothing broke: there is a step the user takes first -- sign in, save a key, pick an org.

    Rendered apart from a failure (no red, no ``error:``), because a missing step reported as a
    crash reads as something inside memrank having broken. ``statement`` says what is missing in
    plain words; each of ``commands`` is one line the user can copy and run, printed on its own.

    ``steps``, when given, are what to do spelled out one by one, and ``statement`` is then only
    what happened -- the shape a run's ending shows (:mod:`memrank.outcome`).
    """

    def __init__(self, statement: str, *commands: str, steps: Sequence[Step] = ()) -> None:
        super().__init__(statement)
        self.statement = statement
        self.commands = commands
        self.steps = tuple(steps) or ((Step("", commands),) if commands else ())

    def __str__(self) -> str:
        rows = [self.statement]
        for step in self.steps:
            rows += ([step.text] if step.text else []) + [f"  {c}" for c in step.commands]
        return "\n".join(rows)


class Concluded(MemrankError):
    """A command's ending, carried to the CLI boundary to be rendered and exited with.

    ``str()`` is the outcome as plain lines, so anything that prints the error without the
    renderer still says what happened and what to do.
    """

    def __init__(self, outcome: Outcome, exit_code: int = 1) -> None:
        super().__init__(outcome.title)
        self.outcome = outcome
        self.exit_code = exit_code

    def __str__(self) -> str:
        from memrank.term.outcome import plain

        return plain(self.outcome)


class MissingOptionalDependency(MemrankError):
    """An optional extra this command needs is not installed."""


def optional_import(module: str, extra: str | None) -> Any:
    """Import ``module``, or fail with the install command for ``extra``.

    The one place a missing optional package is turned into an error. Every lazy import used to
    do this itself, and the four that existed disagreed: three raised ``RuntimeError`` and one
    raised ``ImportError``, so the CLI boundary called all of them internal bugs. A missing
    package is the single most user-fixable failure there is, and it was the one telling people
    to file a traceback.

    ``tests/cli/test_cli_errors.py`` walks every ``except ImportError`` handler under ``memrank/`` and
    fails when one raises something other than a :class:`MemrankError`. That check exists because
    the sibling test -- every exception CLASS subclasses ``MemrankError`` -- structurally cannot see
    this: nothing new is being defined, a builtin is being raised.

    Args:
        module: Import path, e.g. ``"anthropic"``.
        extra: The extra that provides it, e.g. ``"judge"``. ``None`` for a BASE dependency, where
            a failed import means the install is broken rather than incomplete -- telling someone
            to add an extra they already have would send them the wrong way.

    Returns:
        The imported module.

    Raises:
        MissingOptionalDependency: Naming the package and the command that fixes it.
    """
    from importlib import import_module

    try:
        return import_module(module)
    except ImportError as exc:
        # The install's OWN shape, read from its PEP 610 metadata, because an instruction that
        # does not fit the reader's install is a second dead end on top of the missing package.
        # Naming only the contributor's sent an outside user to `uv sync` from a directory with
        # no pyproject.toml; naming a git URL sends a released install to a branch checkout.
        from memrank.provenance.install import dependency_instruction

        fix = (f"which is part of the {extra!r} extra. Install it with: "
               f"{dependency_instruction(module, extra)}"
               if extra else
               f"which is a base dependency, so this install is incomplete rather than missing "
               f"an extra. Repair it with: {dependency_instruction(module)}")
        raise MissingOptionalDependency(
            f"this command needs the {module!r} package, {fix}") from exc


def wants_traceback() -> bool:
    """Whether an unexpected exception should surface its stack instead of one line.

    Read at call time, not import, for the same reason ``style.wants_color`` is: tests and
    wrapper scripts set it per invocation.

    There is deliberately no value that enables local variables. ``submit`` decrypts org
    credentials into a frame one line before the crash that prompted this module, and a
    traceback that renders locals would print them to the terminal -- on a laptop, which is
    exactly where those credentials are. The stack is what debugging needs; the bindings are
    only a leak.
    """
    return bool(os.environ.get("MEMRANK_DEBUG"))
