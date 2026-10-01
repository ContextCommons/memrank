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
"""Every command name and flag the CLI used to answer to, and what replaced each.

The surface is migrating to the interface model, and the cutover is deliberately hard: an old
name does not keep working, and it does not fail silently either -- it says what to type
instead. A script that breaks breaks loudly, once, with the fix in the
message.

Tables rather than a stub beside each rename, so the rule is enumerated and testable: nothing
can be retired without a pointer, because the tests walk these dicts. Later retirements add a
row here, not another mechanism.
"""
from __future__ import annotations

import typer

from memrank.term import style

#: old name -> its replacement, as the user should type it (without the `memrank` prefix).
#: A space in the old name retires a sub-app command ("secrets list") on that sub-app.
RETIRED: dict[str, str] = {
    "login": "auth login",
    "logout": "auth logout",
    "whoami": "auth status",
    "list-benchmarks": "evals ls",
    "list-runs": "runs ls",
    "secrets list": "secrets ls",
    # A fifth flat alias the model never had, and one word two planes owned: `auth status` is
    # the session's, and this one meant one run's record.
    "status": "runs show",
    # Credentials are the wallet's plane and a target's requirements are the catalog's; this
    # command answered for both, and `doctor` -- the environment plane it was meant to fold into
    # -- was dropped from the model rather than built.
    "preflight": "targets show",
}

#: Retired `submit` flag -> the sentence that tells the caller what to do instead. The value is
#: a full clause, not a noun, because not every retirement HAS a replacement: some flags named a
#: thing the positional arguments name now, and some named a thing that stopped existing.
#:
#: A retired flag stays declared (hidden) rather than being deleted outright: a deleted option
#: gets Click's "No such option", which says what is wrong and not what to type instead -- and
#: the person typing the old form is exactly the person who needs to be told.
RETIRED_FLAGS: dict[str, str] = {
    # The first three named the SUBJECT before `targets` existed, which the positional arguments
    # do now; two ways to say one thing is what the model's flag ontology exists to prevent, and
    # the cloud path already refused them.
    "--adapter": "name the target positionally -- `memrank submit <target> <eval>`",
    "--benchmark": "name the eval positionally -- `memrank submit <target> <eval>`",
    "--all-adapters": "sweep with a comma -- `memrank submit target-a,target-b <eval>`",
    # A tier and a slice are not knobs on an evaluation, they SELECT one: `beam:100k-smoke` is a
    # different evaluation from `beam:1m`, with a different question count and non-comparable
    # numbers. As flags they read like `--workers`, and two runs of "beam" at different slices
    # wrote the same artifact filename, so the second silently replaced the first.
    "--tier": "it rides the eval ref -- `memrank submit <target> beam:100k`",
    "--slice": "it rides the eval ref -- `memrank submit <target> locomo:smoke`",
    # These two were deleted outright rather than retired, so they answered with Click's bare
    # "No such option" -- the exact failure this table exists to prevent, and worse than the
    # flags that at least pointed somewhere. Neither has a replacement to point AT, which is
    # what forced the value to become a clause: the gate `--ack-egress` acknowledged still
    # stands, derived from the benchmark's own `is_synthetic`, and the judge's ceiling is now
    # the question count of the eval you named.
    "--ack-egress": "drop it -- judged egress is gated by the benchmark, not acknowledged per run",
    "--max-judge-calls": "drop it -- the eval ref you name is what bounds the judge",
}

#: Click's convention for a usage error -- the caller asked for something that does not exist.
#: Distinct from 1, which the surviving commands use for "ran, and the answer is no".
RETIRED_EXIT_CODE = 2


def refuse_retired_flags(given: dict[str, object]) -> None:
    """Refuse any retired flag the caller supplied, saying what to do instead.

    Args:
        given: Retired flag spelling -> what the parser bound to it. A value that is ``None`` or
            ``False`` was not supplied; anything else was.

    Raises:
        typer.BadParameter: On the first retired flag present, which exits
            :data:`RETIRED_EXIT_CODE` -- the same code the retired command names use.
    """
    for flag, instead in RETIRED_FLAGS.items():
        if given.get(flag):
            raise typer.BadParameter(f"`{flag}` is retired; {instead}")


def _make_stub(old: str, new: str):
    """A command that only reports its own replacement.

    Extra arguments are swallowed rather than parsed: someone retyping a whole old command
    line (``memrank run baseline demo --slice smoke``) must reach the message, not a parser
    error about a flag this stub never declared.
    """
    def stub(ctx: typer.Context) -> None:
        style.say(f"`memrank {old}` is now `memrank {new}`.")
        raise typer.Exit(code=RETIRED_EXIT_CODE)

    stub.__name__ = f"retired_{old.replace('-', '_')}"
    stub.__doc__ = f"Retired: use `memrank {new}`."
    return stub


def register_retired(app: typer.Typer) -> None:
    """Attach every retired name to ``app``, hidden from help.

    Hidden because ``memrank --help`` should show the surface as it is now, not a museum of
    what it was; registered because the person typing the old name is exactly the person who
    needs to be told. Must run after every ``add_typer``: a retirement row naming a sub-app
    that is not registered raises KeyError -- a bug, not something to skip.
    """
    groups = {g.name: g.typer_instance for g in app.registered_groups
              if g.name and g.typer_instance}
    for old, new in RETIRED.items():
        prefix, _, rest = old.partition(" ")
        target, name = (groups[prefix], rest) if rest else (app, old)
        target.command(
            name,
            hidden=True,
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        )(_make_stub(old, new))
