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
"""``memrank config`` -- the standing defaults, and where each one comes from.

``ls`` shows resolved values *with their source*, because the failure this plane produces is
never "the setting is missing" -- it is "the setting is not the one I set". A file value silently
overridden by an inherited environment variable is invisible until it costs a run.
"""
from __future__ import annotations

import typer

from memrank import settings
from memrank.term import fmt, style, table

config_app = typer.Typer(help="Standing defaults (organisation, API address).")


@config_app.command("set")
def cli_config_set(
    key: str = typer.Argument(..., help="setting name, e.g. defaults.org"),
    value: str = typer.Argument(..., help="the value to store"),
) -> None:
    """Store a setting."""
    try:
        stored = settings.put(key, value)
    except settings.UnknownSetting as exc:
        raise typer.BadParameter(str(exc).strip("'")) from exc
    resolved, source = settings.resolve(key)
    # The stored form, not the argument: a path setting is absolutized on write, and echoing
    # the raw input would hide exactly the normalization the user needs to see.
    style.say(f"{key} = {stored}")
    if source == settings.ENV:
        # Storing a value the environment already answers is the exact confusion `ls` exists to
        # prevent; saying so at the moment of the write is better than letting it be discovered.
        spec = settings.setting(key)
        style.note(f"{spec.env} is set in this environment, so {key} resolves to "
                   f"{resolved!r} until it is unset")


@config_app.command("get")
def cli_config_get(
    key: str = typer.Argument(..., help="setting name, e.g. defaults.org"),
) -> None:
    """Print one setting's effective value."""
    try:
        value, _ = settings.resolve(key)
    except settings.UnknownSetting as exc:
        raise typer.BadParameter(str(exc).strip("'")) from exc
    if value is None:
        raise typer.Exit(code=1)
    style.out(value)


@config_app.command("ls")
def cli_config_ls() -> None:
    """Print every setting: its effective value, and which source supplied it."""
    columns = (
        table.Column("KEY", styler=style.accent),
        # An unset setting shows the same em dash the rest of the CLI uses for "no answer",
        # dimmed so it does not read as a value someone chose.
        table.Column("VALUE", ellipsis=True),
        table.Column("SOURCE", styler=style.unit),
        table.Column("DESCRIPTION", styler=style.dim),
    )
    rows = [[spec.key,
             value if value else table.Cell(fmt.UNKNOWN, style.dim),
             source,
             spec.help]
            for spec, value, source in settings.resolved()]
    table.emit(columns, rows, title="Settings")
