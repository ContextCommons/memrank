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
"""Deprecated: the hidden `memrank targets` command. Evaluate an agent with `memrank run`
instead; see the README.

The ``memrank targets`` command surface -- list and inspect what is under test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from memrank.docs import doc_url
from memrank.targets import Manifest, ManifestError, list_targets, resolve_target
from memrank.targets.catalog import checkout_status, secret_status
from memrank.targets.manifest import to_dict
from memrank.term import detail, fmt, style, table

targets_app = typer.Typer(help="Inspect the catalog of things under test.")


def _resolve_or_exit(ref: str, overrides: list[str]) -> Manifest:
    """Resolve a ref, reporting a malformed target as a one-line error rather than a traceback."""
    try:
        return resolve_target(ref, overrides)
    except ManifestError as exc:
        style.error(str(exc))
        raise typer.Exit(1) from exc


def _components_line(target: Manifest) -> str:
    """One-line summary of a target's configured components."""
    if not target.components:
        return "—"
    parts = []
    for role in sorted(target.components):
        comp = target.components[role]
        model = comp.model or "unset"
        dims = f" ({comp.dims})" if comp.dims else ""
        parts.append(f"{role}={comp.provider or '?'}/{model}{dims}")
    return "  ".join(parts)


@targets_app.command("ls")
def ls() -> None:
    """List every known target."""
    from memrank.targets.catalog import builtin_dir, origins

    where = origins()
    builtin = builtin_dir()
    rows = []
    for name in list_targets():
        target = resolve_target(name)
        # A base is listed rather than hidden: someone reading the catalog should see where a
        # variant's engine and graph come from. Marked, because submitting it is refused.
        kind = "base" if target.abstract else target.kind
        # Shipped targets say nothing; anything else names its directory, so "where did this come
        # from" is answerable without knowing the precedence rules.
        origin = "" if where.get(name) == builtin else str(where[name])
        # An unlinked target is listed, not hidden: it exists, it just cannot run here yet, and
        # saying so in the catalog is cheaper than finding out at submit time.
        checkout = checkout_status(target)
        components = _components_line(target)
        if checkout is not None and checkout[1] is None:
            components = f"{components}  {style.bad('✘ not linked')}".lstrip()
        rows.append([name, kind, components, origin])
    columns = [
        table.Column("NAME", width=20, styler=style.accent),
        table.Column("KIND", width=12, styler=style.unit),
        table.Column("COMPONENTS"),
    ]
    # A catalog of only shipped targets has nothing to say about origin, and a column of blanks
    # under a heading reads as data that went missing rather than as a question nobody asked.
    if any(row[3] for row in rows):
        columns.append(table.Column("ORIGIN", styler=style.dim, ellipsis=True))
    else:
        rows = [row[:3] for row in rows]
    table.emit(tuple(columns), rows, title="Targets")


@targets_app.command("show")
def show(
    ref: str = typer.Argument(..., help="target ref, e.g. hindsight:matched"),
    overrides: list[str] = typer.Argument(None, help="key=value overrides"),
    as_json: bool = typer.Option(False, "--json", help="machine-readable JSON output"),
) -> None:
    """Show a target's fully-resolved manifest and the secrets it needs."""
    target = _resolve_or_exit(ref, overrides or [])
    if as_json:
        style.out(json.dumps(to_dict(target), indent=2, sort_keys=True))
        return
    detail.emit(detail.panel(target.name, [
        ("kind", target.kind),
        ("adapter", target.adapter),
        ("components", _components_line(target)),
        ("depends", ", ".join(target.depends) or fmt.UNKNOWN),
    ]))
    secrets = secret_status(target)
    if secrets:
        detail.emit(detail.rule("secrets", indent=2))
        detail.emit(detail.fields(
            [(name, style.good("✔") if ok else style.bad("✘"))
             for name, ok in secrets], indent=4))
    checkout = checkout_status(target)
    if checkout is not None:
        ref, path = checkout
        detail.emit(detail.rule("checkout", indent=2))
        detail.emit(detail.fields([(
            ref,
            f"{style.good('✔')} {path}" if path
            else f"{style.bad('✘')} not linked -- memrank targets link {ref} <path>")], indent=4))


def _verifiable_or_exit(target: Manifest) -> None:
    """Refuse a target whose adapter is not a translator.

    Only the native contract has a published spec a third party can be measured against. Every
    other adapter is memrank's own code, already covered by the conformance suite in this repo.
    """
    if target.adapter != "native":
        style.error(
            f"{target.name!r} uses the {target.adapter!r} adapter, which memrank implements "
            f"itself -- there is no third-party contract to verify. `verify` checks a translator "
            f"against {doc_url('system-contract.md')}.")
        raise typer.Exit(1)
    if target.binding is None:
        style.error(
            f"{target.name!r} declares no source binding, so there is no translator for memrank "
            f"to launch. Add binding/launch/network (see {doc_url('system-contract.md')} "
            f"section 9).")
        raise typer.Exit(1)


@targets_app.command("link")
def link(
    ref: str = typer.Argument(None, help="target ref, e.g. myengine:dev; omit to list every link"),
    path: str = typer.Argument(None, help="the checkout this machine should run for that target"),
    remove: bool = typer.Option(False, "--remove", help="forget the link for REF"),
) -> None:
    """Point a source target at a checkout on this machine.

    A source target evaluates a working tree, and where that tree lives is not the descriptor's to
    know: the repository holding the descriptor and the engine repository know nothing about each
    other, and the checkout may be anywhere. The link is per TARGET, not per repository, so two arms
    of the same engine -- a worktree, a commit under comparison -- stay distinct.

    Stored in this machine's config directory and never committed, like ``go.work`` or
    ``.cargo/config.toml``. A value may also be an ``${ENV_VAR}`` reference if you would rather keep
    the path in the environment.
    """
    from memrank.targets import checkouts

    if remove:
        if not ref:
            raise typer.BadParameter("--remove needs a target ref")
        style.say(f"unlinked {ref}" if checkouts.delete(ref) else f"{ref} was not linked")
        return
    if ref is None:
        rows = [[name, value] for name, value in checkouts.links()]
        if not rows:
            style.say("(nothing linked -- memrank targets link <ref> <path>)")
            return
        table.emit((table.Column("TARGET", width=24, styler=style.accent),
                    table.Column("CHECKOUT", ellipsis=True)), rows, title="Links")
        return
    if path is None:
        raise typer.BadParameter(f"give the checkout to link {ref} to")
    # Resolved before storing so a typo fails here rather than at submit time, when a run has
    # already announced itself and started spending.
    stored = checkouts.put(ref, path)
    if not path.startswith("${") and not Path(stored).is_dir():
        checkouts.delete(ref)
        raise typer.BadParameter(f"{stored} is not a directory")
    style.say(f"{ref} -> {stored}")


@targets_app.command("verify")
def verify(
    ref: str = typer.Argument(..., help="target ref, e.g. myengine:dev"),
    overrides: list[str] = typer.Argument(None, help="key=value overrides"),
) -> None:
    """Check a translator against the memrank system contract.

    Launches the target, exercises the contract, and reports every finding at once. Strictly a
    conformance check on someone else's implementation of a published spec -- it does not diagnose
    the machine, credentials, or the engine's quality.
    """
    import tempfile

    from memrank.adapters.conformance import contract_checks
    from memrank.adapters.native import Native
    from memrank.placement.base import PlacementError
    from memrank.placement.workspace import WorkspacePlacement

    target = _resolve_or_exit(ref, overrides or [])
    _verifiable_or_exit(target)
    log_path = Path(tempfile.gettempdir()) / f"memrank-verify-{target.adapter}-{target.engine.port}.log"
    style.say(f"launching {target.name} -- translator log: {log_path}")
    try:
        with WorkspacePlacement(target=target, log_path=log_path) as placement:
            endpoint = placement.provision(target)
            adapter = Native(base_url=endpoint.base_url)
            try:
                rows = contract_checks(adapter)
            finally:
                adapter.close()
    except PlacementError as exc:
        style.error(str(exc))
        raise typer.Exit(1) from exc
    _report_or_exit(rows, log_path)


def _report_or_exit(rows: list, log_path: Path) -> None:
    """Print every finding, then exit non-zero if any failed."""
    columns = (
        table.Column("", width=1),
        table.Column("CHECK", width=12, styler=style.accent),
        table.Column("DETAIL"),
    )
    table.emit(columns,
               [[table.Cell("✔" if row.ok else "✘", style.good if row.ok else style.bad),
                 row.name, row.detail] for row in rows],
               title="Contract checks")
    # The fix is advice about the answer rather than the answer, so it stays on stderr -- and
    # below the whole table rather than interleaved, which a bordered table has no room for.
    for row in rows:
        if not row.ok and row.fix:
            style.say(f"  {row.name}: {row.fix}")
    failed = [row.name for row in rows if not row.ok]
    if failed:
        style.error(f"{len(failed)} contract check(s) failed: {', '.join(failed)}. "
                    f"Translator output: {log_path}")
        raise typer.Exit(1)
    style.say("translator conforms to the system contract")


def _render_cloud(target: Manifest, aws_json: str | None) -> dict:
    """Render the ECS task definition exactly as `memrank submit --on cloud` would.

    Goes through the same :func:`memrank.placement.cloud.render_from_context` the submit path uses.
    It used to build an ``AwsContext`` from the raw JSON keys, which meant `--aws
    .aws-context.json` never actually worked -- that file has a different shape -- and anything
    inspected here was not necessarily what would launch. Inspection is only worth having if it
    shows the real thing.
    """
    from memrank import config
    from memrank.placement.cloud import render_from_context

    ctx = json.loads(Path(aws_json).read_text()) if aws_json and aws_json != "-" else (
        json.loads(sys.stdin.read()) if aws_json == "-" else config.aws_context())
    return render_from_context(
        target, ctx,
        command=f"memrank submit {target.name} <benchmark> --output-dir /work/results",
        image_tag="<image-tag>")


@targets_app.command("render")
def render(
    ref: str = typer.Argument(..., help="target ref, e.g. hindsight:matched"),
    for_: str = typer.Option(..., "--for", help="placement to render for: cloud or local"),
    aws: str = typer.Option(None, "--aws", help="AWS context JSON ('-' for stdin); defaults to "
                                                "the same file `memrank submit --on cloud` uses"),
    overrides: list[str] = typer.Option(None, "--set", help="key=value overrides"),
) -> None:
    """Render the deployment document that runs a target, generated from its manifest."""
    # This is what replaced the hand-authored per-adapter ECS task definitions, which had the
    # components welded into one template per *adapter* -- so ``mem0:voyage`` could not be
    # expressed in the cloud at all; it ran bge-small under a row labelled voyage. In a comment
    # rather than the docstring because typer prints a command's docstring in `--help`, and those
    # templates lived in a deployment directory an installed copy does not have (ATO-2125).
    from memrank.placement.cloud import CloudRenderError

    target = _resolve_or_exit(ref, list(overrides or []))
    if for_ not in ("cloud", "local"):
        style.error(f"unknown --for {for_!r}; known: cloud, local")
        raise typer.Exit(1)
    try:
        if for_ == "cloud":
            document = _render_cloud(target, aws)
        else:
            from memrank.placement.local import render_compose

            document = render_compose(target, project=f"memrank-{target.adapter}")
    # An in-process arm, a missing artifact or an unsupplied secret ARN are all operator errors
    # with an actionable message; a traceback would bury it.
    except (CloudRenderError, ValueError) as exc:
        style.error(str(exc))
        raise typer.Exit(1) from exc
    style.out(json.dumps(document, indent=2, sort_keys=True))
