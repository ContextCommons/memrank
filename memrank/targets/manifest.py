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
"""Target manifest value types and validation.

A manifest is the complete description of one thing under test: which adapter drives it, whether it
needs containers, and which LLM/embedder it is configured with. Validation is strict and loud -- a
manifest that cannot be stated exactly is rejected rather than defaulted, because a silently wrong
component value produces a run whose receipt is internally truthful and externally a lie.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any

from memrank.errors import MemrankError
from memrank.secrets.names import is_secret_key

KINDS: tuple[str, ...] = ("stack", "in-process")
COMPONENT_ROLES: tuple[str, ...] = ("embedder", "llm")
# How memrank talks to the engine. Declared rather than inferred because it is a methodology
# constraint, not an implementation detail: mem0 must be exercised over HTTP because the public
# mem0ai SDK cannot replicate the running server (see tech-debt.md).
# "translator" is its own class rather than a flavour of "http": a translator is an extra process
# and an extra hop in front of the engine, so its wall-clock latency includes overhead an engine
# driven directly never pays. Latency compares only within a transport class (docs/methodology.md),
# so naming it is what stops a translator row being silently ranked against a direct one.
TRANSPORTS: tuple[str, ...] = ("http", "sdk", "in-process", "translator")
# How much retrieved context this target may hand the reader. "matched" is the default and the
# project's central fairness control (PRD_Memrank_v2.md:84, "fixed retrieval token budget"): every
# target is capped at the same --token-budget, so a row cannot win by dumping more text. "uncapped"
# exists solely for the full-context arm; "none" for the no-memory arm, which supplies nothing.
CONTEXT_BUDGETS: tuple[str, ...] = ("matched", "uncapped", "none")
_COMPONENT_FIELDS: frozenset[str] = frozenset({"provider", "model", "dims", "endpoint"})
_WORKSPACE_FIELDS: frozenset[str] = frozenset({"command", "requires"})
_WORKSPACE_PLACEHOLDERS: frozenset[str] = frozenset({"port"})
_INTERFACE_FIELDS: frozenset[str] = frozenset({"adapter", "transport"})
_BINDING_FIELDS: frozenset[str] = frozenset({"kind", "root", "rootFrom"})
_NETWORK_FIELDS: frozenset[str] = frozenset({"port", "readiness"})
_READINESS_FIELDS: frozenset[str] = frozenset({"path"})
_SOURCE_FIELDS: frozenset[str] = frozenset({
    "schema_version", "name", "kind", "interface", "binding", "launch", "network",
    "components", "context_budget", "retrieval", "ingest", "partitioning", "sdk_config", "secrets",
    "engine_env",
})


class ManifestError(MemrankError):
    """A manifest is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Component:
    """One configured component of a target (its embedder or its extraction LLM)."""

    provider: str | None = None
    model: str | None = None
    dims: int | None = None
    #: Where this component's provider is reached, for providers that have no fixed address.
    #:
    #: Every other provider carries its address implicitly -- a vendor's API is where the vendor
    #: says it is, `transformers` runs in-process, `regex` is the engine's own code. Only
    #: `openai-compatible` names a protocol rather than a place, so without this the URL travelled
    #: through the ambient environment and never reached the receipt: two runs against different
    #: deployments produced byte-identical component records, which is precisely the ambiguity a
    #: receipt exists to remove.
    endpoint: str | None = None


@dataclass(frozen=True)
class Engine:
    """The engine artifact a stack target runs.

    ``artifact`` names the image within the engines registry; the *placement* resolves registry and
    tag, following purl's rule that ``repository_url`` is a discovery hint while the digest is
    identity. ``port`` is the port the container itself listens on -- the image's own default, not a
    published host port, which the placement chooses.
    """

    artifact: str | None = None
    port: int = 8000


@dataclass(frozen=True)
class Workspace:
    """How Memrank starts this engine from a developer's source checkout."""

    command: str
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceBinding:
    """The local checkout a named development target executes.

    ``root`` is ``None`` only for a ``rootFrom: link`` binding this machine has not linked yet.
    That is a reportable state rather than a parse failure: ``targets ls`` and ``targets show`` must
    still describe a target nobody has linked, and the refusal belongs at the moment of running it.
    """

    kind: str
    root: str | None
    #: The target ref whose link supplies ``root``, or ``None`` when the descriptor states a path.
    link: str | None = None


@dataclass(frozen=True)
class Readiness:
    """The HTTP path which proves a launched source engine is ready."""

    path: str


@dataclass(frozen=True)
class Network:
    """The stable listen contract for a source-bound engine."""

    port: int
    readiness: Readiness


@dataclass(frozen=True)
class Manifest:
    """A fully-resolved description of one target."""

    name: str
    kind: str
    adapter: str
    transport: str | None = None
    context_budget: str = "matched"
    engine: Engine = field(default_factory=Engine)
    workspace: Workspace | None = None
    schema_version: int | None = None
    binding: SourceBinding | None = None
    launch: Workspace | None = None
    network: Network | None = None
    depends: tuple[str, ...] = ()
    components: dict[str, Component] = field(default_factory=dict)
    #: The Compose file declaring this target's containers, resolved beside the manifest.
    #: Absent for in-process targets, which have no containers at all.
    compose: str | None = None
    #: Which service in that file the adapter drives. Named rather than assumed, following the
    #: Dev Container spec's `service`: a graph's primary container is not derivable from its shape.
    service: str | None = None
    #: A shared declaration other manifests inherit from, not a system to run: it leaves one of its
    #: adapter's component knobs blank for each variant to fill, and a blank knob is not neutral --
    #: the engine picks its own default and no receipt records it. Submitting one is refused;
    #: resolving one still works, because inheritance depends on it.
    #:
    #: `mem0` was the worked example until 2026-08-18, when it took mem0's own embedder and became
    #: runnable. No BUILTIN target is abstract today; the field governs operator manifests reached
    #: through `targets.path` or `$MEMRANK_CONFIG_DIR/targets`, which share the same namespace.
    abstract: bool = False
    #: Credentials this target needs, as ``{name memrank resolves: variable the engine reads}``.
    #:
    #: memrank delivers these and interprets NOTHING about them. That is the point: a provider
    #: table can only express "one vendor, one key", so an engine wanting a user and a password, a
    #: client id plus a tenant, or a token under a name nobody guessed was simply unexpressible.
    #: Resolution goes through the one credential chokepoint (:func:`memrank.config.secret` --
    #: environment, then org, then wallet), which has always taken an arbitrary name; only the
    #: *derivation* of which names to ask for was hardcoded.
    #:
    #: ADDED TO what a target's providers imply, never replacing it: declaring
    #: ``llm: {provider: anthropic}`` still implies ``ANTHROPIC_API_KEY``, because that inference is
    #: right for the engines memrank ships and dropping it would churn every built-in for no gain.
    secrets: dict[str, list[str]] = field(default_factory=dict)
    #: Environment the engine process must run under, as ``{VAR: value}``, injected by every
    #: placement and interpreted by memrank not at all.
    #:
    #: For POLICY an engine exposes but a component cannot express. The case this field was
    #: built for: an engine whose extractor falls back to a cheaper deterministic one when its
    #: model call fails needs that fallback DISABLED, so a failed extraction is fatal rather than
    #: silently answered -- without it a run against a dead endpoint scores as though the model had
    #: replied, which happened twice before this field existed. That is true of the EVAL, not of
    #: the engine: a product server wants the fallback, because degraded memory beats a failed
    #: ingest. So it belongs to the target, not to `ENGINE_SETTINGS`, which is adapter-wide and
    #: would impose the policy on every variant of that engine.
    #:
    #: SERIALIZED, so a receipt records the policy a run was measured under. That is most of the
    #: value: `fallback_to_rules` currently exists only in the engine's own log, which memrank
    #: captures and never reads.
    #:
    #: Two things it may NOT carry, both enforced in :func:`_engine_env`: a key any component or
    #: declared secret already owns (a target could otherwise assert one configuration and run
    #: another), and anything credential-shaped (this dict is written to disk and synced -- use
    #: ``secrets:``).
    engine_env: dict[str, str] = field(default_factory=dict)
    #: Engine-specific retrieval settings, passed to the adapter verbatim.
    #:
    #: Deliberately NOT a fixed schema: engines do not agree on what retrieval even is. Hindsight
    #: takes a token budget and a fusion tier and has no top-k at all; mem0 takes top_k. A shared
    #: shape would have to invent a lowest common denominator and then translate, which is how a
    #: declared value stops matching what the engine received.
    #:
    #: Matched-mode targets leave this empty and let --token-budget do the capping. It exists for
    #: FAITHFUL variants (decision-matched-and-faithful-run-modes.md), which must reproduce a
    #: vendor's own depth -- and, being on the manifest, that depth lands in the receipt instead of
    #: living as a constant inside an adapter.
    retrieval: dict[str, Any] = field(default_factory=dict)
    #: Engine-specific settings that take effect when memory is WRITTEN, passed to the adapter
    #: verbatim. Same non-schema as ``retrieval`` above, for the same reason.
    #:
    #: Separate from ``retrieval`` because of when a setting can still change the answer, not
    #: because ingest and retrieval are different subsystems. A ``retrieval`` value is per-query:
    #: send a different ``budget`` on the next recall and the same stored memory yields a different
    #: result. An ingest value is decided before the first document lands and is unchangeable
    #: afterwards -- hindsight's ``enable_observations`` is fixed at bank creation and determines
    #: which of its four memory networks are ever written (audit F6).
    #:
    #: Recording the two under one key would be the misdescription this project exists to catch:
    #: two runs differing only in which memory types exist would read, in the receipt, as two runs
    #: differing in retrieval depth.
    #:
    #: Matched-mode targets leave this empty and get the engine's own defaults. It exists for
    #: FAITHFUL variants, where a vendor's benchmark harness configured ingest differently from the
    #: product they ship -- which is exactly hindsight's case.
    ingest: dict[str, Any] = field(default_factory=dict)
    #: How the engine scopes memory, for variants reproducing a vendor's own partitioning.
    #:
    #: Same non-schema as ``retrieval`` above, for the same reason: engines do not agree on what a
    #: partition IS. mem0's is a ``user_id`` and its published eval keeps one per SPEAKER,
    #: retrieving from each; hindsight's is a bank. There is no shared shape to validate against
    #: that would not have to be widened by the next engine.
    #:
    #: Hindsight needs no block here despite AMB looking like it partitions differently: AMB keys
    #: banks by USER (``bank_id_for(user_id)``), which is one bank per question on LongMemEval only
    #: because a LongMemEval unit IS a question. The runner's per-unit scoping already matches it.
    #:
    #: Matched-mode targets leave this empty and get one partition per benchmark unit, which is
    #: what makes cross-engine rows comparable -- every engine sees the same scope. It exists for
    #: FAITHFUL variants, where retrieving from two partitions instead of one is part of the
    #: configuration whose number we are trying to reproduce (audit F3/F13).
    partitioning: dict[str, Any] = field(default_factory=dict)
    #: Native configuration handed to an in-process SDK verbatim, for `transport: sdk` targets.
    #:
    #: The third block of this shape, and the one with the least memrank in it. `components` says
    #: what a target runs in memrank's vocabulary and `engine_env.py` renders that onto the
    #: variables a SERVER reads -- a translation that only works because memrank knows those
    #: variables. An in-process SDK has no environment to render onto: it takes a config object,
    #: and its shape belongs to the library.
    #:
    #: So this is passed through untouched, which is also the point. Reproducing a third party's
    #: run means expressing THEIR configuration -- memory-arena builds mem0 with Chroma, which no
    #: memrank component vocabulary describes and which the server image cannot do at all
    #: (`main.py:238` hardcodes pgvector, with no environment variable to change it).
    #:
    #: Empty for every containerised target, where `components` is the right vocabulary and this
    #: one would be a second way to say the same thing.
    sdk_config: dict[str, Any] = field(default_factory=dict)


def _retrieval(raw: Any) -> dict[str, Any]:
    """Validate the shape of a ``retrieval:`` block without policing its keys.

    Keys are the ENGINE's vocabulary, so this cannot check them against a list -- hindsight's
    ``budget``/``max_tokens`` and mem0's ``top_k`` are both correct, for different engines. What is
    checkable is that it is a mapping at all: a scalar or list here would reach an adapter as
    ``**{}`` or explode at call time, far from the manifest that caused it.

    An adapter that does not accept retrieval settings simply never receives them
    (memrank/targets/factory.py), so a stray block is inert rather than silently misapplied.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError(
            f"retrieval must be a mapping of engine-specific settings, got {type(raw).__name__}")
    return dict(raw)


def _ingest(raw: Any) -> dict[str, Any]:
    """Validate the shape of an ``ingest:`` block without policing its keys.

    Deliberately identical in spirit to :func:`_retrieval`: the keys are the ENGINE's vocabulary
    (``enable_observations`` for hindsight), so only the shape is checkable here.

    An adapter that does not accept ingest settings never receives them
    (memrank/targets/factory.py), so a stray block is inert. That is the right failure for this
    field and NOT for :func:`_partitioning`, which is refused instead: an unhonoured ingest setting
    leaves the engine on its own documented default, whereas an unhonoured partitioning block would
    let a receipt claim memory was split when it was not.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError(
            f"ingest must be a mapping of engine-specific settings, got {type(raw).__name__}")
    return dict(raw)


def _partitioning(raw: Any) -> dict[str, Any]:
    """Validate the shape of a ``partitioning:`` block without policing its keys.

    Deliberately identical in spirit to :func:`_retrieval`: what a partition means is the ENGINE's
    vocabulary (``by: speaker`` for mem0, a bank per question for hindsight), so only the shape is
    checkable here.

    An adapter that does not accept partitioning never receives it
    (memrank/targets/factory.py), so a block on the wrong target is inert rather than silently
    changing how memory is scoped -- which would be the worst possible failure for this field,
    since a mis-scoped run still produces a number.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError(
            f"partitioning must be a mapping of engine-specific settings, got {type(raw).__name__}")
    return dict(raw)


def _sdk_config(raw: Any) -> dict[str, Any]:
    """Validate the shape of an ``sdk_config:`` block without policing its keys.

    Same treatment as :func:`_retrieval` and :func:`_partitioning`, and for a stronger reason: this
    is not memrank's vocabulary at all, it is the SDK's. mem0's config nests a provider and its own
    sub-config under each of ``vector_store``, ``llm``, ``embedder`` and more; another library's
    would look nothing like it. Only the outermost shape is checkable here.

    Consistency with ``components`` is enforced at construction rather than here -- see
    ``memrank/targets/factory.py``, which refuses a target that declares both.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError(
            f"sdk_config must be a mapping passed verbatim to the SDK, got {type(raw).__name__}")
    _refuse_credentials(raw, "sdk_config")
    return dict(raw)


def _refuse_credentials(node: Any, path: str) -> None:
    """Reject any field whose NAME says it holds a credential, at any depth.

    A manifest is committed to git, so a credential written into one is already leaked -- checking
    at parse time is the last moment before that happens. The adapter fills keys at run time from
    :func:`memrank.config.secret`, so there is never a reason to put one here.

    Uses the same detector that redacts receipts (:mod:`memrank.secrets.names`) rather than a second
    list, because two definitions of "looks like a secret" would drift and the drifting one decides
    what leaks. It canonicalises spelling, so ``apiKey``, ``x-api-key`` and ``API_KEY`` are all
    caught while ``token_budget`` is not.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}"
            if is_secret_key(str(key)):
                raise ManifestError(
                    f"{here} looks like a credential, and a manifest is committed to source "
                    f"control. Declare the name under `secrets:` instead -- the adapter resolves "
                    f"the value at run time through the environment, the org, then the wallet.")
            _refuse_credentials(value, here)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _refuse_credentials(value, f"{path}[{index}]")


def _engine_env(raw: Any, *, owned: frozenset[str]) -> dict[str, str]:
    """Validate an ``engine_env:`` declaration: ``{VAR: value}``, both plain strings.

    Two refusals, and both exist because this field could otherwise make a receipt lie.

    A key already owned by a component or a declared secret is rejected. Without that, a target
    could declare ``llm: {model: am-slm-next}`` and then set ``AM_HG_EXTRACTOR_MODEL`` to something
    else here: the receipt would assert the first while the engine ran the second. That is the
    exact class of defect the endpoint and model fixes were about, so this field must not reopen it.

    A credential-shaped key is rejected because this dict IS serialized -- a receipt is written to
    disk and synced, and ``secrets:`` exists to carry names while resolving values at launch.

    Values are not otherwise interpreted. There is no table of known engine variables to check
    against, and inventing one would re-create the guessing that ``secrets:`` removed.

    Args:
        raw: The declaration as written.
        owned: Variable names the manifest already controls, which may not be overridden.

    Returns:
        The declaration as ``{VAR: value}``.

    Raises:
        ManifestError: On a non-mapping, a blank or non-string key or value, a collision with
            ``owned``, or a credential-shaped name.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError(
            f"engine_env must be a mapping of variable to value, got {type(raw).__name__}")
    out: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ManifestError(f"engine_env keys must be non-empty variable names, got {name!r}")
        if not isinstance(value, str):
            raise ManifestError(
                f"engine_env.{name} must be a string, got {type(value).__name__} -- quote it if it "
                f"is a number or a boolean, since an environment variable is always text")
        if name in owned:
            raise ManifestError(
                f"engine_env.{name} is already set from this target's components or secrets. "
                f"Overriding it here would let the manifest assert one configuration while the "
                f"engine ran another; change the component or the secret instead.")
        if is_secret_key(name):
            raise ManifestError(
                f"engine_env.{name} looks like a credential, and engine_env is written into the "
                f"receipt. Declare it under `secrets:`, which records the name and resolves the "
                f"value at launch.")
        out[name] = value
    return out


def _secrets(raw: Any) -> dict[str, list[str]]:
    """Normalize a ``secrets:`` declaration to ``{resolved name: [engine variables]}``.

    A list when the two names match -- the common case, where the engine reads the same variable
    memrank stores. A mapping when they differ, which is how a target expresses a rename without
    memrank holding a table of who calls what (hindsight's ``HINDSIGHT_API_LLM_API_KEY`` is the
    standing example).

    A mapping's value may be a LIST, because one credential can feed several variables: an engine
    whose extractor and embedder are the same deployment reads one token under two names, and a
    mapping to a single string cannot say so (the two entries would need the same key, and the
    later would silently win). OpenClaw's ``providerAuthEnvVars`` is an array per provider for the
    same reason.

    Names are not validated against anything. There is no list of known credentials to check
    against, and inventing one would re-create exactly the guessing this field removes.
    """
    if raw is None:
        return {}
    if isinstance(raw, list):
        entries = {}
        for name in raw:
            if not isinstance(name, str) or not name.strip():
                raise ManifestError(
                    f"secrets entries must be non-empty env-var names, got {name!r}")
            entries[name] = [name]
        return entries
    if isinstance(raw, dict):
        return {name: _secret_vars(name, value) for name, value in raw.items()}
    raise ManifestError(
        f"secrets must be a list of names or a mapping of name to engine variable(s), "
        f"got {type(raw).__name__}")


def _secret_vars(name: str, raw: Any) -> list[str]:
    """The engine variable(s) one declared credential is injected under."""
    values = raw if isinstance(raw, list) else [raw]
    if not values:
        raise ManifestError(
            f"secrets.{name} names no variable; drop the entry or name the one the engine reads")
    for var in values:
        if not isinstance(var, str) or not var.strip():
            raise ManifestError(
                f"secrets.{name} must name the variable(s) the engine reads, got {var!r}")
    return list(values)


def _component(role: str, raw: Any) -> Component:
    """Build one Component, rejecting unknown fields and an unstated embedder dimension."""
    if not isinstance(raw, dict):
        raise ManifestError(f"components.{role} must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _COMPONENT_FIELDS
    if unknown:
        raise ManifestError(f"components.{role} has unknown field(s): {', '.join(sorted(unknown))}")
    dims = raw.get("dims")
    if role == "embedder" and raw.get("model") and dims is None:
        raise ManifestError(
            "components.embedder.dims is required whenever components.embedder.model is set "
            "(dimensions do not follow from the model name; an inherited value would be wrong)")
    if dims is not None and not isinstance(dims, int):
        raise ManifestError(f"components.{role}.dims must be an integer, got {dims!r}")
    endpoint = raw.get("endpoint")
    if endpoint is not None and (not isinstance(endpoint, str) or not endpoint.strip()):
        raise ManifestError(
            f"components.{role}.endpoint must be a non-empty URL, got {endpoint!r}")
    return Component(provider=raw.get("provider"), model=raw.get("model"), dims=dims,
                     endpoint=endpoint)


def _workspace(raw: Any) -> Workspace | None:
    """Validate an optional native-source launcher without executing it."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ManifestError(f"workspace must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _WORKSPACE_FIELDS
    if unknown:
        raise ManifestError(f"workspace has unknown field(s): {', '.join(sorted(unknown))}")
    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ManifestError("workspace.command must be a non-empty string")
    placeholders = {name for _, name, _, _ in Formatter().parse(command) if name}
    unknown_placeholders = placeholders - _WORKSPACE_PLACEHOLDERS
    if unknown_placeholders:
        raise ManifestError(
            "workspace.command has unknown placeholder(s): "
            f"{', '.join(sorted(unknown_placeholders))}; known: port")
    # `requires` is optional -- it buys an early, clear error on a wrong checkout, and a launcher
    # with no distinctive marker file legitimately has none. Normalized BEFORE the type check
    # rather than with `or ()`, which turned both an omitted key and an explicit `[]` into a tuple
    # and then rejected them for not being a list.
    requires = raw.get("requires")
    if requires is None:
        requires = []
    if not isinstance(requires, list) or not all(isinstance(item, str) for item in requires):
        raise ManifestError("workspace.requires must be a list of relative paths")
    for required in requires:
        path = required.replace("\\", "/")
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise ManifestError(
                f"workspace.requires entry {required!r} must stay within the source directory")
    return Workspace(command=command, requires=tuple(requires))


def _only_fields(raw: Any, *, label: str, allowed: frozenset[str]) -> dict[str, Any]:
    """Return a mapping after rejecting misspelled fields."""
    if not isinstance(raw, dict):
        raise ManifestError(f"{label} must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - allowed
    if unknown:
        raise ManifestError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    return raw


def _root(raw: Any, base_dir: Path | None) -> str:
    """Resolve ``binding.root`` against the directory the descriptor was read from.

    Optional, defaulting to ``"."`` -- the descriptor's own directory. That is what lets a target
    file sit beside the translator it launches and be copied, shared, or committed verbatim: an
    absolute path baked into the file is true on exactly one machine.

    One rule covers every case because ``pathlib`` treats an absolute right-hand side as absolute,
    so ``base_dir / "/srv/engine"`` is still ``/srv/engine``. Without a ``base_dir`` (a manifest
    built from a mapping rather than read from disk) the path must already be absolute, since there
    is nothing to resolve against and a cwd-relative guess would mean a different directory
    depending on where the command was run from.
    """
    if raw is None:
        raw = "."
    if not isinstance(raw, str) or not raw:
        raise ManifestError("binding.root must be a path string")
    if base_dir is None:
        if not Path(raw).is_absolute():
            raise ManifestError(
                f"binding.root {raw!r} is relative and this manifest has no file to resolve it "
                f"against; use an absolute path")
        return str(Path(raw))
    return str((base_dir / raw).resolve())


def _binding_location(binding: dict[str, Any], base_dir: Path | None,
                      name: Any) -> tuple[str | None, str | None]:
    """Where this target's checkout is, and what supplied it.

    ``root:`` states a path in the file. ``rootFrom: link`` says the path is not the descriptor's to
    know -- this machine records it per target, under `memrank targets link`. The two are mutually
    exclusive, refused rather than ranked: a file carrying both would resolve differently depending
    on a precedence rule nobody reading it can see.

    An unlinked ``rootFrom: link`` yields ``(None, ref)`` rather than raising, so the catalog can
    still list and describe the target. Running it is what refuses.
    """
    declared_from = binding.get("rootFrom")
    if declared_from is None:
        return _root(binding.get("root"), base_dir), None
    if "root" in binding:
        raise ManifestError(
            "binding declares both root and rootFrom; use one -- root for a path this descriptor "
            "can state, rootFrom for a location only this machine knows")
    if declared_from != "link":
        raise ManifestError(f"unknown binding.rootFrom {declared_from!r}; known: link")
    if not isinstance(name, str) or not name:
        raise ManifestError("binding.rootFrom requires the manifest to declare a name")
    from memrank.targets import checkouts

    return checkouts.resolve(name), name


def _source_fields(data: dict[str, Any],
                   base_dir: Path | None) -> tuple[str, str, SourceBinding, Workspace, Network]:
    """Validate and normalize the complete source-target contract."""
    unknown = set(data) - _SOURCE_FIELDS
    if unknown:
        raise ManifestError(f"source target has unknown field(s): {', '.join(sorted(unknown))}")
    if data.get("schema_version") != 1:
        raise ManifestError("source targets require schema_version: 1")
    interface = _only_fields(
        data.get("interface"), label="interface", allowed=_INTERFACE_FIELDS)
    adapter, transport = interface.get("adapter"), interface.get("transport")
    if not adapter or not transport:
        raise ManifestError("interface requires adapter and transport")
    if transport not in TRANSPORTS:
        raise ManifestError(
            f"unknown transport {transport!r}; known: {', '.join(TRANSPORTS)}")
    binding = _only_fields(data.get("binding"), label="binding", allowed=_BINDING_FIELDS)
    if binding.get("kind") != "source":
        raise ManifestError("binding.kind must be 'source'")
    root, link = _binding_location(binding, base_dir, data.get("name"))
    launch = _workspace(data.get("launch"))
    if launch is None:
        raise ManifestError("source targets require launch")
    network = _only_fields(data.get("network"), label="network", allowed=_NETWORK_FIELDS)
    port = network.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ManifestError("network.port must be an integer from 1 to 65535")
    ready = _only_fields(
        network.get("readiness"), label="network.readiness", allowed=_READINESS_FIELDS)
    path = ready.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ManifestError("network.readiness.path must start with '/'")
    return adapter, transport, SourceBinding("source", root, link), launch, Network(
        port, Readiness(path))


def _validate_shape(data: dict[str, Any]) -> None:
    """Check the required top-level fields and the kind-specific constraints."""
    source = data.get("binding") is not None
    for required in (("name", "kind") if source else ("name", "kind", "adapter")):
        if not data.get(required):
            raise ManifestError(f"manifest is missing required field {required!r}")
    if data["kind"] not in KINDS:
        raise ManifestError(f"unknown kind {data['kind']!r}; known: {', '.join(KINDS)}")
    if source and data["kind"] == "in-process":
        raise ManifestError("in-process targets may not declare a source binding")
    if data.get("abstract") and data["kind"] == "in-process":
        raise ManifestError(
            f"{data['name']!r} is abstract and kind: in-process, which cannot both be true -- an "
            f"in-process arm has no components to leave for a variant to fill")
    if data["kind"] == "in-process" and data.get("depends"):
        raise ManifestError(
            f"{data['name']!r} is kind: in-process and may not declare depends "
            "(in-process targets run without containers)")
    transport = data.get("transport")
    if transport is not None and transport not in TRANSPORTS:
        raise ManifestError(
            f"unknown transport {transport!r}; known: {', '.join(TRANSPORTS)}")
    engine = data.get("engine")
    if engine is not None:
        if not isinstance(engine, dict):
            raise ManifestError(f"engine must be a mapping, got {type(engine).__name__}")
        unknown_engine = set(engine) - {"artifact", "port"}
        if unknown_engine:
            raise ManifestError(
                f"engine has unknown field(s): {', '.join(sorted(unknown_engine))}")
        if data["kind"] == "in-process":
            raise ManifestError(
                f"{data['name']!r} is kind: in-process and may not declare an engine")
    budget = data.get("context_budget")
    if budget is not None and budget not in CONTEXT_BUDGETS:
        raise ManifestError(
            f"unknown context_budget {budget!r}; known: {', '.join(CONTEXT_BUDGETS)}")
    if data["kind"] == "in-process" and (data.get("compose") or data.get("service")):
        raise ManifestError(
            f"{data['name']!r} is kind: in-process and may not declare compose/service "
            "(in-process targets run without containers)")
    if data["kind"] == "in-process" and data.get("secrets"):
        raise ManifestError(
            f"{data['name']!r} is kind: in-process and may not declare secrets -- it runs inside "
            f"memrank, so there is no engine process to inject credentials into. A judged run's "
            f"own judge key is a run-level requirement, gathered separately.")
    if data["kind"] == "in-process" and data.get("workspace"):
        raise ManifestError(
            f"{data['name']!r} is kind: in-process and may not declare workspace "
            "(there is no engine process to launch)")
    if data.get("compose") and not data.get("service"):
        raise ManifestError(
            f"{data['name']!r} declares a compose file but no service: the adapter has to be told "
            "which container it drives, and a graph's primary service is not derivable from its "
            "shape (see the Dev Container spec's `service`)")
    if source and any(data.get(field) for field in (
            "adapter", "transport", "engine", "workspace", "compose", "service", "depends")):
        raise ManifestError(
            "source targets use interface/binding/launch/network and may not declare legacy "
            "adapter, transport, engine, workspace, compose, service, or depends fields")


def from_dict(data: dict[str, Any], *, base_dir: Path | None = None) -> Manifest:
    """Build and validate a Manifest from a raw mapping.

    Args:
        data: The flattened manifest mapping (inheritance and overrides already applied).
        base_dir: The directory the descriptor was read from. Used for exactly one thing --
            resolving a relative ``binding.root`` -- so a target file can sit beside the translator
            it launches and stay true wherever the directory is copied to.

    Returns:
        The validated Manifest.

    Raises:
        ManifestError: On any missing, unknown, or inconsistent field.
    """
    _validate_shape(data)
    source = data.get("binding") is not None
    source_fields = _source_fields(data, base_dir) if source else None
    raw_components = data.get("components") or {}
    unknown_roles = set(raw_components) - set(COMPONENT_ROLES)
    if unknown_roles:
        raise ManifestError(
            f"unknown component role {sorted(unknown_roles)[0]!r}; "
            f"known: {', '.join(COMPONENT_ROLES)}")
    adapter = source_fields[0] if source_fields else data["adapter"]
    transport = source_fields[1] if source_fields else data.get("transport")
    binding = source_fields[2] if source_fields else None
    launch = source_fields[3] if source_fields else None
    network = source_fields[4] if source_fields else None
    secrets = _secrets(data.get("secrets"))
    return Manifest(
        name=data["name"],
        kind=data["kind"],
        adapter=adapter,
        transport=transport,
        context_budget=data.get("context_budget") or "matched",
        engine=(Engine(port=network.port) if network else Engine(**(data.get("engine") or {}))),
        workspace=launch or _workspace(data.get("workspace")),
        schema_version=data.get("schema_version"),
        binding=binding,
        launch=launch,
        network=network,
        depends=tuple(data.get("depends") or ()),
        components={role: _component(role, raw) for role, raw in raw_components.items()},
        compose=data.get("compose"),
        service=data.get("service"),
        abstract=bool(data.get("abstract", False)),
        secrets=secrets,
        # Variables the declared secrets already deliver are off limits here. The adapter's own
        # component variables are refused too, but that check lives in `engine_env.py` where the
        # per-adapter table is -- importing it back into this module would be circular.
        engine_env=_engine_env(
            data.get("engine_env"),
            owned=frozenset(var for variables in secrets.values() for var in variables),
        ),
        retrieval=_retrieval(data.get("retrieval")),
        ingest=_ingest(data.get("ingest")),
        partitioning=_partitioning(data.get("partitioning")),
        sdk_config=_sdk_config(data.get("sdk_config")),
    )


def _component_dict(component: Component) -> dict[str, Any]:
    """One component, with ``endpoint`` present only when it was stated.

    An unstated endpoint is omitted rather than serialized as ``null`` so that adding the field did
    not change :func:`definition_digest` for every target that predates it -- a digest identifies
    the definition a run used, and a schema addition is not a definition change.
    """
    out = asdict(component)
    if out.get("endpoint") is None:
        del out["endpoint"]
    return out


def to_dict(target: Manifest, *, include_local_binding: bool = True) -> dict[str, Any]:
    """Serialize a target without exposing internal source-normalization fields."""
    if target.binding is None:
        whole = asdict(target)
        whole["components"] = {role: _component_dict(component)
                               for role, component in target.components.items()}
        # Omitted when unstated, for the reason `_component_dict` omits an unstated endpoint: a
        # digest identifies the definition a run used, and adding a field to the schema is not a
        # change to the definition of every target that predates it
        # (tests/targets/test_component_endpoint.py pins this).
        if not target.ingest:
            whole.pop("ingest", None)
        if not target.partitioning:
            whole.pop("partitioning", None)
        if not target.sdk_config:
            whole.pop("sdk_config", None)
        if not target.engine_env:
            whole.pop("engine_env", None)
        return whole
    assert target.launch is not None and target.network is not None
    binding: dict[str, Any] = {"kind": target.binding.kind}
    if include_local_binding and target.binding.root is not None:
        binding["root"] = target.binding.root
    if target.binding.link is not None:
        binding["rootFrom"] = "link"
    out: dict[str, Any] = {
        "schema_version": target.schema_version,
        "name": target.name,
        "kind": target.kind,
        "interface": {"adapter": target.adapter, "transport": target.transport},
        "binding": binding,
        "launch": asdict(target.launch),
        "network": asdict(target.network),
        "components": {role: _component_dict(component)
                       for role, component in target.components.items()},
        "context_budget": target.context_budget,
    }
    if target.secrets:
        # Names only. The values are resolved at launch and never enter a manifest, a receipt, or
        # anything that syncs.
        out["secrets"] = dict(target.secrets)
    if target.engine_env:
        # Values included, unlike `secrets`: this is the policy a run was measured under, and a
        # receipt that omits it cannot say whether a failed extraction was allowed to be answered
        # by something other than the extractor under test. `_engine_env` refuses
        # credential-shaped keys precisely because this line writes them out.
        out["engine_env"] = dict(target.engine_env)
    if target.retrieval:
        out["retrieval"] = dict(target.retrieval)
    if target.ingest:
        out["ingest"] = dict(target.ingest)
    if target.partitioning:
        out["partitioning"] = dict(target.partitioning)
    if target.sdk_config:
        out["sdk_config"] = dict(target.sdk_config)
    return out


def definition_digest(*targets: Manifest) -> str:
    """Identify the exact local definitions used by a background source run.

    Order-sensitive and covers every source target in a sweep: the guard's job is to prove the
    background process re-read what the foreground did, and digesting only the first would leave
    the rest free to change in between.
    """
    payload = json.dumps([to_dict(target) for target in targets],
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
