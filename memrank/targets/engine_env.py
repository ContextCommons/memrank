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
"""The env vars that configure an engine's components -- named in exactly one place.

Three surfaces must agree on these names: the engine process itself, the local compose renderer, and
the cloud taskdef renderer. Before this table that agreement was a coincidence. It held for mem0
only because mem0's server happens to read the same names the adapter convention assumes, and it
already failed elsewhere: hindsight reads ``HINDSIGHT_API_LLM_MODEL`` while the convention would
produce ``HINDSIGHT_LLM_MODEL``, and atomicmemory reads a bare ``EMBEDDING_MODEL``.

Per the `CLAUDE.md` chokepoint rule, the names live here and both renderers derive from them, rather
than each restating its own copy. What this function returns is the *only* part of an engine's
environment that must be byte-identical across placements; network topology (hostnames, ports,
service discovery) legitimately differs and is the placement's own business.
"""

from __future__ import annotations

from dataclasses import dataclass

from memrank.docs import doc_url
from memrank.errors import MemrankError
from memrank.targets.manifest import Manifest

# adapter -> {(component role, field) -> the env var the ENGINE PROCESS reads}
#
# Transcribed from the ECS task definitions, which are what the running cloud engines actually got:
#   mem0          deploy/ecs/taskdef.mem0.json.tpl:60-64
#   hindsight     deploy/ecs/taskdef.hindsight.json.tpl:44-45
#   atomicmemory  deploy/ecs/taskdef.atomicmemory.json.tpl:49-53
#
# An adapter present with an EMPTY mapping is a positive statement -- "this engine exposes no
# component knobs" -- and is different from an adapter that is absent, which is an error.
ENGINE_ENV: dict[str, dict[tuple[str, str], str]] = {
    "mem0": {
        ("llm", "provider"): "MEM0_LLM_PROVIDER",
        ("llm", "model"): "MEM0_LLM_MODEL",
        ("embedder", "provider"): "MEM0_EMBEDDER_PROVIDER",
        ("embedder", "model"): "MEM0_EMBEDDER_MODEL",
        ("embedder", "dims"): "MEM0_EMBEDDING_DIMS",
        # The fork's server (mem0-internal server/main.py) passes these to the OpenAI SDK as
        # its base URL for an openai-family provider.
        ("llm", "endpoint"): "MEM0_LLM_ENDPOINT",
        ("embedder", "endpoint"): "MEM0_EMBEDDER_ENDPOINT",
    },
    # No embedder knob: hindsight exposes only its extraction LLM
    # (docs/research/2026-07-29-memory-engine-configurability-and-forks.md).
    "hindsight": {
        ("llm", "provider"): "HINDSIGHT_API_LLM_PROVIDER",
        ("llm", "model"): "HINDSIGHT_API_LLM_MODEL",
        # Read by 0.6.2's config.py; its OpenAI-compatible providers send to it.
        ("llm", "endpoint"): "HINDSIGHT_API_LLM_BASE_URL",
    },
    # Unprefixed by design -- the engine owns its whole container, so its config vars are not
    # namespaced. They must never be read out of memrank's own environment for that reason.
    # The endpoint pair is what core reads for its `openai-compatible` provider on either side
    # (packages/core/src/config.ts:1342, services/llm.ts:419, services/embedding.ts:281). Naming
    # them here is what moves a component's address out of the ambient environment and into the
    # manifest, so the receipt can say WHICH deployment served a run.
    "atomicmemory": {
        ("llm", "provider"): "LLM_PROVIDER",
        ("llm", "model"): "LLM_MODEL",
        ("llm", "endpoint"): "LLM_API_URL",
        ("embedder", "provider"): "EMBEDDING_PROVIDER",
        ("embedder", "model"): "EMBEDDING_MODEL",
        ("embedder", "dims"): "EMBEDDING_DIMENSIONS",
        ("embedder", "endpoint"): "EMBEDDING_API_URL",
    },
    "supermemory": {},
    # Empty for a different reason than supermemory's: not "this engine has no knobs" but "memrank
    # cannot know which variables a stranger's engine reads, so it must not pretend to configure
    # one". A native target therefore declares no components at all; the translator configures its
    # engine in its own launch environment and REPORTS the result via the contract's describe
    # endpoint, which is cross-checked by targets/factory.py. Declaring a component here would be
    # a promise memrank cannot keep -- component_env() below refuses it for exactly that reason.
    "native": {},
}


# adapter -> the env var THE HARNESS reads to find the engine. Not a uniform "{PREFIX}HTTP_URL":
# only mem0 uses that shape, and assuming it configured hindsight with a variable it never reads.
BASE_URL_ENV: dict[str, str] = {
    "mem0": "MEM0_HTTP_URL",
    "hindsight": "HINDSIGHT_API_URL",
    "atomicmemory": "ATOMICMEMORY_API_URL",
    "supermemory": "SUPERMEMORY_BASE_URL",
    # Where the translator is listening, not where the engine is -- memrank never learns the
    # engine's own address, which is the point.
    "native": "NATIVE_API_URL",
}

# adapter -> engine settings that are true WHEREVER it runs. Neighbour addresses are deliberately
# absent: a hostname is topology, differs per placement, and is the placement's own business.
ENGINE_SETTINGS: dict[str, dict[str, str]] = {
    "mem0": {
        "AUTH_DISABLED": "true",
        "MEM0_TELEMETRY": "false",
        "HISTORY_DB_PATH": "/tmp/mem0-history.db",
        "APP_DB_NAME": "mem0_app",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "postgres",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_COLLECTION_NAME": "memories",
    },
    "hindsight": {
        "HINDSIGHT_DATA_DIR": "/tmp/hindsight-data",
        "HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES": "8",
    },
    "atomicmemory": {"DATABASE_URL": "embedded", "RAW_STORAGE_DEPLOYMENT_ENV": "local"},
    "supermemory": {"SUPERMEMORY_DATA_DIR": "/tmp/supermemory"},
    # A translator's settings are its author's business and travel in its own launch environment.
    # memrank injecting anything here would be guessing at a process it did not write.
    "native": {},
}

# adapter -> {canonical credential name -> the env var THE ENGINE reads for it}. Absent entries mean
# the engine reads the canonical name unchanged; only hindsight renames one.
#
# This applies to credentials memrank DERIVES from a target's providers. A target can now state a
# rename itself -- `secrets: {ANTHROPIC_API_KEY: HINDSIGHT_API_LLM_API_KEY}` says the same thing
# without memrank holding the fact -- and a declared entry wins, being the more specific statement.
# Kept because the derivation is still right for the engines memrank ships: migrating them would
# churn nine manifests and their preflight tests for no change in behaviour.
SECRET_ENV: dict[str, dict[str, str]] = {
    "hindsight": {"ANTHROPIC_API_KEY": "HINDSIGHT_API_LLM_API_KEY"},
}


# adapter -> the shell command that starts its engine, overriding the image's own CMD, or None when
# the image already starts correctly. Takes {port}.
#
# ``None`` is a positive statement -- "this image's CMD is right" -- and is different from an adapter
# being absent, which is an error. Deciding is the point: the compose renderer used to apply mem0's
# command to EVERY engine because mem0 was the only stack target when it was written, so
# `--on local` told a Node image to run a Python migration tool. Cloud guarded the same line with
# `if target.adapter == "mem0"` and local did not, which is exactly the kind of divergence a table
# read by both renderers makes impossible.
ENGINE_COMMAND: dict[str, str | None] = {
    # The published image's default CMD starts uvicorn WITHOUT migrating, which leaves the schema
    # absent and every endpoint returning 500. A property of the image, not of either placement.
    "mem0": "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port {port}",
    "hindsight": None,
    "atomicmemory": None,
    "supermemory": None,
    # Native targets are source-bound and start from the manifest's own `launch.command`, so no
    # image CMD is ever overridden. None states that decision rather than leaving the table silent.
    "native": None,
}


def engine_command(target: Manifest) -> list[str] | None:
    """The command that starts ``target``'s engine, or ``None`` to keep the image's own CMD.

    Args:
        target: A resolved manifest.

    Returns:
        An exec-form command ready for compose or ECS, or ``None`` when the image starts itself.

    Raises:
        EngineEnvError: When the adapter has no entry. Inheriting another engine's command does not
            fail loudly at render time -- it fails at container start, looking like a broken image.
    """
    if target.adapter not in ENGINE_COMMAND:
        raise EngineEnvError(
            f"no start command decision for adapter {target.adapter!r}; add one to ENGINE_COMMAND "
            f"in memrank/targets/engine_env.py, using None if the image's own CMD is correct. "
            f"Defaulting would silently give it whichever engine's command was written first.")
    template = ENGINE_COMMAND[target.adapter]
    if template is None:
        return None
    return ["sh", "-c", template.format(port=target.engine.port)]


# adapter -> how to ask its engine "are you ready?", as a shell probe against {port}.
#
# Transcribed from the retired ECS templates. NOT generalisable: mem0 answers /configure, hindsight
# and atomicmemory answer /health, supermemory answers /. Two of them have no curl in the image and
# must probe with node. Assuming mem0's endpoint for all four left hindsight's task stuck in
# PENDING -- the engine container ran fine, its health check probed a path it does not serve, and
# the harness waited on `dependsOn: HEALTHY` until the task was killed.
#
# `start_period` differs per engine too: how long a build takes to become ready is a property of
# the build, not something to standardise.
_CURL_PROBE = "curl -fsS http://localhost:{port}{path} || exit 1"
# atomicmemory and supermemory images carry node but not curl.
_NODE_PROBE = ("node -e \"fetch('http://localhost:{port}{path}')"
               ".then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))\"")

@dataclass(frozen=True)
class Readiness:
    """How to ask one engine whether it is ready, and how long to allow it."""

    probe: str          # a shell command template taking {port} and {path}
    path: str
    start_period: int   # seconds ECS allows before failing health checks count


# How a health check is retried once the start period is over. ECS semantics: `start_period` is a
# GRACE window in which failures do not count, after which `retries` failures at `interval` seconds
# are still tolerated. So the real budget an engine gets is start_period + interval * retries -- not
# start_period, which is what a placement reading only that field would wrongly allow.
HEALTHCHECK_INTERVAL_S = 15
HEALTHCHECK_RETRIES = 10


READINESS: dict[str, Readiness] = {
    # /configure is also what describe_engine() calls, so "ready" means "can answer the question
    # memrank actually asks". /auth/setup-status returns 500 until the schema exists.
    "mem0": Readiness(_CURL_PROBE, "/configure", 180),
    "hindsight": Readiness(_CURL_PROBE, "/health", 120),
    "atomicmemory": Readiness(_NODE_PROBE, "/health", 180),
    "supermemory": Readiness(_NODE_PROBE, "/", 150),
    # describe doubles as the readiness probe, following mem0's precedent above: "ready" must mean
    # "can answer the question memrank is about to ask". A translator that answered before its
    # engine could serve would turn an engine start-up failure into a benchmark result. The grace
    # period is generous because memrank cannot know what the translator is starting behind it.
    "native": Readiness(_CURL_PROBE, "/memrank/v1/describe", 180),
}


def readiness_probe(target: Manifest) -> tuple[str, int]:
    """The shell command that tells whether ``target``'s engine is ready, and its start period.

    Raises:
        EngineEnvError: When the adapter has no entry. Guessing an endpoint does not fail loudly --
            it hangs, because the harness waits on a health check that can never pass.
    """
    spec = READINESS.get(target.adapter)
    if spec is None:
        raise EngineEnvError(
            f"no readiness probe for adapter {target.adapter!r}; add one to READINESS in "
            f"memrank/targets/engine_env.py. Guessing an endpoint does not fail loudly, it hangs: "
            f"the harness waits on a health check that never passes until the task is killed.")
    return spec.probe.format(port=target.engine.port, path=spec.path), spec.start_period


def readiness_budget_s(target: Manifest) -> int:
    """Total seconds ``target``'s engine may take to become ready, on ANY placement.

    Cloud expresses this as a grace period plus a retry policy; a placement that polls directly
    needs the sum. Reading ``start_period`` alone gave a local run a strictly shorter budget than
    the same engine gets in the cloud -- so an engine could be declared dead locally and healthy
    remotely, which is the same class of divergence the component gate exists to prevent.
    """
    _, start_period = readiness_probe(target)
    return start_period + HEALTHCHECK_INTERVAL_S * HEALTHCHECK_RETRIES


def readiness_path(target: Manifest) -> str:
    """The HTTP path that tells whether ``target``'s engine is ready.

    The same per-engine fact :func:`readiness_probe` wraps in a shell command; the local placement
    polls it with httpx instead, so it needs the path alone.
    """
    spec = READINESS.get(target.adapter)
    if spec is None:
        raise EngineEnvError(
            f"no readiness path for adapter {target.adapter!r}; add one to READINESS in "
            f"memrank/targets/engine_env.py.")
    return spec.path


# How long an adapter waits on one HTTP call. Whole-document ingest of a large doc (BEAM) is ONE
# synchronous call whose LLM extraction runs for minutes; the adapters' own 60s default aborts it
# mid-run. 900 is what scripts/cloud-run.sh has always used. Dropping this was invisible on
# `demo --slice smoke` and would have failed the first real benchmark, looking like an engine fault.
ENGINE_TIMEOUT_S = 900

# A bearer token two containers in the same task must agree on. NOT a credential -- it never leaves
# the task -- so it is a fixed dev value rather than a wallet entry, matching cloud-run.sh.
INTRA_TASK_TOKEN = "local-dev-key"
# adapter -> (harness var, engine var). One value, two containers, two names. Omitting the pair
# cost a live 401: the harness sent no token and the engine rejected the first call.
PAIRED_TOKENS: dict[str, tuple[str, str]] = {
    "atomicmemory": ("ATOMICMEMORY_API_KEY", "CORE_API_KEY"),
}

# adapter -> the provenance suffixes ITS adapter reads, appended to {ADAPTER}_. Per engine because
# they differ: only mem0 records source describe/dirty (it is built from a git fork), only
# supermemory records a binary version. Transcribed from the retired templates.
# Two digests, not one. The INDEX digest names the release -- one document listing a manifest per
# platform -- and the PLATFORM digest names the binary that executed. `hindsight:latest` on an arm64
# laptop and on Fargate share the first and differ in the second: same release, different machine
# code. With one digest those are indistinguishable from a genuinely different build, which is what
# made a comparison across machines unreadable.
PROVENANCE_FIELDS: dict[str, tuple[str, ...]] = {
    "mem0": ("ENGINE_VERSION", "ENGINE_IMAGE_DIGEST", "ENGINE_IMAGE_INDEX_DIGEST",
             "ENGINE_IMAGE_PLATFORM", "ENGINE_IMAGE_REPO", "ENGINE_SOURCE_SHA",
             "ENGINE_SOURCE_DESCRIBE", "ENGINE_SOURCE_DIRTY"),
    "hindsight": ("ENGINE_VERSION", "ENGINE_IMAGE_DIGEST", "ENGINE_IMAGE_INDEX_DIGEST",
             "ENGINE_IMAGE_PLATFORM", "ENGINE_IMAGE_REPO"),
    "atomicmemory": ("ENGINE_VERSION", "ENGINE_IMAGE_DIGEST", "ENGINE_IMAGE_INDEX_DIGEST",
             "ENGINE_IMAGE_PLATFORM", "ENGINE_IMAGE_REPO",
                     "ENGINE_SOURCE_SHA"),
    "supermemory": ("ENGINE_VERSION", "ENGINE_IMAGE_DIGEST", "ENGINE_IMAGE_INDEX_DIGEST",
             "ENGINE_IMAGE_PLATFORM", "ENGINE_IMAGE_REPO",
                    "ENGINE_BINARY_VERSION"),
    # Source fields only. There is no image to pin: memrank launches a translator from a checkout
    # and never sees the engine behind it, so an image digest here would name the wrong artifact.
    # What IS identifiable -- the translator's commit and working-tree delta -- the workspace
    # placement injects directly.
    "native": ("ENGINE_VERSION", "ENGINE_SOURCE_SHA"),
}
# supermemory publishes a binary version distinct from its image tag; cloud-run.sh defaulted this.
_SUPERMEMORY_BINARY_VERSION = "0.0.3"


def harness_env(target: Manifest, *, url: str, engines_repo: str = "", tag: str = "",
                digest: str = "", index_digest: str = "",
                platform: str = "") -> dict[str, str]:
    """Everything the memrank container needs to talk to, and describe, this engine.

    Assembled HERE rather than by the caller. It used to be a dict built at the call site, and that
    is precisely how five variables went missing when the renderer replaced the ECS templates --
    including the request timeout, whose absence is silent until a long benchmark aborts mid-ingest.
    The renderer now owns the whole harness environment, the same argument that makes the drift
    gate work for components.

    Args:
        target: The resolved manifest.
        url: Where the engine is reachable from the harness.
        engines_repo: Registry the engine image came from -- a discovery hint.
        tag: The engine image tag; doubles as the source SHA for engines built from a git fork.
        digest: The PLATFORM digest -- the binary that actually executed. This, not the tag, is
            what a receipt pins.
        index_digest: The multi-arch index digest, i.e. which release was requested. Empty for a
            single-architecture image, which has no index; absent and equal are different facts.
        platform: The ``os/arch`` the image ran as, so a reader knows which of two rows sharing a
            release is comparable on latency and which only on quality.

    Returns:
        ``{ENV_VAR: value}`` for the harness container.
    """
    env = dict(base_url_env(target, url))
    prefix = target.adapter.upper()
    env[f"{prefix}_TIMEOUT_S"] = str(ENGINE_TIMEOUT_S)

    values = {
        # The tag as DECLARED, not reassembled. This used to be f"{artifact}-{tag}", rebuilding
        # our own ECR naming convention from two halves -- which produced "mem0-mem0-poc-1", a
        # version string naming a build that does not exist, when the adapter was used in place of
        # the artifact. It also had no meaning for an image we do not publish: a vendor's tag is
        # "latest", not "<something>-latest".
        "ENGINE_VERSION": tag,
        "ENGINE_IMAGE_DIGEST": digest,
        "ENGINE_IMAGE_INDEX_DIGEST": index_digest,
        "ENGINE_IMAGE_PLATFORM": platform,
        "ENGINE_IMAGE_REPO": engines_repo,
        # The fork's short SHA IS the tag for engines we build; cloud-run.sh did the same.
        "ENGINE_SOURCE_SHA": tag,
        "ENGINE_SOURCE_DESCRIBE": "",     # only a local checkout can produce `git describe`
        "ENGINE_SOURCE_DIRTY": "false",   # a published image is never built from a dirty tree
        "ENGINE_BINARY_VERSION": _SUPERMEMORY_BINARY_VERSION,
    }
    for suffix in PROVENANCE_FIELDS.get(target.adapter, ()):
        env[f"{prefix}_{suffix}"] = values[suffix]

    paired = PAIRED_TOKENS.get(target.adapter)
    if paired:
        env[paired[0]] = INTRA_TASK_TOKEN
    return env


def engine_token_env(target: Manifest) -> dict[str, str]:
    """The engine side of a shared intra-task token, under the name the ENGINE reads."""
    paired = PAIRED_TOKENS.get(target.adapter)
    return {paired[1]: INTRA_TASK_TOKEN} if paired else {}


class EngineEnvError(MemrankError):
    """A target declares a component this engine has no env var for."""


def base_url_env(target: Manifest, url: str) -> dict[str, str]:
    """The single env var telling the harness where ``target``'s engine is.

    Raises:
        EngineEnvError: When the adapter's variable is unknown -- guessing it would leave the
            adapter pointed at its compiled-in default while the engine ran somewhere else.
    """
    var = BASE_URL_ENV.get(target.adapter)
    if var is None:
        raise EngineEnvError(
            f"no base-URL env var known for adapter {target.adapter!r}; add one to "
            f"memrank/targets/engine_env.py so the harness can be told where its engine is.")
    return {var: url}


def secret_env_name(adapter: str, canonical: str) -> str:
    """What ``adapter``'s engine calls the credential memrank knows as ``canonical``."""
    return SECRET_ENV.get(adapter, {}).get(canonical, canonical)


def engine_secret_vars(target: Manifest) -> dict[str, list[str]]:
    """Every credential ``target`` needs: ``{name memrank resolves: [variables the engine reads]}``.

    Both placements derive from this rather than each computing its own set. They did not, and the
    two answers had drifted: the local renderer honoured a target's declared ``secrets:`` while the
    cloud renderer built its list from the provider table alone, so a target that declared its own
    credentials launched correctly under ``--on local`` and reached Fargate with those credentials
    absent. Nothing failed loudly -- the wallet preflight passes on names the taskdef then omits.

    Args:
        target: A resolved manifest.

    Returns:
        Derived names first, then declared. A declared entry REPLACES the derived variables for the
        same credential: a target naming its variables explicitly has said something more specific
        than an adapter-wide rename table can.
    """
    from memrank.secrets import requirements

    providers = requirements.keyed_providers(target.components)
    derived = requirements.required_secrets(
        target.adapter, embedder=providers.get("embedder"), llm=providers.get("llm"))
    out: dict[str, list[str]] = {
        name: [secret_env_name(target.adapter, name)] for name in derived}
    out.update({name: list(variables) for name, variables in target.secrets.items()})
    return out


def unstated_components(target: Manifest) -> list[str]:
    """Roles this adapter can be configured for that the manifest says nothing about.

    A knob left blank does not stay blank. `mem0` exposes an embedder and declared none, so the
    engine filled it with `text-embedding-3-small` -- a real OpenAI call per document, recorded in
    no manifest and no receipt. The run either dies on a missing key or, worse, succeeds while
    measuring something nobody chose.

    Only ROLES are checked, not every field: an adapter that names an embedder provider and model
    but no dims has stated its intent, and the missing dimension is caught by the manifest schema.

    Args:
        target: A resolved manifest.

    Returns:
        Sorted role names, empty when the manifest is complete.
    """
    names = ENGINE_ENV.get(target.adapter)
    if not names:
        return []
    exposed = {role for role, _ in names}
    return sorted(exposed - set(target.components))


def component_env(target: Manifest) -> dict[str, str]:
    """The env vars configuring ``target``'s components, identical on every placement.

    Args:
        target: A resolved manifest.

    Returns:
        ``{ENV_VAR: value}`` for every component field the manifest states. Unstated fields are
        omitted rather than blanked, because an empty string is a value the engine would act on.

    Raises:
        EngineEnvError: When the adapter has no table, or states a component field the table cannot
            name. Both are silent-drift bugs: the manifest would promise a configuration the engine
            never received, and the receipt would assert it anyway.
    """
    names = ENGINE_ENV.get(target.adapter)
    if names is None:
        raise EngineEnvError(
            f"no engine-env table for adapter {target.adapter!r}; add one to "
            f"memrank/targets/engine_env.py naming the vars its process reads, otherwise its "
            f"declared components would never reach it.")

    unstated = unstated_components(target)
    if unstated and not target.abstract:
        raise EngineEnvError(
            f"{target.name!r} leaves {', '.join(unstated)} unstated, and its adapter exposes "
            f"{'that knob' if len(unstated) == 1 else 'those knobs'}. A blank knob is not neutral: "
            f"the engine picks its own default and the receipt reports nothing, so the run measures "
            f"a configuration nobody declared. Declare it, or mark the manifest `abstract: true` if "
            f"it is a base for variants to fill.")

    env: dict[str, str] = {}
    for role, component in sorted(target.components.items()):
        for fld in ("provider", "model", "dims", "endpoint"):
            value = getattr(component, fld)
            if value is None:
                continue
            var = names.get((role, fld))
            if var is None:
                raise EngineEnvError(
                    f"{target.name!r} declares {role}.{fld} but {target.adapter!r} has no env var "
                    f"for it, so the value would be dropped on the way to the engine. Either add "
                    f"the name to ENGINE_ENV or stop declaring it."
                    + (
                        " A native target declares no components at all: memrank does not know "
                        "which variables your engine reads, so your launch command configures it "
                        "and your translator reports the result from /memrank/v1/describe "
                        f"({doc_url('system-contract.md')} section 4)."
                        if target.adapter == "native" else ""))
            env[var] = str(value)
    return env


def declared_env(target: Manifest) -> dict[str, str]:
    """A target's ``engine_env:`` block, refused if it would contradict its own components.

    `manifest.py` validates the shape and rejects collisions with declared secrets, but it cannot
    see `ENGINE_ENV` -- importing this module there would be circular. So the adapter-specific half
    of the guard lives here, at the point both placements assemble the environment.

    The collision matters because the two halves disagree silently: a target declaring
    ``llm: {model: am-slm-next}`` and ``engine_env: {AM_HG_EXTRACTOR_MODEL: other}`` would put
    `other` on the wire while the receipt asserted `am-slm-next`.

    Args:
        target: A resolved manifest.

    Returns:
        ``{VAR: value}``, empty when nothing is declared.

    Raises:
        EngineEnvError: When a declared variable is one this adapter derives from components.
    """
    if not target.engine_env:
        return {}
    owned = set(component_env(target))
    if collisions := sorted(set(target.engine_env) & owned):
        raise EngineEnvError(
            f"{target.name!r} sets {', '.join(collisions)} in engine_env, but {target.adapter!r} "
            f"derives {'that variable' if len(collisions) == 1 else 'those variables'} from its "
            f"components. The manifest would assert one configuration and the engine would run "
            f"another; change the component instead.")
    return dict(target.engine_env)
