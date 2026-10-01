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
"""Central configuration for memrank services.

Exposes exactly two least-privilege database URLs:

- ``database_url()`` -- publisher/leaderboard role (read + limited write);
  used by leaderboard-publish and leaderboard-gc.
- ``ingest_database_url()`` -- ingest role (append-only on both tables);
  used ONLY by leaderboard-ingest; never the publisher's DATABASE_URL.

No fallback values. Missing required configuration raises ``ConfigError`` loudly.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from memrank.errors import MemrankError

_DOTENV_LOADED = False

# The CLI's own env file, anchored to THIS package rather than the invoker's working
# directory. A cwd-relative `.env` made the repo root the CLI's de facto config file --
# the exact filename and DATABASE_URL variable dbmate auto-loads, so running `dbmate`
# from the root silently picked up the leaderboard publisher's DSN as a migration DSN.
_DEFAULT_ENV_FILE = Path(__file__).resolve().with_name(".env")


class ConfigError(MemrankError):
    """Raised when a required configuration value is absent (no fallbacks)."""


def _ensure_dotenv() -> None:
    """Load this process's env file once. ``MEMRANK_ENV_FILE`` names it; default ``memrank/.env``.

    The override exists so a service can own its settings instead of sharing one file with
    everything else -- the accounts API has no business being able to read the Arena's token
    secret. ``dev.sh`` points the API at ``memrank/api/.env``; the CLI sets nothing and
    reads the package's own ``memrank/.env`` regardless of where it is invoked from. The
    default file is allowed to be absent (a demo run needs no config at all); each setting
    still fails by its own name when nothing answers it.

    It is an override, NOT a search path. A named file that answers some settings and lets
    another file answer the rest would mean a value the service never declared could still
    satisfy it -- the same silent degradation as a DSN that fails 30 seconds after startup
    instead of at it. One file answers; anything missing raises with its own name.

    Raises:
        ConfigError: If ``MEMRANK_ENV_FILE`` names a file that does not exist. Naming a file
            that is not there is a misconfiguration, not a reason to load nothing.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    named = os.environ.get("MEMRANK_ENV_FILE")
    if named:
        if not Path(named).is_file():
            raise ConfigError(f"MEMRANK_ENV_FILE points at {named!r}, which does not exist")
        load_dotenv(named)
    else:
        load_dotenv(_DEFAULT_ENV_FILE)
    _DOTENV_LOADED = True


def database_url() -> str:
    """Runtime publisher URL (least-priv, pooled). Used by leaderboard-publish/-gc. The ONLY
    DB URL the publishing app loads; the admin/migration connection is applied out-of-band by
    dbmate (see scripts/internal/db-reset.sh), never by this app."""
    _ensure_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ConfigError("DATABASE_URL is required (set it in memrank/.env or the environment)")
    return url


def ingest_database_url() -> str:
    """Ingest role URL (append-only on both tables). Used ONLY by leaderboard-ingest; never
    the publisher's DATABASE_URL."""
    _ensure_dotenv()
    url = os.environ.get("INGEST_DATABASE_URL")
    if not url:
        raise ConfigError("INGEST_DATABASE_URL is required for ingest")
    return url




def cli_token_from_env() -> str | None:
    """The CLI session token supplied by the environment, if any.

    Highest-priority source in ``CredentialStore``: CI, containers, and SSH sessions have no
    OS keyring, so an environment override must always be possible. Optional by design --
    absence means "look in the keyring next", not a misconfiguration.

    Lives here because this module is the one place permitted to read a credential from the
    environment; a test enumerates the rest of the package to keep that true.
    """
    _ensure_dotenv()
    return os.environ.get("MEMRANK_TOKEN") or None


def memrank_api_url() -> str:
    """Base URL of the memrank API the CLI authenticates against.

    Required rather than defaulted to localhost: a CLI that quietly falls back to a local
    address would, on a machine where nothing is listening, report a connection error
    instead of "you have not configured an API", and on a machine where something *is*
    listening would send a credential to it.

    Resolved through :mod:`memrank.settings` so ``memrank config set api.url`` actually takes
    effect; the environment still wins, which is what keeps ``MEMRANK_API_URL=... memrank ...``
    working. Imported inside the function because settings resolves through this module.
    """
    from memrank import settings

    url = settings.get("api.url")
    if not url:
        raise ConfigError("MEMRANK_API_URL is required (e.g. http://127.0.0.1:8100) -- "
                          "set it, or run `memrank config set api.url <url>`")
    return url.rstrip("/")







def artifact_bucket() -> str:
    """Return the S3 bucket name holding raw run artifacts.

    REQUIRED -- no fallback default. AWS region and credentials are deliberately NOT read
    here: boto3 resolves them from the standard AWS chain (env, AWS_PROFILE, instance role),
    the same chain every other AWS call uses. A missing region fails loudly in botocore.
    """
    _ensure_dotenv()
    bucket = os.environ.get("LEADERBOARD_ARTIFACT_BUCKET")
    if not bucket:
        raise ConfigError("missing storage config: LEADERBOARD_ARTIFACT_BUCKET")
    return bucket


def fake_storage_enabled() -> bool:
    """Test-only: route ingest/gc through the in-memory FakeStorage when MEMRANK_FAKE_STORAGE=1."""
    _ensure_dotenv()
    return os.environ.get("MEMRANK_FAKE_STORAGE") == "1"


def mlflow_enabled() -> bool:
    """Return True only when ``MEMRANK_MLFLOW_ENABLED`` is exactly ``"1"`` (default OFF).

    The "connected" switch for MLflow mirroring: when on, every ``memrank submit`` mirrors its
    just-recorded cells into the configured MLflow store (see
    :func:`memrank.tracking.export.mirror_run`).
    Default OFF leaves the eval loop untouched. When on but the ``mlflow`` extra is missing, the
    mirror raises loudly -- never a silent fallback.

    Returns:
        True only when ``MEMRANK_MLFLOW_ENABLED`` equals the exact string ``"1"``.
    """
    _ensure_dotenv()
    return os.environ.get("MEMRANK_MLFLOW_ENABLED") == "1"


def mlflow_tracking_uri() -> str:
    """MLflow tracking URI runs are mirrored into.

    Defaults to the local SQLite store ``sqlite:///mlflow.db`` (matches
    :data:`memrank.tracking.export.DEFAULT_TRACKING_URI`) -- a safe local default, not a fallback for
    missing required infra. Override with ``MEMRANK_MLFLOW_TRACKING_URI``.
    """
    _ensure_dotenv()
    return os.environ.get("MEMRANK_MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"


def mlflow_experiment() -> str:
    """MLflow experiment runs are mirrored under (default ``"memrank"``, override with
    ``MEMRANK_MLFLOW_EXPERIMENT``)."""
    _ensure_dotenv()
    return os.environ.get("MEMRANK_MLFLOW_EXPERIMENT") or "memrank"


def arena_beta_mode() -> bool:
    """Return True only when ``ARENA_BETA_MODE`` is exactly ``"1"`` (default OFF).

    Explicit, default-OFF opt-in for the Arena curation/clearance layer. When
    on, beta mode ASSUMES source/license signoff (BEAM license, sensitive-source,
    ``question_text_public``) and WAIVES the hard two-sided conflict-of-interest
    gate so AtomicStrata can self-clear AtomicMemory battles for pre-GA interface
    testing. Battles built/cleared under beta are stamped
    ``beta_source_signoff_assumed`` and labeled "beta -- illustrative, not vetted
    evidence"; the human PII/traceability/safety rubric still runs. All beta
    assumptions REVERT at GA (enforced gates + hard COI). Must be loud whenever
    active -- never a silent fallback.

    Returns:
        True only when ``ARENA_BETA_MODE`` equals the exact string ``"1"``;
        absent or any other value reads False.
    """
    _ensure_dotenv()
    return os.environ.get("ARENA_BETA_MODE") == "1"


#: Credentials belonging to the org a run was explicitly launched for, loaded once at the
#: start of that run. Empty unless ``--org`` was passed, so every existing workflow resolves
#: exactly as it did before this existed.
_ORG_SECRETS: dict[str, str] = {}


def set_org_secrets(values: dict[str, str]) -> None:
    """Use ``values`` as the credentials for this process's run.

    Ranked below the process environment and above the local wallet. Below the environment
    because an explicit ``export`` is a deliberate override an operator must always retain;
    above the wallet because a run launched *for an org* must not quietly spend whichever
    personal key happens to sit in that developer's wallet.
    """
    _ORG_SECRETS.clear()
    _ORG_SECRETS.update(values)


def clear_org_secrets() -> None:
    """Forget any org credentials loaded into this process."""
    _ORG_SECRETS.clear()


def _resolve_ref(value: str, *, noun: str = "secret") -> str:
    """Resolve a stored value: an ``${ENV_VAR}`` reference reads the env; else it's a literal.

    Reads the REAL environment, never ``.env`` -- see :func:`secret` for why.

    ``noun`` names the kind of stored value in the failure, so a checkout link that points at an
    unset variable does not report itself as a secret. One resolver rather than one per store: the
    reference syntax is the same wherever a value is stored, and a second copy would be the place
    the ``.env`` rule quietly stopped holding.

    Raises:
        ConfigError: When a ``${VAR}`` reference points at an unset environment variable.
    """
    if value.startswith("${") and value.endswith("}"):
        var = value[2:-1]
        resolved = os.environ.get(var)
        if resolved is None:
            raise ConfigError(f"{noun} references ${{{var}}} but it is unset")
        return resolved
    return value


def expand_ref(value: str, *, noun: str) -> str:
    """Resolve a stored non-secret value that may be an ``${ENV_VAR}`` reference.

    The public form of :func:`_resolve_ref`, for stores that borrow the wallet's value contract
    without holding credentials -- currently :mod:`memrank.targets.checkouts`.
    """
    return _resolve_ref(value, noun=noun)


def secret(name: str) -> str | None:
    """Resolve a secret by env-var name: the process environment wins, else the credentials pool.

    **Deliberately does not read ``.env``.** A file that populates the process behind your back
    makes a credential requirement look satisfied when it is genuinely unmet everywhere else, and
    that has cost real failures twice: a preflight that demanded a key it never used stayed wrong
    for weeks because ``.env`` answered it, surfacing only when an ECS task with no file and no
    terminal refused to start. The bug was invisible where it was cheap to find and visible only
    where it was expensive.

    The process environment is still read, and that is not a compromise: ECS resolves SSM
    parameters *into* the container's environment, CI injects ``${{ secrets.* }}`` the same way,
    and a shell ``export`` is a deliberate act. Only the implicit file is gone.

    Non-secret configuration (``DATABASE_URL``, Arena settings, feature flags) still reads ``.env``
    through its own accessors -- those are not secrets and local dev depends on them.

    Returns:
        The value, or ``None`` when neither the environment nor the wallet has it. Use
        ``memrank secrets set <NAME>`` (or ``secrets import-env``) to store one.
    """
    from memrank.secrets import wallet

    if name in os.environ:
        return os.environ[name]
    if name in _ORG_SECRETS:
        return _ORG_SECRETS[name]
    raw = wallet.get(name)
    return _resolve_ref(raw) if raw is not None else None


def missing_requirements(engine: str, *, embedder: str | None = None,
                         llm: str | None = None) -> list[str]:
    """The required secret env-vars for ``engine`` that resolve to nothing (env nor pool)."""
    from memrank.secrets import requirements

    return [name for name in requirements.required_secrets(engine, embedder=embedder, llm=llm)
            if secret(name) is None]


def preflight(engine: str, *, embedder: str | None = None, llm: str | None = None) -> None:
    """Fail loudly (before launch) if ``engine`` is missing any credential it needs to instantiate.

    Raises:
        ConfigError: Listing EVERY missing key at once, with a fix hint.
    """
    missing = missing_requirements(engine, embedder=embedder, llm=llm)
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(
            f"cannot launch {engine!r}: missing credential(s) {joined}. "
            f"Set them with `memrank secrets set <NAME>` (or export the env var).")


def resolved_secret_env(engine: str, *, embedder: str | None = None,
                        llm: str | None = None) -> dict[str, str]:
    """The ``{ENV_VAR: value}`` a launcher must inject for ``engine``. Every required secret.

    Preflights first (raises on any missing), then resolves each required secret from the pool.

    It used to skip names already in ``os.environ``, on the reasoning that a value already set wins
    and needs no injection. That is true of a process that inherits its own environment and false of
    the only thing this function serves: a **container**, which inherits nothing. Since ``.env``
    holds the same credentials, any caller that had touched a `_ensure_dotenv()`-backed setting first
    -- the retired ``engines_registry()`` did, and ``placement_for()`` called it before provisioning
    -- got an empty dict and launched an engine with no credentials at all. It surfaced as
    "connection refused" after a full readiness budget, naming nothing.
    """
    from memrank.secrets import requirements

    preflight(engine, embedder=embedder, llm=llm)
    out: dict[str, str] = {}
    for name in requirements.required_secrets(engine, embedder=embedder, llm=llm):
        resolved = secret(name)
        if resolved is not None:
            out[name] = resolved
    return out


def _stdin_is_interactive() -> bool:
    """Whether we can ask a human for a credential. Patched in tests."""
    return sys.stdin.isatty()


def _prompt_for_secret(name: str) -> str:
    """Ask for one credential with the input hidden. Patched in tests."""
    import typer

    return typer.prompt(f"{name} (paste key)", hide_input=True)


def ensure_secrets(names: Sequence[str]) -> None:
    """Make every credential in ``names`` resolvable, asking for any that are missing.

    This is the promise ``memrank secrets``/``preflight`` were built to keep: an experiment should
    never die part-way through because a key was absent. Anything already resolvable (environment,
    ``.env``, or the wallet) is left alone; anything missing is prompted for **once** and written to
    the wallet, so every later run with the same requirement just works.

    Interactive only. With no TTY -- CI, a cron job, a detached child -- prompting would hang forever
    with nobody to answer, so this raises instead, listing every missing key at once rather than
    one per attempt.

    Args:
        names: Credential env-var names; duplicates and already-present names are fine.

    Raises:
        ConfigError: When keys are missing and stdin is not interactive, or when a prompt is
            answered with blank input.
    """
    from memrank.secrets import wallet

    missing = [name for name in dict.fromkeys(names) if secret(name) is None]
    if not missing:
        return
    if not _stdin_is_interactive():
        joined = ", ".join(missing)
        raise ConfigError(
            f"missing credential(s) {joined}, and there is no terminal to ask on. "
            f"Set them with `memrank secrets set <NAME>` (or export the env var) and re-run.")
    for name in missing:
        value = _prompt_for_secret(name).strip()
        if not value:
            raise ConfigError(f"{name} was left blank; nothing stored.")
        wallet.put(name, value)
        # Local, and it is the only use of `style` in this module. `memrank.term.style` imports
        # `typer` -- the command-line framework -- and `memrank.config` is on the library run
        # path, so a module-level import here put the whole terminal stack into the process of
        # every caller who only ever ran an evaluation. This line is reached solely
        # when a human at a terminal has just been asked for a credential, which a library run
        # never is: it refuses above when there is no terminal to ask on.
        # `tests/repo/test_import_weight.py` holds a completed run to that budget.
        from memrank.term import style

        style.say(f"stored {name} -> {wallet.store_path()}")


_AWS_CONTEXT_FIELDS: tuple[str, ...] = (
    "region", "cluster", "subnet", "security_group", "log_group", "execution_role_arn",
    "task_role_arn", "artifact_bucket", "runner_repository", "engines_repository", "secret_arns")


#: The required keys as a message renders them. Built from :data:`_AWS_CONTEXT_FIELDS` rather
#: than written out again, so the list a failure quotes cannot drift from the list it checks --
#: the three messages below used to send the reader to a generator script instead, which is
#: operator tooling and absent from an installed copy (ATO-2125). The FILE is the contract, and
#: an operator running their own ECS can write it by hand from these names.
_AWS_CONTEXT_KEY_LIST = ", ".join(_AWS_CONTEXT_FIELDS)

# Where the context is looked for by default: the working directory, so the common case needs no
# environment variable at all. Requiring one to point at a file we told you to create at a
# specific path is ceremony, not safety.
DEFAULT_AWS_CONTEXT = ".aws-context.json"


def aws_context() -> dict:
    """The AWS coordinates a cloud eval needs.

    Read from ``MEMRANK_AWS_CONTEXT`` when set, else ``.aws-context.json`` in the working directory
    -- the path ``scripts/internal/aws-context.sh`` writes.

    The generator is operator tooling for this deployment's own terraform and does not ship in the
    public tree; the FILE is the contract, and any operator running their own ECS can write it by
    hand. What memrank requires is the fields below, not the script that happened to produce them.

    memrank deliberately does NOT shell out to terraform: that would make the terraform binary,
    state credentials and the ``infra/`` directory hard runtime requirements of every eval.
    ``scripts/internal/aws-context.sh`` resolves the values once and writes this file; both that script's
    other consumer (``scripts/cloud-run.sh``) and ``memrank submit --on cloud`` read the same artifact,
    so the list of terraform output names lives in exactly one place.

    Returns:
        The parsed context. Every key in :data:`_AWS_CONTEXT_FIELDS` is guaranteed present.

    Raises:
        ConfigError: When no context file is found, is unparseable, or is missing a required field --
            naming the fields and how to regenerate it. A partially-written context would otherwise
            surface as a ``KeyError`` deep inside a paid launch.
    """
    import json

    _ensure_dotenv()
    configured = os.environ.get("MEMRANK_AWS_CONTEXT")
    path = configured or DEFAULT_AWS_CONTEXT
    if not configured and not Path(path).exists():
        raise ConfigError(
            f"no AWS context for --on cloud: {DEFAULT_AWS_CONTEXT!r} is not in the working "
            f"directory and MEMRANK_AWS_CONTEXT is unset. Write it as a JSON object with the "
            f"keys {_AWS_CONTEXT_KEY_LIST} -- the coordinates of the ECS cluster the run launches "
            f"into -- or point MEMRANK_AWS_CONTEXT at a copy you already have.")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(
            f"MEMRANK_AWS_CONTEXT points at {path!r}, which does not exist. Point it at an "
            f"existing file, or write one there as a JSON object with the keys "
            f"{_AWS_CONTEXT_KEY_LIST}.") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path!r} is not valid JSON: {exc}") from exc

    missing = [f for f in _AWS_CONTEXT_FIELDS if not data.get(f)]
    if missing:
        raise ConfigError(
            f"{path!r} is missing {', '.join(missing)} (generated_at "
            f"{data.get('generated_at', 'unknown')}). It is probably from an older schema or a "
            f"partial write -- every one of {_AWS_CONTEXT_KEY_LIST} must be present and "
            f"non-empty.")
    return data

