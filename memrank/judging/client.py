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
"""The provider-backed completer, with response-only caching and a billable-call counter.

Each role -- the reader and the judge -- resolves to its provider through
:mod:`memrank.judging.providers`, or to a Level 2 completer of your own.

The counter sits at the single egress point so a judged run reports exactly what it spent. It used
to be a hard CAP as well; that ceiling is gone, because bounding the scorer would have made which
queries got graded depend on where a counter ran out. What a run costs is bounded by choosing how
many units to run, which is where every evaluation framework puts it.

The cache stores only the response keyed by a content hash -- the prompt body (which may contain
memory content) is never persisted.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memrank.rate_limit import RateLimitGate

from memrank.atomic_json import staging_path
from memrank.judging.judge import JUDGE, READER, Completer, JudgeConfig
from memrank.judging.prompts import JUDGE_PROMPT_VERSION
from memrank.judging.providers import (
    ADAPTERS,
    BUILTINS,
    ProviderError,
    ResolvedJudge,
    Send,
    resolve,
)

_MAX_TOKENS = 512

# The credential the deprecated submit and cloud path judges with: it stays Anthropic-only.
JUDGE_SECRET = BUILTINS["anthropic"].key_name


def _cache_dir() -> Path:
    base = os.environ.get("MEMRANK_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".memrank"
    out = root / "judge-cache"
    out.mkdir(parents=True, exist_ok=True)
    # Cached responses are derived from private memory (generated answers + judge
    # rationales can echo retrieved content), so keep the cache owner-only.
    try:
        out.chmod(0o700)
    except OSError:
        pass
    return out


def cache_dir() -> Path:
    """The owner-only directory judge responses are cached in (``$MEMRANK_CACHE_DIR``)."""
    return _cache_dir()


def cached_completer(inner: Completer, *, cache_dir: Path, prompt_version: str,
                     key_prefix: str = "") -> Completer:
    """``inner`` with its responses cached under a key of ``key_prefix`` and the prompt.

    ``key_prefix`` names who answers (:func:`cache_prefix`, or ``custom:<name>|``); it is empty for
    the built-in Anthropic provider, whose keys are byte-identical to the ones written before
    providers existed.
    """
    def complete(model: str, system: str, user: str) -> str:
        key = hashlib.sha256(
            f"{key_prefix}{prompt_version}|{model}|{_MAX_TOKENS}|{system}|{user}".encode()
        ).hexdigest()
        path = cache_dir / f"{key}.txt"
        if path.exists():
            try:
                path.chmod(0o600)  # remediate pre-existing loose perms on read
            except OSError:
                pass
            return path.read_text(encoding="utf-8")
        resp = inner(model, system, user)
        # Staged then renamed, never written in place. `path.exists()` above is the whole
        # cache-hit test, so a reader that catches a partial write reads a truncated response AS A
        # VERDICT -- silently, and only under concurrency. `staging_path` is per-process and
        # per-thread, which is what makes two workers racing the same key safe rather than fatal
        # (the constant-temp-name version of this bug discarded a 4h50m run on 2026-08-12).
        tmp = staging_path(path)
        tmp.write_text(resp, encoding="utf-8")  # response only -- never the prompt
        try:
            tmp.chmod(0o600)  # owner-only BEFORE it is visible: responses can echo private memory
        except OSError:
            pass
        tmp.replace(path)
        return resp

    return complete


def counting_completer(inner: Completer) -> tuple[Completer, Callable[[], int]]:
    """Count the calls that go out, and refuse none of them.

    This used to be ``capped_completer``, raising once a ceiling was passed. The ceiling is gone: it
    bounded the MEASUREMENT rather than the system under test, and which queries got graded would
    have depended on where a counter ran out -- on ordering, not on quality. No evaluation framework
    caps its scorer; they bound examples instead, which memrank spells `--unit` and its eval slices.
    Nor was there a runaway to catch: gradings per query are fixed by the shape, queries by the
    slice, and a verdict is re-asked at most ``judge.MAX_VERDICT_ATTEMPTS`` times, so spend is
    bounded by construction. Measured, a judge that never returns parseable JSON spends LESS than
    the plan, because a grading that exhausts its attempts abandons the rest of its query.

    The lock stays, because the count outlived the cap and is now the receipt's `judge_calls_made`.
    Read-modify-write across judge workers drops increments, and a spend figure that under-reports
    is worse than none. It is held only around the counter, never across ``inner`` -- holding it over
    the API call would serialise judging back into what that concurrency exists to escape.
    """
    state = {"n": 0}
    lock = threading.Lock()

    def complete(model: str, system: str, user: str) -> str:
        with lock:
            state["n"] += 1
        return inner(model, system, user)

    return complete, (lambda: state["n"])


#: How long a judge call may spend waiting out rate limits before the run gives up. Matches the
#: runner's ingest/retrieve deadline: the gate makes sustained saturation self-resolve, so this is
#: the backstop for what no amount of waiting fixes -- a revoked key, zero quota -- not a contention
#: budget. Bounded because "no degraded modes" means stopping, not spinning while looking healthy.
_JUDGE_RETRY_DEADLINE_S = 600.0
_JUDGE_BACKOFF_BASE_S = 1.0


def gated_completer(inner: Completer, *,
                    gate: Callable[[], RateLimitGate] | None = None) -> Completer:
    """Wait out a rate limit on the shared gate instead of failing the run.

    ``gate`` is a zero-arg resolver for the CURRENT gate (e.g. ``GateScope.get``), read per
    call because a sweep's judge runtime outlives any one cell's gate. ``None`` gives this
    completer a private gate of its own -- real coordination between ITS workers, shared
    with nobody else.

    Judging was the one caller that never did this. Sequentially it was survivable -- one call at a
    time rarely saturates an account. Concurrently it is not: `_judge_one_query` catches only
    `UnparseableVerdict`, so a 429 propagates and kills the run in its LAST stage, after every
    ingest and retrieve has been paid for.

    The same gate ingest and retrieve use, deliberately. The limit is account-wide, so a judge
    worker that ignored the gate would keep the window saturated while the others waited on it --
    throttling the workers and not the account.
    """
    from memrank import rate_limit

    if gate is None:
        own = rate_limit.RateLimitGate()
        resolve_gate = lambda: own  # noqa: E731 - a closure, not a def, is the point here
    else:
        resolve_gate = gate

    def complete(model: str, system: str, user: str) -> str:
        gate = resolve_gate()
        deadline = time.monotonic() + _JUDGE_RETRY_DEADLINE_S
        attempt = 0
        while True:
            attempt += 1
            gate.wait()
            try:
                return inner(model, system, user)
            except Exception as exc:  # noqa: BLE001 - re-raised unless it is a rate limit
                if not rate_limit.is_rate_limited(exc) or time.monotonic() >= deadline:
                    raise
                stated = rate_limit.retry_after_seconds(exc)
                delay = stated if stated is not None else min(
                    _JUDGE_BACKOFF_BASE_S * (2 ** (attempt - 1)), rate_limit.MAX_PAUSE_S)
                gate.pause(delay)

    return complete


def sampling_params(role: str, cfg: JudgeConfig) -> dict[str, float]:
    """The sampling kwargs one call gets, by ROLE: pinned for the reader, none for the judge.

    Reader calls carry ``temperature=cfg.answer_temperature`` (0 by default -- the reproducibility
    pin BEAM's own protocol specifies). Judge calls carry nothing: claude-opus-4-8 rejects the
    parameter, so the judge runs provider-default sampling and ``_judge_receipt_config`` records
    that split rather than implying both were pinned. The role decides, not the model: it used to
    be ``model == answer_model``, which gave a judge that shared the reader's model a temperature.
    A provider that rejects ``temperature`` for the reader fails with its own 400; the parameter is
    never dropped.
    """
    cfg.spec(role)  # refuses an unknown role
    return {"temperature": cfg.answer_temperature} if role == READER else {}


def cache_prefix(target: ResolvedJudge) -> str:
    """What a cache key carries before today's ``prompt_version|model|...`` for ``target``.

    The built-in ``anthropic`` carries nothing, so every existing cache entry stays valid and old
    runs do not pay again. Any other provider carries its adapter and endpoint -- and a named
    endpoint its fingerprint too -- so moving or replacing an endpoint invalidates its entries.
    The cache is local files named by a sha256, so the URL never leaves the machine.
    """
    if target.endpoint is None:
        return "" if target.adapter == "anthropic" else f"{target.adapter}|{target.api_base}|"
    return f"{target.adapter}|{target.api_base}|{target.fingerprint}|"


def _api_key(name: str, keys: Mapping[str, str]) -> str:
    """``name``'s value: given by the caller, else memrank's credential chokepoint."""
    from memrank import config
    from memrank.secrets import wallet

    # Resolve through memrank's one credential chokepoint (environment -> wallet) rather than
    # reading os.environ, and pass the key explicitly rather than letting an SDK read the
    # environment itself -- otherwise a key held only in the wallet would never reach the client.
    api_key = keys.get(name) or config.secret(name)
    if not api_key:
        raise RuntimeError(
            f"{name} not found. Checked the environment and the wallet at "
            f"{wallet.store_path()}. Set it with `memrank secrets set {name}`, "
            f"`memrank secrets import-env`, or export it. A `.env` file is deliberately NOT "
            f"consulted for credentials.")
    return api_key


@dataclass(frozen=True)
class _Route:
    """Where one role's calls go: the cache prefix, the model sent, and the sender."""

    prefix: str
    model: str
    send: Completer


def _provider_send(cfg: JudgeConfig, role: str, target: ResolvedJudge,
                   keys: Mapping[str, str]) -> Completer:
    """A sender for ``target`` whose key is looked up on its first call, not before.

    Deferred so a configuration whose reader is never called -- judging a recorded run -- does not
    need the reader's key.
    """
    sampling = sampling_params(role, cfg)
    built: list[Send] = []
    lock = threading.Lock()

    def complete(model: str, system: str, user: str) -> str:
        with lock:
            if not built:
                built.append(ADAPTERS[target.adapter](
                    target, _api_key(target.key_name, keys), max_tokens=_MAX_TOKENS))
        return built[0](model, system, user, sampling)

    return complete


def _route(cfg: JudgeConfig, role: str, keys: Mapping[str, str]) -> _Route:
    if cfg.completer is not None:
        return _Route(f"custom:{cfg.completer_name}|", cfg.model(role), cfg.completer)
    target = resolve(cfg.spec(role), cfg.endpoints)
    return _Route(cache_prefix(target), target.model, _provider_send(cfg, role, target, keys))


#: Anthropic's answers that mean "not this key": invalid (401), not allowed (403), and 400,
#: which is how an account with no credit refuses a request.
_KEY_REFUSALS = (400, 401, 403)


def key_refusal(target: ResolvedJudge, api_key: str) -> str | None:
    """``None`` when ``target``'s provider accepts ``api_key`` for judging, else its reason.

    One tiny request to the judge's model through the judge's own adapter, so a key is checked
    exactly as judging will use it -- the only check that finds an Anthropic account with no
    credit, which is why a key pasted at the terminal gets it. Any other failure -- unreachable,
    overloaded -- is not an answer about the key and is raised.
    """
    # The judge role's sampling (none) is the same whatever the configuration names.
    complete = _provider_send(JudgeConfig(), JUDGE, target, {target.key_name: api_key})
    try:
        complete(target.model, "Reply with the word OK.", "OK?")
    except ProviderError as exc:
        if exc.status_code not in _KEY_REFUSALS:
            raise
        return exc.message
    return None


def build_completer(cfg: JudgeConfig, *,
                    gate: Callable[[], RateLimitGate] | None = None,
                    keys: Mapping[str, str] | None = None,
                    ) -> tuple[Completer, Callable[[], int]]:
    """Compose cache(count(gate(provider))) per role; return (completer, billable-calls getter).

    The returned completer takes a ROLE (:data:`~memrank.judging.judge.READER` or
    :data:`~memrank.judging.judge.JUDGE`) where a model would go, and sends it to that role's
    provider and model. ``keys`` maps key names to values the caller already holds (``memrank
    run`` judges with the organisation's saved key, decision 0037); any other key resolves on
    this machine.

    The counter wraps only the egress, so it reflects actual API calls -- cache hits bypass it, which
    is what makes `judge_calls_made` a spend figure rather than a workload figure.

    The gate sits INSIDE the counter, and the order carries two decisions. A cache hit must not wait
    on a quota it is never going to spend, so gating below the cache is required. And a 429 is a
    rejected call that cost nothing, so its retries must not each be counted as spend -- the counter
    counts logical calls, the gate absorbs the retries beneath it.

    A ``cfg.completer`` (Level 2: your own code) is gated too. Excluding it would mean the
    composition under test is not the composition that runs.
    """
    if cfg.completer is None and resolve(cfg.judge, cfg.endpoints).decides:
        raise ValueError(f"judge {cfg.judge!r} decides typed questions and writes no text; it "
                         f"judges a recorded run (memrank run, memrank.loop.judge), not through "
                         f"a completer")
    held = dict(keys or {})
    routes = {role: _route(cfg, role, held) for role in (READER, JUDGE)}

    def route_of(role: str) -> _Route:
        if role not in routes:
            raise ValueError(f"a judging completer is called with a role ({READER!r} or "
                             f"{JUDGE!r}), got {role!r}")
        return routes[role]

    def egress(role: str, system: str, user: str) -> str:
        route = route_of(role)
        return route.send(route.model, system, user)

    counted, counter = counting_completer(gated_completer(egress, gate=gate))

    def uncached(role: str, system: str, user: str) -> str:
        route_of(role)  # an unknown role is refused before it is counted
        return counted(role, system, user)

    if not cfg.cache:
        return uncached, counter
    cache_dir = _cache_dir()
    cached = {role: cached_completer(_bound(counted, role), cache_dir=cache_dir,
                                     prompt_version=JUDGE_PROMPT_VERSION, key_prefix=route.prefix)
              for role, route in routes.items()}

    def complete(role: str, system: str, user: str) -> str:
        route = route_of(role)
        return cached[role](route.model, system, user)

    return complete, counter


def _bound(counted: Completer, role: str) -> Completer:
    """``counted`` called for ``role`` whatever model the cache layer passes it."""
    return lambda _model, system, user: counted(role, system, user)
