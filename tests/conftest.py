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
"""Root pytest configuration -- registers shared fixture plugins."""

import os
import pathlib
import tempfile

import pytest

# Registered conditionally, not hard-wired. The Postgres fixtures serve the internal accounts and
# arena suites only, and they live under the `tests/internal/` prefix that `publish.toml` drops
# wholesale. Naming the module unconditionally makes its absence break collection of the ENTIRE
# suite -- every public test included -- rather than the handful that need a database.
pytest_plugins = [
    f"tests.internal.{_name}"
    for _name in ("conftest_pg",)
    if (pathlib.Path(__file__).parent / "internal" / f"{_name}.py").exists()
]

# Isolated at IMPORT time, not in a fixture, because the catalog is read during COLLECTION:
# `tests/targets/test_engine_env.py` parametrizes over `list_targets()` inside a decorator, which
# runs before any session fixture. Left to the fixture below, a developer who has followed the
# documented setup -- `memrank config set targets.path ~/evals` -- would see this suite fail on
# targets memrank has never heard of, at collection, with no env var exported.
#
# The directory is deliberately not cleaned up: it must outlive collection AND the whole session,
# and a few empty temp dirs are cheaper than a suite whose result depends on the developer's
# machine. `_isolate_user_config` still runs and still owns the per-session directory.
os.environ["MEMRANK_CONFIG_DIR"] = tempfile.mkdtemp(prefix="memrank-collect-")
os.environ.pop("MEMRANK_TARGETS_PATH", None)

# --- The rendering environment, pinned for the whole session -------------------------------
#
# Typer renders help and usage errors through rich, and decides ONCE, at
# ``typer.rich_utils`` import time, whether it is writing to a terminal:
#
#     FORCE_TERMINAL = True if getenv("GITHUB_ACTIONS") or getenv("FORCE_COLOR")
#                              or getenv("PY_COLORS") else None
#
# On a GitHub Actions runner that is True, so every CliRunner result carries ANSI escapes --
# and thirteen tests that assert plain substrings on CLI output failed in CI while passing on
# every developer's machine (ATO-1931). Rendering is presentation; it must not decide whether a
# test passes.
#
# Pinned here rather than per test for two reasons. It has to be an ENVIRONMENT variable set
# before ``typer.rich_utils`` is first imported, because the constant above is module level --
# a fixture runs far too late. And a chokepoint in the root conftest covers every CLI test
# there is and every one anyone adds later, which thirteen patched assertions would not.
# ``tests/term/test_rendering_is_pinned.py`` is the guard that this is actually in effect.
#
# ``_TYPER_FORCE_DISABLE_TERMINAL`` is typer's own knob for exactly this and wins over all
# three detections above. The two colour variables are dropped as well so that a developer who
# exports them does not get a different answer from CI: rich consoles built elsewhere in
# ``memrank.term`` consult them directly, where typer's knob does not reach. Nothing here
# leaves the test session, so a real user's terminal renders exactly as before.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("PY_COLORS", None)


@pytest.fixture(autouse=True, scope="session")
def _isolate_run_registry():
    """Point the run registry at a tmp dir so `memrank run` in tests never writes
    the accumulating ``runs/`` folder into the repo."""
    previous = os.environ.get("MEMRANK_RUNS_DIR")
    with tempfile.TemporaryDirectory(prefix="memrank-runs-") as tmp:
        os.environ["MEMRANK_RUNS_DIR"] = tmp
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("MEMRANK_RUNS_DIR", None)
            else:
                os.environ["MEMRANK_RUNS_DIR"] = previous


@pytest.fixture(autouse=True, scope="session")
def _isolate_user_config():
    """Point per-user state at a tmp dir so no test reads the developer's own settings.

    Standing defaults (``defaults.on``, ``defaults.org``) decide where a submission goes. Without
    this, whether the suite exercised the local or the cloud path would depend on whether the
    person running it happens to have logged in -- the test equivalent of a run whose result
    depends on the machine. The wallet in ``secrets.wallet`` shares this directory and is isolated
    by the same fixture.
    """
    previous = os.environ.get("MEMRANK_CONFIG_DIR")
    with tempfile.TemporaryDirectory(prefix="memrank-config-") as tmp:
        os.environ["MEMRANK_CONFIG_DIR"] = tmp
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("MEMRANK_CONFIG_DIR", None)
            else:
                os.environ["MEMRANK_CONFIG_DIR"] = previous


@pytest.fixture(autouse=True, scope="session")
def _signed_out_by_default():
    """No test inherits the developer's session.

    ``CredentialStore`` reads ``MEMRANK_TOKEN``, then the OS keyring, then a file -- so a
    developer who has run `memrank auth login` had every test believing it was signed in, and
    the CLI's logged-out behaviour was exercised on nobody's machine but CI's. Signed out is
    the default; a test that wants a session says so.

    Isolated at the store's three INPUTS rather than by patching its methods, so the tests that
    exercise CredentialStore itself keep testing the real thing: they inject their own keyring
    and path, which this leaves untouched.
    """
    import sys
    import types

    from memrank.accounts import credentials

    empty_keyring = types.ModuleType("keyring")
    empty_keyring.get_password = lambda service, username: None
    empty_keyring.set_password = lambda service, username, value: None
    empty_keyring.delete_password = lambda service, username: None

    previous_env = os.environ.pop("MEMRANK_TOKEN", None)
    previous_module = sys.modules.get("keyring")
    original_path = credentials.default_path
    sys.modules["keyring"] = empty_keyring
    with tempfile.TemporaryDirectory(prefix="memrank-creds-") as tmp:
        credentials.default_path = lambda: pathlib.Path(tmp) / "credentials"
        try:
            yield
        finally:
            credentials.default_path = original_path
            if previous_module is not None:
                sys.modules["keyring"] = previous_module
            else:
                sys.modules.pop("keyring", None)
            if previous_env is not None:
                os.environ["MEMRANK_TOKEN"] = previous_env


@pytest.fixture(autouse=True, scope="session")
def _never_load_the_repo_dotenv():
    """Stop the repository's real ``.env`` reaching any test.

    ``config._ensure_dotenv()`` calls ``load_dotenv(".env")`` relative to CWD and sets a module
    global that is never reset. Tests run with CWD at the repo root, so the first non-secret config
    accessor pulled the developer's real ANTHROPIC_API_KEY into ``os.environ`` **for the remainder
    of the session** -- after which every later assertion about a missing credential was answered by
    a key the test never provided.

    ``tests/secrets/`` worked around this per-fixture with ``monkeypatch.chdir(tmp_path)``. This
    closes it once, for everything: a test that needs a credential must set it or store it.
    """
    from memrank import config

    config._DOTENV_LOADED = True      # "already loaded", so load_dotenv is never called
    yield
