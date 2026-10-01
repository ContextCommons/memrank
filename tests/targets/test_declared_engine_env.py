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
"""A target declares the environment its engine runs under, and the receipt records it.

The case this exists for: an engine whose extractor silently falls back to a cheaper deterministic
one when its model call fails is only a measurement of the model when that fallback is DISABLED,
which makes a failed extraction fatal. As an exported variable that was forgettable, and forgetting
was invisible in the result -- one recorded run substituted the fallback for 36 of 36 messages and
returned the fallback arm's score.

It is deliberately NOT `ENGINE_SETTINGS`: that table is adapter-wide, so it would impose the policy
on every variant of that engine, including ones where a transient provider error becoming fatal is
the wrong trade.

The engine here is `hindsight` because the field is generic; the target that drove the requirement
now lives outside this repo, and re-framing kept the coverage rather than deleting it with it.
"""

from __future__ import annotations

import pytest

from memrank.targets.engine_env import EngineEnvError, declared_env
from memrank.targets.manifest import ManifestError, from_dict, to_dict

STACK = {"name": "t", "kind": "stack", "adapter": "hindsight",
         "components": {"llm": {"provider": "anthropic", "model": "claude-haiku-4-5"}}}


def target(**overrides):
    return from_dict({**STACK, **overrides})


def test_a_declared_variable_reaches_the_engine():
    got = target(engine_env={"EXTRACTOR_FALLBACK": "0"})

    assert declared_env(got) == {"EXTRACTOR_FALLBACK": "0"}


def test_nothing_declared_is_an_empty_environment():
    assert declared_env(target()) == {}


def test_the_declaration_is_in_the_receipt():
    """Unlike `secrets`, values are recorded: this is the policy the run was measured under."""
    serialized = to_dict(target(engine_env={"EXTRACTOR_FALLBACK": "0"}))

    assert serialized["engine_env"] == {"EXTRACTOR_FALLBACK": "0"}


def test_a_variable_the_components_own_is_refused():
    """Otherwise the manifest asserts one model and the engine runs another.

    `HINDSIGHT_API_LLM_MODEL` is what ENGINE_ENV renders `llm.model` to for this adapter, so
    setting it here would put a different model on the wire than the receipt claims.
    """
    got = target(engine_env={"HINDSIGHT_API_LLM_MODEL": "something-else"})

    with pytest.raises(EngineEnvError, match="derives that variable from its components"):
        declared_env(got)


def test_a_variable_a_declared_secret_owns_is_refused():
    with pytest.raises(ManifestError, match="already set from this target"):
        target(secrets={"HINDSIGHT_TOKEN": ["HINDSIGHT_API_LLM_API_KEY"]},
               engine_env={"HINDSIGHT_API_LLM_API_KEY": "not-a-real-key"})


def test_a_credential_shaped_name_is_refused():
    """engine_env is serialized; `secrets:` records a name and resolves the value at launch."""
    with pytest.raises(ManifestError, match="looks like a credential"):
        target(engine_env={"OPENAI_API_KEY": "sk-whatever"})


@pytest.mark.parametrize("raw, message", [
    (["EXTRACTOR_FALLBACK=0"], "must be a mapping"),
    ({"": "0"}, "non-empty variable names"),
    ({"EXTRACTOR_FALLBACK": 0}, "must be a string"),
    ({"CASCADE_MODE": True}, "must be a string"),
])
def test_malformed_declarations_are_rejected(raw, message):
    # Numbers and booleans are refused rather than coerced: an environment variable is always text,
    # and `CASCADE_MODE: true` rendering as "True" is the kind of quiet mismatch this field must not
    # introduce.
    with pytest.raises(ManifestError, match=message):
        target(engine_env=raw)
