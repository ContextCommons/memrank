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
"""Presets: HTTP mappings memrank ships, so common agents need no mapping written.

A preset is nothing but an :class:`~memrank.connect.http.HttpSpec` with the agent-specific
parts left open -- the base URL, the credential, and the variables it names. There is no
second code path behind one.

``openai-chat`` speaks the OpenAI Chat Completions API. Each past session is replayed as one
chat request under ``user`` = the attempt's session id, and the question is asked as a new
request under the same ``user``. ``metadata.memrank_op`` says which of the two a request is;
an agent may use it to skip generating a reply to history, and one that ignores it simply
replies and the reply is discarded.
"""

from __future__ import annotations

from typing import Any

from memrank.connect.base import ConnectorConfigError

_CHAT_PATH = "/v1/chat/completions"

PRESETS: dict[str, dict[str, Any]] = {
    "openai-chat": {
        "required_vars": ("model",),
        "mapping": {
            "feed": {"per": "session", "method": "POST", "path": _CHAT_PATH,
                     "body": {"model": "{model}", "user": "{session_id}",
                              "messages": "{messages}", "metadata": {"memrank_op": "feed"}}},
            "ask": {"method": "POST", "path": _CHAT_PATH,
                    "body": {"model": "{model}", "user": "{session_id}",
                             "messages": "{messages}", "metadata": {"memrank_op": "ask"}},
                    "answer": "choices[0].message.content"},
        },
    },
}


def preset_mapping(name: str, variables: dict[str, Any]) -> dict[str, Any]:
    """The HTTP mapping preset ``name`` stands for, after checking the variables it needs."""
    if name not in PRESETS:
        raise ConnectorConfigError(f"no preset {name!r}; presets: {', '.join(sorted(PRESETS))}")
    preset = PRESETS[name]
    missing = [v for v in preset["required_vars"] if v not in variables]
    if missing:
        raise ConnectorConfigError(f"preset {name!r} needs {', '.join(missing)} in the agent spec")
    return dict(preset["mapping"])
