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
"""Operator-declared effective LLM/embedder config for HTTP-backed adapters.

The extraction LLM and embedder of an HTTP memory engine live server-side (set
out-of-band in the engine's own ``.env``), so the adapter cannot introspect them.
The operator declares them via ``{PREFIX}LLM_MODEL`` / ``{PREFIX}LLM_PROVIDER`` /
``{PREFIX}EMBEDDER_MODEL`` / ``{PREFIX}EMBEDDING_DIMS`` env vars, and these helpers
read them into the shape ``Memory.effective_config`` expects. These names
match the ones the mem0 server itself reads (``MEM0_LLM_MODEL``,
``MEM0_EMBEDDER_MODEL``, ``MEM0_EMBEDDING_DIMS``), so a single env declaration both
configures the backend and is recorded truthfully. Missing vars are reported as
``None`` (never raise) so the record is always well-formed.
"""

from __future__ import annotations

import os
from typing import Any


def llm_from_env(prefix: str) -> dict[str, Any]:
    """Read the operator-declared extraction LLM for ``prefix`` (e.g. ``"MEM0_"``)."""
    return {"provider": os.environ.get(f"{prefix}LLM_PROVIDER"),
            "model": os.environ.get(f"{prefix}LLM_MODEL")}


def embedder_from_env(prefix: str) -> dict[str, Any]:
    """Read the operator-declared embedder provider + model + dims for ``prefix``.

    Uses ``{prefix}EMBEDDER_PROVIDER`` / ``{prefix}EMBEDDER_MODEL`` / ``{prefix}EMBEDDING_DIMS`` --
    the same names the mem0 server reads -- so declaring the embedder once both configures the
    backend and records it in the receipt.

    ``provider`` was previously omitted even though the eval-config profiles have always set it,
    so a run record could not say whether an embedder was hosted or self-hosted. It is included
    here so this env-derived record and a manifest-derived one describe the same fields.
    """
    dims = os.environ.get(f"{prefix}EMBEDDING_DIMS")
    return {"provider": os.environ.get(f"{prefix}EMBEDDER_PROVIDER"),
            "model": os.environ.get(f"{prefix}EMBEDDER_MODEL"),
            "dims": int(dims) if dims and dims.isdigit() else None}
