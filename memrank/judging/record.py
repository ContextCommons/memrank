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
"""Who a configuration's reader and judge are, as a record states them.

A record names the spec as typed, the provider and model it resolved to, a fingerprint of the
endpoint (never its URL), and the settings actually sent -- so a score is never read without
knowing whose it is. A Level 2 completer is ``custom:<completer_name>``: naming the default Claude
model for a run that never called it is the claim this replaces.
"""

from __future__ import annotations

from typing import Any

from memrank.judging.client import _MAX_TOKENS, sampling_params
from memrank.judging.judge import JUDGE, READER, JudgeConfig
from memrank.judging.providers import resolve

#: The provider recorded for a Level 2 completer.
CUSTOM = "custom"


def role_record(cfg: JudgeConfig, role: str) -> dict[str, Any]:
    """``spec``, ``provider``, ``model``, ``endpoint`` and ``settings`` of the model in ``role``."""
    if cfg.completer is not None:
        # The completer is what answered, so its name stands where a model would: a record's
        # judge is always <provider>:<model>.
        return {"spec": f"{CUSTOM}:{cfg.completer_name}", "provider": CUSTOM,
                "model": cfg.completer_name, "endpoint": None, "settings": {"adapter": CUSTOM}}
    target = resolve(cfg.spec(role), cfg.endpoints)
    settings = {"adapter": target.adapter, "max_tokens": _MAX_TOKENS,
                "max_tokens_field": target.max_tokens_field, **sampling_params(role, cfg)}
    return {"spec": cfg.spec(role), "provider": target.adapter, "model": target.model,
            "endpoint": target.fingerprint, "settings": settings}


def receipt_fields(cfg: JudgeConfig) -> dict[str, Any]:
    """The judge and reader identity a receipt's ``config`` records, flat, by role prefix."""
    fields: dict[str, Any] = {}
    for prefix, role in (("judge", JUDGE), ("reader", READER)):
        record = role_record(cfg, role)
        fields[prefix] = record["spec"]
        fields.update({f"{prefix}_{name}": record[name]
                       for name in ("provider", "model", "endpoint", "settings")})
    return fields
