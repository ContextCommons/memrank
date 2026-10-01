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
"""Whether a field NAME looks like it holds a credential.

One definition, because two callers decide different consequences from the same question and they
must not disagree about the answer:

  - ``memrank/provenance/receipt.py`` redacts a matching field before a receipt is written or synced.
  - ``memrank/targets/manifest.py`` REFUSES a matching field in an ``sdk_config:`` block, because
    manifests are committed to git and a value there is already leaked.

A second copy would drift, and the copy that drifted would be the one deciding what counts as a
leak.

This asks only about the name. Detecting a credential by its *value* is a different problem with
different failure modes, and is not attempted here.
"""

from __future__ import annotations

import re

# Secret detection on a CANONICAL key (lowercased, punctuation stripped) so
# header-/camelCase/snake_case spellings collapse to the same form:
#   "x-api-key" / "apiKey" / "API_KEY" -> "apikey".
# Distinctive forms match as substrings; short ambiguous ones match exactly so
# "token_budget" -> "tokenbudget" and "monkey" are NOT flagged.
_SECRET_SUBSTRINGS = ("apikey", "accesstoken", "authtoken", "secret", "password",
                      "passwd", "authorization", "bearer", "privatekey", "credential")
_SECRET_EXACT = {"key", "token", "auth", "password", "secret"}


def canon_key(name: str) -> str:
    """``name`` lowercased with punctuation stripped, so spellings collapse to one form."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_secret_key(name: str) -> bool:
    """Whether a field called ``name`` should be treated as holding a credential."""
    canon = canon_key(name)
    return canon in _SECRET_EXACT or any(s in canon for s in _SECRET_SUBSTRINGS)
