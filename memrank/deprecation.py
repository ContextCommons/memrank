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
"""The one deprecation notice the pre-agent surface gives, worded once.

Memrank's current interface is `memrank run` with an agent. The older surface -- the in-process
Python route (`evaluation.run(system)`), memory adapters, targets, placement, the cloud sweep
and the hidden engine-hosting commands -- still ships, and warns only where a person enters it.
Nothing here warns on import: `memrank/__init__` imports that surface, so an import-time
warning would reach every `memrank run`.
"""

from __future__ import annotations

import warnings

#: Where a person on the old surface goes instead.
REPLACEMENT = "evaluate an agent with `memrank run` instead (see the README)"


def message(what: str) -> str:
    """The notice for ``what``, the same sentence in Python and on the command line."""
    return f"{what} is deprecated and will be removed; {REPLACEMENT}."


def warn(what: str, *, stacklevel: int = 3) -> None:
    """Raise a DeprecationWarning naming ``what``, attributed to the caller's caller."""
    warnings.warn(message(what), DeprecationWarning, stacklevel=stacklevel)
