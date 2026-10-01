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
"""Where to send a reader who needs the documentation.

``docs/`` is a directory in the repository and is NOT in the wheel: ``pyproject.toml`` excludes it
from the packages that ship, so ``pip install memrank`` leaves no ``docs/`` anywhere on the
machine. A message that says "see docs/system-contract.md" therefore names a file the reader does
not have, at the moment they are already stuck -- which is the defect ATO-2125 exists to remove.

The answer is a URL rather than a path: it resolves from any working directory, on a laptop that
never cloned anything, and it is the same document the maintainers read. One function builds every
such link so the branch and the repository are stated once.
"""

from __future__ import annotations

#: The public repository's documentation tree, on its default branch. The public projection of this
#: repository, which is what an installed copy came from.
DOCS_BASE_URL = "https://github.com/ContextCommons/memrank/blob/main/docs"


def doc_url(relative: str) -> str:
    """The published URL of ``relative``, a path under the repository's ``docs/`` directory.

    Args:
        relative: A path relative to ``docs/``, e.g. ``system-contract.md``.

    Returns:
        An absolute https URL a reader can open from anywhere.
    """
    return f"{DOCS_BASE_URL}/{relative.lstrip('/')}"
