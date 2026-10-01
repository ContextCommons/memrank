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
"""Mark a hidden engine-hosting command deprecated with a one-line stderr notice.

The commands are registered through :func:`deprecated` rather than edited, so the function
behind each stays the one its other callers and tests import.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from memrank import deprecation
from memrank.term import style

F = TypeVar("F", bound=Callable[..., Any])


def notice(command: str) -> None:
    """Print the deprecation notice for ``memrank <command>`` to stderr."""
    style.warn(deprecation.message(f"`memrank {command}`"))


def deprecated(command: str, function: F) -> F:
    """``function``, announcing first that ``memrank <command>`` is deprecated.

    `functools.wraps` keeps the signature Typer reads its options from.
    """

    @functools.wraps(function)
    def announced(*args: Any, **kwargs: Any) -> Any:
        notice(command)
        return function(*args, **kwargs)

    return announced  # type: ignore[return-value]
