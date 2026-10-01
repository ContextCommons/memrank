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
"""A malformed evaluation file, refused before anything runs, naming each case and field.

Every problem in the file is collected and reported together -- ``case 3 (refunds), question 2,
`answer`: is required by the exact grader`` -- so one round of fixes is enough, and the refusal
is a step for the reader (``memrank evals check``), not a failure inside memrank.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from memrank.definitions.schema import ID_PATTERN, NAME_PATTERN, VERSION_PATTERN
from memrank.errors import ActionRequired
from memrank.outcome import Step

#: How many problems a refusal lists before it says how many more there are.
MAX_SHOWN = 20

_PATTERNS = {
    ID_PATTERN: "must be letters, digits, '.', '_' or '-', starting with a letter or digit "
                "(64 at most)",
    NAME_PATTERN: "must be lowercase letters, digits, '.', '_' or '-', starting with a letter "
                  "or digit (64 at most)",
    VERSION_PATTERN: "must be letters, digits, '.', '_' or '-' (32 at most)",
}
_MESSAGES = {"missing": "is required",
             "extra_forbidden": "is not a field memrank knows (check its spelling)"}
#: The lists whose items a location names by number: ``question 2``, not ``questions.1``.
_ITEMS = {"questions": "question", "history": "history session", "turns": "turn"}


class EvaluationInvalid(ActionRequired):
    """An evaluation file memrank will not run as written; ``problems`` says where and why."""

    def __init__(self, path: str, problems: list[str]) -> None:
        shown = [f"  - {problem}" for problem in problems[:MAX_SHOWN]]
        if len(problems) > MAX_SHOWN:
            shown.append(f"  ... and {len(problems) - MAX_SHOWN} more")
        count = "1 problem" if len(problems) == 1 else f"{len(problems)} problems"
        super().__init__(
            f"The evaluation {path} has {count}, so nothing ran:\n" + "\n".join(shown),
            steps=(Step("Fix the file, then check it again (this runs no agent):",
                        (f"memrank evals check {path}",)),))
        self.path = path
        self.problems = problems


def _message(error: dict[str, Any]) -> str:
    kind = error.get("type", "")
    if kind == "string_pattern_mismatch":
        return _PATTERNS.get(str(error.get("ctx", {}).get("pattern")), error["msg"])
    return _MESSAGES.get(kind, str(error["msg"]).removeprefix("Value error, "))


def _named(raw: Any, label: str) -> str:
    """``question 2 (q-refund)`` when the item carries an id, else ``question 2``."""
    ident = raw.get("id") if isinstance(raw, dict) else None
    return f"{label} ({ident})" if ident is not None else label


def located(loc: tuple[Any, ...], raw: Any) -> str:
    """A pydantic location read against the raw data: ``question 2 (q1), `grade.kind```."""
    parts: list[str] = []
    fields: list[str] = []
    index = 0
    while index < len(loc):
        key = loc[index]
        following = loc[index + 1] if index + 1 < len(loc) else None
        if key in _ITEMS and isinstance(following, int):
            raw = raw.get(key, [])[following] if isinstance(raw, dict) else None
            parts.append(_named(raw, f"{_ITEMS[key]} {following + 1}"))
            index += 2
            continue
        fields.append(str(key))
        raw = raw.get(key) if isinstance(raw, dict) else None
        index += 1
    if fields:
        parts.append(f"`{'.'.join(fields)}`")
    return ", ".join(parts)


def problems_of(exc: ValidationError, where: str, raw: Any) -> list[str]:
    """One line per error in ``exc``, each prefixed with ``where`` (``case 3 (refunds)``)."""
    lines = []
    for error in exc.errors():
        spot = ", ".join(part for part in (where, located(tuple(error["loc"]), raw)) if part)
        lines.append(f"{spot}: {_message(dict(error))}" if spot else _message(dict(error)))
    return lines
