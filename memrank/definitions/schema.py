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
"""The shape of an evaluation file, field by field. Unknown keys are refused.

A YAML file carries a header (``name``, ``version``, ``description``, a default ``grade``) and
its ``cases`` -- written inline, read from a JSONL file (``cases: {file: cases.jsonl}``), or
printed by a program (``cases: {command: [...]}``). A bare ``.jsonl`` file is a list of cases
named after the file. Each case is past conversations to feed (``history``, optional) and
``questions``; a case written as one question on its own is a question-only case.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: The built-in graders, by the name a file uses.
BUILTIN_GRADERS = ("exact", "choice", "numeric", "judge", "rubric")
#: The graders that call the judge model, and so need the organisation's Anthropic key.
MODEL_GRADERS = frozenset({"judge", "rubric"})

#: Ids travel in session ids and URLs, so they keep to a shell- and URL-safe alphabet.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
#: Evaluation names are recorded in run ids too, and are lowercase so two spellings are one.
NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$"

#: How long a case command or a command grader may take unless the file says otherwise.
DEFAULT_TIMEOUT_S = 600.0


def _text(value: Any) -> Any:
    """YAML reads ``2026-01-05`` as a date and ``2`` as a number; both are text here."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    return value


class _Strict(BaseModel):
    """Unknown keys refused; the fields named in ``TEXT`` read dates and numbers as text."""

    model_config = ConfigDict(extra="forbid")
    TEXT: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _as_text(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {key: _text(item) if key in cls.TEXT else item for key, item in value.items()}


class GraderSpec(_Strict):
    """How a question is graded: a built-in grader, or your program."""

    kind: Literal["exact", "choice", "numeric", "judge", "rubric"] | None = None
    command: list[str] | None = Field(default=None, min_length=1)
    tolerance: float | None = Field(default=None, ge=0)
    timeout_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _named(cls, value: Any) -> Any:
        return {"kind": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def _one_way(self) -> GraderSpec:
        if (self.kind is None) == (self.command is None):
            raise ValueError("give a built-in grader (one of "
                             f"{', '.join(BUILTIN_GRADERS)}) or a `command`, not both")
        if self.tolerance is not None and self.kind != "numeric":
            raise ValueError("`tolerance` is for the numeric grader")
        if self.timeout_s is not None and self.command is None:
            raise ValueError("`timeout_s` is for a command grader")
        return self

    @property
    def name(self) -> str:
        return self.kind or "command"

    @property
    def uses_model(self) -> bool:
        return self.kind in MODEL_GRADERS


class TurnSpec(_Strict):
    TEXT = ("text", "timestamp", "speaker")
    role: Literal["user", "assistant"] = "user"
    speaker: str | None = None
    text: str = Field(min_length=1)
    timestamp: str | None = None



class SessionSpec(_Strict):
    TEXT = ("id", "timestamp")
    id: str | None = Field(default=None, pattern=ID_PATTERN)
    timestamp: str | None = None
    turns: list[TurnSpec] = Field(min_length=1)



class QuestionSpec(_Strict):
    TEXT = ("id", "question", "category", "timestamp")
    id: str | None = Field(default=None, pattern=ID_PATTERN)
    question: str = Field(min_length=1)
    #: The reference answer, or a list of acceptable ones.
    answer: list[str] | None = None
    choices: list[str] | None = Field(default=None, min_length=2)
    rubric: list[str] | None = Field(default=None, min_length=1)
    category: str | None = None
    timestamp: str | None = None
    grade: GraderSpec | None = None


    @field_validator("answer", mode="before")
    @classmethod
    def _answers(cls, value: Any) -> Any:
        if value is None:
            return None
        values = value if isinstance(value, list) else [value]
        return [_text(item) for item in values]

    @field_validator("choices", "rubric", mode="before")
    @classmethod
    def _items(cls, value: Any) -> Any:
        return [_text(item) for item in value] if isinstance(value, list) else value


class CaseSpec(_Strict):
    TEXT = ("id",)
    id: str | None = Field(default=None, pattern=ID_PATTERN)
    history: list[SessionSpec] = Field(default_factory=list)
    #: Checked one by one (:class:`QuestionSpec`), so each question's problems are all found.
    questions: list[Any] = Field(min_length=1)



class SourceSpec(_Strict):
    """Cases kept outside the YAML: a JSONL file, or a program printing JSON lines."""

    file: str | None = None
    command: list[str] | None = Field(default=None, min_length=1)
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S, gt=0)

    @model_validator(mode="after")
    def _one_way(self) -> SourceSpec:
        if (self.file is None) == (self.command is None):
            raise ValueError("give the cases a `file` or a `command`, not both")
        return self


class HeaderSpec(_Strict):
    """An evaluation file's own fields; its ``cases`` are read apart, one by one."""

    TEXT = ("name", "version")

    name: str = Field(pattern=NAME_PATTERN)
    version: str | None = Field(default=None, pattern=VERSION_PATTERN)
    description: str | None = None
    grade: GraderSpec | None = None

