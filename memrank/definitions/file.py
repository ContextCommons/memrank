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
"""An evaluation file -- YAML, or a bare JSONL list of cases -- as a definition.

Reading the file checks everything that can be checked without running anything: the header,
every inline or JSONL case, every question against its grader. Cases a program prints are
checked the same way as soon as it has printed them -- once per seed, since the program is
given ``--seed``. A file that fails any check is refused whole (:class:`EvaluationInvalid`).

The fingerprint covers the file and every file its commands name (a case program, a grader
script, a JSONL case list), so editing any of them makes it a different evaluation.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from memrank.definitions import commands
from memrank.definitions.base import Case, Definition, Grader
from memrank.definitions.cases import Checked, read_case
from memrank.definitions.graders import grader_of
from memrank.definitions.problems import EvaluationInvalid, problems_of
from memrank.definitions.schema import GraderSpec, HeaderSpec, SourceSpec
from memrank.service.protocol import EVALUATIONS, QuestionOutcome

YAML_SUFFIXES = (".yaml", ".yml")
JSONL_SUFFIX = ".jsonl"
#: The evaluation-file format's version, recorded as the run's task version.
FORMAT_VERSION = 1


def is_file_ref(ref: str) -> bool:
    """Whether ``ref`` names an evaluation file rather than a shipped evaluation."""
    return "/" in ref or ref.endswith((*YAML_SUFFIXES, JSONL_SUFFIX))


def _jsonl(text: str, label: str, default: GraderSpec | None, checked: Checked) -> None:
    """Every non-blank line of ``text`` as a case, located as ``<label> line N``."""
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            checked.problems.append(f"{label} line {number}: not JSON ({exc.msg})")
            continue
        read_case(raw, f"{label} line {number}", len(checked.cases), default, checked)


def _read(path: Path, given: str) -> tuple[bytes, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvaluationInvalid(given, [f"it cannot be read: {exc.strerror}"]) from exc
    if path.suffix == JSONL_SUFFIX:
        return raw, None
    try:
        return raw, yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise EvaluationInvalid(given, [f"it is not valid YAML: {exc}"]) from exc


class FileDefinition(Definition):
    """An evaluation written by the person running it."""

    task_version = FORMAT_VERSION
    dataset_version: str | None = None
    name: str
    version: str | None
    description: str | None

    def __init__(self, path: Path) -> None:
        self.given = str(path)  # as the person wrote it, for every message about the file
        self.path = path.resolve()
        self.folder = self.path.parent
        self.source = str(self.path)
        raw, data = _read(self.path, self.given)
        self.default: GraderSpec | None = None
        self.cases_source: SourceSpec | None = None
        self.inline: Checked | None = None
        if data is None:  # a bare JSONL file: every line a case, named after the file
            problems = self._header({"name": self.path.stem.lower()}, "the file's name, ")
            self.inline = Checked()
            _jsonl(raw.decode("utf-8", errors="replace"), self.path.name, None, self.inline)
        elif isinstance(data, dict):
            problems = self._header({k: v for k, v in data.items() if k != "cases"}, "")
            problems += self._cases(data.get("cases"))
        else:
            problems = ["it must be a mapping with `name` and `cases`"]
        self._refuse_problems(problems + (self.inline.problems if self.inline else []))
        self.evaluation = f"{self.name}@{self.version}" if self.version else self.name
        self.fingerprint = self._fingerprint(raw)

    def _header(self, fields: dict[str, Any], where: str) -> list[str]:
        """The file's own fields, or what is wrong with them; its cases are checked anyway."""
        try:
            header = HeaderSpec.model_validate(fields)
        except ValidationError as exc:
            return [f"{where}{p}" for p in problems_of(exc, "", fields)]
        self.name, self.version, self.description = (header.name, header.version,
                                                     header.description)
        self.default = header.grade
        if header.name in EVALUATIONS:
            return [f"{where}`name`: {header.name} is a shipped evaluation's name; choose "
                    "another"]
        return []

    def _cases(self, cases: Any) -> list[str]:
        """Read inline or JSONL cases now; a command's cases wait for their seed."""
        if isinstance(cases, list) and cases:
            self.inline = Checked()
            for index, raw in enumerate(cases):
                read_case(raw, f"case {index + 1}", index, self.default, self.inline)
            return []
        if not isinstance(cases, dict):
            return ["`cases`: give a list of cases, {file: cases.jsonl} or "
                    "{command: [program, ...]}"]
        try:
            self.cases_source = SourceSpec.model_validate(cases)
        except ValidationError as exc:
            return problems_of(exc, "`cases`", cases)
        if self.cases_source.file is not None:
            return self._case_file(self.folder / self.cases_source.file)
        return []

    def _case_file(self, path: Path) -> list[str]:
        if not path.is_file():
            return [f"`cases.file`: {path} does not exist"]
        self.inline = Checked()
        _jsonl(path.read_text(encoding="utf-8", errors="replace"), path.name, self.default,
               self.inline)
        return []

    def _refuse_problems(self, problems: list[str]) -> None:
        if problems:
            raise EvaluationInvalid(self.given, problems)

    def _commands(self) -> list[list[str]]:
        """Every command the file itself names: its case program and its graders."""
        found = [self.cases_source.command] if self.cases_source and \
            self.cases_source.command else []
        graders = [self.default] if self.default else []
        if self.inline is not None:
            graders += [GraderSpec.model_validate(q["grader"]) for c in self.inline.cases
                        for q in c.queries]
        return found + [g.command for g in graders if g.command]

    def _fingerprint(self, raw: bytes) -> str:
        digest = hashlib.sha256(raw)
        named = [self.folder / self.cases_source.file] if self.cases_source and \
            self.cases_source.file else []
        for command in self._commands():
            named += commands.command_files(command, self.folder)
        for path in sorted(set(named)):
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
        return f"sha256:{digest.hexdigest()}"

    def load_cases(self, seed: int) -> list[Case]:
        if self.inline is not None:
            return list(self.inline.cases)
        assert self.cases_source is not None and self.cases_source.command is not None
        return list(_printed(self, seed))

    def grader_for(self, question: QuestionOutcome) -> Grader:
        return grader_of(GraderSpec.model_validate(question.grading["grader"]), self.folder)

    def uses_model(self, seed: int) -> bool:
        return any(GraderSpec.model_validate(q["grader"]).uses_model
                   for case in self.load_cases(seed) for q in case.queries)


def _printed(definition: FileDefinition, seed: int) -> tuple[Case, ...]:
    return _run_case_command(definition.source, definition.fingerprint, seed)


@lru_cache(maxsize=8)
def _run_case_command(source: str, fingerprint: str, seed: int) -> tuple[Case, ...]:
    """The cases the file's program prints for ``seed``: run once per file version and seed."""
    definition = FileDefinition(Path(source))
    if definition.fingerprint != fingerprint:
        raise EvaluationInvalid(definition.given, ["it changed while it was being read; run "
                                                   "it again"])
    spec = definition.cases_source
    assert spec is not None and spec.command is not None
    out = commands.run([*spec.command, "--seed", str(seed)], definition.folder, stdin="",
                       timeout_s=spec.timeout_s, what="The case command")
    checked = Checked()
    _jsonl(out, "the case command's output", definition.default, checked)
    if not checked.cases and not checked.problems:
        checked.problems.append("the case command printed no cases")
    definition._refuse_problems(checked.problems)
    return tuple(checked.cases)
