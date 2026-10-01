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
"""Evaluations as ``memrank run`` takes them: a shipped benchmark or an evaluation file.

    resolve("locomo")            # a shipped evaluation
    resolve("./my-eval.yaml")    # your own
"""

from __future__ import annotations

from pathlib import Path

from memrank.definitions.base import Case, Definition, Grader, ModelJudge, Verdict
from memrank.definitions.file import FileDefinition, is_file_ref
from memrank.definitions.shipped import BenchmarkDefinition

__all__ = ["Case", "Definition", "Grader", "ModelJudge", "Verdict", "is_file_ref", "resolve"]


def resolve(ref: str) -> Definition:
    """The evaluation ``ref`` names: a path to an evaluation file, else a shipped ref."""
    if is_file_ref(ref):
        return FileDefinition(Path(ref))
    return BenchmarkDefinition(ref)
