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
"""Jev, TypeSafe's System One model, as a judge: typed questions in, probabilities out.

``--judge typesafe:jev-1.14.0`` grades every answer with one request of typed questions
(:mod:`.questions`), decided by fixed rules in code (:mod:`.rules`). It is a judge memrank offers,
not its default.
"""

from __future__ import annotations

from memrank.judging.jev.client import Decision, JevClient
from memrank.judging.jev.grade import JevJudge
from memrank.judging.jev.questions import QUESTION_SET
from memrank.judging.jev.rules import THRESHOLD

__all__ = ["QUESTION_SET", "THRESHOLD", "Decision", "JevClient", "JevJudge"]
