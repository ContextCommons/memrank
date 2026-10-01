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
"""A first number, with its measure and its decider, in about a second.

    uv run python examples/01-first-result/run.py

`TFIDF` and `Demo` are the quick-start pair memrank ships: no engine, no
network, no key. Both are imported by name, so an editor follows them to their
own definition. Printing the result is the whole read -- memrank lays it out,
and nothing here formats it.
"""

from memrank.evaluations import Demo
from memrank.systems import TFIDF

system = TFIDF()
evaluation = Demo()
result = evaluation.run(system=system)

print(result)
