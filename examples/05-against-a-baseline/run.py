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
"""Your system against the two controls, so a number has something to mean.

    uv run python examples/05-against-a-baseline/run.py

`no-context` is told nothing and answers anyway: the floor a system must beat
to have done anything at all. `full-context` is handed every document with no
retrieval: the ceiling retrieval is trying to reach. A score without those
two beside it says nothing.

Everything here is named by string, which is the form a config file and the
command line have. `memrank.catalog()` is where the strings come from.
"""

import memrank

# What memrank ships, printed under both the string names used below and the
# Python names that lead to the class and the function themselves.
memrank.catalog()
print()

# One evaluation object stands behind every run below, so the tasks and the
# measures are the same thing in all three.
evaluation = memrank.evaluation("demo")
mine = memrank.system("word-overlap")
result_mine = evaluation.run(system=mine)

for control_name in ("no-context", "full-context"):
    control = memrank.system(control_name)
    result_control = evaluation.run(system=control)
    # A paired reading refuses unless both results are of the same evaluation
    # at the same version, then pairs task by task. It never says "better".
    reading = memrank.paired(result_mine, result_control)
    print(reading)
    print()
