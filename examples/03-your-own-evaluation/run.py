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
"""Your own questions, over your own context, measured by a shipped measure.

    uv run python examples/03-your-own-evaluation/run.py

An evaluation is a named, versioned bundle: its tasks, the measures it ships
with, and the rule for when state is cleared. The one below is written by
hand, and it is the same kind of object `memrank.evaluation("demo")` returns.
"""

import memrank
from memrank import Document, Evaluation, Expected, Task, WordMatch

# The last two notes answer nothing and share the words every question has.
# They are what makes ranking matter: a memory asked for its best two must
# leave them out.
NOTES = (
    Document(id="t1", user_id="acme",
             content="Acme moved to the enterprise plan in March."),
    Document(id="t2", user_id="acme",
             content="Acme's outage was traced to an expired webhook secret."),
    Document(id="t3", user_id="acme",
             content="What is the plan is what we do on the board."),
    Document(id="t4", user_id="acme",
             content="The plan is what the team does on Mondays."),
)

PLAN_TASK = Task(
    id="q_plan", prompt="What plan is Acme on?", group="acme", context=NOTES,
    expected=Expected(answers=("enterprise",),
                      required_spans=("enterprise",),
                      evidence_doc_ids=("t1",)),
)

OUTAGE_TASK = Task(
    id="q_outage", prompt="What caused Acme's outage?", group="acme",
    context=NOTES,
    expected=Expected(required_spans=("webhook secret",),
                      evidence_doc_ids=("t2",)),
)

# A negative task: correct here means the wrong memory is NOT surfaced.
REFUND_TASK = Task(
    id="q_refund", prompt="Was Acme promised a refund?", group="acme",
    context=NOTES,
    expected=Expected(forbidden_spans=("refund approved",),
                      polarity="negative"),
)

# Tasks that share state carry the same group: a run is told a group's
# documents once and clears between groups, never inside one. WordMatch is
# memrank's span proxy, and it says so on every value it produces.
TICKETS = Evaluation(
    name="tickets",
    version="internal@2026-09",
    measures=(WordMatch(),),
    tasks=(PLAN_TASK, OUTAGE_TASK, REFUND_TASK),
)

system = memrank.system("word-overlap")
result = TICKETS.run(system=system)

print(result)

# A trace is where you dig when a number is not what you expected: what was
# given, what came back, and in what order. Position is the rank -- the
# system ranked them, memrank did not.
trace = result.traces_of("q_plan")[0]

print(f"\ntrace q_plan: given {trace.given.count} document(s), "
      f"error {trace.error}")
for rank, document in enumerate(trace.recalled.documents, start=1):
    print(f"  {rank}. {document.id}: {document.content}")
