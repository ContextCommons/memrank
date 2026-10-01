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
"""In-process systems and evaluations the instrument tests run against.

Nothing here reaches a network, a dataset or a model. A fake that needs to fail does it on a
named task, so "the loop never stops on a task's failure" is testable rather than asserted.
"""
from __future__ import annotations

from memrank.core import Document, Recall
from memrank.instrument.evaluation import Clearing, Evaluation
from memrank.instrument.kinds import Memory
from memrank.instrument.measure import Decider, Measure, Scope, Value
from memrank.instrument.run import Answerer
from memrank.instrument.system import Assistant, Model, Retriever
from memrank.instrument.task import Expected, Task


class TinyMemory(Memory):
    """A memory that stores what it is told and returns what shares a word with the question."""

    name = "tiny-memory"
    engine_version = "tiny-1"

    def __init__(self, fails_on: str | None = None, fails_at: str = "retrieve") -> None:
        self.store: dict[str, list[Document]] = {}
        self.isolation: str | None = None
        self.prepared: list[str] = []
        self.cleaned = 0
        self.fails_on = fails_on
        self.fails_at = fails_at

    def prepare(self, isolation_unit: str) -> None:
        if self.fails_at == "prepare" and self.fails_on == isolation_unit:
            raise RuntimeError("prepare refused")
        self.isolation = isolation_unit
        self.prepared.append(isolation_unit)
        self.store.setdefault(isolation_unit, [])

    def ingest(self, documents: list[Document]) -> None:
        if self.fails_at == "ingest" and self.fails_on == self.isolation:
            raise RuntimeError("ingest refused")
        self.store[self.isolation or ""].extend(documents)

    def retrieve(self, query: str, k: int, user_id: str, query_timestamp=None):
        if self.fails_at == "retrieve" and self.fails_on == query:
            raise RuntimeError("retrieve refused")
        words = {w for w in query.lower().split() if w}
        held = self.store.get(user_id) or self.store.get(self.isolation or "") or []
        scored = [(len(words & set(d.content.lower().split())), d) for d in held]
        ranked = [d for score, d in sorted(scored, key=lambda p: -p[0]) if score][:k]
        return Recall(documents=ranked, declared={"results": [d.id for d in ranked]})

    def cleanup(self) -> None:
        self.cleaned += 1
        self.store.pop(self.isolation or "", None)
        self.isolation = None


class HalfAMemory(Memory):
    """A memory missing a required verb, so it cannot be instantiated -- nor run."""

    name = "half-a-memory"

    def prepare(self, isolation_unit: str) -> None: ...
    def ingest(self, documents: list[Document]) -> None: ...
    def cleanup(self) -> None: ...


class TinyModel(Model):
    """A model that answers for itself, so no answer writer is needed."""

    def complete(self, prompt: str) -> str:
        return f"about {prompt.rstrip('?').split()[-1]}"


class TinyAssistant(Assistant):
    def respond(self, messages):
        return messages[-1]["content"].upper()


class TinyRetriever(Retriever):
    """A retriever over a corpus of its own, which is why nothing is given to it."""

    CORPUS = (Document(id="c1", content="blue whales are the largest animal"),
              Document(id="c2", content="portland gets a lot of rain"))

    def rank(self, query: str, k: int):
        words = set(query.lower().split())
        return [d for d in self.CORPUS if words & set(d.content.split())][:k]


class FirstPassage(Answerer):
    """The plainest answer writer there is: it quotes the first passage. Deterministic."""

    name = "first-passage"

    def write(self, task, recalled) -> str:
        return recalled.documents[0].content if recalled.documents else ""


class Counts(Measure):
    """A measure that reads a trace field and counts the traces it was handed."""

    name = "counts"
    scope = Scope.RUN
    reads = ("recalled",)
    decider = Decider.MEMRANK

    def measure(self, traces, values):
        return [Value(measure=self.name, decider=self.decider, value=float(len(traces)))]


class ReadsTheUnknown(Measure):
    """A measure declaring it reads something nothing in a run produces."""

    name = "reads-the-unknown"
    reads = ("vibes",)

    def measure(self, traces, values):  # pragma: no cover - the run refuses first
        raise AssertionError("a refused measure must never run")


class NeedsAnAnswer(Measure):
    """A measure that reads `answered`, which a memory cannot produce by itself."""

    name = "needs-an-answer"
    reads = ("answered",)

    def measure(self, traces, values):
        return [Value(measure=self.name, decider=self.decider, task_id=t.task_id,
                      value=bool(t.answered)) for t in traces]


class Broke(Measure):
    """Whether a task broke, per task. Enough to test the loop without a shipped measure."""

    name = "broke"
    reads = ("error",)
    decider = Decider.MEMRANK

    def measure(self, traces, values):
        return [Value(measure=self.name, decider=self.decider, task_id=t.task_id,
                      value=t.error is not None,
                      why=t.error.message if t.error else None) for t in traces]


DOCUMENTS = (
    Document(id="d1", content="Alex is a marine biologist in Portland."),
    Document(id="d2", content="Dana visits in April for the whale migration."),
)


def two_task_evaluation(measures=(), clearing=Clearing.PER_GROUP, context=DOCUMENTS
                        ) -> Evaluation:
    """Two tasks in one group, over two documents. The smallest thing that is a real run."""
    return Evaluation(
        name="tiny", version="1",
        tasks=(
            Task(id="t_job", prompt="What is Alex?", group="g1", context=tuple(context),
                 expected=Expected(answers=("marine biologist",),
                                   required_spans=("marine biologist",))),
            Task(id="t_visit", prompt="When does Dana visit?", group="g1",
                 context=tuple(context),
                 expected=Expected(answers=("April",), required_spans=("april",))),
        ),
        measures=tuple(measures), clearing=clearing)
