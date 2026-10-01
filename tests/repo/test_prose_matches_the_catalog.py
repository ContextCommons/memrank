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
"""What published prose says ships is what the registries say ships.

0.4.4 added TFIDF, BM25 and SQuAD. `memrank.catalog()` reported eleven systems and six
evaluations the moment the code landed; five pages went on saying five evaluations, three module
lists left SQuAD or TFIDF out, and `reference/measure.md` said four measures ship where
`docs/measures.md` said five. Each was a snapshot of a count, and nothing held a snapshot to the
thing it counted.

Two claim shapes are read here, sentence by sentence:

- **a count of what ships** -- "ships five evaluations", "supplies eleven systems and six
  evaluations", "Five measures ship", "all six registered benchmarks" -- which must equal
  `len(memrank.systems.SHIPPED)`, `len(memrank.evaluations.SHIPPED)`, the measure classes
  `memrank/instrument/measures.py` defines, or `len(memrank.benchmarks.REGISTRY)`;
- **a membership list** -- "`memrank.systems` -- the module: `A`, `B`, ..." and "Evaluations
  ship for `a`, `b` ..." -- which must name exactly the shipped set.

A count of a SUBSET ("three shipped measures produce a quality value") is not one of these shapes
and is not read. A count phrased some third way is missed; that is this guard's honest limit.
"""
from __future__ import annotations

import inspect
import re

from memrank import evaluations, systems
from memrank.benchmarks import REGISTRY as BENCHMARKS
from memrank.instrument import measures as measures_module
from memrank.instrument.measure import Measure
from tests.repo.prose import Sentence, published_sentences, sentences_of

NUMBERS = {word: value for value, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
NUMBER = r"(\d+|" + "|".join(NUMBERS) + r")"
KIND = r"(systems?|evaluations?|measures?|benchmarks?)(?:\(s\))?"

MEASURE_CLASSES = sorted(
    name for name, value in inspect.getmembers(measures_module, inspect.isclass)
    if issubclass(value, Measure) and value.__module__ == measures_module.__name__)

#: What each noun counts. A benchmark is the command line's word for an evaluation's loader.
SHIPPED_COUNT = {"system": len(systems.SHIPPED), "evaluation": len(evaluations.SHIPPED),
                 "measure": len(MEASURE_CLASSES), "benchmark": len(BENCHMARKS)}

#: "ships N kind", "supplies N kind and N kind", "all N registered kind", "N kind (memrank) ship(s)".
COUNTS = (
    re.compile(rf"\b(?:ships?|supplies|supply)\s+((?:(?:the\s+)?{NUMBER}\s+(?:\[)?{KIND}"
               rf"(?:\]\([^)]*\))?(?:,\s*|\s+and\s+)?)+)", re.I),
    re.compile(rf"\ball\s+({NUMBER}\s+(?:registered|shipped)\s+{KIND})", re.I),
    re.compile(rf"\b({NUMBER}\s+{KIND})\s+(?:memrank\s+)?ships?\b", re.I),
)
PAIR = re.compile(rf"{NUMBER}\s+(?:registered\s+|shipped\s+)?(?:\[)?{KIND}", re.I)

#: "`memrank.systems` -- the module...: `A`, `B`" and "Evaluations ship for `a`, `b`".
MODULE_LIST = re.compile(r"`memrank\.(systems|evaluations)` -- the module[^:]*:(.*)")
SHIP_FOR = re.compile(r"\b(Systems|Evaluations) ship for (.*)", re.I)
NAME = re.compile(r"`([\w-]+)`")


def _number(word: str) -> int:
    return int(word) if word.isdigit() else NUMBERS[word.lower()]


def count_offences(sentence: Sentence) -> list[str]:
    """Every count of what ships, in one sentence, that disagrees with the registry."""
    found = []
    for pattern in COUNTS:
        for claim in pattern.finditer(sentence.text):
            for number, kind in PAIR.findall(claim.group(1)):
                noun = kind.lower().rstrip("s")
                if _number(number) != SHIPPED_COUNT[noun]:
                    found.append(f"{sentence.where} says {number} {kind}; "
                                 f"{SHIPPED_COUNT[noun]} ship: {sentence.text!r}")
    return found


def list_offences(sentence: Sentence) -> list[str]:
    """A membership list of what ships that is not exactly what ships."""
    found = []
    for pattern, by in ((MODULE_LIST, "python_name"), (SHIP_FOR, "name")):
        for module, listed in pattern.findall(sentence.text):
            shipped = systems.SHIPPED if module.lower().startswith("system") else evaluations.SHIPPED
            expected = {getattr(entry, by) for entry in shipped}
            named = set(NAME.findall(listed))
            if named != expected:
                found.append(f"{sentence.where} lists {module}: missing "
                             f"{sorted(expected - named)}, extra {sorted(named - expected)}")
    return found


def test_every_count_of_what_ships_is_the_registrys():
    found = [o for s in published_sentences() for o in count_offences(s)]
    assert not found, "\n".join(found)


def test_every_list_of_what_ships_is_the_registrys():
    found = [o for s in published_sentences() for o in list_offences(s)]
    assert not found, "\n".join(found)


def test_the_scan_reads_claims_in_the_corpus():
    """Guards the guard: a pattern that matched nothing would make both tests pass."""
    counted = [s for s in published_sentences()
               if any(p.search(s.text) for p in COUNTS)]
    listed = [s for s in published_sentences()
              if MODULE_LIST.search(s.text) or SHIP_FOR.search(s.text)]
    assert len(counted) >= 5, f"only {len(counted)} count claim(s) read"
    assert len(listed) >= 3, f"only {len(listed)} membership list(s) read"


def test_the_scan_catches_the_claims_that_shipped_wrong():
    """The 0.4.4 leftovers, each caught -- counts restated against today's five evaluations."""
    def caught(text: str) -> int:
        return sum(len(count_offences(s)) + len(list_offences(s))
                   for s in sentences_of("d.md", text))

    assert caught("Memrank ships six evaluations, each bundling its own measures.") == 1
    assert caught("Memrank supplies eleven [systems](system.md) and six "
                  "[evaluations](evaluation.md), with what each needs.") == 1
    assert caught("Five measures ship with the package.") == 0
    assert caught("Four measures ship with the package.") == 1
    assert caught("Declaring nothing is the default: all six registered benchmarks do.") == 1
    assert caught("- `memrank.evaluations` -- the module: `Demo`, `RelationGraph`, `LoCoMo`, "
                  "`BEAM`.") == 1
    assert caught("Three shipped measures produce a quality value.") == 0
