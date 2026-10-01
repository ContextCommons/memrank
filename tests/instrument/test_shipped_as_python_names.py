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
"""`memrank.systems`, `memrank.evaluations`, `memrank.catalog()`: the catalog without a string.

`memrank.system("word-overlap")` presupposes the string. Nothing in an editor follows a string,
so a reader who has one cannot reach the class, and a reader who has none cannot find out what
exists. These hold the three things that fixes: the systems are the registered classes and not
copies of them, every shipped evaluation is an `Evaluation` subclass whose constructor builds
what the string form builds, and one call prints the whole catalog.

The downloading three are checked WITHOUT downloading: the class exists, its constructor is
typed, and a tier it does not have is refused before anything is fetched. A test that pulled
LoCoMo to prove `LoCoMo` is a class would be a test nobody runs offline.
"""
from __future__ import annotations

import inspect
import typing

import pytest

from memrank import evaluations, systems
from memrank.adapters import REGISTRY as SYSTEM_REGISTRY
from memrank.benchmarks import REGISTRY as EVALUATION_REGISTRY
from memrank.instrument.catalog import QUICK_START, catalog, evaluation, system
from memrank.instrument.evaluation import Evaluation

#: The evaluations whose `load()` may reach the network. Named here, never constructed.
DOWNLOADS = ("LoCoMo", "LongMemEval", "BEAM")
#: The evaluations that are wholly in-tree, so a test may actually build one.
OFFLINE = ("Demo", "RelationGraph")


@pytest.mark.parametrize("entry", systems.SHIPPED, ids=lambda e: e.python_name)
def test_every_shipped_system_is_the_registered_class_object(entry):
    """The same class, not a near-copy: a parallel hierarchy fails every registry isinstance."""
    named = getattr(systems, entry.python_name)

    assert named is SYSTEM_REGISTRY[entry.name]
    assert named.name == entry.name


def test_the_systems_module_covers_the_registry_exactly():
    """A registered system with no Python name is one a reader can only reach by guessing."""
    assert {e.name for e in systems.SHIPPED} == set(SYSTEM_REGISTRY)
    assert len(systems.SHIPPED) == len(SYSTEM_REGISTRY)


#: The one entry per shipped evaluation, keyed by the Python name the classes are reached under.
BY_PYTHON_NAME = {entry.python_name: entry for entry in evaluations.SHIPPED}


@pytest.mark.parametrize("python_name", OFFLINE)
def test_an_offline_evaluation_class_builds_what_the_string_form_builds(python_name: str):
    """Built for real, because these two need nothing. `Demo()` is `evaluation("demo")`."""
    entry = BY_PYTHON_NAME[python_name]

    built = getattr(evaluations, python_name)()
    by_string = evaluation(entry.name)

    assert isinstance(built, Evaluation)
    assert built.name == entry.name
    assert built.tasks == by_string.tasks
    assert (built.version, built.clearing, built.metadata) == (
        by_string.version, by_string.clearing, by_string.metadata)
    assert [type(m) for m in built.measures] == [type(m) for m in by_string.measures]


@pytest.mark.parametrize("python_name", OFFLINE)
def test_an_offline_evaluation_instance_copies_and_revalidates(python_name: str):
    """A subclass of a frozen model still has to survive the two moves every model gets."""
    built = getattr(evaluations, python_name)()

    copied = built.model_copy()
    revalidated = Evaluation(**{field: getattr(built, field) for field in Evaluation.model_fields})

    assert copied.tasks == built.tasks
    assert type(revalidated) is Evaluation
    assert (revalidated.name, revalidated.version, revalidated.tasks) == (
        built.name, built.version, built.tasks)


@pytest.mark.parametrize("python_name", DOWNLOADS)
def test_a_downloading_evaluation_class_exists_and_is_typed(python_name: str):
    """Signature only -- constructing one fetches a dataset, which is not this test's business."""
    shipped = getattr(evaluations, python_name)
    hints = typing.get_type_hints(shipped.__init__)

    assert isinstance(shipped, type)
    assert issubclass(shipped, Evaluation)
    assert all(parameter in hints
               for parameter in inspect.signature(shipped.__init__).parameters
               if parameter != "self")


def test_an_unknown_beam_tier_is_refused_before_anything_is_fetched():
    """The benchmark's own error, raised by its constructor, so no download is even begun."""
    with pytest.raises(ValueError) as refused:
        evaluations.BEAM(tier="9000k")

    assert "9000k" in str(refused.value)


def test_the_evaluations_module_covers_the_registry_exactly():
    """A registered evaluation with no class is one a reader can only reach by guessing."""
    assert {e.name for e in evaluations.SHIPPED} == set(EVALUATION_REGISTRY)
    assert all(issubclass(getattr(evaluations, e.python_name), Evaluation)
               for e in evaluations.SHIPPED)


def test_the_catalog_lists_every_registered_name_once(capsys):
    """One call, nothing presupposed: every name that exists, printed and returned."""
    listed = catalog()
    printed = capsys.readouterr().out

    assert len(listed.systems) == len(SYSTEM_REGISTRY)
    assert len(listed.evaluations) == len(EVALUATION_REGISTRY)
    assert {e.name for e in listed.systems} == set(SYSTEM_REGISTRY)
    assert {e.name for e in listed.evaluations} == set(EVALUATION_REGISTRY)
    for name in (*SYSTEM_REGISTRY, *EVALUATION_REGISTRY):
        assert printed.count(f"{name!r}") == 1, f"{name} is printed {printed.count(name)} times"


def test_the_refusals_point_at_the_python_names():
    """A reader who guessed a string is the reader who did not know the catalog existed."""
    with pytest.raises(ValueError) as system_refused:
        system("word-overlaps")
    with pytest.raises(Exception) as evaluation_refused:
        evaluation("demonstration")

    assert "memrank.systems" in str(system_refused.value)
    assert "memrank.evaluations" in str(evaluation_refused.value)



def test_the_quick_start_names_a_shipped_system_and_a_shipped_evaluation():
    """The pair is declared once, and a declared name nothing ships would fail the catalog."""
    assert QUICK_START.system in {e.python_name for e in systems.SHIPPED}
    assert QUICK_START.evaluation in {e.python_name for e in evaluations.SHIPPED}


def test_the_catalog_opens_with_the_declared_quick_start(capsys):
    """The import hints `catalog()` prints are the declaration's, not a string of their own.

    They printed `WordOverlap` and `Demo` for a release after 0.4.4 made the pair TFIDF and SQuAD,
    because the hint was a literal beside the table rather than read from anything.
    """
    first_system, first_evaluation = catalog().quick_start()
    printed = capsys.readouterr().out

    assert f"from memrank.systems import {QUICK_START.system}`" in printed
    assert f"from memrank.evaluations import {QUICK_START.evaluation}`" in printed
    assert f'memrank.system("{first_system.name}")' in printed
    assert f'memrank.evaluation("{first_evaluation.name}")' in printed
