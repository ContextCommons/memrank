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
"""A benchmark can no longer overwrite a field of the result without being told.

``rollup()`` and ``config_for_receipt()`` are the two surfaces a benchmark names its own keys
on, and both are spread LAST into the artifact dict -- deliberately, and pinned there by
``test_golden_run.py``. That position is what let a benchmark naming one of its keys
``composite`` replace the measured value with its own: the artifact stayed well-formed, the
run reported success, and the number a reader ranked on was not the measurement.

The spread position is unchanged here. What changed is that a key which WOULD shadow a fixed
field is refused at assembly. This exercises it through a real ``memrank.run`` rather than by
constructing an ``EvalResult`` directly, because the claim is that a RUN fails -- a guard that
only fires when a test builds the object by hand would not have caught the case it exists for.

The last test is the enumeration: it walks the benchmark registry rather than checking the
five benchmarks that exist today, so a SIXTH whose keys collide fails here.
"""
from __future__ import annotations

import pytest

# `memrank.run` is now the typed run over the seven (memrank/instrument/); the cell run
# that produces an `EvalResult` is imported from its own module, which is where the CLI
# and the cloud reach it too.
from memrank.benchmarks import REGISTRY
from memrank.evaluation.api import run as run_cell
from memrank.evaluation.result import _CONTRACT_FIELDS, ShadowedResultField
from tests.fakes import FakeAdapter, FakeBenchmark

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    return tmp_path


def _adapter() -> FakeAdapter:
    return FakeAdapter(name="fake", responses={"q1": [], "q2": []})


class _ShadowsViaRollup(FakeBenchmark):
    """Names its run-level metric ``composite`` -- the field the reader ranks on."""

    name = "shadows-via-rollup"

    def rollup(self, per_unit_scores, *, ranked=True):
        return {"composite": 9.9}


class _ShadowsViaConfig(FakeBenchmark):
    """Names a reproducibility knob ``receipt`` -- the field that makes a score mean something."""

    name = "shadows-via-config"

    def config_for_receipt(self):
        return {**super().config_for_receipt(), "receipt": {"faked": True}}


class _DeclaresItsOwnKey(FakeBenchmark):
    """The ordinary case: a run-level metric the schema does not own."""

    name = "declares-its-own-key"

    def rollup(self, per_unit_scores, *, ranked=True):
        return {"evidence_recall": 0.5}


def test_a_rollup_key_that_shadows_a_result_field_fails_the_run(isolated):
    with pytest.raises(ShadowedResultField) as raised:
        run_cell(_adapter(), _ShadowsViaRollup(), repeats=1)

    message = str(raised.value)
    assert "shadows-via-rollup" in message, "the message must name the benchmark to fix"
    assert "'composite'" in message, "the message must name the key that collided"
    assert "rollup()" in message, "and the method that declared it, not just 'a rollup key'"


def test_a_config_for_receipt_key_that_shadows_a_result_field_fails_the_run(isolated):
    """The other open surface. It is checked separately because it is a different method.

    A message that said "rollup" for a `config_for_receipt` collision would send the author to
    a method that does not contain the key.
    """
    with pytest.raises(ShadowedResultField) as raised:
        run_cell(_adapter(), _ShadowsViaConfig(), repeats=1)

    message = str(raised.value)
    assert "shadows-via-config" in message
    assert "'receipt'" in message
    assert "config_for_receipt()" in message


def test_a_benchmark_declaring_its_own_key_still_reaches_the_artifact(isolated):
    """The open merge stays open -- this is a guard, not a closed key set.

    Refusing collisions must not cost a benchmark the ability to carry its own run-level
    numbers, which is the whole reason the spread is there.
    """
    result = run_cell(_adapter(), _DeclaresItsOwnKey(), repeats=1)

    assert result.to_dict()["evidence_recall"] == 0.5


def test_no_registered_benchmark_declares_a_key_the_result_owns():
    """Every benchmark in the registry, at both `ranked` settings.

    Both settings because `rollup`'s contract makes the adapter's `ranks_results` an input: a
    benchmark may report a different set of keys for an arm that returns its store unordered,
    and a collision reachable only on that path is one nobody would run into until they did.
    """
    offenders = []
    for name, cls in sorted(REGISTRY.items()):
        benchmark = cls()
        declared = {
            "rollup(ranked=True)": set(benchmark.rollup([], ranked=True)),
            "rollup(ranked=False)": set(benchmark.rollup([], ranked=False)),
            "config_for_receipt()": set(benchmark.config_for_receipt()),
        }
        for surface, keys in declared.items():
            for key in sorted(keys & _CONTRACT_FIELDS):
                offenders.append(f"{name}.{surface} declares {key!r}")
    assert offenders == [], (
        "these declare a key the result owns as a fixed field of its own, so every run of "
        "them would now be refused at assembly:\n  " + "\n  ".join(offenders) + "\n\n"
        "Rename the benchmark's key. The names that are taken are the fields of "
        "memrank.evaluation.result.EvalResult.")
