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
"""The shared launch core: run-id shape and the engine-tag source rules.

The run id is a shell token, an S3 prefix, and a directory name at once, so its alphabet
is a contract -- ``remote_command`` refuses anything outside it, and the accounts API's
runs table enforces it as a check constraint. One shape, defined once, pinned here.
"""
from __future__ import annotations

import re

from memrank.runs.registry import mint_run_id
from tests import withheld

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def test_minted_ids_stay_inside_the_safe_alphabet():
    """Every generated id must satisfy the alphabet remote_command and the DB enforce."""
    for run_id in (mint_run_id("demo"), mint_run_id("beam", "smoke"),
                   mint_run_id("locomo", "smoke", "fast")):
        assert _SAFE.match(run_id), run_id


def test_minted_ids_carry_benchmark_slice_and_tier_in_order():
    run_id = mint_run_id("locomo", "smoke", "fast")
    stamp, benchmark, slice_, tier, suffix = run_id.split("__")
    assert (benchmark, slice_, tier) == ("locomo", "smoke", "fast")
    assert len(suffix) == 6


def test_back_to_back_ids_differ():
    """The uuid suffix keeps two runs in the same second distinct."""
    assert mint_run_id("demo") != mint_run_id("demo")


def test_the_engine_tag_recorded_is_the_one_the_graph_declares(monkeypatch):
    """A tag used to arrive from the bench context or the environment and be pasted into an image
    reference, so a launch could fail with "engine_tags in the bench context has no entry for
    'mem0'" on a target whose image was never in doubt. It is read off the reference that will
    actually be pulled, which for a vendor image is the vendor's own tag."""
    from memrank.placement.graph import engine_image, reference_parts
    from memrank.targets import resolve_target

    monkeypatch.setenv("MEMRANK_ENGINES_REGISTRY", "registry.example/engines")
    tag = lambda ref: reference_parts(engine_image(resolve_target(ref)))[1]  # noqa: E731
    assert tag("hindsight") == "0.6.2"
    withheld.require("mem0")
    assert tag("mem0") == "mem0-server-mem0-poc-1"
