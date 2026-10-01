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
"""An eval variant is an evaluation, not an option on one.

`--tier 100k --slice smoke` reads like `--workers 4` -- a knob on a run. It is not: it selects
WHICH evaluation is being run, and numbers do not carry across those selections. BEAM alone is
three tiers x three slices, all previously called "beam".

The leaderboard already understood this (`board_key` segregates by tier and slice), but the naming
did not, and the artifact layer paid for it: two runs of one benchmark at different slices wrote
the same two filenames, so the second silently replaced the first.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memrank.runner import app

runner = CliRunner()

_FIXTURE = [
    {"sample_id": f"s{i}",
     "conversation": {
         "speaker_a": "A", "speaker_b": "B",
         "session_1": [{"dia_id": f"D{i}:1", "speaker": "A", "text": "we met twice"}],
         "session_1_date_time": "1:00 pm on 8 May, 2023"},
     "qa": [{"category": 1, "question": "how many times?", "answer": "twice",
             "evidence": [f"D{i}:1"]}]}
    for i in range(1, 4)
]


def _run(tmp_path, monkeypatch, eval_arg, run_id):
    """One in-process run against the no-memory arm, writing into a shared output dir."""
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps(_FIXTURE), encoding="utf-8")
    monkeypatch.setenv("LOCOMO_DATA_PATH", str(data))
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    # `--no-judge` because locomo now judges by default -- the judge IS its metric -- and what is
    # under test here is where artifacts are FILED, not what they score. Without it these runs
    # would refuse for want of an ANTHROPIC_API_KEY.
    return runner.invoke(app, ["submit", "no-context", *eval_arg, "--on", "none", "--no-judge",
                               "--output-dir", str(tmp_path / "out"), "--run-id", run_id])


def test_two_slices_of_one_benchmark_do_not_overwrite_each_other(tmp_path, monkeypatch):
    """The headline. Before eval refs, both runs wrote `no-context__locomo.json` and
    `summary__locomo.json`, so whichever ran second was the only one left on disk -- two
    different evaluations, one filename, no warning."""
    first = _run(tmp_path, monkeypatch, ["locomo:smoke"], "r1")
    assert first.exit_code == 0, first.output
    second = _run(tmp_path, monkeypatch, ["locomo:mini"], "r2")
    assert second.exit_code == 0, second.output

    out = tmp_path / "out"
    cells = sorted(p.name for p in out.glob("*__*.json") if not p.name.startswith("summary__"))
    summaries = sorted(p.name for p in out.glob("summary__*.json"))

    assert len(cells) == 2, f"one artifact per evaluation, got {cells}"
    assert len(summaries) == 2, f"one summary per evaluation, got {summaries}"
    # And they describe different evaluations, not the same one written twice.
    unit_counts = {json.loads((out / c).read_text(encoding="utf-8"))["n_units"] for c in cells}
    assert unit_counts == {1, 3}, f"smoke is 1 conversation, mini is 3; got {unit_counts}"


def test_every_listed_ref_resolves_to_the_evaluation_it_names():
    """`evals ls` is only useful if everything it prints can be run, and round-trips: the ref a
    run is filed under must rebuild the same evaluation."""
    from memrank.benchmarks import resolve_eval
    from memrank.benchmarks.refs import list_eval_refs

    for ref in list_eval_refs():
        bench, canonical = resolve_eval(ref)
        assert canonical == ref, f"{ref} canonicalised to {canonical}"
        again, _ = resolve_eval(canonical)
        # getattr on both: `demo` declares no slices and stores none, which is why the ref
        # layer never hands it one.
        def shape(b):
            return b.name, getattr(b, "tier", None), getattr(b, "slice", None)
        assert shape(again) == shape(bench)


def test_the_bare_form_canonicalises_to_an_explicit_evaluation():
    """`beam` was runnable and meant "whatever tier is first". It still runs -- but what it RAN is
    recorded as `beam:100k`, so nothing downstream has to reconstruct the default."""
    from memrank.benchmarks.refs import parse_eval_ref

    assert parse_eval_ref("beam")[2] == "beam:100k"
    assert parse_eval_ref("locomo")[2] == "locomo", "no tiers, so the bare form IS canonical"


def test_an_unknown_variant_names_the_ones_that_exist():
    from memrank.benchmarks.refs import parse_eval_ref
    from memrank.targets.resolve import RefError

    with pytest.raises(RefError) as exc:
        parse_eval_ref("beam:2m")
    assert "beam:100k" in str(exc.value) and "beam:1m" in str(exc.value)


def test_compose_ref_overlays_settings_onto_the_bare_form():
    """The application layer still says `eval + tier/slice settings`; with the flags retired,
    the rendered command carries ONLY the ref, so the ref must absorb the settings -- a bare
    `beam` beside tier=500k means beam:500k, not the default the bare form canonicalises to."""
    from memrank.benchmarks.refs import compose_ref

    assert compose_ref("beam", tier="500k") == "beam:500k"
    assert compose_ref("beam", slice="smoke") == "beam:100k-smoke"
    assert compose_ref("beam", tier="1m", slice="mini") == "beam:1m-mini"
    assert compose_ref("locomo", slice="smoke") == "locomo:smoke"
    assert compose_ref("beam") == "beam:100k"


def test_compose_ref_refines_a_preset_but_refuses_a_contradiction():
    """`beam:1m` beside slice=smoke is a refinement; beside tier=100k it names two different
    evaluations, and arbitrating would mislabel a run. Matching values pass through -- the
    config path derives them from the same ref."""
    from memrank.benchmarks.refs import compose_ref
    from memrank.targets.resolve import RefError

    assert compose_ref("beam:1m-mini", tier="1m", slice="mini") == "beam:1m-mini"
    assert compose_ref("beam:1m", slice="smoke") == "beam:1m-smoke"
    with pytest.raises(RefError):
        compose_ref("beam:1m", tier="100k")


def test_the_retired_flags_point_at_the_ref():
    """`--slice smoke` silently did nothing on a benchmark with no slices, and on one with slices
    it produced an artifact name that collided. Refused, with the replacement spelled out."""
    result = runner.invoke(app, ["submit", "no-context", "demo", "--slice", "smoke", "--on", "none"])
    assert result.exit_code != 0
    assert "eval ref" in result.output
