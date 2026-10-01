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
"""How far along a run is, counted in the work it does.

Every number here is calibrated against one real run -- hindsight x locomo, 272 documents and
4,620 retrievals, measured at 5.16 s/document and 0.293 s/retrieval. That run reported 0% for its
first five minutes and moved ten times in forty-six, because progress was
``queries_done / total_queries`` computed at unit boundaries.
"""
from __future__ import annotations

import pytest

from memrank.runs.progress import RunProgress, StageTimer

DOCUMENTS, RETRIEVALS, UNITS = 272, 4620, 10
INGEST_SECONDS, RETRIEVE_SECONDS = 5.16, 0.293


class _Clock:
    """A wall clock the test drives, standing in for ``time.monotonic``.

    The rate is measured against elapsed time now, the way tqdm measures it, so a test that wants a
    rate has to advance a clock. Recording item latency in a tight loop -- what these tests used to
    do -- describes a run that did all its work in zero seconds, and the old model happily reported
    ``done / summed_latency`` for it: a PER-WORKER rate that ignored concurrency entirely.
    """

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def progress(clock):
    p = RunProgress(_now=clock)
    p.plan(units=UNITS, documents=DOCUMENTS, retrievals=RETRIEVALS)
    return p


def _work(progress, *, documents=0, retrievals=0, workers=1):
    """Do some work at hindsight's measured rates, advancing the clock as it happens.

    ``workers`` divides the wall clock the way concurrency really does: N units in flight means N
    items land per item-latency, so the same work costs 1/N of the elapsed time while the summed
    latency is unchanged. That difference is the whole subject of this module.
    """
    clock = progress._now
    for _ in range(documents):
        clock.advance(INGEST_SECONDS / workers)
        progress.record("ingest", seconds=INGEST_SECONDS)
    for _ in range(retrievals):
        clock.advance(RETRIEVE_SECONDS / workers)
        progress.record("retrieve", seconds=RETRIEVE_SECONDS)


def test_ingest_is_worth_something(progress):
    """The defect, stated. Unit 1's whole ingest -- 29 documents, about two and a half minutes of a
    hindsight run -- used to be worth exactly zero, because progress counted only queries. It is
    now 149 seconds of measured work with a rate and an ETA attached."""
    _work(progress, documents=29)
    ingest = progress.as_dict()["ingest"]

    assert ingest["done"] == 29
    assert ingest["eta_seconds"] == pytest.approx((DOCUMENTS - 29) * INGEST_SECONDS, rel=0.01)


def test_the_overall_percentage_is_weighted_by_time_not_items(progress):
    """81 documents and 924 retrievals: 25.0% of the projected wall clock against 20.5% by item
    count. Truncated, never rounded -- a progress bar must not claim more than it has done, and
    rounding would let 99.6% print as 100% while the run is still working."""
    _work(progress, documents=81, retrievals=924)

    assert progress.pct == 24


def test_the_overall_bar_waits_rather_than_moving_backwards(progress):
    """During unit 1's ingest, retrieval has never run, so 4,620 items cost an unknown amount and
    the run cannot be projected. Reporting against the known stages alone would print 10% and then
    DROP it to 5% when retrieval joined the denominator -- and a progress bar that moves backwards
    is worse than one that waits."""
    _work(progress, documents=29)

    assert progress.pct is None

    _work(progress, retrievals=1)  # retrieval gets a rate; the run becomes projectable

    assert progress.pct is not None


def test_the_overall_bar_never_goes_backwards(progress):
    """The property the wait exists to protect, asserted over a whole simulated run."""
    seen = []
    for _ in range(UNITS):
        _work(progress, documents=27)
        seen.append(progress.pct)
        _work(progress, retrievals=462)
        seen.append(progress.pct)
    measured = [p for p in seen if p is not None]

    assert measured == sorted(measured), f"percentage fell: {measured}"


def test_the_stage_bars_move_from_the_very_first_item(progress):
    """What makes the overall bar's wait tolerable -- the screen is never blank."""
    _work(progress, documents=1)
    ingest = progress.as_dict()["ingest"]

    assert ingest["done"] == 1
    assert ingest["rate_per_second"] is not None and ingest["eta_seconds"] is not None


def test_each_stage_carries_its_own_rate(progress):
    """They differ by ~18x on hindsight; one shared figure would be wrong for both."""
    _work(progress, documents=81, retrievals=924)
    record = progress.as_dict()

    assert record["ingest"]["rate_per_second"] == pytest.approx(1 / INGEST_SECONDS, rel=0.01)
    assert record["retrieve"]["rate_per_second"] == pytest.approx(1 / RETRIEVE_SECONDS, rel=0.01)


def test_each_stage_carries_its_own_eta(progress):
    """What says whether a slow run is slow at ingest or slow at retrieval."""
    _work(progress, documents=81, retrievals=924)
    record = progress.as_dict()

    assert record["ingest"]["eta_seconds"] == pytest.approx((DOCUMENTS - 81) * INGEST_SECONDS,
                                                            rel=0.01)
    assert record["retrieve"]["eta_seconds"] == pytest.approx(
        (RETRIEVALS - 924) * RETRIEVE_SECONDS, rel=0.01)


def test_the_overall_eta_is_the_stages_summed(progress):
    """So the three rendered lines are consistent by construction rather than by coincidence."""
    _work(progress, documents=81, retrievals=924)
    record = progress.as_dict()

    assert record["eta_seconds"] == pytest.approx(
        record["ingest"]["eta_seconds"] + record["retrieve"]["eta_seconds"])


def test_nothing_is_estimated_before_anything_has_happened(progress):
    """The moment someone is most likely to be watching."""
    assert progress.pct is None
    assert progress.eta_seconds is None
    assert progress.as_dict()["ingest"]["rate_per_second"] is None


def test_an_eta_before_retrieval_has_run_is_marked_a_lower_bound(progress):
    """During unit 1's ingest, retrieval has never happened, so its 4,620 items cost an unknown
    amount. Reporting a confident total there understates a hindsight run by ~20 minutes."""
    _work(progress, documents=29)

    assert progress.is_lower_bound
    assert progress.eta_seconds < (DOCUMENTS - 29) * INGEST_SECONDS + 1


def test_it_stops_being_a_lower_bound_once_both_stages_have_run(progress):
    _work(progress, documents=29, retrievals=1)

    assert not progress.is_lower_bound


def test_the_stages_interleave_without_resetting(progress):
    """A run is ingest->retrieve per unit, ten times over. A bar that filled and emptied ten times
    could not answer "how much longer", which is the question this exists for."""
    _work(progress, documents=29, retrievals=462)  # unit 1
    _work(progress, documents=27)                  # unit 2 begins

    record = progress.as_dict()
    assert record["ingest"]["done"] == 56, "cumulative across units, never restarted"
    assert record["retrieve"]["done"] == 462, "held while the other stage moves"
    assert record["stage"] == "ingest", "and says which one is moving now"


def test_progress_never_exceeds_a_hundred(progress):
    """A slower-than-observed tail must not print 104%."""
    _work(progress, documents=DOCUMENTS, retrievals=RETRIEVALS)

    assert progress.pct == 100
    assert progress.eta_seconds == 0


def test_the_timer_records_elapsed_and_count_together(progress):
    """One write surface, so a rate cannot drift away from the work it describes."""
    with StageTimer(progress, "ingest"):
        pass

    assert progress.as_dict()["ingest"]["done"] == 1
    assert progress.as_dict()["ingest"]["rate_per_second"] is not None


def test_an_unknown_stage_is_refused(progress):
    """Silently accepting it would make a whole stage's work vanish from the projection."""
    with pytest.raises(ValueError, match="unknown stage"):
        progress.record("embedding")


def test_entering_a_stage_makes_it_current_before_its_first_item(progress):
    """A stage becomes current when it STARTS. Between "ingesting unit 1" and the first document
    landing, a reader resolving record[record["stage"]] would otherwise find nothing -- which is
    exactly what `watch` printed: `— u1/1  0/0`."""
    progress.enter_stage("ingest")
    record = progress.as_dict()

    assert record["stage"] == "ingest"
    assert record["ingest"]["total"] == DOCUMENTS, "and its totals are already known"


def test_entering_an_unknown_stage_is_refused(progress):
    with pytest.raises(ValueError, match="unknown stage"):
        progress.enter_stage("embedding")


# --- judged runs ------------------------------------------------------------------------------ #
#: The LoCoMo smoke: 69 of its 152 queries are judgeable, and five sequential LLM calls each.
JUDGEMENTS, JUDGE_SECONDS = 69, 13.0


@pytest.fixture
def judged(clock):
    p = RunProgress(_now=clock)
    p.plan(units=UNITS, documents=DOCUMENTS, retrievals=RETRIEVALS, judgements=JUDGEMENTS)
    return p


def _judge(judged, times=1):
    """Judge some queries, advancing the clock as each one costs its measured time."""
    for _ in range(times):
        judged._now.advance(JUDGE_SECONDS)
        judged.record("judge", seconds=JUDGE_SECONDS)


def test_the_judge_denominator_is_the_judgeable_queries(judged):
    """69, not 152. Only some categories are judge-valid, so counting every query would leave the
    bar stopping at 45% on a run that had judged everything it could."""
    assert judged.as_dict()["judge"]["total"] == JUDGEMENTS


def test_an_unjudged_run_has_no_judge_stage(progress):
    """Zero work, so it neither shows nor blocks the projection -- the default plan is unchanged."""
    assert progress.as_dict()["judge"]["total"] == 0


def test_judging_gets_its_own_rate_and_eta(judged):
    """The phase this exists for: ~15 minutes that used to publish nothing at all."""
    _judge(judged, 4)
    judge = judged.as_dict()["judge"]

    assert judge["done"] == 4
    assert judge["rate_per_second"] == pytest.approx(1 / JUDGE_SECONDS, rel=0.01)
    assert judge["eta_seconds"] == pytest.approx((JUDGEMENTS - 4) * JUDGE_SECONDS, rel=0.01)


def test_a_judged_run_withholds_the_overall_percentage_until_it_judges(judged):
    """Five LLM calls per query cost an amount nothing before them can measure, so the overall
    figure waits -- the same trade as unit 1's ingest, over a wider gap. The stage bars carry the
    run in the meantime, and the ETA says out loud that it is a lower bound."""
    _work(judged, documents=DOCUMENTS, retrievals=RETRIEVALS)

    assert judged.pct is None
    assert judged.is_lower_bound
    assert judged.as_dict()["retrieve"]["done"] == RETRIEVALS, "while the bars stay live"

    _judge(judged)

    assert judged.pct is not None
    assert not judged.is_lower_bound


def test_a_judged_run_still_never_goes_backwards(judged):
    """The property the wait protects, asserted across a whole judged run including its tail."""
    seen = []
    for _ in range(UNITS):
        _work(judged, documents=DOCUMENTS // UNITS, retrievals=RETRIEVALS // UNITS)
        seen.append(judged.pct)
    _work(judged, documents=DOCUMENTS % UNITS, retrievals=RETRIEVALS % UNITS)
    for _ in range(JUDGEMENTS):
        _judge(judged)
        seen.append(judged.pct)
    measured = [p for p in seen if p is not None]

    assert measured == sorted(measured), f"percentage fell: {measured}"
    assert judged.pct == 100


# --------------------------------------------------------------------------------------------- #
# The defect this module was rewritten for: an ETA that ignored concurrency
# --------------------------------------------------------------------------------------------- #

def test_the_rate_is_throughput_not_one_workers_speed(clock):
    """`--workers 5` finishes five documents per 5.16s of wall clock, so the rate is ~5x one
    worker's -- and the ETA is a fifth of what it used to say.

    The old model computed `done / summed_latency`, which is one worker's speed no matter how many
    are running. On the mem0 smoke run that quoted 23 minutes at 5/255 for a run that finished in
    11.5, and there is nothing in a per-worker rate that could ever notice the difference.
    """
    sequential = RunProgress(_now=_Clock())
    sequential.plan(units=UNITS, documents=DOCUMENTS, retrievals=0)
    _work(sequential, documents=20, workers=1)

    concurrent = RunProgress(_now=_Clock())
    concurrent.plan(units=UNITS, documents=DOCUMENTS, retrievals=0)
    _work(concurrent, documents=20, workers=5)

    assert concurrent.as_dict()["ingest"]["rate_per_second"] == pytest.approx(
        5 * sequential.as_dict()["ingest"]["rate_per_second"], rel=0.01)
    assert concurrent.as_dict()["ingest"]["eta_seconds"] == pytest.approx(
        sequential.as_dict()["ingest"]["eta_seconds"] / 5, rel=0.01)


def test_the_percentage_is_unmoved_by_concurrency(clock):
    """`pct` is a ratio of WORK-seconds, so N workers inflate both halves identically. It must not
    inherit the wall clock the rate now uses -- that is why `projected_seconds` prices the remainder
    at the mean item cost rather than adding a wall-clock ETA to a summed one."""
    sequential = RunProgress(_now=_Clock())
    sequential.plan(units=UNITS, documents=DOCUMENTS, retrievals=RETRIEVALS)
    _work(sequential, documents=81, retrievals=924, workers=1)

    concurrent = RunProgress(_now=_Clock())
    concurrent.plan(units=UNITS, documents=DOCUMENTS, retrievals=RETRIEVALS)
    _work(concurrent, documents=81, retrievals=924, workers=5)

    assert concurrent.pct == sequential.pct


def test_the_rate_follows_a_run_that_slows_down(clock):
    """Smoothing at tqdm's 0.3 leans recent. A cumulative mean would still be quoting the fast
    opening minute an hour later, and our per-document latency really does spread p50 11.3s to
    p99 39s."""
    progress = RunProgress(_now=clock)
    progress.plan(units=UNITS, documents=DOCUMENTS, retrievals=0)

    _work(progress, documents=40)                       # at INGEST_SECONDS each
    fast = progress.as_dict()["ingest"]["rate_per_second"]
    for _ in range(40):                                 # then four times slower
        clock.advance(INGEST_SECONDS * 4)
        progress.record("ingest", seconds=INGEST_SECONDS * 4)
    slow = progress.as_dict()["ingest"]["rate_per_second"]

    assert slow < fast / 3, "an EMA should converge toward the new rate, not average it away"


def test_a_stage_is_not_charged_for_the_time_it_waited(clock):
    """Ingest and retrieve alternate per unit. Re-entering a stage restarts its clock, so the
    minutes spent in the other stage do not read as this one running slowly."""
    progress = RunProgress(_now=clock)
    progress.plan(units=UNITS, documents=DOCUMENTS, retrievals=RETRIEVALS)

    _work(progress, documents=10)
    before = progress.as_dict()["ingest"]["rate_per_second"]
    clock.advance(600)                                   # ten minutes of retrieval
    progress.enter_stage("ingest")
    _work(progress, documents=10)

    assert progress.as_dict()["ingest"]["rate_per_second"] == pytest.approx(before, rel=0.2)
