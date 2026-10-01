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
"""Scoring arithmetic -- everything that turns a run into a number without asking a model.

`headline` owns the single rule for which number represents a cell, so the CLI, the API and the
leaderboard can never disagree about what a run scored. The rest are the measurements it chooses
between: `scoring` decides whether retrieval actually surfaced the gold evidence (by content, not
by engine-assigned id, and with no "non-empty means hit" fallback), `evidence_recall` and
`retrieval` implement the benchmarks' own published judge-free metrics, and `ordering` scores an
aligned sequence against a reference. `cost` prices a query's context in tokens.

Everything here is pure and deterministic. Judgement -- anything that needs a model's opinion --
lives in :mod:`memrank.judging` instead.
"""

from memrank.metrics.scoring import (
    EvidenceSpec,
    MatchResult,
    SpanRecall,
    normalize,
    score_query,
    spec_from_query,
)

#: Re-exported here because this is where a person looks for them. They were reachable only as
#: `memrank.metrics.scoring.score_query` and were not in any `__all__`, so the deterministic
#: scorer memrank ships had no public name and "my questions, memrank's scorer" could not be
#: written at all (ATO-2135). `SpanRecall` is that scorer as one object; the four names below
#: it are the pieces, for a scorer that needs the per-query verdict rather than the unit mean.
__all__ = [
    "EvidenceSpec",
    "MatchResult",
    "SpanRecall",
    "normalize",
    "score_query",
    "spec_from_query",
]
