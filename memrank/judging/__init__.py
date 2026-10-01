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
"""Judgement -- the parts of scoring that need a model's opinion.

`judge` is the scorer, and it is deliberately pure: it takes a `Completer` -- any callable from
(model, system, user) to text -- so the grading logic can be tested without a network and swapped
without touching methodology. `client` is the only thing that actually talks to a provider, which
makes it the single place caching, 429 back-off and the billable-call counter can live without
being duplicated. `prompts` holds the prompt text itself, checked in and versioned, because a
prompt is methodology: change it and every number before the change means something different.
`shape` declares how a given benchmark grades -- a binary verdict, a per-nugget rubric, an event
ordering -- and what one query costs, so the runner never grows a branch on benchmark name.

`judge` and `shape` reference each other. The cycle is broken on purpose: `judge` imports `shape`
only under `TYPE_CHECKING` plus one deferred call-site import. Keep it that way.

Judge-free arithmetic -- anything deterministic -- belongs in :mod:`memrank.metrics`.
"""
