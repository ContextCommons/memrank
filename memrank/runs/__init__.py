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
"""A run's life on disk -- where it is, what it did, and how it gets here.

`registry` is the filesystem source of truth: every run lives under the runs root, and this is the
only module that knows how to find one, enumerate its cells, and read metrics back out. `status`
is the heartbeat -- a small `status.json` per run, holding the state machine and enough for another
process to tell a live run from an abandoned one. `progress` is the pure model behind it: stage
rates and an ETA, with no I/O and no opinion about how it is drawn (that is
:mod:`memrank.term.progress`). `checkpoint` snapshots a cell before judging so a crash costs the
judge pass, not the whole run. `record` builds the small durable artifact that gets uploaded, with
the bulk payloads stripped. `artifacts` pulls a cloud run's outputs back down, and `reconcile` is
the one routine that combines the two -- fetch, then restamp the local heartbeat -- shared by every
caller so "catch this run up" means the same thing everywhere.

`registry` has the widest fan-in of anything in memrank. Treat its layout as a contract.
"""
