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
"""Deprecated: the cloud and sweep orchestration behind `memrank submit`. Evaluate an agent with
`memrank run` instead; see the README. `sweep._mirror_to_mlflow` is reached lazily by
`runs/reconcile` and does not warn.

The run lifecycle: everything a recorded run wires AROUND the eval library.

What belongs here: the observers that bridge `memrank.evaluation`'s narration into the
terminal and the run heartbeat, and (as the runner split proceeds) target resolution,
sweep execution, persistence, sync and cloud submission. What does not: the measurement
loop itself (`memrank.evaluation`) and Typer -- refusals raise `MemrankError` subclasses
and the CLI translates them at its boundary.
"""
