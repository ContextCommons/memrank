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
"""Reference agents memrank ships, run locally, as baselines to compare an agent against."""

#: How a reference agent reports a failed model call (``detail`` of its 502): the provider it
#: called, the key it called it with, and whether the provider answered at all. The runner reads
#: this to say whose side the failure is on (:mod:`memrank.loop.explain`), so it lives here,
#: importable without the web framework the agents themselves run on.
PROVIDER, KEY_NAME = "anthropic", "ANTHROPIC_API_KEY"
PROVIDER_REFUSED, PROVIDER_FAILED = "provider_refused", "provider_failed"
