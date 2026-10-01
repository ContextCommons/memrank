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
"""The evaluation service: owns the tasks, the step order, judging and the results.

One of three roles that do not know about each other (decision 0034). The service hands a
runner one step at a time -- reset, feed, ask -- and enforces the order server-side
(:mod:`memrank.service.machine`), so an agent cannot see a question before its history or
carry state from one case into the next. The runner (:mod:`memrank.loop`) carries each step to
the agent through a connector (:mod:`memrank.connect`). The agent is a black box.

``memrank serve`` runs it locally. The web framework lives in the ``service`` extra, so only
:mod:`memrank.service.app` imports it; the wire shapes in :mod:`memrank.service.protocol` are
plain pydantic and are what the runner imports.
"""
