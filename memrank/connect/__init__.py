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
"""Connectors: how the runner reaches an agent without the agent knowing memrank.

An agent is described by a spec file (:mod:`memrank.connect.spec`): a preset, a declarative
HTTP mapping, or a command. Each turns reset, feed and ask into calls the agent already
understands.
"""
