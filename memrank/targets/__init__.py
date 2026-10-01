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
"""Deprecated: the target catalog behind `memrank submit` and `memrank targets`. Evaluate an
agent with `memrank run` instead; see the README. `resolve.parse_ref` is used by the current
path (`benchmarks/refs`) and does not warn.

Target catalog -- named, reproducible descriptions of what is under test."""

from memrank.targets.catalog import TargetNotFound, list_targets, resolve_target
from memrank.targets.manifest import Manifest, ManifestError

# `factory` is deliberately NOT re-exported here: it imports memrank.adapters, and importing the
# catalog to read a manifest should not drag in every adapter module. Import it as
# `from memrank.targets.factory import build_adapter`.
__all__ = ["Manifest", "ManifestError", "TargetNotFound", "list_targets", "resolve_target"]
