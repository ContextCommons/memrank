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
"""Where a number came from -- the evidence published beside every result.

A benchmark score is worthless if nobody can say what produced it, so memrank refuses to emit one
without a receipt. `receipt` is that format: the config hash, the versions, the pins. `engine`
records the operator-declared identity of the thing under test as purl/CycloneDX-shaped
provenance, merged with whatever the environment injected. `environment` records where the run
executed -- cloud or laptop, arch, memory, cpu -- deliberately non-identifying and deliberately
*not* part of the config hash, because the same run on a bigger machine is the same run.
`install` reads PEP 610 metadata so `memrank version` reports the commit actually installed
rather than the one someone believes is installed.

The standing rule for this package: never weaken what a receipt records, and never let a secret
value reach one.
"""
