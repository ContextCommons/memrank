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
#
# Used by the current path (loop/judge): it sits in a deprecated package but is not
# deprecated itself, and it must not warn.
"""Shared evaluation defaults.

Kept free of ``memrank`` imports so any layer -- library, CLI, API config -- can
take a default from here without pulling the evaluation stack in behind it.
"""
from __future__ import annotations

#: Queries graded concurrently by the judge stage, everywhere a judge default is needed.
#: 4 because the bound is ONE provider account's rate limit, not the machine: the calls are
#: independent and cheap to have in flight, and a single sequential grade wastes most of the
#: budget. It is a throughput knob only -- classify and reduce are sequential and ordered, so
#: metrics are bit-identical at any worker count.
DEFAULT_JUDGE_WORKERS = 4
