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
"""Credentials -- where they live, and which ones a run needs before it starts.

`wallet` is the local store: a plaintext `0600` file beside the user's config, holding the keys
this machine knows. `requirements` declares, per engine, which credentials a launch actually
needs -- derived from the engine's providers and independent of where it runs, so the same
declaration gates a laptop run and a cloud task. `names` answers one question ("is this field
name a secret?") in one place, so redaction in receipts and refusal in manifests can never
disagree about what counts.

Resolution order and org-vault lookup live in :mod:`memrank.config`; this package is only about
storage and requirements.
"""
