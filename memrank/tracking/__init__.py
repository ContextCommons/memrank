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
"""Deprecated: the MLflow mirror of the `submit` sweep. Evaluate an agent with `memrank run`
instead; see the README.

Experiment tracking -- mirroring finished runs into MLflow.

memrank's own run registry is the source of truth; this package is a one-way mirror for teams who
already read their experiments somewhere else. `export` walks the local registry and writes one
MLflow run per result cell, idempotently, so re-running it converges rather than duplicating.

The command that first pulls cloud runs down from object storage and then hands them to `export`
is :mod:`memrank.ops.mlflow_sync`, on the operator side. It used to live here as `sync`, which put
an internal-only module inside a package the CLI publishes, and that is why it moved. The
direction of the dependency is unchanged -- `ops` knows about `tracking`, never the reverse.

MLflow is an optional dependency, imported lazily. Nothing else in memrank may depend on this
package -- the mirror is allowed to know about the registry, never the reverse.
"""
