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
"""The command surfaces -- one module per noun the CLI exposes.

Everything here is presentation. A command parses flags, calls into
:mod:`memrank.application` or the domain package that owns the work, and renders the answer
through :mod:`memrank.term`; it does not decide anything a second caller would need to decide the
same way. That split is what lets the MCP server in :mod:`memrank.mcp_server` be a second
serialization of the same operations rather than a fork of them.

`runner` still owns the Typer app itself and registers these; `retired` is the odd one out, a
table of removed commands and flags that fails loudly with the replacement rather than silently
doing nothing.

These modules are also weight-constrained: the CLI must import without dragging in the `api`
extra's stack, and ``tests/repo/test_import_weight.py`` enforces it.
"""
