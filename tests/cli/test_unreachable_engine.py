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
"""`memrank submit mem0 locomo --on none` with no mem0 running is the user's fix, not a memrank bug.

It printed "internal error: ConnectError: [Errno 61] Connection refused / this is a bug in
memrank": the manifest cross-check asked mem0 for its configuration before anything else, and the
raw httpx error escaped to the CLI boundary. Driven through ``runner.main`` because the boundary
lives on ``_BoundedTyper.__call__`` (see tests/cli/test_cli_errors.py); only the socket layer is
replaced, so the refusal is deterministic and no port is opened.
"""
from __future__ import annotations

import httpx
import pytest

from memrank.adapters import errors as adapter_errors


class _RefusingSocket(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)


#: (target, evaluation, address variable, where the refusal lands). mem0 is the reported case: the
#: manifest cross-check asks it for /configure. It is withheld from the public projection (ATO-1831),
#: so hindsight -- which ships there, and fails at the preflight probe -- keeps the public suite honest.
_CASES = [
    ("mem0", "locomo", "MEM0_HTTP_URL", "http://localhost:8888/configure"),
    ("hindsight", "demo", "HINDSIGHT_API_URL", "http://localhost:8888/v1/default/banks/"),
]


@pytest.mark.parametrize(("target", "evaluation", "url_env", "tried"), _CASES)
def test_a_refused_engine_is_reported_as_the_users_fix(
        monkeypatch, tmp_path, capsys, target, evaluation, url_env, tried):
    from memrank import runner as runner_module
    from memrank.targets.catalog import list_targets

    if target not in list_targets():
        pytest.skip(f"target {target!r} is withheld from this tree")
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv(url_env, "http://operator:hunter2@localhost:8888")
    monkeypatch.delenv("MEMRANK_DEBUG", raising=False)
    monkeypatch.setattr(adapter_errors.httpx, "HTTPTransport", _RefusingSocket)
    monkeypatch.setattr("sys.argv", ["memrank", "submit", target, evaluation, "--on", "none",
                                     "--output-dir", str(tmp_path / "out")])

    with pytest.raises(SystemExit) as exited:
        runner_module.main()
    printed = capsys.readouterr()
    output = printed.err + printed.out

    assert exited.value.code == 1
    assert f"engine {target!r} is not answering at {tried}" in output
    assert url_env in output and "--on local" in output
    assert "bug in memrank" not in output and "internal error" not in output
    assert "hunter2" not in output and "operator" not in output
