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
"""The current agent path says nothing about deprecation.

The pre-agent surface warns where a person enters it (`memrank.deprecation`), and `import
memrank` still loads it, so a stray warning would reach every `memrank run`. A whole judged run
through the loop, the connector and the service is the witness.
"""

from __future__ import annotations

import warnings

from memrank.loop.run import run_agent
from tests.loop.test_run import LOCOMO, full_context, org, quiet  # noqa: F401 - org is a fixture


def test_a_judged_agent_run_emits_no_deprecation(engine, tmp_path, org, capsys):  # noqa: F811
    echoed: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), quiet(echoed))
    captured = capsys.readouterr()
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)
            and "memrank" in str(w.filename)] == []
    assert not any("deprecated" in line for line in echoed)
    assert "deprecated" not in captured.out + captured.err
