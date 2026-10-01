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
"""supermemory's provider key must be present at boot and never actually called.

memrank supplies a PLACEHOLDER for that key (`supermemory.compose.yaml`) rather than a real
credential, and drops supermemory's entry from `requirements.py` entirely -- so a sweep needs one
provider account instead of two. Both rest on a claim about someone else's binary:

    the server refuses to start without a provider key, and then never authenticates with it

That was established by booting the image with `sk-ant-invalid-...` and watching ingest and retrieve
return the correct document. It can stop being true without warning -- supermemory ships new
versions, and an LLM-backed retrieval path would make the placeholder a 401 in the middle of a paid
run. This test is what turns that into a caught regression.

Run against a server started with a deliberately invalid key:

    docker run --rm -p 6767:6767 -e ANTHROPIC_API_KEY=sk-ant-invalid-probe <image>
    SUPERMEMORY_BASE_URL=http://127.0.0.1:6767 uv run pytest tests/live/conformance/test_supermemory_llm_unused.py

Skipped when no server is reachable, like every other conformance test here.
"""
from __future__ import annotations

import pytest

from memrank.core import Document
from tests import withheld
from tests.live.conformance.test_adapter_contract import _backend_reachable, _backend_url

#: What the placeholder in supermemory.compose.yaml is worth: nothing. A run that authenticates
#: with this string is a run whose engine started calling an LLM.
INVALID_KEY_MARKER = "unused-by-supermemory"


@pytest.fixture
def adapter():
    from memrank.adapters.supermemory import Supermemory

    url = _backend_url("supermemory")
    if not _backend_reachable(url):
        pytest.skip(f"Supermemory not reachable at {url!r}")
    built = Supermemory()
    built.prepare("llm-unused-probe")
    return built


def test_ingest_and_retrieve_work_without_a_usable_provider_key(adapter):
    """The claim the placeholder rests on, stated as an assertion.

    If this fails against a server holding an invalid key, supermemory has begun calling its LLM
    and three things become wrong at once: the placeholder, the dropped `requirements.py` entry,
    and every receipt that reports no LLM component.
    """
    adapter.ingest([Document(id="d1", user_id="llm-unused-probe",
                             content="Alex adopted a tabby cat named Miso in March 2024.")])

    hits = adapter.retrieve("What pet does Alex have?", 3, "llm-unused-probe")

    assert hits, "ingest+retrieve must work with no usable provider credential"
    assert any("Miso" in str(getattr(hit, "content", hit)) for hit in hits), (
        "the stored document must come back -- a degraded answer would mean the engine fell back "
        "to something that DID need the key")


def test_the_graph_supplies_a_placeholder_rather_than_a_credential():
    """Pinned here rather than in a placement test because it is the same claim: the value is
    inert. A real key appearing in this file would be a credential handed to a process that has
    no use for it, and would pass every other test in the suite."""
    from memrank.placement.local import render_compose
    from memrank.targets import resolve_target

    withheld.require("supermemory")
    service = render_compose(resolve_target("supermemory"), project="p")["services"]["engine"]

    assert service["environment"]["ANTHROPIC_API_KEY"] == INVALID_KEY_MARKER


def test_supermemory_demands_no_credential_from_an_org():
    """The user-visible point of the change: one provider account covers every target."""
    from memrank.secrets import requirements

    assert requirements.required_secrets("supermemory") == []
