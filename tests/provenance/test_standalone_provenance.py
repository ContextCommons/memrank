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
"""The standalone build names itself by its own record, and never sends its user to pip.

The standalone build carries its own interpreter and was installed by a script. Telling its user
to ``pip install`` anything would install into a Python the build never uses, so every instruction
here comes from the record the build carries: the commit it was frozen from, and its origin.
"""
from __future__ import annotations

import json
import sys

import pytest

from memrank.provenance import install

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_ORIGIN = "https://get.example.test"


@pytest.fixture
def standalone(tmp_path, monkeypatch):
    """Present this process as the standalone build, carrying its record."""
    (tmp_path / install.STANDALONE_RECORD).write_text(json.dumps(
        {"commit": _COMMIT, "target": "macos-arm64", "origin": _ORIGIN, "version": "9.9.9"}))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    return tmp_path


@pytest.mark.parametrize(("ask", "expected"), [
    (install.describe_origin, "standalone macos-arm64, commit 0123456"),
    (install.build_commit, _COMMIT),
    (install.upgrade_instruction, f"curl -fsSL {_ORIGIN}/install.sh | sh"),
    (lambda: install.dependency_instruction("anthropic"), f"curl -fsSL {_ORIGIN}/install.sh | sh"),
])
def test_the_standalone_build_answers_from_its_own_record(standalone, ask, expected):
    assert ask() == expected


def test_an_extra_it_does_not_carry_is_named_as_a_python_install(standalone):
    instruction = install.dependency_instruction("mlflow", "mlflow")

    assert instruction.startswith("a Python install of memrank, pip install 'memrank[mlflow]'")
    assert instruction.endswith("this standalone build does not carry it")


def test_a_standalone_build_without_its_record_fails_loudly(standalone):
    (standalone / install.STANDALONE_RECORD).unlink()

    with pytest.raises(FileNotFoundError):
        install.version_line()


def test_a_python_install_has_no_standalone_record():
    assert install.standalone_build() is None


def test_the_standalone_build_ships_the_api_its_record_names(standalone, monkeypatch):
    """A build for staging talks to the staging API with nothing configured (ATO-2360)."""
    from memrank import settings

    record = standalone / install.STANDALONE_RECORD
    record.write_text(json.dumps({**json.loads(record.read_text()),
                                  "api_url": "https://api-staging.example.test"}))
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(standalone / "config"))
    monkeypatch.delenv("MEMRANK_API_URL", raising=False)

    assert settings.resolve("api.url") == ("https://api-staging.example.test", settings.DEFAULT)
    monkeypatch.setenv("MEMRANK_API_URL", "https://elsewhere.example.test")
    assert settings.resolve("api.url") == ("https://elsewhere.example.test", settings.ENV)
