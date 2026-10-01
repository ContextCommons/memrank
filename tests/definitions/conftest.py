"""Evaluation files written into tmp_path, and the fixture evaluations beside the tests."""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "evaluations"


@pytest.fixture
def write(tmp_path) -> Callable[[str, str], Path]:
    """Write ``text`` (dedented) to ``tmp_path/name`` and return the path."""

    def written(text: str, name: str = "eval.yaml") -> Path:
        path = tmp_path / name
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return path

    return written
