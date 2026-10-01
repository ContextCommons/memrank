"""Template substitution: whole-string values keep their type, typos are refused."""

from __future__ import annotations

import pytest

from memrank.connect import template
from memrank.connect.base import ConnectorConfigError
from memrank.service.protocol import Op, Question, Session, Step, Turn

SESSION = Session(id="s1", timestamp="2023-05-08", turns=[
    Turn(role="user", speaker="Ann", text="My car is blue."),
    Turn(role="assistant", speaker="assistant", text="Noted.")])


def test_a_whole_placeholder_keeps_its_type_and_an_embedded_one_becomes_text():
    rendered = template.render({"m": "{messages}", "path": "/s/{case_id}"},
                               {"messages": [{"role": "user"}], "case_id": "c1"})
    assert rendered == {"m": [{"role": "user"}], "path": "/s/c1"}


def test_an_unknown_placeholder_is_refused_with_what_is_available():
    with pytest.raises(ConnectorConfigError, match="questoin.*available: case_id"):
        template.render("{questoin}", {"case_id": "c1"})


def test_a_session_becomes_dated_attributed_chat():
    assert template.session_messages(SESSION) == [
        {"role": "system", "content": "Session date: 2023-05-08"},
        {"role": "user", "content": "Ann: My car is blue."},
        {"role": "assistant", "content": "Noted."}]


def test_the_ask_carries_the_question_date_as_a_system_line():
    step = Step(op=Op.ASK, case_id="c", session_id="k",
                question=Question(id="q", text="What colour?", timestamp="2023-06-01"))
    assert template.ask_vars(step, {})["messages"] == [
        {"role": "system", "content": "Current date: 2023-06-01"},
        {"role": "user", "content": "What colour?"}]
