import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from answer_text import AnswerTextError, get_answer_text  # noqa: E402

ANSWER  = "Legal certainty requires that rules be clear, public, and predictable."
SOURCES = [{"name": "EU AI Act", "type": "national_legislation", "jurisdiction": "EU"}]


def test_plain_string():
    assert get_answer_text(ANSWER) == ANSWER


def test_parsed_dict():
    cell = {
        "raw": json.dumps({"response": ANSWER, "sources": SOURCES}),
        "parsed": {"response": ANSWER, "sources": SOURCES},
    }
    text = get_answer_text(cell)
    assert text == ANSWER
    assert "EU AI Act" not in text


def test_unparsed_raw_json():
    # parse_i1_response failed at collection time (parsed is None): fenced
    # JSON with a literal newline inside the string value
    raw = '```json\n{"response": "Line one.\nLine two.", "sources": [{"name": "GDPR"}]}\n```'
    cell = {"raw": raw, "parsed": None}
    text = get_answer_text(cell)
    assert text == "Line one.\nLine two."
    assert "GDPR" not in text


# v2 records: response text + metadata (see collect_llm_responses.make_record)
META = {"model_version": "gpt-6.1-sol", "finish_reason": "stop",
        "timestamp": "2026-10-20T20:00:00+00:00",
        "answer_tokens": 12, "reasoning_tokens": 40, "input_tokens": 90}


def test_v2_free_text_record():
    # I2: no "parsed" key, text is the answer
    assert get_answer_text({"text": ANSWER, **META}) == ANSWER


def test_v2_json_record_parsed():
    # I1: text is the raw JSON output, parsed holds response + sources
    cell = {"text": json.dumps({"response": ANSWER, "sources": SOURCES}), **META,
            "parsed": {"response": ANSWER, "sources": SOURCES}}
    text = get_answer_text(cell)
    assert text == ANSWER
    assert "EU AI Act" not in text and "gpt-6.1-sol" not in text


def test_v2_json_record_unparsed():
    cell = {"text": '{"response": "Line one.\nLine two.", "sources": []}', **META,
            "parsed": None}
    assert get_answer_text(cell) == "Line one.\nLine two."


@pytest.mark.parametrize("cell", [
    {**dict.fromkeys(META), "text": None},                          # v2 failed I2 call
    {**dict.fromkeys(META), "text": None, "parsed": None},          # v2 failed I1 call
    {"raw": "Sorry, I can't answer that as JSON.", "parsed": None},  # raw not JSON
    {"raw": '{"answer": "wrong key"}', "parsed": None},              # no "response"
    {"raw": None, "parsed": None},                                   # nothing stored
    None,                                                            # wrong type
])
def test_broken_cell_raises(cell):
    with pytest.raises(AnswerTextError):
        get_answer_text(cell)
