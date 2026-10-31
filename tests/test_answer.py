import pandas as pd
import pytest

from answer import format_passages, parse_answer
from evaluate import summarise_answers


def test_format_passages_numbers_and_labels():
    rows = pd.DataFrame([{"speaker": "Jane Doe", "party": "Australian Greens", "house": "senate", "date": "2025-03-01",
                          "major": "BILLS", "minor": "Housing Bill", "text": "Rents are rising."}])
    assert format_passages(rows) == (
        "[1] Jane Doe (Australian Greens), senate, 2025-03-01, BILLS: Housing Bill\nRents are rising.")


def test_parse_answer_keeps_valid_cites_only():
    raw = {"found": True, "sentences": [{"text": "A.", "cites": [1, 7]}, {"text": "B.", "cites": []}]}
    assert parse_answer(raw, n_passages=5) == {
        "found": True, "sentences": [{"text": "A.", "cites": [1]}, {"text": "B.", "cites": []}]}


def test_parse_answer_not_found_drops_sentences():
    assert parse_answer({"found": False, "sentences": [{"text": "x", "cites": [1]}]}, 5) == {
        "found": False, "sentences": []}


def test_summarise_answers():
    rows = pd.DataFrame([
        {"kind": "answerable", "found": True, "correct": "yes", "supported": 2, "partial": 0, "unsupported": 0},
        {"kind": "answerable", "found": True, "correct": "partial", "supported": 1, "partial": 1, "unsupported": 1},
        {"kind": "answerable", "found": False, "correct": None, "supported": 0, "partial": 0, "unsupported": 0},
        {"kind": "unanswerable", "found": False, "correct": None, "supported": 0, "partial": 0, "unsupported": 0},
        {"kind": "unanswerable", "found": True, "correct": None, "supported": 0, "partial": 0, "unsupported": 1},
    ])
    s = summarise_answers(rows)
    assert s["answered"] == pytest.approx(2 / 3)
    assert s["correct"] == pytest.approx(1 / 3)
    assert s["correct_or_partial"] == pytest.approx(2 / 3)
    assert s["refused_unanswerable"] == pytest.approx(0.5)
    assert s["sentences_supported"] == pytest.approx(3 / 6)
    assert s["sentences_unsupported"] == pytest.approx(2 / 6)
