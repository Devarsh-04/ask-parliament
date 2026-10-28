import pandas as pd

from search import index_text, rrf


def test_rrf_rewards_agreement():
    assert rrf([[1, 2, 3], [3, 1, 4]], k=60)[:2] == [1, 3]


def test_rrf_includes_items_from_either_list():
    assert set(rrf([[1], [2]])) == {1, 2}


def test_index_text_adds_context():
    row = pd.Series({"major": "BILLS", "minor": "Housing Bill; Second Reading", "speaker": "Jane Doe",
                     "party": "Australian Greens", "text": "Rents are rising."})
    assert index_text(row) == "BILLS: Housing Bill; Second Reading. Jane Doe (Australian Greens): Rents are rising."


def test_index_text_handles_missing_minor_heading():
    row = pd.Series({"major": "QUESTIONS WITHOUT NOTICE", "minor": None, "speaker": "A", "party": "X", "text": "Q?"})
    assert index_text(row) == "QUESTIONS WITHOUT NOTICE. A (X): Q?"
