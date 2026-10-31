import pandas as pd
import pytest

from evalset import parse_json, pick_chunks
from evaluate import first_hit, summarise


def test_pick_chunks_skips_short_and_presiding():
    chunks = pd.DataFrame({
        "chunk_id": ["a", "b", "c", "d"],
        "party": ["Australian Greens", "SPK", "Liberal Party", "Independent"],
        "text": [" ".join(["w"] * 150), " ".join(["w"] * 150), "too short", " ".join(["w"] * 200)],
    })
    picked = pick_chunks(chunks, n=5, seed=1)
    assert sorted(picked.chunk_id) == ["a", "d"]


def test_parse_json_strips_code_fence():
    assert parse_json('```json\n{"question": "Q?", "answer": "A."}\n```') == {"question": "Q?", "answer": "A."}


def test_first_hit_is_rank_of_gold_speech():
    assert first_hit(["s9", "s1", "s1", "s2"], "s1") == 2
    assert first_hit(["s9"], "s1") is None


def test_summarise_hit_rates_and_mrr():
    ranks = pd.DataFrame({"method": ["bm25"] * 4, "rank": [1, 3, None, 8]})
    row = summarise(ranks).iloc[0]
    assert row["hit@1"] == pytest.approx(0.25)
    assert row["hit@5"] == pytest.approx(0.5)
    assert row["hit@10"] == pytest.approx(0.75)
    assert row["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 8) / 4)
