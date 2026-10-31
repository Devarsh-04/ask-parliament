import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from answer import answer, closed_book, complete_json
from search import METHODS, Searcher

EVAL = Path("eval")
CACHE = Path("data/answer_cache.jsonl")
JUDGE_MODEL = "qwen/qwen3.8-27b"

JUDGE_PROMPT = """You are grading an answer about Australian parliament debates.

Question: {question}
Reference answer: {reference}

Each answer sentence is listed with the passages it cites:
{sentences}

For each sentence, decide whether its cited passages support it:
- "supported": every claim in the sentence is stated in the cited passages
- "partial": some claims are supported, others are not
- "unsupported": the cited passages do not back the sentence, or it cites nothing
Then decide whether the whole answer agrees with the reference answer: "yes", "partial" or "no".
If the reference answer is "none", set "correct" to null.

Do not explain. Reply with only this JSON: {{"sentences": ["supported", ...], "correct": "yes"}}"""


def first_hit(speech_ids: list[str], gold: str) -> int | None:
    return next((i + 1 for i, s in enumerate(speech_ids) if s == gold), None)


def summarise(ranks: pd.DataFrame, by: list[str] = ["method"]) -> pd.DataFrame:
    r = ranks.assign(**{f"hit@{k}": ranks["rank"] <= k for k in (1, 5, 10)},
                     mrr=(1 / ranks["rank"]).fillna(0))
    return r.groupby(by, sort=False)[["hit@1", "hit@5", "hit@10", "mrr"]].mean().reset_index()


def summarise_answers(rows: pd.DataFrame) -> dict:
    a, u = rows[rows.kind == "answerable"], rows[rows.kind == "unanswerable"]
    total = rows[["supported", "partial", "unsupported"]].sum()
    return {
        "answered": a.found.mean(),
        "correct": (a.correct == "yes").mean(),
        "correct_or_partial": a.correct.isin(["yes", "partial"]).mean(),
        "refused_unanswerable": (~u.found.astype(bool)).mean(),
        "sentences_supported": total.supported / total.sum(),
        "sentences_partial": total.partial / total.sum(),
        "sentences_unsupported": total.unsupported / total.sum(),
    }


def load_questions() -> pd.DataFrame:
    q = pd.read_csv(EVAL / "questions.csv")
    return q[q.check.fillna("") != "drop"]


def retrieval() -> None:
    questions = load_questions().query("kind == 'answerable'")
    speech_of = pd.read_parquet("data/chunks.parquet", columns=["id"]).id.tolist()
    searcher = Searcher()
    ranks = pd.DataFrame([
        {"qid": q.qid, "method": m, "names_speaker": q.names_speaker,
         "rank": first_hit([speech_of[i] for i in searcher.search(q.question, m, k=10)], q.speech_id)}
        for q in questions.itertuples() for m in METHODS
    ])
    ranks.to_csv(EVAL / "retrieval_ranks.csv", index=False)
    summary = summarise(ranks)
    summary.to_csv(EVAL / "retrieval_summary.csv", index=False)
    print(f"{len(questions)} questions\n")
    print(summary.round(3).to_markdown(index=False))
    print("\nby whether the question names the speaker:\n")
    print(summarise(ranks, ["names_speaker", "method"]).round(3).to_markdown(index=False))


def cached(cache: dict, key: str, fn):
    if key not in cache:
        cache[key] = fn()
        with CACHE.open("a") as f:
            f.write(json.dumps({"key": key, "value": cache[key]}) + "\n")
    return cache[key]


def judge(client: Groq, q, ans: dict, passages: pd.DataFrame) -> dict:
    lines = []
    for i, s in enumerate(ans["sentences"], start=1):
        cited = "\n".join(f"  [{c}] {passages.iloc[c - 1].text}" for c in s["cites"]) or "  (no citation)"
        lines.append(f"Sentence {i}: {s['text']}\n{cited}")
    reference = q.answer if q.kind == "answerable" else "none"
    return complete_json(client, JUDGE_PROMPT.format(question=q.question, reference=reference,
                                                     sentences="\n\n".join(lines)), model=JUDGE_MODEL)


def answers() -> None:
    load_dotenv(".env")
    client = Groq(max_retries=10)
    cache = {r["key"]: r["value"] for r in map(json.loads, CACHE.open())} if CACHE.exists() else {}
    questions = load_questions()
    chunks = pd.read_parquet("data/chunks.parquet")
    searcher = Searcher()
    rows, records = [], []
    for q in questions.itertuples():
        hits = searcher.search(q.question, "rerank", k=5)
        passages = chunks.iloc[hits]
        ans = cached(cache, f"rag:{q.qid}", lambda: answer(client, q.question, passages))
        verdicts = {"sentences": [], "correct": None}
        if ans["found"]:
            verdicts = cached(cache, f"judge:{q.qid}", lambda: judge(client, q, ans, passages))
        labels = [v if s["cites"] else "unsupported" for s, v in zip(ans["sentences"], verdicts["sentences"])]
        labels += ["unsupported"] * (len(ans["sentences"]) - len(labels))
        rows.append({"qid": q.qid, "kind": q.kind, "found": ans["found"], "correct": verdicts.get("correct"),
                     **{k: labels.count(k) for k in ("supported", "partial", "unsupported")}})
        records.append({"qid": q.qid, "kind": q.kind, "question": q.question, "answer": ans,
                        "labels": labels, "correct": verdicts.get("correct"),
                        "chunk_ids": passages.chunk_id.tolist()})
        if q.kind == "unanswerable":
            cb = cached(cache, f"closed:{q.qid}", lambda: closed_book(client, q.question))
            records[-1]["closed_book"] = cb
    rows = pd.DataFrame(rows)
    rows.to_csv(EVAL / "answer_scores.csv", index=False)
    (EVAL / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = summarise_answers(rows)
    closed = [r["closed_book"]["found"] for r in records if "closed_book" in r]
    summary["closed_book_refused_unanswerable"] = 1 - sum(closed) / len(closed)
    (EVAL / "answer_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: round(v, 3) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    {"retrieval": retrieval, "answers": answers}[sys.argv[1] if len(sys.argv) > 1 else "retrieval"]()
