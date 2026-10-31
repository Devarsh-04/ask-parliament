import json
import os

import pandas as pd
from groq import Groq

MODEL = os.environ.get("ANSWER_MODEL", "openai/gpt-oss-120b")

RAG_PROMPT = """You answer questions about Australian federal parliament debates (Hansard), using ONLY the numbered passages below.

Rules:
- Every sentence must cite the passage numbers that support it, e.g. [1] or [2, 3].
- Say who said what (name and party). Do not add facts, numbers or opinions that are not in the passages.
- Each passage header gives the date of the speech. If a passage uses a relative date ("next Sunday", "last year", "tomorrow"), do not work out a calendar date. Give the speech date and quote the phrase, e.g. On 3 May 2025, Jane Doe said it would open "next Sunday".
- If the passages do not answer the question, set "found" to false and leave "sentences" empty.
- Keep it to 1-4 sentences.

Question: {question}

Passages:
{passages}

Return JSON: {{"found": true, "sentences": [{{"text": "...", "cites": [1]}}]}}"""

CLOSED_BOOK_PROMPT = """Answer this question about Australian federal parliament from your own knowledge, in 1-4 sentences.
If you do not know, or it did not happen, set "found" to false.

Question: {question}

Return JSON: {{"found": true, "answer": "..."}}"""


def format_passages(rows: pd.DataFrame) -> str:
    return "\n\n".join(
        f"[{i}] {r.speaker} ({r.party}), {r.house}, {r.date}, "
        f"{f'{r.major}: {r.minor}' if pd.notna(r.minor) else r.major}\n{r.text}"
        for i, r in enumerate(rows.itertuples(), start=1))


def parse_answer(raw: dict, n_passages: int) -> dict:
    if not raw.get("found"):
        return {"found": False, "sentences": []}
    return {"found": True, "sentences": [
        {"text": s["text"], "cites": [c for c in s.get("cites", []) if isinstance(c, int) and 1 <= c <= n_passages]}
        for s in raw.get("sentences", [])]}


def complete_json(client: Groq, prompt: str, model: str = MODEL) -> dict:
    extra = ({"reasoning_effort": "low", "max_completion_tokens": 2000} if "gpt-oss" in model
             else {"reasoning_effort": "none", "max_completion_tokens": 900})
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"},
        temperature=0, **extra)
    return json.loads(r.choices[0].message.content)


def answer(client: Groq, question: str, rows: pd.DataFrame) -> dict:
    raw = complete_json(client, RAG_PROMPT.format(question=question, passages=format_passages(rows)))
    return parse_answer(raw, len(rows))


def closed_book(client: Groq, question: str) -> dict:
    raw = complete_json(client, CLOSED_BOOK_PROMPT.format(question=question))
    return {"found": bool(raw.get("found")), "answer": raw.get("answer")}
