import json
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

PRESIDING = {"SPK", "PRES", "DPRES"}
MODEL = "openai/gpt-oss-120b"
OUT = Path("eval/questions.csv")
CACHE = Path("data/question_cache.jsonl")
N_ANSWERABLE, N_UNANSWERABLE, N_CHECK = 120, 30, 50

QUESTION_PROMPT = """You are building a test set for a search engine over Australian federal parliament debates (Hansard).

Passage from {house} on {date}, debate "{heading}", speaker {speaker} ({party}):
\"\"\"{text}\"\"\"

Write one question a journalist or curious voter might type that this passage answers well.
- It must be answerable from this passage alone, and specific enough that this passage is a strong answer.
- Paraphrase: do not copy distinctive phrases from the passage.
- Under 25 words.
- {speaker_rule}
Also give a 1-2 sentence answer drawn only from the passage.
If the passage is purely procedural (moving motions, thanking members, points of order) return {{"question": null}}.
Return JSON: {{"question": "...", "answer": "..."}}"""

UNANSWERABLE_PROMPT = """You are building a test set for a search engine over Australian federal parliament debates from February 2024 to October 2026.
Write {n} questions a user might type that these debates CANNOT answer. Mix these kinds:
- invented bills, programs or inquiries with specific, plausible-sounding names
- events after October 2026
- matters handled only by state parliaments or local councils
- personal trivia about MPs that would never come up in debate
Make them sound like real search questions, under 25 words each.
Return JSON: {{"questions": ["...", "..."]}}"""


def pick_chunks(chunks: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    ok = chunks[~chunks.party.isin(PRESIDING) & (chunks.text.str.split().str.len() >= 120)]
    return ok.sample(min(n, len(ok)), random_state=seed)


def parse_json(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return json.loads(text)


def ask(client: Groq, prompt: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, max_completion_tokens=2000, temperature=0.7, reasoning_effort="low",
    )
    return parse_json(r.choices[0].message.content)


def question_for(client: Groq, row: pd.Series, names_speaker: bool) -> dict:
    heading = f"{row.major}: {row.minor}" if pd.notna(row.minor) else row.major
    rule = (f"Refer to {row.speaker} by name." if names_speaker
            else "Do not name the speaker; ask about the topic or claim.")
    q = ask(client, QUESTION_PROMPT.format(house=row.house, date=row.date, heading=heading, speaker=row.speaker,
                                           party=row.party, text=row.text, speaker_rule=rule))
    return {"kind": "answerable", "question": q.get("question"), "answer": q.get("answer"),
            "names_speaker": names_speaker, "speech_id": row.id, "chunk_id": row.chunk_id, "speaker": row.speaker,
            "party": row.party, "date": row.date, "heading": heading, "passage": row.text}


def main() -> None:
    load_dotenv()
    client = Groq(max_retries=10)
    picked = pick_chunks(pd.read_parquet("data/chunks.parquet"), N_ANSWERABLE + 30)
    rng = random.Random(0)
    flags = [rng.random() < 0.5 for _ in range(len(picked))]
    done = {r["chunk_id"]: r for r in map(json.loads, CACHE.open())} if CACHE.exists() else {}
    with CACHE.open("a") as cache:
        for (_, row), flag in zip(picked.iterrows(), flags):
            if row.chunk_id not in done:
                done[row.chunk_id] = question_for(client, row, flag)
                cache.write(json.dumps(done[row.chunk_id]) + "\n")
                cache.flush()
    rows = [done[c] for c in picked.chunk_id]
    answerable = [r for r in rows if r["question"]][:N_ANSWERABLE]
    unanswerable = [{"kind": "unanswerable", "question": q}
                    for q in ask(client, UNANSWERABLE_PROMPT.format(n=N_UNANSWERABLE))["questions"]]
    df = pd.DataFrame(answerable + unanswerable)
    df.insert(0, "qid", [f"q{i:03d}" for i in range(len(df))])
    check = set(df[df.kind == "answerable"].sample(40, random_state=0).index) | set(
        df[df.kind == "unanswerable"].sample(10, random_state=0).index)
    df.insert(1, "check", ["?" if i in check else "" for i in df.index])
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.kind.value_counts().to_dict(), f"{len(rows) - len([r for r in rows if r['question']])} procedural skipped")


if __name__ == "__main__":
    main()
