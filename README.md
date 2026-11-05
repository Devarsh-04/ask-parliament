# Ask Parliament

Ask questions about Australian federal parliament debates (Hansard, February 2024 to October 2026) and get short answers where every sentence cites the speech it came from. If the debates don't cover it, the app says so.

**Try it: [btn3a7fez2hnpm4mgnqclg.streamlit.app](https://btn3a7fez2hnpm4mgnqclg.streamlit.app/)**

The project also measures how well this works, because a confident wrong answer about what a politician said is worse than no answer.

## How it works

1. **Ingest** (`ingest.py`): parses OpenAustralia's Hansard XML into 134,710 passages of up to 300 words, each with speaker, party, date and a link to the source.
2. **Search** (`search.py`): combines BM25 and `bge-small-en-v1.5` embeddings with reciprocal rank fusion, then reranks the top 30 with a cross-encoder.
3. **Answer** (`answer.py`): `gpt-oss-120b` on Groq writes 1-4 sentences from the top 5 passages only, citing each sentence, or refuses.
4. **Evaluate** (`evalset.py`, `evaluate.py`): tests search and answers on 149 questions. Results are in `eval/`.

## Results

**Search**, on 120 answerable questions: does the source speech come back?

| Method | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| BM25 | .542 | .800 | .892 | .645 |
| Dense | .367 | .617 | .717 | .479 |
| Hybrid | .475 | .750 | .808 | .592 |
| **Hybrid + rerank** | **.617** | **.825** | **.892** | **.706** |

Dense search did worse than plain BM25 and pulled hybrid below it. Reranking is what beat BM25, by a small margin at hit@5.

**Answers**

| Measure | Result |
|---|---|
| Fully correct (120 answerable) | 80.0% |
| Correct or partly correct | 87.5% |
| Fully correct when the source speech was in the top 5 (99) | 89.9% |
| Fully correct when it was not (21) | 33.3% |
| Answer sentences fully supported by their citations | 91.2% |
| Answer sentences with no support | 1.1% |
| Refused unanswerable questions (29), using passages | 96.6% |
| Refused unanswerable questions (29), no passages | 79.3% |

Most wrong answers trace back to search, so that is where to improve next.

## Limits of these numbers

- **Generated test set.** A model wrote the questions from individual passages, so they may suit this kind of search better than real questions would.
- **Not human-checked.** Claude, not a person, reviewed the questions. A second model (`qwen3.8-27b`) graded the answers.
- **Small unanswerable set.** One question moves the refusal rate by 3.4 points, and about two "unanswerable" questions may actually be answerable.
- **Mixed prompt versions.** All answers come from `gpt-oss-120b`, but only 36 (including all 29 unanswerable questions) use the final prompt. The other 113 predate the rule on relative dates. Running 12 of them with both prompts changed no decision to answer or refuse, but those answers were not regraded.
- **Relative dates.** Answers quote phrases like "next Sunday" with the speech date instead of calculating a date, because the model got the arithmetic wrong in testing.

## Run it

Needs Python 3.13 and a [Groq API key](https://console.groq.com) in `.env` as `GROQ_API_KEY`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

The search index is too large for git and is kept in a private Hugging Face dataset, so running the app needs your own copy in `data/`:

1. Put OpenAustralia's debate XML in `data/raw/representatives_debates` and `data/raw/senate_debates`, and its member lists in `data/representatives.xml` and `data/senators.xml`.
2. Run `.venv/bin/pip install lxml`, then `.venv/bin/python ingest.py` and `.venv/bin/python search.py`.
3. Run `embed_colab.ipynb` on a GPU and put the resulting `dense.npy` in `data/index/`.

Tests: `.venv/bin/pip install pytest lxml`, then `.venv/bin/pytest`.

## Data and licence

Hansard is © Commonwealth of Australia, provided under [CC BY-NC-ND 3.0 AU](https://creativecommons.org/licenses/by-nc-nd/3.0/au/) via [OpenAustralia](https://www.openaustralia.org.au/api/). This is a non-commercial project. AI-written answers can be wrong and are not an official record, so check the linked Hansard source.
