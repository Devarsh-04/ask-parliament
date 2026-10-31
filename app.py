import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from answer import answer
from search import Searcher

EXAMPLES = [
    "How does David Gillespie say gas shortages are hurting Australian manufacturing?",
    "When is the Western Sydney International Airport expected to open to passengers?",
    "What steps is the government taking to improve soft-plastic recycling at supermarkets?",
]

st.set_page_config(page_title="Ask Parliament", page_icon="🏛️", layout="centered")


@st.cache_resource
def load():
    load_dotenv(".env")
    return Searcher(), pd.read_parquet("data/chunks.parquet"), Groq(max_retries=3)


st.title("Ask Parliament")
st.caption("Ask about Australian federal parliament debates, Feb 2024 – Oct 2026. "
           "Answers are written only from Hansard passages, with every sentence cited.")

searcher, chunks, client = load()

example = st.pills("Try one", EXAMPLES)
question = st.text_input("Your question", value=example or "")

if question:
    with st.spinner("Searching Hansard..."):
        passages = chunks.iloc[searcher.search(question, "rerank", k=5)]
        result = answer(client, question, passages)
    if result["found"]:
        st.markdown(" ".join(f"{s['text']} {''.join(f'[{c}]' for c in s['cites'])}"
                             for s in result["sentences"]))
    else:
        st.info("Not found in the debates. The closest passages are below.")
    st.subheader("Sources")
    for i, p in enumerate(passages.itertuples(), start=1):
        heading = f"{p.major}: {p.minor}" if pd.notna(p.minor) else p.major
        with st.expander(f"[{i}] {p.speaker} ({p.party}) · {p.house} · {p.date}"):
            st.markdown(f"**{heading}**")
            st.write(p.text)
            st.markdown(f"[Read in Hansard]({p.url})")

with st.expander("How well does it work?"):
    st.markdown("Search, on 120 test questions (does the right speech come back?):")
    st.dataframe(pd.read_csv("eval/retrieval_summary.csv").round(3), hide_index=True)
    summary = Path("eval/answer_summary.json")
    if summary.exists():
        st.markdown("Answers, graded by a second model:")
        st.json({k: round(v, 3) for k, v in json.loads(summary.read_text()).items()})
