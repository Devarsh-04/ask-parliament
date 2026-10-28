from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer
from sentence_transformers import CrossEncoder, SentenceTransformer

INDEX = Path("data/index")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
METHODS = ["bm25", "dense", "hybrid", "rerank"]
STEMMER = Stemmer.Stemmer("english")


def index_text(row: pd.Series) -> str:
    heading = f"{row.major}: {row.minor}" if pd.notna(row.minor) else row.major
    return f"{heading}. {row.speaker} ({row.party}): {row.text}"


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def build(chunks: pd.DataFrame) -> None:
    texts = chunks.apply(index_text, axis=1).tolist()
    INDEX.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": texts}).to_parquet(INDEX / "texts.parquet", index=False)
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER))
    bm25.save(INDEX / "bm25")


class Searcher:
    def __init__(self):
        self.bm25 = bm25s.BM25.load(INDEX / "bm25")
        self.vectors = np.load(INDEX / "dense.npy").astype(np.float32)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.texts = pd.read_parquet(INDEX / "texts.parquet").text.tolist()

    def bm25_search(self, query: str, k: int) -> list[int]:
        docs, _ = self.bm25.retrieve(bm25s.tokenize([query], stopwords="en", stemmer=STEMMER, show_progress=False),
                                     k=k, show_progress=False)
        return docs[0].tolist()

    def dense_search(self, query: str, k: int) -> list[int]:
        q = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        return np.argsort(-(self.vectors @ q))[:k].tolist()

    def hybrid_search(self, query: str, k: int) -> list[int]:
        return rrf([self.bm25_search(query, 100), self.dense_search(query, 100)])[:k]

    def rerank_search(self, query: str, k: int, candidates: int = 30) -> list[int]:
        pool = self.hybrid_search(query, candidates)
        scores = self.reranker.predict([(query, self.texts[i]) for i in pool])
        return [pool[i] for i in np.argsort(-scores)[:k]]

    def search(self, query: str, method: str = "rerank", k: int = 5) -> list[int]:
        return getattr(self, f"{method}_search")(query, k)


if __name__ == "__main__":
    build(pd.read_parquet("data/chunks.parquet"))
