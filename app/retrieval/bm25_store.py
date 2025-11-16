import os
import pickle
from typing import List, Dict
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# ensure punkt is available; users must download via README instructions

class BM25Store:
    def __init__(self, index_dir: str = "./indexes/bm25"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.docs: List[Dict] = []
        self.tokenized: List[List[str]] = []
        self.bm25 = None
        self._load()

    def _load(self):
        path = os.path.join(self.index_dir, "bm25.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.docs = data.get("docs", [])
                self.tokenized = data.get("tokenized", [])
                if self.tokenized:
                    self.bm25 = BM25Okapi(self.tokenized)

    def add_documents(self, records: List[Dict]):
        for r in records:
            self.docs.append(r)
            tokens = word_tokenize(r["text"].lower())
            self.tokenized.append(tokens)
        if self.tokenized:
            self.bm25 = BM25Okapi(self.tokenized)

    def save(self):
        path = os.path.join(self.index_dir, "bm25.pkl")
        with open(path, "wb") as f:
            pickle.dump({"docs": self.docs, "tokenized": self.tokenized}, f)

    def query(self, q: str, top_k: int = 10):
        if not self.bm25:
            return []
        tokens = word_tokenize(q.lower())
        scores = self.bm25.get_scores(tokens)
        ranked_ix = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.docs[i] for i in ranked_ix]
