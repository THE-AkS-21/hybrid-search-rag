import os
import pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class FaissStore:
    def __init__(self, index_dir: str = "./indexes/faiss", model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.emb_model = SentenceTransformer(model_name)
        self.docs: List[Dict] = []
        self.index = None
        self._load()

    def _load(self):
        meta_path = os.path.join(self.index_dir, "meta.pkl")
        index_path = os.path.join(self.index_dir, "faiss.index")
        if os.path.exists(meta_path) and os.path.exists(index_path):
            with open(meta_path, "rb") as f:
                self.docs = pickle.load(f)
            self.index = faiss.read_index(index_path)

    def add_documents(self, records: List[Dict]):
        texts = [r["text"] for r in records]
        embs = self.emb_model.encode(texts, convert_to_numpy=True)
        # normalize embeddings for inner-product cosine similarity
        faiss.normalize_L2(embs)
        if self.index is None:
            dim = embs.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        self.docs.extend(records)

    def save(self):
        meta_path = os.path.join(self.index_dir, "meta.pkl")
        index_path = os.path.join(self.index_dir, "faiss.index")
        with open(meta_path, "wb") as f:
            pickle.dump(self.docs, f)
        if self.index is not None:
            faiss.write_index(self.index, index_path)

    def query(self, q: str, top_k: int = 10):
        if self.index is None:
            return []
        qb_emb = self.emb_model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(qb_emb)
        D, I = self.index.search(qb_emb, top_k)
        results = []
        for pos in I[0]:
            if pos < len(self.docs):
                results.append(self.docs[pos])
        return results
