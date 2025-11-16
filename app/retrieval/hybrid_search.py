from typing import List

class HybridSearch:
    def __init__(self, bm25_store, faiss_store, combine_k: int = 20):
        self.bm25 = bm25_store
        self.faiss = faiss_store
        self.combine_k = combine_k

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        bm25_res = self.bm25.query(query, top_k=self.combine_k) if self.bm25 else []
        faiss_res = self.faiss.query(query, top_k=self.combine_k) if self.faiss else []

        seen = set()
        merged = []
        for r in (bm25_res + faiss_res):
            rid = r.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            merged.append(r)
            if len(merged) >= top_k:
                break
        return merged
