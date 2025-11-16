from typing import List
import requests
from app.core.config import settings

class CohereReranker:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.COHERE_API_KEY
        if self.api_key is None:
            raise RuntimeError("Cohere API key not set in environment")
        self.base = "https://api.cohere.ai/rerank"

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        if not candidates:
            return []
        texts = [c.get("text", "") for c in candidates]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"query": query, "texts": texts}
        try:
            resp = requests.post(self.base, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
        except Exception:
            # fallback to original ordering
            return candidates[:top_k]
        res_json = resp.json()
        results = res_json.get("results", [])
        # sort by score desc
        ordered = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        order = [r.get("index") for r in ordered]
        reranked = [candidates[i] for i in order if i < len(candidates)]
        return reranked[:top_k]
