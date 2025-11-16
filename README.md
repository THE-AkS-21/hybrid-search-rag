# Multi-Document Hybrid Search RAG System — Project Scaffold

This document contains a full-from-scratch project scaffold, all source files, and a complete `README.md` suitable for GitHub for the **Multi-Document Hybrid Search RAG System** (Hybrid BM25 + FAISS search, Cohere re-ranker, LangChain agent, FastAPI microservice, Docker-ready).

---

## Repository layout

```
hybrid-search-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI server entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py            # HTTP endpoints (ingest, query, health)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # configuration loader (.env)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py           # PDF reading & text extraction
│   │   ├── text_splitter.py        # chunking logic
│   │   ├── indexer.py              # build BM25 + FAISS per-document
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_store.py           # BM25 wrapper
│   │   ├── faiss_store.py          # FAISS wrapper
│   │   ├── hybrid_search.py        # combine BM25 + FAISS results
│   │   ├── reranker.py             # Cohere re-ranker integration
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── rag_agent.py            # LangChain agent + custom tools
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── serializers.py          # helpers to serialize docs for indexing
│   │   ├── logging_cfg.py
│   └── models/
│       └── types.py
├── scripts/
│   ├── ingest_sample.sh            # script to ingest a PDF into local indexes
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
└── LICENSE
```

---

> All code files are included below. Copy them into the matching path in your repository.

---

# app/core/config.py

```python
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None
    COHERE_API_KEY: str | None
    FAISS_INDEX_DIR: str = "./indexes/faiss"
    BM25_INDEX_DIR: str = "./indexes/bm25"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

# app/utils/logging_cfg.py

```python
import logging

def configure_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )
```

---

# app/ingestion/pdf_loader.py

```python
from typing import List
from pathlib import Path
import io
from pdfminer.high_level import extract_text


def load_pdf_text(path: str) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    # Using pdfminer to extract text reliably
    text = extract_text(str(path))
    return text


def load_many_pdfs(paths: List[str]) -> List[dict]:
    """Return list of dicts: {"path": str, "text": str}
    """
    results = []
    for p in paths:
        text = load_pdf_text(p)
        results.append({"path": p, "text": text})
    return results
```

---

# app/ingestion/text_splitter.py

```python
from typing import List


def simple_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks (naive sentence-agnostic approach).
    Prefer replacing this with a sentence-aware splitter if needed.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start = end - overlap
    return chunks
```

---

# app/ingestion/indexer.py

```python
import os
from pathlib import Path
from typing import List

from ..utils.serializers import doc_to_record
from ..ingestion.text_splitter import simple_chunk_text
from ..retrieval.faiss_store import FaissStore
from ..retrieval.bm25_store import BM25Store
from ..core.config import settings


def index_document(path: str):
    text = open(path, "r", encoding="utf-8").read() if path.endswith(".txt") else None
    # In most flows, ingestion will call pdf_loader first; here we assume caller passes text
    raise RuntimeError("Use index_documents_from_texts for this simplified scaffold")


def index_documents_from_texts(docs: List[dict]):
    """docs: list of {"id": str, "text": str, "meta": dict}

    This function creates/upserts BM25 and FAISS indexes for all provided docs.
    """
    # ensure index dirs
    os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
    os.makedirs(settings.BM25_INDEX_DIR, exist_ok=True)

    faiss = FaissStore(index_dir=settings.FAISS_INDEX_DIR)
    bm25 = BM25Store(index_dir=settings.BM25_INDEX_DIR)

    for doc in docs:
        doc_id = doc.get("id")
        text = doc.get("text", "")
        meta = doc.get("meta", {})
        chunks = simple_chunk_text(text, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
        records = [doc_to_record(doc_id=doc_id, idx=i, text=c, meta=meta) for i, c in enumerate(chunks)]
        faiss.add_documents(records)
        bm25.add_documents(records)

    # persist stores
    faiss.save()
    bm25.save()

    return {"faiss_index": settings.FAISS_INDEX_DIR, "bm25_index": settings.BM25_INDEX_DIR}
```

---

# app/utils/serializers.py

```python
from typing import Dict


def doc_to_record(doc_id: str, idx: int, text: str, meta: Dict) -> Dict:
    """Create a uniform record stored in indexes.
    Returns: {"id": "{doc_id}::{idx}", "text": ..., "meta": {...}}
    """
    return {"id": f"{doc_id}::{idx}", "text": text, "meta": meta}
```

---

# app/retrieval/bm25_store.py

```python
import os
import pickle
from typing import List, Dict
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


class BM25Store:
    def __init__(self, index_dir: str = "./indexes/bm25"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.docs = []  # list of records
        self.tokenized = []
        self.bm25 = None
        self._load()

    def _load(self):
        path = os.path.join(self.index_dir, "bm25.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.docs = data["docs"]
                self.tokenized = data["tokenized"]
                self.bm25 = BM25Okapi(self.tokenized)

    def add_documents(self, records: List[Dict]):
        for r in records:
            self.docs.append(r)
            tokens = word_tokenize(r["text"].lower())
            self.tokenized.append(tokens)
        # build/refresh
        self.bm25 = BM25Okapi(self.tokenized)

    def save(self):
        path = os.path.join(self.index_dir, "bm25.pkl")
        with open(path, "wb") as f:
            pickle.dump({"docs": self.docs, "tokenized": self.tokenized}, f)

    def query(self, q: str, top_k: int = 10):
        if self.bm25 is None:
            return []
        tokens = word_tokenize(q.lower())
        scores = self.bm25.get_scores(tokens)
        ranked_ix = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.docs[i] for i in ranked_ix]
```

---

# app/retrieval/faiss_store.py

```python
import os
import pickle
from typing import List, Dict
import numpy as np

# use sentence-transformers for embeddings (or OpenAI embeddings).
from sentence_transformers import SentenceTransformer
import faiss


class FaissStore:
    def __init__(self, index_dir: str = "./indexes/faiss", model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.emb_model = SentenceTransformer(model_name)
        self.docs: List[Dict] = []
        self.index: faiss.IndexFlatIP | None = None
        self.id_to_pos = {}
        self._load()

    def _load(self):
        meta_path = os.path.join(self.index_dir, "meta.pkl")
        index_path = os.path.join(self.index_dir, "faiss.index")
        if os.path.exists(meta_path) and os.path.exists(index_path):
            with open(meta_path, "rb") as f:
                self.docs = pickle.load(f)
                self.id_to_pos = {d["id"]: i for i, d in enumerate(self.docs)}
            self.index = faiss.read_index(index_path)

    def add_documents(self, records: List[Dict]):
        texts = [r["text"] for r in records]
        embs = self.emb_model.encode(texts, convert_to_numpy=True)
        if self.index is None:
            dim = embs.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        # normalize for cosine-sim via inner product
        faiss.normalize_L2(embs)
        start_pos = len(self.docs)
        self.docs.extend(records)
        for i, r in enumerate(records, start=start_pos):
            self.id_to_pos[r["id"]] = i
        self.index.add(embs)

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
```

---

# app/retrieval/hybrid_search.py

```python
from typing import List

class HybridSearch:
    def __init__(self, bm25_store, faiss_store, combine_k: int = 20):
        self.bm25 = bm25_store
        self.faiss = faiss_store
        self.combine_k = combine_k

    def retrieve(self, query: str, top_k: int = 5):
        # fetch larger candidate set from both
        bm25_res = self.bm25.query(query, top_k=self.combine_k) if self.bm25 else []
        faiss_res = self.faiss.query(query, top_k=self.combine_k) if self.faiss else []

        # naive merge: keep order but dedupe by id
        seen = set()
        merged = []
        for r in (bm25_res + faiss_res):
            rid = r["id"]
            if rid in seen:
                continue
            seen.add(rid)
            merged.append(r)
            if len(merged) >= top_k:
                break
        return merged
```

---

# app/retrieval/reranker.py

```python
from typing import List
import os
import requests

from ..core.config import settings


class CohereReranker:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.COHERE_API_KEY
        if self.api_key is None:
            raise RuntimeError("Cohere API key not set in environment")
        self.base = "https://api.cohere.ai/rerank"

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        # cohere rerank expects list of texts
        texts = [c["text"] for c in candidates]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"query": query, "texts": texts}
        resp = requests.post(self.base, headers=headers, json=payload, timeout=15)
        if resp.status_code != 200:
            # fallback: return original ordering
            return candidates[:top_k]
        res_json = resp.json()
        # response contains "results": [{"index": int, "score": float}, ...]
        order = [r["index"] for r in sorted(res_json.get("results", []), key=lambda x: x["score"], reverse=True)]
        reranked = [candidates[i] for i in order]
        return reranked[:top_k]
```

---

# app/agent/rag_agent.py

```python
from typing import List
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from ..retrieval.hybrid_search import HybridSearch
from ..retrieval.reranker import CohereReranker
from ..core.config import settings


class RetrievalTool:
    """A simple wrapper exposing retrieve(query) -> list[records].
    This can be exposed to a LangChain Agent as a Tool if desired.
    """
    def __init__(self, hybrid: HybridSearch, reranker: CohereReranker):
        self.hybrid = hybrid
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5):
        candidates = self.hybrid.retrieve(query, top_k=top_k*3)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return reranked


class RAGAgent:
    def __init__(self, retrieval_tool: RetrievalTool):
        # You can replace OpenAI with any LangChain LLM wrapper (Azure/OpenAI/GPT-4 etc.)
        self.llm = OpenAI(openai_api_key=settings.OPENAI_API_KEY)
        self.retrieval_tool = retrieval_tool

    def answer(self, query: str, top_k: int = 5) -> str:
        # get re-ranked contexts
        contexts = self.retrieval_tool.retrieve(query, top_k=top_k)
        context_text = "\n---\n".join([f"(id: {c['id']})\n{c['text']}" for c in contexts])

        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "You are an expert assistant. Use ONLY the provided context to answer the question. "
                "If there is insufficient information in the context, say you don't know and offer a next step.\n"
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
        )
        composed = prompt.format_prompt(query=query, context=context_text)
        # simple single-shot QA
        response = self.llm(composed.to_string())
        return response
```

---

# app/api/endpoints.py

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
import shutil
from pathlib import Path
import uuid
import os
import json

from ..ingestion.pdf_loader import load_pdf_text
from ..ingestion.indexer import index_documents_from_texts
from ..retrieval.bm25_store import BM25Store
from ..retrieval.faiss_store import FaissStore
from ..retrieval.hybrid_search import HybridSearch
from ..retrieval.reranker import CohereReranker
from ..agent.rag_agent import RetrievalTool, RAGAgent
from ..core.config import settings

router = APIRouter()

# singletons for this process — simple pattern for demonstration
_bm25 = None
_faiss = None
_hybrid = None
_reranker = None
_agent = None


@router.post("/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    """Upload PDFs (multipart) and index them.
    Returns index paths.
    """
    docs = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        tmp_path = f"/tmp/{uuid.uuid4()}_{f.filename}"
        with open(tmp_path, "wb") as out_file:
            content = await f.read()
            out_file.write(content)
        text = load_pdf_text(tmp_path)
        docs.append({"id": f.filename, "text": text, "meta": {"source": f.filename}})
    if not docs:
        raise HTTPException(status_code=400, detail="No valid PDFs uploaded")
    info = index_documents_from_texts(docs)
    return JSONResponse(content={"status": "indexed", "info": info})


@router.post("/query")
def query(q: dict):
    """POST body: {"query": "...", "top_k": 5}
    """
    global _bm25, _faiss, _hybrid, _reranker, _agent
    query_text = q.get("query")
    top_k = q.get("top_k", 5)
    if query_text is None:
        raise HTTPException(status_code=400, detail="'query' field required")

    # lazy initialize stores and agent
    if _bm25 is None:
        _bm25 = BM25Store(index_dir=settings.BM25_INDEX_DIR)
    if _faiss is None:
        _faiss = FaissStore(index_dir=settings.FAISS_INDEX_DIR)
    if _hybrid is None:
        _hybrid = HybridSearch(_bm25, _faiss, combine_k=20)
    if _reranker is None:
        _reranker = CohereReranker()
    if _agent is None:
        retrieval_tool = RetrievalTool(_hybrid, _reranker)
        _agent = RAGAgent(retrieval_tool)

    # run agent synchronously
    answer = _agent.answer(query_text, top_k=top_k)
    return JSONResponse(content={"query": query_text, "answer": str(answer)})


@router.get("/health")
def health():
    return JSONResponse(content={"status": "ok"})
```

---

# app/main.py

```python
from fastapi import FastAPI
import uvicorn

from .api.endpoints import router
from .utils.logging_cfg import configure_logging
from .core.config import settings

app = FastAPI(title="Hybrid Search RAG Service")
app.include_router(router, prefix="/api")

configure_logging(settings.LOG_LEVEL)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
```

---

# scripts/ingest_sample.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 file.pdf"
  exit 1
fi

FILE="$1"

curl -X POST "http://localhost:8000/api/ingest" -F "files=@${FILE}" -H "Accept: application/json"
```

---

# docker/Dockerfile

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

# system deps for pdfminer and faiss
RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config python3-dev curl git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# docker/docker-compose.yml

```yaml
version: '3.8'
services:
  rag:
    build: ..
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - COHERE_API_KEY=${COHERE_API_KEY}
    volumes:
      - ..:/app
```

---

# requirements.txt

```
fastapi
uvicorn[standard]
langchain
openai
python-dotenv
pydantic
pdfminer.six
faiss-cpu
sentence-transformers
rank_bm25
nltk
requests
```

> Note: `nltk` tokenizers require downloads — see README below for steps.

---

# .env.example

```
OPENAI_API_KEY="your_openai_key"
COHERE_API_KEY="your_cohere_key"
FAISS_INDEX_DIR="./indexes/faiss"
BM25_INDEX_DIR="./indexes/bm25"
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

---

# README.md (FULL — copy into repository root)

````markdown
# Multi-Document Hybrid Search RAG System

**Hybrid BM25 + FAISS retrieval + Cohere re-ranking + LangChain agent + FastAPI**

This repository is a full scaffold for a production-minded RAG system that: (1) ingests multiple PDFs, (2) indexes them into a hybrid retrieval stack (BM25 + FAISS), (3) re-ranks candidates using Cohere's reranker, and (4) answers questions through a LangChain agent exposed via a FastAPI microservice.

## Project structure

(see repository root for the full file tree)

## Quickstart (local)

1. Clone:

```bash
git clone https://github.com/THE-AkS-21/hybrid-search-rag.git
cd hybrid-search-rag
````

2. Create env and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Setup NLTK tokenizers (used by BM25):

```python
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
```

4. Create `.env` from `.env.example` and set the `OPENAI_API_KEY` and `COHERE_API_KEY`.

5. Start the service:

```bash
uvicorn app.main:app --reload --port 8000
```

6. Ingest a PDF (example):

```bash
./scripts/ingest_sample.sh ./samples/mydoc.pdf
```

7. Query the API:

```bash
curl -X POST "http://localhost:8000/api/query" -H "Content-Type: application/json" -d '{"query":"What is HTTP/2?", "top_k": 5}'
```

## Design decisions & notes

* **Hybrid search**: we fetch larger candidate pools from BM25 and FAISS, deduplicate, then pass to Cohere reranker. This reduces the chance that either sparse or dense search misses crucial paragraphs.
* **Re-ranking**: Cohere's API is used to re-rank candidate chunks. The re-ranker is critical to reduce hallucination by ensuring the LLM receives the most relevant context.
* **Chunking**: simple overlapped chunking is provided. For production, swap in a sentence-aware splitter (e.g., `tiktoken`-aware or `nltk.sent_tokenize` + aggregation) and tune chunk sizes to match your LLM's context window.
* **Embeddings**: this scaffold uses `sentence-transformers` (`all-MiniLM-L6-v2`) locally. You can optionally replace it with OpenAI embeddings for improved accuracy and fewer maintenance needs.

## Deployment (Docker)

Build and run locally with Docker Compose:

```bash
cp .env.example .env
# update .env with keys
cd docker
docker-compose up --build
```

To deploy to AWS EC2:

1. Build image locally or on EC2.
2. Push to a container registry (ECR) and run as a long-running service or with ECS.
3. Ensure environment variables are configured and volumes for `indexes` are persisted if you want to keep indexes across restarts.

## Extending the system

* Add persistent metadata stores (Postgres) to track documents and versions.
* Add a small UI to upload/search and stream responses.
* Replace the simple RAG agent with a chain-of-thought safe prompt pattern and citation generator to cite source ids.
* Add an evaluation pipeline to measure retrieval precision/recall and LLM hallucination rates.

## Security & costs

* Be careful with LLM provider keys; rotate them and use least-privilege when possible.
* Cohere and OpenAI will incur costs (re-ranker calls + LLM tokens).

## Troubleshooting

* `faiss` compilation issues: prefer `faiss-cpu` wheel on Linux/AMD64. On macOS use Docker to build.
* `nltk` errors: ensure tokenizers downloaded.

## License

MIT

```

---

# Additional notes

* This scaffold focuses on clarity and modularity. Replace the simplified components with production-grade substitutes as needed (e.g., vector DB like Weaviate/Chroma, more advanced chunkers, async request handling, and authentication).
* The sample code uses synchronous blocking I/O for clarity. For high throughput, migrate to async file handling and background indexers.

---

End of scaffold. If you want, I can:

* Generate each file as separate files in a zip you can download.
* Convert the BM25 implementation to use a sentence-aware tokenizer or a different tokenizer.
* Replace SentenceTransformers with OpenAI embeddings and update the FAISS wrapper.

Which of those would you like next?

```
