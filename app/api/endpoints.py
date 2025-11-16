from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import uuid
import os

from app.ingestion.pdf_loader import load_pdf_text
from app.ingestion.indexer import index_documents_from_texts
from app.retrieval.bm25_store import BM25Store
from app.retrieval.faiss_store import FaissStore
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import CohereReranker
from app.agent.rag_agent import RetrievalTool, RAGAgent
from app.core.config import settings

router = APIRouter()

# process-global singletons (simple demonstration)
_bm25 = None
_faiss = None
_hybrid = None
_reranker = None
_agent = None

@router.post("/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    """Upload PDFs and index them. Returns index info."""
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
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not docs:
        raise HTTPException(status_code=400, detail="No valid PDFs uploaded")

    info = index_documents_from_texts(docs)
    return JSONResponse(content={"status": "indexed", "info": info})

@router.post("/query")
def query(q: dict):
    """POST body: {"query": "...", "top_k": 5}
    Returns LLM answer using hybrid retrieval + reranking.
    """
    global _bm25, _faiss, _hybrid, _reranker, _agent
    query_text = q.get("query")
    top_k = int(q.get("top_k", 5))
    if not query_text:
        raise HTTPException(status_code=400, detail="'query' field required")

    # initialize stores lazily
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

    answer = _agent.answer(query_text, top_k=top_k)
    return JSONResponse(content={"query": query_text, "answer": str(answer)})

@router.get("/health")
def health():
    return JSONResponse(content={"status": "ok"})
