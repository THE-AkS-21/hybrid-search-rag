import os
from typing import List
from app.utils.serializers import doc_to_record
from app.ingestion.text_splitter import simple_chunk_text
from app.retrieval.faiss_store import FaissStore
from app.retrieval.bm25_store import BM25Store
from app.core.config import settings

def index_documents_from_texts(docs: List[dict]):
    """docs: list of {"id": str, "text": str, "meta": dict}

    Builds/upserts BM25 and FAISS indexes for provided docs.
    Returns index info.
    """
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

    faiss.save()
    bm25.save()

    return {"faiss_index": settings.FAISS_INDEX_DIR, "bm25_index": settings.BM25_INDEX_DIR, "num_docs": len(docs)}
