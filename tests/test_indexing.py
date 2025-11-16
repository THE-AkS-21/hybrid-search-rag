import tempfile
from app.ingestion.text_splitter import simple_chunk_text

def test_chunking():
    text = "".join([str(i) for i in range(5000)])
    chunks = simple_chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) >= 4
    for c in chunks:
        assert len(c) <= 1000
