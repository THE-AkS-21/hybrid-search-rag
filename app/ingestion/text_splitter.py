from typing import List

def simple_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Naive overlapped chunking by character length. For production, replace with tokenizer-aware splitter."""
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end].strip())
        if end == text_length:
            break
        start = end - overlap
    return [c for c in chunks if c]
