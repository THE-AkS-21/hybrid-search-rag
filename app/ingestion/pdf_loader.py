from typing import List
from pathlib import Path
from pdfminer.high_level import extract_text

def load_pdf_text(path: str) -> str:
    """Extract text from a local PDF file using pdfminer."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    text = extract_text(str(path))
    return text

def load_many_pdfs(paths: List[str]) -> List[dict]:
    results = []
    for p in paths:
        text = load_pdf_text(p)
        results.append({"path": p, "text": text})
    return results
