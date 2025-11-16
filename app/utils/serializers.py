from typing import Dict
def doc_to_record(doc_id: str, idx: int, text: str, meta: Dict) -> Dict:
    return {"id": f"{doc_id}::{idx}", "text": text, "meta": meta}
