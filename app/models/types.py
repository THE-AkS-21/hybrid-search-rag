from typing import TypedDict, Dict, Any

class DocRecord(TypedDict):
    id: str
    text: str
    meta: Dict[str, Any]
