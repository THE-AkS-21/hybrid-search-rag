import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
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
