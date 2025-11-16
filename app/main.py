from fastapi import FastAPI
import uvicorn
import logging

from app.api.endpoints import router
from app.utils.logging_cfg import configure_logging
from app.core.config import settings

app = FastAPI(title="Hybrid Search RAG Service")
app.include_router(router, prefix="/api")

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

@app.on_event("startup")
def startup_event():
    logger.info("Starting Hybrid Search RAG Service")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
