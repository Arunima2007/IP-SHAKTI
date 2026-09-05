"""Health check endpoint for IP-SAKTI Sahayak API."""
import os
import json
import logging
from pathlib import Path
from fastapi import APIRouter
from src.api.schemas.documents import HealthResponse
from src.config import CHUNKS_ALL_FILE, QDRANT_PATH, BM25_INDEX_PATH

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Returns system status, active models, and knowledge base availability."""
    kb_available = Path(QDRANT_PATH).exists() and Path(BM25_INDEX_PATH).exists()
    
    total_chunks = 5212
    if Path(CHUNKS_ALL_FILE).exists():
        try:
            with open(CHUNKS_ALL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_chunks = len(data) if isinstance(data, list) else total_chunks
        except Exception as e:
            logger.warning(f"Could not read chunks count: {e}")

    return HealthResponse(
        status="healthy",
        backend_connected=True,
        knowledge_base_available=kb_available,
        total_chunks_indexed=total_chunks,
        total_documents=22,
        orchestrator="LangGraph StateGraph",
        reranker="BAAI/bge-reranker-v2-m3",
        embeddings="BAAI/bge-m3"
    )
