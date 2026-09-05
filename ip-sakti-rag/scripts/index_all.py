"""One-shot indexing script for IP-SAKTI Sahayak (Milestone 2).

Generates BAAI/bge-m3 embeddings, indexes into Qdrant, and builds BM25 index.
"""
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BM25_DIR,
    BM25_INDEX_PATH,
    CHUNKS_DIR,
    EMBEDDINGS_CACHE_PATH,
    INDEXES_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_DIR,
)

from src.embeddings.embedder import BGEM3Embedder
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_indexing(
    chunks_file: Path = CHUNKS_DIR / "all_chunks.json",
    batch_size: int = 64,
    force_reembed: bool = False,
) -> None:

    """Executes embedding generation, Qdrant upsert, and BM25 index build."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("IP-SAKTI SAHAYAK — MILESTONE 2 INDEXING PIPELINE")
    logger.info("=" * 60)

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    total_chunks = len(chunks)
    logger.info(f"Loaded {total_chunks} chunks from {chunks_file}")

    # Ensure index directories exist
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Embeddings Generation (BGE-M3)
    logger.info("\n--- STEP 1: Dense Embeddings (BAAI/bge-m3) ---")
    embedder = BGEM3Embedder(batch_size=batch_size)
    t0 = time.time()
    embedded_chunks = embedder.embed_chunks(
        chunks=chunks,
        batch_size=batch_size,
        force_recompute=force_reembed,
        show_progress=True,
    )
    embed_duration = time.time() - t0
    logger.info(f"Dense embeddings ready in {embed_duration:.2f}s")

    # 2. Qdrant Vector Database Indexing
    logger.info(f"\n--- STEP 2: Qdrant Indexing ('{QDRANT_COLLECTION_NAME}') ---")
    vector_store = QdrantVectorStore(collection_name=QDRANT_COLLECTION_NAME)
    t0 = time.time()
    indexed_count = vector_store.index_chunks(
        chunks_with_vectors=embedded_chunks,
        batch_size=100,
        show_progress=True,
    )
    qdrant_duration = time.time() - t0
    logger.info(f"Qdrant collection populated with {indexed_count} points in {qdrant_duration:.2f}s")

    # 3. BM25 Lexical Indexing
    logger.info("\n--- STEP 3: BM25 Lexical Indexing ---")
    bm25_engine = BM25SearchEngine()
    t0 = time.time()
    bm25_engine.build_index(chunks=chunks, save=True)
    bm25_duration = time.time() - t0
    logger.info(f"BM25 index built with {len(bm25_engine.corpus_chunks)} documents in {bm25_duration:.2f}s")

    total_duration = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("INDEXING COMPLETE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Chunks Processed:   {total_chunks}")
    logger.info(f"Qdrant Points Count:      {indexed_count}")
    logger.info(f"BM25 Indexed Documents:   {len(bm25_engine.corpus_chunks)}")
    logger.info(f"Embedding Cache Path:     {EMBEDDINGS_CACHE_PATH}")
    logger.info(f"BM25 Index Path:          {BM25_INDEX_PATH}")
    logger.info(f"Total Pipeline Runtime:   {total_duration:.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_indexing()
