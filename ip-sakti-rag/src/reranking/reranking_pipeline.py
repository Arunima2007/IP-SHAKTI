"""Integrated Retrieval and Cross-Encoder Reranking Pipeline for IP-SAKTI Sahayak (Milestone 3).

Flow:
User Query
    ↓
Hybrid Search Engine (Vector BGE-M3 + BM25 Lexical + Dynamic Metadata Filtering)
    ↓
RRF Fusion (Top 25–30 candidate chunks)
    ↓
Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)
    ↓
Diversity-Aware Evidence Selector (Exact Citation Calibration + Cross-Domain Balancing)
    ↓
Final Top 5–8 Evidence Chunks (With complete provenance & latency profiling)
"""
import logging
import time
from typing import Any, Dict, List, Optional

from src.config import (
    BM25_TOP_K,
    FINAL_TOP_K,
    FUSION_TOP_K,
    MAX_CHUNKS_PER_DOCUMENT,
    MAX_CHUNKS_PER_DOMAIN,
    RERANK_TOP_K,
    VECTOR_TOP_K,
)
from src.embeddings.embedder import BGEM3Embedder
from src.reranking.diversity_selector import DiversityAwareSelector
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class RetrievalAndRerankingPipeline:
    """Full two-stage retrieval and reranking pipeline for IP-SAKTI Sahayak."""

    def __init__(
        self,
        hybrid_engine: Optional[HybridSearchEngine] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        diversity_selector: Optional[DiversityAwareSelector] = None,
    ):
        # Initialize subcomponents lazily if not provided
        self.hybrid_engine = hybrid_engine or HybridSearchEngine()
        self.reranker = reranker or CrossEncoderReranker()
        self.diversity_selector = diversity_selector or DiversityAwareSelector()

    def search_and_rerank(
        self,
        query: str,
        top_k_vector: int = VECTOR_TOP_K,
        top_k_bm25: int = BM25_TOP_K,
        top_k_fusion: int = FUSION_TOP_K,
        top_k_rerank: int = RERANK_TOP_K,
        final_top_k: int = FINAL_TOP_K,
        max_chunks_per_doc: int = MAX_CHUNKS_PER_DOCUMENT,
        max_chunks_per_domain: int = MAX_CHUNKS_PER_DOMAIN,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end hybrid retrieval, cross-encoder reranking, and diversity selection.

        Returns a dictionary with:
        - 'query': original user query
        - 'final_evidence': list of top 5–8 reranked, diversity-balanced evidence chunks
        - 'reranked_candidates': all candidate chunks sorted by cross-encoder score
        - 'hybrid_candidates': initial candidates from RRF fusion
        - 'inferred_filters': any dynamic metadata filters applied
        - 'latency_breakdown_ms': timing for each stage
        """
        pipeline_t0 = time.time()
        latencies: Dict[str, float] = {}

        # 1. Hybrid Retrieval
        t_hybrid_0 = time.time()
        hybrid_result = self.hybrid_engine.search(
            query=query,
            top_k=top_k_fusion,
            top_k_vector=top_k_vector,
            top_k_bm25=top_k_bm25,
            fusion_method="rrf",
            debug=True,
        )
        latencies["hybrid_retrieval_ms"] = round((time.time() - t_hybrid_0) * 1000, 2)
        latencies["vector_search_ms"] = hybrid_result.get("vector_latency_ms", 0.0)
        latencies["bm25_search_ms"] = hybrid_result.get("bm25_latency_ms", 0.0)
        latencies["rrf_fusion_ms"] = hybrid_result.get("fusion_latency_ms", 0.0)

        raw_candidates = hybrid_result.get("fused_results", [])[:top_k_rerank]

        # 2. Cross-Encoder Reranking
        t_rerank_0 = time.time()
        reranked = self.reranker.rerank(
            query=query,
            candidates=raw_candidates,
            top_k=top_k_rerank,
        )
        latencies["cross_encoder_ms"] = round((time.time() - t_rerank_0) * 1000, 2)

        # 3. Diversity-Aware Evidence Selection
        t_select_0 = time.time()
        final_evidence = self.diversity_selector.select_evidence(
            query=query,
            reranked_candidates=reranked,
            top_k=final_top_k,
            max_chunks_per_doc=max_chunks_per_doc,
            max_chunks_per_domain=max_chunks_per_domain,
        )
        latencies["diversity_selection_ms"] = round((time.time() - t_select_0) * 1000, 2)

        latencies["total_pipeline_ms"] = round((time.time() - pipeline_t0) * 1000, 2)

        result_payload = {
            "query": query,
            "final_evidence": final_evidence,
            "reranked_candidates": reranked,
            "hybrid_candidates": raw_candidates,
            "inferred_filters": hybrid_result.get("inferred_filters"),
            "latency_breakdown_ms": latencies,
        }

        if debug:
            result_payload["vector_results"] = hybrid_result.get("vector_results", [])
            result_payload["bm25_results"] = hybrid_result.get("bm25_results", [])

        return result_payload
