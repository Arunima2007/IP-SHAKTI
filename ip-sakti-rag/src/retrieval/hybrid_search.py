"""Hybrid Retrieval Engine combining BGE-M3 Dense Vector Search with BM25 Okapi."""
import logging
from typing import Any, Dict, List, Literal, Optional, Union

from src.config import (
    DEFAULT_TOP_K_BM25,
    DEFAULT_TOP_K_HYBRID,
    DEFAULT_TOP_K_VECTOR,
    RRF_K,
)
from src.embeddings.embedder import BGEM3Embedder
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.filter_builder import MetadataFilterBuilder
from src.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Coordinates dual-path retrieval (BGE-M3 Qdrant + BM25) and applies
    Reciprocal Rank Fusion (RRF) or Weighted Score Normalization.
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        bm25_engine: Optional[BM25SearchEngine] = None,
        embedder: Optional[BGEM3Embedder] = None,
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.bm25_engine = bm25_engine or BM25SearchEngine()
        self.embedder = embedder or BGEM3Embedder()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_HYBRID,
        top_k_vector: int = DEFAULT_TOP_K_VECTOR,
        top_k_bm25: int = DEFAULT_TOP_K_BM25,
        fusion_method: Literal["rrf", "weighted_score"] = "rrf",
        rrf_k: int = RRF_K,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        auto_filter: bool = True,
        debug: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes hybrid search across vector and lexical indices.

        Args:
            query: Natural language query or legal citation.
            top_k: Number of fused results to return (20-30).
            top_k_vector: Candidates retrieved from Qdrant.
            top_k_bm25: Candidates retrieved from BM25.
            fusion_method: 'rrf' (Reciprocal Rank Fusion) or 'weighted_score'.
            rrf_k: Smoothing parameter for RRF (default 60).
            vector_weight: Weight for vector retrieval.
            bm25_weight: Weight for BM25 retrieval.
            filters: Explicit metadata filters.
            auto_filter: Whether to infer metadata filters if not explicitly provided.
            debug: If True, returns full diagnostic bundle (vector, bm25, fused).

        Returns:
            List of candidate dictionaries, or diagnostic dictionary if debug=True.
        """
        # 1. Filter determination
        active_filters = filters
        filter_confidence = 1.0 if filters else 0.0
        if active_filters is None and auto_filter:
            active_filters, filter_confidence = MetadataFilterBuilder.infer_filters_from_query(query)

        qdrant_filter = MetadataFilterBuilder.build_qdrant_filter(active_filters)

        # 2. Path A: Dense Vector Search
        query_vector = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k_vector,
            filters=qdrant_filter,
        )

        # 3. Path B: BM25 Lexical Search
        bm25_results = self.bm25_engine.search(
            query=query,
            top_k=top_k_bm25,
            filters=active_filters,
        )

        # 4. Fusion & Deduplication
        if fusion_method == "rrf":
            fused_candidates = self._reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                k=rrf_k,
                w_vec=vector_weight,
                w_bm25=bm25_weight,
                top_k=top_k,
            )
        else:
            fused_candidates = self._score_normalization_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                w_vec=vector_weight,
                w_bm25=bm25_weight,
                top_k=top_k,
            )

        if debug:
            return {
                "query": query,
                "inferred_filters": active_filters,
                "filter_confidence": filter_confidence,
                "vector_results": vector_results,
                "bm25_results": bm25_results,
                "fused_results": fused_candidates,
            }

        return fused_candidates

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int,
        w_vec: float,
        w_bm25: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Combines rankings via Reciprocal Rank Fusion."""
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # Process Vector results
        for rank, item in enumerate(vector_results, start=1):
            cid = item["chunk_id"]
            rrf_score = w_vec * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

            chunk_map[cid] = {
                **item,
                "vector_score": item["score"],
                "vector_rank": rank,
                "bm25_score": None,
                "bm25_rank": None,
                "retrieval_method": "vector",
            }

        # Process BM25 results
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            rrf_score = w_bm25 * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

            if cid in chunk_map:
                chunk_map[cid]["bm25_score"] = item["score"]
                chunk_map[cid]["bm25_rank"] = rank
                chunk_map[cid]["retrieval_method"] = "hybrid"
            else:
                chunk_map[cid] = {
                    **item,
                    "vector_score": None,
                    "vector_rank": None,
                    "bm25_score": item["score"],
                    "bm25_rank": rank,
                    "retrieval_method": "bm25",
                }

        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        final_results = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            res = chunk_map[cid]
            res["score"] = round(rrf_scores[cid], 6)
            res["rank"] = rank
            final_results.append(res)

        return final_results

    def _score_normalization_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        w_vec: float,
        w_bm25: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Combines normalized cosine similarity and BM25 scores linearly."""
        chunk_map: Dict[str, Dict[str, Any]] = {}
        combined_scores: Dict[str, float] = {}

        # Normalize Vector scores (Cosine similarity: 0 to 1)
        v_scores = [item["score"] for item in vector_results]
        v_min = min(v_scores) if v_scores else 0.0
        v_max = max(v_scores) if v_scores else 1.0
        v_range = (v_max - v_min) if (v_max - v_min) > 1e-6 else 1.0

        for rank, item in enumerate(vector_results, start=1):
            cid = item["chunk_id"]
            norm_v = (item["score"] - v_min) / v_range
            combined_scores[cid] = combined_scores.get(cid, 0.0) + w_vec * norm_v

            chunk_map[cid] = {
                **item,
                "vector_score": item["score"],
                "vector_rank": rank,
                "bm25_score": None,
                "bm25_rank": None,
                "retrieval_method": "vector",
            }

        # Normalize BM25 scores
        b_scores = [item["score"] for item in bm25_results]
        b_min = min(b_scores) if b_scores else 0.0
        b_max = max(b_scores) if b_scores else 1.0
        b_range = (b_max - b_min) if (b_max - b_min) > 1e-6 else 1.0

        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            norm_b = (item["score"] - b_min) / b_range
            combined_scores[cid] = combined_scores.get(cid, 0.0) + w_bm25 * norm_b

            if cid in chunk_map:
                chunk_map[cid]["bm25_score"] = item["score"]
                chunk_map[cid]["bm25_rank"] = rank
                chunk_map[cid]["retrieval_method"] = "hybrid"
            else:
                chunk_map[cid] = {
                    **item,
                    "vector_score": None,
                    "vector_rank": None,
                    "bm25_score": item["score"],
                    "bm25_rank": rank,
                    "retrieval_method": "bm25",
                }

        sorted_cids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        final_results = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            res = chunk_map[cid]
            res["score"] = round(combined_scores[cid], 6)
            res["rank"] = rank
            final_results.append(res)

        return final_results
