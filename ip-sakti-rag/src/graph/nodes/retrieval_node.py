"""Retrieval Node for IP-SAKTI Sahayak LangGraph.

Executes hybrid retrieval (BGE-M3 Dense Vector Search + BM25 Lexical Search + RRF Fusion)
using the existing HybridSearchEngine without duplicating retrieval infrastructure.
Supports controlled broader query retry when previous evidence was insufficient.
"""
from typing import Dict, Any, Optional
import time
from src.graph.state import GraphState
from src.retrieval.hybrid_search import HybridSearchEngine
from src.config import FUSION_TOP_K, VECTOR_TOP_K, BM25_TOP_K, EXACT_IDENTIFIER_BOOST
from src.retrieval.exact_lookup import exact_legal_lookup


class RetrievalNode:
    """LangGraph node wrapping hybrid search with retry expansion."""

    def __init__(self, hybrid_engine: Optional[HybridSearchEngine] = None):
        self.hybrid_engine = hybrid_engine or HybridSearchEngine()

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Executes hybrid retrieval and returns candidate chunks."""
        t0 = time.perf_counter()
        
        attempt = state.get("retrieval_attempt", 0)
        query = state.get("expanded_query") or state.get("query", "")
        
        # On retry: increase candidate depth
        top_k = FUSION_TOP_K + (15 if attempt > 0 else 0)
        top_vec = VECTOR_TOP_K + (10 if attempt > 0 else 0)
        top_bm25 = BM25_TOP_K + (10 if attempt > 0 else 0)

        # Execute hybrid search
        search_result = self.hybrid_engine.search(
            query=query,
            top_k=top_k,
            top_k_vector=top_vec,
            top_k_bm25=top_bm25,
            fusion_method="rrf",
            debug=True
        )

        candidates = search_result.get("fused_results", []) if isinstance(search_result, dict) else search_result

        # Exact legal identifier lookup (first‑class retrieval)
        exact_candidates = exact_legal_lookup(state)
        # Merge: exact matches first, then hybrid results without duplicates
        if exact_candidates:
            seen_ids = {c.get("chunk_id") for c in exact_candidates}
            merged = exact_candidates + [c for c in candidates if c.get("chunk_id") not in seen_ids]
            candidates = merged

        latency = round((time.perf_counter() - t0) * 1000, 2)

        node_latencies = dict(state.get("node_latencies_ms", {}))
        node_latencies["retrieval_ms"] = latency

        trace_entry = {
            "node": "retrieval",
            "retrieval_attempt": attempt + 1,
            "candidates_retrieved": len(candidates),
            "latency_ms": latency
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "retrieval_candidates": candidates,
            "retrieval_attempt": attempt + 1,
            "retrieval_called": True,
            "retrieval_performed": True,
            "retrieval_diagnostics": {
                "query": query,
                "query_type": state.get("query_type"),
                "identified_document": (state.get("parsed_identifier") or {}).get("canonical_title"),
                "identified_provision": (state.get("parsed_identifier") or {}).get("value"),
                "identifier_match_count": len(exact_candidates),
                "bm25_top_k": search_result.get("bm25_results", [])[:10],
                "vector_top_k": search_result.get("vector_results", [])[:10],
                "rrf_top_k": search_result.get("fused_results", [])[:10],
                "final_candidates": candidates[:10],
            },
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }
