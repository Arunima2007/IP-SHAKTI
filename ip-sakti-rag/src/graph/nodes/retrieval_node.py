"""Retrieval Node for IP-SAKTI Sahayak LangGraph.

Executes hybrid retrieval (BGE-M3 Dense Vector Search + BM25 Lexical Search + RRF Fusion)
using the existing HybridSearchEngine without duplicating retrieval infrastructure.
Supports controlled broader query retry when previous evidence was insufficient.
"""
from typing import Dict, Any, Optional
import time
from src.graph.state import GraphState
from src.retrieval.hybrid_search import HybridSearchEngine
from src.config import FUSION_TOP_K, VECTOR_TOP_K, BM25_TOP_K


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
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }
