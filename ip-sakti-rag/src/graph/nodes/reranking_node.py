"""Reranking Node for IP-SAKTI Sahayak LangGraph.

Wraps BAAI/bge-reranker-v2-m3 cross-encoder and DiversityAwareSelector to score,
filter, and balance evidence chunks across legal domains and statutory tiers.
"""
from typing import Dict, Any, Optional
import time
from src.graph.state import GraphState
from src.reranking.reranker import CrossEncoderReranker
from src.reranking.diversity_selector import DiversityAwareSelector
from src.config import RERANK_TOP_K, FINAL_TOP_K, MAX_CHUNKS_PER_DOCUMENT, MAX_CHUNKS_PER_DOMAIN


class RerankingNode:
    """LangGraph node wrapping cross-encoder reranking and diversity selection."""

    def __init__(
        self,
        reranker: Optional[CrossEncoderReranker] = None,
        diversity_selector: Optional[DiversityAwareSelector] = None
    ):
        self.reranker = reranker or CrossEncoderReranker()
        self.diversity_selector = diversity_selector or DiversityAwareSelector()

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Executes cross-encoder scoring and diversity-aware selection."""
        t0 = time.perf_counter()
        
        query = state.get("query", "")
        candidates = state.get("retrieval_candidates", [])
        
        if not candidates:
            latency = round((time.perf_counter() - t0) * 1000, 2)
            node_latencies = dict(state.get("node_latencies_ms", {}))
            node_latencies["reranking_ms"] = latency
            return {
                "reranked_candidates": [],
                "selected_evidence": [],
                "reranking_called": True,
                "node_latencies_ms": node_latencies
            }

        # 1. Cross-Encoder Reranking
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=RERANK_TOP_K
        )

        # 2. Diversity-Aware Selection
        selected = self.diversity_selector.select_evidence(
            query=query,
            reranked_candidates=reranked,
            top_k=FINAL_TOP_K,
            max_chunks_per_doc=MAX_CHUNKS_PER_DOCUMENT,
            max_chunks_per_domain=MAX_CHUNKS_PER_DOMAIN
        )

        latency = round((time.perf_counter() - t0) * 1000, 2)
        node_latencies = dict(state.get("node_latencies_ms", {}))
        node_latencies["reranking_ms"] = latency

        trace_entry = {
            "node": "reranking",
            "candidates_reranked": len(reranked),
            "evidence_selected": len(selected),
            "top_rerank_score": selected[0].get("reranker_score", selected[0].get("score", 0.0)) if selected else 0.0,
            "latency_ms": latency
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "reranked_candidates": reranked,
            "selected_evidence": selected,
            "reranking_called": True,
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }
