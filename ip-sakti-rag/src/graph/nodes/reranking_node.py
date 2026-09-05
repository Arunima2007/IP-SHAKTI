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
from src.retrieval.legal_identifier_parser import document_matches, provision_matches


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
        parsed = state.get("parsed_identifier") or {}
        
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

        # Explicit legal references are hard constraints, not relevance hints.
        # A candidate for Section 4 must never remain eligible for a Section
        # 3(p) answer merely because the cross-encoder finds it similar.
        if parsed.get("type") and parsed.get("value"):
            exact_candidates = [c for c in candidates if c.get("exact_provision_match")]
            constrained = exact_candidates or [c for c in candidates if provision_matches(c, parsed)
                                                and document_matches(c, parsed.get("canonical_title") or parsed.get("document_hint"))]
            candidates = constrained

        if not candidates:
            return {
                "reranked_candidates": [], "selected_evidence": [], "reranking_called": True,
                "node_latencies_ms": dict(state.get("node_latencies_ms", {})),
            }

        # 1. Cross-Encoder Reranking
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=RERANK_TOP_K
        )
        if parsed.get("type"):
            for item in reranked:
                item["exact_provision_match"] = True
                item["exact_document_match"] = bool(parsed.get("canonical_title"))

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
