"""Safe Refusal Node for IP-SAKTI Sahayak LangGraph.

Handles out-of-scope user requests and insufficient evidence conditions safely
without hallucinations or arbitrary document retrieval, outputting clear domain boundary notices.
"""
from typing import Dict, Any
import time
from src.graph.state import GraphState
from src.config import INSUFFICIENT_EVIDENCE_MESSAGE


class SafeRefusalNode:
    """LangGraph node returning safe refusal statements for out-of-scope or ungrounded queries."""

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Returns standard safe refusal or domain scope clarification in natural sentence case."""
        t0 = time.perf_counter()
        
        query_type = state.get("query_type", "INSUFFICIENT_EVIDENCE")
        scope_status = state.get("scope_status", "IN_SCOPE")

        if scope_status == "OUT_OF_SCOPE" or query_type == "OUT_OF_SCOPE":
            final_message = (
                "I can help with Intellectual Property, Ayurveda/AYUSH regulations, Traditional Knowledge, "
                "Biological Diversity, and related international IP frameworks. This question is outside my supported domain."
            )
            final_type = "SAFE_REFUSAL"
            refusal_reason = state.get("refusal_reason") or state.get("scope_reason") or "unsupported_general_knowledge"
        else:
            final_message = INSUFFICIENT_EVIDENCE_MESSAGE
            final_type = "INSUFFICIENT_EVIDENCE"
            refusal_reason = state.get("evidence_sufficiency_reason") or state.get("failure_reason") or "insufficient_authoritative_evidence"

        latency = round((time.perf_counter() - t0) * 1000, 2)
        node_latencies = dict(state.get("node_latencies_ms", {}))
        node_latencies["safe_refusal_ms"] = latency

        trace_entry = {
            "node": "safe_refusal",
            "query_type": query_type,
            "scope_status": scope_status,
            "refusal_reason": refusal_reason,
            "latency_ms": latency
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "final_answer": final_message,
            "final_answer_type": final_type,
            "is_refusal": True,
            "is_valid": True,
            "claims": [],
            "citations": [],
            "evidence": [],
            "selected_evidence": [],
            "validation_status": "REFUSAL",
            "refusal_reason": refusal_reason,
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }

