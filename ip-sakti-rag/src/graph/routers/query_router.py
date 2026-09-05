"""Query Router for IP-SAKTI Sahayak LangGraph.

Determines whether a query is strictly within domain scope or should be routed
directly to SafeRefusalNode without calling retrieval or generation.
"""
from typing import Literal
from src.graph.state import GraphState


def route_query(state: GraphState) -> Literal["retrieval", "safe_refusal"]:
    """
    Hard Out-of-Scope Gating:
    Directly routes any non-domain query to safe_refusal.
    Zero retrieval calls, zero reranking, zero generation, zero citations.
    """
    scope_status = state.get("scope_status", "IN_SCOPE")
    query_type = state.get("query_type", "FACTUAL")

    if scope_status == "OUT_OF_SCOPE" or query_type == "OUT_OF_SCOPE":
        return "safe_refusal"
        
    return "retrieval"
