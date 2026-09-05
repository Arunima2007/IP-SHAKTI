"""Evidence Sufficiency Router for IP-SAKTI Sahayak LangGraph.

Decides whether to proceed to generation, retry retrieval with broader expansion,
or route directly to safe refusal when knowledge base evidence is insufficient.
"""
from typing import Literal
from src.graph.state import GraphState
from src.config import MAX_RETRIEVAL_RETRIES


def route_sufficiency(state: GraphState) -> Literal["generation", "retrieval", "safe_refusal"]:
    """Evaluates sufficiency state and controls the retrieval retry loop."""
    is_sufficient = state.get("evidence_sufficient", False)
    attempt = state.get("retrieval_attempt", 1)

    if is_sufficient:
        return "generation"

    # If insufficient and retries remaining, loop back to retrieval
    if attempt <= MAX_RETRIEVAL_RETRIES:
        return "retrieval"

    # Otherwise safe refusal
    return "safe_refusal"
