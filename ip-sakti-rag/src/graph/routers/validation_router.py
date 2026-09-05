"""Citation Validation Router for IP-SAKTI Sahayak LangGraph.

Controls the validation decision loop: accepting valid answers, triggering controlled
generation retries with validation feedback, or terminating safely.
"""
from typing import Literal
from src.graph.state import GraphState
from src.config import MAX_GENERATION_RETRIES


def route_validation(state: GraphState) -> Literal["end", "generation", "safe_refusal"]:
    """Routes based on citation verification results and retry budget."""
    validation_status = state.get("validation_status", "VALID")
    attempt = state.get("generation_attempt", 1)

    if validation_status == "VALID":
        return "end"

    # If invalid and retries remaining, loop back to generation with feedback
    if validation_status == "RETRY_GENERATION" and attempt < MAX_GENERATION_RETRIES:
        return "generation"

    # If maximum retries exceeded and answer is completely invalid, safe refusal
    if not state.get("is_valid", True):
        return "safe_refusal"

    return "end"
