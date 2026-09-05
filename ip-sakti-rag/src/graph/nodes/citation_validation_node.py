"""Citation Validation Node for IP-SAKTI Sahayak LangGraph.

Converts internal [E#] evidence citations into human-readable citations, extracts structured
citation objects, executes claim-level factual support verification, and decides whether the answer
passes validation or requires controlled regeneration.
"""
from typing import Dict, Any, Optional
import time
from src.graph.state import GraphState
from src.generation.citation_engine import CitationEngine
from src.generation.citation_validator import ClaimCitationValidator
from src.config import MAX_GENERATION_RETRIES


class CitationValidationNode:
    """LangGraph node wrapping citation conversion, claim extraction, and validation decision."""

    def __init__(
        self,
        citation_engine: Optional[CitationEngine] = None,
        citation_validator: Optional[ClaimCitationValidator] = None
    ):
        self.citation_engine = citation_engine or CitationEngine()
        self.citation_validator = citation_validator or ClaimCitationValidator()

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Validates draft answer citations and decides next workflow state."""
        t0 = time.perf_counter()
        
        draft_answer = state.get("draft_answer", "")
        evidence_map = state.get("evidence_map", {})
        attempt = state.get("generation_attempt", 1)

        # 1. Convert [E#] to [1] and build structured citation objects
        formatted_answer, citation_objects = self.citation_engine.convert_answer_citations(
            answer_text=draft_answer,
            evidence_map=evidence_map
        )

        # 2. Execute Claim-Level Validation
        val_result = self.citation_validator.validate_answer(
            answer_text=formatted_answer,
            evidence_map=evidence_map,
            citation_objects=citation_objects
        )

        is_valid = val_result.get("is_valid", True)
        flagged_issues = val_result.get("flagged_issues", [])
        sanitized_final = val_result.get("sanitized_answer", formatted_answer)
        claims = val_result.get("claims", [])
        metrics = val_result.get("metrics", {})

        # 3. Formulate Validation Feedback if Invalid
        feedback = None
        if not is_valid and attempt < MAX_GENERATION_RETRIES:
            validation_status = "RETRY_GENERATION"
            descriptions = [f["description"] for f in flagged_issues[:2]]
            feedback = "; ".join(descriptions) if descriptions else "Unsupported assertions detected. Rewrite with strict factual containment."
        elif is_valid:
            validation_status = "VALID"
        else:
            validation_status = "INVALID"

        latency = round((time.perf_counter() - t0) * 1000, 2)
        node_latencies = dict(state.get("node_latencies_ms", {}))
        node_latencies["citation_validation_ms"] = latency

        trace_entry = {
            "node": "citation_validation",
            "is_valid": is_valid,
            "validation_status": validation_status,
            "total_claims": metrics.get("total_claims", len(claims)),
            "supported_claims": metrics.get("supported_claims", 0),
            "citations_count": len(citation_objects),
            "flagged_issues_count": len(flagged_issues),
            "latency_ms": latency
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "final_answer": sanitized_final,
            "claims": claims,
            "citations": citation_objects,
            "citation_validation": val_result,
            "validation_status": validation_status,
            "validation_feedback": feedback,
            "is_valid": is_valid,
            "is_refusal": False,
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }
