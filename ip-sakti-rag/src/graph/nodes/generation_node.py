"""Generation Node for IP-SAKTI Sahayak LangGraph.

Invokes strictly grounded answer generation using Gemini / deterministic grounded engine,
preserving Rules 1-7, natural sentence case, and incorporating validation feedback on regeneration attempts.
"""
from typing import Dict, Any, Optional
import time
from src.graph.state import GraphState
from src.generation.answer_generator import AnswerGenerator


class GenerationNode:
    """LangGraph node wrapping strictly grounded LLM answer generation."""

    def __init__(self, answer_generator: Optional[AnswerGenerator] = None):
        self.answer_generator = answer_generator or AnswerGenerator()

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Executes grounded generation and incorporates validation feedback on retries."""
        t0 = time.perf_counter()
        
        query = state.get("query", "")
        formatted_evidence = state.get("formatted_evidence", "")
        evidence_map = state.get("evidence_map", {})
        conflicts = state.get("detected_conflicts", [])
        attempt = state.get("generation_attempt", 0)
        feedback = state.get("validation_feedback")

        # If feedback exists from previous validation failure, enhance prompt context
        active_query = query
        if feedback and attempt > 0:
            active_query = f"{query}\n\n[REGENERATION INSTRUCTION: Previous validation noted: {feedback}. Ensure all claims strictly match cited evidence in natural sentence case without extrapolation.]"

        raw_answer, gen_meta = self.answer_generator.generate(
            query=active_query,
            formatted_evidence=formatted_evidence,
            evidence_map=evidence_map,
            detected_conflicts=conflicts
        )

        latency = round((time.perf_counter() - t0) * 1000, 2)
        node_latencies = dict(state.get("node_latencies_ms", {}))
        node_latencies["generation_ms"] = latency

        trace_entry = {
            "node": "generation",
            "generation_attempt": attempt + 1,
            "status": gen_meta.get("status", ""),
            "model": gen_meta.get("model", ""),
            "had_feedback": bool(feedback),
            "latency_ms": latency
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "draft_answer": raw_answer,
            "generated_answer": raw_answer,
            "generation_attempt": attempt + 1,
            "generation_called": True,
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }
