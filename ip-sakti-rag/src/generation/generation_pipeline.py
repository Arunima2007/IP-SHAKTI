"""Grounded Answer Generation & Citation Pipeline for IP-SAKTI Sahayak.

Orchestrates:
Retrieval (BGE-M3 + BM25) -> Reranking (bge-reranker-v2-m3) -> Intent Diversity Selection ->
Evidence Formatting (Tiers + Conflicts) -> Grounded Generation (Gemini / Rules) ->
Citation Conversion -> Claim-Level Citation Validation -> Hallucination Control.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time
import logging

from src.reranking.reranking_pipeline import RetrievalAndRerankingPipeline
from src.generation.evidence_formatter import EvidenceFormatter
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_engine import CitationEngine
from src.generation.citation_validator import ClaimCitationValidator

logger = logging.getLogger(__name__)


@dataclass
class GroundedAnswerResult:
    """Structured result of grounded generation and verification pipeline."""
    query: str
    final_answer: str
    raw_answer: str
    is_refusal: bool
    is_valid: bool
    structured_citations: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    flagged_issues: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    selected_evidence: List[Dict[str, Any]] = field(default_factory=list)
    detected_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result into a clean structured dictionary."""
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "raw_answer": self.raw_answer,
            "is_refusal": self.is_refusal,
            "is_valid": self.is_valid,
            "structured_citations": self.structured_citations,
            "claims": self.claims,
            "flagged_issues": self.flagged_issues,
            "metrics": self.metrics,
            "selected_evidence": self.selected_evidence,
            "detected_conflicts": self.detected_conflicts,
            "latencies_ms": self.latencies_ms,
            "metadata": self.metadata
        }


class GenerationPipeline:
    """End-to-end grounded generation and citation verification coordinator."""

    def __init__(
        self,
        retrieval_pipeline: Optional[RetrievalAndRerankingPipeline] = None,
        evidence_formatter: Optional[EvidenceFormatter] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        citation_engine: Optional[CitationEngine] = None,
        citation_validator: Optional[ClaimCitationValidator] = None,
    ):
        self.retrieval_pipeline = retrieval_pipeline or RetrievalAndRerankingPipeline()
        self.evidence_formatter = evidence_formatter or EvidenceFormatter()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.citation_engine = citation_engine or CitationEngine()
        self.citation_validator = citation_validator or ClaimCitationValidator()

    def process_query(self, query: str) -> GroundedAnswerResult:
        """
        Executes the full pipeline for a given query:
        Retrieval -> Reranking -> Evidence Selection -> LLM Generation -> Citation Engine -> Validation.
        """
        total_start = time.perf_counter()
        latencies = {}

        # 1. Retrieval & Reranking
        ret_start = time.perf_counter()
        rerank_result = self.retrieval_pipeline.search_and_rerank(query)
        selected_chunks = rerank_result.get("final_evidence", [])
        latencies["retrieval_ms"] = rerank_result.get("latency_breakdown_ms", {}).get("hybrid_retrieval_ms", 0.0)
        latencies["rerank_ms"] = rerank_result.get("latency_breakdown_ms", {}).get("cross_encoder_ms", 0.0)

        # 2. Evidence Formatting & Source Hierarchy
        fmt_start = time.perf_counter()
        formatted_evidence, evidence_map, conflicts = self.evidence_formatter.format_evidence(
            chunks=selected_chunks,
            query=query
        )
        latencies["evidence_formatting_ms"] = round((time.perf_counter() - fmt_start) * 1000, 2)

        # 3. Grounded Answer Generation
        gen_start = time.perf_counter()
        raw_answer, gen_meta = self.answer_generator.generate(
            query=query,
            formatted_evidence=formatted_evidence,
            evidence_map=evidence_map,
            detected_conflicts=conflicts
        )
        latencies["generation_ms"] = round((time.perf_counter() - gen_start) * 1000, 2)

        # 4. Citation Conversion & Sources Section
        cit_start = time.perf_counter()
        final_answer_text, citation_objects = self.citation_engine.convert_answer_citations(
            answer_text=raw_answer,
            evidence_map=evidence_map
        )
        latencies["citation_conversion_ms"] = round((time.perf_counter() - cit_start) * 1000, 2)

        # 5. Claim-Level Citation Validation & Hallucination Audit
        val_start = time.perf_counter()
        validation_output = self.citation_validator.validate_answer(
            answer_text=final_answer_text,
            evidence_map=evidence_map,
            citation_objects=citation_objects
        )
        latencies["validation_ms"] = round((time.perf_counter() - val_start) * 1000, 2)

        # Total latency
        latencies["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        is_refusal = validation_output.get("is_refusal", False) or ("could not find sufficient" in final_answer_text.lower())
        sanitized_final = validation_output.get("sanitized_answer", final_answer_text)

        return GroundedAnswerResult(
            query=query,
            final_answer=sanitized_final,
            raw_answer=raw_answer,
            is_refusal=is_refusal,
            is_valid=validation_output.get("is_valid", True),
            structured_citations=citation_objects,
            claims=validation_output.get("claims", []),
            flagged_issues=validation_output.get("flagged_issues", []),
            metrics=validation_output.get("metrics", {}),
            selected_evidence=list(evidence_map.values()),
            detected_conflicts=conflicts,
            latencies_ms=latencies,
            metadata={
                "model_used": gen_meta.get("model", ""),
                "generation_status": gen_meta.get("status", ""),
                "reranker_model": rerank_result.get("metadata", {}).get("reranker_model", ""),
                "evidence_chunks_count": len(selected_chunks)
            }
        )
