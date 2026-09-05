"""LangGraph State Definition for IP-SAKTI Sahayak.

Defines a strongly typed shared graph state for the orchestration layer,
tracking query understanding, domain scope classification, retrieval, reranking,
evidence sufficiency, grounded generation, citation validation, and decision feedback loops.
"""
from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict, total=False):
    """Strongly typed shared state for the IP-SAKTI Sahayak LangGraph workflow."""

    # 1. User Input & Query Understanding
    query: str
    original_query: str
    expanded_query: Optional[str]
    language: str                  # "English", "Hindi", "Hinglish / Code-Mixed"
    query_type: str                # "FACTUAL", "EXPLANATORY", "EXACT_LOOKUP", "AYURVEDA_IP", "MULTILINGUAL", "CROSS_DOMAIN", "OUT_OF_SCOPE", "INSUFFICIENT_EVIDENCE"
    query_category: str            # "PATENT", "AYURVEDA", "TRADITIONAL_KNOWLEDGE", "BIODIVERSITY", "INTERNATIONAL_IP", "CROSS_DOMAIN", "OUT_OF_SCOPE"
    scope_status: str              # "IN_SCOPE", "OUT_OF_SCOPE", "AMBIGUOUS"
    scope_confidence: float        # Confidence score [0.0, 1.0]
    scope_reason: str              # Explanation for scope decision
    jurisdiction: str              # "India", "WIPO/PCT", "EPO", "International", "Unknown"
    domains: List[str]             # List of recognized domains
    exact_identifiers: List[str]   # Specific statutory sections, rules, articles, or patent numbers
    metadata_filters: Optional[Dict[str, Any]]

    # 2. Retrieval & Reranking
    retrieval_performed: bool
    retrieval_called: bool
    reranking_called: bool
    retrieval_candidates: List[Dict[str, Any]]
    reranked_candidates: List[Dict[str, Any]]
    selected_evidence: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    formatted_evidence: str
    evidence_map: Dict[str, Dict[str, Any]]
    detected_conflicts: List[Dict[str, Any]]
    retrieval_attempt: int

    # 3. Evidence Sufficiency Decision
    evidence_sufficient: bool
    evidence_sufficiency_reason: str

    # 4. Generation & Post-Processing
    generation_called: bool
    draft_answer: str
    generated_answer: str
    claims: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    generation_attempt: int

    # 5. Citation Validation & Validation Loop
    citation_validation: Dict[str, Any]
    validation_status: str         # "VALID", "INVALID", "RETRY_GENERATION", "REFUSAL"
    validation_feedback: Optional[str]
    failure_reason: Optional[str]
    refusal_reason: Optional[str]

    # 6. Final Output & Observability
    final_answer: str
    final_answer_type: str         # "GROUNDED_ANSWER", "SAFE_REFUSAL", "INSUFFICIENT_EVIDENCE"
    is_refusal: bool
    is_valid: bool
    node_latencies_ms: Dict[str, float]
    total_latency_ms: float
    execution_trace: List[Dict[str, Any]]
