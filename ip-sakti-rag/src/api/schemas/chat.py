"""Chat Request & Response Pydantic Schemas for IP-SAKTI Sahayak API."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat query payload from frontend."""
    query: str = Field(..., min_length=1, max_length=2000, description="The legal or Ayurvedic IP question.")
    language: str = Field(default="auto", description="Query language ('auto', 'en', 'hi', or 'hinglish').")


class CitationItem(BaseModel):
    """Structured citation item linked to retrieved authoritative evidence."""
    citation_id: str = Field(..., description="Citation marker id, e.g. 'C1' or '1'")
    evidence_id: str = Field(..., description="Internal evidence id, e.g. 'E1'")
    document: str = Field(..., description="Authoritative document name")
    document_id: Optional[str] = Field(None, description="Document identifier")
    section: Optional[str] = Field(None, description="Section, rule, or article")
    page: Optional[int] = Field(None, description="Document page number")
    jurisdiction: Optional[str] = Field("India", description="Applicable legal jurisdiction")
    domain: Optional[str] = Field(None, description="Legal/AYUSH domain")
    source_tier: Optional[str] = Field("Tier 1", description="Source authority tier (Tier 1 Primary, Tier 2 Guideline, Tier 3 Institutional)")
    excerpt: Optional[str] = Field(None, description="Verbatim text excerpt from authoritative chunk")


class ValidationInfo(BaseModel):
    """Citation and claim validation status."""
    status: str = Field(..., description="Validation status: 'VALID', 'INVALID', or 'REFUSAL'")
    is_valid: bool = Field(True, description="Whether all substantive claims are supported")
    total_claims: int = Field(0, description="Total substantive claims extracted")
    supported_claims: int = Field(0, description="Number of verified supported claims")
    claim_support_rate: float = Field(1.0, description="Ratio of supported claims (0.0 to 1.0)")
    flagged_issues_count: int = Field(0, description="Number of detected unsupported claims or hallucinated citations")


class QueryMetadata(BaseModel):
    """Processing and observability metadata."""
    latency_ms: float = Field(..., description="Total end-to-end latency in milliseconds")
    generation_attempts: int = Field(1, description="Number of LLM generation attempts")
    retrieval_attempts: int = Field(1, description="Number of retrieval attempts")
    node_latencies_ms: Dict[str, float] = Field(default_factory=dict, description="Execution time per graph node")


class ChatResponse(BaseModel):
    """Complete structured response payload for the frontend."""
    answer: str = Field(..., description="Grounded answer text with inline citation tags [1], [2]")
    language: str = Field(..., description="Detected query language")
    query_type: str = Field(..., description="Query classification type")
    jurisdiction: Optional[str] = Field("India", description="Detected jurisdiction")
    domains: List[str] = Field(default_factory=list, description="List of recognized legal/AYUSH domains")
    citations: List[CitationItem] = Field(default_factory=list, description="Structured citation objects")
    is_refusal: bool = Field(False, description="Whether this response is a safe refusal")
    validation: ValidationInfo = Field(..., description="Claim and citation verification summary")
    metadata: QueryMetadata = Field(..., description="Observability and latency metrics")


class ErrorResponse(BaseModel):
    """Standardized user-friendly error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional context if safe")
    status_code: int = Field(500, description="HTTP status code")
