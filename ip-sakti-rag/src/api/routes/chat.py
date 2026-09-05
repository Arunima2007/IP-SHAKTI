"""Chat endpoint for IP-SAKTI Sahayak API."""
import time
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Request, status
from src.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CitationItem,
    ValidationInfo,
    QueryMetadata,
    ErrorResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])


def determine_source_tier(doc_name: str) -> str:
    """Classifies source authority tier based on legal document type."""
    name_lower = (doc_name or "").lower()
    if any(k in name_lower for k in ["act", "treaty", "constitution", "1970", "2002", "1940", "1957", "2000", "1999"]):
        return "Tier 1: Primary Statute"
    elif any(k in name_lower for k in ["guideline", "regulation", "rules", "gazette", "compendium", "directive"]):
        return "Tier 2: Official Guideline"
    else:
        return "Tier 3: Institutional Source"


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def chat_query(payload: ChatRequest, request: Request):
    """
    Executes the full LangGraph orchestration pipeline for a user query.
    Returns grounded answer, structured citations with verbatim evidence excerpts,
    domain awareness badges, and claim validation metrics.
    """
    t_start = time.perf_counter()
    raw_query = payload.query.strip()
    
    if not raw_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query string cannot be empty."
        )

    # Access pre-initialized LangGraph coordinator from app state
    coordinator = getattr(request.app.state, "coordinator", None)
    if coordinator is None:
        from src.graph.graph import IPSAKTILangGraphCoordinator
        coordinator = IPSAKTILangGraphCoordinator()
        request.app.state.coordinator = coordinator

    try:
        # Execute LangGraph StateGraph pipeline
        result = coordinator.process_query(raw_query)
        
        # 1. Answer and Basic Metadata
        final_answer = result.get("final_answer", "")
        detected_language = result.get("language", "English")
        query_type = result.get("query_type", "FACTUAL")
        jurisdiction = result.get("jurisdiction", "India")
        domains = result.get("domains", [])
        is_refusal = result.get("is_refusal", False)
        
        # 2. Build Structured Citation Items with Verbatim Excerpts
        evidence_map = result.get("evidence_map", {})
        raw_citations = result.get("citations", [])
        citation_items: List[CitationItem] = []
        
        for idx, cit in enumerate(raw_citations, 1):
            if isinstance(cit, dict):
                cit_id = str(cit.get("citation_id") or cit.get("id") or idx)
                ev_id = cit.get("evidence_id") or f"E{idx}"
                doc_name = cit.get("document") or "Authoritative Source"
                section_val = cit.get("section") or cit.get("rule") or cit.get("article")
                page_val = cit.get("page")
                tier = cit.get("source_tier") or determine_source_tier(doc_name)
                
                # Fetch verbatim excerpt from evidence map if available
                excerpt = None
                if ev_id in evidence_map:
                    ev_chunk = evidence_map[ev_id]
                    excerpt = ev_chunk.get("text", "")
                elif "excerpt" in cit:
                    excerpt = cit["excerpt"]
                    
                item = CitationItem(
                    citation_id=cit_id,
                    evidence_id=ev_id,
                    document=doc_name,
                    document_id=cit.get("document_id"),
                    section=str(section_val) if section_val else None,
                    page=int(page_val) if page_val is not None else None,
                    jurisdiction=cit.get("jurisdiction", jurisdiction),
                    domain=cit.get("domain"),
                    source_tier=tier,
                    excerpt=excerpt[:600] if excerpt else None
                )
                citation_items.append(item)

        # 3. Validation Summary
        val_status = result.get("validation_status", "VALID")
        val_result = result.get("citation_validation", {})
        val_metrics = val_result.get("metrics", {})
        claims_list = result.get("claims", [])
        
        total_claims = val_metrics.get("total_claims", len(claims_list))
        supported_claims = val_metrics.get("supported_claims", len([c for c in claims_list if c.get("supported")]))
        claim_support_rate = val_metrics.get("claim_support_rate", 1.0 if total_claims == 0 else supported_claims / max(1, total_claims))
        flagged_count = len(val_result.get("flagged_issues", []))
        
        validation_obj = ValidationInfo(
            status=val_status,
            is_valid=result.get("is_valid", True),
            total_claims=total_claims,
            supported_claims=supported_claims,
            claim_support_rate=round(claim_support_rate, 4),
            flagged_issues_count=flagged_count
        )

        # 4. Latency & Observability Metadata
        total_latency = round((time.perf_counter() - t_start) * 1000, 2)
        node_latencies = result.get("node_latencies_ms", {})
        
        metadata_obj = QueryMetadata(
            latency_ms=total_latency,
            generation_attempts=result.get("generation_attempt", 1),
            retrieval_attempts=result.get("retrieval_attempt", 1),
            node_latencies_ms=node_latencies
        )

        return ChatResponse(
            answer=final_answer,
            language=detected_language,
            query_type=query_type,
            jurisdiction=jurisdiction,
            domains=domains,
            citations=citation_items,
            is_refusal=is_refusal,
            validation=validation_obj,
            metadata=metadata_obj
        )

    except Exception as e:
        logger.error(f"Error processing chat query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your legal query through the orchestration pipeline."
        )
