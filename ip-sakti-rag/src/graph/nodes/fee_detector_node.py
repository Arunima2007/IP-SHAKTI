"""
Current-Fee Detector Node for IP-SHAKTI Sahayak LangGraph.

Classifies current-fee and current-regulation queries.
Routes them to specialized handling that REFUSES to answer without
official fee schedules or regulatory documents.

P0 component preventing: "Section 25 hallucinations for fee questions"
"""

from typing import Dict, Any, List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


class FeeDetectorNode:
    """
    Detects current-fee and current-regulation queries.
    
    Distinguishes between:
    - Current-fee queries (must have official fee schedule)
    - Historical/example fee queries (can be answered from provisions)
    - General regulatory queries (normal retrieval)
    """
    
    # Keywords that indicate current-fee intent
    CURRENT_FEE_KEYWORDS = {
        # Direct fee/cost terms
        "fee", "fees", "cost", "charge", "charges", "price", "pricing",
        "rate", "rates", "tariff", "tariffs",
        
        # Modifiers indicating NOW/CURRENT
        "current", "latest", "present", "now", "today", "2024", "2025", "2026",
        "today's", "present-day", "contemporary", "up-to-date",
        
        # Registration/application specific
        "registration fee", "application fee", "filing fee", "renewal fee",
        "provisional fee", "final fee", "examination fee",
        
        # Amount/calculation
        "how much", "amount", "cost", "expensive", "pay", "payment",
        
        # Official terms
        "official fee", "official charge", "fee schedule", "fee table",
        "tariff schedule", "official rate",
    }
    
    # Keywords indicating HISTORICAL/EXAMPLE fees (not current)
    HISTORICAL_FEE_KEYWORDS = {
        "was", "were", "previously", "earlier", "historical", "old",
        "before", "ago", "past", "then", "at that time",
        "example", "for example", "e.g.", "such as", "like",
        "supposing", "hypothetical", "if", "assume",
    }
    
    # Source types that are authoritative for fees
    AUTHORITATIVE_FEE_SOURCES = {
        "official fee schedule",
        "fee notification",
        "regulatory notification",
        "government notification",
        "official gazette",
        "official order",
        "official circular",
    }
    
    def __init__(self):
        """Initialize detector."""
        pass
    
    def detect(self, query: str) -> Tuple[bool, Optional[str], float]:
        """
        Detect if query is asking for current fees/regulations.
        
        Args:
            query: User query string
            
        Returns:
            (is_fee_query: bool, fee_type: Optional[str], confidence: float)
            
        fee_type values:
        - "CURRENT_FEE": Current fee amount (requires official schedule)
        - "FEE_SCHEDULE": Fee table/schedule (requires official schedule)
        - "CURRENT_REGULATION": Current regulatory requirement (needs official source)
        - "HISTORICAL_FEE": Historical/example fee (can answer from text)
        - "GENERAL": Not a fee query
        """
        
        query_lower = query.lower()
        
        # Check for historical/example fee indicators first
        historical_score = self._calculate_keyword_match(
            query_lower, self.HISTORICAL_FEE_KEYWORDS
        )
        
        # Check for current-fee indicators
        fee_score = self._calculate_keyword_match(
            query_lower, self.CURRENT_FEE_KEYWORDS
        )
        
        # If historical indicators present, reduce fee score
        if historical_score > 0.3:
            fee_score *= 0.5  # Reduce confidence for historical queries
            
            if fee_score > 0.4:
                return True, "HISTORICAL_FEE", fee_score
            else:
                return False, "GENERAL", 0.0
        
        # Determine fee type based on query patterns
        if fee_score > 0.5:
            fee_type = self._determine_fee_type(query_lower)
            return True, fee_type, fee_score
        
        return False, "GENERAL", 0.0
    
    @staticmethod
    def _calculate_keyword_match(text: str, keywords: set) -> float:
        """
        Calculate how well query matches keyword set.
        
        Returns:
            0.0 to 1.0 confidence score
        """
        if not keywords:
            return 0.0
        
        matches = sum(1 for kw in keywords if kw in text)
        score = matches / len(keywords)
        
        # Boost for exact phrase matches
        for kw in keywords:
            if f" {kw} " in f" {text} ":
                score += 0.1
        
        return min(1.0, score)
    
    @staticmethod
    def _determine_fee_type(query_lower: str) -> str:
        """Determine specific type of fee query."""
        
        # Current fee amount
        if any(phrase in query_lower for phrase in [
            "what is the fee", "what's the fee", "how much is the fee",
            "cost of", "price of", "charge for", "fee for"
        ]):
            return "CURRENT_FEE"
        
        # Fee schedule/table
        if any(phrase in query_lower for phrase in [
            "fee schedule", "fee table", "tariff schedule", "fee structure"
        ]):
            return "FEE_SCHEDULE"
        
        # Renewal/recurring fees
        if any(phrase in query_lower for phrase in [
            "renewal fee", "annual fee", "maintenance fee"
        ]):
            return "CURRENT_FEE"
        
        # Regulatory requirements
        if any(phrase in query_lower for phrase in [
            "current requirement", "present rule", "existing regulation"
        ]):
            return "CURRENT_REGULATION"
        
        # Default
        return "CURRENT_FEE"
    
    @staticmethod
    def validate_fee_evidence(
        evidence_chunks: List[Dict[str, Any]],
        fee_type: str
    ) -> Tuple[bool, str]:
        """
        Validate that evidence is appropriate for fee queries.
        
        Args:
            evidence_chunks: Retrieved evidence
            fee_type: Type of fee query detected
            
        Returns:
            (is_valid: bool, reason: str)
        """
        
        if fee_type == "HISTORICAL_FEE":
            # Historical fees can be answered from any text mentioning fees
            for chunk in evidence_chunks:
                text = chunk.get("text", "").lower()
                if any(word in text for word in ["fee", "cost", "charge", "rate"]):
                    return True, "Historical fee reference found in text"
            return False, "No historical fee reference in evidence"
        
        if fee_type in ("CURRENT_FEE", "FEE_SCHEDULE"):
            # Current fees MUST come from official sources
            for chunk in evidence_chunks:
                meta = chunk.get("metadata", {})
                text = chunk.get("text", "").lower()
                
                # Check source tier (1-2 is authoritative)
                source_tier = meta.get("source_tier")
                if source_tier in (1, 2):
                    # Check if it's actually about fees
                    if any(keyword in text for keyword in [
                        "fee", "schedule", "rate", "tariff", "cost", "charge"
                    ]):
                        # Verify it's not just mentioning fees in passing
                        if FeeDetectorNode._is_dedicated_fee_content(text):
                            return True, "Official fee schedule found"
            
            return False, "No official current fee schedule in evidence"
        
        if fee_type == "CURRENT_REGULATION":
            # Current regulations need official source
            for chunk in evidence_chunks:
                meta = chunk.get("metadata", {})
                source_tier = meta.get("source_tier")
                
                if source_tier in (1, 2):
                    return True, "Official regulatory source found"
            
            return False, "No official regulatory source in evidence"
        
        return False, "Unable to validate evidence type"
    
    @staticmethod
    def _is_dedicated_fee_content(text: str) -> bool:
        """
        Check if text is primarily about fees (not just mentioning them).
        
        Prevents matching generic sections that happen to mention fees.
        """
        text_lower = text.lower()
        
        # Count fee-related sentences
        sentences = re.split(r'[.!?]+', text_lower)
        fee_sentences = sum(
            1 for s in sentences
            if any(kw in s for kw in ["fee", "charge", "rate", "cost", "tariff"])
        )
        
        # If more than 30% of sentences mention fees, it's dedicated content
        if len(sentences) > 0:
            ratio = fee_sentences / len(sentences)
            return ratio > 0.30
        
        return False
    
    @staticmethod
    def get_refusal_message(fee_type: str) -> str:
        """
        Generate appropriate refusal message for fee queries.
        
        Args:
            fee_type: Type of fee query
            
        Returns:
            Safe refusal message
        """
        
        if fee_type == "CURRENT_FEE":
            return (
                "I could not find the current official fee schedule in the available knowledge base, "
                "so I cannot reliably state the current fee amount. "
                "Please refer to the official IP India website or the relevant authority for current fee information."
            )
        
        if fee_type == "FEE_SCHEDULE":
            return (
                "The current official fee schedule is not available in my knowledge base. "
                "Please consult the official fee tables published by the IP office for the most current information."
            )
        
        if fee_type == "CURRENT_REGULATION":
            return (
                "I could not find sufficient authoritative evidence for the current regulatory requirement. "
                "Please consult official regulatory sources for current requirements."
            )
        
        return (
            "I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."
        )
    
    @staticmethod
    def should_bypass_generation(
        fee_type: str, evidence_valid: bool
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if answer generation should be bypassed due to fee query issues.
        
        Args:
            fee_type: Detected fee type
            evidence_valid: Whether evidence passed validation
            
        Returns:
            (bypass_generation: bool, refusal_message: Optional[str])
        """
        
        # Current fees ALWAYS require valid evidence
        if fee_type in ("CURRENT_FEE", "FEE_SCHEDULE", "CURRENT_REGULATION"):
            if not evidence_valid:
                refusal = FeeDetectorNode.get_refusal_message(fee_type)
                return True, refusal
        
        # Historical fees can proceed with any fee reference
        return False, None


# Graph node wrapper
def fee_detector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node that detects and routes fee queries.
    
    Updates state with:
    - is_fee_query: bool
    - fee_type: str (CURRENT_FEE, HISTORICAL_FEE, etc.)
    - fee_confidence: float
    - fee_evidence_valid: bool
    - should_skip_generation: bool
    - fee_refusal_reason: Optional[str]
    """
    detector = FeeDetectorNode()
    
    query = state.get("query", "")
    evidence = state.get("selected_evidence", [])
    
    # Detect fee query
    is_fee_query, fee_type, confidence = detector.detect(query)
    
    state["is_fee_query"] = is_fee_query
    state["fee_type"] = fee_type or "GENERAL"
    state["fee_confidence"] = confidence
    
    # Validate evidence if fee query detected
    if is_fee_query:
        evidence_valid, validation_reason = detector.validate_fee_evidence(
            evidence, fee_type
        )
        state["fee_evidence_valid"] = evidence_valid
        
        # Determine if we should skip generation
        skip_gen, refusal_msg = detector.should_bypass_generation(fee_type, evidence_valid)
        state["should_skip_fee_generation"] = skip_gen
        
        if skip_gen:
            state["fee_refusal_reason"] = refusal_msg
            logger.info(
                f"Fee query detected ({fee_type}) with invalid evidence. "
                f"Skipping generation: {refusal_msg[:100]}..."
            )
    else:
        state["fee_evidence_valid"] = True
        state["should_skip_fee_generation"] = False
        state["fee_refusal_reason"] = None
    
    return state
