"""
Query-Evidence Validator for IP-SAKTI Sahayak.

Validates that retrieved evidence actually answers the user's requested question.
This is SEPARATE from claim-level citation validation.

Core P0 component preventing: "Well-cited wrong answers"
Example: Query asks for Section 3(p), evidence is Section 4, but answer is
well-supported by Section 4. This validator BLOCKS such answers.
"""

from typing import Dict, List, Any, Optional, NamedTuple, Tuple
import re
import logging

from src.retrieval.legal_identifier_parser import parse as parse_legal_identifier
from src.retrieval.legal_identifier_matcher import ExactProvisionMatcher
from src.retrieval.document_registry import get_document_registry

logger = logging.getLogger(__name__)


class ValidationResult(NamedTuple):
    """Result of query-evidence alignment validation."""
    query_aligned: bool
    alignment_score: float  # 0.0 to 1.0
    alignment_reason: str
    
    # What was requested
    requested_intent: str  # "EXACT_LOOKUP", "CURRENT_FEE", "EXPLANATION", etc.
    requested_document: Optional[str]
    requested_provision: Optional[str]
    required_evidence_targets: List[str]
    
    # What was provided
    provided_documents: List[str]
    provided_provisions: List[str]
    evidence_coverage: Dict[str, bool]  # target -> found
    
    # Missing pieces
    missing_targets: List[str]
    missing_evidence_types: List[str]
    
    # Recommendation
    allow_answer: bool
    recommendation: str


class QueryEvidenceValidator:
    """
    Validates that evidence actually answers the query.
    
    Distinguishes between:
    1. Claim Support: Does evidence support the generated claim? (citation validation)
    2. Query Alignment: Does evidence answer the requested question? (this validator)
    
    Both must pass for a strong answer.
    """
    
    def __init__(self):
        self.registry = get_document_registry()
        self.matcher = ExactProvisionMatcher()
    
    def validate(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
        parsed_intent: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate that evidence aligns with query intent and requested provision.
        
        Args:
            query: Original user query
            evidence_chunks: Retrieved evidence (from retrieval/reranking)
            parsed_intent: Pre-parsed intent dict (optional, will parse if None)
            
        Returns:
            ValidationResult with alignment decision
        """
        
        # Parse query if not provided
        if parsed_intent is None:
            parsed_intent = parse_legal_identifier(query)
        
        # Detect query intent
        intent = self._detect_intent(query, parsed_intent)
        
        # Determine what evidence is required
        required_targets = self._get_required_evidence_targets(query, intent, parsed_intent)
        
        # Extract what was provided
        provided_docs, provided_provisions = self._extract_evidence_content(evidence_chunks)
        
        # Check coverage
        coverage, missing = self._check_evidence_coverage(
            required_targets, evidence_chunks, parsed_intent
        )
        
        # Make alignment decision
        aligned, score, reason = self._make_alignment_decision(
            intent, required_targets, coverage, missing, evidence_chunks
        )
        
        missing_types = [target for target, found in coverage.items() if not found]
        
        # Recommendation
        allow_answer = aligned and score > 0.60
        recommendation = self._get_recommendation(
            allow_answer, intent, missing, missing_types
        )
        
        return ValidationResult(
            query_aligned=aligned,
            alignment_score=score,
            alignment_reason=reason,
            requested_intent=intent,
            requested_document=parsed_intent.get("canonical_title"),
            requested_provision=parsed_intent.get("value"),
            required_evidence_targets=required_targets,
            provided_documents=list(set(provided_docs)),
            provided_provisions=list(set(provided_provisions)),
            evidence_coverage=coverage,
            missing_targets=missing,
            missing_evidence_types=missing_types,
            allow_answer=allow_answer,
            recommendation=recommendation,
        )
    
    @staticmethod
    def _detect_intent(query: str, parsed: Dict[str, Any]) -> str:
        """
        Detect what the user is actually asking for.
        
        Returns:
            "EXACT_LOOKUP", "EXPLANATION", "CURRENT_FEE", "COMPARISON", "DEFINITION", etc.
        """
        query_lower = query.lower()
        
        # Current-fee pattern
        if any(kw in query_lower for kw in [
            "current fee", "latest fee", "registration fee",
            "application fee", "renewal fee", "how much",
            "cost", "charge", "fee schedule"
        ]):
            return "CURRENT_FEE"
        
        # Exact legal lookup
        if parsed.get("type") and any(word in query_lower for word in [
            "what does", "what is", "state", "provision", "section", "article", "rule"
        ]):
            # Check if asking for text or explanation
            if any(word in query_lower for word in ["mean", "interpret", "explain"]):
                return "EXPLANATION"
            else:
                return "EXACT_LOOKUP"
        
        # Comparison pattern
        if any(word in query_lower for word in ["differ", "compare", "vs", "between", "contrast"]):
            return "COMPARISON"
        
        # Definition/meaning
        if any(word in query_lower for word in ["define", "definition", "what is"]):
            return "DEFINITION"
        
        # Applicability
        if any(word in query_lower for word in ["apply", "applicable", "applies to"]):
            return "APPLICABILITY"
        
        # Default: general information
        return "GENERAL_INFO"
    
    @staticmethod
    def _get_required_evidence_targets(
        query: str, intent: str, parsed: Dict[str, Any]
    ) -> List[str]:
        """
        Determine what evidence MUST be present for this query.
        
        Returns:
            List of required evidence targets (e.g., ["Patents Act Section 3(p)"])
        """
        targets = []
        
        if intent == "EXACT_LOOKUP" and parsed.get("type"):
            # For exact lookup, the exact provision is required
            doc = parsed.get("canonical_title") or parsed.get("document_hint")
            prov = parsed.get("value")
            prov_type = parsed.get("type")
            
            if doc and prov:
                targets.append(f"{doc}|{prov_type}:{prov}")
            elif prov:
                targets.append(f"{prov_type}:{prov}")
        
        elif intent == "CURRENT_FEE":
            # For fees, need official fee schedule or regulatory document
            targets.append("OFFICIAL_FEE_SCHEDULE")
            targets.append("REGULATORY_NOTIFICATION")
        
        elif intent == "COMPARISON":
            # Need at least 2 distinct provisions
            targets.append("MULTI_PROVISION_EVIDENCE")
        
        elif intent == "APPLICABILITY":
            # Need relevant statute/rule
            if parsed.get("canonical_title"):
                targets.append(f"STATUTE:{parsed.get('canonical_title')}")
        
        # Fallback: if we have parsed provision, require it
        if not targets and parsed.get("type"):
            targets.append(f"{parsed.get('type')}:{parsed.get('value')}")
        
        # If no specific requirement, general evidence is OK
        if not targets:
            targets.append("GENERAL_EVIDENCE")
        
        return targets
    
    @staticmethod
    def _extract_evidence_content(
        chunks: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Extract document and provision info from evidence chunks."""
        docs = []
        provisions = []
        
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            
            # Extract document
            doc = meta.get("document")
            if doc:
                docs.append(doc)
            
            # Extract provision
            section = meta.get("section")
            article = meta.get("article")
            rule = meta.get("rule")
            
            if section:
                clause = meta.get("clause")
                prov = f"Section {section}({clause})" if clause else f"Section {section}"
                provisions.append(prov)
            elif article:
                provisions.append(f"Article {article}")
            elif rule:
                provisions.append(f"Rule {rule}")
        
        return docs, provisions
    
    def _check_evidence_coverage(
        self,
        required_targets: List[str],
        chunks: List[Dict[str, Any]],
        parsed: Dict[str, Any],
    ) -> Tuple[Dict[str, bool], List[str]]:
        """
        Check if evidence covers required targets.
        
        Returns:
            (coverage_dict, missing_list)
        """
        coverage = {}
        missing = []
        
        for target in required_targets:
            found = self._target_found_in_chunks(target, chunks, parsed)
            coverage[target] = found
            if not found:
                missing.append(target)
        
        return coverage, missing
    
    def _target_found_in_chunks(
        self, target: str, chunks: List[Dict[str, Any]], parsed: Dict[str, Any]
    ) -> bool:
        """Check if a required target is satisfied by any chunk."""
        
        # Special targets
        if target == "OFFICIAL_FEE_SCHEDULE":
            # Look for fee schedule indicators
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                text = chunk.get("text", "").lower()
                
                if any(kw in text for kw in ["fee", "schedule", "rate", "cost"]):
                    if meta.get("source_tier") in [1, 2]:  # Authoritative source
                        return True
            return False
        
        if target == "REGULATORY_NOTIFICATION":
            # Look for official notification
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                if meta.get("source_tier") in [1, 2]:
                    return True
            return False
        
        if target == "GENERAL_EVIDENCE":
            return len(chunks) > 0
        
        if target == "MULTI_PROVISION_EVIDENCE":
            # Need at least 2 distinct provisions
            provisions = set()
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                prov = f"{meta.get('section')}{meta.get('article')}{meta.get('rule')}"
                if prov != "NoneNoneNone":
                    provisions.add(prov)
            return len(provisions) >= 2
        
        if target.startswith("STATUTE:"):
            doc_name = target.split(":", 1)[1]
            canonical = self.registry.get_canonical_title(doc_name)
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                chunk_doc = meta.get("document")
                if chunk_doc:
                    chunk_canonical = self.registry.get_canonical_title(chunk_doc)
                    if chunk_canonical == canonical:
                        return True
            return False
        
        # Specific provision target: "Patents Act|section:3(p)"
        if "|" in target:
            doc_part, prov_part = target.split("|", 1)
            canonical_doc = self.registry.get_canonical_title(doc_part)
            
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                chunk_doc = meta.get("document")
                
                if chunk_doc:
                    chunk_canonical = self.registry.get_canonical_title(chunk_doc)
                    if chunk_canonical == canonical_doc:
                        # Document matches, now check provision
                        if self._provision_in_chunk(prov_part, chunk, parsed):
                            return True
            return False
        
        # Simple provision target: "section:3(p)"
        for chunk in chunks:
            if self._provision_in_chunk(target, chunk, parsed):
                return True
        
        return False
    
    def _provision_in_chunk(
        self, target: str, chunk: Dict[str, Any], parsed: Dict[str, Any]
    ) -> bool:
        """Check if a provision is in a chunk."""
        meta = chunk.get("metadata", {})
        
        # Use matcher for structured comparison
        result = self.matcher.match(chunk, parsed)
        return result.exact_match
    
    @staticmethod
    def _make_alignment_decision(
        intent: str,
        required_targets: List[str],
        coverage: Dict[str, bool],
        missing: List[str],
        chunks: List[Dict[str, Any]],
    ) -> Tuple[bool, float, str]:
        """
        Make alignment decision based on coverage.
        
        Returns:
            (aligned: bool, score: float, reason: str)
        """
        
        # For exact lookups and current fees, missing required targets = no alignment
        if intent in ("EXACT_LOOKUP", "CURRENT_FEE"):
            if missing:
                return False, 0.0, f"Missing required evidence: {', '.join(missing)}"
        
        # For other intents, high coverage = aligned
        if coverage:
            found = sum(1 for v in coverage.values() if v)
            total = len(coverage)
            score = found / total if total > 0 else 0.0
            
            if score >= 0.80:
                return True, score, "Evidence covers required targets"
            elif score >= 0.50:
                return True, score, f"Partial coverage ({found}/{total} targets)"
            else:
                return False, score, f"Insufficient coverage ({found}/{total} targets)"
        
        # No coverage data but have evidence
        if chunks:
            return True, 0.70, "Evidence available but exact match not confirmed"
        
        return False, 0.0, "No evidence provided"
    
    @staticmethod
    def _get_recommendation(
        allow_answer: bool, intent: str, missing: List[str], missing_types: List[str]
    ) -> str:
        """Generate actionable recommendation."""
        
        if allow_answer:
            return "Proceed with answer generation"
        
        if intent == "EXACT_LOOKUP" and missing:
            return f"Cannot answer: exact provision not found. Missing: {', '.join(missing)}"
        
        if intent == "CURRENT_FEE" and missing:
            return "Cannot answer: official fee schedule not in knowledge base. Provide safe refusal."
        
        if missing_types:
            return f"Insufficient evidence for {intent}. Missing: {', '.join(missing_types)}. Provide safe refusal or retry retrieval."
        
        return "Insufficient evidence alignment. Provide safe refusal."


def create_validator() -> QueryEvidenceValidator:
    """Factory function to create validator instance."""
    return QueryEvidenceValidator()
