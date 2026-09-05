"""
Strict Legal Identifier Matcher for IP-SHAKTI Sahayak.

Enforces exact structural validation for legal provisions.
NEVER reduces "Section 3(p)" to "(p)".

This is the core P0 component that prevents wrong-provision retrieval.
"""

from typing import Dict, Any, Optional, NamedTuple
import re
import logging

from src.retrieval.document_registry import get_document_registry

logger = logging.getLogger(__name__)


class MatchResult(NamedTuple):
    """Result of exact provision matching."""
    exact_match: bool
    match_confidence: float  # 0.0 to 1.0
    match_reason: str
    
    # What was requested
    requested_document: Optional[str]
    requested_section: Optional[str]
    requested_clause: Optional[str]
    requested_full_provision: Optional[str]
    
    # What was found
    candidate_document: Optional[str]
    candidate_section: Optional[str]
    candidate_clause: Optional[str]
    candidate_full_provision: Optional[str]
    
    # Mismatch details
    document_mismatch: bool
    section_mismatch: bool
    clause_mismatch: bool


class ExactProvisionMatcher:
    """
    Strict matcher for legal identifiers.
    
    Validates that retrieved evidence matches the EXACT legal provision requested,
    not just semantic similarity or substring matches.
    
    Example failures that this prevents:
    - Query: Section 3(p) → Candidate: Section 4(p) ✗ (section mismatch)
    - Query: Section 3(p) → Candidate: "(p)" without parent ✗ (no parent section proof)
    - Query: Patents Act Section 25 → Candidate: Trade Marks Act Section 25 ✗ (document mismatch)
    """
    
    def __init__(self):
        self.registry = get_document_registry()
    
    def match(
        self,
        chunk: Dict[str, Any],
        parsed_identifier: Dict[str, Any],
        requested_document: Optional[str] = None,
    ) -> MatchResult:
        """
        Check if a chunk exactly matches the requested legal identifier.
        
        Args:
            chunk: Retrieved document chunk with metadata
            parsed_identifier: Parsed query identifier (from legal_identifier_parser)
            requested_document: Canonical document title (if known)
            
        Returns:
            MatchResult with exact_match flag and detailed mismatch info
        """
        
        # Extract what was requested
        requested_type = parsed_identifier.get("type")  # "section", "article", "rule", "patent"
        requested_value = parsed_identifier.get("value")  # "3(p)", "43bis", etc.
        
        # Parse the requested provision
        if not requested_type or not requested_value:
            return MatchResult(
                exact_match=False,
                match_confidence=0.0,
                match_reason="No legal identifier in query",
                requested_document=None,
                requested_section=None,
                requested_clause=None,
                requested_full_provision=None,
                candidate_document=chunk.get("metadata", {}).get("document"),
                candidate_section=chunk.get("metadata", {}).get("section"),
                candidate_clause=chunk.get("metadata", {}).get("clause"),
                candidate_full_provision=None,
                document_mismatch=False,
                section_mismatch=False,
                clause_mismatch=False,
            )
        
        # Normalize requested provision
        requested_section, requested_clause = self._parse_full_provision(
            requested_value, requested_type
        )
        requested_full = self._construct_full_provision(
            requested_type, requested_section, requested_clause
        )
        
        # Extract candidate metadata
        meta = chunk.get("metadata", {})
        candidate_document = meta.get("document")
        candidate_section = meta.get("section")
        candidate_clause = meta.get("clause")
        candidate_text = chunk.get("text", "")
        
        # Construct candidate provision
        candidate_full = self._construct_full_provision(
            requested_type, candidate_section, candidate_clause
        )
        
        # ========== VALIDATION CHECKS ==========
        
        # 1. Document check (if document was specified in query)
        document_match = self._check_document_match(
            requested_document, candidate_document
        )
        
        # 2. Exact structural match
        structural_match = self._check_structural_match(
            requested_type,
            requested_section,
            requested_clause,
            candidate_section,
            candidate_clause,
        )
        
        # 3. Text-based fallback (for legacy/cross-boundary chunks)
        text_match = self._check_text_match(
            requested_type, requested_section, requested_clause, candidate_text
        )
        
        # Determine overall match
        if requested_document:
            # Document was specified: must match both document AND provision
            overall_match = document_match and (structural_match or text_match)
            reason = self._explain_result(
                document_match, structural_match, text_match, requested_document, candidate_document
            )
        else:
            # No document specified: just check provision
            overall_match = structural_match or text_match
            reason = self._explain_result(
                True, structural_match, text_match, None, None
            )
        
        confidence = self._calculate_confidence(
            requested_type, structural_match, text_match, document_match
        )
        
        return MatchResult(
            exact_match=overall_match,
            match_confidence=confidence,
            match_reason=reason,
            requested_document=requested_document,
            requested_section=requested_section,
            requested_clause=requested_clause,
            requested_full_provision=requested_full,
            candidate_document=candidate_document,
            candidate_section=candidate_section,
            candidate_clause=candidate_clause,
            candidate_full_provision=candidate_full,
            document_mismatch=not document_match,
            section_mismatch=(requested_section != candidate_section),
            clause_mismatch=(requested_clause != candidate_clause),
        )
    
    @staticmethod
    def _parse_full_provision(
        value: str, provision_type: str
    ) -> tuple:
        """
        Parse a provision value into components.
        
        Examples:
        - "3(p)" -> ("3", "p")
        - "43bis" -> ("43bis", None)
        - "3" -> ("3", None)
        
        Returns:
            (section/article/rule, clause) tuple
        """
        # Normalize: remove spaces, lowercase
        value = re.sub(r"\s+", "", str(value).lower()).strip()
        
        if provision_type == "section":
            # Match: 3(p), 3, 3a, 3a(1), etc.
            match = re.match(r"^(\d+[a-z]?)(?:\(([a-z0-9]+)\))?$", value)
            if match:
                return match.group(1), match.group(2)
            return value, None
        
        elif provision_type == "article":
            # Match: 3, 27, 3a, etc.
            match = re.match(r"^(\d+[a-z]?)(?:\(([a-z0-9]+)\))?$", value)
            if match:
                return match.group(1), match.group(2)
            return value, None
        
        elif provision_type == "rule":
            # Match: 43bis, 43, 43ter, etc.
            return value, None
        
        elif provision_type == "patent":
            # Patent numbers are atomic
            return value, None
        
        return value, None
    
    @staticmethod
    def _construct_full_provision(
        provision_type: str, section: Optional[str], clause: Optional[str]
    ) -> str:
        """
        Construct canonical full provision string for comparison.
        
        Examples:
        - ("section", "3", "p") -> "section:3:p"
        - ("section", "3", None) -> "section:3"
        - ("rule", "43bis", None) -> "rule:43bis"
        """
        if not section:
            return ""
        
        parts = [provision_type, section]
        if clause:
            parts.append(clause)
        
        return ":".join(parts)
    
    def _check_document_match(
        self, requested_canonical: Optional[str], candidate_document: Optional[str]
    ) -> bool:
        """
        Check if candidate document matches requested document.
        
        Args:
            requested_canonical: Canonical title from query
            candidate_document: Document from chunk metadata
            
        Returns:
            True if match, False if mismatch, True if no request
        """
        if not requested_canonical:
            # No document specified in query
            return True
        
        if not candidate_document:
            # Candidate has no document metadata
            return False
        
        # Normalize candidate and compare
        candidate_canonical = self.registry.get_canonical_title(candidate_document)
        return candidate_canonical == requested_canonical
    
    @staticmethod
    def _check_structural_match(
        provision_type: str,
        requested_section: Optional[str],
        requested_clause: Optional[str],
        candidate_section: Optional[str],
        candidate_clause: Optional[str],
    ) -> bool:
        """
        Check if metadata fields match exactly.
        
        This is the primary exact match, not fuzzy/substring.
        """
        if provision_type == "section":
            # Must match section and clause (if clause was requested)
            section_match = (
                requested_section and candidate_section and
                ExactProvisionMatcher._normalize_identifier(requested_section) ==
                ExactProvisionMatcher._normalize_identifier(candidate_section)
            )
            
            if requested_clause:
                clause_match = (
                    candidate_clause and
                    ExactProvisionMatcher._normalize_identifier(requested_clause) ==
                    ExactProvisionMatcher._normalize_identifier(candidate_clause)
                )
                return section_match and clause_match
            else:
                return section_match
        
        elif provision_type == "article":
            return (
                requested_section and candidate_section and
                ExactProvisionMatcher._normalize_identifier(requested_section) ==
                ExactProvisionMatcher._normalize_identifier(candidate_section)
            )
        
        elif provision_type == "rule":
            return (
                requested_section and candidate_section and
                ExactProvisionMatcher._normalize_identifier(requested_section) ==
                ExactProvisionMatcher._normalize_identifier(candidate_section)
            )
        
        elif provision_type == "patent":
            return (
                requested_section and candidate_section and
                ExactProvisionMatcher._normalize_identifier(requested_section) ==
                ExactProvisionMatcher._normalize_identifier(candidate_section)
            )
        
        return False
    
    @staticmethod
    def _check_text_match(
        provision_type: str,
        requested_section: Optional[str],
        requested_clause: Optional[str],
        candidate_text: str,
    ) -> bool:
        """
        Fallback: check if provision is explicitly mentioned in chunk text.
        
        Useful for legacy chunks that cross section boundaries but preserve
        the actual section in the text body.
        
        This is less precise than structural match but helps with edge cases.
        """
        if not candidate_text or not requested_section:
            return False
        
        text_lower = candidate_text.lower()
        
        if provision_type == "section":
            # Look for "Section 3(p)" or "S. 3(p)" pattern in text
            if requested_clause:
                # Full: section 3(p)
                patterns = [
                    rf"(?:section|sec\.?|s\.?)\s*{re.escape(requested_section)}\s*\(\s*{re.escape(requested_clause)}\s*\)",
                    rf"\({re.escape(requested_clause)}\)\s*(?:of\s+)?(?:section|sec\.?|s\.?)\s*{re.escape(requested_section)}",
                ]
            else:
                # Just section: section 3
                patterns = [
                    rf"(?:section|sec\.?|s\.?)\s*{re.escape(requested_section)}(?:\s|[.()\[\]]|$)",
                ]
            
            return any(re.search(pat, text_lower, re.IGNORECASE) for pat in patterns)
        
        elif provision_type == "article":
            patterns = [
                rf"(?:article|art\.?)\s*{re.escape(requested_section)}(?:\s|[.()\[\]]|$)",
            ]
            return any(re.search(pat, text_lower, re.IGNORECASE) for pat in patterns)
        
        elif provision_type == "rule":
            patterns = [
                rf"(?:rule|r\.?)\s*{re.escape(requested_section)}(?:\s|[.()\[\]]|$)",
            ]
            return any(re.search(pat, text_lower, re.IGNORECASE) for pat in patterns)
        
        return False
    
    @staticmethod
    def _normalize_identifier(value: Optional[str]) -> str:
        """Normalize identifier for comparison."""
        if not value:
            return ""
        return re.sub(r"\s+", "", str(value).lower()).strip()
    
    @staticmethod
    def _explain_result(
        document_match: bool,
        structural_match: bool,
        text_match: bool,
        requested_doc: Optional[str],
        candidate_doc: Optional[str],
    ) -> str:
        """Generate human-readable explanation of match result."""
        reasons = []
        
        if not document_match:
            reasons.append(
                f"Document mismatch: requested {requested_doc}, found {candidate_doc}"
            )
        
        if not structural_match and not text_match:
            reasons.append("Provision metadata does not match requested identifier")
        elif structural_match:
            reasons.append("Structural metadata match")
        elif text_match:
            reasons.append("Text-based match (legacy chunk)")
        
        return " | ".join(reasons) if reasons else "No match"
    
    @staticmethod
    def _calculate_confidence(
        provision_type: str,
        structural_match: bool,
        text_match: bool,
        document_match: bool,
    ) -> float:
        """
        Calculate match confidence score.
        
        Returns:
            0.0 to 1.0 where 1.0 is highest confidence
        """
        if not structural_match and not text_match:
            return 0.0
        
        confidence = 0.0
        
        # Structural match is most reliable
        if structural_match:
            confidence = 0.95
        # Text match is decent fallback
        elif text_match:
            confidence = 0.70
        
        # Document match improves confidence
        if document_match:
            confidence += 0.05
        
        return min(1.0, confidence)


def create_exact_matcher() -> ExactProvisionMatcher:
    """Factory function to create matcher instance."""
    return ExactProvisionMatcher()
