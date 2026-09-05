"""
Legal Precision Benchmark for IP-SAKTI Sahayak.

Comprehensive regression test suite for legal correctness hardening.
Measures exact provision accuracy, document accuracy, query-evidence alignment,
and prevents regressions in the core fixes.

Run with: pytest tests/test_legal_precision.py -v
"""

import pytest
from typing import Dict, Any, List
import logging

from src.retrieval.document_registry import get_document_registry
from src.retrieval.legal_identifier_matcher import create_exact_matcher
from src.retrieval.legal_identifier_parser import parse as parse_legal_identifier
from src.generation.query_evidence_validator import create_validator
from src.graph.nodes.fee_detector_node import FeeDetectorNode

logger = logging.getLogger(__name__)


class TestDocumentRegistry:
    """Tests for canonical document registry."""
    
    def test_patents_act_normalization(self):
        """Test that Patents Act aliases normalize correctly."""
        registry = get_document_registry()
        
        # All these should map to the same canonical title
        aliases = [
            "Patents Act",
            "Patent Act 1970",
            "The Patents Act, 1970",
            "Indian Patents Act",
            "Patents Act, 1970",
        ]
        
        canonical = "The Patents Act, 1970"
        for alias in aliases:
            result = registry.get_canonical_title(alias)
            assert result == canonical, f"Failed to normalize '{alias}'"
    
    def test_trade_marks_act_normalization(self):
        """Test Trade Marks Act normalization."""
        registry = get_document_registry()
        
        aliases = [
            "Trade Marks Act",
            "Trademarks Act",
            "Trade Mark Act",
            "The Trade Marks Act, 1999",
        ]
        
        canonical = "The Trade Marks Act, 1999"
        for alias in aliases:
            result = registry.get_canonical_title(alias)
            assert result == canonical, f"Failed to normalize '{alias}'"
    
    def test_document_metadata_retrieval(self):
        """Test that document metadata is correctly retrieved."""
        registry = get_document_registry()
        
        canonical = "The Patents Act, 1970"
        assert registry.get_document_id(canonical) == "patent_act_1970"
        assert registry.get_tier(canonical) == 1
        assert registry.get_weight(canonical) == 1.0
        assert "Patents Act, 1970" in registry.get_label(canonical)
    
    def test_invalid_document_handling(self):
        """Test handling of unrecognized documents."""
        registry = get_document_registry()
        
        result = registry.get_canonical_title("Nonexistent Act 2050")
        assert result is None


class TestExactProvisionMatcher:
    """Tests for legal identifier matching."""
    
    def test_exact_section_match_success(self):
        """Test that exact section matches work correctly."""
        matcher = create_exact_matcher()
        
        # Mock chunk for Section 3(p) of Patents Act
        chunk = {
            "text": "3(p) This section defines traditional knowledge",
            "metadata": {
                "document": "The Patents Act, 1970",
                "section": "3",
                "clause": "p",
            }
        }
        
        # Query for Section 3(p)
        parsed = {
            "type": "section",
            "value": "3(p)",
            "canonical_title": "The Patents Act, 1970",
        }
        
        result = matcher.match(chunk, parsed, "The Patents Act, 1970")
        assert result.exact_match is True, f"Failed to match Section 3(p): {result.match_reason}"
    
    def test_exact_section_match_fails_wrong_section(self):
        """Test that Section 4(p) does NOT match query for Section 3(p)."""
        matcher = create_exact_matcher()
        
        # Chunk has Section 4(p)
        chunk = {
            "text": "4(p) This is a different section",
            "metadata": {
                "document": "The Patents Act, 1970",
                "section": "4",
                "clause": "p",
            }
        }
        
        # Query for Section 3(p)
        parsed = {
            "type": "section",
            "value": "3(p)",
            "canonical_title": "The Patents Act, 1970",
        }
        
        result = matcher.match(chunk, parsed, "The Patents Act, 1970")
        assert result.exact_match is False, "Incorrectly matched Section 4(p) as Section 3(p)"
        assert result.section_mismatch is True
    
    def test_exact_section_match_fails_wrong_document(self):
        """Test that Trade Marks Act Section 3 does NOT match Patents Act Section 3."""
        matcher = create_exact_matcher()
        
        # Chunk has Trade Marks Act Section 3
        chunk = {
            "text": "Section 3 of Trade Marks Act",
            "metadata": {
                "document": "The Trade Marks Act, 1999",
                "section": "3",
            }
        }
        
        # Query for Patents Act Section 3
        parsed = {
            "type": "section",
            "value": "3",
            "canonical_title": "The Patents Act, 1970",
        }
        
        result = matcher.match(chunk, parsed, "The Patents Act, 1970")
        assert result.exact_match is False, "Incorrectly matched Section 3 across different acts"
        assert result.document_mismatch is True
    
    def test_exact_rule_match(self):
        """Test exact rule matching (e.g., Rule 43bis)."""
        matcher = create_exact_matcher()
        
        chunk = {
            "text": "Rule 43bis - Patent cooperation treaty",
            "metadata": {
                "document": "PCT Applicant's Guide — International Phase",
                "rule": "43bis",
            }
        }
        
        parsed = {
            "type": "rule",
            "value": "43bis",
            "canonical_title": "PCT Applicant's Guide — International Phase",
        }
        
        result = matcher.match(chunk, parsed, "PCT Applicant's Guide — International Phase")
        assert result.exact_match is True, f"Failed to match Rule 43bis: {result.match_reason}"
    
    def test_exact_patent_number_match(self):
        """Test exact patent number matching."""
        matcher = create_exact_matcher()
        
        chunk = {
            "text": "Patent 429737 relates to...",
            "metadata": {
                "document": "Patent Database",
                "patent_number": "429737",
            }
        }
        
        parsed = {
            "type": "patent",
            "value": "429737",
        }
        
        result = matcher.match(chunk, parsed)
        assert result.exact_match is True, f"Failed to match patent number: {result.match_reason}"


class TestQueryEvidenceValidator:
    """Tests for query-evidence alignment validation."""
    
    def test_exact_lookup_validation_pass(self):
        """Test that exact lookup validates correctly when provision matches."""
        validator = create_validator()
        
        query = "What does Section 3(p) of the Patents Act state?"
        
        # Evidence with correct provision
        evidence = [
            {
                "text": "Section 3(p) defines traditional knowledge",
                "metadata": {
                    "document": "The Patents Act, 1970",
                    "section": "3",
                    "clause": "p",
                    "source_tier": 1,
                }
            }
        ]
        
        result = validator.validate(query, evidence)
        assert result.query_aligned is True, f"Failed validation: {result.alignment_reason}"
        assert result.allow_answer is True
    
    def test_exact_lookup_validation_fails_wrong_provision(self):
        """Test that validation fails when provision does NOT match."""
        validator = create_validator()
        
        query = "What does Section 3(p) of the Patents Act state?"
        
        # Evidence with WRONG provision (Section 4 instead of Section 3)
        evidence = [
            {
                "text": "Section 4(p) is about something else",
                "metadata": {
                    "document": "The Patents Act, 1970",
                    "section": "4",
                    "clause": "p",
                    "source_tier": 1,
                }
            }
        ]
        
        result = validator.validate(query, evidence)
        assert result.query_aligned is False, "Should fail when provision doesn't match"
        assert result.allow_answer is False
    
    def test_current_fee_validation_requires_official_source(self):
        """Test that current fee queries require official fee schedule."""
        validator = create_validator()
        
        query = "What is the current trademark registration fee in India?"
        
        # Evidence without official fee schedule (just generic Section 25)
        evidence = [
            {
                "text": "Section 25 deals with registration of trademarks",
                "metadata": {
                    "document": "The Trade Marks Act, 1999",
                    "section": "25",
                    "source_tier": 1,
                }
            }
        ]
        
        result = validator.validate(query, evidence)
        assert result.requested_intent == "CURRENT_FEE"
        assert result.query_aligned is False, "Current fee query should fail without fee schedule"
        assert result.allow_answer is False


class TestFeeDetector:
    """Tests for current-fee detection and routing."""
    
    def test_detect_current_fee_query(self):
        """Test detection of current fee queries."""
        detector = FeeDetectorNode()
        
        queries = [
            "What is the current trademark registration fee?",
            "How much does it cost to file a patent application?",
            "What is the latest registration fee in India?",
        ]
        
        for query in queries:
            is_fee, fee_type, confidence = detector.detect(query)
            assert is_fee is True, f"Failed to detect fee query: {query}"
            assert fee_type in ("CURRENT_FEE", "FEE_SCHEDULE"), f"Wrong fee type: {fee_type}"
            assert confidence > 0.5, f"Low confidence for: {query}"
    
    def test_detect_historical_fee_query(self):
        """Test detection of historical/example fee queries."""
        detector = FeeDetectorNode()
        
        query = "What was the registration fee in 2020?"
        is_fee, fee_type, confidence = detector.detect(query)
        
        assert is_fee is True
        assert fee_type == "HISTORICAL_FEE", f"Should detect as historical, got {fee_type}"
    
    def test_non_fee_query_detection(self):
        """Test that non-fee queries are not misidentified."""
        detector = FeeDetectorNode()
        
        queries = [
            "What does Section 3(p) of the Patents Act state?",
            "How long is the patent term?",
            "What is the procedure for filing a trademark?",
        ]
        
        for query in queries:
            is_fee, fee_type, confidence = detector.detect(query)
            assert is_fee is False, f"Incorrectly identified as fee query: {query}"
    
    def test_current_fee_evidence_validation(self):
        """Test that current fee queries validate evidence correctly."""
        detector = FeeDetectorNode()
        
        # Generic section about trademarks (NOT a fee schedule)
        evidence = [
            {
                "text": "Section 25 of the Trade Marks Act deals with registration",
                "metadata": {
                    "document": "The Trade Marks Act, 1999",
                    "section": "25",
                    "source_tier": 1,
                }
            }
        ]
        
        is_valid, reason = detector.validate_fee_evidence(evidence, "CURRENT_FEE")
        assert is_valid is False, "Should fail: Section 25 is not a fee schedule"
    
    def test_current_fee_refusal_message(self):
        """Test that appropriate refusal message is generated."""
        detector = FeeDetectorNode()
        
        message = detector.get_refusal_message("CURRENT_FEE")
        assert "official" in message.lower()
        assert "fee" in message.lower()
        assert len(message) > 50


class TestProvisionParsing:
    """Tests for legal identifier parsing."""
    
    def test_parse_section_with_clause(self):
        """Test parsing of 'Section 3(p)'."""
        parsed = parse_legal_identifier("What does Section 3(p) state?")
        
        assert parsed.get("type") == "section"
        assert parsed.get("value") == "3(p)"
    
    def test_parse_section_without_clause(self):
        """Test parsing of 'Section 3' without clause."""
        parsed = parse_legal_identifier("What does Section 3 state?")
        
        assert parsed.get("type") == "section"
        assert "3" in parsed.get("value", "")
    
    def test_parse_patent_number(self):
        """Test parsing of patent numbers."""
        parsed = parse_legal_identifier("Patent No. 429737")
        
        assert parsed.get("type") == "patent"
        assert "429737" in parsed.get("value", "")
    
    def test_parse_rule(self):
        """Test parsing of rules."""
        parsed = parse_legal_identifier("PCT Rule 43bis")
        
        assert parsed.get("type") == "rule"
        assert "43bis" in parsed.get("value", "").lower()
    
    def test_parse_article(self):
        """Test parsing of articles."""
        parsed = parse_legal_identifier("Article 3 of WIPO treaty")
        
        assert parsed.get("type") == "article"
        assert "3" in parsed.get("value", "")


class TestLegalPrecisionMetrics:
    """Meta-test: verify all success criteria are testable."""
    
    def test_success_criteria_a_exact_provision(self):
        """Success Criterion A: Section 3(p) query returns Section 3(p), NOT Section 4."""
        matcher = create_exact_matcher()
        
        # This should PASS
        chunk_correct = {
            "metadata": {"document": "The Patents Act, 1970", "section": "3", "clause": "p"}
        }
        parsed = {"type": "section", "value": "3(p)", "canonical_title": "The Patents Act, 1970"}
        result = matcher.match(chunk_correct, parsed, "The Patents Act, 1970")
        assert result.exact_match is True
        
        # This should FAIL
        chunk_wrong = {
            "metadata": {"document": "The Patents Act, 1970", "section": "4", "clause": "p"}
        }
        result = matcher.match(chunk_wrong, parsed, "The Patents Act, 1970")
        assert result.exact_match is False
    
    def test_success_criteria_d_current_fee_routing(self):
        """Success Criterion D: Current fee queries route to safe refusal if schedule unavailable."""
        detector = FeeDetectorNode()
        
        query = "What is the current trademark registration fee?"
        is_fee, fee_type, confidence = detector.detect(query)
        
        assert is_fee is True
        assert fee_type == "CURRENT_FEE"
        
        # Insufficient evidence
        evidence = [{"metadata": {"source_tier": 1}, "text": "Section 25..."}]
        is_valid, _ = detector.validate_fee_evidence(evidence, fee_type)
        
        assert is_valid is False
        skip_gen, refusal = detector.should_bypass_generation(fee_type, is_valid)
        assert skip_gen is True
        assert refusal is not None
    
    def test_success_criteria_e_verification_labels(self):
        """Success Criterion E: Separate indicators for claim support vs query alignment."""
        # This is tested through the validator returning both flags
        validator = create_validator()
        
        query = "What does Section 3(p) state?"
        evidence = [
            {
                "metadata": {
                    "document": "The Patents Act, 1970",
                    "section": "3",
                    "clause": "p",
                    "source_tier": 1,
                },
                "text": "Section 3(p) defines TK"
            }
        ]
        
        result = validator.validate(query, evidence)
        # Both should be true for a valid answer
        assert result.query_aligned is True
        assert result.allow_answer is True


# Integration test
class TestIntegration:
    """End-to-end integration tests."""
    
    def test_section_3p_workflow(self):
        """Full workflow: query Section 3(p), ensure correct provision retrieved."""
        registry = get_document_registry()
        matcher = create_exact_matcher()
        validator = create_validator()
        
        # User query
        query = "What does Section 3(p) of the Patents Act, 1970 state regarding traditional knowledge?"
        
        # Parse query
        parsed = parse_legal_identifier(query)
        assert parsed.get("type") == "section"
        
        # Normalize document
        canonical_doc = registry.get_canonical_title("Patents Act, 1970")
        assert canonical_doc == "The Patents Act, 1970"
        
        # Mock retrieved evidence (correct)
        evidence = [
            {
                "text": "Section 3(p) traditional knowledge",
                "metadata": {
                    "document": "The Patents Act, 1970",
                    "section": "3",
                    "clause": "p",
                    "source_tier": 1,
                }
            }
        ]
        
        # Match check
        result_match = matcher.match(evidence[0], parsed, canonical_doc)
        assert result_match.exact_match is True
        
        # Validation check
        result_validate = validator.validate(query, evidence, parsed)
        assert result_validate.query_aligned is True
        assert result_validate.allow_answer is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
