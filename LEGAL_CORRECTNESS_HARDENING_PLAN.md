# IP-SHAKTI Legal Correctness Hardening Plan

## Overview

This document outlines the comprehensive hardening pass to fix the core legal-RAG correctness issues: ensuring the system retrieves and answers from the **correct legal provision** using the **correct document**, not merely providing well-cited evidence that answers the wrong question.

## Core Problems Identified

### Problem 1: Wrong Provision Retrieved (Section Mismatch)
- **Symptom**: Query asks for "Section 3(p)" but system retrieves and answers from "Section 4"
- **Root Cause**: 
  - Legal identifier parser reduces "Section 3(p)" to just "(p)" during exact matching
  - BM25 and semantic similarity treat "(p)" as matching any clause (p) from any section
  - Reranker does not validate that the parent section matches
- **Impact**: Answer may be well-cited but factually wrong

### Problem 2: Wrong Document Retrieved (Document Mismatch)
- **Symptom**: Query asks about Patents Act but retrieval includes Trade Marks Act
- **Root Cause**:
  - Multiple document aliases map to same normalized form
  - No canonical document registry enforced across modules
  - Reranker does not validate document alignment with query
- **Impact**: Cross-document confusion

### Problem 3: Query-Evidence Misalignment (Intent Mismatch)
- **Symptom**: "What is the current trademark fee?" answered with Section 25 of Trade Marks Act
- **Root Cause**:
  - Current-fee intent not validated before generation
  - Reranker promotes general trademark provisions regardless of currency/fee specificity
  - Generator accepts any provided evidence without validating intent match
- **Impact**: Answers hallucinate fees from generic provisions

### Problem 4: Misleading Verification Badge
- **Symptom**: UI shows "Evidence Verified (100% Claim Support)" even when query-evidence alignment fails
- **Root Cause**:
  - Citation validation only checks claim-to-evidence match, not query-to-evidence match
  - No distinction between "claim is supported" and "requested provision was retrieved"
- **Impact**: User sees false confidence in wrong answer

## Implementation Strategy

### Phase 1: Canonical Document Registry (P1 Foundation)

**File**: `src/retrieval/document_registry.py` (NEW)

Create single source of truth for all document normalization:

```python
class DocumentRegistry:
    """Canonical document normalization and validation."""
    
    def __init__(self):
        # Exact, authoritative canonical forms
        self.CANONICAL_DOCUMENTS = {
            "The Patents Act, 1970": {
                "aliases": [
                    "Patents Act", "Patent Act", "Patent Act 1970",
                    "The Patents Act, 1970", "Indian Patents Act",
                    "Patents Act, 1970", "Patents Act 1970",
                ],
                "jurisdictions": ["India"],
                "document_id": "patent_act_1970",
                "tier": 1,
            },
            "The Trade Marks Act, 1999": {
                "aliases": [...],
                "jurisdictions": ["India"],
                "document_id": "trade_marks_act_1999",
                "tier": 1,
            },
            # ... all other documents
        }
    
    def normalize(self, text: str) -> Optional[str]:
        """Return canonical title or None."""
    
    def get_document_id(self, canonical: str) -> str:
        """Return document_id for canonical title."""
    
    def get_tier(self, canonical: str) -> int:
        """Return authority tier."""
```

**Update files**:
- `config.py`: Remove `LEGAL_DOCUMENT_ALIASES` and reference registry
- `legal_identifier_parser.py`: Use registry instead of hard-coded `_DOCUMENTS`
- `exact_lookup.py`: Use registry in `_build_filters()`
- `retrieval_node.py`: Pass canonical document from registry

### Phase 2: Exact Legal Identifier Matching (P0 Critical)

**File**: `src/retrieval/legal_identifier_matcher.py` (NEW)

Implement strict structural validation:

```python
class ExactProvisionMatcher:
    """
    Validates exact legal identifier matches without reducing to sub-components.
    
    NEVER reduce:
      Section 3(p) → (p)
    
    ALWAYS require structural proof:
      document_matches AND section_matches AND clause_matches
    """
    
    def match(
        self,
        chunk: Dict[str, Any],
        parsed_identifier: Dict[str, Any],
        requested_document: Optional[str],
    ) -> MatchResult:
        """
        Returns:
            MatchResult(
                exact_match=bool,
                match_reason=str,
                requested_section=str,
                requested_clause=Optional[str],
                candidate_section=str,
                candidate_clause=Optional[str],
            )
        """
```

**Update files**:
- `exact_lookup.py`: Use matcher instead of inline regex
- `reranking_node.py`: Enforce matcher result (no promotion of non-exact candidates as exact)
- `legal_identifier_parser.py`: Extract full provision structure (document + section + clause)

### Phase 3: Query-Evidence Validation Layer (P0 Critical)

**File**: `src/generation/query_evidence_validator.py` (NEW)

Pre-generation validation:

```python
class QueryEvidenceValidator:
    """
    Validates that retrieved evidence actually answers the user's query.
    
    SEPARATE concepts:
    1. Claim Support: Does evidence support the generated claim?
    2. Query Alignment: Does evidence answer the requested question?
    """
    
    def validate(
        self,
        query: str,
        parsed_intent: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Returns:
            ValidationResult(
                query_aligned=bool,
                alignment_score=float,
                reason=str,
                required_document=Optional[str],
                required_provision=Optional[str],
                retrieved_documents=List[str],
                retrieved_provisions=List[str],
                missing_targets=List[str],
            )
        """
```

### Phase 4: Current-Fee / Current-Regulation Detection (P0)

**File**: `src/graph/nodes/fee_detector_node.py` (NEW)

Enhanced query classifier for fee/regulation queries:

```python
class FeeDetectorNode:
    """
    Classify current-fee and current-regulation queries.
    
    Triggers SAFE_REFUSAL if official fee schedule not in evidence.
    """
    
    FEE_KEYWORDS = [
        "current fee", "latest fee", "registration fee",
        "application fee", "renewal fee", "official fee",
        "how much does it cost", "current charges",
        "fee schedule", "current cost",
    ]
    
    def detect(self, query: str) -> Tuple[bool, str]:
        """Returns (is_fee_query, fee_type)"""
```

### Phase 5: Answer Generation with Intent Awareness (P1)

**File**: `src/generation/answer_generator.py` (UPDATE)

Restructure generation based on detected intent:

```python
class IntentAwareAnswerGenerator:
    """
    Tailor answer structure to query intent.
    
    EXACT_LOOKUP: Direct provision text
    EXPLANATION: Interpretation + examples
    CURRENT_FEE: Official schedule only, refuse if unavailable
    COMPARISON: Side-by-side comparison
    """
```

**Key Changes**:
- Parse intent from query + parsed_identifier
- Validate evidence matches intent before generation
- Pass explicit structural instructions to Gemini
- Format output based on intent

### Phase 6: Response Formatting (P1)

**Files**: `src/generation/answer_generator.py`, `src/generation/evidence_formatter.py` (UPDATE)

Fix output formatting:
- Remove ALL_CAPS headings
- Use normal sentence case
- Group related provisions, not all retrieved provisions
- Remove internal scoring/debugging info
- Keep citations clickable

### Phase 7: Reranker Safety (P1)

**File**: `src/graph/nodes/reranking_node.py` (UPDATE)

```python
def reranking_node(state: GraphState) -> GraphState:
    """
    Strict reranker safety:
    
    1. Exact candidates MUST have exact_provision_match == True
    2. Do NOT promote candidates based on relevance alone
    3. Do NOT override legal identifier mismatch
    4. Maintain domain diversity without sacrificing provision correctness
    """
```

### Phase 8: Metadata Enrichment (P1)

**File**: `src/ingestion/chunk_processor.py` (UPDATE)

Ensure chunks have complete metadata:

```python
chunk.metadata = {
    "document_id": "patent_act_1970",  # From registry
    "document_name": "The Patents Act, 1970",
    "jurisdiction": "India",
    "provision_type": "section",
    "section": "3",
    "clause": "p",  # NOT just "(p)"
    "full_provision": "section:3:p",  # Canonical form
    "page": 42,
    "heading": "...",
    "source_tier": 1,
}
```

### Phase 9: Multilingual Handling (P2)

**File**: `src/graph/nodes/query_understanding_node.py` (UPDATE)

Preserve legal identifiers through Hindi/Hinglish:

```python
# Hinglish: "Sections 3(p) क्या है"
# Must preserve: section:3:p
# NOT reduce to: (p)
```

### Phase 10: Legal Precision Benchmark (P2)

**File**: `tests/test_legal_precision.py` (NEW)

Comprehensive regression test suite:

```python
class LegalPrecisionBenchmark:
    """
    Test exact provision retrieval accuracy.
    
    Metrics:
    - exact_provision_accuracy
    - document_accuracy
    - query_evidence_alignment
    - claim_support_rate
    - citation_precision
    - citation_recall
    - safe_refusal_accuracy
    - current_fee_routing_accuracy
    - cross_domain_coverage
    - answer_readability
    """
```

## Files to Create

1. `src/retrieval/document_registry.py` - Canonical document mapping
2. `src/retrieval/legal_identifier_matcher.py` - Structured exact matching
3. `src/generation/query_evidence_validator.py` - Query-evidence alignment validation
4. `src/graph/nodes/fee_detector_node.py` - Current-fee classification
5. `src/generation/intent_classifier.py` - Answer intent detection
6. `tests/test_legal_precision.py` - Regression test suite
7. `LEGAL_HARDENING_DEBUG_TRACE.md` - Debug trace format specification

## Files to Update

1. `src/config.py` - Remove legacy aliases, add registry reference
2. `src/retrieval/legal_identifier_parser.py` - Use registry, extract full structure
3. `src/retrieval/exact_lookup.py` - Use ExactProvisionMatcher
4. `src/graph/nodes/retrieval_node.py` - Add exact vs. hybrid logic
5. `src/graph/nodes/reranking_node.py` - Enforce exact match constraints
6. `src/graph/nodes/query_understanding_node.py` - Preserve multilingual identifiers
7. `src/graph/nodes/evidence_sufficiency_node.py` - Use QueryEvidenceValidator
8. `src/graph/nodes/generation_node.py` - Pass intent and validation flags
9. `src/generation/answer_generator.py` - Intent-aware formatting, normal case output
10. `src/generation/citation_validator.py` - Separate claim support from query alignment
11. `src/generation/evidence_formatter.py` - Remove irrelevant provisions
12. `src/graph/graph.py` - Add fee_detector_node in flow
13. `src/graph/state.py` - Add new state fields for validation results

## New State Fields

```python
class GraphState(TypedDict, total=False):
    # Query Intent & Validation
    query_intent: str  # EXACT_LOOKUP, EXPLANATION, CURRENT_FEE, COMPARISON, etc.
    is_fee_query: bool
    fee_type: Optional[str]
    
    # Exact Provision Matching
    exact_match_found: bool
    requested_document: Optional[str]
    requested_provision: Optional[str]
    exact_candidates: List[Dict[str, Any]]
    
    # Query-Evidence Validation
    query_evidence_validated: bool
    query_aligned: bool
    alignment_reason: str
    alignment_score: float
    
    # Verification Status
    claim_supported: bool
    query_aligned: bool
    final_answer_allowed: bool
    
    # Debug Trace
    legal_retrieval_trace: Dict[str, Any]
```

## Success Criteria

### A. Exact Provision Retrieval
```
Query: "What does Section 3(p) of the Patents Act state?"
✓ System retrieves Section 3(p), NOT Section 4
✓ System clearly states if Section 3(p) not found
```

### B. Exact Document Matching
```
Query: "What is Section 25 of the Trade Marks Act?"
✓ System retrieves Trade Marks Act Section 25, not other acts
```

### C. Current-Fee Routing
```
Query: "What is the current trademark registration fee?"
✓ System detects fee intent
✓ If official fee schedule absent: SAFE_REFUSAL
✓ NOT answered from Section 25
```

### D. Query Alignment
```
Query: "What does Section 3(p) state?"
Evidence: Section 4
✓ Query Aligned = FALSE
✓ Final Answer Allowed = FALSE
✓ Safe refusal returned
```

### E. Verification Labels
```
✓ "Evidence Verified" only when BOTH:
  - Claim Supported = TRUE
  - Query Aligned = TRUE
✓ Shows separate indicators for each
```

### F. Formatting
```
✓ Normal sentence case only
✓ No ALL-CAPS headings
✓ Relevant provisions only
✓ No internal scores/debug info
```

### G. Multilingual
```
✓ Hindi/Hinglish queries preserve legal identifiers
✓ Section 3(p) in Hindi still retrieves Section 3(p)
```

## Testing Strategy

### Unit Tests
- `test_document_registry.py` - Document normalization
- `test_legal_identifier_matcher.py` - Exact matching logic
- `test_query_evidence_validator.py` - Alignment validation
- `test_fee_detector.py` - Fee classification

### Integration Tests
- `test_exact_provision_flow.py` - Full exact lookup pipeline
- `test_current_fee_flow.py` - Fee query routing
- `test_multilingual_flow.py` - Language preservation

### Regression Suite
- `test_legal_precision.py` - 10 benchmark queries
- `test_milestone_1_6_compat.py` - Existing functionality preserved

## Rollback & Compatibility

- All changes are additive or non-breaking
- Existing APIs remain stable
- LangGraph state backward-compatible (new fields with defaults)
- Fallback to safe refusal if new validation fails
- No removal of working components

## Metrics & Observability

### New Metrics
- `exact_provision_accuracy` - Correct provision retrieved
- `query_alignment_accuracy` - Evidence answers requested question
- `document_accuracy` - Correct document retrieved
- `safe_refusal_precision` - Correct refusals for missing evidence
- `current_fee_routing_accuracy` - Fee queries routed correctly

### Debug Trace Format
```json
{
  "query": "...",
  "intent": "EXACT_LOOKUP",
  "requested_document": "Patents Act, 1970",
  "requested_provision": "section:3:p",
  "exact_candidates_found": 3,
  "exact_candidates_valid": 1,
  "invalid_exact_candidates": 2,
  "top_candidate": {...},
  "query_evidence_alignment": true,
  "alignment_score": 0.95,
  "generation_allowed": true
}
```

## Timeline

1. **Phase 1-3** (Critical): Document registry, exact matcher, query-evidence validator
2. **Phase 4-5** (Critical): Fee detector, intent-aware generation
3. **Phase 6-7** (High): Formatting fixes, reranker safety
4. **Phase 8-9** (Medium): Metadata enrichment, multilingual
5. **Phase 10** (Ongoing): Benchmarking and regression tests

---

**Status**: Ready for implementation  
**Author**: Hardening Initiative  
**Version**: 1.0  
**Last Updated**: 2026-09-05
