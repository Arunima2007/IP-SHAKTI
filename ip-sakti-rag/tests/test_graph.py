"""Unit and Integration Tests for Milestone 5 LangGraph Orchestration."""
import pytest
from src.graph.state import GraphState
from src.graph.nodes.query_understanding_node import QueryUnderstandingNode
from src.graph.nodes.evidence_sufficiency_node import EvidenceSufficiencyNode
from src.graph.nodes.safe_refusal_node import SafeRefusalNode
from src.graph.routers.query_router import route_query
from src.graph.routers.sufficiency_router import route_sufficiency
from src.graph.routers.validation_router import route_validation
from src.graph.graph import build_ip_sakti_graph, IPSAKTILangGraphCoordinator
from src.config import INSUFFICIENT_EVIDENCE_MESSAGE


def test_query_understanding_classification():
    node = QueryUnderstandingNode()

    # 1. Exact Legal Lookup
    state1: GraphState = {"query": "What does Section 3(p) of the Patents Act state?"}
    res1 = node(state1)
    assert res1["query_type"] == "EXACT_LOOKUP"
    assert "Section 3(p)" in res1["exact_identifiers"]
    assert res1["language"] == "English"
    assert res1["jurisdiction"] == "India"

    # 2. Multilingual Hindi
    state2: GraphState = {"query": "अश्वगंधा के पेटेंट नियम क्या हैं?"}
    res2 = node(state2)
    assert res2["language"] == "Hindi"
    assert res2["expanded_query"] is not None
    assert "withania somnifera" in res2["expanded_query"].lower()

    # 3. Code-Mixed Hinglish
    state3: GraphState = {"query": "Can an Ayurvedic product ko patent kiya ja sakta hai?"}
    res3 = node(state3)
    assert res3["language"] == "Hinglish / Code-Mixed"

    # 4. Out of Scope
    state4: GraphState = {"query": "Who won the cricket match in California yesterday?"}
    res4 = node(state4)
    assert res4["query_type"] == "OUT_OF_SCOPE"


def test_routers():
    # Query Router
    assert route_query({"query_type": "OUT_OF_SCOPE"}) == "safe_refusal"
    assert route_query({"query_type": "EXACT_LOOKUP"}) == "retrieval"
    assert route_query({"query_type": "FACTUAL"}) == "retrieval"

    # Sufficiency Router
    assert route_sufficiency({"evidence_sufficient": True, "retrieval_attempt": 1}) == "generation"
    assert route_sufficiency({"evidence_sufficient": False, "retrieval_attempt": 1}) == "retrieval"
    assert route_sufficiency({"evidence_sufficient": False, "retrieval_attempt": 2}) == "safe_refusal"

    # Validation Router
    assert route_validation({"validation_status": "VALID", "generation_attempt": 1, "is_valid": True}) == "end"
    assert route_validation({"validation_status": "RETRY_GENERATION", "generation_attempt": 1, "is_valid": False}) == "generation"
    assert route_validation({"validation_status": "RETRY_GENERATION", "generation_attempt": 2, "is_valid": False}) == "safe_refusal"


def test_evidence_sufficiency_node():
    node = EvidenceSufficiencyNode()

    # Case A: Empty evidence
    state_empty: GraphState = {"query": "What is Section 3(p)?", "selected_evidence": []}
    res_empty = node(state_empty)
    assert res_empty["evidence_sufficient"] is False

    # Case B: Out of scope keyword
    state_oos: GraphState = {
        "query": "What are Brazilian patent procedures under 1996 Law?",
        "selected_evidence": [{"text": "sample text", "rerank_score": 0.85}]
    }
    res_oos = node(state_oos)
    assert res_oos["evidence_sufficient"] is False

    # Case C: Sufficient evidence
    state_valid: GraphState = {
        "query": "What does Section 3(p) state?",
        "selected_evidence": [
            {
                "chunk_id": "c1",
                "document": "Patent Act-1970.pdf",
                "section": "3(p)",
                "heading": "What are not inventions",
                "text": "The following are not inventions: (p) traditional knowledge.",
                "rerank_score": 0.95
            }
        ]
    }
    res_valid = node(state_valid)
    assert res_valid["evidence_sufficient"] is True
    assert "E1" in res_valid["evidence_map"]


def test_evidence_sufficiency_rejects_wrong_provision_even_with_high_score():
    node = EvidenceSufficiencyNode()
    state = {
        "query": "What does Section 3(p) of the Patents Act state?",
        "query_type": "EXACT_LOOKUP",
        "parsed_identifier": {"type": "section", "value": "3(p)", "canonical_title": "Patents Act, 1970"},
        "exact_identifiers": ["Section 3(p)"],
        "selected_evidence": [{
            "chunk_id": "wrong", "document": "Patent Act-1970.pdf", "section": "4",
            "text": "Section 4 concerns atomic energy.", "reranker_score": 0.99,
        }],
    }
    result = node(state)
    assert result["evidence_sufficient"] is False


def test_current_fee_requires_fee_schedule_evidence():
    node = EvidenceSufficiencyNode()
    state = {
        "query": "What is the exact current trademark registration fee in India?",
        "query_type": "CURRENT_FEE_LOOKUP",
        "selected_evidence": [{
            "document": "The Trade Marks Act, 1999", "section": "25",
            "text": "Registration is for ten years and may be renewed.", "reranker_score": 0.99,
        }],
    }
    assert node(state)["evidence_sufficient"] is False


def test_safe_refusal_node():
    node = SafeRefusalNode()
    # Out of scope
    state_oos: GraphState = {"query_type": "OUT_OF_SCOPE", "scope_status": "OUT_OF_SCOPE", "refusal_reason": "unsupported_general_knowledge"}
    res_oos = node(state_oos)
    assert res_oos["is_refusal"] is True
    assert "outside my supported domain" in res_oos["final_answer"]
    assert res_oos["final_answer_type"] == "SAFE_REFUSAL"

    # Insufficient evidence
    state_ie: GraphState = {"query_type": "FACTUAL", "scope_status": "IN_SCOPE", "evidence_sufficiency_reason": "Insufficient evidence"}
    res_ie = node(state_ie)
    assert res_ie["is_refusal"] is True
    assert INSUFFICIENT_EVIDENCE_MESSAGE in res_ie["final_answer"]
    assert res_ie["final_answer_type"] == "INSUFFICIENT_EVIDENCE"


def test_compiled_langgraph_out_of_scope():
    coordinator = IPSAKTILangGraphCoordinator()
    query = "What is the orbital velocity of the Chandrayaan-3 propulsion module?"
    result = coordinator.process_query(query)

    assert result["is_refusal"] is True
    assert "outside my supported domain" in result["final_answer"]
    assert result.get("retrieval_called", False) is False
    assert result.get("generation_called", False) is False
    assert len(result.get("citations", [])) == 0
    assert any(t["node"] == "safe_refusal" for t in result["execution_trace"])


def test_compiled_langgraph_in_scope():
    coordinator = IPSAKTILangGraphCoordinator()
    query = "What does Section 3(p) of the Indian Patents Act, 1970 state?"
    result = coordinator.process_query(query)

    assert result["is_refusal"] is False
    assert result["is_valid"] is True
    assert len(result["citations"]) >= 1
    assert "Patents Act" in result["final_answer"]
    assert "### Sources" in result["final_answer"]
    
    # Trace validation
    node_names = [t["node"] for t in result["execution_trace"]]
    assert "query_understanding" in node_names
    assert "retrieval" in node_names
    assert "reranking" in node_names
    assert "evidence_sufficiency" in node_names
    assert "generation" in node_names
    assert "citation_validation" in node_names
