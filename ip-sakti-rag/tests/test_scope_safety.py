"""Regression tests for Milestone 6 Scope Safety, Zero-Leakage Gate, and Latency.

Ensures that:
1. All out-of-scope queries are immediately classified as OUT_OF_SCOPE.
2. Retrieval, reranking, and generation are NEVER invoked for out-of-scope queries (0% leakage).
3. Citation list is strictly empty.
4. Response is natural sentence-case safe refusal.
5. In-scope queries are properly routed and processed.
"""
import pytest
import time
from src.graph.nodes.query_understanding_node import QueryUnderstandingNode
from src.graph.graph import IPSAKTILangGraphCoordinator


OUT_OF_SCOPE_TEST_QUERIES = [
    "Who is Virat Kohli?",
    "What is the capital of France?",
    "Tell me today's weather.",
    "Write a Python program to sort an array.",
    "Who won the FIFA World Cup?",
    "What is the stock price of Apple?",
    "Give me a recipe for pasta.",
    "Explain quantum physics.",
    "What is the population of India?",
    "Who is Elon Musk?",
    "Write a poem about rain.",
    "What is the latest IPL score?",
    "How do I learn Java programming?",
    "What is Bitcoin cryptocurrency?",
    "Give me a travel itinerary for Delhi.",
    "What is the distance between Delhi and Mumbai?",
    "Who is the Prime Minister of Japan?",
    "Explain photosynthesis in plants.",
    "Recommend a good gaming laptop.",
    "What is today's breaking news headlines?",
    "How do I bake a chocolate cake?",
    "Who directed the movie Inception?"
]

IN_SCOPE_TEST_QUERIES = [
    ("What is Section 3(p) of the Patents Act?", "PATENTS", "EXACT_LOOKUP"),
    ("What is inventive step under Indian patent law?", "PATENTS", "EXPLANATORY"),
    ("What role does TKDL play in protecting traditional knowledge?", "TRADITIONAL_KNOWLEDGE", "EXPLANATORY"),
    ("When is NBA approval required under Biological Diversity Act?", "BIOLOGICAL_DIVERSITY", "EXPLANATORY"),
    ("What are PCT requirements for entering the national phase in India?", "INTERNATIONAL_IP", "EXPLANATORY"),
    ("Can an Ayurvedic formulation be patented in India?", "AYUSH", "AYURVEDA_IP"),
    ("What are the patentability requirements for an Ayurvedic invention involving biological resources?", "AYUSH", "CROSS_DOMAIN"),
]


@pytest.fixture(scope="module")
def coordinator():
    return IPSAKTILangGraphCoordinator()


@pytest.fixture(scope="module")
def query_node():
    return QueryUnderstandingNode()


def test_query_understanding_out_of_scope_classifier(query_node):
    """Test that all 22 out-of-scope queries are classified as OUT_OF_SCOPE by QueryUnderstandingNode."""
    for query in OUT_OF_SCOPE_TEST_QUERIES:
        res = query_node({"query": query})
        assert res["scope_status"] == "OUT_OF_SCOPE", f"Failed for query: {query}"
        assert res["query_type"] == "OUT_OF_SCOPE", f"Failed for query: {query}"
        assert res["scope_confidence"] >= 0.90, f"Low confidence for out-of-scope query: {query}"
        assert res["domains"] == [], f"Detected unexpected domain for: {query}"


def test_query_understanding_in_scope_classifier(query_node):
    """Test that in-scope queries are classified as IN_SCOPE."""
    for query, expected_domain, expected_type in IN_SCOPE_TEST_QUERIES:
        res = query_node({"query": query})
        assert res["scope_status"] == "IN_SCOPE", f"False out-of-scope for: {query}"
        assert res["query_type"] != "OUT_OF_SCOPE", f"False out-of-scope type for: {query}"
        assert res["scope_confidence"] >= 0.75, f"Low confidence for in-scope query: {query}"
        assert len(res["domains"]) > 0, f"No domain detected for: {query}"


@pytest.mark.parametrize("query", OUT_OF_SCOPE_TEST_QUERIES)
def test_out_of_scope_zero_leakage_and_latency(coordinator, query):
    """Verify 0% leakage, fast latency, and clear refusal on full LangGraph execution."""
    t0 = time.perf_counter()
    result = coordinator.process_query(query)
    latency_ms = (time.perf_counter() - t0) * 1000

    # 1. Classification & Refusal Contract
    assert result["is_refusal"] is True
    assert result["scope_status"] == "OUT_OF_SCOPE"
    assert result["query_type"] == "OUT_OF_SCOPE"
    assert result["final_answer_type"] == "SAFE_REFUSAL"
    assert "outside the scope of IP-SAKTI Sahayak" in result["final_answer"]

    # 2. Strict Zero Leakage Checks
    assert result.get("retrieval_called", False) is False, f"Retrieval leaked for {query}"
    assert result.get("reranking_called", False) is False, f"Reranking leaked for {query}"
    assert result.get("generation_called", False) is False, f"Generation leaked for {query}"
    assert result.get("citations") == [], f"Citations leaked for {query}"
    assert result.get("selected_evidence") == [], f"Evidence leaked for {query}"

    # 3. Execution trace check
    executed_nodes = [t["node"] for t in result["execution_trace"]]
    assert "retrieval" not in executed_nodes
    assert "reranking" not in executed_nodes
    assert "generation" not in executed_nodes
    assert "safe_refusal" in executed_nodes

    # 4. Latency Requirement (< 200ms)
    assert latency_ms < 200, f"Latency too high for out-of-scope: {latency_ms:.2f}ms"


def test_in_scope_execution_integrity(coordinator):
    """Verify that legitimate in-scope queries execute the full retrieval-reranking-generation pipeline."""
    query = "What is Section 3(p) of the Patents Act?"
    result = coordinator.process_query(query)

    assert result["is_refusal"] is False
    assert result["scope_status"] == "IN_SCOPE"
    assert result["retrieval_called"] is True
    assert result["reranking_called"] is True
    assert result["generation_called"] is True
    assert len(result["citations"]) >= 1
    assert "Patents Act" in result["final_answer"] or "Section 3(p)" in result["final_answer"]
