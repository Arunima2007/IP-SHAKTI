"""StateGraph Orchestrator for IP-SAKTI Sahayak.

Builds and compiles the complete LangGraph StateGraph workflow:
Query Understanding -> Router -> Hybrid Retrieval -> Cross-Encoder Reranker ->
Evidence Sufficiency Check (with retrieval retry) -> Grounded Generation ->
Citation Validation (with generation feedback retry) -> Safe Refusal / Final Output.
"""
from typing import Dict, Any, Optional
import time
import logging

from langgraph.graph import StateGraph, START, END

from src.graph.state import GraphState
from src.graph.nodes.query_understanding_node import QueryUnderstandingNode
from src.graph.nodes.retrieval_node import RetrievalNode
from src.graph.nodes.reranking_node import RerankingNode
from src.graph.nodes.evidence_sufficiency_node import EvidenceSufficiencyNode
from src.graph.nodes.generation_node import GenerationNode
from src.graph.nodes.citation_validation_node import CitationValidationNode
from src.graph.nodes.safe_refusal_node import SafeRefusalNode

from src.graph.routers.query_router import route_query
from src.graph.routers.sufficiency_router import route_sufficiency
from src.graph.routers.validation_router import route_validation

logger = logging.getLogger(__name__)


def build_ip_sakti_graph(
    query_node: Optional[QueryUnderstandingNode] = None,
    retrieval_node: Optional[RetrievalNode] = None,
    reranking_node: Optional[RerankingNode] = None,
    sufficiency_node: Optional[EvidenceSufficiencyNode] = None,
    generation_node: Optional[GenerationNode] = None,
    citation_node: Optional[CitationValidationNode] = None,
    refusal_node: Optional[SafeRefusalNode] = None,
) -> StateGraph:
    """Constructs and compiles the IP-SAKTI Sahayak LangGraph workflow."""
    
    # 1. Instantiate Nodes
    q_node = query_node or QueryUnderstandingNode()
    ret_node = retrieval_node or RetrievalNode()
    rerank_node = reranking_node or RerankingNode()
    suff_node = sufficiency_node or EvidenceSufficiencyNode()
    gen_node = generation_node or GenerationNode()
    cit_node = citation_node or CitationValidationNode()
    ref_node = refusal_node or SafeRefusalNode()

    # 2. Build StateGraph
    builder = StateGraph(GraphState)

    # Add Nodes
    builder.add_node("query_understanding", q_node)
    builder.add_node("retrieval", ret_node)
    builder.add_node("reranking", rerank_node)
    builder.add_node("evidence_sufficiency", suff_node)
    builder.add_node("generation", gen_node)
    builder.add_node("citation_validation", cit_node)
    builder.add_node("safe_refusal", ref_node)

    # 3. Add Edges & Conditional Routing
    # START -> Query Understanding
    builder.add_edge(START, "query_understanding")

    # Query Understanding -> (Router) -> Retrieval OR Safe Refusal
    builder.add_conditional_edges(
        "query_understanding",
        route_query,
        {
            "retrieval": "retrieval",
            "safe_refusal": "safe_refusal"
        }
    )

    # Retrieval -> Reranking -> Evidence Sufficiency
    builder.add_edge("retrieval", "reranking")
    builder.add_edge("reranking", "evidence_sufficiency")

    # Evidence Sufficiency -> (Router) -> Generation OR Retrieval Retry OR Safe Refusal
    builder.add_conditional_edges(
        "evidence_sufficiency",
        route_sufficiency,
        {
            "generation": "generation",
            "retrieval": "retrieval",
            "safe_refusal": "safe_refusal"
        }
    )

    # Generation -> Citation Validation
    builder.add_edge("generation", "citation_validation")

    # Citation Validation -> (Router) -> END OR Generation Retry OR Safe Refusal
    builder.add_conditional_edges(
        "citation_validation",
        route_validation,
        {
            "end": END,
            "generation": "generation",
            "safe_refusal": "safe_refusal"
        }
    )

    # Safe Refusal -> END
    builder.add_edge("safe_refusal", END)

    return builder.compile()


class IPSAKTILangGraphCoordinator:
    """Coordinator class providing a clean execution interface to the compiled LangGraph."""

    def __init__(self, compiled_graph=None):
        self.graph = compiled_graph or build_ip_sakti_graph()

    def process_query(self, query: str) -> Dict[str, Any]:
        """Executes the full LangGraph pipeline for a user query."""
        t_start = time.perf_counter()
        
        initial_state: GraphState = {
            "query": query.strip(),
            "original_query": query.strip(),
            "node_latencies_ms": {},
            "execution_trace": []
        }

        # Execute StateGraph
        final_state = self.graph.invoke(initial_state)
        
        total_latency = round((time.perf_counter() - t_start) * 1000, 2)
        final_state["total_latency_ms"] = total_latency

        return final_state

    def run(self, query: str) -> Dict[str, Any]:
        """Convenience alias for process_query."""
        return self.process_query(query)

