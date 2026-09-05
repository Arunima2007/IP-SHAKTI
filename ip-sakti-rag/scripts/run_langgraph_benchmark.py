#!/usr/bin/env python3
"""
Run LangGraph Benchmark (42 Queries)
Measures:
- Routing accuracy
- Evidence sufficiency decisions
- Retrieval & Reranking stats
- Generation & Citation Validation results
- Retry & Regeneration behaviors
- Node-level latency (mean, median, P95)
- Overall execution latency
- Comprehensive evaluation traces
"""

import os
import sys
import time
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.graph.graph import IPSAKTILangGraphCoordinator
from src.evaluation.langgraph_benchmark import BENCHMARK_QUERIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("langgraph_benchmark_runner")

def compute_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    p95_idx = int(0.95 * n)
    if p95_idx >= n:
        p95_idx = n - 1
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(sorted_v[p95_idx], 3)
    }

def main():
    logger.info("Initializing IP-SAKTI LangGraph Coordinator...")
    coordinator = IPSAKTILangGraphCoordinator()
    
    total_queries = len(BENCHMARK_QUERIES)
    logger.info(f"Starting Milestone 5 LangGraph Benchmark on {total_queries} queries...")
    
    results = []
    node_latencies: Dict[str, List[float]] = {}
    total_latencies: List[float] = []
    
    routing_correct = 0
    sufficiency_correct = 0
    regeneration_count = 0
    refusal_count = 0
    successful_answers = 0
    
    for idx, item in enumerate(BENCHMARK_QUERIES, 1):
        q_id = item["id"]
        category = item["category"]
        query_text = item["query"]
        expected_sufficiency = item["expected_sufficiency"]
        
        logger.info(f"[{idx}/{total_queries}] [{q_id}] ({category}) Query: {query_text[:60]}...")
        start_t = time.time()
        
        try:
            final_state = coordinator.run(query=query_text)
            elapsed = time.time() - start_t
            total_latencies.append(elapsed)
            
            # Record node latencies
            lat_by_node = final_state.get("latency_by_node", {})
            for n_name, lat_val in lat_by_node.items():
                node_latencies.setdefault(n_name, []).append(lat_val)
                
            # Evaluate routing & sufficiency
            actual_type = final_state.get("query_type", "UNKNOWN")
            actual_suff = final_state.get("evidence_sufficient", False)
            val_status = final_state.get("validation_status", "UNKNOWN")
            gen_attempts = final_state.get("generation_attempt", 1)
            ret_attempts = final_state.get("retrieval_attempt", 1)
            final_ans = final_state.get("final_answer", "")
            citations = final_state.get("citations", [])
            claims = final_state.get("claims", [])
            
            is_suff_match = (actual_suff == expected_sufficiency)
            if is_suff_match:
                sufficiency_correct += 1
                
            if category in ["OUT_OF_SCOPE", "INSUFFICIENT_EVIDENCE"]:
                is_route_correct = (actual_type in ["OUT_OF_SCOPE", "INSUFFICIENT_EVIDENCE"]) or (not actual_suff)
            elif category in ["HINDI", "HINGLISH"]:
                is_route_correct = (actual_type == "MULTILINGUAL" or final_state.get("language") in ["Hindi", "Hinglish", "hi"])
            elif category == "CROSS_DOMAIN":
                is_route_correct = (actual_type == "CROSS_DOMAIN" or len(final_state.get("domains", [])) > 1)
            elif category == "EXACT_LOOKUP":
                is_route_correct = (actual_type == "EXACT_LOOKUP" or len(final_state.get("exact_identifiers", [])) > 0)
            else:
                is_route_correct = True
                
            if is_route_correct:
                routing_correct += 1
                
            if gen_attempts > 1:
                regeneration_count += 1
                
            if "I could not find sufficient authoritative evidence" in final_ans or "outside the scope" in final_ans:
                refusal_count += 1
            else:
                successful_answers += 1
                
            result_record = {
                "id": q_id,
                "category": category,
                "query": query_text,
                "language": final_state.get("language"),
                "query_type": actual_type,
                "domains": final_state.get("domains", []),
                "jurisdiction": final_state.get("jurisdiction", "India"),
                "exact_identifiers": final_state.get("exact_identifiers", []),
                "retrieval_candidates_count": len(final_state.get("retrieval_candidates", [])),
                "selected_evidence_count": len(final_state.get("selected_evidence", [])),
                "evidence_sufficient": actual_suff,
                "validation_status": val_status,
                "generation_attempts": gen_attempts,
                "retrieval_attempts": ret_attempts,
                "claims_count": len(claims),
                "citations_count": len(citations),
                "latency_by_node": lat_by_node,
                "total_latency": round(elapsed, 3),
                "final_answer": final_ans,
                "claims": claims,
                "citations": citations,
                "execution_trace": final_state.get("execution_trace", [])
            }
            results.append(result_record)
            
        except Exception as e:
            logger.error(f"Error executing query {q_id}: {str(e)}", exc_info=True)
            results.append({
                "id": q_id,
                "category": category,
                "query": query_text,
                "error": str(e),
                "total_latency": round(time.time() - start_t, 3)
            })

    # Summary Statistics
    latency_summary = {
        "overall": compute_percentiles(total_latencies)
    }
    for n_name, lats in node_latencies.items():
        latency_summary[n_name] = compute_percentiles(lats)
        
    summary = {
        "total_queries": total_queries,
        "successful_executions": len([r for r in results if "error" not in r]),
        "failed_executions": len([r for r in results if "error" in r]),
        "routing_accuracy": round(routing_correct / total_queries * 100, 2),
        "sufficiency_accuracy": round(sufficiency_correct / total_queries * 100, 2),
        "successful_grounded_answers": successful_answers,
        "safe_refusals": refusal_count,
        "regenerations_triggered": regeneration_count,
        "latency_statistics": latency_summary
    }
    
    output_path = project_root / "data" / "metadata" / "langgraph_benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    logger.info("=" * 60)
    logger.info("LANGGRAPH BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Queries: {total_queries}")
    logger.info(f"Routing Accuracy: {summary['routing_accuracy']}%")
    logger.info(f"Sufficiency Accuracy: {summary['sufficiency_accuracy']}%")
    logger.info(f"Safe Refusals: {refusal_count}")
    logger.info(f"Regenerations Triggered: {regeneration_count}")
    logger.info(f"Mean Latency: {latency_summary['overall']['mean']}s (P95: {latency_summary['overall']['p95']}s)")
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
