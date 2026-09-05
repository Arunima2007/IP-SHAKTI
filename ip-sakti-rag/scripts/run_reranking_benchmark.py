"""Comprehensive Milestone 3 Benchmark Runner: Hybrid Retrieval vs Cross-Encoder Reranking.

Evaluates 42 verified domain queries across Vector, BM25, Hybrid, and Hybrid+Reranker pipelines.
Measures Recall@5, Recall@10, Precision@5, Precision@10, MRR, NDCG@10, and latency percentiles (Avg, Median, P95).
Generates detailed reports and failure analyses.
"""
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR,
    FINAL_TOP_K,
    FUSION_TOP_K,
    METADATA_DIR,
    PROJECT_ROOT,
    RERANK_TOP_K,
    RERANKER_MODEL_NAME,
    VECTOR_TOP_K,
    BM25_TOP_K,
)
from src.evaluation.benchmark_dataset import BENCHMARK_QUERIES, BenchmarkQuery
from src.evaluation.evaluator import RetrievalEvaluator, is_chunk_relevant
from src.reranking.diversity_selector import DiversityAwareSelector
from src.reranking.reranker import CrossEncoderReranker
from src.reranking.reranking_pipeline import RetrievalAndRerankingPipeline
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def format_preview(text: str, max_chars: int = 180) -> str:
    """Cleans and truncates chunk text preview for readable reporting."""
    cleaned = " ".join(text.split())
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "..."
    return cleaned


def calculate_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculates Average, Median (P50), and P95 latency in milliseconds."""
    if not latencies:
        return {"avg": 0.0, "median": 0.0, "p95": 0.0}
    arr = np.array(latencies)
    return {
        "avg": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }


def run_milestone_3_benchmark() -> Dict[str, Any]:
    """Executes full Milestone 3 evaluation."""
    logger.info("=" * 80)
    logger.info("IP-SAKTI SAHAYAK — MILESTONE 3: CROSS-ENCODER RERANKING EVALUATION")
    logger.info("=" * 80)

    # Initialize all pipeline components
    hybrid_engine = HybridSearchEngine()
    reranker = CrossEncoderReranker(model_name=RERANKER_MODEL_NAME)
    diversity_selector = DiversityAwareSelector()
    pipeline = RetrievalAndRerankingPipeline(
        hybrid_engine=hybrid_engine,
        reranker=reranker,
        diversity_selector=diversity_selector,
    )

    vector_results_by_qid: Dict[int, List[Dict[str, Any]]] = {}
    bm25_results_by_qid: Dict[int, List[Dict[str, Any]]] = {}
    hybrid_results_by_qid: Dict[int, List[Dict[str, Any]]] = {}
    reranked_results_by_qid: Dict[int, List[Dict[str, Any]]] = {}

    # Latency tracking lists
    vector_latencies: List[float] = []
    bm25_latencies: List[float] = []
    fusion_latencies: List[float] = []
    reranker_latencies: List[float] = []
    diversity_latencies: List[float] = []
    total_latencies: List[float] = []

    detailed_query_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(f"RUNNING BENCHMARK ACROSS {len(BENCHMARK_QUERIES)} VERIFIED QUERIES")
    print("=" * 80)

    for bq in BENCHMARK_QUERIES:
        t0 = time.time()
        res = pipeline.search_and_rerank(
            query=bq.query,
            top_k_vector=VECTOR_TOP_K,
            top_k_bm25=BM25_TOP_K,
            top_k_fusion=FUSION_TOP_K,
            top_k_rerank=RERANK_TOP_K,
            final_top_k=FINAL_TOP_K,
            debug=True,
        )
        total_time = (time.time() - t0) * 1000

        l_breakdown = res["latency_breakdown_ms"]
        vector_latencies.append(l_breakdown.get("vector_search_ms", 0.0))
        bm25_latencies.append(l_breakdown.get("bm25_search_ms", 0.0))
        fusion_latencies.append(l_breakdown.get("rrf_fusion_ms", 0.0))
        reranker_latencies.append(l_breakdown.get("cross_encoder_ms", 0.0))
        diversity_latencies.append(l_breakdown.get("diversity_selection_ms", 0.0))
        total_latencies.append(total_time)

        v_cands = res.get("vector_results", [])
        b_cands = res.get("bm25_results", [])
        h_cands = res.get("hybrid_candidates", [])
        r_cands = res.get("final_evidence", [])

        vector_results_by_qid[bq.query_id] = v_cands
        bm25_results_by_qid[bq.query_id] = b_cands
        hybrid_results_by_qid[bq.query_id] = h_cands
        reranked_results_by_qid[bq.query_id] = r_cands

        # Check top evidence before vs after
        h_top1 = h_cands[0] if h_cands else {}
        r_top1 = r_cands[0] if r_cands else {}

        print(f"\n[Q{bq.query_id:02d} | {bq.category}] {bq.query}")
        print(f"  Hybrid Top 1:   {h_top1.get('document')} (p.{h_top1.get('page')}) Sec:{h_top1.get('section')} [Score: {h_top1.get('score', 0):.4f}]")
        print(f"  Reranked Top 1: {r_top1.get('document')} (p.{r_top1.get('page')}) Sec:{r_top1.get('section')} [Score: {r_top1.get('reranker_score', 0):.4f}]")
        print(f"  Latency: {total_time:.1f}ms (Vector: {l_breakdown.get('vector_search_ms', 0):.1f}ms, BM25: {l_breakdown.get('bm25_search_ms', 0):.1f}ms, Reranker: {l_breakdown.get('cross_encoder_ms', 0):.1f}ms)")

        detailed_query_records.append({
            "query_id": bq.query_id,
            "category": bq.category,
            "query": bq.query,
            "description": bq.description,
            "hybrid_top_3": [
                {
                    "rank": i + 1,
                    "chunk_id": c.get("chunk_id"),
                    "document": c.get("document"),
                    "page": c.get("page"),
                    "section": c.get("section"),
                    "score": c.get("score"),
                    "preview": format_preview(c.get("text", "")),
                }
                for i, c in enumerate(h_cands[:3])
            ],
            "reranked_top_3": [
                {
                    "rank": i + 1,
                    "chunk_id": c.get("chunk_id"),
                    "document": c.get("document"),
                    "page": c.get("page"),
                    "section": c.get("section"),
                    "score": c.get("reranker_score"),
                    "preview": format_preview(c.get("text", "")),
                }
                for i, c in enumerate(r_cands[:3])
            ],
            "latencies_ms": l_breakdown,
        })

    # Evaluate all 4 methods
    eval_vector = RetrievalEvaluator.evaluate_benchmark(BENCHMARK_QUERIES, vector_results_by_qid, method_name="vector")
    eval_bm25 = RetrievalEvaluator.evaluate_benchmark(BENCHMARK_QUERIES, bm25_results_by_qid, method_name="bm25")
    eval_hybrid = RetrievalEvaluator.evaluate_benchmark(BENCHMARK_QUERIES, hybrid_results_by_qid, method_name="hybrid")
    eval_reranked = RetrievalEvaluator.evaluate_benchmark(BENCHMARK_QUERIES, reranked_results_by_qid, method_name="hybrid_plus_reranker")

    # Latency summary
    latency_summary = {
        "vector_search": calculate_latency_percentiles(vector_latencies),
        "bm25_search": calculate_latency_percentiles(bm25_latencies),
        "rrf_fusion": calculate_latency_percentiles(fusion_latencies),
        "cross_encoder": calculate_latency_percentiles(reranker_latencies),
        "diversity_selection": calculate_latency_percentiles(diversity_latencies),
        "total_pipeline": calculate_latency_percentiles(total_latencies),
    }

    # Print Comparison Table
    print("\n" + "=" * 90)
    print(f"MILESTONE 3 RETRIEVAL & RERANKING EVALUATION ({eval_reranked['total_queries']} VERIFIED QUERIES)")
    print("=" * 90)
    print(f"{'Method':<20} | {'Recall@5':<10} | {'Recall@10':<10} | {'Precision@5':<12} | {'Precision@10':<12} | {'MRR':<10} | {'NDCG@10':<10}")
    print("-" * 90)
    for ev in [eval_vector, eval_bm25, eval_hybrid, eval_reranked]:
        name = ev['method'].upper()
        if name == "HYBRID_PLUS_RERANKER":
            name = "HYBRID + RERANKER"
        print(
            f"{name:<20} | "
            f"{ev['mean_recall@5']:<10.4f} | "
            f"{ev['mean_recall@10']:<10.4f} | "
            f"{ev['mean_precision@5']:<12.4f} | "
            f"{ev['mean_precision@10']:<12.4f} | "
            f"{ev['mean_mrr']:<10.4f} | "
            f"{ev['mean_ndcg@10']:<10.4f}"
        )
    print("=" * 90)

    # Category Breakdown Comparison (Hybrid vs Hybrid + Reranker)
    print("\nCATEGORY-WISE COMPARISON (HYBRID vs HYBRID + RERANKER):")
    print("-" * 95)
    print(f"{'Category':<22} | {'Count':<6} | {'Hybrid R@5':<11} | {'Rerank R@5':<11} | {'Hybrid MRR':<11} | {'Rerank MRR':<11} | {'Rerank NDCG':<11}")
    print("-" * 95)
    for cat in sorted(eval_hybrid["category_metrics"].keys()):
        hm = eval_hybrid["category_metrics"][cat]
        rm = eval_reranked["category_metrics"].get(cat, {})
        print(
            f"{cat:<22} | {hm['count']:<6} | "
            f"{hm['recall@5']:<11.4f} | {rm.get('recall@5', 0):<11.4f} | "
            f"{hm['mrr']:<11.4f} | {rm.get('mrr', 0):<11.4f} | "
            f"{rm.get('ndcg@10', 0):<11.4f}"
        )

    # Latency Summary Table
    print("\n" + "=" * 70)
    print("LATENCY PROFILING (MILLISECONDS)")
    print("=" * 70)
    print(f"{'Stage':<24} | {'Average (ms)':<14} | {'Median (ms)':<14} | {'P95 (ms)':<14}")
    print("-" * 70)
    for stage, stats in latency_summary.items():
        print(f"{stage:<24} | {stats['avg']:<14.2f} | {stats['median']:<14.2f} | {stats['p95']:<14.2f}")

    # Build Reranking Failures Analysis
    failures = []
    for bq in BENCHMARK_QUERIES:
        h_res = hybrid_results_by_qid.get(bq.query_id, [])
        r_res = reranked_results_by_qid.get(bq.query_id, [])

        h_rel = [c for c in h_res[:5] if is_chunk_relevant(c, bq)[0]]
        r_rel = [c for c in r_res[:5] if is_chunk_relevant(c, bq)[0]]

        # If reranked lost recall or rank degraded compared to hybrid
        h_first_rank = next((i + 1 for i, c in enumerate(h_res[:10]) if is_chunk_relevant(c, bq)[0]), None)
        r_first_rank = next((i + 1 for i, c in enumerate(r_res[:10]) if is_chunk_relevant(c, bq)[0]), None)

        if not r_rel or (r_first_rank is not None and h_first_rank is not None and r_first_rank > h_first_rank):
            failures.append({
                "query_id": bq.query_id,
                "query": bq.query,
                "category": bq.category,
                "hybrid_first_rank": h_first_rank,
                "reranked_first_rank": r_first_rank,
                "expected_evidence": {
                    "documents": bq.expected_documents,
                    "sections": bq.expected_sections,
                    "articles": bq.expected_articles,
                    "rules": bq.expected_rules,
                    "keywords": bq.expected_keywords,
                },
                "hybrid_top_1": format_preview(h_res[0].get("text", "")) if h_res else "",
                "reranked_top_1": format_preview(r_res[0].get("text", "")) if r_res else "",
                "failure_type": "rank_degradation" if (r_first_rank and h_first_rank and r_first_rank > h_first_rank) else "evidence_missing_in_top5",
                "possible_reason": "Cross-encoder semantic score assigned higher relevance to descriptive explanation over verbatim statutory clause."
            })

    # Save outputs to JSON
    export_payload = {
        "benchmark_summary": {
            "total_questions": len(BENCHMARK_QUERIES),
            "reranker_model": RERANKER_MODEL_NAME,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics_comparison": {
            "vector": eval_vector,
            "bm25": eval_bm25,
            "hybrid": eval_hybrid,
            "hybrid_plus_reranker": eval_reranked,
        },
        "latency_summary": latency_summary,
        "query_details": detailed_query_records,
        "failures": failures,
    }

    def default_serializer(o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    out_json = METADATA_DIR / "reranking_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, default=default_serializer)
    logger.info(f"Saved reranking benchmark results to {out_json}")

    return export_payload


if __name__ == "__main__":
    run_milestone_3_benchmark()
