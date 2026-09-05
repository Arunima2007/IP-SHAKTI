"""Comprehensive Benchmark Runner & Failure Analysis for IP-SAKTI Sahayak (Milestone 2).

Executes 36 canonical queries across Vector, BM25, and Hybrid retrieval pipelines.
Evaluates Recall@5, Recall@10, Precision@5, Precision@10, MRR, and NDCG@10.
Analyzes failure modes, generates `retrieval_failures.json`, and verifies dynamic metadata filtering.
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, METADATA_DIR, PROJECT_ROOT
from src.embeddings.embedder import BGEM3Embedder
from src.evaluation.benchmark_dataset import BENCHMARK_QUERIES, BenchmarkQuery
from src.evaluation.evaluator import RetrievalEvaluator, is_chunk_relevant
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.filter_builder import MetadataFilterBuilder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def format_preview(text: str, max_chars: int = 220) -> str:
    """Creates a clean text preview without excessive linebreaks."""
    cleaned = " ".join(text.split())
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "..."
    return cleaned


def analyze_retrieval_failures(
    benchmark_queries: List[BenchmarkQuery],
    vector_results: Dict[int, List[Dict[str, Any]]],
    bm25_results: Dict[int, List[Dict[str, Any]]],
    hybrid_results: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Analyzes retrieval results to identify failure patterns, causes, and fixes."""
    failures = []

    for bq in benchmark_queries:
        fused_cands = hybrid_results.get(bq.query_id, [])
        v_cands = vector_results.get(bq.query_id, [])
        b_cands = bm25_results.get(bq.query_id, [])

        # Check hybrid top-5
        top5_rel = [c for c in fused_cands[:5] if is_chunk_relevant(c, bq)[0]]
        top10_rel = [c for c in fused_cands[:10] if is_chunk_relevant(c, bq)[0]]

        # Failure or degradation check
        if not top5_rel:
            failure_type = "missing_document"
            possible_reason = "No relevant chunks retrieved in top 5."
            rec_fix = "Expand query expansion or adjust rank fusion weights."

            # Classify failure
            if bq.category == "MULTILINGUAL":
                failure_type = "multilingual_failure"
                possible_reason = "Multilingual query semantics diverged from predominantly English legal corpus text."
                rec_fix = "Add transliteration/translation query preprocessing and multilingual domain dictionary expansion."
            elif bq.category == "CROSS_DOMAIN":
                failure_type = "cross_domain_failure"
                possible_reason = "Query spans multiple legal statutes; single retrieval list biased toward highest semantic cluster."
                rec_fix = "Implement multi-query decomposition in Milestone 3 query planner or reciprocal rank fusion with domain grouping."
            elif bq.category == "EXACT_LOOKUP":
                failure_type = "keyword_mismatch"
                possible_reason = "Exact identifier/citation did not match lexical token index format."
                rec_fix = "Augment legal tokenizer regular expressions."
            elif any(c.get("document_id") not in bq.expected_documents for c in fused_cands[:5]):
                failure_type = "wrong_document"
                possible_reason = "Retrieved candidate documents belong to different domain/jurisdiction."
                rec_fix = "Apply stricter metadata pre-filtering or query domain classifier."

            failures.append({
                "query_id": bq.query_id,
                "query": bq.query,
                "category": bq.category,
                "failure_type": failure_type,
                "expected_evidence": {
                    "documents": bq.expected_documents,
                    "sections": bq.expected_sections,
                    "articles": bq.expected_articles,
                    "rules": bq.expected_rules,
                    "keywords": bq.expected_keywords,
                },
                "retrieved_evidence": [
                    {
                        "rank": i + 1,
                        "chunk_id": c.get("chunk_id"),
                        "document": c.get("document"),
                        "section": c.get("section"),
                        "score": c.get("score"),
                        "preview": format_preview(c.get("text", ""), 120),
                    }
                    for i, c in enumerate(fused_cands[:3])
                ],
                "possible_reason": possible_reason,
                "recommended_fix": rec_fix,
            })

    return failures


def run_metadata_filter_tests(hybrid_engine: HybridSearchEngine) -> List[Dict[str, Any]]:
    """Runs explicit validation on dynamic metadata filtering."""
    test_cases = [
        {
            "name": "India-only query",
            "query": "What are the requirements under the Indian Patents Act for biological material disclosure?",
            "expected_jurisdiction": "India",
        },
        {
            "name": "WIPO/PCT query",
            "query": "What is the PCT international phase procedure under WIPO?",
            "expected_jurisdiction": ["WIPO/PCT", "International"],
        },
        {
            "name": "EPO query",
            "query": "What are the inventive step examination rules in the EPO Guidelines?",
            "expected_jurisdiction": "EPO",
        },
        {
            "name": "Ayurveda Aahara query",
            "query": "What are the FSSAI regulations for Ayurveda Aahara food products?",
            "expected_domain": "ayurveda_aahara",
        },
        {
            "name": "Traditional Knowledge query",
            "query": "What are the WIPO principles for protection of Traditional Knowledge?",
            "expected_jurisdiction": ["WIPO/PCT", "International"],
        },
        {
            "name": "Ambiguous query (should NOT aggressively filter)",
            "query": "What is the meaning of inventive step?",
            "expected_jurisdiction": None,
        },
    ]

    filter_results = []
    print("\n" + "=" * 70)
    print("RUNNING DYNAMIC METADATA FILTERING TEST SUITE")
    print("=" * 70)

    for tc in test_cases:
        inferred_filter, conf = MetadataFilterBuilder.infer_filters_from_query(tc["query"])
        res = hybrid_engine.search(tc["query"], top_k=5, debug=True)
        retrieved_docs = list(set(c.get("document_id") for c in res["fused_results"]))
        
        passed = True
        exp_j = tc.get("expected_jurisdiction")
        exp_d = tc.get("expected_domain")

        if exp_j is None and exp_d is None:
            passed = (inferred_filter is None)
        elif exp_j is not None:
            if isinstance(exp_j, list):
                passed = inferred_filter is not None and any(
                    j in (inferred_filter.get("jurisdiction") if isinstance(inferred_filter.get("jurisdiction"), list) else [inferred_filter.get("jurisdiction")])
                    for j in exp_j
                )
            else:
                passed = inferred_filter is not None and (
                    inferred_filter.get("jurisdiction") == exp_j or
                    (isinstance(inferred_filter.get("jurisdiction"), list) and exp_j in inferred_filter.get("jurisdiction"))
                )
        elif exp_d is not None:
            passed = inferred_filter is not None and (
                inferred_filter.get("domain") == exp_d or
                (isinstance(inferred_filter.get("domain"), list) and exp_d in inferred_filter.get("domain"))
            )

        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] Test: {tc['name']}")
        print(f"   Query: \"{tc['query']}\"")
        print(f"   Inferred Filter: {inferred_filter} (Confidence: {conf:.2f})")
        print(f"   Retrieved Top Docs: {retrieved_docs[:3]}")

        filter_results.append({
            "test_name": tc["name"],
            "query": tc["query"],
            "inferred_filter": inferred_filter,
            "confidence": conf,
            "passed": passed,
            "top_docs": retrieved_docs[:3],
        })

    return filter_results


def run_benchmark() -> Dict[str, Any]:
    """Runs full 36-query benchmark suite across Vector, BM25, and Hybrid methods."""
    logger.info("=" * 75)
    logger.info("IP-SAKTI SAHAYAK — COMPREHENSIVE RETRIEVAL BENCHMARK (36 QUERIES)")
    logger.info("=" * 75)

    embedder = BGEM3Embedder()
    vector_store = QdrantVectorStore()
    bm25_engine = BM25SearchEngine()
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        bm25_engine=bm25_engine,
        embedder=embedder,
    )

    vector_results_by_qid: Dict[int, List[Dict[str, Any]]] = {}
    bm25_results_by_qid: Dict[int, List[Dict[str, Any]]] = {}
    hybrid_results_by_qid: Dict[int, List[Dict[str, Any]]] = {}
    detailed_query_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 75)
    print(f"EXECUTING {len(BENCHMARK_QUERIES)} BENCHMARK QUERIES (VECTOR, BM25, HYBRID)")
    print("=" * 75)

    for bq in BENCHMARK_QUERIES:
        t0 = time.time()
        debug_bundle = hybrid_engine.search(
            query=bq.query,
            top_k=25,
            top_k_vector=20,
            top_k_bm25=20,
            fusion_method="rrf",
            debug=True,
        )
        latency_ms = (time.time() - t0) * 1000

        v_res = debug_bundle["vector_results"]
        b_res = debug_bundle["bm25_results"]
        h_res = debug_bundle["fused_results"]

        vector_results_by_qid[bq.query_id] = v_res
        bm25_results_by_qid[bq.query_id] = b_res
        hybrid_results_by_qid[bq.query_id] = h_res

        # Record top 10 for all three methods
        def format_top10(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            formatted = []
            for rank, c in enumerate(cands[:10], start=1):
                meta = c.get("metadata", {})
                formatted.append({
                    "rank": rank,
                    "chunk_id": c.get("chunk_id"),
                    "document": c.get("document") or meta.get("document"),
                    "page": c.get("page") or meta.get("page"),
                    "section": c.get("section") or meta.get("section"),
                    "article": c.get("article") or meta.get("article"),
                    "rule": c.get("rule") or meta.get("rule"),
                    "heading": c.get("heading") or meta.get("heading"),
                    "score": round(float(c.get("score") or c.get("vector_score") or c.get("bm25_score") or 0.0), 5),
                    "retrieval_method": c.get("retrieval_method", "unknown"),
                    "text_preview": format_preview(c.get("text", "")),
                })
            return formatted

        detailed_query_records.append({
            "query_id": bq.query_id,
            "category": bq.category,
            "query": bq.query,
            "description": bq.description,
            "ground_truth_status": bq.ground_truth_status,
            "expected_documents": bq.expected_documents,
            "expected_sections": bq.expected_sections,
            "expected_articles": bq.expected_articles,
            "expected_rules": bq.expected_rules,
            "expected_keywords": bq.expected_keywords,
            "expected_chunk_ids": bq.expected_chunk_ids,
            "latency_ms": round(latency_ms, 2),
            "vector_top_10": format_top10(v_res),
            "bm25_top_10": format_top10(b_res),
            "hybrid_top_10": format_top10(h_res),
        })

        # Print top 10 Hybrid preview
        print(f"\n[Q{bq.query_id:02d} | {bq.category}] {bq.query}")
        print(f"Status: {bq.ground_truth_status} | Latency: {latency_ms:.1f}ms")
        for item in format_top10(h_res)[:3]:
            struct = f"Sec: {item['section']}" if item['section'] else (f"Art: {item['article']}" if item['article'] else "")
            print(f"  Rank {item['rank']}: {item['document']} (p.{item['page']}) {struct} [Score: {item['score']}] -> \"{item['text_preview'][:80]}...\"")

    # Evaluate all three methods across verified queries
    eval_vector = RetrievalEvaluator.evaluate_benchmark(BENCHMARK_QUERIES, vector_results_by_qid, method_name="vector")
    eval_bm25 = RetrievalEvaluator.evaluate_benchmark(BENCHMARK_QUERIES, bm25_results_by_qid, method_name="bm25")
    eval_hybrid = RetrievalEvaluator.evaluate_benchmark(BENCHMARK_QUERIES, hybrid_results_by_qid, method_name="hybrid")

    # Print Comparison Table
    print("\n" + "=" * 85)
    print(f"RETRIEVAL METRICS COMPARISON ({eval_hybrid['total_queries']} VERIFIED BENCHMARK QUERIES)")
    print("=" * 85)
    print(f"{'Method':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'Precision@5':<12} | {'Precision@10':<12} | {'MRR':<10} | {'NDCG@10':<10}")
    print("-" * 85)
    for ev in [eval_vector, eval_bm25, eval_hybrid]:
        print(
            f"{ev['method'].upper():<10} | "
            f"{ev['mean_recall@5']:<10.4f} | "
            f"{ev['mean_recall@10']:<10.4f} | "
            f"{ev['mean_precision@5']:<12.4f} | "
            f"{ev['mean_precision@10']:<12.4f} | "
            f"{ev['mean_mrr']:<10.4f} | "
            f"{ev['mean_ndcg@10']:<10.4f}"
        )
    print("=" * 85)

    # Category breakdown for Hybrid
    print("\nCATEGORY-WISE BREAKDOWN (HYBRID RETRIEVAL):")
    print("-" * 80)
    print(f"{'Category':<22} | {'Count':<6} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<10} | {'NDCG@10':<10}")
    print("-" * 80)
    for cat, m in eval_hybrid["category_metrics"].items():
        print(f"{cat:<22} | {m['count']:<6} | {m['recall@5']:<10.4f} | {m['recall@10']:<10.4f} | {m['mrr']:<10.4f} | {m['ndcg@10']:<10.4f}")

    # Analyze failure cases
    failures = analyze_retrieval_failures(BENCHMARK_QUERIES, vector_results_by_qid, bm25_results_by_qid, hybrid_results_by_qid)
    print(f"\nTotal Significant Failure / Suboptimal Cases: {len(failures)}")

    # Run Metadata Filter Tests
    filter_test_results = run_metadata_filter_tests(hybrid_engine)

    # Export results
    export_payload = {
        "benchmark_summary": {
            "total_questions": len(BENCHMARK_QUERIES),
            "verified_ground_truth_questions": eval_hybrid["total_queries"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics_comparison": {
            "vector": {
                "recall@5": eval_vector["mean_recall@5"],
                "recall@10": eval_vector["mean_recall@10"],
                "precision@5": eval_vector["mean_precision@5"],
                "precision@10": eval_vector["mean_precision@10"],
                "mrr": eval_vector["mean_mrr"],
                "ndcg@10": eval_vector["mean_ndcg@10"],
            },
            "bm25": {
                "recall@5": eval_bm25["mean_recall@5"],
                "recall@10": eval_bm25["mean_recall@10"],
                "precision@5": eval_bm25["mean_precision@5"],
                "precision@10": eval_bm25["mean_precision@10"],
                "mrr": eval_bm25["mean_mrr"],
                "ndcg@10": eval_bm25["mean_ndcg@10"],
            },
            "hybrid": {
                "recall@5": eval_hybrid["mean_recall@5"],
                "recall@10": eval_hybrid["mean_recall@10"],
                "precision@5": eval_hybrid["mean_precision@5"],
                "precision@10": eval_hybrid["mean_precision@10"],
                "mrr": eval_hybrid["mean_mrr"],
                "ndcg@10": eval_hybrid["mean_ndcg@10"],
            },
        },
        "category_breakdown_hybrid": eval_hybrid["category_metrics"],
        "metadata_filter_tests": filter_test_results,
        "query_evaluations": detailed_query_records,
    }

    # Helper serialization
    def default_serializer(o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    # 1. Save detailed benchmark results
    benchmark_res_path = METADATA_DIR / "retrieval_benchmark_results.json"
    with open(benchmark_res_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, default=default_serializer)
    logger.info(f"Saved benchmark results to {benchmark_res_path}")

    # 2. Save failures report
    failures_meta_path = METADATA_DIR / "retrieval_failures.json"
    failures_root_path = PROJECT_ROOT / "retrieval_failures.json"
    failures_payload = {
        "total_failures": len(failures),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "failures": failures,
    }
    with open(failures_meta_path, "w", encoding="utf-8") as f:
        json.dump(failures_payload, f, indent=2, default=default_serializer)
    with open(failures_root_path, "w", encoding="utf-8") as f:
        json.dump(failures_payload, f, indent=2, default=default_serializer)
    logger.info(f"Saved retrieval failures to {failures_root_path}")

    return export_payload


if __name__ == "__main__":
    run_benchmark()
