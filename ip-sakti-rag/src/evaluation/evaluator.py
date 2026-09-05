"""Retrieval Evaluation Engine: Recall@K, MRR, Precision@K, NDCG@10 against Ground Truth."""
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.benchmark_dataset import BenchmarkQuery

logger = logging.getLogger(__name__)


@dataclass
class QueryEvaluationResult:
    query_id: int
    query: str
    category: str
    ground_truth_status: str  # "verified" or "needs_manual_verification"
    retrieval_method: str  # "vector", "bm25", "hybrid"
    top_candidates: List[Dict[str, Any]]
    first_relevant_rank: Optional[int] = None
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    reciprocal_rank: float = 0.0
    ndcg_at_10: float = 0.0
    relevant_chunks_found: List[str] = field(default_factory=list)


def is_chunk_relevant(chunk: Dict[str, Any], query_spec: BenchmarkQuery) -> Tuple[bool, str]:
    """
    Evaluates whether a candidate chunk meets ground truth relevance criteria.
    Criteria checked:
    1. Matching expected chunk IDs (exact match)
    2. Matching expected document IDs
    3. Matching expected sections / articles / rules
    4. Matching expected keywords / citations in chunk text
    """
    chunk_id = chunk.get("chunk_id", "")
    if query_spec.expected_chunk_ids and chunk_id in query_spec.expected_chunk_ids:
        return True, f"Exact expected chunk_id match: {chunk_id}"

    doc_id = chunk.get("document_id") or chunk.get("metadata", {}).get("document_id", "")
    text_lower = chunk.get("text", "").lower()
    meta = chunk.get("metadata", {})
    sec = str(meta.get("section", "") or "")
    art = str(meta.get("article", "") or "")
    rule = str(meta.get("rule", "") or "")
    heading = str(meta.get("heading", "") or "").lower()

    # Document check
    doc_match = doc_id in query_spec.expected_documents

    # Section / Article / Rule match
    struct_match = False
    if query_spec.expected_sections and sec in query_spec.expected_sections:
        struct_match = True
    if query_spec.expected_articles and art in query_spec.expected_articles:
        struct_match = True
    if query_spec.expected_rules and any(r.lower() in rule.lower() for r in query_spec.expected_rules):
        struct_match = True

    # Keyword match in text or heading
    kw_matches = [kw for kw in query_spec.expected_keywords if kw.lower() in text_lower or kw.lower() in heading]
    keyword_match = len(kw_matches) >= 1

    # Exact lookup query special handling
    if query_spec.category == "EXACT_LOOKUP":
        if query_spec.expected_keywords and all(kw.lower() in text_lower or kw.lower() in heading for kw in query_spec.expected_keywords):
            return True, f"Exact keyword match: {query_spec.expected_keywords}"
        if struct_match:
            return True, f"Exact structural match: sec={sec}, art={art}, rule={rule}"

    # General query handling
    if doc_match and (struct_match or keyword_match):
        return True, f"Doc '{doc_id}' match with keywords/sections: {kw_matches}"
    elif doc_match and not query_spec.expected_sections and not query_spec.expected_articles and not query_spec.expected_rules:
        # Document matches and general domain
        if keyword_match:
            return True, f"Document match with keyword: {doc_id}"

    return False, "Not relevant"


def compute_ndcg_at_k(relevance_scores: List[int], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k (binary or graded relevance)."""
    if not relevance_scores:
        return 0.0
    
    actual_scores = relevance_scores[:k]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(actual_scores))
    
    # Ideal DCG: sort actual relevance scores descending (assuming at least 1 relevant item exists)
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_scores))
    
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


class RetrievalEvaluator:
    """Computes automated and ground-truth-based retrieval metrics."""

    @staticmethod
    def evaluate_query(
        query_spec: BenchmarkQuery,
        candidates: List[Dict[str, Any]],
        method_name: str = "hybrid",
    ) -> QueryEvaluationResult:
        """Evaluates top candidates for a single benchmark query."""
        relevant_at_5 = 0
        relevant_at_10 = 0
        first_relevant_rank = None
        relevant_cids = []
        binary_relevances = []

        for rank, cand in enumerate(candidates[:10], start=1):
            is_rel, reason = is_chunk_relevant(cand, query_spec)
            if is_rel:
                binary_relevances.append(1)
                relevant_cids.append(cand["chunk_id"])
                if first_relevant_rank is None:
                    first_relevant_rank = rank
                if rank <= 5:
                    relevant_at_5 += 1
                if rank <= 10:
                    relevant_at_10 += 1
            else:
                binary_relevances.append(0)

        r5 = 1.0 if relevant_at_5 > 0 else 0.0
        r10 = 1.0 if relevant_at_10 > 0 else 0.0
        p5 = relevant_at_5 / 5.0
        p10 = relevant_at_10 / 10.0
        mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
        ndcg10 = compute_ndcg_at_k(binary_relevances, k=10)

        return QueryEvaluationResult(
            query_id=query_spec.query_id,
            query=query_spec.query,
            category=query_spec.category,
            ground_truth_status=getattr(query_spec, "ground_truth_status", "verified"),
            retrieval_method=method_name,
            top_candidates=candidates,
            first_relevant_rank=first_relevant_rank,
            recall_at_5=r5,
            recall_at_10=r10,
            precision_at_5=p5,
            precision_at_10=p10,
            reciprocal_rank=mrr,
            ndcg_at_10=ndcg10,
            relevant_chunks_found=relevant_cids,
        )

    @classmethod
    def evaluate_benchmark(
        cls,
        benchmark_queries: List[BenchmarkQuery],
        results_by_query_id: Dict[int, List[Dict[str, Any]]],
        method_name: str = "hybrid",
        only_verified: bool = True,
    ) -> Dict[str, Any]:
        """Runs evaluation over the benchmark suite."""
        eval_results: List[QueryEvaluationResult] = []

        for bq in benchmark_queries:
            if only_verified and getattr(bq, "ground_truth_status", "verified") != "verified":
                continue
            candidates = results_by_query_id.get(bq.query_id, [])
            eval_res = cls.evaluate_query(bq, candidates, method_name=method_name)
            eval_results.append(eval_res)

        total_queries = len(eval_results)
        mean_recall_at_5 = sum(r.recall_at_5 for r in eval_results) / total_queries if total_queries else 0.0
        mean_recall_at_10 = sum(r.recall_at_10 for r in eval_results) / total_queries if total_queries else 0.0
        mean_mrr = sum(r.reciprocal_rank for r in eval_results) / total_queries if total_queries else 0.0
        mean_precision_at_5 = sum(r.precision_at_5 for r in eval_results) / total_queries if total_queries else 0.0
        mean_precision_at_10 = sum(r.precision_at_10 for r in eval_results) / total_queries if total_queries else 0.0
        mean_ndcg_at_10 = sum(r.ndcg_at_10 for r in eval_results) / total_queries if total_queries else 0.0

        # Breakdown by category
        categories = sorted(list(set(r.category for r in eval_results)))
        category_metrics = {}
        for cat in categories:
            cat_results = [r for r in eval_results if r.category == cat]
            c_len = len(cat_results)
            category_metrics[cat] = {
                "count": c_len,
                "recall@5": round(sum(r.recall_at_5 for r in cat_results) / c_len, 4),
                "recall@10": round(sum(r.recall_at_10 for r in cat_results) / c_len, 4),
                "mrr": round(sum(r.reciprocal_rank for r in cat_results) / c_len, 4),
                "precision@5": round(sum(r.precision_at_5 for r in cat_results) / c_len, 4),
                "ndcg@10": round(sum(r.ndcg_at_10 for r in cat_results) / c_len, 4),
            }

        return {
            "method": method_name,
            "total_queries": total_queries,
            "mean_recall@5": round(mean_recall_at_5, 4),
            "mean_recall@10": round(mean_recall_at_10, 4),
            "mean_mrr": round(mean_mrr, 4),
            "mean_precision@5": round(mean_precision_at_5, 4),
            "mean_precision@10": round(mean_precision_at_10, 4),
            "mean_ndcg@10": round(mean_ndcg_at_10, 4),
            "category_metrics": category_metrics,
            "query_results": eval_results,
        }

