"""Cross-Encoder Reranker Module for IP-SAKTI Sahayak (Milestone 3).

Uses multilingual cross-encoders (default: BAAI/bge-reranker-v2-m3) with content-hash caching,
batching, and complete provenance preservation.
"""
import hashlib
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from sentence_transformers import CrossEncoder

from src.config import (
    INDEXES_DIR,
    RERANKER_BATCH_SIZE,
    RERANKER_CACHE_PATH,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    """Computes sigmoid to normalize cross-encoder logits into [0, 1] relevance probabilities."""
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class CrossEncoderReranker:
    """Multilingual Cross-Encoder Reranker with disk caching and provenance preservation."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        max_length: int = RERANKER_MAX_LENGTH,
        batch_size: int = RERANKER_BATCH_SIZE,
        cache_path: Path = RERANKER_CACHE_PATH,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, float] = {}
        self._cache_dirty: bool = False
        self._model: Optional[CrossEncoder] = None

        # Threading optimization for multi-core CPUs
        num_threads = min(8, max(1, os.cpu_count() or 4))
        torch.set_num_threads(num_threads)

        self._load_cache()

    @property
    def model(self) -> CrossEncoder:
        """Lazy loader for cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading Cross-Encoder model '{self.model_name}' (max_length={self.max_length})...")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device="cpu",
            )
            logger.info(f"Cross-Encoder model '{self.model_name}' loaded successfully.")
        return self._model

    def _load_cache(self) -> None:
        """Loads disk cache of previously computed query-chunk scores."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached reranker scores from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Could not load reranker cache from {self.cache_path}: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self) -> None:
        """Flushes in-memory cache to disk if modified."""
        if self._cache_dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved {len(self.cache)} reranker scores to {self.cache_path}")
                self._cache_dirty = False
            except Exception as e:
                logger.warning(f"Failed to save reranker cache: {e}")

    @staticmethod
    def _compute_cache_key(query: str, chunk_id: str, chunk_text: str) -> str:
        """Computes deterministic SHA-256 cache key for a (query, chunk) pair."""
        raw = f"{query.strip().lower()}:::{chunk_id}:::{chunk_text.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        apply_sigmoid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunks against the given query.

        Preserves ALL original chunk metadata and attaches:
        - 'retrieval_score': original score from vector/BM25/RRF
        - 'retrieval_rank': original rank before reranking
        - 'reranker_score': calibrated cross-encoder relevance score
        - 'reranker_raw_score': raw logit from model
        - 'rank': updated post-reranking rank (1-indexed)
        """
        if not candidates:
            return []

        clean_query = query.strip()
        pairs_to_predict: List[Tuple[str, str]] = []
        indices_to_predict: List[int] = []
        cache_keys: List[str] = []

        scores: List[float] = [0.0] * len(candidates)
        raw_scores: List[float] = [0.0] * len(candidates)

        # Check cache first
        for idx, cand in enumerate(candidates):
            cid = cand.get("chunk_id", f"idx_{idx}")
            text = cand.get("text", "")
            ckey = self._compute_cache_key(clean_query, cid, text)
            cache_keys.append(ckey)

            if ckey in self.cache:
                raw_score = self.cache[ckey]
                raw_scores[idx] = raw_score
                scores[idx] = sigmoid(raw_score) if apply_sigmoid else raw_score
            else:
                pairs_to_predict.append((clean_query, text))
                indices_to_predict.append(idx)

        # Compute missing scores in batches
        if pairs_to_predict:
            logger.debug(f"Predicting cross-encoder scores for {len(pairs_to_predict)} uncached pairs...")
            pred_logits = self.model.predict(
                pairs_to_predict,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            for p_idx, orig_idx in enumerate(indices_to_predict):
                raw_logit = float(pred_logits[p_idx])
                raw_scores[orig_idx] = raw_logit
                scores[orig_idx] = sigmoid(raw_logit) if apply_sigmoid else raw_logit

                ckey = cache_keys[orig_idx]
                self.cache[ckey] = raw_logit
                self._cache_dirty = True

            self.save_cache()

        # Build reranked list with full metadata preservation
        reranked_candidates: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            orig_meta = cand.get("metadata", {})
            orig_rank = cand.get("rank", idx + 1)
            orig_score = cand.get("score") or cand.get("vector_score") or cand.get("bm25_score") or 0.0

            # Clone dict to ensure no destructive modification of original object
            item: Dict[str, Any] = {
                # Core Identifiers & Provenance
                "chunk_id": cand.get("chunk_id"),
                "document_id": cand.get("document_id") or orig_meta.get("document_id"),
                "document": cand.get("document") or orig_meta.get("document"),
                "source": cand.get("source") or orig_meta.get("source"),
                "page": cand.get("page") or orig_meta.get("page"),
                "section": cand.get("section") or orig_meta.get("section"),
                "subsection": cand.get("subsection") or orig_meta.get("subsection"),
                "clause": cand.get("clause") or orig_meta.get("clause"),
                "article": cand.get("article") or orig_meta.get("article"),
                "rule": cand.get("rule") or orig_meta.get("rule"),
                "paragraph": cand.get("paragraph") or orig_meta.get("paragraph"),
                "guideline": cand.get("guideline") or orig_meta.get("guideline"),
                "regulation": cand.get("regulation") or orig_meta.get("regulation"),
                "schedule": cand.get("schedule") or orig_meta.get("schedule"),
                "chapter": cand.get("chapter") or orig_meta.get("chapter"),
                "heading": cand.get("heading") or orig_meta.get("heading"),
                "subheading": cand.get("subheading") or orig_meta.get("subheading"),
                "jurisdiction": cand.get("jurisdiction") or orig_meta.get("jurisdiction"),
                "domain": cand.get("domain") or orig_meta.get("domain"),
                "category": cand.get("category") or orig_meta.get("category"),
                "language": cand.get("language") or orig_meta.get("language", "en"),
                "year": cand.get("year") or orig_meta.get("year"),
                "version": cand.get("version") or orig_meta.get("version"),
                "patent_number": cand.get("patent_number") or orig_meta.get("patent_number"),
                "title": cand.get("title") or orig_meta.get("title"),
                "applicant": cand.get("applicant") or orig_meta.get("applicant"),
                "inventor": cand.get("inventor") or orig_meta.get("inventor"),
                "priority_date": cand.get("priority_date") or orig_meta.get("priority_date"),
                "filing_date": cand.get("filing_date") or orig_meta.get("filing_date"),
                "publication_date": cand.get("publication_date") or orig_meta.get("publication_date"),
                "ipc": cand.get("ipc") or orig_meta.get("ipc"),
                "cpc": cand.get("cpc") or orig_meta.get("cpc"),
                "status": cand.get("status") or orig_meta.get("status"),
                # Text and Context
                "text": cand.get("text", ""),
                "context_header": cand.get("context_header", ""),
                "token_count": cand.get("token_count", 0),
                "metadata": orig_meta,
                # Scores
                "retrieval_method": cand.get("retrieval_method", "hybrid"),
                "retrieval_score": round(float(orig_score), 6),
                "retrieval_rank": orig_rank,
                "reranker_score": round(float(scores[idx]), 6),
                "reranker_raw_score": round(float(raw_scores[idx]), 6),
                "score": round(float(scores[idx]), 6),  # primary score is now reranker_score
            }
            reranked_candidates.append(item)

        # Sort descending by reranker_score
        reranked_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Update 1-indexed ranks
        for rank, item in enumerate(reranked_candidates, start=1):
            item["rank"] = rank

        if top_k is not None and top_k > 0:
            return reranked_candidates[:top_k]
        return reranked_candidates
