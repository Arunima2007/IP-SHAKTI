"""BGE-M3 Dense Multilingual Embedding Generator with Incremental Caching."""
import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_CACHE_PATH,
    INDEXES_DIR,
)

logger = logging.getLogger(__name__)


def compute_chunk_content_hash(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Computes a deterministic SHA-256 hash for chunk text and its metadata."""
    hasher = hashlib.sha256()
    hasher.update(text.encode("utf-8"))
    if metadata:
        meta_str = json.dumps(metadata, sort_keys=True, default=str)
        hasher.update(meta_str.encode("utf-8"))
    return hasher.hexdigest()


class BGEM3Embedder:
    """Embedding manager wrapping BAAI/bge-m3 with caching and batch processing."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_path: Union[str, Path] = EMBEDDINGS_CACHE_PATH,
        device: Optional[str] = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size

        # CPU with multi-threading is fastest and most reliable on Apple Silicon for XLM-RoBERTa
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        torch.set_num_threads(8)
        logger.info(f"Initialized BGEM3Embedder with device: {self.device} (threads={torch.get_num_threads()})")
        self._model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loads SentenceTransformer model to save memory until needed."""
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = 384
            if hasattr(self._model, "tokenizer") and self._model.tokenizer is not None:
                self._model.tokenizer.model_max_length = 384
        return self._model



    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Loads embedding cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded {len(cache)} cached embeddings from {self.cache_path}")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_path}: {e}. Starting fresh.")
                return {}
        return {}

    def _save_cache(self) -> None:
        """Persists embedding cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved {len(self._cache)} embeddings to cache at {self.cache_path}")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generates dense embeddings for a list of raw texts without caching."""
        bs = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str, normalize: bool = True) -> List[float]:
        """Generates a dense embedding for a single query."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        force_recompute: bool = False,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Embeds a list of chunk dictionaries with caching based on content hash.

        Each chunk dict MUST contain:
        - chunk_id: str
        - text: str
        - metadata: dict

        Returns:
            List of chunk dicts, each augmented with a 'vector' key (List[float]).
        """
        bs = batch_size or self.batch_size
        results: List[Dict[str, Any]] = []
        chunks_to_embed: List[Tuple[int, Dict[str, Any], str]] = []  # (index, chunk, hash)

        for i, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            metadata = chunk.get("metadata", {})
            content_hash = compute_chunk_content_hash(text, metadata)

            if (
                not force_recompute
                and chunk_id in self._cache
                and self._cache[chunk_id].get("hash") == content_hash
            ):
                cached_vec = self._cache[chunk_id]["vector"]
                chunk_copy = dict(chunk)
                chunk_copy["vector"] = cached_vec.tolist() if isinstance(cached_vec, np.ndarray) else cached_vec
                results.append((i, chunk_copy))
            else:
                chunks_to_embed.append((i, chunk, content_hash))

        logger.info(
            f"Embedding chunks: {len(results)} found in cache, {len(chunks_to_embed)} to compute."
        )

        if chunks_to_embed:
            texts = [item[1]["text"] for item in chunks_to_embed]
            logger.info(f"Encoding {len(texts)} chunks in batches of {bs}...")
            embeddings = self.embed_texts(
                texts,
                batch_size=bs,
                show_progress=show_progress,
                normalize=True,
            )

            for count, ((orig_idx, chunk, c_hash), vec) in enumerate(zip(chunks_to_embed, embeddings), start=1):
                chunk_id = chunk["chunk_id"]
                vec_list = vec.tolist()

                # Update cache
                self._cache[chunk_id] = {
                    "hash": c_hash,
                    "vector": vec,
                    "document_id": chunk.get("metadata", {}).get("document_id"),
                }

                chunk_copy = dict(chunk)
                chunk_copy["vector"] = vec_list
                results.append((orig_idx, chunk_copy))

                if count % 500 == 0:
                    self._save_cache()

            # Final persist updated cache
            self._save_cache()


        # Sort back to original order
        results.sort(key=lambda x: x[0])
        return [item[1] for item in results]
