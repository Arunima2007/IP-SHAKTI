"""Qdrant Vector Store implementation for IP-SAKTI Sahayak."""
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from src.config import (
    DEFAULT_TOP_K_VECTOR,
    EMBEDDING_DIM,
    QDRANT_COLLECTION_NAME,
    QDRANT_DIR,
)

logger = logging.getLogger(__name__)


def generate_point_id(chunk_id: str) -> str:
    """Generates a deterministic UUID from chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantVectorStore:
    """Manages Qdrant collection creation, indexing, payload indices, and search."""

    _client_cache: Dict[str, QdrantClient] = {}

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION_NAME,
        storage_path: Union[str, Path] = QDRANT_DIR,
        dim: int = EMBEDDING_DIM,
    ):
        self.collection_name = collection_name
        self.storage_path = Path(storage_path)
        self.dim = dim

        self.storage_path.mkdir(parents=True, exist_ok=True)
        path_str = str(self.storage_path.resolve())
        if path_str not in self._client_cache:
            logger.info(f"Initializing QdrantClient at {self.storage_path}...")
            self._client_cache[path_str] = QdrantClient(path=path_str)
        self.client = self._client_cache[path_str]

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates collection and payload indices if they do not already exist."""
        existing_collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing_collections:
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' (dim={self.dim}, distance=Cosine)..."
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            self._create_payload_indices()
        else:
            logger.info(f"Qdrant collection '{self.collection_name}' already exists.")

    def _create_payload_indices(self) -> None:
        """Creates keyword/text payload indices for efficient filtering."""
        fields_to_index = [
            ("jurisdiction", models.PayloadSchemaType.KEYWORD),
            ("category", models.PayloadSchemaType.KEYWORD),
            ("domain", models.PayloadSchemaType.KEYWORD),
            ("document_type", models.PayloadSchemaType.KEYWORD),
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("document", models.PayloadSchemaType.KEYWORD),
            ("language", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.KEYWORD),
            ("section", models.PayloadSchemaType.KEYWORD),
            ("article", models.PayloadSchemaType.KEYWORD),
            ("rule", models.PayloadSchemaType.KEYWORD),
            ("chapter", models.PayloadSchemaType.KEYWORD),
            ("guideline", models.PayloadSchemaType.KEYWORD),
            ("patent_number", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, field_type in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                logger.debug(f"Payload index for '{field_name}' already exists or notice: {e}")

    def index_chunks(
        self,
        chunks_with_vectors: List[Dict[str, Any]],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        Indexes chunks into Qdrant collection.

        Each item must contain 'chunk_id', 'text', 'vector', and 'metadata'.
        """
        points: List[PointStruct] = []
        for item in chunks_with_vectors:
            chunk_id = item["chunk_id"]
            vector = item["vector"]
            text = item["text"]
            metadata = item.get("metadata", {})

            # Prepare flat payload + full metadata
            payload = {
                "chunk_id": chunk_id,
                "text": text,
                "context_header": item.get("context_header", ""),
                "token_count": item.get("token_count", 0),
                "document_id": metadata.get("document_id"),
                "document": metadata.get("document"),
                "document_type": metadata.get("document_type"),
                "category": metadata.get("category"),
                "domain": metadata.get("domain", []),
                "jurisdiction": metadata.get("jurisdiction"),
                "part": metadata.get("part"),
                "chapter": metadata.get("chapter"),
                "section": metadata.get("section"),
                "subsection": metadata.get("subsection"),
                "clause": metadata.get("clause"),
                "article": metadata.get("article"),
                "rule": metadata.get("rule"),
                "paragraph": metadata.get("paragraph"),
                "guideline": metadata.get("guideline"),
                "regulation": metadata.get("regulation"),
                "schedule": metadata.get("schedule"),
                "heading": metadata.get("heading"),
                "subheading": metadata.get("subheading"),
                "page": metadata.get("page"),
                "language": metadata.get("language"),
                "source": metadata.get("source"),
                "year": metadata.get("year"),
                "patent_number": metadata.get("patent_number"),
                "title": metadata.get("title"),
                "metadata": metadata,
            }

            point_id = generate_point_id(chunk_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert
        total_points = len(points)
        logger.info(f"Upserting {total_points} points into '{self.collection_name}'...")
        iterator = range(0, total_points, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Qdrant Indexing")

        for i in iterator:
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

        count = self.get_count()
        logger.info(f"Successfully indexed points. Collection count: {count}")
        return count

    def search(
        self,
        query_vector: List[float],
        top_k: int = DEFAULT_TOP_K_VECTOR,
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes dense vector similarity search in Qdrant.

        Returns structured candidate list.
        """
        # Query Qdrant
        if hasattr(self.client, "query_points"):
            query_res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=filters,
                limit=top_k,
                with_payload=True,
            )
            scored_points = query_res.points
        else:
            scored_points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filters,
                limit=top_k,
                with_payload=True,
            )

        candidates = []
        for scored_point in scored_points:
            payload = scored_point.payload or {}
            chunk_metadata = payload.get("metadata", {})


            candidate = {
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "document": payload.get("document"),
                "page": payload.get("page"),
                "section": payload.get("section"),
                "article": payload.get("article"),
                "rule": payload.get("rule"),
                "guideline": payload.get("guideline"),
                "chapter": payload.get("chapter"),
                "heading": payload.get("heading"),
                "jurisdiction": payload.get("jurisdiction"),
                "category": payload.get("category"),
                "domain": payload.get("domain", []),
                "score": float(scored_point.score),
                "retrieval_method": "vector",
                "text": payload.get("text", ""),
                "metadata": chunk_metadata,
            }
            candidates.append(candidate)

        return candidates

    def get_count(self) -> int:
        """Returns total number of points in the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0
