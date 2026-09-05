"""Document & Knowledge Base Schemas for IP-SAKTI Sahayak API."""
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Metadata for an authoritative legal/AYUSH document in the knowledge base."""
    id: str = Field(..., description="Document identifier/filename")
    title: str = Field(..., description="Full official title of the legal/regulatory document")
    category: str = Field(..., description="Document category (Patent Law, AYUSH, Traditional Knowledge, Biodiversity, International IP)")
    jurisdiction: str = Field(..., description="Jurisdiction (India, WIPO/PCT, EPO, International)")
    authority_tier: str = Field(..., description="Authority tier (Tier 1 Primary Statute, Tier 2 Official Guideline, Tier 3 Institutional)")
    year: Optional[int] = Field(None, description="Enactment or publication year")
    chunk_count: int = Field(..., description="Total structured chunks indexed")


class DocumentListResponse(BaseModel):
    """Knowledge base documents overview."""
    total_documents: int = Field(..., description="Total authoritative documents (22)")
    total_chunks: int = Field(..., description="Total indexed chunks (5,212)")
    categories: List[str] = Field(..., description="Distinct categories represented")
    documents: List[DocumentInfo] = Field(..., description="List of all 22 authoritative documents")


class HealthResponse(BaseModel):
    """System health and component status."""
    status: str = Field("healthy", description="Overall health status")
    backend_connected: bool = Field(True, description="FastAPI server connection")
    knowledge_base_available: bool = Field(True, description="Qdrant and BM25 index status")
    total_chunks_indexed: int = Field(..., description="Total processed chunks (5,212)")
    total_documents: int = Field(22, description="Total authoritative documents")
    orchestrator: str = Field("LangGraph StateGraph", description="Active orchestration engine")
    reranker: str = Field("BAAI/bge-reranker-v2-m3", description="Active cross-encoder reranker")
    embeddings: str = Field("BAAI/bge-m3", description="Active embedding model")
