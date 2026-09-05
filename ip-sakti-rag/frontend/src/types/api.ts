export interface CitationItem {
  citation_id: string;
  evidence_id: string;
  document: string;
  document_id?: string;
  section?: string;
  page?: number;
  jurisdiction?: string;
  domain?: string;
  source_tier?: string;
  excerpt?: string;
}

export interface ValidationInfo {
  status: 'VALID' | 'INVALID' | 'REFUSAL';
  is_valid: boolean;
  total_claims: number;
  supported_claims: number;
  claim_support_rate: number;
  flagged_issues_count: number;
}

export interface QueryMetadata {
  latency_ms: number;
  generation_attempts: number;
  retrieval_attempts: number;
  node_latencies_ms?: Record<string, number>;
}

export interface ChatResponse {
  answer: string;
  language: string;
  query_type: string;
  jurisdiction?: string;
  domains: string[];
  citations: CitationItem[];
  is_refusal: boolean;
  validation: ValidationInfo;
  metadata: QueryMetadata;
}

export interface ChatRequest {
  query: string;
  language?: string;
}

export interface DocumentInfo {
  id: string;
  title: string;
  category: string;
  jurisdiction: string;
  authority_tier: string;
  year?: number;
  chunk_count: number;
}

export interface DocumentListResponse {
  total_documents: number;
  total_chunks: number;
  categories: string[];
  documents: DocumentInfo[];
}

export interface HealthResponse {
  status: string;
  backend_connected: boolean;
  knowledge_base_available: boolean;
  total_chunks_indexed: number;
  total_documents: number;
  orchestrator: string;
  reranker: string;
  embeddings: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  timestamp: string;
  query?: string;
  response?: ChatResponse;
  error?: string;
}
