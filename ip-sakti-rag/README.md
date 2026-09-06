# IP-SAKTI Sahayak

IP-SAKTI Sahayak is a multilingual, evidence-grounded legal and Ayurveda IP assistant for Indian patents, AYUSH regulations, Traditional Knowledge, biological diversity, and selected international IP frameworks.

The system combines hybrid retrieval, cross-encoder reranking, query-aware evidence sufficiency, deterministic or Gemini-backed generation, and claim-level citation validation. It refuses when authoritative evidence is not sufficient.

## Architecture

```mermaid
flowchart TD
    U[User / React UI] --> API[FastAPI /api/chat]
    API --> Q[QueryUnderstandingNode]
    Q -->|out of scope| R0[SafeRefusalNode]
    Q --> RET[RetrievalNode]

    RET --> V[Qdrant dense search\nBAAI/bge-m3]
    RET --> B[BM25 sparse search]
    V --> F[RRF fusion]
    B --> F
    F --> RR[RerankingNode\nBAAI/bge-reranker-v2-m3]
    RR --> DS[Diversity-aware evidence selection]
    DS --> ES[EvidenceSufficiencyNode]
    ES -->|insufficient, retry| RET
    ES -->|insufficient after retry| R1[SafeRefusalNode]
    ES -->|sufficient| G[GenerationNode]
    G --> CG[Grounded answer + [E#] citations]
    CG --> CV[CitationValidationNode]
    CV -->|invalid, retry| G
    CV -->|valid| OUT[Answer + citations + evidence metadata]
    CV -->|invalid after retries| R2[SafeRefusalNode]
    R0 --> OUT
    R1 --> OUT
    R2 --> OUT
```

### Request path

1. **Query understanding** detects language, jurisdiction, domains, query type, and exact legal identifiers. Narrow AYUSH patentability questions receive deterministic retrieval vocabulary expansion.
2. **Hybrid retrieval** searches the existing Qdrant dense index and BM25 index, then merges results with reciprocal rank fusion.
3. **Reranking and diversity selection** score candidates with `BAAI/bge-reranker-v2-m3` and limit document/domain monopolization.
4. **Evidence sufficiency** evaluates the evidence set collectively using relevance scores, domain coverage, jurisdiction, authority tier, and relevant evidence count. Exact sections remain strict; conceptual questions do not require a statutory identifier.
5. **Grounded generation** uses Gemini when configured, otherwise the deterministic offline generator. Generated claims cite evidence IDs such as `[E1]`.
6. **Citation validation** checks claim support, provision alignment, entity containment, and fabricated citations. Invalid answers are regenerated once before safe refusal.

## Repository layout

```text
src/
  api/                 FastAPI routes and response schemas
  graph/               LangGraph state, nodes, and routers
  retrieval/           Qdrant, BM25, fusion, exact legal lookup
  reranking/           Cross-encoder and diversity selection
  generation/          Evidence formatting, answer generation, citations
  ingestion/           PDF extraction, cleaning, chunking, metadata
  evaluation/          Retrieval, generation, and LangGraph benchmarks
data/processed/chunks/ Indexed JSON evidence chunks
indexes/               Local Qdrant, BM25, embedding, and reranker artifacts
frontend/              React + TypeScript + Vite UI
tests/                 Unit, integration, legal precision, and safety tests
scripts/               Ingestion, indexing, and benchmark commands
```

## Requirements

- Python 3.11 or newer
- Node.js 20 or newer and npm
- Approximately 4 GB of memory for the local embedding and reranker models
- Existing local indexes under `indexes/`, or permission to build them
- Optional `GEMINI_API_KEY` for Gemini generation; without it, deterministic grounded fallback mode is used

## Setup

From the repository root:

```bash
cd ip-sakti-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env` only if live Gemini generation is required. The local fallback does not require an API key.

## Indexes

The backend expects these local artifacts:

- `indexes/qdrant/` for dense vectors
- `indexes/bm25/bm25_index.pkl` for lexical retrieval
- `indexes/embeddings_cache.pkl` and `indexes/reranker_cache.pkl` for model caches

If the indexes need to be rebuilt:

```bash
python scripts/preprocess_documents.py
python scripts/index_all.py
```

Indexing can download large model weights and may take several minutes. Do not run multiple local Qdrant processes against the same `indexes/qdrant/` directory.

## Run the application

Start the backend from `ip-sakti-rag`, without `--reload` when using the local Qdrant store:

```bash
source .venv/bin/activate
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd ip-sakti-rag/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The API is available at <http://localhost:8000>, with interactive documentation at <http://localhost:8000/docs>.

## API

### Health

```bash
curl http://localhost:8000/health
```

### Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What does Section 3(p) of the Indian Patents Act, 1970 state regarding traditional knowledge?",
    "language": "auto"
  }'
```

The response includes the answer, detected language, query type, domains, citations, validation metrics, and refusal-only diagnostics when evidence is insufficient.

## Tests and benchmarks

Run the complete Python suite:

```bash
python -m pytest -q
```

Useful focused checks:

```bash
python -m pytest -q tests/test_graph.py
python -m pytest -q tests/test_generation.py
python -m pytest -q tests/test_legal_precision.py
python -m pytest -q tests/test_scope_safety.py
```

Run benchmark scripts after the indexes and optional model/API prerequisites are available:

```bash
python scripts/run_retrieval_benchmark.py
python scripts/run_reranking_benchmark.py
python scripts/run_generation_benchmark.py
python scripts/run_langgraph_benchmark.py
```

Benchmark outputs are written under `data/metadata/` and reports under `reports/`.

## Evidence and refusal policy

- Tier 1 primary statutes and treaties and Tier 2 official guidelines are preferred.
- Exact queries such as `Section 3(p)` require an exact authoritative provision match.
- Conceptual queries use collective evidence and do not require an exact section number.
- Current fee queries require an authoritative fee schedule or notification; neighboring statutory provisions are not treated as fee evidence.
- Unsupported, out-of-scope, or citation-invalid answers are refused rather than fabricated.

## Development notes

- Run the backend from `ip-sakti-rag`; otherwise Python may not resolve the `src` package.
- Use one backend process at a time with the local Qdrant path.
- Do not commit generated model caches or local index artifacts unless intentionally publishing them.
