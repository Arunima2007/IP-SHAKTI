# IP-SAKTI Sahayak (आयुष एवं बौद्धिक संपदा सहायक)
### Intelligent Multilingual AI Legal Assistant for Indian Patents, AYUSH Regulations, Traditional Knowledge (TKDL), Biological Diversity & International IP

---

## System Overview

**IP-SAKTI Sahayak** is an authoritative, strictly grounded RAG and LangGraph-orchestrated research system built for the Smart India Hackathon (SIH 2026). It indexes and reasons across **22 primary statutes, official examination guidelines, gazette notifications, and international treaties** (5,212 structured chunks) with claim-level citation validation and source authority hierarchy awareness.

### Core Architecture
- **Knowledge Base**: 22 authoritative legal documents (Patents Act 1970, Biological Diversity Act 2002, Drugs & Cosmetics Act 1940, Ayurveda Aahara 2022, WIPO Treaties, EPO Guidelines).
- **Retrieval Layer**: `BAAI/bge-m3` dense embeddings + Qdrant vector database + BM25 sparse index + Reciprocal Rank Fusion (RRF).
- **Reranking Layer**: `BAAI/bge-reranker-v2-m3` cross-encoder reranking with multi-domain diversity preservation.
- **Orchestration**: LangGraph StateGraph managing query classification, evidence sufficiency checking, controlled retrieval/generation retries, and safe refusal routing.
- **Generation & Citations**: Gemini with strict evidence grounding, bracketed citation resolution, and claim-level verification.
- **API & UI**: FastAPI backend + React 19 + TypeScript + Tailwind CSS with interactive clickable citation evidence inspection and source authority badges (Tier 1 Primary Statute, Tier 2 Official Guideline, Tier 3 Institutional).

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.11+
- Node.js 20+ & npm 10+

### 2. Backend Setup
```bash
# Navigate to project root
cd ip-sakti-rag

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install "fastapi>=0.111.0" "uvicorn[standard]>=0.30.0" "httpx>=0.27.0"

# Configure environment variables
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# Run FastAPI backend server (port 8000)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server (port 5173)
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health, chunk count (5,212), and index availability |
| `POST` | `/api/chat` | Main query endpoint executing LangGraph StateGraph |
| `GET` | `/api/documents` | Complete catalog of 22 authoritative legal documents |

### Example Chat Request:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does Section 3(p) of the Indian Patents Act, 1970 state regarding traditional knowledge?",
    "language": "auto"
  }'
```

---

## Running Automated Tests

```bash
# Run complete test suite (27 tests across M1-M6)
pytest tests/

# Run API tests specifically
pytest tests/test_api.py

# Run frontend build check
cd frontend && npm run build
```

---

## Authoritative Document Pillars
1. **Indian Patent Law**: The Patents Act 1970, The Trade Marks Act 1999, The Copyright Act 1957, The Designs Act 2000.
2. **AYUSH & Drug Regulations**: The Drugs and Cosmetics Act 1940 & Rules 1945, AYUSH-Related Inventions Guidelines 2025, Ayurveda Aahara Regulations 2022, ASU Compendiums.
3. **Biological Diversity**: The Biological Diversity Act 2002 & NBA Section 6 / Section 19 regulations.
4. **Traditional Knowledge**: Guidelines for Patent Applications relating to Traditional Knowledge and Biological Material (2012), TKDL prior art resources.
5. **International IP & Treaties**: WIPO GR/TK Treaty 2024, PCT Applicant's Guide (WIPO), EPO Examination Guidelines 2026, WHO Ayurveda Benchmarks.
