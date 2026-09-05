"""Configuration module for IP-SAKTI Sahayak."""
from pathlib import Path
from typing import Dict, List, Set

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTRACTED_DIR = PROCESSED_DATA_DIR / "extracted"
CLEANED_DIR = PROCESSED_DATA_DIR / "cleaned"
CHUNKS_DIR = PROCESSED_DATA_DIR / "chunks"
ALL_CHUNKS_PATH = CHUNKS_DIR / "all_chunks.json"
CHUNKS_ALL_FILE = ALL_CHUNKS_PATH
METADATA_DIR = DATA_DIR / "metadata"

INDEXES_DIR = PROJECT_ROOT / "indexes"
QDRANT_DIR = INDEXES_DIR / "qdrant"
QDRANT_PATH = QDRANT_DIR
BM25_DIR = INDEXES_DIR / "bm25"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Metadata Files
INVENTORY_JSON_PATH = METADATA_DIR / "document_inventory.json"
INVENTORY_CSV_PATH = METADATA_DIR / "document_inventory.csv"
AYURVEDA_TERMS_PATH = METADATA_DIR / "ayurveda_terms.json"

# Chunking Configuration
TARGET_CHUNK_MIN_TOKENS: int = 400
TARGET_CHUNK_MAX_TOKENS: int = 800
CHUNK_OVERLAP_TOKENS: int = 100
DEFAULT_ENCODING: str = "cl100k_base"

# Recognized Categories
CATEGORIES: Set[str] = {
    "indian_ip",
    "ayush",
    "ayurveda",
    "traditional_knowledge",
    "biological_resources",
    "international_ip",
    "patents",
    "regulatory",
    "other",
}

# Recognized Jurisdictions
JURISDICTIONS: Set[str] = {
    "India",
    "WIPO/PCT",
    "EPO",
    "International",
    "Unknown",
}

# Recognized Query & Domain Classifications
DOMAINS: List[str] = [
    "patentability",
    "prior_art",
    "patent_procedure",
    "trademark",
    "copyright",
    "design",
    "traditional_knowledge",
    "biological_material",
    "ayush_regulation",
    "ayurveda",
    "international_patent",
    "general_information",
]

# Retrieval & Embeddings Configuration
EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
EMBEDDING_DIM: int = 1024
EMBEDDING_BATCH_SIZE: int = 64
EMBEDDINGS_CACHE_PATH: Path = INDEXES_DIR / "embeddings_cache.pkl"


QDRANT_COLLECTION_NAME: str = "ip_sakti_documents"
BM25_INDEX_PATH: Path = BM25_DIR / "bm25_index.pkl"

DEFAULT_TOP_K_VECTOR: int = 25
DEFAULT_TOP_K_BM25: int = 25
DEFAULT_TOP_K_HYBRID: int = 30
RRF_K: int = 60

# Milestone 3 Reranking & Diversity Configuration
import os
RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_BATCH_SIZE: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
RERANKER_MAX_LENGTH: int = 512
RERANKER_CACHE_PATH: Path = INDEXES_DIR / "reranker_cache.pkl"

VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "25"))
BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "25"))
FUSION_TOP_K: int = int(os.getenv("FUSION_TOP_K", "30"))
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "25"))
FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "6"))

MAX_CHUNKS_PER_DOCUMENT: int = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "2"))
MAX_CHUNKS_PER_DOMAIN: int = int(os.getenv("MAX_CHUNKS_PER_DOMAIN", "3"))
DIVERSITY_ENABLED: bool = os.getenv("DIVERSITY_ENABLED", "True").lower() in ("true", "1", "yes")

# Milestone 4 Generation & Citation Configuration
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GENERATION_TEMPERATURE: float = float(os.getenv("GENERATION_TEMPERATURE", "0.0"))
GENERATION_MAX_TOKENS: int = int(os.getenv("GENERATION_MAX_TOKENS", "2048"))
GENERATION_TOP_P: float = float(os.getenv("GENERATION_TOP_P", "0.95"))

# Fallback Refusal Message for Insufficient Evidence
INSUFFICIENT_EVIDENCE_MESSAGE: str = (
    "I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."
)

# Source Authority Hierarchy
# Tier 1: Primary Legislation, Official Gazettes, Official Treaties, Examination Guidelines
# Tier 2: Official Guidelines, Departmental Compendiums, Regulatory Notifications
# Tier 3: Institutional Studies, Toolkits, Training Benchmarks
SOURCE_HIERARCHY: Dict[str, Dict[str, any]] = {
    "patent_act_1970": {"tier": 1, "label": "Tier 1: Primary Statute", "weight": 1.0},
    "biological_diversity_act_2002": {"tier": 1, "label": "Tier 1: Primary Statute", "weight": 1.0},
    "drugs_and_cosmetics_act_1940": {"tier": 1, "label": "Tier 1: Primary Statute", "weight": 1.0},
    "trade_marks_act_1999": {"tier": 1, "label": "Tier 1: Primary Statute", "weight": 1.0},
    "copyright_act_1957": {"tier": 1, "label": "Tier 1: Primary Statute", "weight": 1.0},
    "designs_act_2000": {"tier": 1, "label": "Tier 1: Primary Statute", "weight": 1.0},
    "wipo_gr_tk_treaty_2024": {"tier": 1, "label": "Tier 1: International Treaty", "weight": 1.0},
    "pct_applicant_guide_international_phase": {"tier": 1, "label": "Tier 1: International Treaty Guide", "weight": 0.95},
    "epo_guidelines_for_examination_2026": {"tier": 1, "label": "Tier 1: Examination Guidelines", "weight": 0.95},
    "epo_pct_guidelines_2026": {"tier": 1, "label": "Tier 1: Examination Guidelines", "weight": 0.95},

    "ayush_related_inventions_guidelines_2025": {"tier": 2, "label": "Tier 2: Official Patent Guidelines", "weight": 0.9},
    "guidelines_tk_biological_material_2012": {"tier": 2, "label": "Tier 2: Official Patent Guidelines", "weight": 0.9},
    "fssai_ayurveda_aahara_regulations_2022": {"tier": 2, "label": "Tier 2: Statutory Regulation", "weight": 0.9},
    "order_fssai_ayurveda_aahara_schedules_2025": {"tier": 2, "label": "Tier 2: Official Regulatory Order", "weight": 0.88},
    "compendium_advertising_claims_regulations_2022": {"tier": 2, "label": "Tier 2: Official Regulatory Compendium", "weight": 0.85},
    "compendium_licensing_regulations_2021": {"tier": 2, "label": "Tier 2: Official Regulatory Compendium", "weight": 0.85},
    "gsr_669_e_drugs_rules_2024": {"tier": 2, "label": "Tier 2: Official Gazette Notification", "weight": 0.88},

    "who_benchmarks_practice_ayurveda": {"tier": 3, "label": "Tier 3: Institutional Standard", "weight": 0.75},
    "who_benchmarks_training_ayurveda": {"tier": 3, "label": "Tier 3: Institutional Standard", "weight": 0.75},
    "wipo_ip_gr_tk_tce_overview": {"tier": 3, "label": "Tier 3: Institutional Study", "weight": 0.75},
    "wipo_documenting_tk_toolkit": {"tier": 3, "label": "Tier 3: Institutional Toolkit", "weight": 0.75},
    "wipo_patent_disclosure_gr_tk": {"tier": 3, "label": "Tier 3: Institutional Study", "weight": 0.75},
}

# Exact identifier retrieval boost (higher value prioritises exact legal matches)
EXACT_IDENTIFIER_BOOST = 10.0

# Mapping of user-friendly document names to their canonical titles used in metadata
LEGAL_DOCUMENT_ALIASES = {
    "Patents Act": "The Patents Act, 1970",
    "Designs Act": "The Designs Act, 2000",
    "Biological Diversity Act": "The Biological Diversity Act, 2002",
    "Trade Marks Act": "The Trade Marks Act, 1999",
    "Copyright Act": "The Copyright Act, 1957",
}

# Added configuration constants for exact legal identifier boost and document alias mapping



# Validation & Grounding Thresholds
MIN_CLAIM_SUPPORT_CONFIDENCE: float = 0.50
RELEVANCE_SCORE_THRESHOLD: float = 0.05
MAX_EVIDENCE_CHUNKS: int = 6

# Milestone 5 LangGraph Orchestration Configuration
MAX_RETRIEVAL_RETRIES: int = int(os.getenv("MAX_RETRIEVAL_RETRIES", "1"))
MAX_GENERATION_RETRIES: int = int(os.getenv("MAX_GENERATION_RETRIES", "2"))
MIN_EVIDENCE_SCORE: float = float(os.getenv("MIN_EVIDENCE_SCORE", "0.05"))
MIN_DOMAIN_COVERAGE: float = float(os.getenv("MIN_DOMAIN_COVERAGE", "0.50"))
MIN_SUFFICIENCY_CHUNKS: int = int(os.getenv("MIN_SUFFICIENCY_CHUNKS", "1"))



