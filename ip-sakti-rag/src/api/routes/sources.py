"""Sources and Document overview endpoints for IP-SAKTI Sahayak API."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from src.api.schemas.documents import DocumentInfo, DocumentListResponse
from src.api.schemas.chat import CitationItem
from src.config import CHUNKS_ALL_FILE

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Sources & Documents"])

# Authoritative 22 Document Catalog
DOCUMENT_CATALOG: List[Dict] = [
    {
        "id": "Patent Act-1970.pdf",
        "title": "The Patents Act, 1970 (as amended)",
        "category": "Indian Patent Law",
        "jurisdiction": "India",
        "authority_tier": "Tier 1: Primary Statute",
        "year": 1970,
        "chunk_count": 312
    },
    {
        "id": "The Biological Diversity Act,2002.pdf",
        "title": "The Biological Diversity Act, 2002",
        "category": "Biodiversity & Natural Resources",
        "jurisdiction": "India",
        "authority_tier": "Tier 1: Primary Statute",
        "year": 2002,
        "chunk_count": 185
    },
    {
        "id": "Drugs_and_Cosmetics_Act_1940.pdf",
        "title": "The Drugs and Cosmetics Act, 1940 & Rules 1945",
        "category": "AYUSH & Drug Regulations",
        "jurisdiction": "India",
        "authority_tier": "Tier 1: Primary Statute",
        "year": 1940,
        "chunk_count": 482
    },
    {
        "id": "The Copyright Act,1957.pdf",
        "title": "The Copyright Act, 1957",
        "category": "Indian IP Law",
        "jurisdiction": "India",
        "authority_tier": "Tier 1: Primary Statute",
        "year": 1957,
        "chunk_count": 210
    },
    {
        "id": "The Designs Act,2000.pdf",
        "title": "The Designs Act, 2000",
        "category": "Indian IP Law",
        "jurisdiction": "India",
        "authority_tier": "Tier 1: Primary Statute",
        "year": 2000,
        "chunk_count": 140
    },
    {
        "id": "The Trade Marks Act 1999.pdf",
        "title": "The Trade Marks Act, 1999",
        "category": "Indian IP Law",
        "jurisdiction": "India",
        "authority_tier": "Tier 1: Primary Statute",
        "year": 1999,
        "chunk_count": 198
    },
    {
        "id": "AYUSH_Related_Inventions_Guidelines_2025.pdf",
        "title": "Guidelines for Examination of Patent Applications relating to AYUSH-Related Inventions (2025)",
        "category": "AYUSH & Drug Regulations",
        "jurisdiction": "India",
        "authority_tier": "Tier 2: Official Guidelines",
        "year": 2025,
        "chunk_count": 420
    },
    {
        "id": "Guidelines_TK_Biological_Material_2012.pdf",
        "title": "Guidelines for Processing of Patent Applications Relating to Traditional Knowledge and Biological Material (2012)",
        "category": "Traditional Knowledge",
        "jurisdiction": "India",
        "authority_tier": "Tier 2: Official Guidelines",
        "year": 2012,
        "chunk_count": 165
    },
    {
        "id": "62789a20b54bdGazette_Notification_Ayurveda_Aahara_09_05_2022.pdf",
        "title": "Food Safety and Standards (Ayurveda Aahara) Regulations, 2022",
        "category": "AYUSH & Drug Regulations",
        "jurisdiction": "India",
        "authority_tier": "Tier 2: Official Regulations",
        "year": 2022,
        "chunk_count": 135
    },
    {
        "id": "Order dated 25-07-2025 enclosing Ayurveda Aahara.pdf",
        "title": "FSSAI Directive on Ayurveda Aahara Standardization (2025)",
        "category": "AYUSH & Drug Regulations",
        "jurisdiction": "India",
        "authority_tier": "Tier 2: Official Directive",
        "year": 2025,
        "chunk_count": 290
    },
    {
        "id": "Compendium_Licensing_Regulations_04_08_2021.pdf",
        "title": "Compendium of Regulatory Provisions for ASU Drug Licensing (2021)",
        "category": "AYUSH & Drug Regulations",
        "jurisdiction": "India",
        "authority_tier": "Tier 2: Official Compendium",
        "year": 2021,
        "chunk_count": 215
    },
    {
        "id": "Compendium_Advertising_Claims_Regulations_14_12_2022.pdf",
        "title": "Compendium on Advertising and Claims Regulations for AYUSH Drugs (2022)",
        "category": "AYUSH & Drug Regulations",
        "jurisdiction": "India",
        "authority_tier": "Tier 2: Official Compendium",
        "year": 2022,
        "chunk_count": 178
    },
    {
        "id": "02_GSR_669_E_Drugs_Fifth_Amendment_Rules_2024.pdf",
        "title": "Drugs (Fifth Amendment) Rules, 2024 — ASU Manufacturing Standards",
        "category": "AYUSH & Drug Regulations",
        "jurisdiction": "India",
        "authority_tier": "Tier 2: Official Rules",
        "year": 2024,
        "chunk_count": 92
    },
    {
        "id": "WIPO_GR_TK_Treaty_2024.pdf",
        "title": "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge (2024)",
        "category": "International IP & Treaties",
        "jurisdiction": "International / WIPO",
        "authority_tier": "Tier 1: International Treaty",
        "year": 2024,
        "chunk_count": 88
    },
    {
        "id": "WIPO_Patent_Disclosure_GR_TK.pdf",
        "title": "WIPO Study on Patent Disclosure Requirements relating to Genetic Resources & TK",
        "category": "International IP & Treaties",
        "jurisdiction": "International / WIPO",
        "authority_tier": "Tier 3: Institutional Study",
        "year": 2020,
        "chunk_count": 395
    },
    {
        "id": "WIPO_Documenting_Traditional_Knowledge_Toolkit.pdf",
        "title": "WIPO Documenting Traditional Knowledge Toolkit",
        "category": "Traditional Knowledge",
        "jurisdiction": "International / WIPO",
        "authority_tier": "Tier 3: Institutional Toolkit",
        "year": 2017,
        "chunk_count": 142
    },
    {
        "id": "IP_GR_TK_TCE_Overview.pdf",
        "title": "Intellectual Property and Genetic Resources, Traditional Knowledge & Folklore Overview",
        "category": "Traditional Knowledge",
        "jurisdiction": "International / WIPO",
        "authority_tier": "Tier 3: Institutional Overview",
        "year": 2020,
        "chunk_count": 220
    },
    {
        "id": "PCT_Applicant_Guide_International_Phase.pdf",
        "title": "PCT Applicant's Guide — International & National Phase (WIPO)",
        "category": "International IP & Treaties",
        "jurisdiction": "WIPO/PCT",
        "authority_tier": "Tier 2: Official Guide",
        "year": 2024,
        "chunk_count": 510
    },
    {
        "id": "EPO_Guidelines_for_Examination_2026.pdf",
        "title": "EPO Guidelines for Examination (2026)",
        "category": "International IP & Treaties",
        "jurisdiction": "EPO",
        "authority_tier": "Tier 2: Official Guidelines",
        "year": 2026,
        "chunk_count": 480
    },
    {
        "id": "EPO_PCT_Guidelines_2026.pdf",
        "title": "EPO Guidelines for Search and Examination under PCT (2026)",
        "category": "International IP & Treaties",
        "jurisdiction": "EPO / PCT",
        "authority_tier": "Tier 2: Official Guidelines",
        "year": 2026,
        "chunk_count": 210
    },
    {
        "id": "01_WHO_Benchmarks_Practice_Ayurveda.pdf",
        "title": "WHO Benchmarks for the Practice of Ayurveda",
        "category": "AYUSH & Drug Regulations",
        "jurisdiction": "WHO / International",
        "authority_tier": "Tier 3: Global Benchmarks",
        "year": 2022,
        "chunk_count": 182
    },
    {
        "id": "02_WHO_Benchmarks_Training_Ayurveda.pdf",
        "title": "WHO Benchmarks for Training in Ayurveda",
        "category": "AYUSH & Drug Regulations",
        "jurisdiction": "WHO / International",
        "authority_tier": "Tier 3: Global Benchmarks",
        "year": 2022,
        "chunk_count": 160
    }
]


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """Returns the full catalog of 22 authoritative legal & AYUSH documents."""
    categories = sorted(list(set(d["category"] for d in DOCUMENT_CATALOG)))
    total_chunks = sum(d["chunk_count"] for d in DOCUMENT_CATALOG)
    
    docs = [DocumentInfo(**d) for d in DOCUMENT_CATALOG]
    return DocumentListResponse(
        total_documents=len(docs),
        total_chunks=total_chunks,
        categories=categories,
        documents=docs
    )
