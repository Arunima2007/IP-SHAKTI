"""Automatic Document Discovery, Hashing, and Categorization."""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import pymupdf

from src.config import (
    RAW_DATA_DIR,
    INVENTORY_JSON_PATH,
    INVENTORY_CSV_PATH,
    METADATA_DIR,
)
from src.ingestion.metadata import DocumentInventoryItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Known document catalog for high-precision domain metadata mapping
KNOWN_DOCUMENTS_CATALOG: Dict[str, Dict] = {
    "01_WHO_Benchmarks_Practice_Ayurveda.pdf": {
        "document_id": "who_benchmarks_practice_ayurveda",
        "document": "WHO Benchmarks for the Practice of Ayurveda",
        "category": "ayurveda",
        "document_type": "guidelines",
        "domain": ["ayurveda", "ayush", "traditional_medicine", "practice_standards"],
        "jurisdiction": "International",
        "language": "en",
        "year": "2022",
        "version": "1.0",
        "source": "World Health Organization (WHO)",
    },
    "02_WHO_Benchmarks_Training_Ayurveda.pdf": {
        "document_id": "who_benchmarks_training_ayurveda",
        "document": "WHO Benchmarks for the Training of Ayurveda",
        "category": "ayurveda",
        "document_type": "guidelines",
        "domain": ["ayurveda", "ayush", "training", "traditional_medicine"],
        "jurisdiction": "International",
        "language": "en",
        "year": "2022",
        "version": "1.0",
        "source": "World Health Organization (WHO)",
    },
    "02_GSR_669_E_Drugs_Fifth_Amendment_Rules_2024.pdf": {
        "document_id": "gsr_669_e_drugs_rules_2024",
        "document": "Drugs (Fifth Amendment) Rules, 2024 (G.S.R. 669(E))",
        "category": "regulatory",
        "document_type": "rules_amendment",
        "domain": ["ayush", "drugs_cosmetics", "regulatory", "traditional_medicine"],
        "jurisdiction": "India",
        "language": "mul",
        "year": "2024",
        "version": "G.S.R. 669(E)",
        "source": "Ministry of Health and Family Welfare / Gazette of India",
    },
    "62789a20b54bdGazette_Notification_Ayurveda_Aahara_09_05_2022.pdf": {
        "document_id": "fssai_ayurveda_aahara_regulations_2022",
        "document": "Food Safety and Standards (Ayurveda Aahara) Regulations, 2022",
        "category": "regulatory",
        "document_type": "regulations",
        "domain": ["ayurveda", "food_safety", "ayurveda_aahara", "regulatory"],
        "jurisdiction": "India",
        "language": "mul",
        "year": "2022",
        "version": "F.No. Stds/SP(Nutraceuticals)/Ayur Aahar-01/2021",
        "source": "Food Safety and Standards Authority of India (FSSAI)",
    },
    "AYUSH_Related_Inventions_Guidelines_2025.pdf": {
        "document_id": "ayush_related_inventions_guidelines_2025",
        "document": "Guidelines for Examination of Patent Applications Relating to AYUSH Related Inventions, 2025",
        "category": "indian_ip",
        "document_type": "examination_guidelines",
        "domain": ["ayush", "ayurveda", "patents", "traditional_knowledge", "patent_examination", "section_3p"],
        "jurisdiction": "India",
        "language": "en",
        "year": "2025",
        "version": "2025 Edition",
        "source": "Office of the Controller General of Patents, Designs and Trade Marks (CGPDTM)",
    },
    "Compendium_Advertising_Claims_Regulations_14_12_2022.pdf": {
        "document_id": "compendium_advertising_claims_regulations_2022",
        "document": "Compendium of Food Safety and Standards (Advertising and Claims) Regulations, 2018",
        "category": "regulatory",
        "document_type": "compendium",
        "domain": ["advertising", "claims", "food_safety", "regulatory"],
        "jurisdiction": "India",
        "language": "en",
        "year": "2022",
        "version": "Updated Dec 2022",
        "source": "FSSAI",
    },
    "Compendium_Licensing_Regulations_04_08_2021.pdf": {
        "document_id": "compendium_licensing_regulations_2021",
        "document": "Compendium of Food Safety and Standards (Licensing and Registration of Food Businesses) Regulations, 2011",
        "category": "regulatory",
        "document_type": "compendium",
        "domain": ["licensing", "registration", "food_safety", "regulatory"],
        "jurisdiction": "India",
        "language": "en",
        "year": "2021",
        "version": "Updated Aug 2021",
        "source": "FSSAI",
    },
    "Drugs_and_Cosmetics_Act_1940.pdf": {
        "document_id": "drugs_and_cosmetics_act_1940",
        "document": "The Drugs and Cosmetics Act, 1940 and Drugs Rules, 1945",
        "category": "regulatory",
        "document_type": "act_legislation",
        "domain": ["drugs_cosmetics", "ayush", "ayurveda_siddha_unani", "legislation"],
        "jurisdiction": "India",
        "language": "en",
        "year": "1940",
        "version": "As amended",
        "source": "Ministry of Health and Family Welfare / Legislative Department",
    },
    "EPO_Guidelines_for_Examination_2026.pdf": {
        "document_id": "epo_guidelines_for_examination_2026",
        "document": "Guidelines for Examination in the European Patent Office",
        "category": "international_ip",
        "document_type": "examination_guidelines",
        "domain": ["patents", "epo", "patent_examination", "international_ip"],
        "jurisdiction": "EPO",
        "language": "en",
        "year": "2026",
        "version": "April 2026 Edition",
        "source": "European Patent Office (EPO)",
    },
    "EPO_PCT_Guidelines_2026.pdf": {
        "document_id": "epo_pct_guidelines_2026",
        "document": "Guidelines for Search and Examination at the European Patent Office as PCT Authority",
        "category": "international_ip",
        "document_type": "examination_guidelines",
        "domain": ["patents", "pct", "epo", "international_search_examination"],
        "jurisdiction": "EPO",
        "language": "en",
        "year": "2026",
        "version": "April 2026 Edition",
        "source": "European Patent Office (EPO)",
    },
    "Guidelines_TK_Biological_Material_2012.pdf": {
        "document_id": "guidelines_tk_biological_material_2012",
        "document": "Guidelines for Processing of Patent Applications Relating to Traditional Knowledge and Biological Material",
        "category": "indian_ip",
        "document_type": "examination_guidelines",
        "domain": ["traditional_knowledge", "biological_resources", "patents", "patent_examination", "section_3p"],
        "jurisdiction": "India",
        "language": "en",
        "year": "2012",
        "version": "2012 Edition",
        "source": "CGPDTM",
    },
    "IP_GR_TK_TCE_Overview.pdf": {
        "document_id": "wipo_ip_gr_tk_tce_overview",
        "document": "Intellectual Property and Genetic Resources, Traditional Knowledge and Traditional Cultural Expressions: An Overview",
        "category": "international_ip",
        "document_type": "overview_guide",
        "domain": ["genetic_resources", "traditional_knowledge", "traditional_cultural_expressions", "wipo"],
        "jurisdiction": "International",
        "language": "en",
        "year": "2020",
        "version": "2020 Edition",
        "source": "World Intellectual Property Organization (WIPO)",
    },
    "Order dated 25-07-2025 enclosing Ayurveda Aahara.pdf": {
        "document_id": "order_fssai_ayurveda_aahara_schedules_2025",
        "document": "FSSAI Direction / Order enclosing Permissible Ingredients and Recipes for Ayurveda Aahara",
        "category": "regulatory",
        "document_type": "order_schedule",
        "domain": ["ayurveda", "ayurveda_aahara", "food_safety", "recipes_schedules", "regulatory"],
        "jurisdiction": "India",
        "language": "mul",
        "year": "2025",
        "version": "Order dated 25-07-2025",
        "source": "FSSAI",
    },
    "PCT_Applicant_Guide_International_Phase.pdf": {
        "document_id": "pct_applicant_guide_international_phase",
        "document": "PCT Applicant's Guide - International Phase",
        "category": "international_ip",
        "document_type": "guide",
        "domain": ["patents", "pct", "wipo", "international_phase", "patent_procedure"],
        "jurisdiction": "WIPO/PCT",
        "language": "en",
        "year": "2026",
        "version": "April 2026 Edition",
        "source": "WIPO",
    },
    "Patent Act-1970.pdf": {
        "document_id": "patent_act_1970",
        "document": "The Patents Act, 1970",
        "category": "indian_ip",
        "document_type": "act_legislation",
        "domain": ["patents", "patent_law", "indian_ip", "legislation", "patentability"],
        "jurisdiction": "India",
        "language": "en",
        "year": "1970",
        "version": "As amended by Act 15 of 2005",
        "source": "Legislative Department, Ministry of Law and Justice, Government of India",
    },
    "The Biological Diversity Act,2002.pdf": {
        "document_id": "biological_diversity_act_2002",
        "document": "The Biological Diversity Act, 2002",
        "category": "indian_ip",
        "document_type": "act_legislation",
        "domain": ["biological_resources", "biodiversity", "traditional_knowledge", "nba_approvals", "legislation"],
        "jurisdiction": "India",
        "language": "en",
        "year": "2002",
        "version": "Act No. 18 of 2003",
        "source": "Legislative Department, Ministry of Law and Justice, Government of India",
    },
    "The Copyright Act,1957.pdf": {
        "document_id": "copyright_act_1957",
        "document": "The Copyright Act, 1957",
        "category": "indian_ip",
        "document_type": "act_legislation",
        "domain": ["copyright", "authors_rights", "indian_ip", "legislation"],
        "jurisdiction": "India",
        "language": "en",
        "year": "1957",
        "version": "As amended by Act 27 of 2012",
        "source": "Legislative Department, Ministry of Law and Justice, Government of India",
    },
    "The Designs Act,2000.pdf": {
        "document_id": "designs_act_2000",
        "document": "The Designs Act, 2000",
        "category": "indian_ip",
        "document_type": "act_legislation",
        "domain": ["design", "industrial_designs", "indian_ip", "legislation"],
        "jurisdiction": "India",
        "language": "en",
        "year": "2000",
        "version": "Act No. 16 of 2000",
        "source": "Legislative Department, Ministry of Law and Justice, Government of India",
    },
    "The Trade Marks Act 1999.pdf": {
        "document_id": "trade_marks_act_1999",
        "document": "The Trade Marks Act, 1999",
        "category": "indian_ip",
        "document_type": "act_legislation",
        "domain": ["trademark", "trademark_law", "indian_ip", "legislation"],
        "jurisdiction": "India",
        "language": "en",
        "year": "1999",
        "version": "Act No. 47 of 1999",
        "source": "Legislative Department, Ministry of Law and Justice, Government of India",
    },
    "WIPO_Documenting_Traditional_Knowledge_Toolkit.pdf": {
        "document_id": "wipo_documenting_tk_toolkit",
        "document": "Documenting Traditional Knowledge - A Toolkit",
        "category": "international_ip",
        "document_type": "toolkit",
        "domain": ["traditional_knowledge", "documentation", "intellectual_property", "wipo"],
        "jurisdiction": "International",
        "language": "en",
        "year": "2017",
        "version": "2017 Edition",
        "source": "WIPO",
    },
    "WIPO_GR_TK_Treaty_2024.pdf": {
        "document_id": "wipo_gr_tk_treaty_2024",
        "document": "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge",
        "category": "international_ip",
        "document_type": "treaty_international_instrument",
        "domain": ["genetic_resources", "traditional_knowledge", "mandatory_disclosure", "treaty", "wipo"],
        "jurisdiction": "International",
        "language": "en",
        "year": "2024",
        "version": "GRATK/DC/7",
        "source": "WIPO Diplomatic Conference",
    },
    "WIPO_Patent_Disclosure_GR_TK.pdf": {
        "document_id": "wipo_patent_disclosure_gr_tk",
        "document": "Key Questions on Patent Disclosure Requirements for Genetic Resources and Traditional Knowledge",
        "category": "international_ip",
        "document_type": "study_guide",
        "domain": ["patents", "patent_disclosure", "genetic_resources", "traditional_knowledge", "wipo"],
        "jurisdiction": "International",
        "language": "en",
        "year": "2020",
        "version": "Second edition 2020",
        "source": "WIPO",
    },
}


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def scan_and_inventory_documents(
    raw_dir: Path = RAW_DATA_DIR,
    overrides: Optional[Dict[str, Dict]] = None,
) -> List[DocumentInventoryItem]:
    """Scan raw directory, analyze PDFs, classify documents, and generate inventory."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted([
        f for f in raw_dir.glob("*.pdf")
        if not f.name.startswith(".") and "__MACOSX" not in str(f)
    ])
    
    logger.info(f"Discovered {len(pdf_files)} PDF documents in {raw_dir}")
    
    inventory: List[DocumentInventoryItem] = []
    
    for pdf_path in pdf_files:
        file_hash = calculate_sha256(pdf_path)
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)
        
        # Analyze text extractability and detect scanned pages
        empty_or_short_pages = []
        total_extracted_chars = 0
        
        for idx in range(page_count):
            page_text = doc[idx].get_text().strip()
            total_extracted_chars += len(page_text)
            # If text is extremely small, mark page for inspection/OCR
            if len(page_text) < 40:
                # Check if page has images
                images = doc[idx].get_images()
                if len(images) > 0:
                    empty_or_short_pages.append(idx + 1)
        
        doc.close()
        
        has_selectable_text = total_extracted_chars > 500
        requires_ocr = len(empty_or_short_pages) > 0 and (len(empty_or_short_pages) / max(1, page_count) > 0.1)
        
        # Classify from catalog or derive defaults
        info = KNOWN_DOCUMENTS_CATALOG.get(pdf_path.name, {})
        if overrides and pdf_path.name in overrides:
            info.update(overrides[pdf_path.name])
            
        doc_id = info.get("document_id", pdf_path.stem.lower().replace(" ", "_").replace("-", "_"))
        category = info.get("category", "other")
        doc_type = info.get("document_type", "unknown")
        domain = info.get("domain", ["general_information"])
        jurisdiction = info.get("jurisdiction", "Unknown")
        language = info.get("language", "en")
        year = info.get("year", "")
        version = info.get("version", "")
        source = info.get("source", "")
        
        item = DocumentInventoryItem(
            document_id=doc_id,
            filename=pdf_path.name,
            path=str(pdf_path.relative_to(raw_dir.parent.parent)),
            file_hash=file_hash,
            page_count=page_count,
            category=category,
            document_type=doc_type,
            domain=domain,
            jurisdiction=jurisdiction,
            language=language,
            year=year,
            version=version,
            source=source,
            has_selectable_text=has_selectable_text,
            requires_ocr=requires_ocr,
            ocr_pages=empty_or_short_pages if requires_ocr else [],
        )
        inventory.append(item)
    
    # Save inventory as JSON
    inventory_data = [item.model_dump() for item in inventory]
    with open(INVENTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(inventory_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved document inventory to {INVENTORY_JSON_PATH}")
    
    # Save inventory as CSV
    df = pd.DataFrame(inventory_data)
    # Convert list column for CSV representation
    df["domain"] = df["domain"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    df["ocr_pages"] = df["ocr_pages"].apply(lambda x: f"{len(x)} pages" if isinstance(x, list) else x)
    df.to_csv(INVENTORY_CSV_PATH, index=False)
    logger.info(f"Saved document inventory CSV to {INVENTORY_CSV_PATH}")
    
    return inventory
