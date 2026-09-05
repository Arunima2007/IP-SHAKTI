"""End-to-end preprocessing pipeline for Milestone 1."""
import json
import logging
import sys
import time
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_DATA_DIR,
    EXTRACTED_DIR,
    CLEANED_DIR,
    CHUNKS_DIR,
    INVENTORY_JSON_PATH,
)
from src.ingestion.inventory import scan_and_inventory_documents
from src.ingestion.pdf_loader import extract_all_documents
from src.ingestion.cleaner import clean_all_extracted_documents
from src.ingestion.chunker import chunk_all_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("IP-SAKTI Sahayak — Milestone 1 Preprocessing Pipeline")
    print("=" * 70)
    
    # 1. Document Discovery & Inventory
    print("\n[Step 1/4] Discovering and inventorying documents...")
    inventory = scan_and_inventory_documents()
    print(f"-> Discovered {len(inventory)} PDF documents.")
    
    # 2. Text Extraction & Selective OCR
    print("\n[Step 2/4] Extracting text with PyMuPDF and OCR fallback...")
    extracted_pages = extract_all_documents(inventory, RAW_DATA_DIR, EXTRACTED_DIR)
    ocr_count = sum(1 for p in extracted_pages if p.is_ocr)
    print(f"-> Extracted {len(extracted_pages)} total pages ({ocr_count} pages processed via OCR).")
    
    # 3. Conservative Cleaning
    print("\n[Step 3/4] Cleaning extracted text (conserving legal and domain structures)...")
    cleaned_pages = clean_all_extracted_documents(EXTRACTED_DIR, CLEANED_DIR)
    print(f"-> Cleaned {len(cleaned_pages)} pages.")
    
    # 4. Structure Detection & Domain-Aware Semantic Chunking
    print("\n[Step 4/4] Detecting structure and generating semantic chunks...")
    chunks = chunk_all_documents(inventory, CLEANED_DIR, CHUNKS_DIR)
    print(f"-> Generated {len(chunks)} domain-aware chunks with metadata.")
    
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.2f} seconds.")
    print("=" * 70)
    
    return inventory, extracted_pages, chunks


if __name__ == "__main__":
    run_pipeline()
