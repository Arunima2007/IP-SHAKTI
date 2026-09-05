"""PDF text extraction module using PyMuPDF with selective OCR fallback."""
import json
import logging
from pathlib import Path
from typing import List, Optional
import pymupdf
from tqdm import tqdm

from src.config import EXTRACTED_DIR, RAW_DATA_DIR
from src.ingestion.metadata import ExtractedPage, DocumentInventoryItem
from src.ingestion.ocr import extract_page_with_ocr, should_ocr_page

logger = logging.getLogger(__name__)


def extract_document_pages(
    pdf_path: Path,
    document_id: str,
    output_dir: Path = EXTRACTED_DIR,
    force_ocr_pages: Optional[List[int]] = None,
) -> List[ExtractedPage]:
    """Extract page-by-page text from a single PDF document."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{document_id}.json"
    
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    extracted_pages: List[ExtractedPage] = []
    
    force_set = set(force_ocr_pages or [])
    
    logger.info(f"Extracting {pdf_path.name} ({total_pages} pages)...")
    
    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        
        # Primary extraction with PyMuPDF
        text = page.get_text()
        is_ocr = False
        
        # Check if OCR is needed
        if page_num in force_set or should_ocr_page(page, min_char_threshold=50):
            ocr_text = extract_page_with_ocr(page)
            if len(ocr_text) > len(text):
                text = ocr_text
                is_ocr = True
        
        page_obj = ExtractedPage(
            document_id=document_id,
            page_number=page_num,
            text=text,
            source_file=pdf_path.name,
            is_ocr=is_ocr,
            character_count=len(text.strip()),
        )
        extracted_pages.append(page_obj)
        
    doc.close()
    
    # Save extracted document to JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in extracted_pages], f, indent=2, ensure_ascii=False)
        
    logger.info(f"Extracted {len(extracted_pages)} pages for {document_id} -> {out_file}")
    return extracted_pages


def extract_all_documents(
    inventory: List[DocumentInventoryItem],
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = EXTRACTED_DIR,
) -> List[ExtractedPage]:
    """Extract all documents listed in the inventory."""
    all_pages: List[ExtractedPage] = []
    for item in tqdm(inventory, desc="Extracting PDFs"):
        pdf_path = raw_dir / item.filename
        if not pdf_path.exists():
            # Try path from item
            pdf_path = raw_dir.parent.parent / item.path
            
        if not pdf_path.exists():
            logger.error(f"File not found: {pdf_path}")
            continue
            
        pages = extract_document_pages(
            pdf_path=pdf_path,
            document_id=item.document_id,
            output_dir=output_dir,
            force_ocr_pages=item.ocr_pages,
        )
        all_pages.extend(pages)
        
    return all_pages
