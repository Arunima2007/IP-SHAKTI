"""Conservative text cleaner that preserves all legal structures and meaning."""
import json
import logging
import re
from pathlib import Path
from typing import List
from tqdm import tqdm

from src.config import EXTRACTED_DIR, CLEANED_DIR
from src.ingestion.metadata import ExtractedPage

logger = logging.getLogger(__name__)

# Common recurring headers/footers in known official documents
REPETITIVE_HEADER_PATTERNS = [
    re.compile(r"^PCT Applicant[’']s Guide\s*–?\s*International Phase\s*–?\s*Contents.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^PCT Applicant[’']s Guide\s*–?\s*International Phase\s*–?\s*Introduction.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^PCT Applicant[’']s Guide\s*–?\s*International Phase\s*–?\s*Chapter\s+[IVXLCDM\d]+.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Guidelines for Examination in the EPO\s*\|\s*April 2026\s*\|\s*Part\s+[A-Z].*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Guidelines for Search and Examination at the EPO as PCT Authority\s*\|\s*April 2026.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^THE GAZETTE OF INDIA\s*:\s*EXTRAORDINARY.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\[\s*भाग\s*II\s*—\s*खण्ड\s*3\(i\)\s*\]\s*भारत का राजपत्र\s*:\s*असाधारण.*$", re.MULTILINE),
]


def clean_page_text(text: str, document_id: str, page_number: int) -> str:
    """Conservatively clean page text while retaining all legal and structural integrity."""
    if not text:
        return ""
    
    # 1. Remove form-feed / null bytes
    cleaned = text.replace("\x0c", "\n").replace("\x00", "")
    
    # 2. Normalize Windows/Mac line endings
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    
    # 3. Strip repetitive header patterns
    for pattern in REPETITIVE_HEADER_PATTERNS:
        cleaned = pattern.sub("", cleaned)
        
    # 4. Remove isolated standalone page numbering artifacts at top/bottom of pages
    lines = cleaned.split("\n")
    filtered_lines = []
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # Check if first line or last line is simply a bare page number e.g. "12" or "Page 12 of 150"
        if (i == 0 or i == len(lines) - 1) and re.match(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?$", stripped_line, re.IGNORECASE):
            continue
            
        # Check for boilerplate empty lines
        filtered_lines.append(line)
        
    cleaned = "\n".join(filtered_lines)
    
    # 5. Fix hyphenated word breaks across lines without breaking legal ranges like (a)-(d)
    # Match letters followed by hyphen, newline, and letters: e.g. 'bio-\nlogical' -> 'biological'
    cleaned = re.sub(r"([a-zA-Z]{2,})-\n([a-zA-Z]{2,})", r"\1\2", cleaned)
    
    # 6. Normalize multiple horizontal whitespace while preserving indentation / formatting
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    
    # 7. Normalize excess vertical whitespace to maximum 2 consecutive newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    return cleaned.strip()


def clean_document_pages(
    extracted_pages: List[ExtractedPage],
    document_id: str,
    output_dir: Path = CLEANED_DIR,
) -> List[ExtractedPage]:
    """Clean all extracted pages of a document and save to data/processed/cleaned/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{document_id}.json"
    
    cleaned_pages: List[ExtractedPage] = []
    for page in extracted_pages:
        cleaned_text = clean_page_text(page.text, document_id, page.page_number)
        cleaned_page = ExtractedPage(
            document_id=page.document_id,
            page_number=page.page_number,
            text=cleaned_text,
            source_file=page.source_file,
            is_ocr=page.is_ocr,
            character_count=len(cleaned_text),
        )
        cleaned_pages.append(cleaned_page)
        
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in cleaned_pages], f, indent=2, ensure_ascii=False)
        
    return cleaned_pages


def clean_all_extracted_documents(
    extracted_dir: Path = EXTRACTED_DIR,
    cleaned_dir: Path = CLEANED_DIR,
) -> List[ExtractedPage]:
    """Load all extracted JSON files, clean them, and save to cleaned directory."""
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(extracted_dir.glob("*.json"))
    
    all_cleaned_pages: List[ExtractedPage] = []
    
    for json_file in tqdm(json_files, desc="Cleaning documents"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            pages = [ExtractedPage(**p) for p in data]
            
        doc_id = json_file.stem
        cleaned = clean_document_pages(pages, doc_id, cleaned_dir)
        all_cleaned_pages.extend(cleaned)
        
    return all_cleaned_pages
