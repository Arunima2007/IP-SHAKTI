"""Domain-aware semantic chunker preserving legal, guideline, and hierarchical context with explicit metadata."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import tiktoken

from src.config import (
    CHUNKS_DIR,
    CLEANED_DIR,
    TARGET_CHUNK_MIN_TOKENS,
    TARGET_CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    DEFAULT_ENCODING,
)
from src.ingestion.metadata import ChunkMetadata, ProcessedChunk, DocumentInventoryItem
from src.ingestion.structure_parser import DocumentStructureParser, StructuralElement

logger = logging.getLogger(__name__)

tokenizer = tiktoken.get_encoding(DEFAULT_ENCODING)


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base."""
    return len(tokenizer.encode(text))


def split_text_by_tokens(
    text: str,
    max_tokens: int = TARGET_CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """Split text along natural paragraph boundaries respecting token limits with overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        
    chunks = []
    current_paragraphs = []
    current_token_count = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        # If single paragraph exceeds max_tokens, split on sentences
        if para_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sent_tokens = count_tokens(sentence)
                if current_token_count + sent_tokens > max_tokens and current_paragraphs:
                    chunks.append("\n\n".join(current_paragraphs))
                    overlap_buffer = []
                    overlap_count = 0
                    for prev_p in reversed(current_paragraphs):
                        t_count = count_tokens(prev_p)
                        if overlap_count + t_count <= overlap_tokens:
                            overlap_buffer.insert(0, prev_p)
                            overlap_count += t_count
                        else:
                            break
                    current_paragraphs = overlap_buffer
                    current_token_count = overlap_count
                    
                current_paragraphs.append(sentence)
                current_token_count += sent_tokens
        else:
            if current_token_count + para_tokens > max_tokens and current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                overlap_buffer = []
                overlap_count = 0
                for prev_p in reversed(current_paragraphs):
                    t_count = count_tokens(prev_p)
                    if overlap_count + t_count <= overlap_tokens:
                        overlap_buffer.insert(0, prev_p)
                        overlap_count += t_count
                    else:
                        break
                current_paragraphs = overlap_buffer
                current_token_count = overlap_count
                
            current_paragraphs.append(para)
            current_token_count += para_tokens
            
    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))
        
    return chunks


def build_context_breadcrumb(
    doc_title: str,
    jurisdiction: str,
    part: Optional[str] = None,
    chapter: Optional[str] = None,
    section: Optional[str] = None,
    subsection: Optional[str] = None,
    clause: Optional[str] = None,
    article: Optional[str] = None,
    rule: Optional[str] = None,
    guideline: Optional[str] = None,
    regulation: Optional[str] = None,
    schedule: Optional[str] = None,
    heading: Optional[str] = None,
    page: int = 1,
) -> str:
    """Generate structured provenance breadcrumb prepended to chunk content."""
    parts = [f"Document: {doc_title}", f"Jurisdiction: {jurisdiction}"]
    if part:
        parts.append(f"Part: {part}")
    if chapter:
        parts.append(f"Chapter: {chapter}")
    if schedule:
        parts.append(f"Schedule: {schedule}")
    if section:
        sec_label = f"Section: {section}"
        if subsection:
            sec_label += f"({subsection})"
        if clause:
            sec_label += f"({clause})"
        parts.append(sec_label)
    if article:
        parts.append(f"Article: {article}")
    if rule:
        parts.append(f"Rule: {rule}")
    if guideline:
        parts.append(f"Guideline: {guideline}")
    if regulation:
        parts.append(f"Regulation: {regulation}")
    if heading and heading != section and heading != article and heading != rule:
        parts.append(f"Heading: {heading}")
    parts.append(f"Page: {page}")
    return "[" + " | ".join(parts) + "]"


def coalesce_small_elements(elements: List[StructuralElement], max_coalesce_tokens: int = 500) -> List[StructuralElement]:
    """Coalesce very small adjacent structural elements belonging to the same chapter/part/page into coherent blocks."""
    if not elements:
        return []
        
    coalesced: List[StructuralElement] = []
    current_elem: Optional[StructuralElement] = None
    
    for elem in elements:
        if not elem.content.strip():
            continue
            
        elem_tokens = count_tokens(elem.content)
        
        if current_elem is None:
            current_elem = elem
            continue
            
        current_tokens = count_tokens(current_elem.content)
        
        # Check if elements share same context (e.g. same section or same chapter or page) and total tokens < max_coalesce_tokens
        can_merge = (
            (current_elem.section and current_elem.section == elem.section) or
            (current_elem.element_type == "page_block" and elem.element_type == "page_block" and current_elem.chapter == elem.chapter and current_elem.part == elem.part) or
            (current_tokens < 120 and elem_tokens < 150 and current_elem.chapter == elem.chapter and current_elem.part == elem.part)
        ) and (current_tokens + elem_tokens <= max_coalesce_tokens)
        
        if can_merge:
            current_elem.content = f"{current_elem.content}\n\n{elem.content}".strip()
            current_elem.end_page = max(current_elem.end_page, elem.end_page)
            if not current_elem.heading and elem.heading:
                current_elem.heading = elem.heading
            if not current_elem.section and elem.section:
                current_elem.section = elem.section
            if not current_elem.clause and elem.clause:
                current_elem.clause = elem.clause
        else:
            coalesced.append(current_elem)
            current_elem = elem
            
    if current_elem:
        coalesced.append(current_elem)
        
    return coalesced


def chunk_document(
    doc_info: DocumentInventoryItem,
    cleaned_pages_data: List[Dict],
    output_dir: Path = CHUNKS_DIR,
) -> List[ProcessedChunk]:
    """Chunk a single cleaned document into domain-aware semantic chunks with explicit structured metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{doc_info.document_id}_chunks.json"
    
    # Parse document structure
    raw_elements = DocumentStructureParser.parse_document(
        pages_text=cleaned_pages_data,
        document_id=doc_info.document_id,
        category=doc_info.category,
        document_type=doc_info.document_type,
        jurisdiction=doc_info.jurisdiction,
    )
    
    # Coalesce small adjacent micro-elements
    elements = coalesce_small_elements(raw_elements, max_coalesce_tokens=550)
    
    chunks: List[ProcessedChunk] = []
    chunk_index = 1
    
    for elem in elements:
        if not elem.content.strip():
            continue
            
        elem_tokens = count_tokens(elem.content)
        
        # Split into sub-chunks if larger than TARGET_CHUNK_MAX_TOKENS
        if elem_tokens > TARGET_CHUNK_MAX_TOKENS:
            sub_texts = split_text_by_tokens(elem.content, TARGET_CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)
        else:
            sub_texts = [elem.content]
            
        for sub_text in sub_texts:
            if not sub_text.strip():
                continue
                
            breadcrumb = build_context_breadcrumb(
                doc_title=doc_info.filename,
                jurisdiction=doc_info.jurisdiction,
                part=elem.part,
                chapter=elem.chapter,
                section=elem.section,
                subsection=elem.subsection,
                clause=elem.clause,
                article=elem.article,
                rule=elem.rule,
                guideline=elem.guideline,
                regulation=elem.regulation,
                schedule=elem.schedule,
                heading=elem.heading,
                page=elem.page,
            )
            
            contextualized_text = f"{breadcrumb}\n\n{sub_text}"
            token_count = count_tokens(contextualized_text)
            
            chunk_id = f"{doc_info.document_id}_p{elem.page}_c{chunk_index}"
            
            # Explicit structured metadata assignment
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=doc_info.document_id,
                document=doc_info.filename,
                document_type=doc_info.document_type,
                category=doc_info.category,
                domain=doc_info.domain,
                jurisdiction=doc_info.jurisdiction,
                
                # Structural hierarchy fields
                part=elem.part,
                chapter=elem.chapter,
                section=elem.section,
                subsection=elem.subsection,
                clause=elem.clause,
                article=elem.article,
                rule=elem.rule,
                paragraph=elem.paragraph,
                guideline=elem.guideline,
                regulation=elem.regulation,
                schedule=elem.schedule,
                heading=elem.heading,
                subheading=elem.subheading,
                
                page=elem.page,
                language=doc_info.language,
                source=doc_info.source,
                year=doc_info.year,
                version=doc_info.version,
                
                # Patent details
                patent_number=elem.patent_number,
                applicant=elem.applicant,
                inventor=elem.inventor,
            )
            
            chunk_obj = ProcessedChunk(
                chunk_id=chunk_id,
                text=contextualized_text,
                context_header=breadcrumb,
                token_count=token_count,
                metadata=meta,
            )
            chunks.append(chunk_obj)
            chunk_index += 1
            
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, indent=2, ensure_ascii=False)
        
    logger.info(f"Generated {len(chunks)} chunks for {doc_info.document_id} -> {out_file}")
    return chunks


def chunk_all_documents(
    inventory: List[DocumentInventoryItem],
    cleaned_dir: Path = CLEANED_DIR,
    output_dir: Path = CHUNKS_DIR,
) -> List[ProcessedChunk]:
    """Chunk all cleaned documents and generate combined all_chunks.json."""
    all_chunks: List[ProcessedChunk] = []
    
    for item in inventory:
        clean_file = cleaned_dir / f"{item.document_id}.json"
        if not clean_file.exists():
            logger.warning(f"Cleaned file not found for {item.document_id}")
            continue
            
        with open(clean_file, "r", encoding="utf-8") as f:
            cleaned_pages = json.load(f)
            
        doc_chunks = chunk_document(item, cleaned_pages, output_dir)
        all_chunks.extend(doc_chunks)
        
    # Save master chunks file
    all_chunks_file = output_dir / "all_chunks.json"
    with open(all_chunks_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_chunks], f, indent=2, ensure_ascii=False)
        
    logger.info(f"Total chunks created across all documents: {len(all_chunks)} -> {all_chunks_file}")
    return all_chunks
