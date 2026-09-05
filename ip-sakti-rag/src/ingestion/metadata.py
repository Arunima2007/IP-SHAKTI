"""Metadata models for IP-SAKTI Sahayak knowledge base."""
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentInventoryItem(BaseModel):
    """Metadata representation for a raw document in the inventory."""
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    path: str = Field(..., description="Relative or absolute path to the file")
    file_hash: str = Field(..., description="SHA-256 hash of the file")
    page_count: int = Field(0, description="Total number of pages")
    category: str = Field("other", description="Primary category of the document")
    document_type: str = Field("unknown", description="Type of document, e.g. act_legislation, guidelines, etc.")
    domain: List[str] = Field(default_factory=list, description="List of domain tags")
    jurisdiction: str = Field("Unknown", description="Jurisdiction: India, WIPO/PCT, EPO, International")
    language: str = Field("en", description="Language of the document, e.g. en, hi, mul")
    year: str = Field("", description="Publication or enactment year")
    version: str = Field("", description="Version or edition details")
    source: str = Field("", description="Source authority or issuing body")
    has_selectable_text: bool = Field(True, description="Whether selectable text was found")
    requires_ocr: bool = Field(False, description="Whether document contains scanned pages requiring OCR")
    ocr_pages: List[int] = Field(default_factory=list, description="List of 1-indexed page numbers requiring OCR")


class ExtractedPage(BaseModel):
    """Extracted text representation of a single PDF page."""
    document_id: str
    page_number: int
    text: str
    source_file: str
    is_ocr: bool = False
    character_count: int = 0


class ChunkMetadata(BaseModel):
    """Comprehensive metadata schema attached to each indexed chunk with explicit structural fields."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk, e.g. patent_act_1970_p12_c1")
    document_id: str = Field(..., description="ID of source document")
    document: str = Field(..., description="Human readable filename/title of the document")
    document_type: str = Field(..., description="Type of document")
    category: str = Field(..., description="Primary category")
    domain: List[str] = Field(default_factory=list, description="Domain tags")
    jurisdiction: str = Field(..., description="Jurisdiction: India, WIPO/PCT, EPO, International")
    
    # Explicit Structural Hierarchical Metadata Fields
    part: Optional[str] = Field(None, description="Part identifier/title, e.g. 'Part A', 'Part I'")
    chapter: Optional[str] = Field(None, description="Chapter identifier/title, e.g. 'Chapter II'")
    section: Optional[str] = Field(None, description="Section number e.g. '3', '3(p)', '25'")
    subsection: Optional[str] = Field(None, description="Subsection identifier, e.g. '(1)', '(2)'")
    clause: Optional[str] = Field(None, description="Clause identifier, e.g. '(a)', '(p)'")
    article: Optional[str] = Field(None, description="Article number e.g. '3', '8', '27'")
    rule: Optional[str] = Field(None, description="Rule number e.g. '43bis', '33bis', 'Rule 5'")
    paragraph: Optional[str] = Field(None, description="Paragraph/paragraph-level identifier e.g. '[1.001]'")
    guideline: Optional[str] = Field(None, description="Guideline identifier e.g. 'Guideline 3.1'")
    regulation: Optional[str] = Field(None, description="Regulation number e.g. 'Regulation 4'")
    schedule: Optional[str] = Field(None, description="Schedule name e.g. 'Schedule I', 'First Schedule'")
    heading: Optional[str] = Field(None, description="Section, article, or chapter heading")
    subheading: Optional[str] = Field(None, description="Subheading if present")
    
    page: int = Field(..., description="1-indexed source page number")
    language: str = Field("en", description="Language")
    source: str = Field("", description="Issuing body or authority")
    year: str = Field("", description="Year of enactment or publication")
    version: str = Field("", description="Version details")
    
    # Optional Patent-specific fields
    patent_number: Optional[str] = None
    title: Optional[str] = None
    applicant: Optional[str] = None
    inventor: Optional[str] = None
    priority_date: Optional[str] = None
    filing_date: Optional[str] = None
    publication_date: Optional[str] = None
    ipc: Optional[str] = None
    cpc: Optional[str] = None
    status: Optional[str] = None


class ProcessedChunk(BaseModel):
    """Complete chunk containing text, breadcrumb context, token count, and metadata."""
    chunk_id: str
    text: str
    context_header: str
    token_count: int
    metadata: ChunkMetadata
