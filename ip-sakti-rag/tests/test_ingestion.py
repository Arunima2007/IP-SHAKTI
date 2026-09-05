"""Unit and regression tests for ingestion, extraction, structure parsing, and chunking."""
import json
import pytest
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    INVENTORY_JSON_PATH,
    INVENTORY_CSV_PATH,
    AYURVEDA_TERMS_PATH,
    CLEANED_DIR,
    CHUNKS_DIR,
)
from src.ingestion.cleaner import clean_page_text
from src.ingestion.structure_parser import DocumentStructureParser, RE_SECTION_START, RE_ARTICLE, RE_RULE
from src.ingestion.chunker import count_tokens, split_text_by_tokens, build_context_breadcrumb


def test_inventory_files_exist():
    """Verify inventory and terminology files are present."""
    assert AYURVEDA_TERMS_PATH.exists(), "Ayurveda terms JSON must exist"
    with open(AYURVEDA_TERMS_PATH, "r", encoding="utf-8") as f:
        terms_data = json.load(f)
    assert "terms" in terms_data
    assert len(terms_data["terms"]) >= 15


def test_clean_page_text():
    """Test that cleaner removes headers/footers but keeps legal provisions intact."""
    sample_text = (
        "PCT Applicant's Guide – International Phase – Contents\n"
        "12\n"
        "CHAPTER II\n"
        "3. What are not inventions.—\n"
        "The following are not inventions within the meaning of this Act, namely:—\n"
        "(p) an invention which in effect, is tradi-\n"
        "tional knowledge or which is an aggregation or duplication.\n"
        "Page 12 of 150"
    )
    cleaned = clean_page_text(sample_text, "patent_act_1970", 12)
    assert "PCT Applicant's Guide" not in cleaned
    assert "CHAPTER II" in cleaned
    assert "3. What are not inventions." in cleaned
    assert "traditional knowledge" in cleaned  # Verifies hyphenation repair
    assert "(p)" in cleaned


def test_regex_legal_patterns():
    """Test regex matching on standard legal constructs."""
    sec_line = "3. What are not inventions.—"
    match = RE_SECTION_START.match(sec_line)
    assert match is not None
    assert match.group(1) == "3"
    
    art_line = "ARTICLE 8\nMandatory Patent Disclosure"
    match_art = RE_ARTICLE.match(art_line)
    assert match_art is not None
    assert match_art.group(1) == "8"
    
    rule_line = "Rule 43bis.1\nSearch Report"
    match_rule = RE_RULE.match(rule_line)
    assert match_rule is not None
    assert "43bis" in match_rule.group(1)


def test_breadcrumb_builder():
    """Test provenance context header generation."""
    breadcrumb = build_context_breadcrumb(
        doc_title="Patent Act-1970.pdf",
        jurisdiction="India",
        chapter="Chapter II",
        section="3",
        heading="What are not inventions",
        page=12,
    )
    assert "[Document: Patent Act-1970.pdf" in breadcrumb
    assert "Jurisdiction: India" in breadcrumb
    assert "Section: 3" in breadcrumb
    assert "Page: 12" in breadcrumb


def test_token_splitting():
    """Test token counter and text splitter bounds."""
    text = "This is a test paragraph for token counting. " * 50
    tokens = count_tokens(text)
    assert tokens > 100
    splits = split_text_by_tokens(text, max_tokens=200, overlap_tokens=20)
    for s in splits:
        assert count_tokens(s) <= 220
