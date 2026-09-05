"""Unit tests for Milestone 2 retrieval components."""
import pytest
from src.embeddings.embedder import BGEM3Embedder, compute_chunk_content_hash
from src.retrieval.bm25_search import BM25SearchEngine, tokenize_legal_technical
from src.retrieval.filter_builder import MetadataFilterBuilder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.vector_store import QdrantVectorStore
from src.retrieval.legal_identifier_parser import parse, provision_matches, document_matches


def test_content_hash_deterministic():
    text = "Section 3(p) excludes traditional knowledge from patentability."
    meta = {"document": "patent_act_1970", "section": "3(p)"}
    h1 = compute_chunk_content_hash(text, meta)
    h2 = compute_chunk_content_hash(text, meta)
    assert h1 == h2
    assert len(h1) == 64

    # Different text should change hash
    h3 = compute_chunk_content_hash(text + " modified", meta)
    assert h1 != h3


def test_legal_tokenizer():
    sample_text = (
        "Under Section 3(p) and Article 3 of WIPO GR/TK Treaty, "
        "Patent No. 429737 regarding Withania somnifera and PCT Rule 43bis."
    )
    tokens = tokenize_legal_technical(sample_text)

    # Check key legal tokens
    assert "section_3_p" in tokens or "3(p)" in tokens
    assert "article_3" in tokens or "art_3" in tokens
    assert "patent_429737" in tokens or "429737" in tokens
    assert "withania_somnifera" in tokens
    assert "pct_rule_43bis" in tokens or "rule_43bis" in tokens or "43bis" in tokens


def test_bm25_search_in_memory(tmp_path):
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Section 3(p) prohibits patenting of traditional knowledge inventions in India.",
            "metadata": {"document_id": "patent_act_1970", "section": "3(p)", "jurisdiction": "India"},
        },
        {
            "chunk_id": "c2",
            "text": "Rule 43bis requires an international search authority written opinion under PCT.",
            "metadata": {"document_id": "pct_guide", "rule": "43bis", "jurisdiction": "WIPO/PCT"},
        },
        {
            "chunk_id": "c3",
            "text": "Withania somnifera formulation synergism under patent guidelines.",
            "metadata": {"document_id": "ayush_guidelines", "jurisdiction": "India"},
        },
    ]

    engine = BM25SearchEngine(index_path=tmp_path / "test_bm25.pkl")
    engine.build_index(chunks, save=True)

    # Test Section 3(p) lookup
    res = engine.search("Section 3(p)", top_k=2)
    assert len(res) > 0
    assert res[0]["chunk_id"] == "c1"

    # Test Withania somnifera lookup
    res_botanical = engine.search("Withania somnifera", top_k=2)
    assert len(res_botanical) > 0
    assert res_botanical[0]["chunk_id"] == "c3"


def test_metadata_filter_builder():
    filters, conf = MetadataFilterBuilder.infer_filters_from_query("What is the PCT international phase?")
    assert filters is not None
    assert "WIPO/PCT" in filters.get("jurisdiction")
    assert conf >= 0.75

    filters_india, conf_india = MetadataFilterBuilder.infer_filters_from_query("Can an Ayurvedic formulation be patented in India?")
    assert filters_india is not None
    assert filters_india.get("jurisdiction") == "India"


def test_legal_identifier_variants_and_document_normalization():
    for query in ("Section 3(p) of the Patents Act", "sec. 3(p) Patents Act", "s. 3(p) of Patent Act-1970.pdf"):
        parsed = parse(query)
        assert parsed["type"] == "section"
        assert parsed["value"] == "3(p)"
        assert parsed["canonical_title"] == "Patents Act, 1970"


def test_exact_identifier_matches_clause_in_legacy_cross_section_chunk():
    chunk = {
        "document": "Patent Act-1970.pdf",
        "section": "4",  # legacy metadata is the following section
        "text": "(p) an invention which, in effect, is traditional knowledge.\n4. Atomic energy.",
        "metadata": {"document": "Patent Act-1970.pdf", "section": "4"},
    }
    parsed = parse("What does Section 3(p) of the Patents Act state?")
    assert provision_matches(chunk, parsed)
    assert document_matches(chunk, parsed["canonical_title"])
