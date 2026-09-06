"""Unit and Integration Tests for Milestone 4 Grounded Answer Generation & Citation Verification."""
import pytest
from src.generation.evidence_formatter import EvidenceFormatter
from src.generation.citation_engine import CitationEngine
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_validator import ClaimCitationValidator
from src.generation.generation_pipeline import GenerationPipeline, GroundedAnswerResult
from src.config import INSUFFICIENT_EVIDENCE_MESSAGE


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "patent_act_1970_chunk_0042",
            "document_id": "patent_act_1970",
            "document": "Patent Act-1970.pdf",
            "jurisdiction": "India",
            "page": 9,
            "page_start": 9,
            "page_end": 9,
            "section": "3(p)",
            "heading": "What are not inventions",
            "text": "The following are not inventions within the meaning of this Act: (p) an invention which in effect, is traditional knowledge or which is an aggregation or duplication of known properties of traditionally known component or components.",
            "rerank_score": 0.9650,
            "category": "patents"
        },
        {
            "chunk_id": "ayush_guidelines_chunk_0015",
            "document_id": "ayush_related_inventions_guidelines_2025",
            "document": "Guidelines-for-Examination-of-Patent-Applications-related-to-AYUSH-2025.pdf",
            "jurisdiction": "India",
            "page": 14,
            "page_start": 14,
            "page_end": 14,
            "section": "3(e)",
            "heading": "Assessment of Synergism in Herbal Compositions",
            "text": "For herbal compositions involving known medicinal plants, the applicant must demonstrate synergistic efficacy supported by comparative experimental data to overcome objections under Section 3(e) and 3(p).",
            "rerank_score": 0.8920,
            "category": "ayush"
        },
        {
            "chunk_id": "wipo_treaty_chunk_0008",
            "document_id": "wipo_gr_tk_treaty_2024",
            "document": "WIPO Treaty on GR and TK 2024.pdf",
            "jurisdiction": "WIPO/PCT",
            "page": 5,
            "page_start": 5,
            "page_end": 5,
            "article": "3",
            "heading": "Mandatory Disclosure Requirement",
            "text": "Where the claimed invention in a patent application is based on genetic resources, each Contracting Party shall require applicants to disclose the country of origin of the genetic resources.",
            "rerank_score": 0.7850,
            "category": "international_ip"
        }
    ]


def test_evidence_formatter(sample_chunks):
    formatter = EvidenceFormatter(max_evidence_chunks=5)
    formatted_text, evidence_map, conflicts = formatter.format_evidence(sample_chunks, query="What is Section 3(p)?")

    assert "[E1]" in formatted_text
    assert "[E2]" in formatted_text
    assert "[E3]" in formatted_text
    assert "E1" in evidence_map
    assert "E2" in evidence_map
    assert "E3" in evidence_map

    # Authority tier tagging: Tier 1 items come before Tier 2 items
    assert evidence_map["E1"]["tier"] == 1
    assert "Tier 1" in evidence_map["E1"]["tier_label"]
    assert evidence_map["E2"]["tier"] == 1  # WIPO Treaty is Tier 1
    assert evidence_map["E3"]["tier"] == 2  # AYUSH Guidelines is Tier 2
    assert "3(p)" in evidence_map["E1"]["section"]

    # Conflict detection between India jurisdiction and WIPO jurisdiction
    assert len(conflicts) >= 1
    assert conflicts[0]["type"] in ("jurisdictional_variation", "regulatory_boundary")


def test_citation_engine(sample_chunks):
    formatter = EvidenceFormatter()
    _, evidence_map, _ = formatter.format_evidence(sample_chunks)

    engine = CitationEngine()
    test_text = """### Answer
Section 3(p) excludes traditional knowledge from patentability [E1].

### Explanation
Under the Patents Act, 1970, an invention aggregating known traditional properties is not patentable [E1].
International applications must disclose genetic resource origins [E2].
Furthermore, herbal compositions must demonstrate synergistic efficacy under Section 3(e) [E3].
"""
    final_text, citations = engine.convert_answer_citations(test_text, evidence_map)

    assert "[1]" in final_text
    assert "[2]" in final_text
    assert "[3]" in final_text
    assert "[E1]" not in final_text
    assert len(citations) == 3

    # Validate structured citation objects
    c1 = citations[0]
    assert c1["citation_id"] == "C1"
    assert c1["evidence_id"] == "E1"
    assert "Patents Act, 1970" in c1["formatted_citation"]
    assert "Section 3(p)" in c1["formatted_citation"]
    assert "### Sources" in final_text


def test_answer_generator_fallback(sample_chunks):
    formatter = EvidenceFormatter()
    fmt_text, evidence_map, conflicts = formatter.format_evidence(sample_chunks)

    generator = AnswerGenerator()
    query = "What does Section 3(p) state regarding traditional knowledge?"
    answer, meta = generator.generate(query, fmt_text, evidence_map, conflicts)

    assert "### Short answer" in answer or "### Answer" in answer
    assert "### Explanation" in answer
    assert "[E1]" in answer
    assert meta["status"] in ("success_offline_grounded", "success_live_llm")


def test_answer_generator_refusal_on_empty():
    generator = AnswerGenerator()
    answer, meta = generator.generate(
        query="What is the patent registration procedure in Brazil?",
        formatted_evidence="",
        evidence_map={}
    )
    assert INSUFFICIENT_EVIDENCE_MESSAGE in answer
    assert meta["status"] == "refused_empty_evidence"


def test_citation_validator_supported(sample_chunks):
    formatter = EvidenceFormatter()
    _, evidence_map, _ = formatter.format_evidence(sample_chunks)

    engine = CitationEngine()
    test_text = """### Answer
Section 3(p) of the Patents Act excludes traditional knowledge from patentability [E1].

### Explanation
An invention which is an aggregation or duplication of known properties of traditional knowledge is not patentable [E1].
Where an invention is based on genetic resources, applicants shall disclose the country of origin [E2].
Applicants combining herbal components must demonstrate synergistic efficacy under Section 3(e) [E3].
"""
    final_text, citations = engine.convert_answer_citations(test_text, evidence_map)

    validator = ClaimCitationValidator()
    result = validator.validate_answer(final_text, evidence_map, citations)

    assert result["is_valid"] is True
    assert result["metrics"]["total_claims"] >= 2
    assert result["metrics"]["claim_support_rate"] >= 0.80
    assert result["metrics"]["citation_precision"] >= 0.80
    assert result["metrics"]["unsupported_claim_rate"] <= 0.20



def test_citation_validator_detects_hallucination(sample_chunks):
    formatter = EvidenceFormatter()
    _, evidence_map, _ = formatter.format_evidence(sample_chunks)

    # Intentionally corrupt text with fabricated section and unsupported assertions
    corrupt_text = """### Answer
Under Section 999(z) of the Patent Act, all software code is automatically granted a 50-year patent term [1].

### Explanation
The Act states that quantum computers must be registered with the NASA space center [1].
"""
    cit_objects = [
        {
            "citation_id": "C1",
            "citation_number": 1,
            "evidence_id": "E1",
            "chunk_id": sample_chunks[0]["chunk_id"],
            "document": "Patent Act-1970.pdf",
            "section": "3(p)",
            "page": "9",
            "formatted_citation": "Patents Act, 1970 — Section 3(p), p. 9"
        }
    ]

    validator = ClaimCitationValidator()
    result = validator.validate_answer(corrupt_text, evidence_map, cit_objects)

    assert len(result["flagged_issues"]) >= 1
    assert result["metrics"]["unsupported_claim_rate"] > 0.50
    # Remediation should have sanitized or refused
    assert (result["sanitized_answer"] == INSUFFICIENT_EVIDENCE_MESSAGE) or (result["is_valid"] is False)


def test_citation_validator_rejects_wrong_cited_provision(sample_chunks):
    validator = ClaimCitationValidator()
    wrong_evidence = dict(sample_chunks[0])
    wrong_evidence["section"] = "4"
    result = validator.validate_answer(
        "Section 3(p) excludes traditional knowledge. [1]",
        {"E1": wrong_evidence},
        [{"citation_id": "C1", "citation_number": 1, "evidence_id": "E1"}],
    )
    assert result["is_valid"] is False
    assert any(issue["type"] == "citation_mismatch" for issue in result["flagged_issues"])


def test_exact_lookup_does_not_emit_neighboring_section_claim(sample_chunks):
    formatter = EvidenceFormatter()
    formatted_evidence, evidence_map, conflicts = formatter.format_evidence([sample_chunks[0]])
    answer, _ = AnswerGenerator().generate(
        "What does Section 3(p) of the Patents Act state?",
        formatted_evidence,
        evidence_map,
        conflicts,
    )
    assert "Specifically," not in answer
    assert "Section 3(p)" in answer


def test_ayush_patentability_answer_is_conditional(sample_chunks):
    formatter = EvidenceFormatter()
    ayush_chunk = dict(sample_chunks[1])
    ayush_chunk["text"] = (
        "Guiding Principle 5: In case a multi-ingredient formulation is known to have "
        "a specific therapeutic activity as per the prior art, merely selecting one "
        "or more ingredient for the same said therapeutic activity cannot be considered as inventive."
    )
    formatted_evidence, evidence_map, conflicts = formatter.format_evidence([ayush_chunk])
    answer, _ = AnswerGenerator().generate(
        "Can a classical Ayurvedic formulation described in the Ayurvedic Pharmacopoeia of India be patented as an invention?",
        formatted_evidence,
        evidence_map,
        conflicts,
    )
    assert "cannot be considered inventive" in answer
    assert "automatically" not in answer.lower()
