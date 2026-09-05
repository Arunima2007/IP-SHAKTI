# legal_identifier_parser.py
"""Deterministic parsing and matching for legal identifiers.

Shared by query understanding, retrieval, reranking, and validation so explicit
legal references cannot be diluted by semantic similarity later in the pipeline.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

_PROVISION_PATTERNS = (
    ("section", re.compile(r"\b(?:section|sec\.?|s\.?|धारा)\s*([0-9]+[a-z]?(?:\([a-z0-9]+\))*)", re.I)),
    ("article", re.compile(r"\b(?:article|art\.?|अनुच्छेद)\s*([0-9]+[a-z]*(?:\([a-z0-9]+\))*)", re.I)),
    ("rule", re.compile(r"\b(?:rule|r\.?|नियम)\s*([0-9]+(?:bis|ter|quater)?[a-z]*(?:\.\d+)?)", re.I)),
    ("regulation", re.compile(r"\b(?:regulation|reg\.?)\s*([0-9]+[a-z]*(?:\([a-z0-9]+\))*)", re.I)),
    ("schedule", re.compile(r"\b((?:first|second|third|fourth|[0-9]+)(?:\s+schedule)?|schedule\s+[ivxlcdm0-9]+)", re.I)),
    ("patent", re.compile(r"\bpatent\s*(?:no\.?|number)?\s*([0-9]{5,10})\b", re.I)),
)
_DOCUMENTS = {
    "Patents Act, 1970": ("patents act", "patent act", "patent act 1970"),
    "Trade Marks Act, 1999": ("trade marks act", "trademarks act", "trade mark act", "trademark act"),
    "Designs Act, 2000": ("designs act", "design act"),
    "Copyright Act, 1957": ("copyright act",),
    "Biological Diversity Act, 2002": ("biological diversity act", "biodiversity act"),
    "Drugs and Cosmetics Act, 1940": ("drugs and cosmetics act",),
    "PCT": ("pct", "patent cooperation treaty"),
    "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge (2024)": ("wipo gr/tk", "wipo treaty", "genetic resources treaty"),
}


def normalize_identifier(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower().strip(" .:;,")


def canonical_document_title(value: str) -> Optional[str]:
    normalized = re.sub(r"[_.\-]+", " ", value.lower()).replace(".pdf", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for canonical, aliases in _DOCUMENTS.items():
        if any(alias in normalized for alias in aliases):
            return canonical
    return None


def parse(query: str) -> Dict[str, Optional[str]]:
    canonical = canonical_document_title(query)
    jurisdiction = "WIPO/PCT" if canonical == "PCT" or re.search(r"\b(?:pct|wipo)\b", query, re.I) else ("India" if canonical else None)
    for provision_type, pattern in _PROVISION_PATTERNS:
        match = pattern.search(query)
        if match:
            value = normalize_identifier(match.group(1))
            return {"type": provision_type, "value": value, "provision_type": provision_type,
                    "provision_number": value, "canonical_title": canonical,
                    "document_hint": canonical, "jurisdiction": jurisdiction}
    return {"type": None, "value": None, "provision_type": None, "provision_number": None,
            "canonical_title": canonical, "document_hint": canonical, "jurisdiction": jurisdiction}


def provision_value(metadata: Dict[str, Any], provision_type: str) -> str:
    meta = metadata.get("metadata", metadata)
    if provision_type == "section":
        section, clause = normalize_identifier(str(meta.get("section") or "")), normalize_identifier(str(meta.get("clause") or ""))
        return f"{section}({clause})" if section and clause else section
    return normalize_identifier(str(meta.get("patent_number" if provision_type == "patent" else provision_type) or ""))


def document_matches(item: Dict[str, Any], canonical_title: Optional[str]) -> bool:
    if not canonical_title:
        return True
    meta = item.get("metadata", item)
    candidates: Iterable[str] = (str(item.get("document") or ""), str(meta.get("document") or ""),
                                 str(item.get("document_id") or ""), str(meta.get("document_id") or ""),
                                 str(meta.get("canonical_title") or ""))
    return any(canonical_document_title(value) == canonical_title for value in candidates if value)


def provision_matches(item: Dict[str, Any], parsed: Dict[str, Any]) -> bool:
    provision_type, wanted = parsed.get("type"), parsed.get("value")
    if not provision_type or not wanted:
        return False
    wanted = normalize_identifier(str(wanted))
    if provision_value(item, provision_type) == wanted:
        return True
    meta = item.get("metadata", {})
    corpus = " ".join(str(item.get(k) or "") for k in ("text", "heading", "context_header"))
    corpus += " " + " ".join(str(meta.get(k) or "") for k in ("heading", "section", "article", "rule", "patent_number"))
    if provision_type == "section":
        pattern = rf"(?:section|sec\.?|s\.?)\s*{re.escape(wanted)}\b"
        if "(" in wanted:
            pattern += rf"|\({re.escape(wanted.split('(', 1)[1][:-1])}\)"
    elif provision_type == "patent":
        pattern = rf"patent\s*(?:no\.?|number)?\s*{re.escape(wanted)}\b"
    else:
        pattern = rf"\b{re.escape(provision_type)}\s*{re.escape(wanted)}\b"
    return bool(re.search(pattern, corpus, re.I))
