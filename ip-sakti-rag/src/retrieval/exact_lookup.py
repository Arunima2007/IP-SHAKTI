# src/retrieval/exact_lookup.py
"""Exact legal identifier lookup service.

Uses parsed identifiers from the query (provided in GraphState by QueryUnderstandingNode)
and retrieves matching chunks via BM25 with strict metadata filters.
Boosts are applied to prioritize exact matches.
"""

from typing import List, Dict, Any
import re

from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.bm25_search import tokenize_legal_technical
from src.config import EXACT_IDENTIFIER_BOOST, LEGAL_DOCUMENT_ALIASES
from src.retrieval.legal_identifier_parser import document_matches, provision_matches


def _exact_excerpt(text: str, parsed: Dict[str, Any]) -> str:
    """Return the smallest preserved source passage for a matched provision."""
    value, kind = str(parsed.get("value") or ""), parsed.get("type")
    if kind == "section" and "(" in value:
        clause = re.escape(value.split("(", 1)[1].rstrip(")"))
        match = re.search(rf"\({clause}\)\s*(.*?)(?=\n\s*\d+[A-Z]?\.\s|\Z)", text, re.I | re.S)
        if match:
            return f"Section {value}: ({clause}) " + match.group(1).strip()
    if kind in {"section", "article", "rule", "regulation"}:
        match = re.search(rf"(?:{kind}|sec\.?|art\.?|r\.?)\s*{re.escape(value)}\b(.*?)(?=\n\s*(?:section|article|rule|regulation)?\s*\d+[A-Z]?(?:\.|\s)|\Z)", text, re.I | re.S)
        if match:
            return f"{kind.title()} {value} " + match.group(1).strip()
    return text


def _is_unambiguous_match(chunk: Dict[str, Any], previous_chunk: Dict[str, Any], parsed: Dict[str, Any]) -> bool:
    """Reject incidental ``(p)`` occurrences from unrelated provisions.

    Some legacy Act chunks cross a page/section boundary.  Such a chunk is
    accepted only when its immediately preceding source chunk establishes the
    same parent section; this preserves the real Section 3(p) passage without
    treating every clause (p) in the Act as Section 3(p).
    """
    value = str(parsed.get("value") or "")
    if parsed.get("type") != "section" or "(" not in value:
        return True
    if str(chunk.get("section") or chunk.get("metadata", {}).get("section") or "").lower() == value.lower():
        return True
    if re.search(rf"(?:section|sec\.?)\s*{re.escape(value)}\b", chunk.get("text", ""), re.I):
        return True
    parent = value.split("(", 1)[0]
    current_section = str(chunk.get("section") or chunk.get("metadata", {}).get("section") or "")
    clause_match = re.search(rf"\({re.escape(value.split('(', 1)[1].rstrip(')'))}\)\s*(.{0,120})", chunk.get("text", ""), re.I | re.S)
    if clause_match and "w.e.f." in clause_match.group(1).lower():
        return False
    if current_section == parent and clause_match:
        return True
    previous_section = str(previous_chunk.get("metadata", {}).get("section") or "")
    # A cross-boundary legacy chunk may be labelled with the next section
    # (e.g. Section 4 after the final clause of Section 3), but not an
    # unrelated later Schedule provision.
    try:
        next_section = str(int(parent) + 1)
    except ValueError:
        next_section = ""
    current_is_next = current_section == next_section
    return previous_section == parent and current_is_next and bool(clause_match)


def _build_filters(parsed_identifier: Dict[str, Any], document_hint: str) -> Dict[str, Any]:
    """Construct metadata filters based on the identifier type and optional document hint.
    """
    filters: Dict[str, Any] = {}
    id_type = parsed_identifier.get("type")
    value = parsed_identifier.get("value")
    if not id_type or not value:
        return filters
    if id_type == "section":
        filters["section"] = value
    elif id_type == "article":
        filters["article"] = value
    elif id_type == "rule":
        filters["rule"] = value
    elif id_type == "patent":
        filters["patent_number"] = value
    # Document hint mapping to canonical title
    if document_hint:
        canonical = LEGAL_DOCUMENT_ALIASES.get(document_hint)
        if canonical:
            filters["document"] = canonical
    return filters


def exact_legal_lookup(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of candidate chunks that exactly match a legal identifier.

    The function reads ``state['exact_identifiers']`` (list of strings) and the
    ``parsed_identifier`` dict added by ``QueryUnderstandingNode``. It performs a
    BM25 search constrained by metadata filters and returns the raw candidate
    dictionaries with an additional ``exact_match_boost`` field.
    """
    parsed = state.get("parsed_identifier") or {}
    if not parsed.get("type") or not parsed.get("value"):
        return []

    bm25_engine = BM25SearchEngine()
    # Do not rely on equality filters here: legacy chunks can carry a parent
    # section while the clause itself appears in the chunk body.  Deterministic
    # matching inspects both structural metadata and preserved source text.
    query_tokens = tokenize_legal_technical(state.get("query", ""))
    lexical_scores = bm25_engine.bm25.get_scores(query_tokens) if query_tokens and bm25_engine.bm25 else []
    score_by_id = dict(zip(bm25_engine.corpus_chunk_ids, lexical_scores))
    exact = []
    canonical = parsed.get("canonical_title") or parsed.get("document_hint")
    for index, chunk in enumerate(bm25_engine.corpus_chunks):
        previous = bm25_engine.corpus_chunks[index - 1] if index else {}
        if not document_matches(chunk, canonical) or not provision_matches(chunk, parsed) or not _is_unambiguous_match(chunk, previous, parsed):
            continue
        meta = chunk.get("metadata", {})
        excerpt = _exact_excerpt(chunk.get("text", ""), parsed)
        # Substantive provision language ranks above amendment footnotes that
        # happen to repeat the same clause marker.
        passage_bonus = min(8.0, len(excerpt.split()) / 25.0)
        if re.search(r"\b(?:w\.e\.f\.|subs\.\s+by|ins\.\s+by|omitted\s+by)\b", excerpt, re.I):
            passage_bonus -= 8.0
        item = {
            "chunk_id": chunk.get("chunk_id"), "document_id": meta.get("document_id"),
            "document": meta.get("document"), "page": meta.get("page"),
            "section": meta.get("section"), "article": meta.get("article"),
            "rule": meta.get("rule"), "heading": meta.get("heading"),
            "jurisdiction": meta.get("jurisdiction"), "category": meta.get("category"),
            "domain": meta.get("domain", []), "text": excerpt,
            "metadata": meta, "retrieval_method": "exact_identifier",
            "score": EXACT_IDENTIFIER_BOOST + float(score_by_id.get(chunk.get("chunk_id"), 0.0)) + passage_bonus, "exact_provision_match": True,
            "exact_document_match": bool(canonical), "matched_provision_type": parsed["type"],
            "matched_provision": parsed["value"],
        }
        # The matched identifier is the citation identity even when a legacy
        # chunk's parent metadata is stale or spans a subsequent section.
        if parsed["type"] == "section":
            item["section"] = parsed["value"]
        else:
            item[parsed["type"]] = parsed["value"]
        exact.append(item)

    # Prefer chunks structurally tagged with the provision, then stable source order.
    exact.sort(key=lambda c: (c.get("score", 0.0), c["chunk_id"]), reverse=True)
    return exact[:10]
