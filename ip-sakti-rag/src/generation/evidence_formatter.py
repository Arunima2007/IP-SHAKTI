"""Evidence Formatter for IP-SAKTI Sahayak.

Formats retrieved and reranked chunks into structured evidence blocks with stable
identifiers (E1, E2, ...), metadata provenance, source authority tier tagging,
and source conflict detection.
"""
from typing import Dict, List, Any, Tuple, Optional
import re
from src.config import SOURCE_HIERARCHY, MAX_EVIDENCE_CHUNKS


class EvidenceFormatter:
    """Formats retrieved chunks into strictly structured evidence blocks for LLM consumption."""

    def __init__(self, max_evidence_chunks: int = MAX_EVIDENCE_CHUNKS):
        self.max_evidence_chunks = max_evidence_chunks

    def format_evidence(
        self,
        chunks: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> Tuple[str, Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Formats candidate chunks into structured evidence blocks with stable IDs (E1, E2, ...).

        Returns:
            - formatted_text: String representation for the LLM prompt.
            - evidence_map: Dict mapping 'E1' -> enriched chunk dict with citation metadata.
            - detected_conflicts: List of detected source conflict alerts, if any.
        """
        if not chunks:
            return "No relevant authoritative evidence found.", {}, []

        # Sort chunks primarily by Source Tier (Tier 1 > Tier 2 > Tier 3) and secondarily by reranker score
        enriched_chunks = []
        for c in chunks:
            doc_id = c.get("document_id") or c.get("metadata", {}).get("document_id", "")
            tier_info = SOURCE_HIERARCHY.get(doc_id, {"tier": 2, "label": "Tier 2: Official Document", "weight": 0.85})
            c_copy = dict(c)
            c_copy["_tier"] = tier_info["tier"]
            c_copy["_tier_label"] = tier_info["label"]
            c_copy["_authority_weight"] = tier_info["weight"]
            enriched_chunks.append(c_copy)

        # Stable sort: keep reranker preference but ensure top authority tier items are prominent
        enriched_chunks.sort(key=lambda x: (x["_tier"], -float(x.get("rerank_score", x.get("score", 0.0)))))
        selected_chunks = enriched_chunks[:self.max_evidence_chunks]

        evidence_map: Dict[str, Dict[str, Any]] = {}
        formatted_blocks: List[str] = []

        for idx, chunk in enumerate(selected_chunks, start=1):
            evidence_id = f"E{idx}"
            chunk_id = chunk.get("chunk_id", "")
            doc_name = chunk.get("document") or chunk.get("metadata", {}).get("document_name", "Unknown Document")
            jurisdiction = chunk.get("jurisdiction") or chunk.get("metadata", {}).get("jurisdiction", "Unknown")
            
            # Page resolution
            page = chunk.get("page") or chunk.get("page_start") or chunk.get("metadata", {}).get("page_start", 1)
            page_end = chunk.get("page_end") or chunk.get("metadata", {}).get("page_end", page)
            page_str = str(page) if page == page_end else f"{page}-{page_end}"
            
            section = chunk.get("section") or chunk.get("metadata", {}).get("section", None)
            article = chunk.get("article") or chunk.get("metadata", {}).get("article", None)
            rule = chunk.get("rule") or chunk.get("metadata", {}).get("rule", None)
            heading = chunk.get("heading") or chunk.get("metadata", {}).get("heading", "General Provision")
            tier_label = chunk.get("_tier_label", "Tier 2: Official Document")
            
            # Text cleanup
            text = chunk.get("text", "").strip()

            evidence_item = {
                "evidence_id": evidence_id,
                "chunk_id": chunk_id,
                "document": doc_name,
                "document_id": chunk.get("document_id", ""),
                "jurisdiction": jurisdiction,
                "page": page_str,
                "page_num": page,
                "section": section,
                "article": article,
                "rule": rule,
                "heading": heading,
                "tier_label": tier_label,
                "tier": chunk.get("_tier", 2),
                "text": text,
                "rerank_score": chunk.get("rerank_score", chunk.get("score", 0.0)),
                "category": chunk.get("category", "")
            }
            evidence_map[evidence_id] = evidence_item

            # Build readable block
            block = [
                f"[{evidence_id}]",
                f"Chunk ID: {chunk_id}",
                f"Document: {doc_name}",
                f"Authority: {tier_label}",
                f"Jurisdiction: {jurisdiction}",
                f"Page: {page_str}",
            ]
            if section:
                block.append(f"Section: {section}")
            if article:
                block.append(f"Article: {article}")
            if rule:
                block.append(f"Rule: {rule}")
            if heading and heading != "None":
                block.append(f"Heading: {heading}")
            
            block.append(f"Content:\n\"\"\"\n{text}\n\"\"\"")
            formatted_blocks.append("\n".join(block))

        # Check for potential source conflicts
        detected_conflicts = self._detect_source_conflicts(list(evidence_map.values()))

        formatted_text = "\n\n" + ("\n\n" + "="*50 + "\n\n").join(formatted_blocks)
        if detected_conflicts:
            conflict_notice = "\n\n[NOTICE: POTENTIAL SOURCE VARIATION / MULTI-JURISDICTION CONTEXT DETECTED]\n"
            for conf in detected_conflicts:
                conflict_notice += f"- {conf['description']} (Between {conf['source_a']['evidence_id']} and {conf['source_b']['evidence_id']})\n"
            formatted_text = conflict_notice + formatted_text

        return formatted_text, evidence_map, detected_conflicts

    def _detect_source_conflicts(self, evidence_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects potential source differences or jurisdictional variations across retrieved evidence.
        Flags when different statutory tiers or differing jurisdictional frameworks provide diverging provisions.
        """
        conflicts = []
        n = len(evidence_items)
        for i in range(n):
            for j in range(i + 1, n):
                item_a = evidence_items[i]
                item_b = evidence_items[j]
                
                # Check 1: Cross-Jurisdictional Scope Differences (e.g. India Patent Act vs EPO/PCT vs WIPO)
                if item_a["jurisdiction"] != item_b["jurisdiction"] and item_a["jurisdiction"] != "Unknown" and item_b["jurisdiction"] != "Unknown":
                    corpus_a = (item_a["text"] + " " + (item_a.get("heading") or "") + " " + item_a.get("document", "")).lower()
                    corpus_b = (item_b["text"] + " " + (item_b.get("heading") or "") + " " + item_b.get("document", "")).lower()
                    keywords = [
                        "traditional knowledge", "tk", "genetic resources", "patent", "patentability",
                        "invention", "inventions", "exclusion", "disclosure", "biological material", "prior art"
                    ]
                    matched_kw = [kw for kw in keywords if kw in corpus_a and kw in corpus_b]
                    if matched_kw:
                        conflicts.append({
                            "type": "jurisdictional_variation",
                            "source_a": item_a,
                            "source_b": item_b,
                            "description": f"Different jurisdictional rules for {', '.join(matched_kw[:3])} between {item_a['jurisdiction']} ({item_a['document']}) and {item_b['jurisdiction']} ({item_b['document']})."
                        })
                
                # Check 2: Regulatory vs Statutory Scope Difference (e.g. FSSAI Ayurveda Aahara vs Drugs & Cosmetics Act)
                doc_a = item_a.get("document_id", "").lower()
                doc_b = item_b.get("document_id", "").lower()
                if ("fssai" in doc_a and "drugs" in doc_b) or ("drugs" in doc_a and "fssai" in doc_b):
                    conflicts.append({
                        "type": "regulatory_boundary",
                        "source_a": item_a,
                        "source_b": item_b,
                        "description": "Boundary distinction between Food Safety (FSSAI Ayurveda Aahara) and Ayurvedic Drug/Medicine regulations (Drugs & Cosmetics Act)."
                    })
        return conflicts
