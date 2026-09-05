"""Domain-Aware & Diversity-Aware Evidence Selector (Milestone 3).

Prevents single-document/domain monopolization for cross-domain queries while preserving
exact statutory matches for focused legal and taxon queries.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import (
    AYURVEDA_TERMS_PATH,
    DIVERSITY_ENABLED,
    FINAL_TOP_K,
    MAX_CHUNKS_PER_DOCUMENT,
    MAX_CHUNKS_PER_DOMAIN,
)

logger = logging.getLogger(__name__)


class DiversityAwareSelector:
    """Selects final high-quality evidence chunks with domain diversity and exact citation alignment."""

    def __init__(
        self,
        max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
        max_chunks_per_domain: int = MAX_CHUNKS_PER_DOMAIN,
        final_top_k: int = FINAL_TOP_K,
        diversity_enabled: bool = DIVERSITY_ENABLED,
        ayurveda_terms_path: Path = AYURVEDA_TERMS_PATH,
    ):
        self.max_chunks_per_document = max_chunks_per_document
        self.max_chunks_per_domain = max_chunks_per_domain
        self.final_top_k = final_top_k
        self.diversity_enabled = diversity_enabled
        self.ayurveda_terms_path = Path(ayurveda_terms_path)
        self.term_synonyms_map: Dict[str, Set[str]] = {}

        self._load_ayurveda_terms()

    def _load_ayurveda_terms(self) -> None:
        """Loads Ayurveda botanical synonyms from JSON for query recognition and alignment."""
        if self.ayurveda_terms_path.exists():
            try:
                with open(self.ayurveda_terms_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("terms", []):
                    all_variants = set()
                    for key in ["common_name", "scientific_name", "sanskrit_name", "hindi_name"]:
                        val = item.get(key)
                        if val:
                            all_variants.add(val.lower())
                    for alt in item.get("alternative_spellings", []):
                        all_variants.add(alt.lower())
                    for syn in item.get("retrieval_synonyms", []):
                        all_variants.add(syn.lower())

                    for variant in all_variants:
                        self.term_synonyms_map[variant] = all_variants
            except Exception as e:
                logger.warning(f"Could not load ayurveda terms from {self.ayurveda_terms_path}: {e}")

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyzes query intent:
        - is_exact_citation: checks for exact Section, Rule, Article, Patent No.
        - is_multidomain: checks if query spans multiple distinct domains (e.g. Patent + Ayurveda + Biodiversity)
        - target_identifiers: extracted section/rule/patent tokens
        - target_botanical_terms: recognized botanical taxa/synonyms
        """
        q_lower = query.lower()
        intent = {
            "is_exact_citation": False,
            "is_multidomain": False,
            "detected_domains": set(),
            "target_sections": [],
            "target_articles": [],
            "target_rules": [],
            "target_patents": [],
            "target_botanical_terms": set(),
        }

        # 1. Exact citations
        sec_matches = re.findall(r'\b(?:section|sec\.?)\s*(\d+[a-z]?(?:\([a-z0-9]+\))*)', q_lower)
        if sec_matches:
            intent["target_sections"] = sec_matches
            intent["is_exact_citation"] = True

        sec_sub = re.findall(r'\b(\d+[a-z]?\([a-z0-9]+\))\b', q_lower)
        if sec_sub:
            intent["target_sections"].extend(sec_sub)
            intent["is_exact_citation"] = True

        art_matches = re.findall(r'\b(?:article|art\.?)\s*(\d+[a-z]*)', q_lower)
        if art_matches:
            intent["target_articles"] = art_matches
            intent["is_exact_citation"] = True

        rule_matches = re.findall(r'\b(?:rule|r\.)\s*(\d+[a-z]*)', q_lower)
        if rule_matches:
            intent["target_rules"] = rule_matches
            intent["is_exact_citation"] = True

        pat_matches = re.findall(r'(?:patent\s*(?:no\.?|number)?\s*|patent\s+)(\d{5,8})', q_lower)
        if pat_matches:
            intent["target_patents"] = pat_matches
            intent["is_exact_citation"] = True

        # 2. Botanical taxa check
        for variant, syn_group in self.term_synonyms_map.items():
            if variant in q_lower:
                intent["target_botanical_terms"].update(syn_group)

        # 3. Domain detection
        if re.search(r'\b(patent|patents|patentability|inventive\s+step|novelty|specification|controller)\b', q_lower):
            intent["detected_domains"].add("patents")
        if re.search(r'\b(ayush|ayurveda|ayurvedic|aahara|rasayana|bhasma|formulation|vaidya|formulary)\b', q_lower):
            intent["detected_domains"].add("ayurveda")
        if re.search(r'\b(traditional\s+knowledge|tk|tkdl|indigenous|folklore|prior\s+art|defensive\s+protection)\b', q_lower):
            intent["detected_domains"].add("traditional_knowledge")
        if re.search(r'\b(biological\s+resources?|biological\s+material|biodiversity|nba|national\s+biodiversity\s+authority|access\s+and\s+benefit\s+sharing|abs)\b', q_lower):
            intent["detected_domains"].add("biological_resources")
        if re.search(r'\b(pct|wipo|epo|international\s+phase|international\s+search|isa|treaty)\b', q_lower):
            intent["detected_domains"].add("international_ip")
        if re.search(r'\b(fssai|licensing|advertising|claims|food\s+safety|fbo|regulation|order)\b', q_lower):
            intent["detected_domains"].add("regulatory")

        if len(intent["detected_domains"]) >= 2:
            intent["is_multidomain"] = True

        return intent

    def adjust_scores_for_exact_match(
        self,
        candidates: List[Dict[str, Any]],
        intent: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Applies exact-match calibration to prevent semantically adjacent chapters from overtaking exact sections."""
        adjusted = []
        for cand in candidates:
            c = dict(cand)
            meta = c.get("metadata", {})
            sec = str(c.get("section") or meta.get("section") or "").lower()
            art = str(c.get("article") or meta.get("article") or "").lower()
            rule = str(c.get("rule") or meta.get("rule") or "").lower()
            heading = str(c.get("heading") or meta.get("heading") or "").lower()
            pat_num = str(c.get("patent_number") or meta.get("patent_number") or "").lower()
            text_lower = c.get("text", "").lower()

            boost = 0.0

            # Exact section match
            for t_sec in intent.get("target_sections", []):
                clean_t = t_sec.replace("(", "").replace(")", "").lower()
                clean_s = sec.replace("(", "").replace(")", "").lower()
                if clean_t and (clean_t == clean_s or f"section {clean_t}" in heading or f"section_{clean_t}" in text_lower):
                    boost += 0.08

            # Exact article match
            for t_art in intent.get("target_articles", []):
                if t_art.lower() == art or f"article {t_art.lower()}" in heading:
                    boost += 0.08

            # Exact rule match
            for t_rule in intent.get("target_rules", []):
                if t_rule.lower() in rule or f"rule {t_rule.lower()}" in heading:
                    boost += 0.08

            # Exact patent number match
            for t_pat in intent.get("target_patents", []):
                if t_pat in pat_num or t_pat in text_lower or f"patent no. {t_pat}" in text_lower:
                    boost += 0.15

            # Botanical taxon match
            botanical_terms = intent.get("target_botanical_terms", set())
            if botanical_terms:
                if any(bt in text_lower or bt in heading for bt in botanical_terms):
                    boost += 0.05

            if boost > 0.0:
                original_score = c.get("reranker_score", 0.0)
                # Keep score bounded in [0, 1]
                c["reranker_score"] = min(1.0, original_score + boost)
                c["score"] = c["reranker_score"]
                c["exact_match_boost"] = round(boost, 4)

            adjusted.append(c)

        adjusted.sort(key=lambda x: x["reranker_score"], reverse=True)
        for rank, item in enumerate(adjusted, start=1):
            item["rank"] = rank
        return adjusted

    def select_evidence(
        self,
        query: str,
        reranked_candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        max_chunks_per_domain: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Selects top evidence chunks with domain-aware diversity control.

        Rules:
        1. If query is multi-domain (e.g. cross-domain query spanning Patent Act + AYUSH + Biodiversity Act),
           applies max_chunks_per_doc and max_chunks_per_domain to ensure multi-statute representation.
        2. If query is single-domain / exact lookup (e.g. "Section 3(p)"), allows top chunks from the same document.
        3. Preserves all metadata provenance and re-ranks final evidence.
        """
        if not reranked_candidates:
            return []

        k = top_k or self.final_top_k
        max_per_doc = max_chunks_per_doc or self.max_chunks_per_document
        max_per_dom = max_chunks_per_domain or self.max_chunks_per_domain

        intent = self.analyze_query_intent(query)
        candidates = self.adjust_scores_for_exact_match(reranked_candidates, intent)

        # If diversity is disabled or query is a focused exact statutory lookup, return top K directly
        if not self.diversity_enabled or (intent["is_exact_citation"] and not intent["is_multidomain"]):
            final_list = candidates[:k]
            for r, item in enumerate(final_list, start=1):
                item["rank"] = r
            return final_list

        # Multi-domain or diversity selection
        selected: List[Dict[str, Any]] = []
        doc_counts: Dict[str, int] = {}
        domain_counts: Dict[str, int] = {}

        # First pass: pick best candidates respecting diversity limits
        remaining: List[Dict[str, Any]] = []
        for cand in candidates:
            doc_id = cand.get("document_id") or cand.get("metadata", {}).get("document_id", "unknown_doc")
            dom = cand.get("category") or cand.get("metadata", {}).get("category", "general")

            current_doc_count = doc_counts.get(doc_id, 0)
            current_dom_count = domain_counts.get(dom, 0)

            # Check constraints
            if current_doc_count < max_per_doc and current_dom_count < max_per_dom:
                selected.append(cand)
                doc_counts[doc_id] = current_doc_count + 1
                domain_counts[dom] = current_dom_count + 1
            else:
                remaining.append(cand)

            if len(selected) >= k:
                break

        # Second pass: if diversity constraints were too strict and we have fewer than k, fill with highest remaining
        if len(selected) < k and remaining:
            for cand in remaining:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= k:
                    break

        # Update final 1-indexed ranks
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank

        return selected
