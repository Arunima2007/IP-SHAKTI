"""Citation Engine for IP-SAKTI Sahayak.

Parses internal [E#] evidence references from LLM-generated answers, converts them into
human-readable formatted citations and structured citation objects (C1, C2, ...),
and formats the final standardized Sources section.
"""
from typing import Dict, List, Any, Tuple, Optional, Set
import re


class CitationEngine:
    """Handles citation extraction, structured object creation, and human-readable formatting."""

    def __init__(self):
        # Pattern to capture [E1], [E2], [E1, E2], [E1][E2], [E1,E2]
        self.evidence_tag_pattern = re.compile(r'\[(E\d+(?:\s*,\s*E\d+)*)\]')

    def extract_evidence_ids(self, text: str) -> List[str]:
        """Extracts unique evidence IDs cited in the text in order of appearance."""
        cited_ids: List[str] = []
        for match in self.evidence_tag_pattern.finditer(text):
            inner = match.group(1)
            ids = [x.strip() for x in inner.split(',') if x.strip()]
            for eid in ids:
                if eid not in cited_ids:
                    cited_ids.append(eid)
        return cited_ids

    def build_structured_citations(
        self,
        cited_evidence_ids: List[str],
        evidence_map: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
        """
        Creates structured citation objects and bi-directional ID mappings.

        Returns:
            - citation_objects: List of structured dicts (C1, C2, ...)
            - eid_to_cid: Mapping from 'E1' -> 'C1'
            - eid_to_num: Mapping from 'E1' -> '[1]'
        """
        citation_objects = []
        eid_to_cid = {}
        eid_to_num = {}

        for idx, eid in enumerate(cited_evidence_ids, start=1):
            cid = f"C{idx}"
            num_str = f"[{idx}]"
            eid_to_cid[eid] = cid
            eid_to_num[eid] = num_str

            evidence = evidence_map.get(eid, {})
            doc_name = evidence.get("document", "Unknown Document")
            clean_doc_name = self._format_doc_title(doc_name)
            
            page = evidence.get("page", "1")
            section = evidence.get("section")
            article = evidence.get("article")
            rule = evidence.get("rule")
            heading = evidence.get("heading")
            jurisdiction = evidence.get("jurisdiction", "India")
            chunk_id = evidence.get("chunk_id", "")

            # Construct human-readable citation string
            parts = [clean_doc_name]
            if section:
                parts.append(f"Section {section}")
            elif article:
                parts.append(f"Article {article}")
            elif rule:
                parts.append(f"Rule {rule}")
            
            if heading and heading not in ("None", "General Provision", clean_doc_name):
                # Shorten very long headings
                short_heading = heading if len(heading) <= 45 else heading[:42] + "..."
                parts.append(f'"{short_heading}"')
            
            if page:
                parts.append(f"p. {page}")

            formatted_citation = " — ".join(parts)

            citation_obj = {
                "citation_id": cid,
                "evidence_id": eid,
                "citation_number": idx,
                "chunk_id": chunk_id,
                "document": doc_name,
                "document_title": clean_doc_name,
                "jurisdiction": jurisdiction,
                "page": page,
                "section": section,
                "article": article,
                "rule": rule,
                "heading": heading,
                "formatted_citation": formatted_citation,
                "tier": evidence.get("tier", 2),
                "tier_label": evidence.get("tier_label", "Tier 2: Official Document")
            }
            citation_objects.append(citation_obj)

        return citation_objects, eid_to_cid, eid_to_num

    def convert_answer_citations(
        self,
        answer_text: str,
        evidence_map: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Converts internal [E#] tags into numbered human-readable citations [1], [2],
        builds structured citation objects, and appends a formatted ### Sources section
        if one was not already present.
        """
        cited_eids = self.extract_evidence_ids(answer_text)
        citation_objects, eid_to_cid, eid_to_num = self.build_structured_citations(cited_eids, evidence_map)

        # Replace [E1] / [E1, E2] with [1], [2]
        def _replace_tag(match):
            inner = match.group(1)
            ids = [x.strip() for x in inner.split(',') if x.strip()]
            nums = []
            for eid in ids:
                if eid in eid_to_num:
                    nums.append(eid_to_num[eid].strip('[]'))
                elif eid in evidence_map:
                    # Unindexed cited eid fallback
                    nums.append(eid)
            if nums:
                return f"[{', '.join(nums)}]"
            return match.group(0)

        converted_text = self.evidence_tag_pattern.sub(_replace_tag, answer_text)

        # Ensure ### Sources section contains exact formatted citations
        sources_section = self._render_sources_section(citation_objects)
        
        # If the answer already has a ### Sources section, replace or standardize it
        if "### Sources" in converted_text:
            base_text = converted_text.split("### Sources")[0].strip()
            # If there's an Important Note after Sources, keep it
            after_sources = ""
            if "### Important Note" in converted_text:
                parts = converted_text.split("### Important Note")
                if len(parts) > 1:
                    after_sources = "\n\n### Important Note\n" + parts[1].strip()
            
            final_text = f"{base_text}\n\n{sources_section}{after_sources}".strip()
        else:
            final_text = f"{converted_text.strip()}\n\n{sources_section}".strip()

        return final_text, citation_objects

    def _render_sources_section(self, citation_objects: List[Dict[str, Any]]) -> str:
        """Renders the standard markdown Sources section."""
        if not citation_objects:
            return "### Sources\n\nNo specific statutory sources cited."
        
        lines = ["### Sources\n"]
        for c in citation_objects:
            lines.append(f"[{c['citation_number']}] {c['formatted_citation']}")
        return "\n".join(lines)

    def _format_doc_title(self, doc_filename: str) -> str:
        """Formats raw file names into clean authoritative legal titles."""
        clean = doc_filename.replace(".pdf", "").replace("-", " ").replace("_", " ")
        mapping = {
            "patent act 1970": "Patents Act, 1970",
            "patent act-1970": "Patents Act, 1970",
            "biological diversity act 2002": "Biological Diversity Act, 2002",
            "drugs and cosmetics act 1940": "Drugs and Cosmetics Act, 1940",
            "trade marks act 1999": "Trade Marks Act, 1999",
            "copyright act 1957": "Copyright Act, 1957",
            "designs act 2000": "Designs Act, 2000",
            "ayush related inventions guidelines 2025": "Guidelines for Examination of Patent Applications Related to AYUSH (2025)",
            "guidelines tk biological material 2012": "Guidelines for Patent Applications Relating to Traditional Knowledge and Biological Material (2012)",
            "fssai ayurveda aahara regulations 2022": "FSSAI (Ayurveda Aahara) Regulations, 2022",
            "order fssai ayurveda aahara schedules 2025": "FSSAI Order on Ayurveda Aahara Schedules (2025)",
            "compendium advertising claims regulations 2022": "FSSAI Advertising & Claims Compendium (2022)",
            "compendium licensing regulations 2021": "FSSAI Licensing & Registration Compendium (2021)",
            "gsr 669 e drugs rules 2024": "Gazette Notification G.S.R. 669(E) Drugs Rules (2024)",
            "wipo gr tk treaty 2024": "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge (2024)",
            "pct applicant guide international phase": "WIPO PCT Applicant's Guide (International Phase)",
            "epo guidelines for examination 2026": "EPO Guidelines for Examination (2026)",
            "epo pct guidelines 2026": "EPO Guidelines for PCT Search and Examination (2026)",
            "who benchmarks practice ayurveda": "WHO Benchmarks for the Practice of Ayurveda",
            "who benchmarks training ayurveda": "WHO Benchmarks for the Training of Ayurveda",
            "wipo ip gr tk tce overview": "WIPO Overview of IP, Genetic Resources, and TK",
            "wipo documenting tk toolkit": "WIPO Documenting Traditional Knowledge Toolkit",
            "wipo patent disclosure gr tk": "WIPO Patent Disclosure Requirements Relating to GR & TK"
        }
        lower = clean.lower().strip()
        for key, title in mapping.items():
            if key in lower or lower in key:
                return title
        return clean.title()
