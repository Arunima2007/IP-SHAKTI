"""
Canonical Document Registry for IP-SAKTI Sahayak.

Single source of truth for all legal document normalization.
Ensures consistent document identity across retrieval, reranking, and generation.

Prevents document confusion by mapping all user-facing aliases to a single
canonical title and metadata structure.
"""

from typing import Dict, Optional, Set, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """
    Centralized registry for legal document normalization and validation.
    
    Ensures:
    - Single canonical title per document
    - Consistent metadata across all modules
    - No duplicate or conflicting aliases
    """
    
    # Canonical document definitions
    # Format: "Canonical Title" -> {aliases, metadata}
    CANONICAL_DOCUMENTS: Dict[str, Dict] = {
        "The Patents Act, 1970": {
            "aliases": [
                "patents act", "patent act", "patent act 1970",
                "the patents act, 1970", "indian patents act",
                "patents act, 1970", "patents act 1970",
                "patent act 1970", "patents act,1970",
            ],
            "jurisdictions": ["India"],
            "document_id": "patent_act_1970",
            "tier": 1,
            "label": "Tier 1: Primary Statute",
            "weight": 1.0,
            "category": "indian_ip",
        },
        
        "The Trade Marks Act, 1999": {
            "aliases": [
                "trade marks act", "trademarks act", "trade mark act",
                "trademark act", "trade marks act, 1999",
                "the trade marks act, 1999", "trademark act 1999",
                "trade marks act 1999", "trademarks act 1999",
            ],
            "jurisdictions": ["India"],
            "document_id": "trade_marks_act_1999",
            "tier": 1,
            "label": "Tier 1: Primary Statute",
            "weight": 1.0,
            "category": "indian_ip",
        },
        
        "The Designs Act, 2000": {
            "aliases": [
                "designs act", "design act", "designs act, 2000",
                "the designs act, 2000", "designs act 2000",
                "design act 2000",
            ],
            "jurisdictions": ["India"],
            "document_id": "designs_act_2000",
            "tier": 1,
            "label": "Tier 1: Primary Statute",
            "weight": 1.0,
            "category": "indian_ip",
        },
        
        "The Copyright Act, 1957": {
            "aliases": [
                "copyright act", "copyright act, 1957",
                "the copyright act, 1957", "copyright act 1957",
            ],
            "jurisdictions": ["India"],
            "document_id": "copyright_act_1957",
            "tier": 1,
            "label": "Tier 1: Primary Statute",
            "weight": 1.0,
            "category": "indian_ip",
        },
        
        "The Biological Diversity Act, 2002": {
            "aliases": [
                "biological diversity act", "biodiversity act",
                "biological diversity act, 2002",
                "the biological diversity act, 2002",
                "biological diversity act 2002",
                "biodiversity act 2002",
            ],
            "jurisdictions": ["India"],
            "document_id": "biological_diversity_act_2002",
            "tier": 1,
            "label": "Tier 1: Primary Statute",
            "weight": 1.0,
            "category": "biological_resources",
        },
        
        "Drugs and Cosmetics Act, 1940": {
            "aliases": [
                "drugs and cosmetics act", "drugs and cosmetics act, 1940",
                "drugs and cosmetics act 1940", "the drugs and cosmetics act, 1940",
            ],
            "jurisdictions": ["India"],
            "document_id": "drugs_and_cosmetics_act_1940",
            "tier": 1,
            "label": "Tier 1: Primary Statute",
            "weight": 1.0,
            "category": "regulatory",
        },
        
        "AYUSH-Related Inventions Guidelines, 2025": {
            "aliases": [
                "ayush related inventions guidelines", "ayush guidelines",
                "ayush-related inventions guidelines, 2025",
                "ayush related inventions guidelines 2025",
            ],
            "jurisdictions": ["India"],
            "document_id": "ayush_related_inventions_guidelines_2025",
            "tier": 2,
            "label": "Tier 2: Official Patent Guidelines",
            "weight": 0.9,
            "category": "ayush",
        },
        
        "Guidelines for Patent Applications Relating to Traditional Knowledge and Biological Material, 2012": {
            "aliases": [
                "guidelines for patent applications relating to traditional knowledge and biological material",
                "guidelines tk biological material", "guidelines traditional knowledge 2012",
                "traditional knowledge guidelines 2012",
            ],
            "jurisdictions": ["India"],
            "document_id": "guidelines_tk_biological_material_2012",
            "tier": 2,
            "label": "Tier 2: Official Patent Guidelines",
            "weight": 0.9,
            "category": "traditional_knowledge",
        },
        
        "Ayurveda Aahara Regulations, 2022": {
            "aliases": [
                "ayurveda aahara regulations", "ayurveda aahara regulations, 2022",
                "ayurveda aahara regulations 2022", "fssai ayurveda aahara regulations 2022",
            ],
            "jurisdictions": ["India"],
            "document_id": "fssai_ayurveda_aahara_regulations_2022",
            "tier": 2,
            "label": "Tier 2: Statutory Regulation",
            "weight": 0.9,
            "category": "ayurveda",
        },
        
        "PCT Applicant's Guide — International Phase": {
            "aliases": [
                "pct applicant's guide", "pct applicant guide",
                "pct applicant's guide international phase",
                "pct applicant guide international phase",
                "patent cooperation treaty guide",
            ],
            "jurisdictions": ["WIPO/PCT"],
            "document_id": "pct_applicant_guide_international_phase",
            "tier": 1,
            "label": "Tier 1: International Treaty Guide",
            "weight": 0.95,
            "category": "international_ip",
        },
        
        "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge (2024)": {
            "aliases": [
                "wipo gr/tk treaty", "wipo treaty", "wipo gr/tk",
                "genetic resources treaty", "wipo gr tk treaty",
                "wipo intellectual property genetic resources treaty",
            ],
            "jurisdictions": ["WIPO/PCT", "International"],
            "document_id": "wipo_gr_tk_treaty_2024",
            "tier": 1,
            "label": "Tier 1: International Treaty",
            "weight": 1.0,
            "category": "international_ip",
        },
        
        "EPO Guidelines for Examination, 2026": {
            "aliases": [
                "epo guidelines", "epo guidelines for examination",
                "epo examination guidelines", "epo guidelines 2026",
                "european patent office guidelines",
            ],
            "jurisdictions": ["EPO"],
            "document_id": "epo_guidelines_for_examination_2026",
            "tier": 1,
            "label": "Tier 1: Examination Guidelines",
            "weight": 0.95,
            "category": "international_ip",
        },
        
        "EPO PCT Guidelines, 2026": {
            "aliases": [
                "epo pct guidelines", "epo pct guidelines 2026",
                "european patent office pct guidelines",
            ],
            "jurisdictions": ["EPO", "WIPO/PCT"],
            "document_id": "epo_pct_guidelines_2026",
            "tier": 1,
            "label": "Tier 1: Examination Guidelines",
            "weight": 0.95,
            "category": "international_ip",
        },
    }
    
    def __init__(self):
        """Initialize registry and build reverse lookup."""
        self._build_reverse_lookup()
    
    def _build_reverse_lookup(self) -> None:
        """Build alias-to-canonical mapping for fast lookups."""
        self.alias_to_canonical: Dict[str, str] = {}
        for canonical, metadata in self.CANONICAL_DOCUMENTS.items():
            for alias in metadata.get("aliases", []):
                normalized = self._normalize_alias(alias)
                self.alias_to_canonical[normalized] = canonical
    
    @staticmethod
    def _normalize_alias(alias: str) -> str:
        """Normalize an alias for comparison."""
        # Replace punctuation and underscores with spaces
        normalized = re.sub(r"[_.\-,]+", " ", alias.lower())
        # Replace .pdf suffix
        normalized = normalized.replace(".pdf", "")
        # Collapse whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
    
    def get_canonical_title(self, text: str) -> Optional[str]:
        """
        Resolve a document reference to its canonical title.
        
        Args:
            text: Document name, alias, or filename
            
        Returns:
            Canonical title or None if not recognized
        """
        if not text:
            return None
        
        normalized = self._normalize_alias(text)
        
        # Exact match in reverse lookup
        if normalized in self.alias_to_canonical:
            return self.alias_to_canonical[normalized]
        
        # Fallback: substring match (less precise but more forgiving)
        for alias, canonical in self.alias_to_canonical.items():
            if alias in normalized or normalized in alias:
                return canonical
        
        return None
    
    def get_document_id(self, canonical_title: str) -> Optional[str]:
        """Get document_id for a canonical title."""
        metadata = self.CANONICAL_DOCUMENTS.get(canonical_title)
        return metadata.get("document_id") if metadata else None
    
    def get_tier(self, canonical_title: str) -> Optional[int]:
        """Get source authority tier (1, 2, or 3)."""
        metadata = self.CANONICAL_DOCUMENTS.get(canonical_title)
        return metadata.get("tier") if metadata else None
    
    def get_weight(self, canonical_title: str) -> Optional[float]:
        """Get authority weight for ranking."""
        metadata = self.CANONICAL_DOCUMENTS.get(canonical_title)
        return metadata.get("weight") if metadata else None
    
    def get_label(self, canonical_title: str) -> Optional[str]:
        """Get human-readable tier label."""
        metadata = self.CANONICAL_DOCUMENTS.get(canonical_title)
        return metadata.get("label") if metadata else None
    
    def get_jurisdictions(self, canonical_title: str) -> Optional[list]:
        """Get applicable jurisdictions."""
        metadata = self.CANONICAL_DOCUMENTS.get(canonical_title)
        return metadata.get("jurisdictions") if metadata else None
    
    def get_category(self, canonical_title: str) -> Optional[str]:
        """Get document category (patents, ayush, etc.)."""
        metadata = self.CANONICAL_DOCUMENTS.get(canonical_title)
        return metadata.get("category") if metadata else None
    
    def is_valid_canonical(self, title: str) -> bool:
        """Check if a title is a valid canonical document."""
        return title in self.CANONICAL_DOCUMENTS
    
    def get_all_canonical_titles(self) -> Set[str]:
        """Return all canonical document titles."""
        return set(self.CANONICAL_DOCUMENTS.keys())
    
    def get_all_aliases(self, canonical_title: str) -> Optional[list]:
        """Get all aliases for a canonical title."""
        metadata = self.CANONICAL_DOCUMENTS.get(canonical_title)
        return metadata.get("aliases") if metadata else None
    
    def get_metadata(self, canonical_title: str) -> Optional[Dict]:
        """Get complete metadata for a canonical title."""
        return self.CANONICAL_DOCUMENTS.get(canonical_title)


# Global singleton instance
_REGISTRY: Optional[DocumentRegistry] = None


def get_document_registry() -> DocumentRegistry:
    """
    Get the global document registry singleton.
    
    Returns:
        DocumentRegistry instance
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DocumentRegistry()
    return _REGISTRY


def normalize_document_reference(text: str) -> Optional[str]:
    """
    Convenience function to normalize any document reference.
    
    Args:
        text: Document name, alias, or filename
        
    Returns:
        Canonical title or None
    """
    return get_document_registry().get_canonical_title(text)
