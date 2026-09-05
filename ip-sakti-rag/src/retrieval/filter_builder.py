"""Dynamic Metadata Filter Generator for IP-SAKTI Sahayak."""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client.http import models

logger = logging.getLogger(__name__)


class MetadataFilterBuilder:
    """Detects implicit/explicit metadata intent from queries and builds Qdrant & BM25 filters."""

    @staticmethod
    def infer_filters_from_query(query: str, confidence_threshold: float = 0.75) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Analyzes query to infer potential metadata restrictions.
        Returns (filters_dict, confidence_score). If confidence is low or ambiguous, returns (None, score).
        """
        q_lower = query.lower()
        filters: Dict[str, Any] = {}
        confidence = 0.0

        # International / WIPO / PCT checks
        if re.search(r'\b(pct|wipo|international\s+phase|international\s+search|rule\s+43bis|wipo\s+gr\/tk|gr\/tk\s+treaty)\b', q_lower):
            filters["jurisdiction"] = ["WIPO/PCT", "International"]
            confidence = 0.90
        # EPO checks
        elif re.search(r'\b(epo|european\s+patent|european\s+patent\s+office|guidelines\s+for\s+examination\s+in\s+the\s+epo)\b', q_lower):
            filters["jurisdiction"] = "EPO"
            confidence = 0.90
        # India-specific checks
        elif re.search(r'\b(in\s+india|indian\s+patents?\s+act|indian\s+patent\s+law|indian\s+law|ayush|ayurveda\s+aahara|fssai|national\s+biodiversity\s+authority|nba|section\s+3\(p\)|section\s+3|drugs\s+and\s+cosmetics)\b', q_lower):
            filters["jurisdiction"] = "India"
            confidence = 0.85

        # Domain checks
        if "ayurveda aahara" in q_lower or "fssai" in q_lower or "food safety" in q_lower:
            filters["domain"] = ["ayurveda_aahara", "ayush_regulation", "regulatory"]
            confidence = max(confidence, 0.85)

        if confidence < confidence_threshold:
            # Low confidence or ambiguous query -> do not over-filter
            return None, confidence

        return filters, confidence

    @staticmethod
    def build_qdrant_filter(filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        """Converts filter dictionary into Qdrant models.Filter object."""
        if not filters:
            return None

        must_conditions = []
        for key, val in filters.items():
            if val is None:
                continue

            if isinstance(val, list):
                # Match any of the list values (OR within the specified field)
                should_conditions = [
                    models.FieldCondition(key=key, match=models.MatchValue(value=v))
                    for v in val
                ]
                must_conditions.append(models.Filter(should=should_conditions))
            else:
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=val))
                )

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)
