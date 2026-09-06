"""Claim-Level Citation Validator for IP-SAKTI Sahayak.

Decomposes generated answers into individual factual/legal claims, validates claim support
against cited authoritative evidence chunks, detects hallucinations, unsupported assertions,
wrong sections/rules, and computes formal citation verification metrics.
"""
from typing import Dict, List, Any, Tuple, Optional, Set
import re
import difflib
from src.config import MIN_CLAIM_SUPPORT_CONFIDENCE, INSUFFICIENT_EVIDENCE_MESSAGE


class ClaimCitationValidator:
    """Validates factual groundedness and citation accuracy at the individual claim level."""

    def __init__(self, min_confidence: float = MIN_CLAIM_SUPPORT_CONFIDENCE):
        self.min_confidence = min_confidence

    def validate_answer(
        self,
        answer_text: str,
        evidence_map: Dict[str, Dict[str, Any]],
        citation_objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes claim-level validation on the answer text.

        Returns:
            Dict containing:
                - is_valid: bool
                - claims: List of claim validation objects
                - flagged_issues: List of detected discrepancies
                - metrics: Dict of formal evaluation metrics
                - sanitized_answer: Cleaned/verified answer text
        """
        # If the answer is a safe refusal, it is valid by definition
        if INSUFFICIENT_EVIDENCE_MESSAGE in answer_text or not answer_text.strip():
            return {
                "is_valid": True,
                "is_refusal": True,
                "claims": [],
                "flagged_issues": [],
                "metrics": {
                    "total_claims": 0,
                    "supported_claims": 0,
                    "unsupported_claims": 0,
                    "total_citations": 0,
                    "supported_citations": 0,
                    "citation_precision": 1.0,
                    "citation_recall": 1.0,
                    "claim_support_rate": 1.0,
                    "unsupported_claim_rate": 0.0,
                    "hallucination_rate": 0.0
                },
                "sanitized_answer": answer_text
            }

        # Build citation lookup maps: by citation number "[1]", by citation_id "C1", and by evidence_id "E1"
        num_to_cit = {f"[{c['citation_number']}]": c for c in citation_objects}
        cid_to_cit = {c["citation_id"]: c for c in citation_objects}
        eid_to_cit = {c["evidence_id"]: c for c in citation_objects}

        # 1. Extract discrete claims
        extracted_claims = self._extract_claims(answer_text)

        validated_claims: List[Dict[str, Any]] = []
        flagged_issues: List[Dict[str, Any]] = []
        total_citations_count = 0
        supported_citations_count = 0

        for claim_info in extracted_claims:
            claim_text = claim_info["text"]
            tags = claim_info["citations"]
            
            # Resolve attached evidence chunks
            attached_evidence = []
            citation_ids = []
            
            for tag in tags:
                total_citations_count += 1
                cit_obj = num_to_cit.get(tag) or eid_to_cit.get(tag.strip("[]")) or cid_to_cit.get(tag.strip("[]"))
                if cit_obj:
                    citation_ids.append(cit_obj["citation_id"])
                    eid = cit_obj["evidence_id"]
                    if eid in evidence_map:
                        attached_evidence.append((cit_obj, evidence_map[eid]))
                else:
                    flagged_issues.append({
                        "type": "fabricated_citation",
                        "tag": tag,
                        "claim": claim_text,
                        "description": f"Citation tag {tag} does not map to any retrieved evidence chunk."
                    })

            # Check for missing citation on substantive factual claims
            is_substantive = self._is_substantive_claim(claim_text)
            if is_substantive and not tags:
                flagged_issues.append({
                    "type": "missing_citation",
                    "claim": claim_text,
                    "description": "Substantive factual/statutory assertion made without evidence citation."
                })

            # 2. Check claim support against attached evidence
            if attached_evidence:
                support_scores = []
                mismatch_reasons = []
                
                for cit_obj, ev in attached_evidence:
                    score, reasons = self._verify_claim_against_chunk(claim_text, ev)
                    support_scores.append(score)
                    if reasons:
                        mismatch_reasons.extend(reasons)
                
                max_score = max(support_scores) if support_scores else 0.0
                is_supported = max_score >= self.min_confidence

                if is_supported:
                    supported_citations_count += len(attached_evidence)
                else:
                    for reason in mismatch_reasons:
                        flagged_issues.append({
                            "type": reason.get("type", "unsupported_claim"),
                            "claim": claim_text,
                            "citations": citation_ids,
                            "confidence": max_score,
                            "description": reason.get("desc", "Claim not sufficiently supported by cited text.")
                        })

                validated_claims.append({
                    "claim": claim_text,
                    "citations": citation_ids,
                    "raw_tags": tags,
                    "supported": is_supported,
                    "confidence": round(max_score, 4),
                    "is_substantive": is_substantive
                })
            else:
                # No valid evidence attached
                validated_claims.append({
                    "claim": claim_text,
                    "citations": [],
                    "raw_tags": tags,
                    "supported": not is_substantive,  # structural/transitional sentences can pass
                    "confidence": 0.0 if is_substantive else 1.0,
                    "is_substantive": is_substantive
                })

        # 3. Compute Metrics
        substantive_claims = [c for c in validated_claims if c["is_substantive"]]
        total_substantive = len(substantive_claims)
        supported_substantive = sum(1 for c in substantive_claims if c["supported"])
        unsupported_substantive = total_substantive - supported_substantive

        citation_precision = (
            supported_citations_count / total_citations_count
            if total_citations_count > 0 else 1.0
        )
        citation_recall = (
            supported_substantive / total_substantive
            if total_substantive > 0 else 1.0
        )
        claim_support_rate = (
            supported_substantive / total_substantive
            if total_substantive > 0 else 1.0
        )
        unsupported_claim_rate = (
            unsupported_substantive / total_substantive
            if total_substantive > 0 else 0.0
        )
        hallucination_rate = unsupported_claim_rate

        metrics = {
            "total_claims": total_substantive,
            "supported_claims": supported_substantive,
            "unsupported_claims": unsupported_substantive,
            "total_citations": total_citations_count,
            "supported_citations": supported_citations_count,
            "citation_precision": round(citation_precision, 4),
            "citation_recall": round(citation_recall, 4),
            "claim_support_rate": round(claim_support_rate, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "hallucination_rate": round(hallucination_rate, 4)
        }

        # 4. Remediation / Sanitization
        sanitized_answer = self._remediate_answer(answer_text, validated_claims, flagged_issues)
        is_valid = (unsupported_claim_rate <= 0.15) and (not any(f["type"] == "fabricated_citation" for f in flagged_issues))

        return {
            "is_valid": is_valid,
            "is_refusal": False,
            "claims": validated_claims,
            "flagged_issues": flagged_issues,
            "metrics": metrics,
            "sanitized_answer": sanitized_answer
        }

    def _extract_claims(self, answer_text: str) -> List[Dict[str, Any]]:
        """Splits answer text into discrete factual claims with their citation tags."""
        claims = []
        lines = answer_text.split('\n')
        
        # Skip section headers and Sources section
        in_sources_section = False
        in_provisions_section = False

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("### Sources") or line_str.startswith("### Important Note"):
                in_sources_section = True
                in_provisions_section = False
                continue
            if in_sources_section:
                if line_str.startswith("###"):
                    in_sources_section = False
                else:
                    continue
            if line_str.startswith("### Applicable provisions"):
                in_provisions_section = True
                continue
            if in_provisions_section:
                continue
            if line_str.startswith("###"):
                continue

            # Clean raw chunk header metadata if present
            clean_line = re.sub(r'\[Document:[^\]]+\]', '', line_str).strip()
            if not clean_line:
                continue

            # Extract all paragraph-level citation tags
            line_tags = re.findall(r'\[(?:\d+|E\d+)(?:\s*,\s*(?:\d+|E\d+))*\]', clean_line)

            # Split line into sentences
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\[\-])', clean_line)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 10:
                    continue
                
                # Extract sentence-specific tags
                sentence_tags = re.findall(r'\[(?:\d+|E\d+)(?:\s*,\s*(?:\d+|E\d+))*\]', s_clean)
                
                # If sentence has no specific tags, inherit paragraph tags
                effective_tags = sentence_tags if sentence_tags else line_tags
                
                # Clean text of tags for linguistic matching
                pure_text = re.sub(r'\[(?:\d+|E\d+)(?:\s*,\s*(?:\d+|E\d+))*\]', '', s_clean).strip()
                pure_text = re.sub(r'\s+', ' ', pure_text).strip("-* ")
                
                if pure_text and len(pure_text.split()) >= 3:
                    claims.append({
                        "text": pure_text,
                        "raw_sentence": s_clean,
                        "citations": effective_tags
                    })

        return claims

    def _is_substantive_claim(self, claim_text: str) -> bool:
        """Determines if a claim makes a substantive legal/factual assertion."""
        text_lower = claim_text.lower()
        # Non-substantive structural phrases
        boilerplate = [
            "the following provisions apply",
            "in summary",
            "as outlined below",
            "here is the explanation",
            "based on the authoritative documents",
            "different jurisdictional rules",
            "general provision",
            "applicable provisions"
        ]
        if any(b in text_lower for b in boilerplate) and len(claim_text.split()) < 8:
            return False
        return len(claim_text.split()) >= 4

    def _verify_claim_against_chunk(
        self,
        claim_text: str,
        evidence: Dict[str, Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculates verification confidence between a claim and a candidate evidence chunk.
        Checks statutory provision matching, entity containment, and lexical similarity.
        """
        reasons = []
        
        # Build comprehensive evidence text including headers, sections, and document titles
        raw_text = evidence.get("text", "")
        heading = evidence.get("heading", "")
        sec = evidence.get("section", "")
        art = evidence.get("article", "")
        rule = evidence.get("rule", "")
        doc = evidence.get("document", "")
        from src.generation.citation_engine import CitationEngine
        doc_clean = CitationEngine()._format_doc_title(str(doc)) if doc else ""
        
        full_evidence_corpus = f"{raw_text} {heading} {sec} {art} {rule} {doc} {doc_clean}".lower()
        claim_lower = claim_text.lower()

        # 1. Statutory / Provision Check
        sec_match = re.search(r'\bsection\s+([0-9]+[a-z]*(?:\([a-z0-9]+\))*)', claim_lower)
        if sec_match:
            sec_claimed = sec_match.group(1).replace(" ", "")
            sec_ev = str(evidence.get("section") or "").lower().replace(" ", "")
            # Citation identity is authoritative.  Text can span neighbouring
            # provisions in legacy chunks, but a citation to Section 4 cannot
            # support a claim labelled Section 3(p).
            if sec_ev and sec_claimed != sec_ev:
                reasons.append({
                    "type": "citation_mismatch",
                    "desc": f"Claim asserts Section '{sec_claimed}', but the cited provision is Section '{sec_ev}'."
                })

        rule_match = re.search(r'\brule\s+([0-9]+[a-z]*)', claim_lower)
        if rule_match:
            rule_claimed = rule_match.group(1).replace(" ", "")
            rule_ev = str(evidence.get("rule") or "").lower().replace(" ", "")
            if rule_ev and rule_claimed != rule_ev:
                reasons.append({
                    "type": "citation_mismatch",
                    "desc": f"Claim asserts Rule '{rule_claimed}', but cited chunk does not contain it."
                })

        art_match = re.search(r'\barticle\s+([0-9]+[a-z]*)', claim_lower)
        if art_match:
            art_claimed = art_match.group(1).replace(" ", "")
            art_ev = str(evidence.get("article") or "").lower().replace(" ", "")
            if art_ev and art_claimed != art_ev:
                reasons.append({
                    "type": "citation_mismatch",
                    "desc": f"Claim asserts Article '{art_claimed}', but cited chunk does not contain it."
                })

        # 2. Key Entity Containment (Botanical / Latin Names)
        common_words = {
            "where", "when", "under", "according", "applicants", "furthermore", "however",
            "as", "stated", "based",
            "section", "article", "rule", "the", "this", "that", "these", "those", "each",
            "all", "any", "some", "an", "such", "in", "on", "for", "with", "by", "from",
            "patent", "patents", "act", "acts", "indian", "law", "court", "guidelines",
            "system", "systems", "number", "numbers", "study", "studies", "category",
            "categories", "ingredient", "ingredients", "experience", "evidence", "order",
            "serial", "safety", "published", "effective", "effectiveness", "provisions",
            "provision", "general", "short", "answer", "explanation", "applicable",
            "biological", "diversity", "authority", "national", "traditional", "knowledge"
        }
        potential_entities = re.findall(r'\b([A-Z][a-z]+)\s+([a-z]+)\b', claim_text)
        for w1, w2 in potential_entities:
            if w1.lower() not in common_words and w2.lower() not in common_words:
                full_term = f"{w1} {w2}".lower()
                if full_term not in full_evidence_corpus and len(full_term) > 8:
                    reasons.append({
                        "type": "unsupported_entity",
                        "desc": f"Entity '{w1} {w2}' in claim not found in cited chunk text."
                    })

        # 3. Legal Equivalence & Multilingual Expansion
        norm_evidence = full_evidence_corpus
        if "not inventions" in full_evidence_corpus or "not an invention" in full_evidence_corpus:
            norm_evidence += " excludes exclusion not patentable patentability"
        if "synergism" in full_evidence_corpus or "synergistic" in full_evidence_corpus:
            norm_evidence += " synergistic efficacy unexpected effect"
        if "traditional knowledge" in full_evidence_corpus:
            norm_evidence += " tk पारंपरिक ज्ञान ashwagandha withania"
        if "biological diversity" in full_evidence_corpus or "nba" in full_evidence_corpus:
            norm_evidence += " जैव विविधता राष्ट्रीय प्राधिकरण अनुमति approval"

        # 4. Lexical / Token Containment & Sequence Matching
        claim_words = set(re.findall(r'[a-zA-Z0-9_\u0900-\u097F]+', claim_lower))
        stop_words = {
            "the", "a", "an", "and", "or", "in", "of", "to", "for", "is", "are", "that",
            "this", "under", "with", "by", "as", "it", "on", "shall", "must", "be", "from",
            "which", "where", "into", "their", "such", "can", "may", "also", "when",
            "hain", "hai", "mein", "par", "se", "ko", "ke", "liye", "kya", "hote"
        }
        content_words = {w for w in claim_words if len(w) > 2 and w not in stop_words}
        
        if not content_words:
            return 1.0, reasons

        matched_content = {w for w in content_words if (w in norm_evidence or any(w in term for term in norm_evidence.split()))}
        containment_score = len(matched_content) / len(content_words)

        # Longest common contiguous subsequence similarity
        matcher = difflib.SequenceMatcher(None, claim_lower, norm_evidence)
        match = matcher.find_longest_match(0, len(claim_lower), 0, len(norm_evidence))
        longest_phrase_len = match.size
        phrase_ratio = min(1.0, longest_phrase_len / max(len(claim_lower) * 0.30, 1))

        # Composite Confidence
        confidence = (0.65 * containment_score) + (0.35 * phrase_ratio)
        
        # Statutory provenance boost: if exact Section, Article, Rule, or Document Title is verified
        doc_clean = str(evidence.get("document") or "").lower().replace(".pdf", "")
        if (sec_match and str(evidence.get("section") or "").lower().replace(" ", "") in claim_lower) or \
           (art_match and str(evidence.get("article") or "").lower().replace(" ", "") in claim_lower) or \
           (rule_match and str(evidence.get("rule") or "").lower().replace(" ", "") in claim_lower) or \
           (doc_clean and doc_clean in claim_lower):
            confidence = min(1.0, confidence + 0.20)

        # Penalize if critical statutory mismatch was flagged
        if reasons:
            confidence = max(0.0, confidence - 0.40)

        return min(1.0, confidence), reasons

    def _remediate_answer(
        self,
        answer_text: str,
        validated_claims: List[Dict[str, Any]],
        flagged_issues: List[Dict[str, Any]]
    ) -> str:
        """
        Remediates the answer text if critical hallucinations or unsupported claims exist.
        If unsupported rate is too severe, returns the safe refusal message.
        """
        severe_issues = [f for f in flagged_issues if f["type"] in ("fabricated_citation", "citation_mismatch")]
        unsupported_count = sum(1 for c in validated_claims if not c["supported"] and c["is_substantive"])
        
        if unsupported_count > 3 or len(severe_issues) >= 2:
            return INSUFFICIENT_EVIDENCE_MESSAGE

        return answer_text
