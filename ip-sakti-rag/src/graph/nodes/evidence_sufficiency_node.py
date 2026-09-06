"""Evidence Sufficiency Node for IP-SAKTI Sahayak LangGraph.

Evaluates whether retrieved and reranked evidence contains sufficient authoritative
statutory and domain coverage to answer the user query conclusively.
Validates domain compatibility, filters irrelevant chunks, formats structured evidence blocks,
and detects multi-jurisdictional source conflicts.
"""
from typing import Dict, Any, List, Optional, Set
import time
import re
from src.graph.state import GraphState
from src.generation.evidence_formatter import EvidenceFormatter
from src.config import (
    MIN_COMBINED_EVIDENCE_SCORE,
    MIN_EVIDENCE_SCORE,
    MIN_DOMAIN_COVERAGE,
    MIN_SUFFICIENCY_CHUNKS,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    SOURCE_HIERARCHY,
)
from src.retrieval.legal_identifier_parser import document_matches, provision_matches


class EvidenceSufficiencyNode:
    """Evaluates evidence sufficiency across scores, exact statutory matches, and domain coverage."""

    def __init__(self, evidence_formatter: Optional[EvidenceFormatter] = None):
        self.evidence_formatter = evidence_formatter or EvidenceFormatter()

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Assesses evidence sufficiency and prepares structured evidence blocks."""
        t0 = time.perf_counter()
        
        query = state.get("query", "")
        query_type = state.get("query_type", "FACTUAL")
        scope_status = state.get("scope_status", "IN_SCOPE")
        selected_evidence = list(state.get("selected_evidence", []))
        exact_identifiers = state.get("exact_identifiers", [])
        parsed_identifier = state.get("parsed_identifier") or {}
        query_domains = state.get("domains", [])

        # 1. Gate: If query was out of scope, evidence must be empty
        if scope_status == "OUT_OF_SCOPE" or query_type == "OUT_OF_SCOPE":
            return self._build_insufficient_result(
                state, t0, "Query is outside the supported legal/AYUSH domain scope."
            )

        # 2. Check if evidence is empty
        if not selected_evidence or len(selected_evidence) < MIN_SUFFICIENCY_CHUNKS:
            return self._build_insufficient_result(
                state, t0, "No relevant candidate chunks found in knowledge base."
            )

        # 3. Domain Compatibility & Semantic Filtering.  The resulting set,
        # rather than only its first item, is the unit of sufficiency.
        filtered_evidence = self._filter_compatible_evidence(query, selected_evidence, query_domains)
        # Diversity selection may omit the only chunks matching a narrow
        # intent. Reuse the existing reranked pool before refusing; retrieval
        # and reranking remain unchanged, and the same relevance/authority
        # checks still apply.
        if not filtered_evidence:
            candidate_pool = list(state.get("reranked_candidates", []))
            candidate_pool.extend(state.get("retrieval_candidates", []))
            seen_ids = set()
            candidate_pool = [
                chunk for chunk in candidate_pool
                if chunk.get("chunk_id") not in seen_ids and not seen_ids.add(chunk.get("chunk_id"))
            ]
            filtered_evidence = self._filter_compatible_evidence(query, candidate_pool, query_domains)
        if len(filtered_evidence) < MIN_SUFFICIENCY_CHUNKS:
            return self._build_insufficient_result(
                state, t0, "Retrieved chunks are weakly related or from incompatible domains."
            )

        diagnostics = self._evidence_diagnostics(state, filtered_evidence)
        if not self._has_sufficient_evidence(state, filtered_evidence, diagnostics):
            return self._build_insufficient_result(
                state,
                t0,
                self._insufficiency_reason(state, diagnostics),
                diagnostics,
            )

        # Current fee questions may only be answered from an authoritative,
        # fee-specific and dated source.  An Act section about duration or
        # renewal is not evidence of a current amount.
        if query_type == "CURRENT_FEE_LOOKUP":
            fee_evidence = []
            for chunk in filtered_evidence:
                corpus = " ".join(str(chunk.get(key) or "") for key in ("text", "heading", "document", "rule", "regulation")).lower()
                has_amount = bool(re.search(r"(?:₹|rs\.?|inr)\s*\d|\d[\d,]*\s*(?:rupees|rs\.?)", corpus))
                official_fee_source = any(term in corpus for term in ("fee schedule", "fees", "notification", "rules"))
                if has_amount and official_fee_source:
                    fee_evidence.append(chunk)
            if not fee_evidence:
                return self._build_insufficient_result(
                    state, t0, "The knowledge base does not contain authoritative current fee-schedule evidence for this request.",
                    diagnostics,
                )
            filtered_evidence = fee_evidence
            diagnostics = self._evidence_diagnostics(state, filtered_evidence)

        # 5. An exact lookup is sufficient only with the requested provision
        # and requested Act.  Semantic score can never waive this invariant.
        if parsed_identifier.get("type") and parsed_identifier.get("value"):
            exact_evidence = [c for c in filtered_evidence if provision_matches(c, parsed_identifier)
                              and document_matches(c, parsed_identifier.get("canonical_title") or parsed_identifier.get("document_hint"))
                              and self._authority_tier(c) in {1, 2}]
            if not exact_evidence:
                return self._build_insufficient_result(
                    state, t0, f"Requested exact provision {exact_identifiers} was not found in the requested authoritative document.",
                    diagnostics,
                )
            filtered_evidence = exact_evidence
            diagnostics = self._evidence_diagnostics(state, filtered_evidence)

        # 6. Format Evidence and Detect Conflicts
        formatted_evidence, evidence_map, conflicts = self.evidence_formatter.format_evidence(
            chunks=filtered_evidence,
            query=query
        )

        latency = round((time.perf_counter() - t0) * 1000, 2)
        node_latencies = dict(state.get("node_latencies_ms", {}))
        node_latencies["evidence_sufficiency_ms"] = latency

        trace_entry = {
            "node": "evidence_sufficiency",
            "evidence_sufficient": True,
            "selected_evidence_count": len(filtered_evidence),
            "top_score": diagnostics["max_score"],
            "evidence_diagnostics": diagnostics,
            "conflicts_detected": len(conflicts),
            "latency_ms": latency
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "evidence_sufficient": True,
            "evidence_sufficiency_reason": f"Sufficient evidence verified ({len(filtered_evidence)} chunks, top score: {diagnostics['max_score']:.4f}).",
            "selected_evidence": filtered_evidence,
            "evidence": filtered_evidence,
            "formatted_evidence": formatted_evidence,
            "evidence_map": evidence_map,
            "detected_conflicts": conflicts,
            "evidence_sufficiency_diagnostics": diagnostics,
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }

    def _filter_compatible_evidence(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        query_domains: List[str]
    ) -> List[Dict[str, Any]]:
        """Filters candidate chunks to ensure substantive relevance and domain compatibility."""
        q_lower = query.lower()

        # Reject explicitly out-of-scope or foreign legal queries immediately
        foreign_jurisdictions = ["brazilian", "brazil", "germany", "german", "california", "australia", "japanese", "uk patent", "us patent"]
        if any(fj in q_lower for fj in foreign_jurisdictions) and not any(ij in q_lower for ij in ["pct", "wipo", "epo", "paris convention"]):
            return []

        # Extract tokens of length >= 1 to capture alphanumeric section identifiers (e.g. 3, 3p, 43bis)
        query_tokens = set(re.findall(r'[a-zA-Z0-9_\u0900-\u097F]+', q_lower))
        
        # Generic words ignored during keyword check
        stop_words = {
            "what", "the", "are", "and", "for", "how", "can", "kya", "hai", "mein", "par",
            "section", "rule", "article", "act", "india", "law", "patent", "patents",
            "is", "an", "of", "in", "to", "does", "state", "regarding"
        }
        distinctive_tokens = query_tokens - stop_words
        patentability_terms = {
            "patent", "patented", "patentable", "patentability", "invention",
            "inventions", "prior", "novelty", "inventive", "obvious", "claim",
            "claims", "pharmacopoeia", "formulation", "composition", "traditional",
        }
        query_requests_patentability = bool(query_tokens.intersection(patentability_terms))
        formulation_terms = {"formulation", "formulations", "pharmacopoeia", "pharmacopoeias", "composition", "compositions"}
        query_requests_formulation = bool(query_tokens.intersection(formulation_terms))
        formulation_evidence_terms = (*formulation_terms, "known plants", "known medicinal")
        patentability_evidence_terms = (
            "patentable", "patentability", "invention", "inventions",
            "prior art", "novelty", "inventive step", "not patentable",
        )

        compatible_chunks = []
        for chunk in chunks:
            chunk_text = str(chunk.get("text") or "").lower()
            doc_val = chunk.get("document") or (chunk.get("metadata") or {}).get("document_name") or ""
            chunk_doc = str(doc_val or "").lower()
            sec_val = chunk.get("section") or (chunk.get("metadata") or {}).get("section") or ""
            chunk_section = str(sec_val or "").lower()
            chunk_full = f"{chunk_text} {chunk_doc} {chunk_section}"

            chunk_score = float(
                chunk.get("rerank_score") or
                chunk.get("reranker_score") or
                chunk.get("score") or
                0.0
            )

            # If distinctive query tokens exist, check overlap
            chunk_domains = self._chunk_domains(chunk)
            domain_match = bool(set(query_domains).intersection(chunk_domains))
            has_patentability_evidence = any(term in chunk_full for term in patentability_evidence_terms)
            has_formulation_evidence = any(term in chunk_full for term in formulation_evidence_terms)
            if distinctive_tokens:
                overlap = [t for t in distinctive_tokens if t in chunk_full]
                if query_requests_patentability and not has_patentability_evidence:
                    continue
                if query_requests_formulation and not has_formulation_evidence:
                    continue
                if len(overlap) > 0 or (domain_match and chunk_score >= MIN_EVIDENCE_SCORE):
                    compatible_chunks.append(chunk)
            else:
                if domain_match and chunk_score >= MIN_EVIDENCE_SCORE:
                    compatible_chunks.append(chunk)

        return compatible_chunks

    def _build_insufficient_result(
        self,
        state: GraphState,
        start_time: float,
        reason: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Builds state dictionary when evidence is deemed insufficient."""
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        node_latencies = dict(state.get("node_latencies_ms", {}))
        node_latencies["evidence_sufficiency_ms"] = latency

        trace_entry = {
            "node": "evidence_sufficiency",
            "evidence_sufficient": False,
            "reason": reason,
            "evidence_diagnostics": diagnostics or {
                "decision": "REFUSE",
                "reason": "NO_SUFFICIENT_EVIDENCE",
                "max_score": 0.0,
                "relevant_evidence_count": 0,
                "domain_coverage": 0.0,
                "exact_identifier_required": bool((state.get("parsed_identifier") or {}).get("type")),
            },
            "latency_ms": latency
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "evidence_sufficient": False,
            "evidence_sufficiency_reason": reason,
            "selected_evidence": [],
            "evidence": [],
            "formatted_evidence": "",
            "evidence_map": {},
            "detected_conflicts": [],
            "evidence_sufficiency_diagnostics": diagnostics or {
                "decision": "REFUSE",
                "reason": "NO_SUFFICIENT_EVIDENCE",
                "max_score": 0.0,
                "relevant_evidence_count": 0,
                "domain_coverage": 0.0,
                "exact_identifier_required": bool((state.get("parsed_identifier") or {}).get("type")),
            },
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }

    @staticmethod
    def _chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
        metadata = chunk.get("metadata") or {}
        return {**metadata, **{k: v for k, v in chunk.items() if v is not None}}

    def _chunk_domains(self, chunk: Dict[str, Any]) -> Set[str]:
        domains = self._chunk_metadata(chunk).get("domain", [])
        if isinstance(domains, str):
            domains = [domains]
        return {str(domain).lower() for domain in domains}

    def _authority_tier(self, chunk: Dict[str, Any]) -> Optional[int]:
        metadata = self._chunk_metadata(chunk)
        tier = metadata.get("tier") or metadata.get("authority_tier")
        if tier is not None:
            try:
                return int(tier)
            except (TypeError, ValueError):
                pass
        document_id = str(metadata.get("document_id") or "")
        if document_id in SOURCE_HIERARCHY:
            return int(SOURCE_HIERARCHY[document_id]["tier"])
        document = str(metadata.get("document") or chunk.get("document") or "").lower()
        if "act" in document or "treaty" in document:
            return 1
        if any(term in document for term in ("guideline", "regulation", "notification", "rules")):
            return 2
        return None

    def _evidence_score(self, chunk: Dict[str, Any]) -> float:
        return float(chunk.get("reranker_score") or chunk.get("rerank_score") or chunk.get("score") or 0.0)

    def _evidence_diagnostics(self, state: GraphState, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        query_domains = {str(domain).lower() for domain in state.get("domains", [])}
        represented_domains = set().union(*(self._chunk_domains(chunk) for chunk in chunks)) if chunks else set()
        covered_domains = query_domains.intersection(represented_domains)
        scores = sorted((self._evidence_score(chunk) for chunk in chunks), reverse=True)
        top_scores = scores[:3]
        top_average = sum(top_scores) / len(top_scores) if top_scores else 0.0
        return {
            "decision": "PASS",
            "reason": "SUFFICIENT_RELEVANT_EVIDENCE",
            "max_score": round(scores[0], 6) if scores else 0.0,
            "top_k_average_score": round(sum(top_scores) / len(top_scores), 6) if top_scores else 0.0,
            "combined_score": round((scores[0] + top_average) if scores else 0.0, 6),
            "relevant_evidence_count": len(chunks),
            "domain_coverage": round(len(covered_domains) / len(query_domains), 4) if query_domains else 1.0,
            "evidence_domains": sorted(represented_domains),
            "authority_tiers": sorted({tier for tier in (self._authority_tier(chunk) for chunk in chunks) if tier is not None}),
            "authoritative_evidence_count": sum(1 for chunk in chunks if self._authority_tier(chunk) in {1, 2}),
            "exact_identifier_required": bool((state.get("parsed_identifier") or {}).get("type")),
        }

    def _has_sufficient_evidence(self, state: GraphState, chunks: List[Dict[str, Any]], diagnostics: Dict[str, Any]) -> bool:
        if not chunks or diagnostics["authoritative_evidence_count"] == 0:
            return False
        if diagnostics["combined_score"] < MIN_COMBINED_EVIDENCE_SCORE:
            return False
        query_domains = set(state.get("domains", []))
        if query_domains and diagnostics["domain_coverage"] < MIN_DOMAIN_COVERAGE:
            return False
        return True

    def _insufficiency_reason(self, state: GraphState, diagnostics: Dict[str, Any]) -> str:
        diagnostics["decision"] = "REFUSE"
        diagnostics["reason"] = "NO_SUFFICIENT_EVIDENCE"
        if diagnostics["authoritative_evidence_count"] == 0:
            return "Relevant evidence was found, but no Tier 1 or Tier 2 authoritative source was represented."
        if diagnostics["domain_coverage"] < MIN_DOMAIN_COVERAGE:
            return "Authoritative evidence does not cover the requested domain(s)."
        return f"Combined relevant evidence score ({diagnostics['combined_score']:.4f}) is below the minimum ({MIN_COMBINED_EVIDENCE_SCORE:.4f})."
