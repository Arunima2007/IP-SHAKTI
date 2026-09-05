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
from src.config import MIN_EVIDENCE_SCORE, MIN_SUFFICIENCY_CHUNKS, INSUFFICIENT_EVIDENCE_MESSAGE
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

        # 3. Check Top Reranker Score
        top_score = float(
            selected_evidence[0].get("reranker_score") or
            selected_evidence[0].get("rerank_score") or
            selected_evidence[0].get("score") or
            0.0
        )
        if top_score < MIN_EVIDENCE_SCORE:
            return self._build_insufficient_result(
                state, t0, f"Top reranker score ({top_score:.4f}) is below minimum threshold ({MIN_EVIDENCE_SCORE})."
            )

        # 4. Domain Compatibility & Semantic Filtering
        filtered_evidence = self._filter_compatible_evidence(query, selected_evidence, query_domains)
        if len(filtered_evidence) < MIN_SUFFICIENCY_CHUNKS:
            return self._build_insufficient_result(
                state, t0, "Retrieved chunks are weakly related or from incompatible domains."
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
                    state, t0, "The knowledge base does not contain authoritative current fee-schedule evidence for this request."
                )
            filtered_evidence = fee_evidence

        # 5. An exact lookup is sufficient only with the requested provision
        # and requested Act.  Semantic score can never waive this invariant.
        if parsed_identifier.get("type") and parsed_identifier.get("value"):
            exact_evidence = [c for c in filtered_evidence if provision_matches(c, parsed_identifier)
                              and document_matches(c, parsed_identifier.get("canonical_title") or parsed_identifier.get("document_hint"))]
            if not exact_evidence:
                return self._build_insufficient_result(
                    state, t0, f"Requested exact provision {exact_identifiers} was not found in the requested authoritative document."
                )
            filtered_evidence = exact_evidence

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
            "top_score": top_score,
            "conflicts_detected": len(conflicts),
            "latency_ms": latency
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "evidence_sufficient": True,
            "evidence_sufficiency_reason": f"Sufficient evidence verified ({len(filtered_evidence)} chunks, top score: {top_score:.4f}).",
            "selected_evidence": filtered_evidence,
            "evidence": filtered_evidence,
            "formatted_evidence": formatted_evidence,
            "evidence_map": evidence_map,
            "detected_conflicts": conflicts,
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
            if distinctive_tokens:
                overlap = [t for t in distinctive_tokens if t in chunk_full]
                if len(overlap) > 0 or chunk_score >= MIN_EVIDENCE_SCORE:
                    compatible_chunks.append(chunk)
            else:
                if chunk_score >= MIN_EVIDENCE_SCORE:
                    compatible_chunks.append(chunk)

        return compatible_chunks

    def _build_insufficient_result(
        self,
        state: GraphState,
        start_time: float,
        reason: str
    ) -> Dict[str, Any]:
        """Builds state dictionary when evidence is deemed insufficient."""
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        node_latencies = dict(state.get("node_latencies_ms", {}))
        node_latencies["evidence_sufficiency_ms"] = latency

        trace_entry = {
            "node": "evidence_sufficiency",
            "evidence_sufficient": False,
            "reason": reason,
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
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }
