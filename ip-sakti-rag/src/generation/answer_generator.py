"""Answer Generator for IP-SAKTI Sahayak.

Invokes Gemini with strict legal/regulatory grounding rules (Rules 1-7),
evidence citation tagging ([E1], [E2], ...), structured markdown output sections,
natural sentence case formatting, and safe refusal handling.
"""
from typing import Dict, List, Any, Optional, Tuple
import os
import re
import time
import logging

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GENERATION_TEMPERATURE,
    GENERATION_MAX_TOKENS,
    GENERATION_TOP_P,
    INSUFFICIENT_EVIDENCE_MESSAGE
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are IP-SAKTI Sahayak, an authoritative AI legal and regulatory assistant specializing in Indian Intellectual Property Law, Traditional Knowledge, AYUSH/Ayurveda regulations, Biological Diversity laws, and international patent systems (PCT, WIPO, EPO).

STRICT GROUNDING RULES (MANDATORY):
1. RULE 1 (Strict Evidence Containment): Only make factual, legal, and regulatory claims that are DIRECTLY SUPPORTED by the provided evidence chunks.
2. RULE 2 (No Fabrication): Do NOT invent or assume laws, sections, subsections, rules, articles, patent numbers, guidelines, dates, or regulatory requirements.
3. RULE 3 (No Pretrained Extrapolation): Do NOT use your pretrained knowledge to fill in missing information or invent legal precedence.
4. RULE 4 (Insufficient Evidence Refusal): If the provided evidence is absent, irrelevant, or insufficient to answer the query conclusively, you MUST respond ONLY with:
   "I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."
5. RULE 5 (Distinguish Support vs Uncertainty): Clearly distinguish directly supported statutory facts from areas where evidence is partial or silent.
6. RULE 6 (Zero Fabricated Citations): Never fabricate a citation. Every substantive claim MUST cite the exact evidence identifier (e.g. [E1], [E2]) from which it is derived.
7. RULE 7 (Accurate Attribution): Do not cite an evidence ID merely because it is listed. The cited evidence MUST genuinely support the specific claim.

STYLE & FORMATTING RULES:
- Write in natural sentence case.
- Do NOT write the answer in ALL CAPS.
- Use concise, well-structured paragraphs.
- Preserve the exact capitalization of legal document titles, statute names, section numbers, rules, and proper nouns (e.g., "Patents Act, 1970", "Section 3(p)", "Ayurveda Aahara Regulations 2022", "National Biodiversity Authority").
- Follow the required markdown section format strictly:

### Short answer
[A clear, direct answer in normal sentence case, citing relevant evidence tags like [E1].]

### Explanation
[Detailed legal/regulatory explanation grounded strictly in the cited evidence. Every substantive sentence must cite evidence tags like [E1], [E2].]

### Applicable provisions
- Section X — [concise explanation of how it applies] [E1]
- Rule Y — [concise explanation of how it applies] [E2]

### Sources
[Leave this for citation post-processing, or list [E1], [E2] tags.]
"""


def normalize_sentence_case(text: str) -> str:
    """Converts ALL-CAPS text blocks or headers into clean natural sentence case."""
    if not text:
        return ""
    
    # If string is mostly uppercase (e.g. headers from official PDFs)
    letters = [c for c in text if c.isalpha()]
    if letters and (sum(1 for c in letters if c.isupper()) / len(letters)) > 0.70:
        # Lowercase everything except first character and legal acronyms
        words = text.split()
        cleaned_words = []
        for w in words:
            # Preserve recognized legal abbreviations/acronyms
            if w.upper() in {"IPR", "PCT", "WIPO", "EPO", "NBA", "SBB", "BMC", "TKDL", "TK", "ASU", "AYUSH", "FSSAI", "GMP", "TRIPS", "WTO"}:
                cleaned_words.append(w.upper())
            elif re.match(r'^(?:[0-9]+[A-Z]*|[A-Z])$', w):
                cleaned_words.append(w)
            else:
                cleaned_words.append(w.lower())
        
        reconstructed = " ".join(cleaned_words)
        # Capitalize first letter of sentences
        return re.sub(r'(^[a-z]|(?<=[.!?]\s)[a-z])', lambda m: m.group(1).upper(), reconstructed)
    
    return text


class AnswerGenerator:
    """Generates grounded answers using Gemini with strict citation and fallback support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = GENERATION_TEMPERATURE,
        max_tokens: int = GENERATION_MAX_TOKENS,
        top_p: float = GENERATION_TOP_P
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        self.model_name = model_name or os.getenv("GEMINI_MODEL") or GEMINI_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._gemini_client = None
        self._init_client()

    def _init_client(self):
        """Initializes the Gemini client if API key is present."""
        if not self.api_key:
            logger.info("GEMINI_API_KEY not configured. Operating in deterministic offline fallback mode.")
            return

        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=self.api_key)
            self._client_type = "genai"
            logger.info(f"Initialized google.genai Client with model: {self.model_name}")
        except Exception:
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self._gemini_client = gai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_PROMPT
                )
                self._client_type = "generativeai"
                logger.info(f"Initialized google.generativeai GenerativeModel with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini SDK client: {e}. Falling back to offline engine.")
                self._gemini_client = None

    def generate(
        self,
        query: str,
        formatted_evidence: str,
        evidence_map: Dict[str, Dict[str, Any]],
        detected_conflicts: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates a strictly grounded response to the query using the formatted evidence."""
        start_time = time.perf_counter()

        # Check for empty evidence
        if not evidence_map or not formatted_evidence.strip() or formatted_evidence.startswith("No relevant authoritative evidence"):
            return INSUFFICIENT_EVIDENCE_MESSAGE, {
                "model": "rule_based_safety_refusal",
                "latency_sec": round(time.perf_counter() - start_time, 4),
                "status": "refused_empty_evidence",
                "tokens": 0
            }

        user_content = self._construct_user_prompt(query, formatted_evidence, detected_conflicts)

        # 1. Live Gemini Generation (if client available)
        if self._gemini_client is not None and self.api_key:
            try:
                raw_answer, usage_meta = self._call_gemini(user_content)
                elapsed = round(time.perf_counter() - start_time, 4)
                
                is_refusal = (
                    INSUFFICIENT_EVIDENCE_MESSAGE in raw_answer
                    or "could not find sufficient authoritative evidence" in raw_answer.lower()
                    or "insufficient evidence" in raw_answer.lower()
                )
                
                return raw_answer, {
                    "model": self.model_name,
                    "latency_sec": elapsed,
                    "status": "refused_by_model" if is_refusal else "success_live_llm",
                    "usage": usage_meta
                }
            except Exception as e:
                logger.error(f"Live Gemini API call failed: {e}. Utilizing deterministic fallback generator.")

        # 2. Deterministic Grounded Fallback Generation (for offline testing / no API key)
        raw_answer = self._generate_deterministic_grounded_answer(query, evidence_map, detected_conflicts)
        elapsed = round(time.perf_counter() - start_time, 4)
        
        is_refusal = INSUFFICIENT_EVIDENCE_MESSAGE in raw_answer
        return raw_answer, {
            "model": f"deterministic_grounded_engine (model_target: {self.model_name})",
            "latency_sec": elapsed,
            "status": "refused_by_rules" if is_refusal else "success_offline_grounded",
            "usage": {"prompt_tokens": len(user_content.split()), "completion_tokens": len(raw_answer.split())}
        }

    def _construct_user_prompt(
        self,
        query: str,
        formatted_evidence: str,
        detected_conflicts: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Assembles the user prompt containing query, grounding instructions, and evidence blocks."""
        prompt = [
            f"USER QUERY:\n{query}\n",
            f"AUTHORITATIVE EVIDENCE CHUNKS:\n{formatted_evidence}\n",
            "INSTRUCTIONS:",
            "- Base your response ONLY on the authoritative evidence above.",
            "- Use [E1], [E2], etc. to cite evidence for every substantive claim.",
            "- Write in natural sentence case. Do NOT write in ALL CAPS.",
            "- Follow the required markdown section format strictly (### Short answer, ### Explanation, ### Applicable provisions, ### Sources).",
            "- If the provided chunks do not contain enough information to answer the query, respond ONLY with:",
            f'  "{INSUFFICIENT_EVIDENCE_MESSAGE}"'
        ]
        return "\n".join(prompt)

    def _call_gemini(self, user_content: str) -> Tuple[str, Dict[str, Any]]:
        """Executes the API request against Gemini."""
        if self._client_type == "genai":
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=self.top_p
            )
            response = self._gemini_client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=config
            )
            text = response.text or ""
            usage = {
                "prompt_token_count": getattr(response.usage_metadata, "prompt_token_count", 0),
                "candidates_token_count": getattr(response.usage_metadata, "candidates_token_count", 0)
            }
            return text, usage
        else:
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": self.top_p
            }
            response = self._gemini_client.generate_content(
                user_content,
                generation_config=generation_config
            )
            text = response.text or ""
            usage = {}
            return text, usage

    def _generate_deterministic_grounded_answer(
        self,
        query: str,
        evidence_map: Dict[str, Dict[str, Any]],
        detected_conflicts: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Synthesizes a strictly grounded answer in clean, natural sentence case."""
        if not evidence_map:
            return INSUFFICIENT_EVIDENCE_MESSAGE

        e1 = evidence_map.get("E1", {})
        e2 = evidence_map.get("E2", {})
        
        doc1 = e1.get("document", "the relevant authoritative document")
        sec1 = e1.get("section")
        art1 = e1.get("article")
        rule1 = e1.get("rule")
        heading1 = e1.get("heading", "")
        text1 = e1.get("text", "")
        
        clean_text1 = re.sub(r'\[Document:[^\]]+\]', '', text1)
        clean_text1 = " ".join(clean_text1.split()).strip()

        prov_label1 = f"Section {sec1}" if sec1 else (f"Article {art1}" if art1 else (f"Rule {rule1}" if rule1 else (heading1 if heading1 else "")))
        
        sentences1 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text1) if len(s.strip()) > 20 and not s.strip().startswith("(")]
        if not sentences1:
            sentences1 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text1) if len(s.strip()) > 10]
        
        raw_sentence = sentences1[0] if sentences1 else clean_text1[:220]
        primary_sentence = normalize_sentence_case(raw_sentence)

        answer_lines = [
            "### Short answer",
            f"Under the provisions of {doc1}" + (f" ({prov_label1})" if prov_label1 else "") + f", {primary_sentence.strip()} [E1]"
        ]

        exp_lines = [
            "\n### Explanation",
            f"According to authoritative documentation in {doc1}" + (f" ({prov_label1})" if prov_label1 else "") + f", {primary_sentence} [E1]"
        ]
        
        if len(sentences1) > 1:
            sec_raw = sentences1[1] if len(sentences1[1]) < 250 else sentences1[1][:240] + "..."
            sec_sentence = normalize_sentence_case(sec_raw)
            exp_lines.append(f"Furthermore, {sec_sentence} [E1]")
        
        if e2:
            doc2 = e2.get("document", "")
            sec2 = e2.get("section")
            art2 = e2.get("article")
            rule2 = e2.get("rule")
            heading2 = e2.get("heading", "")
            prov_label2 = f"Section {sec2}" if sec2 else (f"Article {art2}" if art2 else (f"Rule {rule2}" if rule2 else (heading2 if heading2 else "")))
            clean_text2 = re.sub(r'\[Document:[^\]]+\]', '', e2.get("text", ""))
            clean_text2 = " ".join(clean_text2.split()).strip()
            sentences2 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text2) if len(s.strip()) > 20 and not s.strip().startswith("(")]
            if sentences2:
                norm_sentence2 = normalize_sentence_case(sentences2[0])
                exp_lines.append(f"In addition, under {doc2}" + (f" ({prov_label2})" if prov_label2 else "") + f", {norm_sentence2} [E2]")

        prov_lines = ["\n### Applicable provisions"]
        for eid, e in evidence_map.items():
            doc = e.get("document", "")
            s = e.get("section")
            a = e.get("article")
            r = e.get("rule")
            h = e.get("heading")
            tag = f"Section {s}" if s else (f"Article {a}" if a else (f"Rule {r}" if r else (h if h else "General Provision")))
            prov_lines.append(f"- {tag}, {doc} [{eid}]")

        src_lines = ["\n### Sources"]
        for eid in evidence_map.keys():
            src_lines.append(f"[{eid}]")

        note_lines = []
        if detected_conflicts:
            note_lines.append("\n### Important note")
            for c in detected_conflicts:
                note_lines.append(f"- {c['description']}")

        full_response = "\n".join(answer_lines) + "\n" + "\n".join(exp_lines) + "\n" + "\n".join(prov_lines) + "\n" + "\n".join(src_lines)
        if note_lines:
            full_response += "\n" + "\n".join(note_lines)

        return full_response.strip()
