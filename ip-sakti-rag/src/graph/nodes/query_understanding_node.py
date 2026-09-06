"""Query Understanding Node for IP-SAKTI Sahayak LangGraph.

Performs deterministic two-stage query understanding:
Stage 1: Strict Domain Scope Verification (IN_SCOPE vs OUT_OF_SCOPE with confidence scoring)
Stage 2: In-Scope Classification, entity extraction, statutory provision detection, and multilingual expansion.
"""
from typing import Dict, Any, List, Tuple, Optional, Set
import re
import time
from src.graph.state import GraphState
from src.retrieval.legal_identifier_parser import parse as parse_legal_identifier


class QueryUnderstandingNode:
    """Classifies user queries and extracts domain, jurisdiction, and statutory metadata."""

    def __init__(self):
        # Exact statutory regex patterns
        self.section_pattern = re.compile(r'\b(?:section|sec\.?|धारा)\s+([0-9]+[a-z]*(?:\([a-z0-9]+\))*)', re.IGNORECASE)
        self.rule_pattern = re.compile(r'\b(?:pct\s+)?(?:rule|नियम)\s+([0-9]+[a-z]*(?:\.[0-9]+)*)', re.IGNORECASE)
        self.article_pattern = re.compile(r'\b(?:article|अनुच्छेद)\s+([0-9]+[a-z]*)', re.IGNORECASE)
        self.patent_num_pattern = re.compile(r'\bpatent\s+(?:no\.?\s*)?([0-9]{5,8})\b', re.IGNORECASE)

        # -------------------------------------------------------------
        # 1. Authoritative In-Scope Keywords & Patterns (Comprehensive)
        # -------------------------------------------------------------
        
        # A. Indian Patent Law & Indian IP
        self.patent_keywords = {
            "patent", "patents", "patented", "patentability", "patentable", "patentee",
            "inventive step", "novelty", "prior art", "obviousness", "claims", "specification",
            "complete specification", "provisional specification", "infringement", "compulsory license",
            "compulsory licensing", "true and first inventor", "controller general", "cgpdtm",
            "indian patent office", "ipo", "patent revocation", "patent opposition", "pre-grant opposition",
            "post-grant opposition", "patents act", "patent act", "patent rules", "patent of addition",
            "trademark", "trademarks", "trade mark", "trade marks", "copyright", "copyrights",
            "design", "designs", "designs act", "trade marks act", "copyright act",
            "ipr", "intellectual property", "पेटेंट", "बौद्धिक संपदा", "आविष्कार", "नवीनता", "ट्रेडमार्क", "कॉपीराइट"
        }

        # B. Ayurveda / AYUSH / ASU Drugs
        self.ayurveda_keywords = {
            "ayurveda", "ayurvedic", "ayush", "asu", "siddha", "unani", "sowa-rigpa", "homeopathy",
            "ayurvedic formulation", "classical formulation", "proprietary medicine", "ayurvedic medicine",
            "herbal", "herb", "herbs", "plant extract", "medicinal plant", "medicinal plants",
            "pharmacopoeia", "ayurvedic pharmacopoeia", "drugs and cosmetics", "drugs & cosmetics",
            "drugs and cosmetics act", "rule 158b", "rule 161", "rule 170", "schedule t", "gmp",
            "good manufacturing practices", "ayurveda aahara", "aahara", "fssai", "withania somnifera",
            "ashwagandha", "curcuma", "turmeric", "triphala", "chyawanprash", "neem", "azadirachta",
            "synergy", "synergism", "admixture",
            "आयुर्वेद", "आयुष", "आहार", "औषधि", "जड़ी-बूटी", "अश्वगंधा", "पारंपरिक चिकित्सा", "दवा"
        }

        # C. Traditional Knowledge
        self.tk_keywords = {
            "traditional knowledge", "traditional medicinal knowledge", "tk", "tkdl",
            "traditional knowledge digital library", "defensive protection", "biopiracy",
            "prior art search in tk", "indigenous knowledge", "traditional use", "traditional uses",
            "folklore", "tce", "wipo tk", "documenting traditional knowledge", "पारंपरिक ज्ञान"
        }

        # D. Biological Resources & Biodiversity
        self.biodiversity_keywords = {
            "biological resource", "biological resources", "biological material", "biological diversity",
            "biodiversity act", "biological diversity act", "nba", "national biodiversity authority",
            "state biodiversity board", "sbb", "biodiversity management committee", "bmc",
            "access and benefit sharing", "abs", "peoples biodiversity register", "pbr",
            "bio-resource", "bio-resources", "genetic resource", "genetic resources",
            "material transfer agreement", "mta", "section 6 approval", "form iii",
            "जैव विविधता", "राष्ट्रीय जैव विविधता प्राधिकरण"
        }

        # E. International IP & Treaties
        self.international_keywords = {
            "wipo", "world intellectual property organization", "pct", "patent cooperation treaty",
            "rule 43bis", "rule 51bis", "international searching authority", "isa",
            "international preliminary examining authority", "ipea", "iprp", "international preliminary report",
            "written opinion", "national phase", "international phase", "31 months", "30 months",
            "paris convention", "right of priority", "priority date", "trips", "trips agreement", "wto",
            "epo", "european patent office", "european patent convention", "epc",
            "wipo gr/tk treaty", "wipo treaty on genetic resources"
        }

        # -------------------------------------------------------------
        # 2. Out-of-Scope Negative Patterns (Explicit Red Lines)
        # -------------------------------------------------------------
        self.out_of_scope_patterns = [
            # Sports & Entertainment
            r'\b(?:cricket|ipl|football|fifa|messi|ronaldo|virat|kohli|dhoni|sachin|bollywood|hollywood|movie|movies|actor|actress|cinema|celebrity|oscar|song|songs|trailer)\b',
            # Cooking & Recipes
            r'\b(?:recipe|recipes|baking|bake|brownie|brownies|pasta|cake|pizza|burger|cook|cooking|dish|dishes|cook pasta|make tea|how to cook|ingredients for baking)\b',
            # Geography, General Trivia & News
            r'\b(?:capital of|population of|weather in|weather today|temperature today|distance between|currency of|prime minister of|president of|breaking news|who is [a-z0-9\s]+(?:actor|singer|player|politician))\b',
            # Unrelated Science & Technology
            r'\b(?:quantum mechanics|wave function|hydrogen atom|mitochondria|relativity|black hole|supernova|calculus|derivative of|solve this equation)\b',
            # Programming & IT (Unrelated to Patent Software Eligibility)
            r'\b(?:python program|write a python|sort an array|write code to|learn java|javascript tutorial|html css code|react component|build a website|python mein rag|how to code)\b',
            # Finance, Crypto & Investing (Unrelated to IP)
            r'\b(?:stock price|bitcoin|cryptocurrency|crypto|ethereum|forex|invest in stocks|gold price|mutual fund|credit card|trading strategy)\b',
            # Travel & Lifestyle
            r'\b(?:travel itinerary|hotels in|flights to|tourist places|sightseeing|recommend a laptop|buy a phone|relationship advice|love advice|workout routine|diet plan|horoscope|astrology)\b',
            # Humor & Chatbot Banter
            r'\b(?:tell me a joke|tell a joke|write a poem|sing a song|how are you|are you human|what is your name)\b',
            # Hindi / Hinglish Out-of-Scope phrases
            r'\b(?:kaun hai|kaisa mausam|chutkula sunao|kavita likho|cricket match kisne jita|khana kaise banaye|kaise banaye|python mein)\b',
            # Unrelated Foreign Laws
            r'\b(?:brazil tax|brazilian corporate|germany kstg|german tax|california clean vehicle|faa airspace|australia drone|us tax code|irs tax)\b'
        ]
        self.compiled_out_of_scope = [re.compile(p, re.IGNORECASE) for p in self.out_of_scope_patterns]

        # Multilingual / Hindi expansion dictionary
        self.hindi_term_map = {
            "अश्वगंधा": "withania somnifera ashwagandha",
            "पारंपरिक ज्ञान": "traditional knowledge tk tkdl",
            "आयुर्वेद आहार": "ayurveda aahara food safety fssai",
            "राष्ट्रीय जैव विविधता प्राधिकरण": "national biodiversity authority nba section 6",
            "जैव विविधता": "biological diversity biological resources",
            "पेटेंट": "patent patentability section 3",
            "अनुमति": "approval permission nba",
            "नियम": "rules regulation",
            "दवा": "drugs medicine ayurvedic",
            "ग्रंथ": "authoritative texts classical texts"
        }

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Executes two-stage query understanding and domain scope verification."""
        t0 = time.perf_counter()
        query = state.get("query", "").strip()

        # 1. Detect Language
        language = self._detect_language(query)

        # 2. Extract Exact Statutory Identifiers
        parsed_identifier = parse_legal_identifier(query)
        exact_identifiers = self._extract_identifiers(query)
        if parsed_identifier.get("type"):
            exact_identifiers = [f"{parsed_identifier['type'].title()} {parsed_identifier['value']}"]
        # 3. Detect Jurisdiction
        jurisdiction = self._detect_jurisdiction(query)
        # 4. Stage 1: Domain Scope Verification (IN_SCOPE vs OUT_OF_SCOPE)
        scope_status, scope_confidence, scope_reason, detected_domains = self._verify_scope(
            query=query,
            exact_identifiers=exact_identifiers
        )

        # 5. Stage 2: In-Scope Classification (or OUT_OF_SCOPE)
        if scope_status == "OUT_OF_SCOPE":
            query_type = "OUT_OF_SCOPE"
            query_category = "OUT_OF_SCOPE"
            refusal_reason = "unsupported_general_knowledge"
        else:
            query_type, query_category = self._classify_in_scope_query(
                query=query,
                language=language,
                domains=detected_domains,
                exact_identifiers=exact_identifiers
            )
            refusal_reason = None

        # 6. Expand Query if Multilingual & In-Scope
        expanded_query = None
        if scope_status == "IN_SCOPE":
            expanded_query = self._expand_multilingual_query(query, language)
            if not expanded_query:
                expanded_query = self._expand_domain_query(query, detected_domains)

        latency = round((time.perf_counter() - t0) * 1000, 2)
        node_latencies = dict(state.get("node_latencies_ms", {}))
        node_latencies["query_understanding_ms"] = latency

        trace_entry = {
            "node": "query_understanding",
            "scope_status": scope_status,
            "scope_confidence": scope_confidence,
            "query_type": query_type,
            "query_category": query_category,
            "language": language,
            "jurisdiction": jurisdiction,
            "domains": detected_domains,
            "exact_identifiers": exact_identifiers,
            "identified_document": parsed_identifier.get("canonical_title"),
            "identified_provision": parsed_identifier.get("value"),
            "latency_ms": latency
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        # Add parsed_identifier (including document hint) to returned state
        return {
            "parsed_identifier": parsed_identifier,
            "original_query": query,
            "expanded_query": expanded_query,
            "language": language,
            "scope_status": scope_status,
            "scope_confidence": scope_confidence,
            "scope_reason": scope_reason,
            "query_type": query_type,
            "query_category": query_category,
            "jurisdiction": jurisdiction,
            "domains": detected_domains,
            "domain": detected_domains[0].upper() if detected_domains else None,
            "exact_identifiers": exact_identifiers,
            "refusal_reason": refusal_reason,
            "retrieval_called": False,
            "reranking_called": False,
            "generation_called": False,
            "retrieval_attempt": 0,
            "generation_attempt": 0,
            "node_latencies_ms": node_latencies,
            "execution_trace": trace
        }

    def _verify_scope(
        self,
        query: str,
        exact_identifiers: List[str]
    ) -> Tuple[str, float, str, List[str]]:
        """
        Determines whether the query is strictly within the supported legal & AYUSH domains.
        Returns: (scope_status, scope_confidence, scope_reason, detected_domains)
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # Step A: Detect in-scope domain keywords first
        detected_domains = self._detect_domain_keywords(q_lower)

        # Step B: Check Explicit Out-of-Scope Negative Patterns
        matched_negative = False
        for pattern in self.compiled_out_of_scope:
            if pattern.search(q_lower):
                matched_negative = True
                break

        if matched_negative:
            # If a negative pattern matched (e.g. "cricket", "virat kohli", "python program"):
            # If there are NO supported legal/AYUSH domains or exact statutory identifiers, it is strictly OUT_OF_SCOPE
            if not detected_domains and not exact_identifiers:
                return (
                    "OUT_OF_SCOPE",
                    0.99,
                    "Query matches an explicit out-of-scope domain (e.g. sports, entertainment, cooking, general science, finance, trivia).",
                    []
                )
            
            # If the query is mixed/ambiguous (e.g., "How can I patent a cricket-related Ayurvedic product?"):
            # The presence of strong IP/Ayurveda keywords makes the legal aspect in-scope
            if detected_domains or exact_identifiers:
                confidence = 0.88
                return (
                    "IN_SCOPE",
                    confidence,
                    f"Mixed query with IP/AYUSH domain elements: {', '.join(detected_domains)}.",
                    detected_domains if detected_domains else ["patents"]
                )

        # Step C: Check Exact Statutory Identifiers
        # If an exact legal section/rule/article is present, verify if associated with supported law
        if exact_identifiers:
            # e.g. "Section 3(p)", "Section 3(d)", "Section 6", "Rule 43bis", "Section 53"
            domains = detected_domains if detected_domains else ["patents"]
            return (
                "IN_SCOPE",
                0.97,
                f"Query contains exact statutory provision: {', '.join(exact_identifiers)}.",
                domains
            )

        # Step D: Supported domain keywords found
        if len(detected_domains) > 0:
            confidence = min(0.97, 0.85 + 0.04 * len(detected_domains))
            return (
                "IN_SCOPE",
                confidence,
                f"Query matched supported legal/AYUSH domains: {', '.join(detected_domains)}.",
                detected_domains
            )

        # Step E: No supported domain keywords found -> Hard Out-of-Scope Gate
        return (
            "OUT_OF_SCOPE",
            0.96,
            "Query does not mention any recognized Indian IP, AYUSH, Traditional Knowledge, Biodiversity, or International IP concepts.",
            []
        )

    def _detect_domain_keywords(self, q_lower: str) -> List[str]:
        """Finds all matching legal/AYUSH domains from text."""
        domains: List[str] = []

        # Check Patent Law & Indian IP
        if any(kw in q_lower for kw in self.patent_keywords):
            domains.append("patents")

        # Check Ayurveda / AYUSH
        if any(kw in q_lower for kw in self.ayurveda_keywords):
            domains.append("ayurveda")

        # Check Traditional Knowledge
        if any(kw in q_lower for kw in self.tk_keywords):
            domains.append("traditional_knowledge")

        # Check Biological Resources
        if any(kw in q_lower for kw in self.biodiversity_keywords):
            domains.append("biological_resources")

        # Check International IP
        if any(kw in q_lower for kw in self.international_keywords):
            domains.append("international_ip")

        return domains

    def _classify_in_scope_query(
        self,
        query: str,
        language: str,
        domains: List[str],
        exact_identifiers: List[str]
    ) -> Tuple[str, str]:
        """Classifies an in-scope query into standard pipeline routing types."""
        q_lower = query.lower()

        # A current fee is time-sensitive and needs an official fee schedule,
        # rules, or notification—not a neighbouring substantive provision.
        if "fee" in q_lower and any(term in q_lower for term in ("current", "exact", "registration", "application", "renewal")):
            return "CURRENT_FEE_LOOKUP", "TRADEMARK" if "trademark" in q_lower or "trade mark" in q_lower else "REGULATORY"

        # Exact Lookup
        if exact_identifiers or any(w in q_lower for w in ["section 3(p)", "section 3(d)", "section 6", "rule 43bis", "section 53"]):
            return "EXACT_LOOKUP", "PATENT" if "patents" in domains else (domains[0].upper() if domains else "PATENT")

        # Cross-Domain Multi-Statute
        if len(domains) >= 3 or (
            ("patents" in domains and "ayurveda" in domains and "traditional_knowledge" in domains) or
            ("patents" in domains and "biological_resources" in domains)
        ):
            return "CROSS_DOMAIN", "CROSS_DOMAIN"

        # Multilingual / Code-mixed
        if language == "Hindi":
            return "MULTILINGUAL", domains[0].upper() if domains else "GENERAL"
        if language == "Hinglish / Code-Mixed":
            return "CODE_MIXED", domains[0].upper() if domains else "GENERAL"

        # Ayurveda / AYUSH
        if "ayurveda" in domains:
            return "AYURVEDA_IP", "AYURVEDA"

        # Traditional Knowledge
        if "traditional_knowledge" in domains:
            return "FACTUAL", "TRADITIONAL_KNOWLEDGE"

        # Biological Resources
        if "biological_resources" in domains:
            return "FACTUAL", "BIODIVERSITY"

        # International IP
        if "international_ip" in domains:
            return "FACTUAL", "INTERNATIONAL_IP"

        # Explanatory vs Factual
        if any(q_lower.startswith(w) for w in ["why", "explain", "how does", "compare", "describe"]):
            return "EXPLANATORY", "PATENT"

        return "FACTUAL", "PATENT"

    def _detect_language(self, query: str) -> str:
        """Detects whether query is English, Hindi (Devanagari), or Hinglish (Code-Mixed)."""
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', query))
        total_chars = len(re.findall(r'[a-zA-Z0-9_\u0900-\u097F]', query))
        
        if total_chars == 0:
            return "English"
        
        devanagari_ratio = devanagari_chars / total_chars
        if devanagari_ratio > 0.35:
            return "Hindi"

        # Check for Hinglish / Code-Mixed keywords
        hinglish_words = {
            "kya", "hai", "hain", "mein", "par", "se", "ko", "ke", "liye", "hote", "hota",
            "kar", "sakta", "sakte", "zaroori", "shartein", "karein", "batayein", "kaise", "kyu", "kyun", "mil", "kaun"
        }
        tokens = set(re.findall(r'\b[a-zA-Z]+\b', query.lower()))
        if len(tokens.intersection(hinglish_words)) >= 1 and (len(tokens.intersection(hinglish_words)) >= 2 or any(w in tokens for w in ["kya", "kaun", "kaise", "sakta", "sakte", "hai"])):
            return "Hinglish / Code-Mixed"

        return "English"

    def _extract_identifiers(self, query: str) -> List[str]:
        """Extracts exact sections, rules, articles, and patent numbers."""
        ids = []
        for match in self.section_pattern.finditer(query):
            ids.append(f"Section {match.group(1)}")
        for match in self.rule_pattern.finditer(query):
            ids.append(f"Rule {match.group(1)}")
        for match in self.article_pattern.finditer(query):
            ids.append(f"Article {match.group(1)}")
        for match in self.patent_num_pattern.finditer(query):
            ids.append(f"Patent No. {match.group(1)}")
        return ids

    def _detect_jurisdiction(self, query: str) -> str:
        """Infers legal jurisdiction from keywords."""
        q_lower = query.lower()
        if "pct" in q_lower or "wipo" in q_lower:
            return "WIPO/PCT"
        if "epo" in q_lower or "european patent" in q_lower:
            return "EPO"
        if any(w in q_lower for w in ["india", "indian", "nba", "fssai", "ayush", "patents act", "section 3"]):
            return "India"
        return "India"

    def _expand_multilingual_query(self, query: str, language: str) -> Optional[str]:
        """Expands Hindi/Hinglish query with relevant English legal terminology."""
        if language not in ("Hindi", "Hinglish / Code-Mixed"):
            return None

        expanded = query
        for k, v in self.hindi_term_map.items():
            if k in query:
                expanded += f" ({v})"
        return expanded

    def _expand_domain_query(self, query: str, domains: List[str]) -> Optional[str]:
        """Adds deterministic retrieval vocabulary for narrow AYUSH/IP questions."""
        q_lower = query.lower()
        if "ayurveda" not in domains and "ayush" not in q_lower:
            return None
        if not any(term in q_lower for term in ("patent", "patentability", "invention", "formulation", "pharmacopoeia")):
            return None
        return (
            f"{query} AYUSH patent examination patentability invention "
            "Patents Act Section 3(p) Section 3(e) prior art novelty inventive step traditional knowledge "
            "Ayurvedic formulation pharmacopoeia"
        )
