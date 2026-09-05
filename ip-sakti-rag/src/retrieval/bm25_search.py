"""BM25 Search Index with Legal, Botanical, and Patent Citation Tokenization."""
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import BM25_DIR, BM25_INDEX_PATH, DEFAULT_TOP_K_BM25

logger = logging.getLogger(__name__)

# Basic English stopwords (excluding legal structural words like 'rule', 'section', 'article', 'no', 'patent')
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize_legal_technical(text: str) -> List[str]:
    """
    Tokenizes text while preserving and augmenting legal citations,
    patent numbers, section/article formats, and botanical scientific names.
    """
    if not text:
        return []

    tokens: List[str] = []
    text_lower = text.lower()

    # 1. Extract and preserve exact Section citations: e.g. "Section 3(p)", "3(p)", "Sec. 3(k)"
    section_matches = re.findall(r'(?:section|sec\.?)\s*(\d+[a-z]?(?:\([a-z0-9]+\))*)', text_lower)
    for sm in section_matches:
        tokens.append(f"section_{sm}")
        clean_sm = sm.replace("(", "").replace(")", "")
        tokens.append(f"section_{clean_sm}")
        tokens.append(f"sec_{sm}")
        tokens.append(sm)
        tokens.append(clean_sm)

    # Isolated section-style tokens like "3(p)", "3(k)", "3(d)", "3(e)"
    sec_sub_matches = re.findall(r'\b(\d+[a-z]?\([a-z0-9]+\))\b', text_lower)
    for ssm in sec_sub_matches:
        tokens.append(ssm)
        clean_ssm = ssm.replace("(", "").replace(")", "")
        tokens.append(clean_ssm)
        tokens.append(f"section_{clean_ssm}")

    # Isolated subclauses like "(p)", "(k)", "(d)", "(e)", "(j)"
    clause_matches = re.findall(r'\(([a-z0-9]{1,3})\)', text_lower)
    for cm in clause_matches:
        tokens.append(f"clause_{cm}")
        tokens.append(f"({cm})")
        tokens.append(cm)

    # 2. Extract Article citations: e.g. "Article 3", "Art. 27"
    article_matches = re.findall(r'(?:article|art\.?)\s*(\d+[a-z]*)', text_lower)
    for am in article_matches:
        tokens.append(f"article_{am}")
        tokens.append(f"art_{am}")
        tokens.append(am)
        tokens.append(f"article {am}")

    # 3. Extract Rule citations: e.g. "Rule 43bis", "PCT Rule 43bis", "Rule 43"
    rule_matches = re.findall(r'(?:pct\s+)?(?:rule|r\.)\s*(\d+[a-z]*)', text_lower)
    for rm in rule_matches:
        tokens.append(f"rule_{rm}")
        tokens.append(f"pct_rule_{rm}")
        tokens.append(rm)

    # 4. Extract Patent numbers: e.g. "Patent No. 429737", "IN 429737", "429737"
    patent_matches = re.findall(r'(?:patent\s*(?:no\.?|number)?\s*|patent\s+)(\d{5,8})', text_lower)
    for pm in patent_matches:
        tokens.append(f"patent_{pm}")
        tokens.append(f"patent_no_{pm}")
        tokens.append(pm)

    # Isolated multi-digit patent numbers
    digits_matches = re.findall(r'\b(\d{6,8})\b', text_lower)
    for dm in digits_matches:
        tokens.append(f"patent_{dm}")
        tokens.append(dm)

    # 5. Word tokenization
    cleaned = re.sub(r'[^a-z0-9\-_]', ' ', text_lower)
    words = cleaned.split()

    filtered_words: List[str] = []
    for word in words:
        w_strip = word.strip("-_")
        if len(w_strip) > 1 and w_strip not in STOPWORDS:
            filtered_words.append(w_strip)
            tokens.append(w_strip)
        elif w_strip.isdigit():
            filtered_words.append(w_strip)
            tokens.append(w_strip)

    # 6. Bi-grams for phrases and botanical names (e.g. withania_somnifera, curcuma_longa)
    for i in range(len(filtered_words) - 1):
        bigram = f"{filtered_words[i]}_{filtered_words[i+1]}"
        tokens.append(bigram)

    # 7. Key legal & botanical multi-word terms
    special_phrases = [
        "withania somnifera", "curcuma longa", "azadirachta indica", "ocimum sanctum",
        "traditional knowledge", "genetic resources", "biological resources",
        "biological diversity", "ayurveda aahara", "food safety", "prior art",
        "person skilled in the art", "traditional cultural expressions",
        "mandatory disclosure", "international phase", "preliminary examination"
    ]
    for phrase in special_phrases:
        if phrase in text_lower:
            tokens.append(phrase.replace(" ", "_"))
            tokens.append(phrase)

    return tokens



def create_searchable_document(chunk: Dict[str, Any]) -> str:
    """Builds an enriched document text incorporating all structural metadata."""
    parts: List[str] = []

    # 1. Base text
    text = chunk.get("text", "")
    parts.append(text)

    # 2. Metadata attributes
    meta = chunk.get("metadata", {})
    if meta:
        doc_name = meta.get("document", "")
        if doc_name:
            parts.append(f"Document: {doc_name}")
        heading = meta.get("heading", "")
        if heading:
            parts.append(f"Heading: {heading}")
        subheading = meta.get("subheading", "")
        if subheading:
            parts.append(f"Subheading: {subheading}")
        section = meta.get("section", "")
        if section:
            parts.append(f"Section {section}")
            parts.append(f"Section_{section}")
        article = meta.get("article", "")
        if article:
            parts.append(f"Article {article}")
            parts.append(f"Article_{article}")
        rule = meta.get("rule", "")
        if rule:
            parts.append(f"Rule {rule}")
            parts.append(f"Rule_{rule}")
        guideline = meta.get("guideline", "")
        if guideline:
            parts.append(f"Guideline {guideline}")
        patent_num = meta.get("patent_number", "")
        if patent_num:
            parts.append(f"Patent No. {patent_num}")
            parts.append(f"Patent_{patent_num}")
        title = meta.get("title", "")
        if title:
            parts.append(f"Title: {title}")
        category = meta.get("category", "")
        if category:
            parts.append(category)
        jurisdiction = meta.get("jurisdiction", "")
        if jurisdiction:
            parts.append(jurisdiction)
        domains = meta.get("domain", [])
        if isinstance(domains, list):
            parts.extend(domains)

    return " \n".join(parts)


class BM25SearchEngine:
    """BM25 Okapi search engine with legal tokenization and metadata search."""

    def __init__(
        self,
        index_path: Union[str, Path] = BM25_INDEX_PATH,
    ):
        self.index_path = Path(index_path)
        self.corpus_chunk_ids: List[str] = []
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

        if self.index_path.exists():
            self.load()

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        save: bool = True,
    ) -> None:
        """Tokenizes chunks and constructs the BM25 index."""
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        self.corpus_chunks = chunks
        self.corpus_chunk_ids = [c["chunk_id"] for c in chunks]

        tokenized_corpus: List[List[str]] = []
        for chunk in chunks:
            searchable_text = create_searchable_document(chunk)
            tokens = tokenize_legal_technical(searchable_text)
            tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(tokenized_corpus)} documents.")

        if save:
            self.save()

    def save(self) -> None:
        """Saves index and corpus mapping to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus_chunk_ids": self.corpus_chunk_ids,
            "corpus_chunks": self.corpus_chunks,
            "bm25": self.bm25,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved BM25 index to {self.index_path}")

    def load(self) -> None:
        """Loads BM25 index from disk."""
        logger.info(f"Loading BM25 index from {self.index_path}...")
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self.corpus_chunk_ids = data["corpus_chunk_ids"]
        self.corpus_chunks = data["corpus_chunks"]
        self.bm25 = data["bm25"]
        logger.info(f"Loaded BM25 index with {len(self.corpus_chunk_ids)} documents.")

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_BM25,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches BM25 index with legal tokenization and optional metadata filtering.

        Returns structured candidate list.
        """
        if self.bm25 is None or not self.corpus_chunks:
            raise ValueError("BM25 index is not built or loaded.")

        query_tokens = tokenize_legal_technical(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)

        # Get top indices sorted by score descending
        # Filter scores > 0
        ranked_indices = np.argsort(doc_scores)[::-1]

        candidates: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(doc_scores[idx])
            if score <= 0.0:
                break

            chunk = self.corpus_chunks[idx]
            meta = chunk.get("metadata", {})

            # Optional metadata filter check
            if filters:
                match = True
                for f_key, f_val in filters.items():
                    if f_key == "jurisdiction" and meta.get("jurisdiction") != f_val:
                        match = False
                        break
                    elif f_key == "category" and meta.get("category") != f_val:
                        match = False
                        break
                    elif f_key == "document_type" and meta.get("document_type") != f_val:
                        match = False
                        break
                    elif f_key == "document_id" and meta.get("document_id") != f_val:
                        match = False
                        break
                    elif f_key == "domain":
                        chunk_domains = meta.get("domain", [])
                        if isinstance(f_val, list):
                            if not any(d in chunk_domains for d in f_val):
                                match = False
                                break
                        elif f_val not in chunk_domains:
                            match = False
                            break
                if not match:
                    continue

            candidate = {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": meta.get("document_id"),
                "document": meta.get("document"),
                "page": meta.get("page"),
                "section": meta.get("section"),
                "article": meta.get("article"),
                "rule": meta.get("rule"),
                "guideline": meta.get("guideline"),
                "chapter": meta.get("chapter"),
                "heading": meta.get("heading"),
                "jurisdiction": meta.get("jurisdiction"),
                "category": meta.get("category"),
                "domain": meta.get("domain", []),
                "score": score,
                "retrieval_method": "bm25",
                "text": chunk.get("text", ""),
                "metadata": meta,
            }
            candidates.append(candidate)
            if len(candidates) >= top_k:
                break

        return candidates
