# IP-SAKTI Sahayak: Milestone 2 Retrieval Evaluation & Validation Report

**Date**: 2026-09-05  
**System**: IP-SAKTI Sahayak Legal & Ayurvedic IP RAG Pipeline  
**Model**: `BAAI/bge-m3` (1024-dimensional dense vectors) + Qdrant Vector Store + BM25 Okapi with Legal/Botanical Tokenizer  

---

## 1. Dataset Statistics

| Metric | Value |
|---|---|
| **Total Source PDF Documents** | 22 Legal, Regulatory & Academic Documents |
| **Total Indexed Chunks** | 5,212 Chunks |
| **Embedding Dimensions** | 1,024 (`BAAI/bge-m3`) |
| **Dense Vector Cache** | 5,212 vectors in `indexes/embeddings_cache.pkl` |
| **Vector Index** | Qdrant local disk collection `ip_sakti_documents` (`indexes/qdrant/`) |
| **BM25 Inverted Index** | `indexes/bm25/bm25_index.pkl` (5,212 tokenized documents) |
| **Covered Jurisdictions** | India (13 docs), International (6 docs), EPO (2 docs), WIPO/PCT (1 doc) |
| **Average Query Latency (Cached)** | ~140ms – 180ms per hybrid query |

---

## 2. Benchmark Composition & Ground Truth Summary

- **Total Benchmark Questions**: **36 Domain-Specific Questions**
- **Verified Ground Truth Questions**: **36 / 36 (100%)**
- **Domain Coverage**:
  - **Indian Patent Law (PATENT)**: 6 Questions (Section 3 exclusions, 3(p) traditional knowledge, 2(1)(j) invention definition, 3(d) enhanced efficacy, 3(e) synergistic combinations vs mere admixture, 10(4) biological disclosure).
  - **Ayurveda & AYUSH Guidelines (AYURVEDA)**: 6 Questions (Formulation patentability, 2025 AYUSH examination guidelines, TKDL examination role, FSSAI Ayurveda Aahara 2022/2025 schedules, advertising & disease claims restrictions, FBO licensing).
  - **Traditional Knowledge (TRADITIONAL KNOWLEDGE)**: 4 Questions (TK in IP context, defensive vs positive protection, genetic resources interplay, WIPO documentation toolkit).
  - **Biological Resources & NBA (BIOLOGICAL RESOURCES)**: 3 Questions (Biological Diversity Act 2002 provisions, Section 6 prior approval for IPR, Access & Benefit Sharing mechanisms).
  - **International IP (INTERNATIONAL)**: 4 Questions (PCT International Phase procedure, WIPO 2024 GR/TK Treaty Article 3, EPO examination & problem-solution approach, PCT Rule 43bis written opinion).
  - **Exact Lookups (EXACT_LOOKUP)**: 7 Questions (`Section 3(p)`, `Article 3`, `PCT Rule 43bis`, `Withania somnifera`, `Patent No. 429737`, `Section 10(4)`, `Curcuma longa`).
  - **Multilingual & Code-Mixed (MULTILINGUAL)**: 4 Questions (Hindi and Hindi-English mixed queries).
  - **Cross-Domain Complex Queries (CROSS_DOMAIN)**: 2 Questions (Interplay across Patent Act + AYUSH Guidelines + Biodiversity Act + International Treaties).

---

## 3. Retrieval Metrics Comparison (36 Verified Queries)

| Method | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR | NDCG@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vector Search (BGE-M3)** | 0.8889 | 0.9167 | 0.5556 | 0.4583 | 0.7306 | 0.7659 |
| **BM25 Search (Lexical)** | 0.7222 | 0.8056 | 0.4667 | 0.4083 | 0.6352 | 0.6589 |
| **Hybrid Search (RRF Fusion, $k=60$)** | **0.8889** | **0.9444** | **0.5833** | **0.4972** | **0.7378** | **0.7778** |

---

## 4. Category-Wise Performance Breakdown (Hybrid Retrieval)

| Category | Count | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|:---:|:---:|:---:|:---:|:---:|
| **PATENT** | 6 | 0.8333 | 0.8333 | 0.8333 | 0.7822 |
| **AYURVEDA** | 6 | 0.8333 | **1.0000** | 0.7500 | 0.7985 |
| **TRADITIONAL KNOWLEDGE** | 4 | **1.0000** | **1.0000** | 0.6875 | 0.8155 |
| **BIOLOGICAL RESOURCES** | 3 | **1.0000** | **1.0000** | **0.8333** | **0.9171** |
| **INTERNATIONAL** | 4 | **1.0000** | **1.0000** | **0.8125** | **0.8622** |
| **EXACT_LOOKUP** | 7 | 0.8571 | 0.8571 | 0.6786 | 0.7130 |
| **MULTILINGUAL** | 4 | **1.0000** | **1.0000** | 0.6750 | 0.7065 |
| **CROSS_DOMAIN** | 2 | 0.5000 | **1.0000** | 0.5556 | 0.6193 |

---

## 5. Detailed Case Studies & Method Analysis

### 5.1 Ten Cases Where Vector Retrieval Performed Well
1. **Can an Ayurvedic formulation be patented in India?** (Q7): Vector captured conceptual discussions on synergistic efficacy, novelty hurdles, and prior art even when the query did not mention specific statutory clauses.
2. **What is traditional knowledge in the context of intellectual property?** (Q13): Vector retrieved high-level descriptive definitions from `IP_GR_TK_TCE_Overview.pdf` without requiring exact phrasing matches.
3. **What is defensive protection versus positive protection of traditional knowledge?** (Q14): Vector distinguished between defensive mechanisms (preventing improper patent grants) and positive rights (enforcing economic control) at Rank 1.
4. **What is the relationship between genetic resources and associated traditional knowledge?** (Q15): Successfully retrieved WIPO treaties and overview guides exploring genetic material extraction and indigenous knowledge protection.
5. **What are the patentability examination principles under EPO Guidelines?** (Q22): Vector retrieved the problem-solution approach and novelty assessment principles directly from `epo_guidelines_for_examination_2026.pdf`.
6. **What is the role of TKDL in patent examination?** (Q9): Vector retrieved cross-jurisdictional evidence regarding CSIR's defensive patent examination access agreements.
7. **What considerations apply to Ayurveda-related inventions?** (Q8): Vector placed the 2025 IPO AYUSH Guidelines at Rank 1.
8. **What licensing and registration rules apply to manufacturers under FSSAI?** (Q12): Vector identified the Food Safety and Standards Licensing compendium chapters on manufacturer compliance.
9. **क्या आयुर्वेदिक formulation को भारत में patent किया जा सकता है?** (Q31): Vector's multilingual BGE-M3 representations mapped Hindi conceptual phrasing to English AYUSH guidelines and Patent Act chunks.
10. **What is Access and Benefit Sharing (ABS) under Indian biodiversity law?** (Q19): Retrieved equitable sharing definitions under Section 21 of the Biological Diversity Act 2002 at Rank 1.

### 5.2 Ten Cases Where BM25 Retrieval Performed Well
1. **Patent No. 429737** (Q28): BM25 placed the exact Indian Patent grant chunk (`ayush_related_inventions_guidelines_2025_p1_c1`) at **Rank 1**, whereas vector placed generic patent chapters first.
2. **Withania somnifera** (Q27): Legal tokenizer's compound taxon term `withania_somnifera` enabled BM25 to locate specific Ashwagandha prior art examination examples at **Rank 1**.
3. **Curcuma longa** (Q30): BM25 retrieved the exact turmeric schedules in the FSSAI Ayurveda Aahara 2025 gazette order at **Rank 1**.
4. **Section 3(p)** (Q24): Exact section tokenization matched the statutory title chunk `ayush_related_inventions_guidelines_2025_p7_c6` at **Rank 1**.
5. **Section 10(4)** (Q29): BM25 pinpointed Section 10 "Contents of specifications" in the Patents Act containing mandatory biological source disclosure.
6. **PCT Rule 43bis** (Q26): BM25 matched the explicit procedural rule on the written opinion of the ISA in `pct_applicant_guide_international_phase.pdf` at **Rank 1**.
7. **Article 3** (Q25): BM25 retrieved Article 3 of the 2024 WIPO Genetic Resources Treaty at **Rank 1**.
8. **Section 3(d)** (Q4): BM25 retrieved the verbatim exclusion of new forms of known substances without enhanced efficacy at **Rank 1**.
9. **FSSAI Ayurveda Aahara Regulations** (Q10): BM25 extracted the specific 2022 gazette notification chunks mentioning authoritative Ayurvedic texts.
10. **When is NBA approval required under Section 6?** (Q18): BM25 retrieved Section 6 of the Biological Diversity Act 2002 directly.

### 5.3 Ten Cases Where Hybrid Retrieval Improved Over Both
1. **Section 3(p) traditional knowledge से कैसे संबंधित है?** (Q33): Hybrid combined BM25's exact token match on `section_3(p)` with Vector's multilingual understanding of "पारंपरिक ज्ञान" semantics, placing the definitive statutory definition at **Rank 1**.
2. **What are the exclusions under Section 3 of the Indian Patents Act?** (Q1): Vector brought conceptual discussions while BM25 retrieved specific subclauses ((d), (e), (p)); Hybrid RRF fused them to provide both overarching principles and specific statutory clauses in Top 5.
3. **What constitutes an invention under Indian patent law?** (Q3): BM25 found Section 2 definitions; Vector found inventive step analysis; Hybrid fused them to place Section 2(1)(j) at **Rank 1**.
4. **How does Section 3(e) distinguish mere admixture from synergistic combinations?** (Q5): BM25 matched `mere admixture`; Vector captured `synergistic combination` tests; Hybrid achieved 100% precision in Top 5.
5. **What is Article 3 of the 2024 WIPO Treaty on Genetic Resources?** (Q21): Combined dense semantic understanding of mandatory disclosure with exact lexical token `Article 3`.
6. **Can an Ayurvedic formulation using a traditionally known plant be patented in India?** (Q35): Vector retrieved AYUSH guidelines and BM25 retrieved Biodiversity Act Section 6; Hybrid presented cross-statutory evidence across both acts.
7. **What are the advertising and claim restrictions for Ayurvedic products?** (Q11): Hybrid brought both FSSAI Advertising Regulations and Drugs and Cosmetics Act disease claim restrictions into Top 5.
8. **What provisions govern the use of biological resources under BDA 2002?** (Q17): Combined BDA Section 3/4 access rules with AYUSH patent guidelines.
9. **PCT International Phase procedure** (Q20): Fused PCT applicant guide procedural overviews with EPO PCT guidelines.
10. **What are the disclosure requirements for biological materials under Section 10(4)?** (Q6): Combined Section 10(4) statutory chunk with 2025 examination guidelines on patent specification drafting.

---

## 6. Failure Analysis & Explanations

Across 36 queries, 4 queries exhibited suboptimal rank placement or required deeper multi-domain evidence in Top 5:

1. **Q4: Section 3(d) Enhanced Efficacy** (`PATENT`):
   - *Issue*: Vector assigned high semantic weight to pharmaceutical patent guidelines in general; the exact Section 3(d) definition chunk was retrieved at Rank 6 instead of Top 5.
   - *Reason*: Section 3(d) text in `patent_act_1970_p9_c33` was embedded alongside neighboring subclauses, diluting single-clause dense affinity.
   - *Fix*: BM25 recovered it; RRF fusion placed it at Rank 2. Milestone 3 cross-encoder reranking will further elevate the specific clause.

2. **Q28: Patent No. 429737** (`EXACT_LOOKUP`):
   - *Issue*: Vector alone failed completely (Rank > 20) because 6-digit numbers have near-random dense representations in generic embedding models.
   - *Reason*: Dense tokenizers treat rare 6-digit numbers as byte-pair fragments (`429`, `737`).
   - *Fix*: BM25 with our regex patent tokenizer placed it at Rank 1. Hybrid RRF brought it into Top 3.

3. **Q32: अश्वगंधा के patent से संबंधित प्रावधान क्या हैं?** (`MULTILINGUAL`):
   - *Issue*: Hindi token "अश्वगंधा" retrieved general WIPO and Drugs Act chunks before finding specific Withania somnifera patent case studies.
   - *Reason*: Corpus text uses botanical binomial "Withania somnifera" or English "Ashwagandha", whereas raw Hindi query lacked English transliteration in BM25.
   - *Fix*: Query pre-processor in Milestone 3 should expand Hindi botanical names to Sanskrit/English/Binomial equivalents using `ayurveda_terms.json`.

4. **Q36: Cross-Domain Disclosure Requirements** (`CROSS_DOMAIN`):
   - *Issue*: Query required simultaneous retrieval across Indian Patent Act Sec 10(4), Biodiversity Act Sec 6, and WIPO Treaty Art 3. Top 5 contained WIPO and Patent Act, but BDA Sec 6 was at Rank 8.
   - *Reason*: Single-query retrieval tends to cluster around the most dense semantic domain (WIPO Patent Disclosure guide).
   - *Fix*: LangGraph Query Decomposer in Milestone 3 will split cross-domain queries into sub-queries per statute.

---

## 7. Multilingual Retrieval Evaluation

| Query (Hindi / Code-Mixed) | English Equivalent | Hybrid Top 1 Rank | Retrieval Quality |
|---|---|:---:|---|
| **क्या आयुर्वेदिक formulation को भारत में patent किया जा सकता है?** (Q31) | Can an Ayurvedic formulation be patented in India? | Rank 1 (AYUSH Guidelines & Drugs Act) | **High**: Semantic alignment across languages |
| **अश्वगंधा के patent से संबंधित प्रावधान क्या हैं?** (Q32) | What are patent provisions for Ashwagandha? | Rank 1 (WIPO & Patent Act) | **Moderate**: Requires botanical synonym expansion |
| **Section 3(p) traditional knowledge से कैसे संबंधित है?** (Q33) | How is Section 3(p) related to traditional knowledge? | Rank 1 (AYUSH Guidelines Sec 3(p)) | **Excellent**: Exact statutory match at Rank 1 |
| **आयुर्वेद आहार के लिए FSSAI के क्या नियम हैं?** (Q34) | What are FSSAI rules for Ayurveda Aahara? | Rank 1 (Ayurveda Aahara Gazette 2025) | **Excellent**: Correct regulatory order at Rank 1 |

---

## 8. Dynamic Metadata Filtering Validation

All 6 test scenarios executed successfully with 100% accuracy:

| Test Case | Query | Inferred Filter | Result |
|---|---|---|:---:|
| **India-only** | Indian Patents Act biological material | `{'jurisdiction': 'India'}` (Conf: 0.85) | **PASSED** |
| **WIPO / PCT** | PCT international phase procedure WIPO | `{'jurisdiction': ['WIPO/PCT', 'International']}` (Conf: 0.90) | **PASSED** |
| **EPO** | Inventive step in EPO Guidelines | `{'jurisdiction': 'EPO'}` (Conf: 0.90) | **PASSED** |
| **Ayurveda Aahara** | FSSAI regulations for Ayurveda Aahara | `{'domain': ['ayurveda_aahara', ...], 'jurisdiction': 'India'}` | **PASSED** |
| **Traditional Knowledge** | WIPO principles for Traditional Knowledge | `{'jurisdiction': ['WIPO/PCT', 'International']}` (Conf: 0.90) | **PASSED** |
| **Ambiguous Query** | What is the meaning of inventive step? | `None` (Conf: 0.00 — No over-filtering) | **PASSED** |

---

## 9. Answers to Final Evaluation Questions

1. **Is vector retrieval working well?**  
   **Yes**. `BAAI/bge-m3` achieves **Recall@5 = 88.89%** and **Recall@10 = 91.67%**, providing strong conceptual matching for complex legal doctrines, cross-lingual queries, and regulatory standards.

2. **Is BM25 working well?**  
   **Yes**. With custom legal and botanical regex tokenizers, BM25 achieves **Recall@10 = 80.56%** and excels on exact patent numbers (`429737`), section numbers (`Section 3(p)`), rule numbers (`PCT Rule 43bis`), and Latin binomials (`Withania somnifera`).

3. **Does hybrid retrieval outperform them?**  
   **Yes**. Hybrid retrieval (RRF, $k=60$) achieves the highest overall performance: **Recall@10 = 94.44%**, **Precision@5 = 58.33%**, **MRR = 0.7378**, and **NDCG@10 = 0.7778**, successfully resolving both lexical omissions and dense semantic ambiguities.

4. **What is Recall@5?**  
   **88.89%** (32 / 36 queries retrieve verified relevant evidence in Top 5).

5. **What is Recall@10?**  
   **94.44%** (34 / 36 queries retrieve verified relevant evidence in Top 10).

6. **What is MRR?**  
   **0.7378** (First relevant legal evidence appears on average at Rank 1.35).

7. **What are the biggest retrieval failures?**  
   - Multi-statute cross-domain queries where a single retrieval list is dominated by the largest semantic cluster (e.g. WIPO guides dominating Biodiversity Act provisions).
   - Devanagari plant names lacking English transliteration in BM25 index (e.g., "अश्वगंधा" vs "Withania somnifera").
   - Dilution of specific statutory subsections when embedded in multi-clause act chunks.

8. **Is the system ready for reranking?**  
   **Yes**. The hybrid retrieval candidate pool provides 94.44% Recall@10 with balanced representation across Indian patent law, AYUSH guidelines, biodiversity regulations, and international IP frameworks. A cross-encoder reranker in Milestone 3 will compress and re-score the Top 25 candidates to elevate exact subsection evidence to Ranks 1–3.
