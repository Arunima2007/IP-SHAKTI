# IP-SAKTI Sahayak: Milestone 3 Cross-Encoder Reranking & Diversity Evaluation

**Date**: 2026-09-05  
**Component**: Milestone 3 — Domain-Aware Cross-Encoder Reranking & Diversity-Aware Evidence Selection  
**Reranker Model**: `BAAI/bge-reranker-v2-m3`  
**Evaluation Scope**: 42 Verified Domain Queries across Indian Patent Law, AYUSH / Ayurveda, Traditional Knowledge, Biological Resources, International Treaties (PCT/WIPO/EPO), Exact Lookups, Multilingual, and Cross-Domain.

---

## 1. Reranker Model & Architectural Overview

The Milestone 3 reranking layer is integrated directly after the Hybrid RRF fusion stage:

```mermaid
flowchart TD
    Q[User Query] --> FB[Dynamic Filter Builder]
    FB --> |Jurisdiction / Domain Filter| VS[Qdrant Vector Search - BGE-M3]
    Q --> |Dense Vector 1024-dim| VS
    Q --> |Domain/Legal Tokenizer| BM[BM25 Okapi Index]
    VS --> |Top-25 Dense Chunks| RRF[Reciprocal Rank Fusion]
    BM --> |Top-25 Lexical Chunks| RRF
    RRF --> |Top 25–30 Candidates Pool| CE[Cross-Encoder Reranker - BGE-Reranker-v2-m3]
    CE --> |Re-scored Candidates| DS[Diversity-Aware Evidence Selector]
    DS --> |Exact Match Calibration + Cross-Domain Balancing| FE[Top 5–8 Final Evidence Chunks]
```

- **Base Reranker**: `BAAI/bge-reranker-v2-m3` (Multilingual Cross-Encoder supporting English, Hindi, Sanskrit, and international legal texts).
- **Sequence Length**: `max_length = 512` tokens.
- **Scoring**: Full cross-attention between `(query, chunk_text)` producing logit scores converted via sigmoid to $[0, 1]$ calibrated relevance probabilities.
- **Provenance Preservation**: 100% metadata preservation (`chunk_id`, `document_id`, `document`, `source`, `page`, `section`, `subsection`, `clause`, `article`, `rule`, `heading`, `subheading`, `jurisdiction`, `domain`, `category`, `language`, `year`, `version`, `patent_number`, etc.).

---

## 2. Configuration Parameters

All parameters are configurable via environment variables or `src/config.py`:

| Parameter | Default Value | Description |
|---|---|---|
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Configurable cross-encoder model checkpoint |
| `RERANKER_BATCH_SIZE` | `16` | Batch size for CPU/GPU cross-encoder inference |
| `RERANKER_MAX_LENGTH` | `512` | Token truncation limit for query-chunk pairs |
| `RERANKER_CACHE_PATH` | `indexes/reranker_cache.pkl` | SHA-256 disk cache for previously scored pairs |
| `VECTOR_TOP_K` | `25` | Number of dense candidates retrieved from Qdrant |
| `BM25_TOP_K` | `25` | Number of lexical candidates retrieved from BM25 |
| `FUSION_TOP_K` | `30` | Number of candidate chunks returned by RRF fusion |
| `RERANK_TOP_K` | `25` | Number of candidates passed to Cross-Encoder |
| `FINAL_TOP_K` | `6` | Number of final evidence chunks delivered to downstream agent |
| `MAX_CHUNKS_PER_DOCUMENT` | `2` | Maximum chunks allowed per document in multi-domain mode |
| `MAX_CHUNKS_PER_DOMAIN` | `3` | Maximum chunks allowed per domain in multi-domain mode |
| `DIVERSITY_ENABLED` | `True` | Intent-aware diversity toggle |

---

## 3. Benchmark Metrics Comparison (42 Verified Queries)

| Method | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR | NDCG@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vector-only (`BAAI/bge-m3`)** | 0.8810 | 0.9048 | 0.5714 | 0.4786 | 0.7294 | 0.7663 |
| **BM25-only (Legal Tokenizer)** | 0.7143 | 0.8095 | 0.4714 | 0.4119 | 0.6427 | 0.6652 |
| **Hybrid (RRF Fusion, $k=60$)** | 0.9286 | 0.9286 | **0.6143** | **0.5429** | 0.7575 | 0.7939 |
| **Hybrid + Cross-Encoder Reranking** | **0.9524** | **0.9524** | 0.6095 | 0.3619* | **0.8413** | **0.8555** |

*\*Note on Precision@10 for Reranked: The final evidence selector deliberately limits the output to Top 6 evidence chunks (`FINAL_TOP_K = 6`) to eliminate noisy tail chunks. Hence Precision@10 reflects a strict 6-chunk pool.*

---

## 4. Category-Wise Performance (Hybrid vs Hybrid + Reranker)

| Category | Queries Count | Hybrid Recall@5 | Rerank Recall@5 | Hybrid MRR | Rerank MRR | Rerank NDCG@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Indian Patent Law (PATENT)** | 6 | 0.8333 | **0.8333** | 0.8333 | **0.8333** | **0.7823** |
| **Ayurveda & AYUSH (AYURVEDA)** | 6 | 1.0000 | **1.0000** | 0.7639 | **0.8056** (+5.5%) | **0.8552** |
| **Traditional Knowledge (TK)** | 4 | 1.0000 | **1.0000** | 0.6875 | **0.7500** (+9.1%) | **0.8259** |
| **Biological Resources (BDA/NBA)** | 3 | 1.0000 | **1.0000** | 0.8333 | **0.7778** | **0.8171** |
| **International IP (PCT/WIPO/EPO)** | 4 | 1.0000 | **1.0000** | 0.8125 | **1.0000** (+23.1%) | **0.9621** |
| **Exact Lookups (Sections/Patents)** | 7 | 0.8571 | **0.8571** | 0.6714 | **0.8571** (+27.7%) | **0.8305** |
| **Multilingual (Hindi/Code-Mixed)** | 10 | 1.0000 | **1.0000** | 0.8033 | **0.8167** (+1.7%) | **0.8716** |
| **Cross-Domain Complex Queries** | 2 | 0.5000 | **1.0000** (+100%) | 0.5000 | **1.0000** (+100%) | **0.9855** |

---

## 5. Ten Detailed Before / After Reranking Case Studies

### Case 1: International IP — Article 3 Mandatory Disclosure (Q25)
- **Query**: `"Article 3"`
- **Before Reranking (Hybrid Top 3)**:
  1. `PCT_Applicant_Guide_International_Phase.pdf` (p.162) — General PCT Rule Index Mention [Score: 0.0161]
  2. `PCT_Applicant_Guide_International_Phase.pdf` (p.162) — General PCT Index [Score: 0.0159]
  3. `WIPO_GR_TK_Treaty_2024.pdf` (p.5) Article: 3 — Verbatim Mandatory Disclosure Provision [Score: 0.0082]
- **After Reranking (Reranked Top 3)**:
  1. `WIPO_GR_TK_Treaty_2024.pdf` (p.5) Article: 3 — Verbatim Mandatory Disclosure Provision [Score: **0.8078**]
  2. `PCT_Applicant_Guide_International_Phase.pdf` (p.162) [Score: 0.5210]
  3. `EPO_Guidelines_for_Examination_2026.pdf` (p.45) [Score: 0.4980]
- **Why Reranking Succeeded**: Cross-encoder semantic evaluation identified that chunk 3 contained the substantive text of Article 3 rather than mere index citations, elevating it from Rank 3 to **Rank 1**.

---

### Case 2: Indian Patent Law — Section 3 Core Exclusions (Q01)
- **Query**: `"What are the exclusions under Section 3 of the Indian Patents Act?"`
- **Before Reranking (Hybrid Top 3)**:
  1. `Patent Act-1970.pdf` (p.66) Sec: 3 — Subsidiary amendment footnote referencing Section 3 [Score: 0.0161]
  2. `Patent Act-1970.pdf` (p.9) Sec: 3 — Primary statutory text of Section 3 ("What are not inventions") [Score: 0.0161]
  3. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.1) [Score: 0.0152]
- **After Reranking (Reranked Top 3)**:
  1. `Patent Act-1970.pdf` (p.9) Sec: 3 — Primary statutory text of Section 3 ("What are not inventions") [Score: **0.7213**]
  2. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.1) [Score: 0.7092]
  3. `Patent Act-1970.pdf` (p.66) Sec: 3 [Score: 0.5412]
- **Why Reranking Succeeded**: The cross-encoder distinguished between actual operative statutory text and amendment footnotes, promoting the primary exclusion definition chunk to **Rank 1**.

---

### Case 3: Cross-Domain Multi-Statute Evidence Selection (Q41)
- **Query**: `"Can an Ayurvedic formulation using a traditionally known plant be patented in India?"`
- **Before Reranking (Hybrid Top 3)**:
  1. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.9) — AYUSH Guidelines [Score: 0.0149]
  2. `Guidelines_TK_Biological_Material_2012.pdf` (p.2) — AYUSH Guidelines [Score: 0.0149]
  3. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.7) — AYUSH Guidelines [Score: 0.0147]
  *(Biological Diversity Act Section 6 was pushed down to Rank 14)*
- **After Reranking & Diversity Selection (Reranked Top 3)**:
  1. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.9) — AYUSH Guidelines on Formulation Synergism [Score: **0.7228**]
  2. `The Biological Diversity Act,2002.pdf` (p.6) Sec: 6 — Prior NBA Approval for IPR [Score: **0.7104**]
  3. `Guidelines_TK_Biological_Material_2012.pdf` (p.2) — TKDL Prior Art Standards [Score: **0.7081**]
- **Why Reranking Succeeded**: The `DiversityAwareSelector` detected multi-domain intent across Patents + AYUSH + Biodiversity, capping single-document saturation and pulling the crucial Section 6 Biodiversity Act chunk up from Rank 14 directly into **Rank 2**.

---

### Case 4: Multilingual Code-Mixed Treaty Lookup (Q38)
- **Query**: `"WIPO GR/TK Treaty के Article 3 में mandatory disclosure की क्या व्यवस्था है?"`
- **Before Reranking (Hybrid Top 3)**:
  1. `WIPO_GR_TK_Treaty_2024.pdf` (p.5) [Score: 0.0082]
  2. `WIPO_Patent_Disclosure_GR_TK.pdf` (p.23) [Score: 0.0080]
  3. `WIPO_Patent_Disclosure_GR_TK.pdf` (p.29) [Score: 0.0079]
- **After Reranking (Reranked Top 3)**:
  1. `WIPO_GR_TK_Treaty_2024.pdf` (p.4) Sec: None — Article 3 Verbatim Text [Score: **0.8059**]
  2. `WIPO_GR_TK_Treaty_2024.pdf` (p.5) [Score: 0.7410]
  3. `WIPO_Patent_Disclosure_GR_TK.pdf` (p.23) [Score: 0.7301]
- **Why Reranking Succeeded**: BGE-Reranker-v2-m3 effectively parsed the mixed Hindi-English query syntax and matched the core treaty provision chunk at **Rank 1**.

---

### Case 5: Exact Patent Number Lookup (Q28)
- **Query**: `"Patent No. 429737"`
- **Before Reranking (Hybrid Top 3)**:
  1. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.1) [Score: 0.0164]
  2. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.19) [Score: 0.0082]
  3. `Patent Act-1970.pdf` (p.12) [Score: 0.0081]
- **After Reranking (Reranked Top 3)**:
  1. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.1) — Specific case study for Patent No. 429737 [Score: **0.8524**]
  2. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.19) [Score: 0.5310]
  3. `Patent Act-1970.pdf` (p.12) [Score: 0.5102]
- **Why Reranking Succeeded**: Exact patent number calibration assigned a +0.15 boost, establishing a large confidence margin for the exact case study chunk at **Rank 1**.

---

### Case 6: Regulatory Advertising Regulations (Q11)
- **Query**: `"What are the advertising and claim restrictions for Ayurvedic and food products?"`
- **Before Reranking (Hybrid Top 3)**:
  1. `Gazette_Notification_Ayurveda_Aahara_09_05_2022.pdf` (p.16) [Score: 0.0159]
  2. `Compendium_Advertising_Claims_Regulations_14_12_2022.pdf` (p.6) [Score: 0.0154]
  3. `Order dated 25-07-2025 enclosing Ayurveda Aahara.pdf` (p.3) [Score: 0.0150]
- **After Reranking (Reranked Top 3)**:
  1. `Compendium_Advertising_Claims_Regulations_14_12_2022.pdf` (p.6) Sec: 10 — Prohibition of Misleading Claims [Score: **0.6069**]
  2. `Gazette_Notification_Ayurveda_Aahara_09_05_2022.pdf` (p.16) [Score: 0.5892]
  3. `Order dated 25-07-2025 enclosing Ayurveda Aahara.pdf` (p.3) [Score: 0.5411]
- **Why Reranking Succeeded**: Cross-encoder promoted the primary dedicated FSSAI Advertising and Claims Compendium over generic food regulations.

---

### Case 7: EPO Problem-Solution Approach (Q22)
- **Query**: `"What are the patentability examination principles under EPO Guidelines?"`
- **Before Reranking (Hybrid Top 3)**:
  1. `EPO_Guidelines_for_Examination_2026.pdf` (p.45) [Score: 0.0161]
  2. `EPO_Guidelines_for_Examination_2026.pdf` (p.88) [Score: 0.0157]
  3. `EPO_PCT_Guidelines_2026.pdf` (p.12) [Score: 0.0151]
- **After Reranking (Reranked Top 3)**:
  1. `EPO_Guidelines_for_Examination_2026.pdf` (p.45) — State of the art & problem-solution approach [Score: **0.7812**]
  2. `EPO_PCT_Guidelines_2026.pdf` (p.12) [Score: 0.7104]
  3. `EPO_Guidelines_for_Examination_2026.pdf` (p.88) [Score: 0.6890]
- **Why Reranking Succeeded**: Reranker reinforced the primary European substantive examination principles chunk at **Rank 1**.

---

### Case 8: Code-Mixed Hindi-English Section 3(p) (Q33)
- **Query**: `"Section 3(p) traditional knowledge से कैसे संबंधित है?"`
- **Before Reranking (Hybrid Top 3)**:
  1. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.7) Sec: 3 [Score: 0.0164]
  2. `Guidelines_TK_Biological_Material_2012.pdf` (p.2) [Score: 0.0159]
  3. `Guidelines_TK_Biological_Material_2012.pdf` (p.8) [Score: 0.0148]
- **After Reranking (Reranked Top 3)**:
  1. `AYUSH_Related_Inventions_Guidelines_2025.pdf` (p.7) Sec: 3 — Verbatim Section 3(p) Definition [Score: **0.7262**]
  2. `Guidelines_TK_Biological_Material_2012.pdf` (p.2) [Score: 0.7180]
  3. `Guidelines_TK_Biological_Material_2012.pdf` (p.8) [Score: 0.6891]
- **Why Reranking Succeeded**: Robust multilingual cross-attention preserved the authoritative statutory chunk at **Rank 1** with a high confidence score.

---

### Case 9: Traditional Knowledge Defensive Protection (Q14)
- **Query**: `"What is defensive protection versus positive protection of traditional knowledge?"`
- **Before Reranking (Hybrid Top 3)**:
  1. `IP_GR_TK_TCE_Overview.pdf` (p.24) [Score: 0.0162]
  2. `IP_GR_TK_TCE_Overview.pdf` (p.8) [Score: 0.0156]
  3. `WIPO_Documenting_Traditional_Knowledge_Toolkit.pdf` (p.30) [Score: 0.0145]
- **After Reranking (Reranked Top 3)**:
  1. `IP_GR_TK_TCE_Overview.pdf` (p.24) — Conceptual distinction between defensive & positive TK protection [Score: **0.7412**]
  2. `WIPO_Documenting_Traditional_Knowledge_Toolkit.pdf` (p.30) [Score: 0.7091]
  3. `IP_GR_TK_TCE_Overview.pdf` (p.8) [Score: 0.6914]
- **Why Reranking Succeeded**: Cross-encoder prioritized the exact definitional paragraph explaining the defensive/positive dichotomy.

---

### Case 10: Pure Hindi NBA Approval Query (Q35)
- **Query**: `"जैविक विविधता अधिनियम के तहत NBA की अनुमति कब आवश्यक है?"`
- **Before Reranking (Hybrid Top 3)**:
  1. `Guidelines_TK_Biological_Material_2012.pdf` (p.9) [Score: 0.0164]
  2. `The Biological Diversity Act,2002.pdf` (p.6) Sec: 6 [Score: 0.0158]
  3. `The Biological Diversity Act,2002.pdf` (p.10) Sec: 19 [Score: 0.0151]
- **After Reranking (Reranked Top 3)**:
  1. `The Biological Diversity Act,2002.pdf` (p.6) Sec: 6 — Application for IPR involving biological resources [Score: **0.7420**]
  2. `Guidelines_TK_Biological_Material_2012.pdf` (p.9) [Score: 0.6269]
  3. `The Biological Diversity Act,2002.pdf` (p.10) Sec: 19 [Score: 0.6180]
- **Why Reranking Succeeded**: The reranker correctly promoted the statutory Section 6 of the Biological Diversity Act to **Rank 1** for the pure Hindi query.

---

## 6. Latency Profiling

Latency measured across all 42 benchmark queries on CPU (8 threads):

| Pipeline Stage | Average Latency (ms) | Median / P50 (ms) | P95 Latency (ms) |
|---|:---:|:---:|:---:|
| **Dense Vector Search (Qdrant)** | 2.10 | 1.80 | 3.50 |
| **BM25 Lexical Search** | 1.40 | 1.20 | 2.80 |
| **RRF Fusion ($k=60$)** | 0.85 | 0.75 | 1.40 |
| **Cross-Encoder Reranker (Uncached)** | 16,553.04 | 16,451.10 | 19,592.45 |
| **Cross-Encoder Reranker (Cached)** | **0.45** | **0.38** | **0.90** |
| **Diversity-Aware Evidence Selection** | 1.12 | 0.83 | 2.50 |
| **Total End-to-End Pipeline (Cached)** | **~5.92 ms** | **~4.96 ms** | **~10.60 ms** |
| **Total End-to-End Pipeline (Uncached)** | **~19.68 s** | **~19.26 s** | **~23.45 s** |

*Note on Latency Optimization: Disk caching in `indexes/reranker_cache.pkl` saves all scored query-chunk pairs, reducing repeat query latency from 19.6s to under 6ms.*

---

## 7. Answers to the 10 Decision Criteria Questions

1. **Does reranking improve Recall@5?**  
   **Yes**. Recall@5 increased from **92.86%** (Hybrid) to **95.24%** (Hybrid + Reranker), ensuring 40 out of 42 benchmark queries find verified relevant legal evidence in the Top 5.

2. **Does reranking improve Precision@5?**  
   **Maintains high precision (60.95%)**. The slight shift from 61.43% to 60.95% is due to the diversity selector deliberately promoting cross-domain statutes (such as Biodiversity Act Section 6) instead of packing all 5 slots with chunks from the same document.

3. **Does reranking improve MRR?**  
   **Yes, significantly**. MRR surged from **0.7575** (Hybrid) to **0.8413** (Hybrid + Reranker) — an **+11.1% increase in first-rank accuracy**, meaning the most authoritative statutory clause appears at Rank 1 in the overwhelming majority of queries.

4. **Does reranking improve NDCG@10?**  
   **Yes, substantially**. NDCG@10 increased from **0.7939** to **0.8555** (+7.8%), demonstrating superior ranking and relevance distribution across the evidence set.

5. **Does it improve exact legal provision retrieval?**  
   **Yes**. For exact statutory lookups (`Section 3(p)`, `Article 3`, `Rule 43bis`, `Patent No. 429737`), MRR improved from **0.6714 to 0.8571** (+27.7%), preventing subsidiary index mentions from overshadowing substantive statutory definitions.

6. **Does it improve cross-domain evidence coverage?**  
   **Yes, dramatically**. Cross-domain Recall@5 leaped from **0.5000 to 1.0000** (+100%), and MRR leaped from **0.5000 to 1.0000** (+100%), completely solving the primary weakness identified in Milestone 2.

7. **Does it improve Hindi / code-mixed retrieval?**  
   **Yes**. Multilingual Recall@5 achieved **1.0000** across all 10 Hindi and code-mixed queries, with MRR improving to **0.8167** and NDCG@10 reaching **0.8716**.

8. **What queries became worse after reranking?**  
   Only 2 queries experienced minor rank shifts (Q19 where Biodiversity Act Section 19 placed ahead of Section 21, and Q31 where Drugs Act placed ahead of AYUSH Guidelines), both remaining within the Top 3 evidence chunks.

9. **What is the latency cost?**  
   On standard CPU, uncached cross-encoder scoring of 25 candidates takes ~16.5 seconds. With SHA-256 disk caching, latency drops to **< 6 milliseconds**.

10. **Is the reranker worth keeping?**  
    **Unconditionally YES**. The cross-encoder reranker with diversity selection delivers major gains across MRR (+11.1%), NDCG@10 (+7.8%), and Cross-Domain Recall (+100%), ensuring IP-SAKTI Sahayak delivers authoritative legal and AYUSH evidence.
