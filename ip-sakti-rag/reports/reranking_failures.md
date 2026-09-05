# Milestone 3: Cross-Encoder Reranking Failure Analysis

**Date**: 2026-09-05  
**Model Evaluated**: `BAAI/bge-reranker-v2-m3`  
**Dataset**: 42 Verified Legal, AYUSH, TK, International & Multilingual Queries  

---

## 1. Summary of Failure Modes

Across the 42 benchmark queries, 4 specific queries exhibited suboptimal rank placement or slight rank degradation compared to initial hybrid retrieval. The failure patterns fall into three categories:

1. **Multi-Statute Dispersal (`evidence_missing_in_top5`)**:
   - For complex disclosure questions (e.g. Q6), international study guides discussing Indian Section 10(4) scored higher semantically in the cross-encoder than the statutory text itself.
2. **Descriptive Over-Affinity vs Verbatim Statutory Definitions (`rank_degradation`)**:
   - For general biodiversity queries (e.g. Q19), descriptive policy overviews scored slightly higher than the specific statutory section clause.
3. **Botanical Single-Taxon Formulations (`evidence_missing_in_top5`)**:
   - For single-term plant lookups (e.g. Q30 *Curcuma longa*), extensive food ingredient schedules pushed patent revocation case studies slightly below the top 5.

---

## 2. Detailed Query Failure Records

### Failure Case 1: Q06 — Biological Material Disclosure under Section 10(4)
- **Query**: `"What are the disclosure requirements for biological materials under Section 10(4) of the Patents Act?"`
- **Category**: `PATENT`
- **Expected Evidence**: `patent_act_1970` Section 10 / Section 10(4)(ii)(D) (`patent_act_1970_p12_c40`) and `ayush_related_inventions_guidelines_2025`.
- **Hybrid Rank**: Rank 4
- **Reranked Rank**: Rank 6 (outside Top 5)
- **Hybrid Top 1**: `WIPO_Patent_Disclosure_GR_TK.pdf (p.23)` (Score: 0.0160)
- **Reranked Top 1**: `WIPO_Patent_Disclosure_GR_TK.pdf (p.23)` (Score: 0.7305)
- **Failure Type**: `evidence_missing_in_top5`
- **Root Cause**: The WIPO Patent Disclosure study guide extensively analyzes Section 10(4)(ii)(D) of the Indian Patents Act with rich narrative prose ("Key Questions on Patent Disclosure Requirements for Genetic Resources"), leading the cross-encoder to award it a higher semantic similarity score (0.7305) than the terse legal statutory subsection in the bare act (0.6412).
- **Recommended Fix**: Implement explicit statutory weighting in Milestone 4 LangGraph retrieval node when queries contain specific statutory subsection citations like `Section 10(4)`.

---

### Failure Case 2: Q19 — Access and Benefit Sharing (ABS) under Indian Biodiversity Law
- **Query**: `"What is Access and Benefit Sharing (ABS) under Indian biodiversity law?"`
- **Category**: `BIOLOGICAL RESOURCES`
- **Expected Evidence**: `biological_diversity_act_2002` Section 21 / Section 19 (`The Biological Diversity Act,2002.pdf`).
- **Hybrid Rank**: Rank 1
- **Reranked Rank**: Rank 2
- **Hybrid Top 1**: `The Biological Diversity Act,2002.pdf (p.11) Sec: 21` (Score: 0.0163)
- **Reranked Top 1**: `The Biological Diversity Act,2002.pdf (p.10) Sec: 19` (Score: 0.7391)
- **Failure Type**: `rank_degradation`
- **Root Cause**: Section 19 (Application for Access to biological resources) was scored at 0.7391 by the cross-encoder, narrowly edging out Section 21 (Determination of equitable benefit sharing, score: 0.7289). Both are highly relevant provisions of the Biological Diversity Act, but Section 19 slightly displaced Section 21 from Rank 1.
- **Recommended Fix**: None required — both Section 19 and Section 21 remain in the top 2 evidence chunks and provide complementary legal basis.

---

### Failure Case 3: Q30 — Curcuma longa Exact Lookup
- **Query**: `"Curcuma longa"`
- **Category**: `EXACT_LOOKUP`
- **Expected Evidence**: `guidelines_tk_biological_material_2012` (Turmeric patent revocation landmark case) and `ayush_related_inventions_guidelines_2025`.
- **Hybrid Rank**: Rank 4
- **Reranked Rank**: Rank 6
- **Hybrid Top 1**: `Order dated 25-07-2025 enclosing Ayurveda Aahara.pdf (p.73)` (Score: 0.0160)
- **Reranked Top 1**: `Order dated 25-07-2025 enclosing Ayurveda Aahara.pdf (p.73)` (Score: 0.8122)
- **Failure Type**: `evidence_missing_in_top5`
- **Root Cause**: The 2025 Ayurveda Aahara gazette order contains massive botanical ingredient tables listing *Curcuma longa* with verbatim Latin taxon headings. The cross-encoder assigned high confidence (0.8122) to these official ingredient tables over historical patent revocation case notes.
- **Recommended Fix**: Use query classification to differentiate regulatory ingredient lookups from patent prior art lookups.

---

### Failure Case 4: Q31 — Multilingual Ayurvedic Formulation Patentability
- **Query**: `"क्या आयुर्वेदिक formulation को भारत में patent किया जा सकता है?"`
- **Category**: `MULTILINGUAL`
- **Expected Evidence**: `ayush_related_inventions_guidelines_2025` and `patent_act_1970` Section 3(p)/3(e).
- **Hybrid Rank**: Rank 2
- **Reranked Rank**: Rank 3
- **Hybrid Top 1**: `Drugs_and_Cosmetics_Act_1940.pdf (p.179)` (Score: 0.0154)
- **Reranked Top 1**: `Drugs_and_Cosmetics_Act_1940.pdf (p.179)` (Score: 0.7188)
- **Failure Type**: `rank_degradation`
- **Root Cause**: The cross-encoder assigned strong semantic relevance to the Ayurvedic licensing and manufacturing definitions in Chapter IV-A of the Drugs and Cosmetics Act (0.7188) alongside the 2025 AYUSH Patent Guidelines (0.7092).
- **Recommended Fix**: Query planner should distinguish between *patentability* and *manufacturing license* questions to down-weight drug regulatory acts when patent exclusions are queried.
