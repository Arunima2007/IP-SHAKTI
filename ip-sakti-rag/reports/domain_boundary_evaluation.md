# Domain Boundary Classification & Safe Refusal Evaluation Report

**System**: IP-SAKTI Sahayak  
**Evaluation Date**: September 5, 2026  
**Target Milestone**: Domain Boundary, Zero-Leakage Short-Circuiting, and Natural Formatting  

---

## Executive Summary

This report evaluates the accuracy, safety, and latency of the IP-SAKTI Sahayak domain-boundary classification and safe refusal subsystem. The system is designed to provide authoritative assistance strictly across 11 designated legal/regulatory domains while immediately short-circuiting out-of-scope or general knowledge queries before dense/lexical retrieval or LLM generation.

---

## Benchmark Metrics

| Metric | Target | Achieved | Status |
| :--- | :--- | :--- | :--- |
| **Total Test Queries Evaluated** | — | **46** | Complete |
| **Out-of-Scope Classification Accuracy** | $\ge 95.0\%$ | **100.0%** (29/29) | **PASSED** |
| **In-Scope Query Accuracy** | $\ge 95.0\%$ | **100.0%** (17/17) | **PASSED** |
| **False Acceptance Rate (General Knowledge)** | $\le 5.0\%$ | **0.0%** (0/29) | **PASSED** |
| **False Rejection Rate (Valid IP/AYUSH)** | $\le 5.0\%$ | **0.0%** (0/17) | **PASSED** |
| **Retrieval Short-Circuit Rate (Out-of-Scope)** | $100.0\%$ | **100.0%** (29/29) | **PASSED** |
| **Average Out-of-Scope Latency** | $< 200\text{ ms}$ | **0.75 ms** | **PASSED** |

---

## Detailed Evaluation Breakdown

### 1. Correctly Refused Out-of-Scope Queries (Zero Leakage)

All out-of-scope queries immediately bypass retrieval, reranking, and Gemini answer generation, returning the domain boundary guidance in $< 1\text{ ms}$ average latency.

| # | User Query | Language | Classification | Short-Circuit Verified | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | *Who is Virat Kohli?* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.81 |
| 2 | *What is the capital of India?* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.69 |
| 3 | *What is the capital of France?* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.72 |
| 4 | *Write me a Python program* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.68 |
| 5 | *Write a Python program to sort an array.* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.74 |
| 6 | *What is today's weather?* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.70 |
| 7 | *Tell me a joke.* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.71 |
| 8 | *Who won the cricket match?* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.73 |
| 9 | *How do I cook pasta?* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.75 |
| 10 | *What is Bitcoin?* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.70 |
| 11 | *Give me relationship advice* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.72 |
| 12 | *Virat Kohli kaun hai?* | Hinglish | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.78 |
| 13 | *Python mein RAG kaise banaye?* | Hinglish | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.77 |
| 14 | *Who won the FIFA World Cup?* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.72 |
| 15 | *What is the stock price of Apple?* | English | `OUT_OF_SCOPE` | Yes (0 retrieval calls) | 0.71 |

**Standard Out-of-Scope Refusal Response**:
> *"I can help with Intellectual Property, Ayurveda/AYUSH regulations, Traditional Knowledge, Biological Diversity, and related international IP frameworks. This question is outside my supported domain."*

---

### 2. Correctly Accepted In-Scope Queries (Full Pipeline Execution)

In-scope queries are routed to hybrid retrieval (Dense + BM25), Cross-Encoder reranking, evidence sufficiency verification, and grounded answer generation.

| # | User Query | Language | Domain / Type | Confidence | Route |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | *What is Section 3(p) of the Patents Act?* | English | `PATENT` / `EXACT_LOOKUP` | 0.97 | Full Pipeline |
| 2 | *Can traditional knowledge be patented in India?* | English | `TRADITIONAL_KNOWLEDGE` / `EXPLANATORY` | 0.89 | Full Pipeline |
| 3 | *What is the role of TKDL in patent examination?* | English | `TRADITIONAL_KNOWLEDGE` / `EXPLANATORY` | 0.89 | Full Pipeline |
| 4 | *What is TKDL?* | English | `TRADITIONAL_KNOWLEDGE` / `FACTUAL` | 0.89 | Full Pipeline |
| 5 | *What approval is required from NBA?* | English | `BIODIVERSITY` / `FACTUAL` | 0.89 | Full Pipeline |
| 6 | *When is NBA approval required under Biological Diversity Act?* | English | `BIODIVERSITY` / `EXPLANATORY` | 0.93 | Full Pipeline |
| 7 | *What are the PCT international phase requirements?* | English | `INTERNATIONAL_IP` / `EXPLANATORY` | 0.89 | Full Pipeline |
| 8 | *What is PCT Rule 43bis?* | English | `INTERNATIONAL_IP` / `EXACT_LOOKUP` | 0.97 | Full Pipeline |
| 9 | *Can an Ayurvedic formulation be patented in India?* | English | `AYUSH` / `AYURVEDA_IP` | 0.93 | Full Pipeline |
| 10 | *Can Ayurvedic inventions receive patent protection?* | English | `AYUSH` / `AYURVEDA_IP` | 0.93 | Full Pipeline |
| 11 | *What are the requirements for patent disclosure involving biological resources?* | English | `CROSS_DOMAIN` / `CROSS_DOMAIN` | 0.93 | Full Pipeline |
| 12 | *Explain Section 3(d) in the context of an Ayurvedic invention.* | English | `AYUSH` / `EXACT_LOOKUP` | 0.97 | Full Pipeline |
| 13 | *Section 3(p) kya kehta hai?* | Hinglish | `PATENT` / `EXACT_LOOKUP` | 0.97 | Full Pipeline |
| 14 | *TKDL kya hai?* | Hinglish | `TRADITIONAL_KNOWLEDGE` / `CODE_MIXED` | 0.89 | Full Pipeline |
| 15 | *क्या आयुर्वेदिक आविष्कार का पेटेंट हो सकता है?* | Hindi | `AYUSH` / `MULTILINGUAL` | 0.93 | Full Pipeline |
| 16 | *क्या आयुर्वेदिक औषधि पर पेटेंट मिल सकता है?* | Hindi | `AYUSH` / `MULTILINGUAL` | 0.93 | Full Pipeline |
| 17 | *How can I patent a cricket-related Ayurvedic product?* | English | `AYUSH` / `AYURVEDA_IP` | 0.88 | Full Pipeline |

---

### 3. Evidentiary Safeguards & Wrong-Document Prevention

1. **Domain Consistency Guard**:
   - `EvidenceSufficiencyNode` verifies retrieved chunk domains against query domain intent.
   - Prevents ungrounded fallback when lexical or vector retrieval pulls loosely matched tokens (e.g. `Drugs_and_Cosmetics_Act_1940.pdf` for unrelated general topics).
2. **Refusal Contract Distinction**:
   - **Out-of-Scope Queries**: Returns domain boundary guidance (`SAFE_REFUSAL`).
   - **In-Scope Queries lacking Evidence**: Returns evidentiary safeguard (`INSUFFICIENT_EVIDENCE`):
     *"I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."*

---

### 4. Presentation & Typography Verification

- **Eliminated ALL-CAPS Transforms**: Removed `uppercase` CSS class transforms from markdown headings (`<h4>`) in `ChatArea.tsx`.
- **Document Title Normalization**: Raw filenames (e.g., `Patent Act-1970.pdf`, `DRUGS_AND_COSMETICS_ACT_1940.pdf`) are cleaned into standard legal titles (e.g., `Patents Act, 1970`, `Drugs and Cosmetics Act, 1940`).
- **Standardized Section Structure**: Output uses natural sentence-case legal structure:
  - `### Answer`
  - `### Explanation`
  - `### Applicable provisions`
  - `### Sources`

---

## Conclusion

The domain boundary classifier and zero-leakage LangGraph routing achieve **100% accuracy** on the evaluation suite with **0% false acceptances** and sub-millisecond out-of-scope latency (**0.75 ms**), completely resolving erroneous retrieval on out-of-scope queries and restoring natural, professional presentation across the application.
