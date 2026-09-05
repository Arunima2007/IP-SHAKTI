# Milestone 4: Generation Failures & Flagged Issues Log

**Date:** 2026-09-05 02:27:28  
**Scope:** All queries with flagged claims, citation mismatches, or out-of-scope refusals.

---

## 1. Out-of-Scope Safe Refusals (Expected & Verified)

The following queries tested the system's hallucination resistance on out-of-scope / missing legal domains. All successfully yielded safe refusal without generating unsupported claims:

### `gen_q29`: What are the specific patent registration procedures in Brazil under the Brazilian 1996 Industrial Property Law (Law No. 9.279)?
- **Expected Behavior:** Safe Refusal (`should_refuse: True`)
- **Refusal Status:** `PASSED_REFUSAL`
- **Output Generated:** `I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively.

### Sources

No specific statutory sources cited.`

### `gen_q30`: What are the copyright registration fees for computer software in Japan under the Japanese Copyright Act of 1970?
- **Expected Behavior:** Safe Refusal (`should_refuse: True`)
- **Refusal Status:** `PASSED_REFUSAL`
- **Output Generated:** `I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively.

### Sources

No specific statutory sources cited.`

### `gen_q31`: What are the tax deduction percentages for electric vehicle purchases in California under the 2024 Clean Vehicle Rebate Program?
- **Expected Behavior:** Safe Refusal (`should_refuse: True`)
- **Refusal Status:** `PASSED_REFUSAL`
- **Output Generated:** `I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively.

### Sources

No specific statutory sources cited.`

### `gen_q32`: Explain the orbital mechanics of the Chandrayaan-3 lunar propulsion module during trans-lunar injection.
- **Expected Behavior:** Safe Refusal (`should_refuse: True`)
- **Refusal Status:** `PASSED_REFUSAL`
- **Output Generated:** `I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively.

### Sources

No specific statutory sources cited.`

### `gen_q33`: What is the corporate income tax rate for manufacturing companies in Germany under the Corporate Tax Act (KStG)?
- **Expected Behavior:** Safe Refusal (`should_refuse: True`)
- **Refusal Status:** `PASSED_REFUSAL`
- **Output Generated:** `I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively.

### Sources

No specific statutory sources cited.`

### `gen_q34`: What are the FAA airspace classification rules for commercial drone operations in Australia?
- **Expected Behavior:** Safe Refusal (`should_refuse: True`)
- **Refusal Status:** `PASSED_REFUSAL`
- **Output Generated:** `I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively.

### Sources

No specific statutory sources cited.`

---

## 2. Flagged Inconsistencies & Remediation Actions

### Query `gen_q11`: What are the conditions for patentability of medical treatment methods under Section 3(i) of the Indian Patents Act?
- **Issue Type:** `unsupported_entity`
  - Claim: "In addition, under Drugs_and_Cosmetics_Act_1940.pdf (Chapter III), Drugs and Cosmetics Rules 1945 93 Provided further that for tests requiring sophisticated instrumentation techniques or biological or microbiological methods the licensing authority may permit such test to be conducted by institutions approved by it under Part XV(A) of these Rules for this purpose.] Explanation.−A person who satisfies the following minimum qualifications shall be deemed to be a ―competent person‖ for the purposes of rule 71A or 74A of these rules, namely: − (a) a person who holds the Diploma in Pharmacy approved by the Pharmacy Council of India under the Pharmacy Act, 1948 (VIII of 1948) or a person who is registered under the said Act, or (b) a person who has passed the Intermediate examination with Chemistry as one of the principal subjects or an examination equivalent to it or an examination recognized by the Licensing Authority as equivalent to it; or (c) a person who has passed the Matriculation examination or an examination recognized by the Licensing Authority as equivalent to it and has had not less than four years‘ practical experience in the manufacture, dispensing or repacking of drugs.] 1[71B."
  - Description: Entity 'Chemistry as' in claim not found in cited chunk text.

### Query `gen_q24`: Ayurveda Aahara product packaging par logo aur statutory warnings ke kya rules hain?
- **Issue Type:** `unsupported_entity`
  - Claim: "Furthermore, Information by the Food Business Operator to Food Authority.- The Food Business Operator shall inform the licensing authority in writing, if any, of his existing food products duly licensed to be assigned as an Ayurveda Aahara and the Licen..."
  - Description: Entity 'Operator shall' in claim not found in cited chunk text.

### Query `gen_q28`: What are the IP and regulatory steps to commercialize an Ayurvedic botanical formulation internationally via PCT and in India?
- **Issue Type:** `unsupported_entity`
  - Claim: "Furthermore, Indian scientists at the Tropical Botanic Garden and Research Institute used the tribal know-how to develop the drug."
  - Description: Entity 'Institute used' in claim not found in cited chunk text.
