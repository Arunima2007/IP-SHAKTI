"""
LangGraph Comprehensive Evaluation Benchmark (42 Queries across 10 Categories)
Covers all domain-specific tests required for Milestone 5:
- Simple Factual (5)
- Exact Legal Lookup (5)
- Ayurveda / IP (5)
- Traditional Knowledge (5)
- Biological Resources / NBA (5)
- International IP / PCT / TRIPS (5)
- Hindi Queries (5)
- Hinglish / Code-mixed (5)
- Cross-Domain Intersections (5)
- Out-of-Scope & Insufficient Evidence (5)
"""

BENCHMARK_QUERIES = [
    # 1. Simple Factual (5)
    {
        "id": "SF-01",
        "category": "SIMPLE_FACTUAL",
        "query": "What is the term of a patent in India under the Indian Patents Act, 1970?",
        "expected_domain": "PATENT",
        "expected_type": "FACTUAL",
        "expected_sufficiency": True
    },
    {
        "id": "SF-02",
        "category": "SIMPLE_FACTUAL",
        "query": "What constitutes an invention under Section 2(1)(j) of the Patents Act?",
        "expected_domain": "PATENT",
        "expected_type": "FACTUAL",
        "expected_sufficiency": True
    },
    {
        "id": "SF-03",
        "category": "SIMPLE_FACTUAL",
        "query": "Who is eligible to file a patent application under Indian law?",
        "expected_domain": "PATENT",
        "expected_type": "FACTUAL",
        "expected_sufficiency": True
    },
    {
        "id": "SF-04",
        "category": "SIMPLE_FACTUAL",
        "query": "What is the purpose of the National Biodiversity Authority established under the Biological Diversity Act?",
        "expected_domain": "BIODIVERSITY",
        "expected_type": "FACTUAL",
        "expected_sufficiency": True
    },
    {
        "id": "SF-05",
        "category": "SIMPLE_FACTUAL",
        "query": "What are the key requirements for an international patent application under the PCT?",
        "expected_domain": "INTERNATIONAL_IP",
        "expected_type": "FACTUAL",
        "expected_sufficiency": True
    },

    # 2. Exact Legal Lookup (5)
    {
        "id": "EL-01",
        "category": "EXACT_LOOKUP",
        "query": "What does Section 3(p) of the Indian Patents Act, 1970 state regarding traditional knowledge?",
        "expected_domain": "PATENT",
        "expected_type": "EXACT_LOOKUP",
        "expected_sufficiency": True
    },
    {
        "id": "EL-02",
        "category": "EXACT_LOOKUP",
        "query": "Explain the exact exclusion in Section 3(d) of the Patents Act regarding new forms of known substances.",
        "expected_domain": "PATENT",
        "expected_type": "EXACT_LOOKUP",
        "expected_sufficiency": True
    },
    {
        "id": "EL-03",
        "category": "EXACT_LOOKUP",
        "query": "What are the requirements of Section 6 of the Biological Diversity Act, 2002 for obtaining IPR?",
        "expected_domain": "BIODIVERSITY",
        "expected_type": "EXACT_LOOKUP",
        "expected_sufficiency": True
    },
    {
        "id": "EL-04",
        "category": "EXACT_LOOKUP",
        "query": "What is PCT Rule 43bis regarding written opinions of the International Searching Authority?",
        "expected_domain": "INTERNATIONAL_IP",
        "expected_type": "EXACT_LOOKUP",
        "expected_sufficiency": True
    },
    {
        "id": "EL-05",
        "category": "EXACT_LOOKUP",
        "query": "What are the labeling and safety standards prescribed under the Ayurveda Aahara Regulations 2022?",
        "expected_domain": "AYURVEDA",
        "expected_type": "EXACT_LOOKUP",
        "expected_sufficiency": True
    },

    # 3. Ayurveda / IP (5)
    {
        "id": "AY-01",
        "category": "AYURVEDA_IP",
        "query": "Can a classical Ayurvedic formulation described in the Ayurvedic Pharmacopoeia of India be patented as an invention?",
        "expected_domain": "AYURVEDA",
        "expected_type": "AYURVEDA_IP",
        "expected_sufficiency": True
    },
    {
        "id": "AY-02",
        "category": "AYURVEDA_IP",
        "query": "What regulatory approvals are mandatory for manufacturing and selling Ayurvedic drugs in India?",
        "expected_domain": "AYURVEDA",
        "expected_type": "AYURVEDA_IP",
        "expected_sufficiency": True
    },
    {
        "id": "AY-03",
        "category": "AYURVEDA_IP",
        "query": "How does the Indian Patent Office evaluate inventive step and synergy for herbal and Ayurvedic compositions?",
        "expected_domain": "AYURVEDA",
        "expected_type": "AYURVEDA_IP",
        "expected_sufficiency": True
    },
    {
        "id": "AY-04",
        "category": "AYURVEDA_IP",
        "query": "What are the rules regarding misleading advertisements and claims for ASU (Ayurveda, Siddha, Unani) drugs?",
        "expected_domain": "AYURVEDA",
        "expected_type": "AYURVEDA_IP",
        "expected_sufficiency": True
    },
    {
        "id": "AY-05",
        "category": "AYURVEDA_IP",
        "query": "What criteria distinguish proprietary Ayurvedic medicines from classical Ayurvedic formulations under the Drugs and Cosmetics Act?",
        "expected_domain": "AYURVEDA",
        "expected_type": "AYURVEDA_IP",
        "expected_sufficiency": True
    },

    # 4. Traditional Knowledge (5)
    {
        "id": "TK-01",
        "category": "TRADITIONAL_KNOWLEDGE",
        "query": "What role does the Traditional Knowledge Digital Library (TKDL) play in preventing biopiracy and invalid patent grants?",
        "expected_domain": "TRADITIONAL_KNOWLEDGE",
        "expected_type": "FACTUAL",
        "expected_sufficiency": True
    },
    {
        "id": "TK-02",
        "category": "TRADITIONAL_KNOWLEDGE",
        "query": "How does WIPO protect traditional knowledge and genetic resources at the international multilateral level?",
        "expected_domain": "TRADITIONAL_KNOWLEDGE",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "TK-03",
        "category": "TRADITIONAL_KNOWLEDGE",
        "query": "Why are mere admixtures or known traditional uses excluded from patentability under Indian law?",
        "expected_domain": "TRADITIONAL_KNOWLEDGE",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "TK-04",
        "category": "TRADITIONAL_KNOWLEDGE",
        "query": "What evidence must a patent examiner review to verify if a claimed herbal medicine is part of prior art in traditional knowledge?",
        "expected_domain": "TRADITIONAL_KNOWLEDGE",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "TK-05",
        "category": "TRADITIONAL_KNOWLEDGE",
        "query": "How do international patent offices utilize TKDL access agreements during prior art search and examination?",
        "expected_domain": "TRADITIONAL_KNOWLEDGE",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },

    # 5. Biological Resources / NBA (5)
    {
        "id": "BR-01",
        "category": "BIOLOGICAL_RESOURCES",
        "query": "When is prior approval from the National Biodiversity Authority (NBA) required before applying for a patent in India?",
        "expected_domain": "BIODIVERSITY",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "BR-02",
        "category": "BIOLOGICAL_RESOURCES",
        "query": "What are the consequences of failing to disclose the biological source of an invention in a patent specification under Section 10?",
        "expected_domain": "BIODIVERSITY",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "BR-03",
        "category": "BIOLOGICAL_RESOURCES",
        "query": "What is the benefit-sharing mechanism under the Biological Diversity Act, 2002 for commercial utilization of Indian bio-resources?",
        "expected_domain": "BIODIVERSITY",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "BR-04",
        "category": "BIOLOGICAL_RESOURCES",
        "query": "Are Indian citizens and entities required to intimate State Biodiversity Boards (SBB) before accessing biological resources for commercial utilization?",
        "expected_domain": "BIODIVERSITY",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "BR-05",
        "category": "BIOLOGICAL_RESOURCES",
        "query": "What exemptions exist for local people and indigenous communities under the Biological Diversity Act?",
        "expected_domain": "BIODIVERSITY",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },

    # 6. International IP / PCT / TRIPS (5)
    {
        "id": "INT-01",
        "category": "INTERNATIONAL_IP",
        "query": "What are the timeline and requirements for entering the National Phase in India under the Patent Cooperation Treaty (PCT)?",
        "expected_domain": "INTERNATIONAL_IP",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "INT-02",
        "category": "INTERNATIONAL_IP",
        "query": "How does the TRIPS Agreement define minimum standards for patent protection and public health flexibilities?",
        "expected_domain": "INTERNATIONAL_IP",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "INT-03",
        "category": "INTERNATIONAL_IP",
        "query": "What is the function of the International Preliminary Report on Patentability (IPRP) under PCT Chapter II?",
        "expected_domain": "INTERNATIONAL_IP",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "INT-04",
        "category": "INTERNATIONAL_IP",
        "query": "How does the Paris Convention right of priority apply to subsequent patent filings in member states?",
        "expected_domain": "INTERNATIONAL_IP",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },
    {
        "id": "INT-05",
        "category": "INTERNATIONAL_IP",
        "query": "What are the international publication requirements for PCT applications under Article 21?",
        "expected_domain": "INTERNATIONAL_IP",
        "expected_type": "EXPLANATORY",
        "expected_sufficiency": True
    },

    # 7. Hindi Queries (5)
    {
        "id": "HI-01",
        "category": "HINDI",
        "query": "भारतीय पेटेंट कानून के तहत पारंपरिक ज्ञान (Traditional Knowledge) को पेटेंट क्यों नहीं कराया जा सकता?",
        "expected_domain": "TRADITIONAL_KNOWLEDGE",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },
    {
        "id": "HI-02",
        "category": "HINDI",
        "query": "राष्ट्रीय जैव विविधता प्राधिकरण (NBA) से पेटेंट के लिए अनुमति कब आवश्यक होती है?",
        "expected_domain": "BIODIVERSITY",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },
    {
        "id": "HI-03",
        "category": "HINDI",
        "query": "आयुर्वेद आहार (Ayurveda Aahara) विनियम 2022 के मुख्य नियम क्या हैं?",
        "expected_domain": "AYURVEDA",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },
    {
        "id": "HI-04",
        "category": "HINDI",
        "query": "भारतीय पेटेंट अधिनियम की धारा 3(p) के तहत क्या प्रावधान हैं?",
        "expected_domain": "PATENT",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },
    {
        "id": "HI-05",
        "category": "HINDI",
        "query": "पेटेंट सहयोग संधि (PCT) के तहत भारत में राष्ट्रीय चरण में प्रवेश करने की समय सीमा क्या है?",
        "expected_domain": "INTERNATIONAL_IP",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },

    # 8. Hinglish / Code-mixed Queries (5)
    {
        "id": "HG-01",
        "category": "HINGLISH",
        "query": "Kya Ayurvedic plants aur biological resources use karke banaye gaye invention ko India me patent mil sakta hai?",
        "expected_domain": "AYURVEDA",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },
    {
        "id": "HG-02",
        "category": "HINGLISH",
        "query": "Section 3(p) ke according traditional knowledge ka patent kyu nahi milta?",
        "expected_domain": "PATENT",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },
    {
        "id": "HG-03",
        "category": "HINGLISH",
        "query": "Indian bio-resources use karne par National Biodiversity Authority ka permission kab lena padta hai?",
        "expected_domain": "BIODIVERSITY",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },
    {
        "id": "HG-04",
        "category": "HINGLISH",
        "query": "Ayurvedic drug manufacturing ke liye kaun se regulatory approvals aur licenses compulsory hain?",
        "expected_domain": "AYURVEDA",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },
    {
        "id": "HG-05",
        "category": "HINGLISH",
        "query": "PCT application me Indian Patent Office me national phase file karne ki deadline kitni hoti hai?",
        "expected_domain": "INTERNATIONAL_IP",
        "expected_type": "MULTILINGUAL",
        "expected_sufficiency": True
    },

    # 9. Cross-Domain Intersections (5)
    {
        "id": "CD-01",
        "category": "CROSS_DOMAIN",
        "query": "Can an Ayurvedic invention using traditional knowledge and biological resources be patented in India?",
        "expected_domain": "CROSS_DOMAIN",
        "expected_type": "CROSS_DOMAIN",
        "expected_sufficiency": True
    },
    {
        "id": "CD-02",
        "category": "CROSS_DOMAIN",
        "query": "How do Section 3(p) of the Patents Act, Section 6 of the Biological Diversity Act, and TKDL interact when evaluating an Ayurvedic patent application?",
        "expected_domain": "CROSS_DOMAIN",
        "expected_type": "CROSS_DOMAIN",
        "expected_sufficiency": True
    },
    {
        "id": "CD-03",
        "category": "CROSS_DOMAIN",
        "query": "What are the combined regulatory and intellectual property obligations for commercializing an Ayurvedic nutraceutical under Ayurveda Aahara and Patent Law?",
        "expected_domain": "CROSS_DOMAIN",
        "expected_type": "CROSS_DOMAIN",
        "expected_sufficiency": True
    },
    {
        "id": "CD-04",
        "category": "CROSS_DOMAIN",
        "query": "If a foreign entity files a PCT application claiming an extract from an Indian medicinal plant, what NBA approvals and TKDL prior art checks are triggered?",
        "expected_domain": "CROSS_DOMAIN",
        "expected_type": "CROSS_DOMAIN",
        "expected_sufficiency": True
    },
    {
        "id": "CD-05",
        "category": "CROSS_DOMAIN",
        "query": "How do the disclosure of origin requirements in patent law align with access and benefit sharing (ABS) under the Biodiversity Act for indigenous formulations?",
        "expected_domain": "CROSS_DOMAIN",
        "expected_type": "CROSS_DOMAIN",
        "expected_sufficiency": True
    },

    # 10. Out-of-Scope & Insufficient Evidence (5)
    {
        "id": "OS-01",
        "category": "OUT_OF_SCOPE",
        "query": "Who will win the IPL cricket match tomorrow?",
        "expected_domain": "OUT_OF_SCOPE",
        "expected_type": "OUT_OF_SCOPE",
        "expected_sufficiency": False
    },
    {
        "id": "OS-02",
        "category": "OUT_OF_SCOPE",
        "query": "What is the best recipe for baking chocolate brownies at home?",
        "expected_domain": "OUT_OF_SCOPE",
        "expected_type": "OUT_OF_SCOPE",
        "expected_sufficiency": False
    },
    {
        "id": "OS-03",
        "category": "OUT_OF_SCOPE",
        "query": "Explain the quantum mechanical wave function of a hydrogen atom.",
        "expected_domain": "OUT_OF_SCOPE",
        "expected_type": "OUT_OF_SCOPE",
        "expected_sufficiency": False
    },
    {
        "id": "OS-04",
        "category": "INSUFFICIENT_EVIDENCE",
        "query": "What are the specific tax rates for software exports in Iceland under the 2026 fiscal budget?",
        "expected_domain": "INSUFFICIENT_EVIDENCE",
        "expected_type": "INSUFFICIENT_EVIDENCE",
        "expected_sufficiency": False
    },
    {
        "id": "OS-05",
        "category": "INSUFFICIENT_EVIDENCE",
        "query": "What are the specific maritime boundary treaties between Argentina and Chile in the Beagle Channel?",
        "expected_domain": "INSUFFICIENT_EVIDENCE",
        "expected_type": "INSUFFICIENT_EVIDENCE",
        "expected_sufficiency": False
    }
]
