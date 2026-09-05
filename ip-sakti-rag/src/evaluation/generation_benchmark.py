"""Benchmark Dataset for Milestone 4 Grounded Answer Generation & Citation Verification.

Contains 34 curated questions across 8 distinct query types:
A. Simple Factual
B. Legal/Regulatory Explanation
C. Exact Lookup
D. Ayurveda / AYUSH Inventions
E. Multilingual (Hindi)
F. Code-Mixed (Hinglish)
G. Cross-Domain (Patents + Biodiversity + AYUSH + Treaties)
H. Insufficient Evidence / Out-of-Scope (Testing Refusal & Hallucination Resistance)
"""
from typing import Dict, List, Any

GENERATION_BENCHMARK_DATASET: List[Dict[str, Any]] = [
    # -------------------------------------------------------------
    # Category A: Simple Factual Queries
    # -------------------------------------------------------------
    {
        "id": "gen_q01",
        "category": "simple_factual",
        "query": "What does Section 3(p) of the Indian Patents Act, 1970 state?",
        "expected_documents": ["patent_act_1970", "guidelines_tk_biological_material_2012"],
        "expected_provisions": ["Section 3(p)"],
        "expected_claims": [
            "Section 3(p) excludes traditional knowledge from patentability.",
            "An invention which in effect is traditional knowledge or an aggregation or duplication of known properties of traditionally known components is not patentable."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q02",
        "category": "simple_factual",
        "query": "What is Section 3(e) of the Indian Patents Act?",
        "expected_documents": ["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        "expected_provisions": ["Section 3(e)"],
        "expected_claims": [
            "Section 3(e) excludes a substance obtained by a mere admixture resulting only in the aggregation of the properties of the components.",
            "A process for producing such an admixture is also excluded unless synergistic efficacy is demonstrated."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q03",
        "category": "simple_factual",
        "query": "What is Section 6 of the Biological Diversity Act, 2002?",
        "expected_documents": ["biological_diversity_act_2002"],
        "expected_provisions": ["Section 6"],
        "expected_claims": [
            "Section 6 mandates prior approval from the National Biodiversity Authority (NBA) before applying for any intellectual property right based on biological resources or associated knowledge obtained from India."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q04",
        "category": "simple_factual",
        "query": "What constitutes an invention under Section 2(1)(j) of the Patents Act, 1970?",
        "expected_documents": ["patent_act_1970"],
        "expected_provisions": ["Section 2(1)(j)"],
        "expected_claims": [
            "An invention means a new product or process involving an inventive step and capable of industrial application."
        ],
        "should_refuse": False
    },

    # -------------------------------------------------------------
    # Category B: Legal & Regulatory Explanation Queries
    # -------------------------------------------------------------
    {
        "id": "gen_q05",
        "category": "explanation",
        "query": "Why is traditional knowledge relevant to the determination of patentability in India?",
        "expected_documents": ["patent_act_1970", "guidelines_tk_biological_material_2012", "ayush_related_inventions_guidelines_2025"],
        "expected_provisions": ["Section 3(p)", "Section 2(1)(j)"],
        "expected_claims": [
            "Traditional knowledge forms part of the prior art and public domain.",
            "Inventions merely aggregating or duplicating known properties of traditional knowledge are excluded under Section 3(p).",
            "An applicant must demonstrate novel, non-obvious synergistic technical advancements beyond traditional teachings."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q06",
        "category": "explanation",
        "query": "Explain the requirement of demonstrating synergism for AYUSH-related patent applications.",
        "expected_documents": ["ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012", "patent_act_1970"],
        "expected_provisions": ["Section 3(e)", "Section 3(p)"],
        "expected_claims": [
            "To overcome objections under Section 3(e), applicants combining herbal or AYUSH ingredients must provide experimental evidence of unexpected synergistic activity.",
            "A mere combination exhibiting only the sum or known properties of individual components is not patentable."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q07",
        "category": "explanation",
        "query": "What are the disclosure requirements regarding the source and geographical origin of biological materials in a patent specification?",
        "expected_documents": ["patent_act_1970", "guidelines_tk_biological_material_2012", "wipo_patent_disclosure_gr_tk"],
        "expected_provisions": ["Section 10(4)(d)", "Section 10"],
        "expected_claims": [
            "Section 10(4)(d) of the Patents Act requires the applicant to disclose the source and geographical origin of the biological material in the specification when used in an invention."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q08",
        "category": "explanation",
        "query": "How does the WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge address patent disclosure?",
        "expected_documents": ["wipo_gr_tk_treaty_2024", "wipo_patent_disclosure_gr_tk"],
        "expected_provisions": ["Article 3"],
        "expected_claims": [
            "The WIPO GR/TK Treaty requires Contracting Parties to mandate patent applicants to disclose the country of origin of genetic resources or the indigenous/local community providing associated traditional knowledge."
        ],
        "should_refuse": False
    },

    # -------------------------------------------------------------
    # Category C: Exact Lookup Queries
    # -------------------------------------------------------------
    {
        "id": "gen_q09",
        "category": "exact_lookup",
        "query": "What does PCT Rule 43bis govern in international patent applications?",
        "expected_documents": ["pct_applicant_guide_international_phase", "epo_pct_guidelines_2026"],
        "expected_provisions": ["Rule 43bis", "Rule 43bis.1"],
        "expected_claims": [
            "PCT Rule 43bis governs the written opinion established by the International Searching Authority (ISA).",
            "It requires the ISA to state whether the claimed invention appears to be novel, involve an inventive step, and be industrially applicable."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q10",
        "category": "exact_lookup",
        "query": "What does Article 3 of the WIPO 2024 Treaty on Genetic Resources and Associated Traditional Knowledge require?",
        "expected_documents": ["wipo_gr_tk_treaty_2024"],
        "expected_provisions": ["Article 3"],
        "expected_claims": [
            "Article 3 sets out the mandatory disclosure requirement for patent applications claimed as being based on genetic resources and associated traditional knowledge."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q11",
        "category": "exact_lookup",
        "query": "What are the conditions for patentability of medical treatment methods under Section 3(i) of the Indian Patents Act?",
        "expected_documents": ["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        "expected_provisions": ["Section 3(i)"],
        "expected_claims": [
            "Section 3(i) excludes any process for the medicinal, surgical, curative, prophylactic, diagnostic, therapeutic or other treatment of human beings or animals."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q12",
        "category": "exact_lookup",
        "query": "What does Section 3(j) of the Patents Act exclude from patentability?",
        "expected_documents": ["patent_act_1970", "guidelines_tk_biological_material_2012"],
        "expected_provisions": ["Section 3(j)"],
        "expected_claims": [
            "Section 3(j) excludes plants and animals in whole or any part thereof other than micro-organisms, including seeds, varieties and species and essentially biological processes."
        ],
        "should_refuse": False
    },

    # -------------------------------------------------------------
    # Category D: Ayurveda / AYUSH IP & Regulatory Inventions
    # -------------------------------------------------------------
    {
        "id": "gen_q13",
        "category": "ayurveda_ip",
        "query": "What are the IP and patentability considerations for an Ayurvedic herbal extract formulation?",
        "expected_documents": ["ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012", "patent_act_1970", "biological_diversity_act_2002"],
        "expected_provisions": ["Section 3(p)", "Section 3(e)", "Section 6"],
        "expected_claims": [
            "Ayurvedic formulations must overcome Section 3(p) (traditional knowledge exclusion) and Section 3(e) (mere admixture).",
            "Applicants must demonstrate novel extraction techniques, specific active fractions, or synergistic therapeutic effects.",
            "Prior approval from the National Biodiversity Authority under Section 6 of the Biological Diversity Act is required if biological resources from India are utilized."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q14",
        "category": "ayurveda_ip",
        "query": "What are the regulatory requirements for Ayurveda Aahara products under FSSAI regulations?",
        "expected_documents": ["fssai_ayurveda_aahara_regulations_2022", "order_fssai_ayurveda_aahara_schedules_2025"],
        "expected_provisions": ["Regulation 3", "Regulation 4", "Regulation 5"],
        "expected_claims": [
            "Ayurveda Aahara covers food prepared in accordance with the recipes/methods described in authoritative Ayurvedic texts.",
            "Products must not contain synthetic additives or vitamins/minerals unless naturally occurring.",
            "Labeling must display the specified Ayurveda Aahara logo and clear advisory warnings."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q15",
        "category": "ayurveda_ip",
        "query": "What are the labeling and advertising restrictions for Ayurvedic proprietary medicines under AYUSH regulations?",
        "expected_documents": ["compendium_advertising_claims_regulations_2022", "gsr_669_e_drugs_rules_2024", "drugs_and_cosmetics_act_1940"],
        "expected_provisions": ["Rule 170", "Rule 158"],
        "expected_claims": [
            "Advertisements for Ayurvedic and AYUSH drugs are regulated to prevent misleading or exaggerated therapeutic claims.",
            "Labels must accurately reflect true ingredients and comply with statutory licensing mandates under the Drugs and Cosmetics Rules."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q16",
        "category": "ayurveda_ip",
        "query": "How is Prior Art defined in the Traditional Knowledge Digital Library (TKDL) for Indian patent examination?",
        "expected_documents": ["guidelines_tk_biological_material_2012", "ayush_related_inventions_guidelines_2025"],
        "expected_provisions": ["Section 3(p)", "TKDL"],
        "expected_claims": [
            "TKDL references documented ancient Ayurvedic, Unani, and Siddha texts as searchable prior art for patent examiners to prevent wrongful patenting of traditional knowledge."
        ],
        "should_refuse": False
    },

    # -------------------------------------------------------------
    # Category E: Multilingual Queries (Hindi)
    # -------------------------------------------------------------
    {
        "id": "gen_q17",
        "category": "multilingual_hindi",
        "query": "अश्वगंधा (Withania somnifera) से संबंधित पेटेंट प्राप्त करने के मुख्य कानूनी नियम क्या हैं?",
        "expected_documents": ["patent_act_1970", "ayush_related_inventions_guidelines_2025", "biological_diversity_act_2002"],
        "expected_provisions": ["Section 3(p)", "Section 3(e)", "Section 6"],
        "expected_claims": [
            "अश्वगंधा के पारंपरिक उपयोगों पर पेटेंट धारा 3(p) के तहत वर्जित है।",
            "पेटेंट प्राप्त करने के लिए नवीन और अप्रत्याशित प्रभाव (synergistic efficacy) साबित करना आवश्यक है।",
            "भारतीय जैविक संसाधन का उपयोग करने पर राष्ट्रीय जैव विविधता प्राधिकरण (NBA) से पूर्व अनुमति अनिवार्य है।"
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q18",
        "category": "multilingual_hindi",
        "query": "पारंपरिक ज्ञान (Traditional Knowledge) को भारतीय पेटेंट कानून में कैसे परिभाषित और बहिष्कृत किया गया है?",
        "expected_documents": ["patent_act_1970", "guidelines_tk_biological_material_2012"],
        "expected_provisions": ["Section 3(p)"],
        "expected_claims": [
            "पेटेंट अधिनियम की धारा 3(p) के तहत पारंपरिक ज्ञान से संबंधित किसी भी खोज को पेटेंट योग्य आविष्कार नहीं माना जाता है।"
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q19",
        "category": "multilingual_hindi",
        "query": "एफएसएसएआई (FSSAI) के अनुसार आयुर्वेद आहार के नियम क्या हैं?",
        "expected_documents": ["fssai_ayurveda_aahara_regulations_2022", "order_fssai_ayurveda_aahara_schedules_2025"],
        "expected_provisions": ["Ayurveda Aahara"],
        "expected_claims": [
            "आयुर्वेद आहार केवल प्रामाणिक आयुर्वेदिक ग्रंथों में वर्णित नियमों और सामग्रियों के आधार पर तैयार किया जाना चाहिए।"
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q20",
        "category": "multilingual_hindi",
        "query": "राष्ट्रीय जैव विविधता प्राधिकरण (NBA) से अनुमति कब आवश्यक होती है?",
        "expected_documents": ["biological_diversity_act_2002"],
        "expected_provisions": ["Section 6"],
        "expected_claims": [
            "भारत के किसी जैविक संसाधन या उससे जुड़े पारंपरिक ज्ञान पर आधारित बौद्धिक संपदा अधिकार आवेदन के लिए धारा 6 के तहत पूर्व अनुमति आवश्यक है।"
        ],
        "should_refuse": False
    },

    # -------------------------------------------------------------
    # Category F: Code-Mixed (Hinglish) Queries
    # -------------------------------------------------------------
    {
        "id": "gen_q21",
        "category": "code_mixed_hinglish",
        "query": "Can an Ayurvedic formulation ko India mein patent kiya ja sakta hai?",
        "expected_documents": ["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        "expected_provisions": ["Section 3(p)", "Section 3(e)"],
        "expected_claims": [
            "Ayurvedic formulation ko direct traditional form mein patent nahi kiya ja sakta under Section 3(p).",
            "Patent lene ke liye novel inventive step, synergistic efficacy aur biological resource ke liye NBA permission mandatory hai."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q22",
        "category": "code_mixed_hinglish",
        "query": "Ayurvedic product par patent apply karne ke liye NBA permission kab zaroori hoti hai?",
        "expected_documents": ["biological_diversity_act_2002"],
        "expected_provisions": ["Section 6"],
        "expected_claims": [
            "Jab invention mein Indian biological resource ya usse associated traditional knowledge use hota hai, toh Section 6 of Biological Diversity Act ke tehat NBA approval zaroori hota hai."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q23",
        "category": "code_mixed_hinglish",
        "query": "Section 3(e) mere admixture objection ko kaise overcome karein herbal drugs ke liye?",
        "expected_documents": ["ayush_related_inventions_guidelines_2025", "patent_act_1970"],
        "expected_provisions": ["Section 3(e)"],
        "expected_claims": [
            "Section 3(e) objection ko overcome karne ke liye comparative experimental data se unexpected synergistic effect prove karna hota hai."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q24",
        "category": "code_mixed_hinglish",
        "query": "Ayurveda Aahara product packaging par logo aur statutory warnings ke kya rules hain?",
        "expected_documents": ["fssai_ayurveda_aahara_regulations_2022"],
        "expected_provisions": ["Ayurveda Aahara logo", "Labeling"],
        "expected_claims": [
            "Ayurveda Aahara products par designated logo lagana compulsory hai aur specific consumption warnings deni hoti hain."
        ],
        "should_refuse": False
    },

    # -------------------------------------------------------------
    # Category G: Cross-Domain Multi-Statutory Queries
    # -------------------------------------------------------------
    {
        "id": "gen_q25",
        "category": "cross_domain",
        "query": "Can an Ayurvedic invention using traditional knowledge and biological resources be patented in India, and what NBA approvals and patent disclosures are required?",
        "expected_documents": ["patent_act_1970", "biological_diversity_act_2002", "guidelines_tk_biological_material_2012", "ayush_related_inventions_guidelines_2025"],
        "expected_provisions": ["Section 3(p)", "Section 3(e)", "Section 6", "Section 10(4)(d)"],
        "expected_claims": [
            "The invention must not be a mere duplication of traditional knowledge (Section 3(p)) or mere admixture (Section 3(e)).",
            "The applicant must disclose the source and geographical origin of biological material under Section 10(4)(d) of the Patents Act.",
            "Prior approval of the National Biodiversity Authority is mandatory under Section 6 of the Biological Diversity Act, 2002 before the grant of the patent."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q26",
        "category": "cross_domain",
        "query": "Compare the Indian Patent Law Section 3(p) with the 2024 WIPO Treaty requirements for traditional knowledge and genetic resources.",
        "expected_documents": ["patent_act_1970", "wipo_gr_tk_treaty_2024", "guidelines_tk_biological_material_2012"],
        "expected_provisions": ["Section 3(p)", "Article 3"],
        "expected_claims": [
            "Section 3(p) of the Indian Patents Act is a substantive patentability exclusion disqualifying traditional knowledge.",
            "The WIPO 2024 Treaty establishes international disclosure requirements (Article 3) mandating applicants to declare the country of origin or indigenous source of genetic resources and associated TK."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q27",
        "category": "cross_domain",
        "query": "What is the legal difference between an Ayurvedic Drug under the Drugs and Cosmetics Act and an Ayurveda Aahara product under FSSAI regulations?",
        "expected_documents": ["fssai_ayurveda_aahara_regulations_2022", "drugs_and_cosmetics_act_1940", "gsr_669_e_drugs_rules_2024"],
        "expected_provisions": ["Drugs and Cosmetics Act", "Ayurveda Aahara Regulations"],
        "expected_claims": [
            "Ayurvedic drugs under the Drugs and Cosmetics Act are medicinal formulations intended for therapeutic diagnosis/treatment.",
            "Ayurveda Aahara under FSSAI covers food/dietary items prepared as per authoritative Ayurvedic texts, excluding synthetic medicinal drugs or parenteral products."
        ],
        "should_refuse": False
    },
    {
        "id": "gen_q28",
        "category": "cross_domain",
        "query": "What are the IP and regulatory steps to commercialize an Ayurvedic botanical formulation internationally via PCT and in India?",
        "expected_documents": ["pct_applicant_guide_international_phase", "patent_act_1970", "biological_diversity_act_2002", "ayush_related_inventions_guidelines_2025"],
        "expected_provisions": ["PCT International Phase", "Section 6", "Section 3(p)"],
        "expected_claims": [
            "File international patent application through PCT designating international search and examination authorities.",
            "Ensure compliance with Indian Section 3(p) and NBA Section 6 permissions for Indian biological resources.",
            "Provide proof of unexpected synergistic efficacy and source disclosure."
        ],
        "should_refuse": False
    },

    # -------------------------------------------------------------
    # Category H: Insufficient Evidence / Out-of-Scope (Refusal Test)
    # -------------------------------------------------------------
    {
        "id": "gen_q29",
        "category": "insufficient_evidence",
        "query": "What are the specific patent registration procedures in Brazil under the Brazilian 1996 Industrial Property Law (Law No. 9.279)?",
        "expected_documents": [],
        "expected_provisions": [],
        "expected_claims": [
            "I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."
        ],
        "should_refuse": True
    },
    {
        "id": "gen_q30",
        "category": "insufficient_evidence",
        "query": "What are the copyright registration fees for computer software in Japan under the Japanese Copyright Act of 1970?",
        "expected_documents": [],
        "expected_provisions": [],
        "expected_claims": [
            "I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."
        ],
        "should_refuse": True
    },
    {
        "id": "gen_q31",
        "category": "insufficient_evidence",
        "query": "What are the tax deduction percentages for electric vehicle purchases in California under the 2024 Clean Vehicle Rebate Program?",
        "expected_documents": [],
        "expected_provisions": [],
        "expected_claims": [
            "I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."
        ],
        "should_refuse": True
    },
    {
        "id": "gen_q32",
        "category": "insufficient_evidence",
        "query": "Explain the orbital mechanics of the Chandrayaan-3 lunar propulsion module during trans-lunar injection.",
        "expected_documents": [],
        "expected_provisions": [],
        "expected_claims": [
            "I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."
        ],
        "should_refuse": True
    },
    {
        "id": "gen_q33",
        "category": "insufficient_evidence",
        "query": "What is the corporate income tax rate for manufacturing companies in Germany under the Corporate Tax Act (KStG)?",
        "expected_documents": [],
        "expected_provisions": [],
        "expected_claims": [
            "I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."
        ],
        "should_refuse": True
    },
    {
        "id": "gen_q34",
        "category": "insufficient_evidence",
        "query": "What are the FAA airspace classification rules for commercial drone operations in Australia?",
        "expected_documents": [],
        "expected_provisions": [],
        "expected_claims": [
            "I could not find sufficient authoritative evidence in the available knowledge base to answer this conclusively."
        ],
        "should_refuse": True
    }
]
