"""Comprehensive Benchmark Dataset for Retrieval Evaluation (36 Canonical Queries).

Covers Indian Patent Law, AYUSH / Ayurveda, Traditional Knowledge, Biological Resources,
International IP (PCT, WIPO, EPO), Exact Legal & Taxon Lookups, Multilingual, and Cross-Domain.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkQuery:
    query_id: int
    category: str
    query: str
    description: str
    expected_documents: List[str]
    expected_sections: List[str] = field(default_factory=list)
    expected_articles: List[str] = field(default_factory=list)
    expected_rules: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    ground_truth_status: str = "verified"  # "verified" or "needs_manual_verification"
    notes: Optional[str] = None


BENCHMARK_QUERIES: List[BenchmarkQuery] = [
    # ==========================================
    # A. INDIAN PATENT LAW
    # ==========================================
    BenchmarkQuery(
        query_id=1,
        category="PATENT",
        query="What are the exclusions under Section 3 of the Indian Patents Act?",
        description="Core statutory exclusions under Section 3 of Patent Act 1970",
        expected_documents=["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        expected_sections=["3", "3(a)", "3(b)", "3(c)", "3(d)", "3(e)", "3(p)"],
        expected_keywords=["what are not inventions", "mere admixture", "traditional knowledge", "frivolous"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=2,
        category="PATENT",
        query="What is Section 3(p) of the Indian Patents Act?",
        description="Traditional knowledge exclusion clause in Indian Patents Act",
        expected_documents=["patent_act_1970", "ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012"],
        expected_sections=["3", "3(p)"],
        expected_keywords=["traditional knowledge", "aggregation or duplication", "known properties"],
        expected_chunk_ids=["ayush_related_inventions_guidelines_2025_p7_c6"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=3,
        category="PATENT",
        query="What constitutes an invention under Indian patent law?",
        description="Statutory definition of invention under Section 2(1)(j)",
        expected_documents=["patent_act_1970"],
        expected_sections=["2", "2(1)(j)", "2(1)(ja)"],
        expected_keywords=["invention", "new product or process", "inventive step", "industrial application"],
        expected_chunk_ids=["patent_act_1970_p7_c29"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=4,
        category="PATENT",
        query="What is the requirement under Section 3(d) regarding enhanced efficacy?",
        description="Section 3(d) exclusion of new forms of known substances without enhancement of known efficacy",
        expected_documents=["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        expected_sections=["3", "3(d)"],
        expected_keywords=["efficacy", "known substance", "enhancement of the known efficacy"],
        expected_chunk_ids=["patent_act_1970_p9_c33"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=5,
        category="PATENT",
        query="How does Section 3(e) distinguish mere admixture from synergistic combinations?",
        description="Section 3(e) substance obtained by mere admixture resulting only in aggregation of properties",
        expected_documents=["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        expected_sections=["3", "3(e)"],
        expected_keywords=["mere admixture", "aggregation", "synergistic", "properties"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=6,
        category="PATENT",
        query="What are the disclosure requirements for biological materials under Section 10(4) of the Patents Act?",
        description="Section 10(4)(ii)(D) mandatory disclosure of source and geographical origin of biological material",
        expected_documents=["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        expected_sections=["10", "10(4)"],
        expected_keywords=["biological material", "source and geographical origin", "specification"],
        expected_chunk_ids=["patent_act_1970_p12_c40"],
        ground_truth_status="verified",
    ),

    # ==========================================
    # B. AYURVEDA / AYUSH
    # ==========================================
    BenchmarkQuery(
        query_id=7,
        category="AYURVEDA",
        query="Can an Ayurvedic formulation be patented in India?",
        description="Patentability standards and statutory hurdles for Ayurvedic formulations",
        expected_documents=["ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012"],
        expected_sections=["3(p)", "3(e)", "3(d)"],
        expected_keywords=["ayurvedic formulation", "synergy", "mere admixture", "prior art", "tkdl"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=8,
        category="AYURVEDA",
        query="What examination guidelines apply to AYUSH and Ayurveda-related inventions?",
        description="Specific examination guidelines issued by IPO for AYUSH inventions in 2025",
        expected_documents=["ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012"],
        expected_keywords=["ayush", "novelty", "inventive step", "synergistic effect", "traditional knowledge", "guidelines"],
        expected_chunk_ids=["ayush_related_inventions_guidelines_2025_p1_c1"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=9,
        category="AYURVEDA",
        query="What is the role of the Traditional Knowledge Digital Library (TKDL) in patent examination?",
        description="TKDL database evidence as prior art in Indian and international patent offices",
        expected_documents=["ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012", "wipo_ip_gr_tk_tce_overview"],
        expected_keywords=["tkdl", "traditional knowledge digital library", "prior art", "ayurveda"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=10,
        category="AYURVEDA",
        query="What regulations govern Ayurveda Aahara food products under FSSAI?",
        description="FSSAI Ayurveda Aahara Regulations 2022 and authoritative text schedules",
        expected_documents=["fssai_ayurveda_aahara_regulations_2022", "order_fssai_ayurveda_aahara_schedules_2025"],
        expected_keywords=["ayurveda aahara", "fssai", "schedule", "expert committee", "labelling", "authoritative books"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=11,
        category="AYURVEDA",
        query="What are the advertising and claim restrictions for Ayurvedic and food products?",
        description="FSSAI Advertising and Claims Regulations 2022 preventing misleading health claims",
        expected_documents=["compendium_advertising_claims_regulations_2022", "fssai_ayurveda_aahara_regulations_2022"],
        expected_keywords=["advertising", "claims", "deceptive", "health claims", "disease risk reduction"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=12,
        category="AYURVEDA",
        query="What licensing and registration rules apply to manufacturers under FSSAI?",
        description="Compendium of Food Safety and Standards (Licensing and Registration of Food Businesses) Regulations",
        expected_documents=["compendium_licensing_regulations_2021", "drugs_and_cosmetics_act_1940"],
        expected_keywords=["licensing", "registration", "food business operator", "fbo", "license", "schedule 1"],
        ground_truth_status="verified",
    ),

    # ==========================================
    # C. TRADITIONAL KNOWLEDGE
    # ==========================================
    BenchmarkQuery(
        query_id=13,
        category="TRADITIONAL KNOWLEDGE",
        query="What is traditional knowledge in the context of intellectual property?",
        description="Conceptual definition, characteristics, and IP interface of traditional knowledge",
        expected_documents=["wipo_ip_gr_tk_tce_overview", "wipo_gr_tk_treaty_2024", "wipo_documenting_tk_toolkit"],
        expected_keywords=["traditional knowledge", "indigenous", "defensive protection", "positive protection"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=14,
        category="TRADITIONAL KNOWLEDGE",
        query="What is defensive protection versus positive protection of traditional knowledge?",
        description="Distinction between defensive mechanisms (preventing illegitimate patents) and positive rights",
        expected_documents=["wipo_ip_gr_tk_tce_overview", "wipo_documenting_tk_toolkit", "guidelines_tk_biological_material_2012"],
        expected_keywords=["defensive protection", "positive protection", "prior art", "sui generis"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=15,
        category="TRADITIONAL KNOWLEDGE",
        query="What is the relationship between genetic resources and associated traditional knowledge?",
        description="Interface between genetic resources, associated TK, and benefit sharing",
        expected_documents=["wipo_gr_tk_treaty_2024", "wipo_ip_gr_tk_tce_overview", "biological_diversity_act_2002"],
        expected_keywords=["genetic resources", "associated traditional knowledge", "mandatory disclosure", "benefit sharing"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=16,
        category="TRADITIONAL KNOWLEDGE",
        query="What guidelines exist for documenting traditional knowledge?",
        description="WIPO toolkit on documenting traditional knowledge and safeguarding community rights",
        expected_documents=["wipo_documenting_tk_toolkit", "wipo_ip_gr_tk_tce_overview"],
        expected_keywords=["documenting traditional knowledge", "prior informed consent", "intellectual property toolkit"],
        ground_truth_status="verified",
    ),

    # ==========================================
    # D. BIOLOGICAL RESOURCES
    # ==========================================
    BenchmarkQuery(
        query_id=17,
        category="BIOLOGICAL RESOURCES",
        query="What provisions govern the use of biological resources under the Biological Diversity Act 2002?",
        description="Core statutory requirements of Biological Diversity Act 2002 regarding access and approvals",
        expected_documents=["biological_diversity_act_2002", "guidelines_tk_biological_material_2012"],
        expected_sections=["3", "4", "6", "19", "20"],
        expected_keywords=["biological resources", "national biodiversity authority", "nba", "approval"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=18,
        category="BIOLOGICAL RESOURCES",
        query="When is National Biodiversity Authority (NBA) approval required before applying for an intellectual property right?",
        description="Section 6 of Biological Diversity Act 2002 mandating prior NBA approval for IP applications",
        expected_documents=["biological_diversity_act_2002", "guidelines_tk_biological_material_2012", "patent_act_1970"],
        expected_sections=["6", "19"],
        expected_keywords=["section 6", "intellectual property", "prior approval", "national biodiversity authority", "biological resource"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=19,
        category="BIOLOGICAL RESOURCES",
        query="What is Access and Benefit Sharing (ABS) under Indian biodiversity law?",
        description="Fair and equitable sharing of benefits arising from the utilization of biological resources",
        expected_documents=["biological_diversity_act_2002", "wipo_ip_gr_tk_tce_overview"],
        expected_sections=["21", "19", "20"],
        expected_keywords=["benefit sharing", "equitable sharing", "national biodiversity authority", "local bodies"],
        ground_truth_status="verified",
    ),

    # ==========================================
    # E. INTERNATIONAL IP
    # ==========================================
    BenchmarkQuery(
        query_id=20,
        category="INTERNATIONAL",
        query="What is the PCT international phase procedure?",
        description="PCT International Phase filing, search, publication, and preliminary examination",
        expected_documents=["pct_applicant_guide_international_phase", "epo_pct_guidelines_2026"],
        expected_keywords=["international phase", "receiving office", "international searching authority", "wipo", "pct"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=21,
        category="INTERNATIONAL",
        query="What is Article 3 of the 2024 WIPO Treaty on Genetic Resources and Associated Traditional Knowledge?",
        description="Mandatory disclosure requirement for patent applications based on genetic resources or associated TK",
        expected_documents=["wipo_gr_tk_treaty_2024"],
        expected_articles=["3"],
        expected_keywords=["article 3", "mandatory disclosure requirement", "country of origin", "indigenous peoples"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=22,
        category="INTERNATIONAL",
        query="What are the patentability examination principles under EPO Guidelines?",
        description="EPO Guidelines for Examination regarding novelty, inventive step, and problem-solution approach",
        expected_documents=["epo_guidelines_for_examination_2026", "epo_pct_guidelines_2026"],
        expected_keywords=["state of the art", "inventive step", "problem-solution approach", "examination"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=23,
        category="INTERNATIONAL",
        query="What is PCT Rule 43bis regarding the written opinion of the International Searching Authority?",
        description="PCT Rule 43bis governing the written opinion on novelty, inventive step, and industrial applicability",
        expected_documents=["pct_applicant_guide_international_phase", "epo_pct_guidelines_2026"],
        expected_rules=["43bis", "Rule 43bis"],
        expected_keywords=["rule 43bis", "written opinion", "international searching authority", "isa"],
        ground_truth_status="verified",
    ),

    # ==========================================
    # F. EXACT LOOKUP QUERIES
    # ==========================================
    BenchmarkQuery(
        query_id=24,
        category="EXACT_LOOKUP",
        query="Section 3(p)",
        description="Exact statutory citation for Section 3(p) of the Patents Act",
        expected_documents=["patent_act_1970", "ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012"],
        expected_sections=["3", "3(p)"],
        expected_keywords=["traditional knowledge", "aggregation", "properties"],
        expected_chunk_ids=["ayush_related_inventions_guidelines_2025_p7_c6"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=25,
        category="EXACT_LOOKUP",
        query="Article 3",
        description="Exact treaty article citation for Article 3 (WIPO Treaty / Treaties)",
        expected_documents=["wipo_gr_tk_treaty_2024", "pct_applicant_guide_international_phase", "epo_guidelines_for_examination_2026"],
        expected_articles=["3"],
        expected_keywords=["article 3", "disclosure requirement"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=26,
        category="EXACT_LOOKUP",
        query="PCT Rule 43bis",
        description="Exact procedural rule citation for PCT Rule 43bis",
        expected_documents=["pct_applicant_guide_international_phase", "epo_pct_guidelines_2026"],
        expected_rules=["43bis", "Rule 43bis"],
        expected_keywords=["rule 43bis", "written opinion"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=27,
        category="EXACT_LOOKUP",
        query="Withania somnifera",
        description="Exact botanical binomial lookup for Ashwagandha prior art and patent exam cases",
        expected_documents=["ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012"],
        expected_keywords=["withania somnifera", "ashwagandha"],
        expected_chunk_ids=["ayush_related_inventions_guidelines_2025_p10_c9", "guidelines_tk_biological_material_2012_p5_c7"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=28,
        category="EXACT_LOOKUP",
        query="Patent No. 429737",
        description="Exact Indian patent number lookup cited in AYUSH patent guidelines",
        expected_documents=["ayush_related_inventions_guidelines_2025"],
        expected_keywords=["429737", "patent no. 429737"],
        expected_chunk_ids=["ayush_related_inventions_guidelines_2025_p1_c1"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=29,
        category="EXACT_LOOKUP",
        query="Section 10(4)",
        description="Exact statutory citation for Section 10(4) of Indian Patents Act",
        expected_documents=["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        expected_sections=["10", "10(4)"],
        expected_keywords=["section 10", "specification", "biological material"],
        expected_chunk_ids=["patent_act_1970_p12_c40"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=30,
        category="EXACT_LOOKUP",
        query="Curcuma longa",
        description="Exact botanical taxon lookup for Turmeric prior art cases",
        expected_documents=["ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012"],
        expected_keywords=["curcuma", "turmeric", "haldi", "rhizome"],
        ground_truth_status="verified",
    ),

    # ==========================================
    # G. MULTILINGUAL QUERIES (Hindi & Code-Mixed)
    # ==========================================
    BenchmarkQuery(
        query_id=31,
        category="MULTILINGUAL",
        query="क्या आयुर्वेदिक formulation को भारत में patent किया जा सकता है?",
        description="Hindi query regarding patentability of Ayurvedic formulations under Indian law",
        expected_documents=["ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012", "patent_act_1970"],
        expected_sections=["3(p)", "3(e)"],
        expected_keywords=["ayurvedic", "formulation", "patent", "traditional knowledge", "synergy"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=32,
        category="MULTILINGUAL",
        query="अश्वगंधा के patent से संबंधित प्रावधान क्या हैं?",
        description="Hindi query on patent provisions and prior art relating to Ashwagandha (Withania somnifera)",
        expected_documents=["ayush_related_inventions_guidelines_2025", "guidelines_tk_biological_material_2012"],
        expected_keywords=["withania somnifera", "ashwagandha", "patent", "traditional knowledge"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=33,
        category="MULTILINGUAL",
        query="Section 3(p) traditional knowledge से कैसे संबंधित है?",
        description="Code-mixed Hindi-English query on Section 3(p) relationship with traditional knowledge",
        expected_documents=["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        expected_sections=["3", "3(p)"],
        expected_keywords=["section 3(p)", "traditional knowledge", "aggregation"],
        expected_chunk_ids=["ayush_related_inventions_guidelines_2025_p7_c6"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=34,
        category="MULTILINGUAL",
        query="आयुर्वेद आहार के लिए FSSAI के क्या नियम हैं?",
        description="Hindi query regarding FSSAI regulations for Ayurveda Aahara",
        expected_documents=["fssai_ayurveda_aahara_regulations_2022", "order_fssai_ayurveda_aahara_schedules_2025"],
        expected_keywords=["ayurveda aahara", "fssai", "regulations", "schedules"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=35,
        category="MULTILINGUAL",
        query="जैविक विविधता अधिनियम के तहत NBA की अनुमति कब आवश्यक है?",
        description="Hindi query regarding when NBA approval is required under Biological Diversity Act",
        expected_documents=["biological_diversity_act_2002", "guidelines_tk_biological_material_2012"],
        expected_sections=["6", "19"],
        expected_keywords=["national biodiversity authority", "biological resource", "approval", "intellectual property"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=36,
        category="MULTILINGUAL",
        query="पारंपरिक ज्ञान (Traditional Knowledge) का defensive protection क्या होता है?",
        description="Code-mixed query on defensive protection of traditional knowledge",
        expected_documents=["wipo_ip_gr_tk_tce_overview", "wipo_documenting_tk_toolkit", "guidelines_tk_biological_material_2012"],
        expected_keywords=["defensive protection", "prior art", "traditional knowledge", "tkdl"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=37,
        category="MULTILINGUAL",
        query="Indian Patents Act में Section 3(d) enhancement of efficacy की क्या शर्तें हैं?",
        description="Code-mixed query on Section 3(d) enhanced efficacy requirements in Indian Patent Act",
        expected_documents=["patent_act_1970", "ayush_related_inventions_guidelines_2025"],
        expected_sections=["3", "3(d)"],
        expected_keywords=["known substance", "enhancement of the known efficacy", "efficacy"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=38,
        category="MULTILINGUAL",
        query="WIPO GR/TK Treaty के Article 3 में mandatory disclosure की क्या व्यवस्था है?",
        description="Code-mixed query on mandatory disclosure requirement in Article 3 of WIPO Treaty",
        expected_documents=["wipo_gr_tk_treaty_2024"],
        expected_articles=["3"],
        expected_keywords=["mandatory disclosure", "genetic resources", "traditional knowledge", "country of origin"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=39,
        category="MULTILINGUAL",
        query="हल्दी (Curcuma longa) के patent revocation केस में TKDL की क्या भूमिका थी?",
        description="Code-mixed query on TKDL role in Turmeric patent revocation case",
        expected_documents=["guidelines_tk_biological_material_2012", "wipo_ip_gr_tk_tce_overview"],
        expected_keywords=["turmeric", "curcuma", "prior art", "tkdl", "wound healing"],
        ground_truth_status="verified",
    ),
    BenchmarkQuery(
        query_id=40,
        category="MULTILINGUAL",
        query="Ayurveda Aahara formulations के लिए authoritative texts की क्या सूची है?",
        description="Code-mixed query on list of authoritative texts for Ayurveda Aahara formulations",
        expected_documents=["fssai_ayurveda_aahara_regulations_2022", "order_fssai_ayurveda_aahara_schedules_2025"],
        expected_keywords=["authoritative books", "schedule", "first schedule", "drugs and cosmetics"],
        ground_truth_status="verified",
    ),

    # ==========================================
    # H. CROSS-DOMAIN & COMPLEX RETRIEVAL
    # ==========================================
    BenchmarkQuery(
        query_id=41,
        category="CROSS_DOMAIN",
        query="Can an Ayurvedic formulation using a traditionally known plant be patented in India?",
        description="Cross-domain query spanning Indian Patent Act Sec 3(p)/3(e), AYUSH Guidelines, and Biological Diversity Act NBA approval",
        expected_documents=[
            "patent_act_1970",
            "ayush_related_inventions_guidelines_2025",
            "biological_diversity_act_2002",
            "guidelines_tk_biological_material_2012"
        ],
        expected_sections=["3(p)", "3(e)", "6"],
        expected_keywords=["traditional knowledge", "synergy", "biological resource", "nba", "patent"],
        ground_truth_status="verified",
        notes="Requires cross-domain evidence from Patents Act, AYUSH guidelines, and Biodiversity Act",
    ),
    BenchmarkQuery(
        query_id=42,
        category="CROSS_DOMAIN",
        query="What are the disclosure requirements for patent applications based on Indian genetic resources and traditional knowledge?",
        description="Cross-domain query spanning Patents Act Section 10(4), Biodiversity Act Section 6, and WIPO GR/TK Treaty Article 3",
        expected_documents=[
            "patent_act_1970",
            "biological_diversity_act_2002",
            "wipo_gr_tk_treaty_2024",
            "ayush_related_inventions_guidelines_2025"
        ],
        expected_sections=["10(4)", "6"],
        expected_articles=["3"],
        expected_keywords=["disclosure requirement", "source and geographical origin", "genetic resources", "traditional knowledge"],
        ground_truth_status="verified",
        notes="Requires domestic disclosure (Sec 10(4) / Sec 6 BDA) and international treaty compliance (Art 3 WIPO Treaty)",
    ),
]

