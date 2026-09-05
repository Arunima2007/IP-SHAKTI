"""Structure detection module for legislation, guidelines, treaties, and technical documents."""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StructuralElement:
    """Represents a structural block in a legal or regulatory document."""
    element_type: str  # 'section', 'article', 'rule', 'chapter', 'part', 'guideline', 'regulation', 'schedule', 'illustration', 'patent_case', 'page_block'
    identifier: str    # e.g. '3', '3(p)', '8', '43bis', 'Chapter II', 'Guideline 4', 'Illustration 1'
    heading: str       # e.g. 'What are not inventions', 'List of Terms', 'Assessment of Novelty'
    subheading: Optional[str] = None
    part: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    clause: Optional[str] = None
    article: Optional[str] = None
    rule: Optional[str] = None
    paragraph: Optional[str] = None
    guideline: Optional[str] = None
    regulation: Optional[str] = None
    schedule: Optional[str] = None
    page: int = 1
    end_page: int = 1
    content: str = ""
    sub_elements: List["StructuralElement"] = field(default_factory=list)
    
    # Patent-specific fields for guideline examples
    patent_number: Optional[str] = None
    applicant: Optional[str] = None
    inventor: Optional[str] = None


# Regex Patterns for Legal & Regulatory Hierarchies
RE_CHAPTER = re.compile(r"^(?:CHAPTER|Chapter)\s+([IVXLCDM\d]+)\s*(?:[\n—\-:]+\s*([^\n]+))?", re.MULTILINE)
RE_PART = re.compile(r"^(?:PART|Part)\s+([IVXLCDM\d]+|[A-H]|General\s+Part)\s*(?:[\n—\-:]+\s*([^\n]+))?", re.MULTILINE)
RE_SCHEDULE = re.compile(r"^(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|[IVXLCDM\d]+)?\s*SCHEDULE\s*(?:[—\-:]+\s*([^\n]+))?", re.IGNORECASE | re.MULTILINE)

# Sections in Indian Acts / Rules: e.g. "3. What are not inventions.—" or "Section 3(p)"
RE_SECTION_START = re.compile(
    r"^(?:(?:SECTION|Section)\s+)?(\d+[A-Z]?)\.\s+(?!(?:Ins|Subs|Omitted|Added|See|Vide|w\.e\.f|Rep)\b)([A-Z][a-zA-Z\s,–\-\(\)\'\"]+?)(?:[.—–\-]|(?=\n))",
    re.MULTILINE
)
RE_SECTION_INLINE = re.compile(r"^(?:Section|SECTION)\s+(\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?)\s*:\s*([^\n]+)", re.MULTILINE)

# Articles in Treaties/WIPO/EPO: e.g. "ARTICLE 3" or "Article 8"
RE_ARTICLE = re.compile(r"^(?:ARTICLE|Article)\s+(\d+[A-Za-z]?)\s*(?:[\n—\-:]+\s*([^\n]+))?", re.MULTILINE)

# Rules in PCT / Patent Rules / EPO: e.g. "Rule 43bis", "Rule 33.1", "Rule 5"
RE_RULE = re.compile(r"^(?:Rule|RULE)\s+(\d+(?:bis|ter|quater)?(?:\.\d+)?)\s*(?:[\n—\-:]+\s*([^\n]+))?", re.MULTILINE)

# Guidelines in AYUSH / Examination Guidelines: e.g. "Guideline 3:" or "3.4 Guiding principles"
RE_GUIDELINE = re.compile(r"^(?:Guideline|GUIDELINE|Guiding\s+Principle)\s+(\d+(?:\.\d+)?)\s*(?:[\n—\-:]+\s*([^\n]+))?", re.MULTILINE)
RE_REGULATION = re.compile(r"^(?:Regulation|REGULATION)\s+(\d+[A-Z]?)\s*(?:[\n—\-:]+\s*([^\n]+))?", re.MULTILINE)

# Illustrations / Examples in AYUSH Guidelines
RE_ILLUSTRATION = re.compile(r"^(?:Illustration|Example)\s+(\d+)\s*:\s*([^\n]+)", re.MULTILINE)

# Patent Grants in AYUSH Guidelines: e.g. "Patent No.: 429737"
RE_PATENT_NO = re.compile(r"Patent\s+No\.?:\s*(\d+)", re.IGNORECASE)
RE_APPLICANT = re.compile(r"Applicant:\s*([^\n]+)", re.IGNORECASE)
RE_INVENTOR = re.compile(r"Inventor:\s*([^\n]+)", re.IGNORECASE)


class DocumentStructureParser:
    """Parses raw document text into hierarchical legal/regulatory units."""

    @staticmethod
    def parse_document(
        pages_text: List[Dict],
        document_id: str,
        category: str,
        document_type: str,
        jurisdiction: str,
    ) -> List[StructuralElement]:
        """Parse structured elements across all pages of a document."""
        elements: List[StructuralElement] = []
        
        current_part = None
        current_chapter = None
        current_schedule = None
        
        # In EPO Guidelines, default part based on document structure
        if "epo" in document_id:
            current_part = "General Part"
            
        for p in pages_text:
            page_num = p["page_number"]
            text = p["text"]
            
            lines = text.split("\n")
            current_element: Optional[StructuralElement] = None
            current_buffer = []
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    current_buffer.append(line)
                    continue
                
                # Check for Part header
                part_match = RE_PART.match(line_stripped)
                if part_match:
                    p_num = part_match.group(1)
                    p_title = (part_match.group(2) or "").strip()
                    current_part = f"Part {p_num}" + (f": {p_title}" if p_title else "")
                
                # Check for Chapter header
                ch_match = RE_CHAPTER.match(line_stripped)
                if ch_match:
                    ch_num = ch_match.group(1)
                    ch_title = (ch_match.group(2) or "").strip()
                    current_chapter = f"Chapter {ch_num}" + (f": {ch_title}" if ch_title else "")
                    
                # Check for Schedule
                sched_match = RE_SCHEDULE.match(line_stripped)
                if sched_match:
                    sched_title = (sched_match.group(1) or "").strip()
                    current_schedule = f"Schedule" + (f": {sched_title}" if sched_title else "")
                
                # Check specific structure matches
                sec_match = RE_SECTION_START.match(line_stripped)
                sec_inline_match = RE_SECTION_INLINE.match(line_stripped)
                art_match = RE_ARTICLE.match(line_stripped)
                rule_match = RE_RULE.match(line_stripped)
                guide_match = RE_GUIDELINE.match(line_stripped)
                reg_match = RE_REGULATION.match(line_stripped)
                illus_match = RE_ILLUSTRATION.match(line_stripped)
                
                # Check for Patent Grant Examples (e.g. in AYUSH guidelines page 2)
                pat_match = RE_PATENT_NO.search(line_stripped)
                
                if sec_match and document_type in ("act_legislation", "regulations", "compendium", "rules_amendment"):
                    sec_num = sec_match.group(1)
                    sec_heading = sec_match.group(2).strip()
                    
                    if current_element:
                        current_element.content = "\n".join(current_buffer).strip()
                        current_element.end_page = page_num
                        elements.append(current_element)
                        current_buffer = []
                        
                    current_element = StructuralElement(
                        element_type="section",
                        identifier=sec_num,
                        heading=sec_heading,
                        part=current_part,
                        chapter=current_chapter,
                        section=sec_num,
                        schedule=current_schedule,
                        page=page_num,
                        end_page=page_num,
                    )
                    current_buffer.append(line)
                    
                elif sec_inline_match:
                    sec_num = sec_inline_match.group(1).strip()
                    sec_heading = sec_inline_match.group(2).strip()
                    
                    # Extract subsection/clause if present e.g. "3(p)" -> section "3", clause "(p)"
                    clause_match = re.search(r"\(([a-z\d]+)\)", sec_num)
                    clause_val = clause_match.group(1) if clause_match else None
                    base_sec = re.split(r"[\(\[]", sec_num)[0]
                    
                    if current_element:
                        current_element.content = "\n".join(current_buffer).strip()
                        current_element.end_page = page_num
                        elements.append(current_element)
                        current_buffer = []
                        
                    current_element = StructuralElement(
                        element_type="section",
                        identifier=sec_num,
                        heading=f"Section {sec_num}: {sec_heading}",
                        part=current_part,
                        chapter=current_chapter,
                        section=base_sec,
                        clause=clause_val,
                        page=page_num,
                        end_page=page_num,
                    )
                    current_buffer.append(line)
                    
                elif art_match and jurisdiction in ("International", "WIPO/PCT", "EPO"):
                    art_num = art_match.group(1)
                    art_heading = (art_match.group(2) or "").strip()
                    
                    if current_element:
                        current_element.content = "\n".join(current_buffer).strip()
                        current_element.end_page = page_num
                        elements.append(current_element)
                        current_buffer = []
                        
                    current_element = StructuralElement(
                        element_type="article",
                        identifier=art_num,
                        heading=art_heading or f"Article {art_num}",
                        part=current_part,
                        chapter=current_chapter,
                        article=art_num,
                        page=page_num,
                        end_page=page_num,
                    )
                    current_buffer.append(line)
                    
                elif rule_match:
                    rule_num = rule_match.group(1)
                    rule_heading = (rule_match.group(2) or "").strip()
                    
                    if current_element:
                        current_element.content = "\n".join(current_buffer).strip()
                        current_element.end_page = page_num
                        elements.append(current_element)
                        current_buffer = []
                        
                    current_element = StructuralElement(
                        element_type="rule",
                        identifier=rule_num,
                        heading=rule_heading or f"Rule {rule_num}",
                        part=current_part,
                        chapter=current_chapter,
                        rule=rule_num,
                        page=page_num,
                        end_page=page_num,
                    )
                    current_buffer.append(line)
                    
                elif guide_match or illus_match:
                    if illus_match:
                        ident = f"Illustration {illus_match.group(1)}"
                        hdg = illus_match.group(2).strip()
                        elem_type = "illustration"
                    else:
                        ident = f"Guideline {guide_match.group(1)}"
                        hdg = (guide_match.group(2) or "").strip()
                        elem_type = "guideline"
                    
                    if current_element:
                        current_element.content = "\n".join(current_buffer).strip()
                        current_element.end_page = page_num
                        elements.append(current_element)
                        current_buffer = []
                        
                    current_element = StructuralElement(
                        element_type=elem_type,
                        identifier=ident,
                        heading=hdg or ident,
                        part=current_part,
                        chapter=current_chapter,
                        guideline=ident,
                        page=page_num,
                        end_page=page_num,
                    )
                    current_buffer.append(line)
                    
                elif reg_match and document_type in ("regulations", "compendium"):
                    reg_num = reg_match.group(1)
                    reg_heading = (reg_match.group(2) or "").strip()
                    
                    if current_element:
                        current_element.content = "\n".join(current_buffer).strip()
                        current_element.end_page = page_num
                        elements.append(current_element)
                        current_buffer = []
                        
                    current_element = StructuralElement(
                        element_type="regulation",
                        identifier=reg_num,
                        heading=reg_heading or f"Regulation {reg_num}",
                        part=current_part,
                        chapter=current_chapter,
                        regulation=reg_num,
                        page=page_num,
                        end_page=page_num,
                    )
                    current_buffer.append(line)
                    
                else:
                    # Check for patent details in AYUSH examples
                    if pat_match and current_element:
                        current_element.patent_number = pat_match.group(1)
                    app_match = RE_APPLICANT.search(line_stripped)
                    if app_match and current_element:
                        current_element.applicant = app_match.group(1).strip()
                    inv_match = RE_INVENTOR.search(line_stripped)
                    if inv_match and current_element:
                        current_element.inventor = inv_match.group(1).strip()
                        
                    current_buffer.append(line)
                    
            if current_element:
                current_element.content = "\n".join(current_buffer).strip()
                current_element.end_page = page_num
                elements.append(current_element)
            elif current_buffer:
                fallback_elem = StructuralElement(
                    element_type="page_block",
                    identifier=f"p{page_num}",
                    heading=current_chapter or current_part or f"Page {page_num}",
                    part=current_part,
                    chapter=current_chapter,
                    schedule=current_schedule,
                    page=page_num,
                    end_page=page_num,
                    content="\n".join(current_buffer).strip(),
                )
                if fallback_elem.content:
                    elements.append(fallback_elem)
                    
        return elements
