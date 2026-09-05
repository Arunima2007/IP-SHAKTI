"""Modular OCR processing module supporting Apple Vision, PaddleOCR, and Tesseract engines."""
import io
import logging
import os
from typing import Dict, List, Optional, Tuple
import pymupdf
from PIL import Image

logger = logging.getLogger(__name__)

# Backend selection: auto, ocrmac, paddleocr, tesseract
OCR_BACKEND = os.getenv("OCR_BACKEND", "auto")

# Detect available OCR engines
HAS_OCRMAC = False
try:
    from ocrmac import ocrmac
    HAS_OCRMAC = True
except ImportError:
    pass

HAS_PADDLEOCR = False
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    pass

HAS_TESSERACT = False
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    pass


def extract_page_with_ocr(page: pymupdf.Page, dpi: int = 150) -> str:
    """Extract text from a single PDF page using the best available configured OCR backend."""
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    
    # 1. macOS Apple Vision (High speed on Apple Silicon/Mac)
    if (OCR_BACKEND in ("auto", "ocrmac")) and HAS_OCRMAC:
        try:
            annotations = ocrmac.OCR(img).recognize()
            # Sort vertically then horizontally to maintain reading order
            annotations_sorted = sorted(annotations, key=lambda a: (round(-a[2][1], 2), a[2][0])) if len(annotations) > 0 and len(annotations[0]) > 2 else annotations
            text_lines = [item[0] for item in annotations_sorted]
            return "\n".join(text_lines).strip()
        except Exception as e:
            logger.warning(f"Apple Vision OCR failed on page {page.number + 1}: {e}")
            
    # 2. PaddleOCR (Production Linux/Cloud Engine)
    if (OCR_BACKEND in ("auto", "paddleocr")) and HAS_PADDLEOCR:
        try:
            ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
            result = ocr_engine.ocr(img_bytes, cls=True)
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    text_lines.append(line[1][0])
            return "\n".join(text_lines).strip()
        except Exception as e:
            logger.warning(f"PaddleOCR failed on page {page.number + 1}: {e}")
            
    # 3. Tesseract Fallback
    if (OCR_BACKEND in ("auto", "tesseract")) and HAS_TESSERACT:
        try:
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            logger.warning(f"Tesseract OCR failed on page {page.number + 1}: {e}")
            
    logger.error(f"No functional OCR engine available for page {page.number + 1}")
    return ""


def should_ocr_page(page: pymupdf.Page, min_char_threshold: int = 50) -> bool:
    """Determine if a page is scanned and requires OCR."""
    raw_text = page.get_text().strip()
    if len(raw_text) >= min_char_threshold:
        return False
    images = page.get_images()
    return len(images) > 0


def assess_ocr_quality(text: str) -> Dict:
    """Assess extracted OCR text quality metrics (confidence heuristics)."""
    if not text:
        return {"confidence": 0.0, "status": "empty", "word_count": 0, "non_ascii_ratio": 0.0}
        
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return {"confidence": 0.0, "status": "empty", "word_count": 0, "non_ascii_ratio": 0.0}
        
    # Heuristics: average word length, printable character ratio, garble score
    printable_chars = sum(1 for c in text if c.isprintable())
    printable_ratio = printable_chars / max(1, len(text))
    
    # Non-alphanumeric/non-space ratio
    symbol_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in ".,-()[]/:;%—–")
    symbol_ratio = symbol_chars / max(1, len(text))
    
    garbled = symbol_ratio > 0.15 or printable_ratio < 0.85
    
    confidence = 1.0 - (symbol_ratio * 2.0)
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        "confidence": round(confidence, 2),
        "status": "low_confidence" if garbled else "good",
        "word_count": word_count,
        "symbol_ratio": round(symbol_ratio, 3),
    }
