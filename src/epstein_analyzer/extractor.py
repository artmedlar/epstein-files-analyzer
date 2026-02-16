"""PDF text extraction with PyMuPDF + Tesseract OCR fallback.

Extract-and-discard pipeline: download PDF -> extract text -> delete PDF.
"""

import logging
from pathlib import Path
from typing import Optional

from .config import (
    TEXT_DIR, OCR_MIN_TEXT_LENGTH,
    OCR_ENGINE_MODE, OCR_PAGE_SEG_MODE,
    OCR_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str | Path) -> dict:
    """
    Extract text from a PDF file using PyMuPDF with Tesseract OCR fallback.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        dict with keys:
            text: str - full extracted text
            page_count: int
            ocr_needed: bool - whether any pages required OCR
            ocr_confidence: float - average OCR confidence (0-100)
            pages: list[dict] - per-page info
    """
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []
    full_text_parts = []
    ocr_needed = False
    confidence_scores = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_info = {"page_num": page_num + 1, "ocr_used": False, "confidence": 100.0}

        # Try native text extraction first
        text = page.get_text("text").strip()

        # If text is too short or looks like garbage, try OCR
        if len(text) < OCR_MIN_TEXT_LENGTH or _is_garbage_text(text):
            ocr_text, confidence = _ocr_page(page)
            if ocr_text and len(ocr_text) > len(text):
                text = ocr_text
                page_info["ocr_used"] = True
                page_info["confidence"] = confidence
                ocr_needed = True
                confidence_scores.append(confidence)

        if text:
            full_text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        page_info["text_length"] = len(text)
        pages.append(page_info)

    doc.close()

    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores else 100.0
    )

    return {
        "text": "\n\n".join(full_text_parts),
        "page_count": len(pages),
        "ocr_needed": ocr_needed,
        "ocr_confidence": round(avg_confidence, 1),
        "pages": pages,
    }


def _is_garbage_text(text: str) -> bool:
    """Check if extracted text looks like garbage (random symbols, not real text)."""
    if not text:
        return True
    # If less than 50% of characters are alphanumeric or spaces, it's probably garbage
    alnum_count = sum(1 for c in text if c.isalnum() or c.isspace())
    return alnum_count / len(text) < 0.5


def _ocr_page(page) -> tuple[str, float]:
    """
    OCR a single PDF page using Tesseract.

    Args:
        page: A PyMuPDF page object

    Returns:
        (extracted_text, confidence_score)
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render page to image at 300 DPI for good OCR quality
        mat = page.get_pixmap(matrix=page.derotation_matrix, dpi=300)
        img_data = mat.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        # Run OCR with confidence data
        config = f"--oem {OCR_ENGINE_MODE} --psm {OCR_PAGE_SEG_MODE}"

        # Get detailed data including confidence
        data = pytesseract.image_to_data(
            image, config=config, output_type=pytesseract.Output.DICT
        )

        # Calculate average confidence (excluding -1 which means no text detected)
        confidences = [c for c in data["conf"] if c > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Get the text
        text = pytesseract.image_to_string(image, config=config).strip()

        # If first pass gives poor results, try fully automatic segmentation
        if avg_confidence < OCR_CONFIDENCE_THRESHOLD and text:
            alt_config = f"--oem {OCR_ENGINE_MODE} --psm 3"
            alt_text = pytesseract.image_to_string(image, config=alt_config).strip()
            alt_data = pytesseract.image_to_data(
                image, config=alt_config, output_type=pytesseract.Output.DICT
            )
            alt_conf = [c for c in alt_data["conf"] if c > 0]
            alt_avg = sum(alt_conf) / len(alt_conf) if alt_conf else 0.0

            if alt_avg > avg_confidence:
                text = alt_text
                avg_confidence = alt_avg

        return text, avg_confidence

    except ImportError:
        logger.warning("pytesseract not available, skipping OCR")
        return "", 0.0
    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return "", 0.0


def extract_and_save(pdf_path: str | Path, filename: str) -> dict:
    """
    Extract text from a PDF and save it to the text directory.

    Args:
        pdf_path: Path to the (temporary) PDF file
        filename: Original filename (used to create text filename)

    Returns:
        dict with extraction results plus text_path
    """
    result = extract_text_from_pdf(pdf_path)

    # Save extracted text
    text_filename = Path(filename).stem + ".txt"
    text_path = TEXT_DIR / text_filename
    text_path.write_text(result["text"], encoding="utf-8")

    result["text_path"] = text_filename
    logger.info(
        f"Extracted {filename}: {result['page_count']} pages, "
        f"OCR={'yes' if result['ocr_needed'] else 'no'}, "
        f"confidence={result['ocr_confidence']:.1f}"
    )

    return result


def process_downloaded_documents(download_results: list[dict],
                                 progress_callback=None) -> list[dict]:
    """
    Extract text from all downloaded PDFs and save to text directory.
    Deletes PDFs after successful extraction.

    Args:
        download_results: Output from downloader.download_documents()
        progress_callback: Optional callable for progress updates

    Returns:
        List of extraction result dicts
    """
    from .models import SearchProgress

    successful = [r for r in download_results if r["success"] and r["local_path"]]
    total = len(successful)
    extraction_results = []

    for i, dl in enumerate(successful):
        if progress_callback:
            progress_callback(SearchProgress(
                stage="extracting",
                current=i + 1, total=total,
                message=f"Extracting text: {dl['filename']} ({i+1}/{total})"
            ))

        try:
            result = extract_and_save(dl["local_path"], dl["filename"])
            result["url"] = dl["url"]
            result["filename"] = dl["filename"]
            result["file_hash"] = dl.get("file_hash")
            result["data_set"] = dl.get("data_set")
            extraction_results.append(result)

            # Delete the PDF after successful extraction
            pdf_path = Path(dl["local_path"])
            if pdf_path.exists():
                pdf_path.unlink()
                logger.debug(f"Deleted temporary PDF: {dl['filename']}")

        except Exception as e:
            logger.error(f"Extraction failed for {dl['filename']}: {e}")
            extraction_results.append({
                "url": dl["url"],
                "filename": dl["filename"],
                "error": str(e),
            })

    logger.info(
        f"Extraction complete: {len(extraction_results)} documents processed"
    )
    return extraction_results
