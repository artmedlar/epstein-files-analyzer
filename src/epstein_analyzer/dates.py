"""Extract document dates from text using multiple simple regex patterns.

Each pattern targets a common date format found in legal documents:
letterheads, filing stamps, email headers, deposition transcripts, etc.

Returns (date_str, confidence) where date_str is YYYY-MM-DD and
confidence is 'high' or 'medium'.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

# Patterns are tried in order.  Earlier = higher confidence.
# We search the first ~3000 chars (roughly first 1-2 pages) for
# high-confidence matches, and the rest of the document for medium.

# Pattern 1: "March 15, 2003" or "March 15 2003"
_PAT_MONTH_DAY_YEAR = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|'
    r'October|November|December|'
    r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?'
    r'\s+(\d{1,2}),?\s+(\d{4})\b',
    re.IGNORECASE,
)

# Pattern 2: "15 March 2003" or "15th March, 2003"
_PAT_DAY_MONTH_YEAR = re.compile(
    r'\b(\d{1,2})(?:st|nd|rd|th)?\s+'
    r'(January|February|March|April|May|June|July|August|September|'
    r'October|November|December|'
    r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?,?\s+'
    r'(\d{4})\b',
    re.IGNORECASE,
)

# Pattern 3: "03/15/2003" or "03-15-2003" (MM/DD/YYYY)
_PAT_MMDDYYYY = re.compile(
    r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b'
)

# Pattern 4: "2003-03-15" (ISO format)
_PAT_ISO = re.compile(
    r'\b(\d{4})-(\d{2})-(\d{2})\b'
)

# Pattern 5: "Dated: March 2003" or "Date: March, 2003" (month+year only)
_PAT_MONTH_YEAR = re.compile(
    r'(?:dated?|date)[:\s]+'
    r'(January|February|March|April|May|June|July|August|September|'
    r'October|November|December|'
    r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?,?\s+'
    r'(\d{4})\b',
    re.IGNORECASE,
)

# Pattern 6: Standalone "Month Year" on its own line (letterhead style)
_PAT_LINE_MONTH_YEAR = re.compile(
    r'^\s*(January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+(\d{4})\s*$',
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_year(y: int) -> bool:
    """Check if a year is plausible for Epstein documents."""
    return 1970 <= y <= 2025


def _parse_month(name: str) -> Optional[int]:
    return MONTHS.get(name.lower().rstrip("."))


def extract_date(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract the most likely document date from text.

    Searches the first ~3000 characters first (headers, letterheads)
    for a high-confidence date.  Falls back to the full text for a
    medium-confidence date.

    Returns:
        (date_str, confidence) where date_str is 'YYYY-MM-DD' and
        confidence is 'high' or 'medium', or (None, None) if no
        date found.
    """
    header = text[:3000]
    rest = text[3000:]

    # Try header first (high confidence)
    result = _search_patterns(header)
    if result:
        return result, "high"

    # Try rest of document (medium confidence)
    result = _search_patterns(rest)
    if result:
        return result, "medium"

    return None, None


def _search_patterns(text: str) -> Optional[str]:
    """Run all date patterns against text, return first valid match."""

    # Pattern 1: Month Day, Year
    m = _PAT_MONTH_DAY_YEAR.search(text)
    if m:
        month = _parse_month(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if month and 1 <= day <= 31 and _normalize_year(year):
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Pattern 2: Day Month Year
    m = _PAT_DAY_MONTH_YEAR.search(text)
    if m:
        day = int(m.group(1))
        month = _parse_month(m.group(2))
        year = int(m.group(3))
        if month and 1 <= day <= 31 and _normalize_year(year):
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Pattern 3: MM/DD/YYYY
    m = _PAT_MMDDYYYY.search(text)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31 and _normalize_year(year):
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Pattern 4: ISO YYYY-MM-DD
    m = _PAT_ISO.search(text)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31 and _normalize_year(year):
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Pattern 5: "Dated: Month Year" (day unknown, use 1st)
    m = _PAT_MONTH_YEAR.search(text)
    if m:
        month = _parse_month(m.group(1))
        year = int(m.group(2))
        if month and _normalize_year(year):
            return f"{year:04d}-{month:02d}-01"

    # Pattern 6: Standalone "Month Year" line
    m = _PAT_LINE_MONTH_YEAR.search(text)
    if m:
        month = _parse_month(m.group(1))
        year = int(m.group(2))
        if month and _normalize_year(year):
            return f"{year:04d}-{month:02d}-01"

    return None


def extract_dates_for_documents(doc_ids: list[int] = None):
    """Run date extraction on documents.

    If doc_ids is None, processes all extracted documents.
    Skips documents that already have a date.
    Snippets are generated separately by the LLM via analyzer.summarize_documents.
    """
    from . import database
    from .config import TEXT_DIR

    if doc_ids:
        docs = [database.get_document_by_id(did) for did in doc_ids]
        docs = [d for d in docs if d]
    else:
        docs = database.get_extracted_documents()

    processed = 0
    dates_found = 0

    for doc in docs:
        if doc.get("document_date"):
            continue
        if not doc.get("text_path"):
            continue

        text_path = TEXT_DIR / doc["text_path"]
        if not text_path.exists():
            continue

        text = text_path.read_text(encoding="utf-8", errors="replace")

        date_str, confidence = extract_date(text)
        if date_str:
            database.update_document_date(doc["id"], date_str, confidence)
            dates_found += 1

        processed += 1

    logger.info("Date extraction: %d processed, %d dates found",
                processed, dates_found)
    return {"processed": processed, "dates_found": dates_found}
