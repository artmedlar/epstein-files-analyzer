"""Targeted PDF downloader with threading, resume, and cookie support."""

import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import requests

from .config import (
    DOJ_AGE_COOKIE, DOJ_BASE_URL,
    DOWNLOAD_WORKERS, DOWNLOAD_RETRY_COUNT,
    DOWNLOAD_RETRY_DELAY, DOWNLOAD_TIMEOUT,
    DATA_DIR,
)
from .models import SearchResult

logger = logging.getLogger(__name__)

# Temporary download directory (PDFs deleted after extraction)
TEMP_PDF_DIR = DATA_DIR / "tmp_pdfs"
TEMP_PDF_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_LOCK = threading.Lock()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{DOJ_BASE_URL}/epstein",
}


def _create_session(extra_cookies: Optional[dict] = None) -> requests.Session:
    """Create a requests session with DOJ cookies."""
    session = requests.Session()
    for name, value in DOJ_AGE_COOKIE.items():
        session.cookies.set(name, value, domain=".justice.gov", path="/")
    if extra_cookies:
        for name, value in extra_cookies.items():
            session.cookies.set(name, value, domain=".justice.gov", path="/")
    return session


def _download_single(url: str, session: requests.Session,
                     output_dir: Path) -> tuple[bool, str, Optional[str], Optional[str]]:
    """
    Download a single PDF file.

    Returns: (success, filename, local_path_or_None, error_or_None)
    """
    filename = unquote(url.split("/")[-1].split("?")[0])
    output_path = output_dir / filename

    # Skip if already downloaded (for resume)
    if output_path.exists() and output_path.stat().st_size > 0:
        return True, filename, str(output_path), "already_exists"

    for attempt in range(1, DOWNLOAD_RETRY_COUNT + 1):
        try:
            response = session.get(
                url, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT, stream=True
            )
            response.raise_for_status()

            # Verify we got a PDF, not an HTML error page
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                raise RuntimeError("Got HTML instead of PDF (possible age gate or error)")

            tmp_path = output_path.with_suffix(".tmp")
            hasher = hashlib.sha256()
            first_chunk = True

            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        if first_chunk and not chunk[:4].startswith(b"%PDF"):
                            raise RuntimeError(
                                f"Not a PDF file (starts with {chunk[:20]!r})"
                            )
                        first_chunk = False
                        f.write(chunk)
                        hasher.update(chunk)

            if tmp_path.stat().st_size == 0:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError("Downloaded file is empty")

            # Move to final location
            tmp_path.replace(output_path)
            file_hash = hasher.hexdigest()

            return True, filename, str(output_path), file_hash

        except Exception as e:
            tmp_path = output_path.with_suffix(".tmp")
            tmp_path.unlink(missing_ok=True)

            if attempt == DOWNLOAD_RETRY_COUNT:
                logger.error(f"Failed to download {filename}: {e}")
                return False, filename, None, str(e)

            time.sleep(DOWNLOAD_RETRY_DELAY * attempt)

    return False, filename, None, "Max retries exceeded"


def download_documents(results: list[SearchResult],
                       extra_cookies: Optional[dict] = None,
                       workers: int = DOWNLOAD_WORKERS,
                       progress_callback=None) -> list[dict]:
    """
    Download PDFs for a list of search results.

    Args:
        results: List of SearchResult from DOJ search
        extra_cookies: Additional cookies from Selenium session
        workers: Number of concurrent download threads
        progress_callback: Optional callable for progress updates

    Returns:
        List of dicts: {url, filename, local_path, file_hash, success, error}
    """
    from .models import SearchProgress

    session = _create_session(extra_cookies)
    output_dir = TEMP_PDF_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(results)
    completed = 0
    downloaded = 0
    failed = 0
    skipped = 0
    download_results = []

    logger.info(f"Downloading {total} PDFs with {workers} workers")

    if progress_callback:
        progress_callback(SearchProgress(
            stage="downloading", total=total,
            message=f"Starting download of {total} documents..."
        ))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_result = {
            executor.submit(
                _download_single, r.url, session, output_dir
            ): r for r in results
        }

        for future in as_completed(future_to_result):
            search_result = future_to_result[future]
            completed += 1

            try:
                success, filename, local_path, info = future.result()

                if success and info == "already_exists":
                    skipped += 1
                    status_str = "SKIP"
                elif success:
                    downloaded += 1
                    status_str = "OK"
                else:
                    failed += 1
                    status_str = "FAIL"

                download_results.append({
                    "url": search_result.url,
                    "filename": filename,
                    "local_path": local_path,
                    "file_hash": info if success and info != "already_exists" else None,
                    "success": success,
                    "error": info if not success else None,
                    "data_set": search_result.data_set,
                })

                logger.info(
                    f"[{completed}/{total}] [{status_str}] {filename}"
                )

                if progress_callback:
                    progress_callback(SearchProgress(
                        stage="downloading",
                        current=completed, total=total,
                        message=f"[{completed}/{total}] {status_str}: {filename}",
                        documents_found=downloaded + skipped
                    ))

            except Exception as e:
                failed += 1
                download_results.append({
                    "url": search_result.url,
                    "filename": search_result.filename,
                    "local_path": None,
                    "file_hash": None,
                    "success": False,
                    "error": str(e),
                    "data_set": search_result.data_set,
                })

    logger.info(
        f"Download complete: {downloaded} new, {skipped} skipped, {failed} failed"
    )
    return download_results


def cleanup_temp_pdfs():
    """Remove all temporary PDF files."""
    if TEMP_PDF_DIR.exists():
        for f in TEMP_PDF_DIR.iterdir():
            if f.suffix.lower() in (".pdf", ".tmp"):
                f.unlink(missing_ok=True)
        logger.info("Temporary PDFs cleaned up")
