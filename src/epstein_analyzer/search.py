"""Search engine: DOJ site search via Selenium + local Whoosh index."""

import os
import re
import time
import logging
from typing import Optional
from pathlib import Path
from urllib.parse import urljoin, unquote

from .config import (
    DOJ_SEARCH_URL, DOJ_BASE_URL, DOJ_AGE_COOKIE,
    SELENIUM_HEADLESS, SELENIUM_PAGE_LOAD_TIMEOUT,
    DATA_DIR, INDEX_DIR, TEXT_DIR, MAX_SEARCH_PAGES,
)
from .models import SearchResult

logger = logging.getLogger(__name__)

# PDF URL pattern for DOJ Epstein files
PDF_URL_PATTERN = re.compile(
    r'https?://www\.justice\.gov/epstein/files/[^\s"\'<>]+\.pdf',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# DOJ Site Search via Selenium
# ---------------------------------------------------------------------------

def _create_driver(headless: bool = True, max_attempts: int = 3):
    """Create a browser driver that can bypass Akamai bot protection.

    The DOJ Epstein site uses Akamai Bot Manager which detects and blocks
    headless browsers.  We use ``undetected-chromedriver`` in **visible
    (non-headless)** mode, which is the only reliable way to pass
    Akamai's checks.  A Chrome window will briefly appear during
    searches -- this is expected behaviour.

    Includes retry logic since undetected-chromedriver occasionally
    fails to start cleanly on the first attempt.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        # Primary: undetected-chromedriver (non-headless for Akamai bypass)
        driver = _try_undetected_chrome(headless=False)
        if driver:
            # Verify the driver is actually responsive
            try:
                _ = driver.window_handles
                return driver
            except Exception as e:
                logger.warning(
                    "Driver created but unresponsive (attempt %d): %s",
                    attempt, e,
                )
                try:
                    driver.quit()
                except Exception:
                    pass
                last_error = e
                time.sleep(2)
                continue

        # Fallback: regular Chrome (may fail on Akamai-protected endpoints)
        driver = _try_chrome(headless=headless)
        if driver:
            return driver

        last_error = RuntimeError("Both Chrome drivers failed")
        time.sleep(2)

    raise RuntimeError(
        f"Could not start a browser after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


def _try_undetected_chrome(headless: bool = False):
    """Try undetected-chromedriver, which bypasses Akamai Bot Manager."""
    try:
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1280,900")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-popup-blocking")

        # Detect Chrome major version for compatibility
        chrome_version = None
        try:
            import subprocess
            out = subprocess.check_output(
                ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 "--version"],
                text=True, timeout=5
            ).strip()
            chrome_version = int(out.split()[-1].split(".")[0])
            logger.info(f"Detected Chrome version {chrome_version}")
        except Exception:
            pass

        kwargs = {"options": options}
        if chrome_version:
            kwargs["version_main"] = chrome_version

        driver = uc.Chrome(**kwargs)
        driver.set_page_load_timeout(SELENIUM_PAGE_LOAD_TIMEOUT)
        mode = "headless" if headless else "visible"
        logger.info(f"Using undetected-chromedriver ({mode})")
        return driver
    except Exception as e:
        logger.warning(f"undetected-chromedriver not available: {e}")
        return None


def _try_chrome(headless: bool = True):
    """Try standard Chrome WebDriver (fallback)."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1280,900")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        service = Service(log_output=os.devnull)
        driver = webdriver.Chrome(options=options, service=service)
        driver.set_page_load_timeout(SELENIUM_PAGE_LOAD_TIMEOUT)
        logger.info("Using standard Chrome (fallback)")
        return driver
    except Exception as e:
        logger.warning(f"Chrome not available: {e}")
        return None


def _handle_bot_challenge(driver, max_retries: int = 3):
    """Handle the DOJ 'I am not a robot' challenge page.

    The DOJ presents a page with an 'I am not a robot' button that
    computes SHA256-based authorization cookies and reloads.  We click
    the button and wait for the real page to load.
    """
    from selenium.webdriver.common.by import By

    for attempt in range(max_retries):
        try:
            robot_btn = driver.find_element(
                By.XPATH, "//input[@value='I am not a robot']"
            )
        except Exception:
            return True

        logger.info("Bot challenge detected (attempt %d), clicking...", attempt + 1)
        robot_btn.click()
        time.sleep(8)

        # The click may open a new window/tab — switch to it
        try:
            handles = driver.window_handles
            if handles:
                driver.switch_to.window(handles[-1])
        except Exception:
            pass

        try:
            driver.find_element(By.XPATH, "//input[@value='I am not a robot']")
            logger.warning("Bot challenge still present after click")
        except Exception:
            logger.info("Bot challenge cleared")
            return True

    logger.error("Could not clear bot challenge after %d attempts", max_retries)
    return False


def _handle_age_verification(driver):
    """Click the age verification button if present."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException

    selectors = [
        "//button[contains(text(), 'Yes')]",
        "//button[contains(text(), 'yes')]",
        "//a[contains(text(), 'Yes')]",
        "//a[contains(text(), 'YES')]",
        "//input[@value='Yes']",
        "//input[@value='YES']",
        "//a[contains(@class, 'age-gate-yes')]",
        "//button[contains(@class, 'age-gate-yes')]",
    ]
    for selector in selectors:
        try:
            button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, selector))
            )
            button.click()
            logger.info("Age verification confirmed")
            time.sleep(2)
            return True
        except (TimeoutException, Exception):
            continue

    # Also try setting the cookie directly
    try:
        driver.add_cookie({
            "name": "justiceGovAgeVerified",
            "value": "true",
            "domain": ".justice.gov",
            "path": "/"
        })
    except Exception:
        pass
    return False


def _extract_results_from_page(driver) -> list[SearchResult]:
    """Extract document links from the current search results page."""
    from selenium.webdriver.common.by import By

    results = []
    seen_urls = set()

    # Method 1: Find PDF links inside the #results container
    try:
        results_div = driver.find_element(By.ID, "results")
        links = results_div.find_elements(By.XPATH, ".//a[contains(@href, '.pdf')]")
        for link in links:
            href = link.get_attribute("href")
            if href and href not in seen_urls:
                seen_urls.add(href)
                filename = unquote(href.split("/")[-1])
                data_set = None
                ds_match = re.search(r'DataSet\s*(\d+)', unquote(href))
                if ds_match:
                    data_set = f"DataSet {ds_match.group(1)}"
                title = link.text.strip() if link.text else filename
                results.append(SearchResult(
                    filename=filename, url=href,
                    data_set=data_set, title=title
                ))
        logger.info(f"Extracted {len(results)} PDF links from #results")
    except Exception as e:
        logger.warning(f"Error extracting from #results div: {e}")

    # Method 2: Find all PDF links on the page (broader search)
    if not results:
        try:
            links = driver.find_elements(
                By.XPATH, "//a[contains(@href, '.pdf')]"
            )
            for link in links:
                href = link.get_attribute("href")
                if href and "epstein" in href.lower() and href not in seen_urls:
                    seen_urls.add(href)
                    filename = unquote(href.split("/")[-1])
                    data_set = None
                    ds_match = re.search(r'DataSet\s*(\d+)', unquote(href))
                    if ds_match:
                        data_set = f"DataSet {ds_match.group(1)}"
                    title = link.text.strip() if link.text else filename
                    results.append(SearchResult(
                        filename=filename, url=href,
                        data_set=data_set, title=title
                    ))
            logger.info(f"Extracted {len(results)} PDF links from full page")
        except Exception as e:
            logger.warning(f"Error extracting links via anchors: {e}")

    # Method 3: Regex on page source as final fallback
    if not results:
        try:
            source = driver.page_source
            for url in PDF_URL_PATTERN.findall(source):
                if url not in seen_urls:
                    seen_urls.add(url)
                    filename = unquote(url.split("/")[-1])
                    data_set = None
                    ds_match = re.search(r'DataSet\s*(\d+)', unquote(url))
                    if ds_match:
                        data_set = f"DataSet {ds_match.group(1)}"
                    results.append(SearchResult(
                        filename=filename, url=url,
                        data_set=data_set, title=filename
                    ))
            logger.info(f"Extracted {len(results)} PDF links via regex")
        except Exception as e:
            logger.warning(f"Error extracting links via regex: {e}")

    return results


def _fetch_api_page(driver, query: str, page: int, timeout: int = 30) -> Optional[dict]:
    """Fetch a single page from the DOJ /multimedia-search API.

    Executes a fetch() call *inside* the browser context so that
    the request carries the Akamai authorisation cookies established
    by the visible Chrome session.

    The API is 1-indexed: page=1 returns results 1-10, page=2
    returns results 11-20, etc.

    Returns the parsed JSON dict, or None on failure.
    """
    import json as _json
    from urllib.parse import quote

    encoded_query = quote(query)
    js = f"""
        var callback = arguments[arguments.length - 1];
        fetch('/multimedia-search?keys={encoded_query}&page=' + arguments[0])
            .then(function(r) {{
                if (!r.ok) {{ callback(JSON.stringify({{error: r.status}})); return; }}
                return r.json();
            }})
            .then(function(d) {{ callback(JSON.stringify(d)); }})
            .catch(function(e) {{ callback(JSON.stringify({{error: e.message}})); }});
    """
    try:
        raw = driver.execute_async_script(js, page)
        data = _json.loads(raw)
        if "error" in data:
            logger.warning("API page %d returned error: %s", page, data["error"])
            return None
        return data
    except Exception as e:
        logger.warning("Failed to fetch API page %d: %s", page, e)
        return None


def _parse_api_results(api_data: dict) -> tuple[list[SearchResult], int]:
    """Parse the /multimedia-search JSON response into SearchResult objects.

    The API returns Elasticsearch-style results where each hit is a
    *chunk* of a document (large PDFs are split into multiple indexed
    chunks).  We deduplicate by filename so each unique PDF appears
    only once.

    Returns (list_of_results, total_hit_count).
    """
    hits_obj = api_data.get("hits", {})
    total = hits_obj.get("total", {}).get("value", 0)
    hits = hits_obj.get("hits", [])

    results: list[SearchResult] = []
    seen_filenames: set[str] = set()

    for hit in hits:
        src = hit.get("_source", {})
        filename = src.get("ORIGIN_FILE_NAME", "")
        url = src.get("ORIGIN_FILE_URI", "")
        if not filename or not url:
            continue
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)

        data_set = None
        key = src.get("key", "")
        ds_match = re.search(r'DataSet\s*(\d+)', key)
        if ds_match:
            data_set = f"DataSet {ds_match.group(1)}"

        results.append(SearchResult(
            filename=filename, url=url,
            data_set=data_set, title=filename,
        ))

    return results, total


def search_doj(query: str, max_pages: int = MAX_SEARCH_PAGES,
               headless: bool = SELENIUM_HEADLESS,
               progress_callback=None) -> list[SearchResult]:
    """
    Search the DOJ Epstein site for a query term.

    Opens a visible Chrome window (required for Akamai bypass),
    submits the initial search via the page UI to establish API
    authorisation cookies, then calls the ``/multimedia-search``
    JSON API directly from inside the browser context to paginate
    through *all* results efficiently.

    The API returns Elasticsearch hits that may include multiple
    chunks per document; results are deduplicated by filename.

    Args:
        query: Search term (person name, keyword, etc.)
        max_pages: Maximum number of API pages to fetch (10 results/page)
        headless: Run browser in headless mode (ignored — always visible
                  for Akamai bypass)
        progress_callback: Optional callable(SearchProgress) for UI updates

    Returns:
        List of SearchResult objects with unique document URLs
    """
    from .models import SearchProgress

    RESULTS_PER_PAGE = 10  # API page size is fixed at 10

    all_results: list[SearchResult] = []
    seen_filenames: set[str] = set()

    if progress_callback:
        progress_callback(SearchProgress(
            stage="searching", message=f"Starting search for: {query}"
        ))

    logger.info("Starting DOJ search for: %s", query)
    driver = _create_driver(headless=headless)

    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException

        # ------------------------------------------------------------------
        # Phase 1: Navigate, pass bot challenge, age gate
        # ------------------------------------------------------------------
        if progress_callback:
            progress_callback(SearchProgress(
                stage="searching", message="Loading DOJ search page..."
            ))
        # Navigate to DOJ search — retry if the window is lost
        nav_ok = False
        for nav_attempt in range(3):
            try:
                driver.get(DOJ_SEARCH_URL)
                time.sleep(6)
                handles = driver.window_handles
                if handles:
                    driver.switch_to.window(handles[-1])
                nav_ok = True
                break
            except Exception as nav_err:
                logger.warning(
                    "Navigation attempt %d failed: %s", nav_attempt + 1, nav_err
                )
                if nav_attempt < 2:
                    # Try to recover with a fresh driver
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    time.sleep(3)
                    driver = _create_driver(headless=headless, max_attempts=2)

        if not nav_ok:
            raise RuntimeError("Could not navigate to DOJ search page")

        if not _handle_bot_challenge(driver):
            msg = "Could not pass DOJ bot challenge"
            logger.error(msg)
            if progress_callback:
                progress_callback(SearchProgress(
                    stage="searching", message=f"Error: {msg}"
                ))
            return []
        time.sleep(3)

        # After bot challenge the page may have redirected away
        # from the search page — navigate back if needed
        try:
            cur = driver.current_url
            if "epstein/search" not in cur:
                logger.info(
                    "Bot challenge redirected to %s, navigating back", cur
                )
                driver.get(DOJ_SEARCH_URL)
                time.sleep(6)
                # May need to clear bot challenge again
                _handle_bot_challenge(driver, max_retries=2)
                time.sleep(2)
        except Exception:
            # Window may have closed; try navigating fresh
            logger.warning("Lost browser window, re-navigating")
            driver.get(DOJ_SEARCH_URL)
            time.sleep(8)

        _handle_age_verification(driver)
        time.sleep(2)

        _handle_bot_challenge(driver, max_retries=1)
        time.sleep(1)

        try:
            logger.info("Current URL after gates: %s", driver.current_url)
        except Exception:
            logger.warning("Could not read current_url after gates")

        # ------------------------------------------------------------------
        # Phase 2: Submit the search via the page UI so the JS establishes
        #          the API session/cookies needed for /multimedia-search
        # ------------------------------------------------------------------
        if progress_callback:
            progress_callback(SearchProgress(
                stage="searching", message="Submitting search..."
            ))

        search_input = None
        for by, selector in [
            (By.ID, "searchInput"),
            (By.CSS_SELECTOR, "#searchInput"),
            (By.XPATH, "//input[@id='searchInput']"),
            (By.XPATH, "//input[@placeholder='Type to search...']"),
        ]:
            try:
                search_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((by, selector))
                )
                if search_input:
                    break
            except (TimeoutException, Exception):
                continue

        if not search_input:
            logger.error("Could not find Epstein search input")
            debug_path = Path(DATA_DIR) / "debug_page.html"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(driver.page_source)
            if progress_callback:
                progress_callback(SearchProgress(
                    stage="searching",
                    message="Error: Could not find Epstein search input. "
                            "The site may be blocking automated access."
                ))
            return []

        search_input.clear()
        search_input.send_keys(query)

        try:
            driver.find_element(By.ID, "searchButton").click()
            logger.info("Clicked Epstein searchButton")
        except Exception:
            from selenium.webdriver.common.keys import Keys
            search_input.send_keys(Keys.RETURN)

        # Wait for the initial UI results (confirms API cookies work)
        if progress_callback:
            progress_callback(SearchProgress(
                stage="searching",
                message="Waiting for initial results..."
            ))
        time.sleep(8)

        _handle_bot_challenge(driver, max_retries=2)

        # ------------------------------------------------------------------
        # Phase 3: Use the /multimedia-search API to paginate all results
        # ------------------------------------------------------------------
        if progress_callback:
            progress_callback(SearchProgress(
                stage="searching",
                message="Fetching results via API..."
            ))

        # First API call to get total count (page=1 is the first page)
        first_page = _fetch_api_page(driver, query, page=1)
        if not first_page:
            # Fall back to scraping the DOM for page-1 results only
            logger.warning("API unavailable, falling back to DOM scraping")
            all_results = _extract_results_from_page(driver)
            if progress_callback:
                progress_callback(SearchProgress(
                    stage="searching",
                    message=f"Search complete (DOM fallback): {len(all_results)} documents",
                    documents_found=len(all_results),
                ))
            return all_results

        page_results, total_hits = _parse_api_results(first_page)
        for r in page_results:
            if r.filename not in seen_filenames:
                seen_filenames.add(r.filename)
                all_results.append(r)

        total_api_pages = min(
            max_pages,
            (total_hits + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE,
        )
        logger.info(
            "API: %d total hits across %d pages, capped to %d",
            total_hits, total_api_pages, max_pages,
        )

        if progress_callback:
            progress_callback(SearchProgress(
                stage="searching",
                current=1,
                total=total_api_pages,
                message=f"Page 1/{total_api_pages} — {total_hits:,} hits, "
                        f"{len(all_results)} unique docs so far",
                documents_found=len(all_results),
            ))

        # Fetch remaining pages
        consecutive_empty = 0
        for page_num in range(2, total_api_pages + 1):
            api_data = _fetch_api_page(driver, query, page=page_num)
            if not api_data:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.warning(
                        "3 consecutive failed API pages, stopping at page %d",
                        page_num,
                    )
                    break
                continue
            consecutive_empty = 0

            page_results, _ = _parse_api_results(api_data)
            new_count = 0
            for r in page_results:
                if r.filename not in seen_filenames:
                    seen_filenames.add(r.filename)
                    all_results.append(r)
                    new_count += 1

            logger.info(
                "Page %d/%d: %d hits, %d new unique docs (%d total)",
                page_num, total_api_pages, len(page_results),
                new_count, len(all_results),
            )

            if progress_callback:
                progress_callback(SearchProgress(
                    stage="searching",
                    current=page_num,
                    total=total_api_pages,
                    message=f"Page {page_num}/{total_api_pages} — "
                            f"{len(all_results)} unique docs",
                    documents_found=len(all_results),
                ))

            # Brief pause to avoid hammering the API
            if page_num % 10 == 0:
                time.sleep(1)

    except Exception as e:
        logger.error("Search error: %s", e, exc_info=True)
        if progress_callback:
            progress_callback(SearchProgress(
                stage="searching", message=f"Search error: {e}"
            ))
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    if progress_callback:
        progress_callback(SearchProgress(
            stage="searching",
            current=total_api_pages if "total_api_pages" in dir() else 0,
            total=total_api_pages if "total_api_pages" in dir() else 0,
            message=f"Search complete: {len(all_results)} unique documents found"
                    f" ({total_hits:,} total hits)"
                    if "total_hits" in dir()
                    else f"Search complete: {len(all_results)} documents found",
            documents_found=len(all_results),
        ))

    logger.info("Search complete: %d unique documents found", len(all_results))
    return all_results


def get_session_cookies(driver) -> dict:
    """Extract cookies from Selenium session for use with requests."""
    return {c["name"]: c["value"] for c in driver.get_cookies()}


# ---------------------------------------------------------------------------
# Local Whoosh Index
# ---------------------------------------------------------------------------

def _get_whoosh_schema():
    """Create the Whoosh index schema."""
    from whoosh.fields import Schema, TEXT, ID, KEYWORD, STORED
    from whoosh.analysis import StemmingAnalyzer

    return Schema(
        doc_id=ID(stored=True, unique=True),
        filename=TEXT(stored=True),
        url=ID(stored=True),
        data_set=ID(stored=True),
        content=TEXT(analyzer=StemmingAnalyzer(), stored=False),
        preview=STORED,
        found_via_query=KEYWORD(stored=True, commas=True),
    )


def build_whoosh_index(force: bool = False):
    """Build or rebuild the Whoosh index from extracted text files."""
    from whoosh import index as whoosh_index
    from tqdm import tqdm

    index_dir = str(INDEX_DIR)
    os.makedirs(index_dir, exist_ok=True)

    schema = _get_whoosh_schema()

    if force or not whoosh_index.exists_in(index_dir):
        ix = whoosh_index.create_in(index_dir, schema)
    else:
        ix = whoosh_index.open_dir(index_dir)

    from . import database
    docs = database.get_extracted_documents()

    writer = ix.writer()
    indexed = 0

    for doc in tqdm(docs, desc="Indexing documents"):
        text_path = doc.get("text_path")
        if not text_path:
            continue

        full_path = TEXT_DIR / text_path if not Path(text_path).is_absolute() else Path(text_path)
        if not full_path.exists():
            continue

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            if len(content) < 20:
                continue

            writer.update_document(
                doc_id=str(doc["id"]),
                filename=doc["filename"],
                url=doc["url"],
                data_set=doc.get("data_set", ""),
                content=content,
                preview=content[:500],
                found_via_query=doc.get("found_via_query", ""),
            )
            indexed += 1
        except Exception as e:
            logger.warning(f"Error indexing {doc['filename']}: {e}")

    writer.commit()
    logger.info(f"Whoosh index built: {indexed} documents indexed")
    return indexed


def search_local(query: str, limit: int = 50) -> list[dict]:
    """Search the local Whoosh index."""
    from whoosh import index as whoosh_index
    from whoosh.qparser import MultifieldParser

    index_dir = str(INDEX_DIR)
    if not whoosh_index.exists_in(index_dir):
        logger.warning("No local index found. Download and extract documents first.")
        return []

    ix = whoosh_index.open_dir(index_dir)

    with ix.searcher() as searcher:
        parser = MultifieldParser(
            ["content", "filename"], ix.schema
        )
        parsed = parser.parse(query)
        results = searcher.search(parsed, limit=limit)

        output = []
        for hit in results:
            output.append({
                "doc_id": hit["doc_id"],
                "filename": hit["filename"],
                "url": hit["url"],
                "data_set": hit.get("data_set", ""),
                "preview": hit.get("preview", ""),
                "score": hit.score,
            })

    return output
