"""Configuration and paths for the Epstein Analyzer."""

import os
from pathlib import Path

# Project root is ~/local/epstein
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEXT_DIR = DATA_DIR / "text"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "epstein.db"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Ensure data directories exist
for d in [DATA_DIR, TEXT_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Ollama settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# DOJ site settings
DOJ_BASE_URL = "https://www.justice.gov"
DOJ_SEARCH_URL = f"{DOJ_BASE_URL}/epstein/search"
DOJ_EPSTEIN_URL = f"{DOJ_BASE_URL}/epstein"
DOJ_AGE_COOKIE = {"justiceGovAgeVerified": "true"}

# Selenium settings
SELENIUM_HEADLESS = True
SELENIUM_PAGE_LOAD_TIMEOUT = 30
SELENIUM_IMPLICIT_WAIT = 10

# Download settings
DOWNLOAD_WORKERS = 4
DOWNLOAD_RETRY_COUNT = 3
DOWNLOAD_RETRY_DELAY = 2  # seconds
DOWNLOAD_TIMEOUT = 120  # seconds per file

# Text extraction settings
OCR_MIN_TEXT_LENGTH = 20  # pages with less text than this get OCR'd
OCR_ENGINE_MODE = 3  # LSTM neural net (most accurate)
OCR_PAGE_SEG_MODE = 6  # Assume uniform block of text
OCR_CONFIDENCE_THRESHOLD = 60  # flag pages below this confidence

# Chunking settings for RAG
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 100  # characters

# LLM context / batch settings for map-reduce analysis
LLM_CTX_TOKENS = 6144       # num_ctx sent to Ollama for map batches
LLM_BATCH_CHARS = 12000     # max chars of document text per map-phase batch
LLM_MAP_PREDICT = 300       # max tokens per batch summary (keep concise)
LLM_REDUCE_CTX = 32768      # larger context for the reduce/synthesis phase
LLM_REDUCE_PREDICT = 8000   # max tokens for final synthesis — don't truncate

# Search settings
SEARCH_RESULTS_PER_PAGE = 10  # DOJ API returns 10 results per page (fixed)
MAX_SEARCH_PAGES = 1000  # safety limit (10 results/page = 10,000 docs max)

# Web UI settings
WEB_HOST = "127.0.0.1"
WEB_PORT = 8742  # arbitrary unused port
