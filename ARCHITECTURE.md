# Architecture & Data Flow

This document explains how the Epstein Files Analyzer works internally -- from the moment you type a name into the search box to the final LLM-generated analysis.

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  User Interface (browser)                                           │
│  frontend/index.html + css/styles.css + js/app.js                   │
│                                                                      │
│  Single-page app with four tabs:                                     │
│  [Search] [Documents] [Analysis] [Settings]                          │
└────────┬────────────┬────────────────┬──────────────────────────────┘
         │ HTTP/REST  │ WebSocket      │ WebSocket
         │            │ (search,       │ (analysis
         │            │  download)     │  streaming)
         ▼            ▼                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (app.py)                                           │
│  Runs on 127.0.0.1:8742 — never exposed to the internet             │
│                                                                      │
│  REST:        /api/status, /api/documents, /api/documents/{id}/text  │
│  WebSocket:   /ws/search, /ws/download, /ws/analyze                  │
│  Static:      /static/* → frontend/                                  │
└─────┬──────────┬──────────────┬───────────────┬─────────────────────┘
      │          │              │               │
      ▼          ▼              ▼               ▼
  search.py  downloader.py  extractor.py   analyzer.py
  (Selenium)  (requests)    (PyMuPDF+OCR)  (Ollama LLM)
      │          │              │               │
      │          │              │               │
      ▼          ▼              ▼               ▼
 DOJ Website   DOJ CDN      Tesseract       Ollama
 (justice.gov) (PDF files)  (system)        (localhost:11434)
```

All state is stored in two places:
- **SQLite database** (`data/epstein.db`) -- document metadata, search history, saved analyses
- **Text files** (`data/text/`) -- extracted text from each PDF

## Component Details

### 1. Frontend (`frontend/`)

A single-page application built with vanilla HTML, CSS, and JavaScript (no build tools, no framework). The UI uses a dark theme with tab-based navigation.

**Communication with the backend:**
- **REST** (`fetch`) for reading data: document lists, document text, system status
- **WebSocket** for long-running operations: search, download, analysis

The WebSocket protocol uses JSON messages with a `type` field to distinguish message kinds. For example, during analysis, the frontend receives:

| Message type | Purpose |
|---|---|
| `estimate` | Document count, batch count, estimated time |
| `progress` | Stage name, current/total counts, human-readable message |
| `batch_result` | Summary from one completed batch (shown in expandable section) |
| `time_update` | Refined elapsed/remaining time estimates |
| `token` | One token of the final synthesis (streamed for real-time display) |
| `sources` | List of source document filenames and IDs |
| `heartbeat` | Keep-alive ping every 15 seconds during long operations |
| `error` | Error message (also used for cancellation acknowledgment) |
| `done` | Analysis complete |

The frontend renders LLM output as Markdown. Batch findings are displayed in collapsible `<details>` elements so the user can inspect individual batch results while the final synthesis streams.

### 2. Search Engine (`search.py`)

Searches the DOJ Epstein document library at `justice.gov/epstein/search`.

**The challenge:** The DOJ site uses Akamai Bot Manager, which blocks headless browsers, automated HTTP clients, and most scraping tools.

**The solution:**

1. **Visible Chrome** -- `undetected-chromedriver` opens a real, visible Chrome window. This is the only way to reliably pass Akamai's JavaScript-based bot detection. The Chrome window is expected and will appear briefly during searches.

2. **Bot challenge handling** -- The DOJ presents an "I am not a robot" button that computes SHA256-based authorization cookies. The tool clicks this button and waits for the challenge to clear.

3. **Age verification** -- The Epstein section requires age confirmation. The tool clicks the "Yes" button or sets the `justiceGovAgeVerified` cookie directly.

4. **API pagination** -- After passing the bot challenge and submitting a search through the UI (which establishes the necessary session cookies), the tool switches to calling the DOJ's internal `/multimedia-search` JSON API directly from within the browser context. This API returns Elasticsearch-style results, 10 per page. The tool paginates through all pages, deduplicating by filename (since large PDFs are split into multiple indexed chunks).

**Data flow:**

```
User enters "Leon Black"
    │
    ▼
Chrome opens → navigates to DOJ search page
    │
    ▼
Bot challenge detected? → Click "I am not a robot" → Wait for cookies
    │
    ▼
Age verification? → Click "Yes"
    │
    ▼
Type query into search input → Click search button
    │  (this triggers the DOJ JavaScript to establish API session cookies)
    ▼
Call /multimedia-search?keys=Danny+Hillis&page=1 from browser context
    │
    ▼
Parse response: total hits, document URLs, filenames, data sets
    │
    ▼
Loop through page=2, page=3, ... up to all pages
    │  (deduplicate by filename; brief pause every 10 pages)
    │  (send SearchProgress updates to UI via WebSocket queue)
    ▼
Return list of SearchResult objects
    │
    ▼
Store in SQLite: each unique URL → documents table (status: "found")
Log to search_history table
```

**Why Selenium and not `requests`?** The Akamai Bot Manager checks for a full browser environment: JavaScript execution, DOM APIs, WebGL fingerprinting, mouse events, etc. No HTTP client can replicate this. The tool must establish a real browser session first, then reuse those session cookies for the API calls.

### 3. Downloader (`downloader.py`)

Downloads PDF files from the DOJ CDN using `requests` with a thread pool.

**Design decisions:**

- **Thread pool** (`ThreadPoolExecutor`) with 4 workers for concurrent downloads
- **Resume support** -- If a PDF already exists on disk (from a previous interrupted run), it's skipped
- **DOJ cookies** -- The age verification cookie is set on all requests to avoid HTML error pages instead of PDFs
- **Integrity checks** -- Verifies the response is actually a PDF (checks `Content-Type` header and the `%PDF` magic bytes), computes SHA-256 hash
- **Retry logic** -- 3 attempts with exponential backoff per file
- **Temporary storage** -- Downloads to `data/tmp_pdfs/` with `.tmp` extension, atomically renamed on completion

**Data flow:**

```
List of SearchResult URLs
    │
    ▼
Create requests.Session with DOJ cookies
    │
    ▼
ThreadPoolExecutor(max_workers=4)
    │
    ├─► Worker 1: GET url → stream to tmp_pdfs/file.tmp → rename to file.pdf
    ├─► Worker 2: GET url → stream to tmp_pdfs/file.tmp → rename to file.pdf
    ├─► Worker 3: ...
    └─► Worker 4: ...
    │
    ▼
Return list of {url, filename, local_path, file_hash, success}
```

Progress is reported back through a `SearchProgress` callback that the WebSocket handler (`app.py`) forwards to the frontend.

### 4. Text Extractor (`extractor.py`)

Extracts text from PDFs using a two-tier approach, then deletes the PDF.

**Tier 1: PyMuPDF (`fitz`)** -- Fast extraction of embedded digital text. Works on PDFs that were born digital (typed documents, emails, etc.).

**Tier 2: Tesseract OCR** -- Fallback for scanned documents. Renders each page to a 300 DPI image, runs Tesseract with LSTM neural net engine (OEM 3), tries two page segmentation modes (PSM 6 for uniform text blocks, PSM 3 for fully automatic), and picks whichever gives higher confidence.

**The extract-and-discard pipeline:**

```
PDF on disk (data/tmp_pdfs/EFTA00026703.pdf)
    │
    ▼
Open with PyMuPDF → iterate pages
    │
    ├── Page has digital text (≥20 chars, not garbage)?
    │       → Use that text
    │
    └── Page has little/no text or garbage characters?
            → Render to 300 DPI PNG
            → Run Tesseract OCR
            → If OCR confidence < 60%, try alternate PSM
            → Pick best result
    │
    ▼
Combine all pages with "--- Page N ---" separators
Save to data/text/EFTA00026703.txt
    │
    ▼
DELETE the original PDF ← this is the key design choice
    │
    ▼
Update database: status → "extracted", store text_path, page_count, ocr_needed
Rebuild Whoosh full-text index
```

**Why delete PDFs?** The DOJ Epstein files contain thousands of documents that can be hundreds of pages each. Keeping all PDFs would consume tens of gigabytes. The extracted text is typically ~50 KB per document. The original PDF URL is stored in the database, so any document can be re-downloaded from the DOJ site if needed.

**Garbage detection:** Some PDFs contain embedded text that is actually encoding artifacts (random symbols). The extractor checks whether at least 50% of characters are alphanumeric or spaces; if not, it falls back to OCR.

### 5. LLM Analyzer (`analyzer.py`)

This is the most complex component. It communicates with Ollama's HTTP API to run the `llama3.1:8b` model locally.

#### 5a. Map-Reduce Analysis

The core analysis pipeline uses a **map-reduce** pattern to process ALL documents, not just a sample.

**Why map-reduce?** A local 8B-parameter model has a limited context window. You cannot feed 1,500 documents into a single prompt. Instead:

- **Map phase**: Process documents in batches. Each batch gets a concise summary from the LLM.
- **Reduce phase**: Feed ALL batch summaries into a single prompt for comprehensive synthesis.

**Detailed flow:**

```
analyze_documents("Leon Black", [1, 2, 3, ..., 1512])
    │
    ▼
LOAD: Read text files for all 1512 document IDs
    │  (skip docs with no text or <20 chars)
    │  (yield progress: "Loading 1512 documents...")
    ▼
BUILD BATCHES: Group documents into batches of ~12,000 characters
    │  Documents larger than 12,000 chars are truncated with a note
    │  Result: ~260 batches (large docs = 1 per batch, small docs = many per batch)
    │
    │  (yield estimate: 260 batches, ~55s each, ~4 hours total)
    │  (yield sources: list of all document filenames)
    ▼
MAP PHASE: For each batch...
    │
    │  ┌─────────────────────────────────────────────────────┐
    │  │ Prompt:                                             │
    │  │ "Extract facts about 'Leon Black' from these     │
    │  │  documents. For EACH document:                      │
    │  │  FILENAME: key facts, names, dates, locations..."   │
    │  │                                                     │
    │  │ [document text for this batch]                      │
    │  │                                                     │
    │  │ Context: 6,144 tokens (num_ctx)                     │
    │  │ Max output: 300 tokens (num_predict)                │
    │  │ Temperature: 0.3                                    │
    │  │ Timeout: 600 seconds                                │
    │  └─────────────────────────────────────────────────────┘
    │
    │  (yield progress: "Batch 134/260 - 700/1512 documents")
    │  (yield batch_result: individual batch findings)
    │  (yield time_update: refined estimate based on actual speed)
    │
    │  Each batch takes ~50-60 seconds on CPU
    │  After each batch, check cancel_flag
    │
    ▼
REDUCE PHASE: Synthesize all batch summaries
    │
    │  ┌─────────────────────────────────────────────────────┐
    │  │ Prompt:                                             │
    │  │ "You have extracted findings from 1512 documents.   │
    │  │  Synthesize ALL of them into:                       │
    │  │  1. Key Facts                                       │
    │  │  2. Timeline                                        │
    │  │  3. People Connected                                │
    │  │  4. Locations                                       │
    │  │  5. Financial Details                               │
    │  │  6. Document Types                                  │
    │  │  7. Patterns                                        │
    │  │  8. Most Significant Findings                       │
    │  │  Cite specific document filenames."                  │
    │  │                                                     │
    │  │ [all batch summaries concatenated]                   │
    │  │                                                     │
    │  │ Context: 32,768 tokens (num_ctx)                    │
    │  │ Max output: 8,000 tokens (num_predict)              │
    │  │ Streaming: yes (token by token)                      │
    │  └─────────────────────────────────────────────────────┘
    │
    │  (yield token: each token as it's generated)
    ▼
SAVE: Store synthesis + all batch findings in SQLite analyses table
```

**LLM parameters explained:**

| Parameter | Map Phase | Reduce Phase | Why |
|---|---|---|---|
| `num_ctx` | 6,144 | 32,768 | Map batches are small; reduce needs to hold all summaries |
| `num_predict` | 300 | 8,000 | Map summaries should be concise; final synthesis should be comprehensive |
| `temperature` | 0.3 | 0.3 | Low temperature for factual extraction |
| `timeout` | 600s | 600s | Per-request timeout for Ollama HTTP calls |

**Batching logic:** Documents are packed into batches with a maximum of 12,000 characters total. If a single document exceeds 12,000 characters, it is truncated (with a note) and placed in its own batch. This prevents one huge document from consuming the entire context window and making that batch's LLM call excessively slow.

#### 5b. Correlation Analysis

A specialized mode for finding cross-connections between two or more people.

Unlike the map-reduce analysis (which processes every document), correlation uses a **chunk-based** approach:

1. Split all documents into large text chunks (1,500 chars each)
2. Score chunks by how many of the queried names they mention
3. Multi-mention chunks (containing 2+ names) are prioritized
4. Send the top-ranked chunks to the LLM with a prompt focused on connections, timelines, and document-by-document analysis
5. Stream the response token by token

#### 5c. Embeddings

The `nomic-embed-text` model is used by the correlation analysis to compute vector embeddings for semantic similarity. This is separate from the map-reduce analysis, which processes all documents without any embedding or ranking step.

### 6. Database (`database.py`)

SQLite with WAL mode for concurrent read/write safety. Three tables:

**`documents`** -- One row per unique DOJ document URL.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER PK | Internal ID, used in doc_ids for analysis |
| `filename` | TEXT | Original PDF filename (e.g., EFTA00026703.pdf) |
| `url` | TEXT UNIQUE | Full DOJ URL (for re-download) |
| `data_set` | TEXT | Which DOJ data set (e.g., "DataSet 9") |
| `status` | TEXT | Pipeline stage: found → downloaded → extracted |
| `text_path` | TEXT | Relative path to text file in data/text/ |
| `page_count` | INTEGER | Number of pages in the PDF |
| `ocr_needed` | INTEGER | Whether any page required OCR |
| `ocr_confidence` | REAL | Average OCR confidence (0-100) |
| `file_hash` | TEXT | SHA-256 of the downloaded PDF |
| `found_via_query` | TEXT | The search query that discovered this document |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | ISO timestamp |

**`analyses`** -- One row per completed LLM analysis.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `query` | TEXT | The query/name analyzed |
| `document_ids` | TEXT | Comma-separated list of document IDs |
| `analysis_text` | TEXT | Full LLM output (synthesis + batch findings) |
| `dates_found` | TEXT | Comma-separated dates (if extracted) |
| `people_found` | TEXT | Comma-separated names (if extracted) |
| `locations_found` | TEXT | Comma-separated locations (if extracted) |
| `document_types` | TEXT | Comma-separated document types (if extracted) |
| `model_used` | TEXT | e.g., "llama3.1:8b" |
| `created_at` | TEXT | ISO timestamp |

**`search_history`** -- Log of DOJ searches.

| Column | Type | Purpose |
|---|---|---|
| `query` | TEXT | Search term |
| `results_count` | INTEGER | Number of unique documents found |
| `searched_at` | TEXT | ISO timestamp |

### 7. FastAPI Backend (`app.py`)

The backend serves both REST endpoints and WebSocket connections.

**Why WebSocket?** Three operations can take minutes to hours:
- DOJ search with pagination (minutes)
- Downloading hundreds of PDFs (minutes)
- LLM analysis of all documents (hours)

WebSockets allow the backend to stream real-time progress, batch results, and LLM tokens to the frontend without polling.

**Concurrency model:**

```
Main async event loop (FastAPI/uvicorn)
    │
    ├── WebSocket handler (async)
    │       │
    │       ├── run_in_executor → search/download/analysis thread (sync)
    │       │       │
    │       │       └── Puts items into asyncio.Queue (thread-safe)
    │       │
    │       ├── Main loop: reads from Queue, sends to WebSocket
    │       │
    │       └── watch_disconnect task: listens for client disconnect or cancel
    │
    └── REST handlers (async, trivial DB reads)
```

The key challenge was WebSocket concurrency: only one task can read from a WebSocket at a time. The `watch_disconnect` coroutine is the single reader for incoming messages (disconnect events, cancel commands). The main loop only writes to the WebSocket.

**Heartbeat:** A JSON `{"type": "heartbeat"}` message is sent every 15 seconds during long operations to keep the WebSocket connection alive through proxies and browsers that might otherwise time out.

**Cancellation:** The user can click "Cancel" in the UI, which sends `{"type": "cancel"}` over the WebSocket. The `watch_disconnect` task detects this and sets a `threading.Event` flag. The analysis thread checks this flag between batches and stops early.

### 8. Local Search Index (Whoosh)

After text extraction, a Whoosh full-text index is built/updated over all extracted text files. This enables instant local search across all downloaded documents.

**Schema:**
- `doc_id` -- Document ID (for linking back to the database)
- `filename` -- Searchable filename
- `content` -- Full text, indexed with a stemming analyzer
- `preview` -- First 500 characters (stored but not indexed)
- `url`, `data_set`, `found_via_query` -- Metadata fields

The index uses the `StemmingAnalyzer` so searches for "traveled" also match "traveling", "travel", etc.

## Configuration (`config.py`)

All settings are centralized in one file. Key settings can be overridden via environment variables:

| Setting | Env Var | Default | Purpose |
|---|---|---|---|
| `OLLAMA_URL` | `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `OLLAMA_MODEL` | `llama3.1:8b` | LLM for analysis |
| `OLLAMA_EMBED_MODEL` | `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |

Most other settings (download workers, OCR thresholds, LLM batch sizes) are tuned for local hardware and should not need changing unless you have a faster or slower machine.

## End-to-End Example

Here is the complete flow for analyzing documents about "Leon Black":

```
1. USER types "Leon Black" and clicks Search
        │
2. FRONTEND opens WebSocket to /ws/search
        │
3. BACKEND spawns thread → creates Chrome driver
        │
4. CHROME opens DOJ search page, passes bot challenge, age gate
        │
5. SEARCH types "Leon Black", clicks search button
        │
6. SEARCH calls /multimedia-search API pages 1..N
   Each page: parse results, deduplicate, send progress
        │
7. SEARCH returns 1,512 unique document URLs
   Stored in SQLite (status: "found")
        │
8. USER clicks "Download All"
        │
9. FRONTEND opens WebSocket to /ws/download
        │
10. DOWNLOADER spawns 4 threads, downloads PDFs concurrently
    Each PDF: verify, hash, save to tmp_pdfs/
        │
11. EXTRACTOR processes each PDF:
    - PyMuPDF extracts text (or Tesseract OCR for scanned pages)
    - Text saved to data/text/
    - PDF deleted
    - Database updated (status: "extracted")
        │
12. WHOOSH index rebuilt with new documents
        │
13. USER switches to Analysis tab, types "Leon Black", clicks Analyze
        │
14. FRONTEND opens WebSocket to /ws/analyze
        │
15. ANALYZER loads text for all 1,512 extracted documents
        │
16. ANALYZER builds ~260 batches of ~12,000 chars each
        │
17. MAP PHASE: For each batch, LLM extracts key facts (6K context, 300 tokens)
    ~50-60 seconds per batch → ~4 hours total
    Progress streamed to UI in real-time
        │
18. REDUCE PHASE: All batch summaries synthesized (32K context, 8K tokens)
    Streamed token-by-token to UI
        │
19. Full analysis (synthesis + batch findings) saved to SQLite
        │
20. USER reads comprehensive analysis of all 1,512 documents
```

## Security Notes

- The web server binds to `127.0.0.1` only (localhost). It is never exposed to the network.
- No data leaves the machine. All LLM processing happens locally via Ollama.
- The DOJ Epstein files are public records released by the U.S. Department of Justice.
- Selenium runs a visible Chrome window (not headless) because that is the only way to pass the DOJ's bot detection. This is expected behavior, not a security issue.
