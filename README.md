# EPSTEIN HELPER, an Epstein Files Analyzer

Search, download, and analyze documents from the DOJ Epstein Files release using a local LLM.

This tool automates searching the DOJ's Epstein document library at [justice.gov/epstein](https://www.justice.gov/epstein), downloads matching documents, extracts text (with OCR for scanned pages), and runs comprehensive analysis using a local Ollama LLM -- all through a browser-based UI.

## Features

- **DOJ Search** -- Automated search across all 12 DOJ Epstein data sets (3+ million pages) using the DOJ's own search API, with Selenium to bypass Akamai bot protection.
- **Batch Download** -- Download all matching documents or select specific ones. Multi-threaded with retry logic.
- **Text Extraction** -- PyMuPDF for digital text, Tesseract OCR fallback for scanned pages. PDFs are deleted after extraction to save disk space.
- **Map-Reduce Analysis** -- Feed ALL downloaded documents to your local LLM. Documents are batched, each batch is analyzed for key findings, then all findings are synthesized into a comprehensive report. No arbitrary limits on how many documents are analyzed.
- **Correlation Analysis** -- Cross-reference two or more people (or topics) across the entire document corpus. Finds co-occurrences and analyzes connections.
- **Document Timeline** -- Interactive bar chart showing when documents are dated, with drill-down detail per month.
- **LLM Document Summaries** -- Generate short descriptions for every document using the LLM (parallelized, ~25 min for 2,000 docs). Summaries appear in the timeline detail view.
- **Editable Prompts** -- All LLM prompts (map, reduce, correlation, summary) are fully editable in the Settings tab. Customize what the AI looks for and how it responds.
- **Document Management** -- Documents are grouped by search query. Delete a search set and only orphaned documents (not shared with other sets) are removed.
- **Checkpointing** -- Long-running analyses save progress after every batch. If the process crashes, it resumes from the last completed batch instead of starting over.
- **Local Full-Text Search** -- Whoosh-based index of all extracted text for instant local search.
- **Real-Time Progress** -- WebSocket-based streaming of search progress, download status, and LLM analysis (batch-by-batch with time estimates).
- **Cancel Anytime** -- Long-running analyses can be cancelled at any batch boundary.
- **Everything Local** -- No cloud APIs, no data leaves your machine. Ollama runs locally.

## Platform

This tool is developed and tested on **macOS only**. It has not been tested on Windows or Linux. The code is all open and portable in principle -- users on other platforms are welcome to adapt it, but no support is provided.

## System Requirements

These are system-level dependencies that must be installed separately from the Python packages.

| Dependency | Purpose | Install (macOS) |
|---|---|---|
| **Python 3.10+** | Runtime | Usually pre-installed; or `brew install python` |
| **Tesseract OCR** | OCR for scanned PDFs | `brew install tesseract` |
| **Google Chrome** | Browser for DOJ site search | [google.com/chrome](https://www.google.com/chrome/) |
| **Ollama** | Local LLM runtime | [ollama.com](https://ollama.com/) |

### Ollama Models

After installing Ollama, pull the required models:

```bash
ollama pull llama3.1:8b        # LLM for analysis (~4.7 GB)
ollama pull nomic-embed-text   # Embeddings for correlation search (~274 MB)
```

## Installation

```bash
git clone https://github.com/artmedlar/epstein-files-analyzer.git
cd epstein-files-analyzer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate
python run.py
```

This starts a local web server and opens the UI in your default browser at `http://127.0.0.1:8742`. Press Ctrl+C to stop.

To use a custom port:

```bash
python run.py --port 9000
```

### Typical Workflow

1. **Search**: Enter a name (e.g., "Leon Black") in the Search tab. The tool opens a Chrome window, navigates the DOJ site to pass bot protection, then paginates through the DOJ API collecting all matching document URLs.

2. **Download**: Click "Download All" (or select specific documents). PDFs are downloaded in parallel, text is extracted, and the PDFs are deleted. Only the extracted text and DOJ URLs are kept.

3. **Analyze**: Switch to the Analysis tab, enter the same name, and click "Analyze". The map-reduce pipeline processes every downloaded document:
   - Documents are grouped into batches of ~12,000 characters
   - Each batch is sent to the LLM for key fact extraction
   - All batch findings are synthesized into a comprehensive report
   - Progress is shown in real-time with time estimates
   - If it crashes, it resumes from the last checkpoint

4. **Correlate**: Enter two or more names separated by commas in the correlation input (e.g., "Leon Black, Alan Dershowitz"). The tool scans all extracted documents for co-occurrences and analyzes their connections.

5. **Timeline**: After analysis, a timeline bar chart appears below the results. Click any bar to see the documents from that period, with LLM-generated descriptions. Click "Generate Document Summaries" to create descriptions for all documents.

6. **Browse**: The Documents tab lists all downloaded documents grouped by search query. Click any document to view its full extracted text. Use "Search Local" for instant full-text search. Use the "Remove" button to delete a search set (shared documents are preserved).

7. **Customize Prompts**: The Settings tab lets you edit the LLM prompts that control what the analysis looks for. Each prompt uses `{placeholders}` for runtime data -- an info button explains how they work. Click "Reset" to restore defaults.

## Performance Notes

Analysis time depends on your hardware and the number of documents:

| Documents | Batches | Approx. Time |
|---|---|---|
| 50 | ~8 | ~10 minutes |
| 500 | ~50 | ~45 minutes |
| 1,500 | ~260 | ~4 hours |

The LLM processing speed is the bottleneck. Each batch takes approximately 50-60 seconds on a modern MacBook (CPU inference with llama3.1:8b). A machine with a GPU or more RAM will be significantly faster.

The analysis can be cancelled at any time via the Cancel button. If it crashes, it will resume from the last checkpoint when re-run. The map-reduce approach means the LLM actually reads every document -- there are no shortcuts or sampling that would miss information.

## What Gets Stored

Everything lives in the `data/` directory (created at runtime, excluded from git):

| Path | Contents |
|---|---|
| `data/text/` | Extracted text files (~50 KB per document) |
| `data/index/` | Whoosh full-text search index |
| `data/checkpoints/` | Analysis checkpoint files (auto-cleaned after completion) |
| `data/epstein.db` | SQLite database: document metadata, search history, analysis results, settings |
| `data/tmp_pdfs/` | Temporary PDFs during download (deleted after text extraction) |

**PDFs are not kept.** After text extraction, the PDF is deleted. The DOJ URL is stored so you can re-download or view the original at any time.

## Project Structure

```
epstein-files-analyzer/
├── run.py                          # Entry point
├── requirements.txt                # Python dependencies
├── LICENSE                         # Unlicense (public domain)
├── src/epstein_analyzer/
│   ├── config.py                   # All settings and paths
│   ├── database.py                 # SQLite operations + settings
│   ├── models.py                   # Pydantic data models
│   ├── search.py                   # DOJ search (Selenium) + local Whoosh index
│   ├── downloader.py               # Multi-threaded PDF downloader
│   ├── extractor.py                # PDF text extraction + OCR
│   ├── analyzer.py                 # Map-reduce LLM analysis + correlation + summaries
│   ├── dates.py                    # Date extraction from document text
│   └── app.py                      # FastAPI backend (REST + WebSocket)
├── frontend/
│   ├── index.html                  # Single-page application
│   ├── css/styles.css              # Dark theme
│   └── js/app.js                   # Frontend logic + WebSocket handling
└── data/                           # Created at runtime (not in git)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed explanation of the data flow and internal operations.

## Credits

This tool incorporates patterns from several community projects:

- [the-files-easy](https://github.com/socially-truthful/the-files-easy) -- RAG pipeline, search indexing, PDF extraction
- [DOJ PDF Downloader](https://pastebin.com/vDMNN0cS) -- Selenium pagination and cookie handling
- [efgrabber](https://github.com/segin/efgrabber) -- Age verification cookie discovery, file ID schemes

## License

This is free and unencumbered software released into the public domain. See [LICENSE](LICENSE) for details.

## Disclaimer

This tool is for research and educational purposes. All documents are publicly released by the U.S. Department of Justice. The tool does not modify any documents. AI-generated analyses may contain errors and should always be verified against source documents. Users are responsible for ethical use.
