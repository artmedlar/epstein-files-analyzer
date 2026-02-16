# DOJ-UNFUK, an Epstein Files Analyzer

Search, download, and analyze documents from the DOJ Epstein Files release using a local LLM.

This tool automates searching the DOJ's Epstein document library at [justice.gov/epstein](https://www.justice.gov/epstein), downloads matching documents, extracts text (with OCR for scanned pages), and runs comprehensive analysis using a local Ollama LLM -- all through a desktop or browser UI.

## Features

- **DOJ Search** -- Automated search across all 12 DOJ Epstein data sets (3+ million pages) using the DOJ's own search API, with Selenium to bypass Akamai bot protection.
- **Batch Download** -- Download all matching documents or select specific ones. Multi-threaded with retry logic.
- **Text Extraction** -- PyMuPDF for digital text, Tesseract OCR fallback for scanned pages. PDFs are deleted after extraction to save disk space.
- **Map-Reduce Analysis** -- Feed ALL downloaded documents to your local LLM. Documents are batched, each batch is analyzed for key findings, then all findings are synthesized into a comprehensive report. No arbitrary limits on how many documents are analyzed.
- **Correlation Analysis** -- Search for multiple names to find documents where they co-occur and analyze their cross-connections.
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
# Clone the repo
mkdir epstein-analyzer
cd epstein-analyzer
git clone https://github.com/YOUR_USERNAME/epstein-analyzer.git


# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Desktop App

```bash
source .venv/bin/activate
python run.py
```

Opens a native desktop window (via pywebview) with the full GUI.

### Web Server Mode

```bash
python run.py --web
```

Starts a web server and opens the UI in your default browser at `http://127.0.0.1:8742`. Useful if pywebview has issues on your system.

### Custom Port

```bash
python run.py --web --port 9000
```

### Typical Workflow

1. **Search**: Enter a name (e.g., "Martin Nowak") in the Search tab. The tool opens a Chrome window, navigates the DOJ site to pass bot protection, then paginates through the DOJ API collecting all matching document URLs.

2. **Download**: Click "Download All" (or select specific documents). PDFs are downloaded in parallel, text is extracted, and the PDFs are deleted. Only the extracted text and DOJ URLs are kept.

3. **Analyze**: Switch to the Analysis tab, enter the same name, and click "Analyze". The map-reduce pipeline processes every downloaded document:
   - Documents are grouped into batches of ~12,000 characters
   - Each batch is sent to the LLM for key fact extraction
   - All batch findings are synthesized into a comprehensive report
   - Progress is shown in real-time with time estimates

4. **Correlate**: Enter two or more names separated by commas in the correlation input (e.g., "Leon Black, Alan Dershowitz"). The tool finds documents where both names appear and analyzes their connections.

5. **Browse**: The Documents tab lists all downloaded documents. Click any to view its full extracted text. Use "Search Local" for instant full-text search across all extracted documents.

## Performance Notes

Analysis time depends on your hardware and the number of documents:

| Documents | Batches | Approx. Time |
|---|---|---|
| 50 | ~8 | ~10 minutes |
| 500 | ~50 | ~45 minutes |
| 1,500 | ~260 | ~4 hours |

The LLM processing speed is the bottleneck. Each batch takes approximately 50-60 seconds on a modern MacBook (CPU inference with llama3.1:8b). A machine with a GPU or more RAM will be significantly faster.

The analysis can be cancelled at any time via the Cancel button. The map-reduce approach means the LLM actually reads every document -- there are no shortcuts or sampling that would miss information.

## What Gets Stored

Everything lives in the `data/` directory (created at runtime, excluded from git):

| Path | Contents |
|---|---|
| `data/text/` | Extracted text files (~50 KB per document) |
| `data/index/` | Whoosh full-text search index |
| `data/epstein.db` | SQLite database: document metadata, search history, analysis results |
| `data/tmp_pdfs/` | Temporary PDFs during download (deleted after text extraction) |

**PDFs are not kept.** After text extraction, the PDF is deleted. The DOJ URL is stored so you can re-download or view the original at any time.

## Project Structure

```
epstein-analyzer/
├── run.py                          # Entry point
├── requirements.txt                # Python dependencies
├── LICENSE                         # Unlicense (public domain)
├── src/epstein_analyzer/
│   ├── config.py                   # All settings and paths
│   ├── database.py                 # SQLite operations
│   ├── models.py                   # Pydantic data models
│   ├── search.py                   # DOJ search (Selenium) + local Whoosh index
│   ├── downloader.py               # Multi-threaded PDF downloader
│   ├── extractor.py                # PDF text extraction + OCR
│   ├── analyzer.py                 # Map-reduce LLM analysis + correlation
│   ├── app.py                      # FastAPI backend (REST + WebSocket)
│   └── desktop.py                  # pywebview desktop wrapper
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
