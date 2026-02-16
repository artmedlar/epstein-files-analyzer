"""SQLite database for tracking documents, extractions, and analyses."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import DB_PATH
from .models import AnalysisResult, Document, DocumentStatus


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create database tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                data_set TEXT,
                status TEXT NOT NULL DEFAULT 'found',
                text_path TEXT,
                page_count INTEGER,
                ocr_needed INTEGER DEFAULT 0,
                ocr_confidence REAL,
                file_hash TEXT,
                found_via_query TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_documents_query ON documents(found_via_query);

            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                document_ids TEXT NOT NULL,
                analysis_text TEXT NOT NULL,
                dates_found TEXT,
                people_found TEXT,
                locations_found TEXT,
                document_types TEXT,
                model_used TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_analyses_query ON analyses(query);

            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                results_count INTEGER NOT NULL DEFAULT 0,
                searched_at TEXT NOT NULL
            );
        """)


# --- Document CRUD ---

def insert_document(doc: Document) -> int:
    """Insert a document, returning its ID. Skips if URL already exists."""
    now = datetime.now().isoformat()
    with get_db() as conn:
        try:
            cursor = conn.execute(
                """INSERT INTO documents
                   (filename, url, data_set, status, text_path, page_count,
                    ocr_needed, ocr_confidence, file_hash, found_via_query,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc.filename, doc.url, doc.data_set, doc.status.value,
                 doc.text_path, doc.page_count, int(doc.ocr_needed),
                 doc.ocr_confidence, doc.file_hash, doc.found_via_query,
                 now, now)
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # URL already exists, return existing ID
            row = conn.execute(
                "SELECT id FROM documents WHERE url = ?", (doc.url,)
            ).fetchone()
            return row["id"] if row else -1


def update_document_status(doc_id: int, status: DocumentStatus,
                           text_path: Optional[str] = None,
                           page_count: Optional[int] = None,
                           ocr_needed: Optional[bool] = None,
                           ocr_confidence: Optional[float] = None):
    """Update a document's status and optional fields."""
    now = datetime.now().isoformat()
    with get_db() as conn:
        updates = ["status = ?", "updated_at = ?"]
        params: list = [status.value, now]

        if text_path is not None:
            updates.append("text_path = ?")
            params.append(text_path)
        if page_count is not None:
            updates.append("page_count = ?")
            params.append(page_count)
        if ocr_needed is not None:
            updates.append("ocr_needed = ?")
            params.append(int(ocr_needed))
        if ocr_confidence is not None:
            updates.append("ocr_confidence = ?")
            params.append(ocr_confidence)

        params.append(doc_id)
        conn.execute(
            f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
            params
        )


def get_document_by_url(url: str) -> Optional[dict]:
    """Look up a document by URL."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE url = ?", (url,)
        ).fetchone()
        return dict(row) if row else None


def get_document_by_id(doc_id: int) -> Optional[dict]:
    """Look up a document by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None


def get_documents_by_query(query: str) -> list[dict]:
    """Get all documents found by a specific search query."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE found_via_query = ? ORDER BY filename",
            (query,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_documents_by_status(status: DocumentStatus) -> list[dict]:
    """Get all documents with a given status."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE status = ? ORDER BY filename",
            (status.value,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_extracted_documents() -> list[dict]:
    """Get all documents that have been text-extracted."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE status = 'extracted' ORDER BY filename"
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_documents() -> list[dict]:
    """Get all documents."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_document_count() -> int:
    """Get total number of documents."""
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
        return row["cnt"]


def url_exists(url: str) -> bool:
    """Check if a URL is already in the database."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE url = ?", (url,)
        ).fetchone()
        return row is not None


# --- Analysis CRUD ---

def insert_analysis(result: AnalysisResult) -> int:
    """Insert an analysis result."""
    now = datetime.now().isoformat()
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO analyses
               (query, document_ids, analysis_text, dates_found,
                people_found, locations_found, document_types,
                model_used, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result.query,
             ",".join(str(i) for i in result.document_ids),
             result.analysis_text,
             ",".join(result.dates_found),
             ",".join(result.people_found),
             ",".join(result.locations_found),
             ",".join(result.document_types),
             result.model_used, now)
        )
        return cursor.lastrowid


def get_analyses_by_query(query: str) -> list[dict]:
    """Get all analyses for a query."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE query = ? ORDER BY created_at DESC",
            (query,)
        ).fetchall()
        return [dict(r) for r in rows]


# --- Search History ---

def log_search(query: str, results_count: int):
    """Log a search query."""
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO search_history (query, results_count, searched_at) VALUES (?, ?, ?)",
            (query, results_count, now)
        )


def get_search_history(limit: int = 20) -> list[dict]:
    """Get recent search history."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM search_history ORDER BY searched_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
