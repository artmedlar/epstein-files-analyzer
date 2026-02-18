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
        """)

        # Add date columns if they don't exist yet (migration-safe)
        for col, coltype in [("document_date", "TEXT"),
                             ("date_confidence", "TEXT"),
                             ("snippet", "TEXT")]:
            try:
                conn.execute(
                    f"ALTER TABLE documents ADD COLUMN {col} {coltype}"
                )
            except sqlite3.OperationalError:
                pass  # column already exists

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS document_queries (
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (document_id, query)
            );
            CREATE INDEX IF NOT EXISTS idx_dq_query ON document_queries(query);

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

        # Migrate: populate document_queries from found_via_query for any
        # documents not yet in the junction table.
        conn.execute("""
            INSERT OR IGNORE INTO document_queries (document_id, query, created_at)
            SELECT id, found_via_query, created_at
            FROM documents
            WHERE found_via_query IS NOT NULL
              AND id NOT IN (SELECT document_id FROM document_queries)
        """)


# --- Document CRUD ---

def insert_document(doc: Document) -> int:
    """Insert a document, returning its ID.

    If the URL already exists, links the existing document to the new
    query (many-to-many via document_queries) and returns the existing ID.
    """
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
            doc_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            row = conn.execute(
                "SELECT id FROM documents WHERE url = ?", (doc.url,)
            ).fetchone()
            doc_id = row["id"] if row else -1

        if doc_id > 0 and doc.found_via_query:
            conn.execute(
                """INSERT OR IGNORE INTO document_queries
                   (document_id, query, created_at) VALUES (?, ?, ?)""",
                (doc_id, doc.found_via_query, now)
            )

        return doc_id


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
    """Get all documents linked to a specific search query."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT d.* FROM documents d
               JOIN document_queries dq ON d.id = dq.document_id
               WHERE dq.query = ?
               ORDER BY d.filename""",
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
    """Get all documents with their query memberships."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
        docs = [dict(r) for r in rows]

        # Attach list of queries to each document
        for doc in docs:
            qrows = conn.execute(
                "SELECT query FROM document_queries WHERE document_id = ?",
                (doc["id"],)
            ).fetchall()
            doc["queries"] = [r["query"] for r in qrows]

        return docs


def get_document_count() -> int:
    """Get total number of documents."""
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
        return row["cnt"]


def get_query_list() -> list[dict]:
    """Get all distinct queries with their document counts."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT query, COUNT(*) as doc_count
               FROM document_queries
               GROUP BY query
               ORDER BY query"""
        ).fetchall()
        return [dict(r) for r in rows]


def delete_documents_by_query(query: str) -> dict:
    """Unlink all documents from a query, then garbage-collect any
    documents that no longer belong to any query.

    Returns {"unlinked": N, "deleted": M} where M is the number of
    documents actually removed from disk because they had no remaining
    query associations.
    """
    from .config import TEXT_DIR

    with get_db() as conn:
        # Find documents currently linked to this query
        linked = conn.execute(
            "SELECT document_id FROM document_queries WHERE query = ?",
            (query,)
        ).fetchall()
        linked_ids = [r["document_id"] for r in linked]

        if not linked_ids:
            return {"unlinked": 0, "deleted": 0}

        # Remove the query links
        conn.execute(
            "DELETE FROM document_queries WHERE query = ?", (query,)
        )

        # Find which of those documents are now orphaned (no remaining links)
        placeholders = ",".join("?" * len(linked_ids))
        orphans = conn.execute(
            f"""SELECT id, text_path FROM documents
                WHERE id IN ({placeholders})
                  AND id NOT IN (SELECT document_id FROM document_queries)""",
            linked_ids
        ).fetchall()

        # Delete orphaned text files from disk
        for row in orphans:
            if row["text_path"]:
                text_file = TEXT_DIR / row["text_path"]
                if text_file.exists():
                    text_file.unlink()

        # Delete orphaned document rows
        orphan_ids = [r["id"] for r in orphans]
        if orphan_ids:
            ph = ",".join("?" * len(orphan_ids))
            conn.execute(
                f"DELETE FROM documents WHERE id IN ({ph})", orphan_ids
            )

        return {"unlinked": len(linked_ids), "deleted": len(orphan_ids)}


def url_exists(url: str) -> bool:
    """Check if a URL is already in the database."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE url = ?", (url,)
        ).fetchone()
        return row is not None


def update_document_date(doc_id: int, document_date: str,
                         date_confidence: str,
                         snippet: Optional[str] = None):
    """Set the extracted date and optional snippet for a document."""
    now = datetime.now().isoformat()
    with get_db() as conn:
        if snippet is not None:
            conn.execute(
                "UPDATE documents SET document_date = ?, date_confidence = ?, "
                "snippet = ?, updated_at = ? WHERE id = ?",
                (document_date, date_confidence, snippet, now, doc_id)
            )
        else:
            conn.execute(
                "UPDATE documents SET document_date = ?, date_confidence = ?, "
                "updated_at = ? WHERE id = ?",
                (document_date, date_confidence, now, doc_id)
            )


def update_document_snippet(doc_id: int, snippet: str):
    """Set just the snippet for a document."""
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "UPDATE documents SET snippet = ?, updated_at = ? WHERE id = ?",
            (snippet, now, doc_id)
        )


def get_timeline_data(query: Optional[str] = None) -> list[dict]:
    """Get documents with dates for timeline display."""
    with get_db() as conn:
        if query:
            rows = conn.execute(
                """SELECT d.id, d.filename, d.document_date,
                          d.date_confidence, d.snippet
                   FROM documents d
                   JOIN document_queries dq ON d.id = dq.document_id
                   WHERE dq.query = ? AND d.document_date IS NOT NULL
                   ORDER BY d.document_date""",
                (query,)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, filename, document_date, date_confidence, snippet
                   FROM documents
                   WHERE document_date IS NOT NULL
                   ORDER BY document_date"""
            ).fetchall()
        return [dict(r) for r in rows]


def get_undated_document_count(query: Optional[str] = None) -> int:
    """Count extracted documents that have no date assigned."""
    with get_db() as conn:
        if query:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM documents d
                   JOIN document_queries dq ON d.id = dq.document_id
                   WHERE dq.query = ? AND d.status = 'extracted'
                   AND d.document_date IS NULL""",
                (query,)
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM documents
                   WHERE status = 'extracted' AND document_date IS NULL"""
            ).fetchone()
        return row["cnt"]


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
