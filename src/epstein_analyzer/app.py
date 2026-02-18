"""FastAPI backend with REST API and WebSocket support."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import database, analyzer, dates
from .config import FRONTEND_DIR, TEXT_DIR, WEB_HOST, WEB_PORT
from .models import DocumentStatus, SearchProgress

logger = logging.getLogger(__name__)

app = FastAPI(title="Epstein Analyzer", version="0.1.0")

# Only one analysis can run at a time (protects Ollama from concurrent load)
_analysis_lock = asyncio.Lock()

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    database.init_db()


@app.get("/")
async def index():
    """Serve the main frontend page."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Status endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Get system status including Ollama availability."""
    ollama_status = analyzer.check_ollama()
    doc_count = database.get_document_count()
    return {
        "ollama": ollama_status.model_dump(),
        "document_count": doc_count,
        "index_available": (FRONTEND_DIR.parent / "data" / "index").exists(),
    }


@app.get("/api/history")
async def get_history(limit: int = 20):
    """Get recent search history."""
    return database.get_search_history(limit)


# ---------------------------------------------------------------------------
# Search endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws/search")
async def ws_search(websocket: WebSocket):
    """
    WebSocket endpoint for DOJ search with real-time progress.

    Client sends: {"query": "person name", "max_pages": 100}
    Server sends progress updates and final results.
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data.get("query", "").strip()
        max_pages = data.get("max_pages", 100)

        if not query:
            await websocket.send_json({"type": "error", "message": "Empty query"})
            return

        async def send_progress(progress: SearchProgress):
            await websocket.send_json({
                "type": "progress",
                "data": progress.model_dump()
            })

        # Run the synchronous search in a thread
        from .search import search_doj
        loop = asyncio.get_event_loop()

        # We need a sync-to-async bridge for the callback
        progress_queue = asyncio.Queue()

        def sync_progress_callback(progress: SearchProgress):
            loop.call_soon_threadsafe(progress_queue.put_nowait, progress)

        # Start search in background thread
        search_task = loop.run_in_executor(
            None, lambda: search_doj(
                query, max_pages=max_pages,
                progress_callback=sync_progress_callback
            )
        )

        # Forward progress updates to websocket
        while not search_task.done():
            try:
                progress = await asyncio.wait_for(
                    progress_queue.get(), timeout=0.5
                )
                await websocket.send_json({
                    "type": "progress",
                    "data": progress.model_dump()
                })
            except asyncio.TimeoutError:
                continue

        # Drain any remaining progress messages
        while not progress_queue.empty():
            progress = progress_queue.get_nowait()
            await websocket.send_json({
                "type": "progress", "data": progress.model_dump()
            })

        results = await search_task

        # Store results in database
        for result in results:
            doc = database.get_document_by_url(result.url)
            if not doc:
                from .models import Document
                database.insert_document(Document(
                    filename=result.filename,
                    url=result.url,
                    data_set=result.data_set,
                    found_via_query=query,
                ))

        database.log_search(query, len(results))

        await websocket.send_json({
            "type": "results",
            "data": {
                "query": query,
                "count": len(results),
                "documents": [r.model_dump() for r in results],
            }
        })

    except WebSocketDisconnect:
        logger.info("Search WebSocket disconnected")
    except Exception as e:
        logger.error(f"Search WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/api/search/local")
async def search_local_endpoint(q: str = Query(...), limit: int = 50):
    """Search the local Whoosh index."""
    from .search import search_local
    results = search_local(q, limit=limit)
    return {"query": q, "count": len(results), "results": results}


# ---------------------------------------------------------------------------
# Download + Extract endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws/download")
async def ws_download(websocket: WebSocket):
    """
    WebSocket endpoint for downloading and extracting documents.

    Client sends: {"urls": [...], "query": "original query"}
    Server sends progress updates.
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        urls = data.get("urls", [])
        query = data.get("query", "")

        if not urls:
            await websocket.send_json({"type": "error", "message": "No URLs provided"})
            return

        loop = asyncio.get_event_loop()
        progress_queue = asyncio.Queue()

        def sync_progress(progress: SearchProgress):
            loop.call_soon_threadsafe(progress_queue.put_nowait, progress)

        # Build SearchResult objects
        from .models import SearchResult
        search_results = []
        for url in urls:
            filename = url.split("/")[-1].split("?")[0]
            search_results.append(SearchResult(filename=filename, url=url))

        # Download
        from .downloader import download_documents
        dl_task = loop.run_in_executor(
            None, lambda: download_documents(
                search_results, progress_callback=sync_progress
            )
        )

        while not dl_task.done():
            try:
                progress = await asyncio.wait_for(
                    progress_queue.get(), timeout=0.5
                )
                await websocket.send_json({
                    "type": "progress", "data": progress.model_dump()
                })
            except asyncio.TimeoutError:
                continue

        dl_results = await dl_task

        # Extract text
        from .extractor import process_downloaded_documents
        ext_task = loop.run_in_executor(
            None, lambda: process_downloaded_documents(
                dl_results, progress_callback=sync_progress
            )
        )

        while not ext_task.done():
            try:
                progress = await asyncio.wait_for(
                    progress_queue.get(), timeout=0.5
                )
                await websocket.send_json({
                    "type": "progress", "data": progress.model_dump()
                })
            except asyncio.TimeoutError:
                continue

        ext_results = await ext_task

        # Update database
        for ext in ext_results:
            if "error" in ext:
                continue
            doc = database.get_document_by_url(ext["url"])
            if doc:
                database.update_document_status(
                    doc["id"],
                    DocumentStatus.EXTRACTED,
                    text_path=ext.get("text_path"),
                    page_count=ext.get("page_count"),
                    ocr_needed=ext.get("ocr_needed"),
                    ocr_confidence=ext.get("ocr_confidence"),
                )

        # Extract dates from newly extracted documents
        extracted_doc_ids = []
        for ext in ext_results:
            if "error" in ext:
                continue
            doc = database.get_document_by_url(ext["url"])
            if doc:
                extracted_doc_ids.append(doc["id"])
        if extracted_doc_ids:
            await loop.run_in_executor(
                None, lambda: dates.extract_dates_for_documents(extracted_doc_ids)
            )

        # Rebuild local search index
        from .search import build_whoosh_index
        await loop.run_in_executor(None, build_whoosh_index)

        await websocket.send_json({
            "type": "complete",
            "data": {
                "downloaded": sum(1 for r in dl_results if r["success"]),
                "extracted": sum(1 for r in ext_results if "error" not in r),
                "failed": sum(1 for r in dl_results if not r["success"]),
            }
        })

    except WebSocketDisconnect:
        logger.info("Download WebSocket disconnected")
    except Exception as e:
        logger.error(f"Download WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Analysis endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for LLM analysis with streaming.

    Client sends: {"query": "person name", "doc_ids": [1,2,3]}
    Server streams analysis tokens.

    Only one analysis can run at a time.  If another is already in
    progress, the new connection is rejected with an error message.
    """
    await websocket.accept()

    if _analysis_lock.locked():
        await websocket.send_json({
            "type": "error",
            "content": "An analysis is already running. Wait for it to finish or cancel it first."
        })
        await websocket.close()
        logger.warning("Rejected concurrent analysis request")
        return

    async with _analysis_lock:
        await _run_analysis(websocket)


async def _run_analysis(websocket: WebSocket):
    """Inner analysis handler, called while holding _analysis_lock."""
    try:
        data = await websocket.receive_json()
        query = data.get("query", "").strip()
        doc_ids = data.get("doc_ids", [])
        correlation_names = data.get("correlation_names", [])

        if not query and not correlation_names:
            await websocket.send_json({"type": "error", "content": "No query provided"})
            return

        if not doc_ids:
            if correlation_names and len(correlation_names) >= 2:
                # Correlation always searches ALL extracted documents.
                # The correlation function does its own keyword scanning
                # to find chunks where the named people appear.
                all_docs = database.get_extracted_documents()
                doc_ids = [d["id"] for d in all_docs]
            else:
                # Use all extracted documents matching the single query
                docs = database.get_documents_by_query(query)
                doc_ids = [
                    d["id"] for d in docs
                    if d["status"] == DocumentStatus.EXTRACTED.value
                ]

        if not doc_ids:
            await websocket.send_json({
                "type": "error",
                "content": "No extracted documents found. Download and extract documents first."
            })
            return

        import threading

        loop = asyncio.get_event_loop()
        item_queue: asyncio.Queue = asyncio.Queue()
        cancel_flag = threading.Event()

        if correlation_names and len(correlation_names) >= 2:
            gen_factory = lambda: analyzer.analyze_correlation(
                correlation_names, doc_ids, stream=True,
                cancel_flag=cancel_flag,
            )
        else:
            gen_factory = lambda: analyzer.analyze_documents(
                query, doc_ids, stream=True,
                cancel_flag=cancel_flag,
            )

        # Run generator in a background thread, push items to queue
        def run_generator():
            try:
                for item in gen_factory():
                    if cancel_flag.is_set():
                        break
                    loop.call_soon_threadsafe(item_queue.put_nowait, item)
            except Exception as e:
                if not cancel_flag.is_set():
                    loop.call_soon_threadsafe(
                        item_queue.put_nowait,
                        {"type": "error", "content": str(e)}
                    )
            finally:
                loop.call_soon_threadsafe(
                    item_queue.put_nowait, None  # sentinel
                )

        gen_task = loop.run_in_executor(None, run_generator)

        # Cancel detection: the client closes the WebSocket to cancel.
        # We detect this via a background receive task that only looks
        # for disconnection (no concurrent reads issue since Starlette
        # allows one pending receive).
        async def watch_disconnect():
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") in ("websocket.disconnect",):
                        cancel_flag.set()
                        break
                    # If client sends a JSON cancel message
                    if msg.get("type") == "websocket.receive":
                        text = msg.get("text", "")
                        try:
                            parsed = json.loads(text)
                            if parsed.get("type") == "cancel":
                                cancel_flag.set()
                                break
                        except Exception:
                            pass
            except Exception:
                cancel_flag.set()

        disconnect_task = asyncio.create_task(watch_disconnect())

        # Stream items to the client — NO hard timeout.
        # Heartbeat every 15s keeps the connection alive.
        HEARTBEAT_SECS = 15

        while True:
            try:
                item = await asyncio.wait_for(
                    item_queue.get(), timeout=HEARTBEAT_SECS
                )
            except asyncio.TimeoutError:
                if cancel_flag.is_set():
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "content": "Analysis cancelled"
                        })
                    except Exception:
                        pass
                    break
                # Heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    cancel_flag.set()
                    break
                continue

            if item is None:
                break
            try:
                await websocket.send_json(item)
            except Exception:
                cancel_flag.set()
                break

        cancel_flag.set()
        disconnect_task.cancel()
        await gen_task

    except WebSocketDisconnect:
        logger.info("Analysis WebSocket disconnected by client")
    except Exception as e:
        logger.error("Analysis WebSocket error: %s", e, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------

@app.get("/api/documents")
async def list_documents(query: Optional[str] = None,
                         status: Optional[str] = None):
    """List documents, optionally filtered by query or status."""
    if query:
        docs = database.get_documents_by_query(query)
    elif status:
        docs = database.get_documents_by_status(DocumentStatus(status))
    else:
        docs = database.get_all_documents()
    return {"count": len(docs), "documents": docs}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: int):
    """Get a single document's details."""
    doc = database.get_document_by_id(doc_id)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return doc


@app.get("/api/documents/{doc_id}/text")
async def get_document_text(doc_id: int):
    """Get the extracted text for a document."""
    doc = database.get_document_by_id(doc_id)
    if not doc or not doc.get("text_path"):
        return JSONResponse(status_code=404, content={"error": "Text not available"})

    text_path = TEXT_DIR / doc["text_path"]
    if not text_path.exists():
        return JSONResponse(status_code=404, content={"error": "Text file missing"})

    content = text_path.read_text(encoding="utf-8", errors="replace")
    return {"doc_id": doc_id, "filename": doc["filename"], "text": content}


@app.delete("/api/documents/query/{query}")
async def delete_documents_by_query_endpoint(query: str):
    """Remove a query's association with its documents.

    Documents that still belong to other queries are kept.
    Documents with no remaining query associations are deleted from disk.
    """
    result = database.delete_documents_by_query(query)
    if result["unlinked"] == 0:
        return JSONResponse(status_code=404,
                            content={"error": f"No documents found for query '{query}'"})
    if result["deleted"] > 0:
        loop = asyncio.get_event_loop()
        from .search import build_whoosh_index
        await loop.run_in_executor(None, build_whoosh_index)
    return {"query": query, **result}


@app.get("/api/analyses")
async def list_analyses(query: Optional[str] = None):
    """List analyses, optionally filtered by query."""
    if query:
        analyses = database.get_analyses_by_query(query)
    else:
        analyses = []
    return {"count": len(analyses), "analyses": analyses}


# ---------------------------------------------------------------------------
# Timeline endpoints
# ---------------------------------------------------------------------------

@app.get("/api/timeline")
async def get_timeline(query: Optional[str] = None):
    """Get timeline data for documents.

    Returns dated documents and counts of undated ones for display.
    Optionally filtered by search query.
    """
    timeline_docs = database.get_timeline_data(query)
    undated_count = database.get_undated_document_count(query)

    total_with_dates = len(timeline_docs)

    # Count by confidence
    high_count = sum(1 for d in timeline_docs if d["date_confidence"] == "high")
    medium_count = sum(1 for d in timeline_docs if d["date_confidence"] == "medium")

    return {
        "documents": timeline_docs,
        "total_dated": total_with_dates,
        "high_confidence": high_count,
        "medium_confidence": medium_count,
        "undated": undated_count,
        "total": total_with_dates + undated_count,
    }


@app.post("/api/timeline/backfill")
async def backfill_dates():
    """Run date extraction on all extracted documents that lack a date.

    This is a one-time catch-up for documents downloaded before the
    date extraction feature was added.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, dates.extract_dates_for_documents)
    return result


# ---------------------------------------------------------------------------
# Prompt settings endpoints
# ---------------------------------------------------------------------------

@app.get("/api/prompts")
async def get_prompts():
    """Return all prompt templates (user-customized or defaults)."""
    from .analyzer import PROMPT_KEYS, _get_prompt
    result = {}
    for key in PROMPT_KEYS:
        result[key] = {
            "value": _get_prompt(key),
            "is_custom": database.get_setting(key) is not None,
            "default": PROMPT_KEYS[key],
        }
    return result


@app.put("/api/prompts/{key}")
async def set_prompt(key: str, body: dict):
    """Save a customized prompt template."""
    from .analyzer import PROMPT_KEYS
    if key not in PROMPT_KEYS:
        return JSONResponse(status_code=400,
                            content={"error": f"Unknown prompt key: {key}"})
    value = body.get("value", "").strip()
    if not value:
        return JSONResponse(status_code=400,
                            content={"error": "Prompt cannot be empty"})
    database.set_setting(key, value)
    return {"key": key, "saved": True}


@app.delete("/api/prompts/{key}")
async def reset_prompt(key: str):
    """Reset a prompt template to its default."""
    from .analyzer import PROMPT_KEYS
    if key not in PROMPT_KEYS:
        return JSONResponse(status_code=400,
                            content={"error": f"Unknown prompt key: {key}"})
    database.delete_setting(key)
    return {"key": key, "reset": True, "default": PROMPT_KEYS[key]}


@app.websocket("/ws/summarize")
async def ws_summarize(websocket: WebSocket):
    """WebSocket endpoint for LLM-based document summarization.

    Uses the LLM to generate a short description for every document
    that doesn't have one yet.  Streams progress back to the client.
    """
    await websocket.accept()

    if _analysis_lock.locked():
        await websocket.send_json({
            "type": "error",
            "content": "An analysis is already running. Wait for it to finish."
        })
        await websocket.close()
        return

    async with _analysis_lock:
        import threading
        cancel_event = threading.Event()
        loop = asyncio.get_event_loop()

        # Listen for cancel messages in background
        async def listen_for_cancel():
            try:
                while True:
                    msg = await websocket.receive_json()
                    if msg.get("type") == "cancel":
                        cancel_event.set()
                        break
            except (WebSocketDisconnect, Exception):
                cancel_event.set()

        cancel_task = asyncio.create_task(listen_for_cancel())

        result_queue: asyncio.Queue = asyncio.Queue()

        def run_summarize():
            for update in analyzer.summarize_documents(
                stream=True, cancel_flag=cancel_event
            ):
                loop.call_soon_threadsafe(result_queue.put_nowait, update)
            loop.call_soon_threadsafe(
                result_queue.put_nowait, None  # sentinel
            )

        try:
            task = loop.run_in_executor(None, run_summarize)

            while True:
                item = await result_queue.get()
                if item is None:
                    break
                await websocket.send_json(item)

            await task
        except Exception as e:
            logger.error("Summarize error: %s", e)
            try:
                await websocket.send_json({
                    "type": "error", "content": str(e)
                })
            except Exception:
                pass
        finally:
            cancel_task.cancel()


