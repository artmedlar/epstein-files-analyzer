"""LLM analysis pipeline using Ollama with map-reduce over ALL documents.

Includes checkpoint support: batch summaries are saved to a JSON file
after each batch completes, so a crash only loses the current batch
(~60 seconds) instead of the entire multi-hour analysis.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .config import (
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL,
    TEXT_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    CHECKPOINT_DIR,
    LLM_CTX_TOKENS, LLM_BATCH_CHARS,
    LLM_MAP_PREDICT, LLM_REDUCE_CTX, LLM_REDUCE_PREDICT,
)
from .models import AnalysisResult, OllamaStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def check_ollama() -> OllamaStatus:
    """Check if Ollama is running and which models are available."""
    status = OllamaStatus()
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            status.available = True
            models = resp.json().get("models", [])
            status.models = [m.get("name", "").split(":")[0] for m in models]
            status.has_llm = any(
                OLLAMA_MODEL.split(":")[0] in m for m in status.models
            )
            status.has_embed = any(
                OLLAMA_EMBED_MODEL.split(":")[0] in m for m in status.models
            )
    except Exception:
        pass
    return status


def _get_embedding(text: str) -> Optional[list[float]]:
    """Get an embedding vector from Ollama."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("embedding")
    except Exception as e:
        logger.warning("Embedding error: %s", e)
    return None


def _llm_generate(prompt: str, stream: bool = False,
                  num_predict: int = LLM_MAP_PREDICT,
                  num_ctx: int = LLM_CTX_TOKENS,
                  temperature: float = 0.3,
                  timeout: int = 600):
    """Call Ollama generate with correct num_ctx. Returns response object."""
    return requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        },
        stream=stream,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking at sentence boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind(".")
            if last_period > chunk_size // 2:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1
        chunk = chunk.strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Checkpoint helpers — survive crashes without losing hours of work
# ---------------------------------------------------------------------------

def _checkpoint_key(query: str, doc_ids: list[int]) -> str:
    """Deterministic key for a (query, doc_ids) pair."""
    id_str = ",".join(str(i) for i in sorted(doc_ids))
    raw = f"{query.lower().strip()}|{id_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _checkpoint_path(query: str, doc_ids: list[int]) -> Path:
    key = _checkpoint_key(query, doc_ids)
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in query)
    safe_name = safe_name.strip().replace(" ", "_")[:40]
    return CHECKPOINT_DIR / f"{safe_name}_{key}.json"


def _load_checkpoint(query: str, doc_ids: list[int]) -> Optional[dict]:
    """Load an existing checkpoint if it matches the query and doc set."""
    path = _checkpoint_path(query, doc_ids)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("query") == query
                and data.get("checkpoint_key") == _checkpoint_key(query, doc_ids)):
            return data
    except Exception as e:
        logger.warning("Corrupt checkpoint %s: %s", path, e)
    return None


def _save_checkpoint(query: str, doc_ids: list[int],
                     num_batches: int,
                     batch_summaries: list[str],
                     total_docs: int, total_chars: int):
    """Write checkpoint to disk. Called after each batch completes."""
    path = _checkpoint_path(query, doc_ids)
    data = {
        "query": query,
        "checkpoint_key": _checkpoint_key(query, doc_ids),
        "doc_ids": doc_ids,
        "num_batches": num_batches,
        "total_docs": total_docs,
        "total_chars": total_chars,
        "completed_batches": len(batch_summaries),
        "batch_summaries": batch_summaries,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _delete_checkpoint(query: str, doc_ids: list[int]):
    """Remove checkpoint after successful completion."""
    path = _checkpoint_path(query, doc_ids)
    try:
        path.unlink(missing_ok=True)
        logger.info("Checkpoint deleted: %s", path.name)
    except Exception as e:
        logger.warning("Could not delete checkpoint: %s", e)


# ---------------------------------------------------------------------------
# Map-Reduce analysis — processes ALL documents
# ---------------------------------------------------------------------------

def _load_all_doc_texts(doc_ids: list[int]) -> list[dict]:
    """Load text content for all document IDs. Returns list of
    {doc_id, filename, text} dicts (skips docs with no text)."""
    from . import database

    docs = []
    for doc_id in doc_ids:
        doc = database.get_document_by_id(doc_id)
        if not doc or not doc.get("text_path"):
            continue
        text_path = TEXT_DIR / doc["text_path"]
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 20:
            continue
        docs.append({
            "doc_id": doc_id,
            "filename": doc["filename"],
            "text": text,
        })
    return docs


def _build_batches(doc_texts: list[dict],
                   max_chars: int = LLM_BATCH_CHARS) -> list[list[dict]]:
    """Group documents into batches that fit within max_chars total.

    Documents larger than max_chars are truncated (with a note) so that
    no single batch blows out the LLM context window.
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0

    for doc in doc_texts:
        text = doc["text"]
        if len(text) > max_chars:
            truncated = text[:max_chars] + (
                f"\n\n[... document truncated at {max_chars:,} of "
                f"{len(text):,} chars ...]"
            )
            doc = {**doc, "text": truncated}

        doc_len = len(doc["text"])
        if current_size + doc_len > max_chars and current:
            batches.append(current)
            current = []
            current_size = 0
        current.append(doc)
        current_size += doc_len

    if current:
        batches.append(current)
    return batches


def _map_prompt(query: str, batch_docs: list[dict]) -> str:
    """Build the map-phase prompt for one batch of documents."""
    parts = []
    for doc in batch_docs:
        parts.append(f"[{doc['filename']}]\n{doc['text']}")
    context = "\n\n---\n\n".join(parts)

    return f"""Extract facts about "{query}" from these DOJ Epstein documents. For EACH document use this exact format:
FILENAME: key facts, names mentioned, dates, locations, doc type (email/deposition/court filing/etc)
Skip documents with no relevant info. Be brief — one line per document.

{context}

FINDINGS:"""


def _intermediate_reduce_prompt(query: str, group_summaries: list[str],
                                group_num: int, total_groups: int) -> str:
    """Prompt for an intermediate reduce round — consolidate a subset of
    batch summaries into one shorter summary, preserving all key facts."""
    combined = "\n\n---\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(group_summaries)
    )
    return f"""Below are findings about "{query}" extracted from DOJ Epstein documents (group {group_num}/{total_groups}).

Consolidate these into a single concise summary. Keep ALL:
- Names (with document citations)
- Dates and events
- Locations
- Financial details
- Document types
Do NOT drop any facts. Be concise but complete.

{combined}

CONSOLIDATED SUMMARY:"""


def _reduce_prompt(query: str, batch_summaries: list[str],
                   total_docs: int) -> str:
    """Build the final reduce-phase prompt to synthesize all summaries."""
    combined = "\n\n---\n\n".join(
        f"[Batch {i+1} findings]\n{s}" for i, s in enumerate(batch_summaries)
    )

    return f"""You have extracted findings from {total_docs} DOJ Epstein Files documents related to "{query}".

Below are the findings from each batch. Synthesize ALL of them into a comprehensive, structured analysis:

1. **Key Facts**: The most important facts discovered across all documents
2. **Timeline**: Chronological sequence of dates and events mentioned
3. **People Connected**: Every person mentioned in connection with {query} (with document citations)
4. **Locations**: All locations referenced
5. **Financial Details**: Any dollar amounts, transactions, or financial connections
6. **Document Types**: Summary of what types of documents were found
7. **Patterns**: Recurring themes, repeated connections, or notable patterns across documents
8. **Most Significant Findings**: The 3-5 most important discoveries

Cite specific document filenames for every claim. Do NOT omit findings — this is the final synthesis of {total_docs} documents.

BATCH FINDINGS:
{combined}

COMPREHENSIVE ANALYSIS:"""


# Maximum chars of batch summaries that fit comfortably in the reduce
# context window.  32K tokens ≈ ~100K chars, but we leave generous
# headroom for the prompt template and output tokens.
_REDUCE_MAX_INPUT_CHARS = 60_000


def _hierarchical_reduce(query: str, batch_summaries: list[str],
                         total_docs: int,
                         cancel_flag=None,
                         progress_yield=None) -> list[str]:
    """If batch summaries exceed the reduce context window, run
    intermediate reduction rounds until they fit.

    Returns the (possibly reduced) list of summaries ready for the
    final synthesis prompt.
    """
    def _cancelled():
        return cancel_flag and cancel_flag.is_set()

    round_num = 0
    current = list(batch_summaries)

    while sum(len(s) for s in current) > _REDUCE_MAX_INPUT_CHARS:
        round_num += 1
        total_chars = sum(len(s) for s in current)
        logger.info(
            "Reduce round %d: %d summaries, %d chars (limit %d)",
            round_num, len(current), total_chars, _REDUCE_MAX_INPUT_CHARS,
        )

        # Group summaries into chunks that fit in the reduce context
        groups: list[list[str]] = []
        group: list[str] = []
        group_size = 0
        per_group_limit = _REDUCE_MAX_INPUT_CHARS

        for s in current:
            if group_size + len(s) > per_group_limit and group:
                groups.append(group)
                group = []
                group_size = 0
            group.append(s)
            group_size += len(s)
        if group:
            groups.append(group)

        if progress_yield:
            progress_yield({
                "stage": "reduce",
                "current": 0,
                "total": len(groups),
                "message": (
                    f"Condensing {len(current)} batch summaries — "
                    f"{len(groups)} groups to process (each takes several minutes)"
                ),
            })

        condensed: list[str] = []
        for gi, grp in enumerate(groups):
            if _cancelled():
                return current

            grp_chars = sum(len(s) for s in grp)
            logger.info(
                "Intermediate reduce round %d, group %d/%d: "
                "%d summaries, %d chars",
                round_num, gi + 1, len(groups), len(grp), grp_chars,
            )

            prompt = _intermediate_reduce_prompt(
                query, grp, gi + 1, len(groups),
            )
            try:
                resp = _llm_generate(
                    prompt, stream=False,
                    num_predict=2000,
                    num_ctx=LLM_REDUCE_CTX,
                    timeout=900,
                )
                if resp.status_code == 200:
                    text = resp.json().get("response", "").strip()
                    if text:
                        condensed.append(text)
                    else:
                        condensed.extend(grp)
                else:
                    condensed.extend(grp)
            except Exception as e:
                logger.error("Intermediate reduce group %d failed: %s", gi + 1, e)
                condensed.extend(grp)

            if progress_yield:
                progress_yield({
                    "stage": "reduce",
                    "current": gi + 1,
                    "total": len(groups),
                    "message": (
                        f"Condensing group {gi + 1}/{len(groups)} complete"
                    ),
                })

        current = condensed
        logger.info(
            "Reduce round %d complete: %d summaries, %d chars",
            round_num, len(current), sum(len(s) for s in current),
        )

    return current


def analyze_documents(query: str, doc_ids: list[int],
                      stream: bool = False,
                      progress_callback=None,
                      cancel_flag=None):
    """
    Analyze ALL documents using a map-reduce approach with checkpointing.

    Map phase:  batch documents into groups, ask LLM to extract findings
                from each batch.  Each batch summary is checkpointed to
                disk so a crash only loses ~60 seconds of work.
    Reduce phase: combine all batch summaries and ask LLM to synthesize
                  a comprehensive analysis.

    On restart, an existing checkpoint is detected and the map phase
    resumes from the last completed batch.
    """
    from . import database

    def _cancelled():
        return cancel_flag and cancel_flag.is_set()

    # --- Load all document texts -----------------------------------------
    if stream:
        yield {"type": "progress", "content": {
            "stage": "loading", "current": 0, "total": len(doc_ids),
            "message": f"Loading {len(doc_ids)} documents..."
        }}

    doc_texts = _load_all_doc_texts(doc_ids)
    if not doc_texts:
        if stream:
            yield {"type": "error",
                   "content": "No readable text found in the documents."}
        return

    total_chars = sum(len(d["text"]) for d in doc_texts)
    total_docs = len(doc_texts)

    # --- Build batches ---------------------------------------------------
    batches = _build_batches(doc_texts)
    num_batches = len(batches)

    # --- Check for existing checkpoint -----------------------------------
    checkpoint = _load_checkpoint(query, doc_ids)
    resume_from = 0
    batch_summaries: list[str] = []

    if (checkpoint
            and checkpoint.get("num_batches") == num_batches
            and checkpoint.get("completed_batches", 0) > 0):
        resume_from = checkpoint["completed_batches"]
        batch_summaries = list(checkpoint["batch_summaries"])
        logger.info(
            "Resuming from checkpoint: %d/%d batches already done",
            resume_from, num_batches,
        )
        if stream:
            yield {"type": "progress", "content": {
                "stage": "resuming",
                "current": resume_from,
                "total": num_batches,
                "message": (
                    f"Resuming from checkpoint — "
                    f"{resume_from}/{num_batches} batches already complete"
                ),
            }}

    # --- Time estimate ---------------------------------------------------
    remaining_batches_est = num_batches - resume_from
    est_per_batch = 55
    est_total = remaining_batches_est * est_per_batch + 120

    if stream:
        yield {"type": "estimate", "content": {
            "doc_count": total_docs,
            "batch_count": num_batches,
            "total_chars": total_chars,
            "est_seconds": est_total,
            "resumed_from": resume_from,
        }}

    logger.info(
        "Map-reduce: %d docs, %d chars, %d batches (resume=%d), est %ds",
        total_docs, total_chars, num_batches, resume_from, est_total,
    )

    # --- Send source list ------------------------------------------------
    if stream:
        seen = set()
        source_list = []
        for doc in doc_texts:
            if doc["filename"] not in seen:
                seen.add(doc["filename"])
                source_list.append({
                    "filename": doc["filename"],
                    "doc_id": doc["doc_id"],
                    "score": 1.0,
                })
        yield {"type": "sources", "content": source_list}

    # --- MAP PHASE -------------------------------------------------------
    docs_processed = sum(len(batches[i]) for i in range(resume_from))
    map_start = time.time()

    for batch_idx in range(resume_from, num_batches):
        if _cancelled():
            yield {"type": "error", "content": "Analysis cancelled"}
            return

        batch_docs = batches[batch_idx]
        batch_doc_count = len(batch_docs)
        docs_processed += batch_doc_count

        if stream:
            yield {"type": "progress", "content": {
                "stage": "map",
                "current": batch_idx,
                "total": num_batches,
                "message": (
                    f"Analyzing batch {batch_idx + 1}/{num_batches} "
                    f"({docs_processed}/{total_docs} documents)"
                ),
            }}

        prompt = _map_prompt(query, batch_docs)

        try:
            resp = _llm_generate(
                prompt, stream=False,
                num_predict=LLM_MAP_PREDICT,
                num_ctx=LLM_CTX_TOKENS,
            )
            if resp.status_code == 200:
                summary = resp.json().get("response", "").strip()
                if summary:
                    batch_summaries.append(summary)

                    # ---- CHECKPOINT after every successful batch ----
                    _save_checkpoint(
                        query, doc_ids, num_batches,
                        batch_summaries, total_docs, total_chars,
                    )

                    if stream:
                        filenames = [d["filename"] for d in batch_docs]
                        yield {"type": "batch_result", "content": {
                            "batch": batch_idx + 1,
                            "total_batches": num_batches,
                            "docs_in_batch": batch_doc_count,
                            "docs_processed": docs_processed,
                            "total_docs": total_docs,
                            "filenames": filenames,
                            "summary": summary,
                        }}
            else:
                logger.warning("Ollama returned %d for batch %d",
                               resp.status_code, batch_idx + 1)
        except Exception as e:
            logger.error("Batch %d failed: %s", batch_idx + 1, e)
            if stream:
                yield {"type": "progress", "content": {
                    "stage": "map",
                    "current": batch_idx + 1,
                    "total": num_batches,
                    "message": f"Batch {batch_idx + 1} failed: {e}",
                }}

        # Refine time estimate
        batches_done_this_run = (batch_idx - resume_from) + 1
        if stream and batches_done_this_run > 1:
            elapsed = time.time() - map_start
            avg_per_batch = elapsed / batches_done_this_run
            remaining = num_batches - (batch_idx + 1)
            est_remaining = int(remaining * avg_per_batch) + 60
            yield {"type": "time_update", "content": {
                "elapsed": int(elapsed),
                "est_remaining": est_remaining,
            }}

    # Final map progress
    if stream:
        yield {"type": "progress", "content": {
            "stage": "map",
            "current": num_batches,
            "total": num_batches,
            "message": (
                f"Map phase complete: {num_batches} batches, "
                f"{total_docs} documents processed"
            ),
        }}

    if _cancelled():
        yield {"type": "error", "content": "Analysis cancelled"}
        return

    if not batch_summaries:
        if stream:
            yield {"type": "error",
                   "content": "No findings extracted from any batch."}
        return

    # --- REDUCE PHASE: synthesize all batch summaries --------------------
    if stream:
        yield {"type": "progress", "content": {
            "stage": "reduce",
            "current": 0,
            "total": 1,
            "message": (
                f"Synthesizing findings from {len(batch_summaries)} batches "
                f"({total_docs} documents)..."
            ),
        }}

    # If the combined summaries are too large for the context window,
    # run intermediate reduce rounds to condense them first.
    # We use a queue so progress messages stream in real-time.
    import queue as _queue_mod
    _reduce_q: _queue_mod.Queue = _queue_mod.Queue()

    def _progress_yield(content):
        """Used by _hierarchical_reduce to stream progress in real-time."""
        _reduce_q.put({"type": "progress", "content": content})

    import threading as _thr
    _reduce_result: list = []
    _reduce_error: list = []

    def _run_reduce():
        try:
            r = _hierarchical_reduce(
                query, batch_summaries, total_docs,
                cancel_flag=cancel_flag,
                progress_yield=_progress_yield,
            )
            _reduce_result.append(r)
        except Exception as exc:
            _reduce_error.append(exc)
        finally:
            _reduce_q.put(None)  # sentinel

    reduce_thread = _thr.Thread(target=_run_reduce, daemon=True)
    reduce_thread.start()

    # Drain progress messages in real-time
    if stream:
        while True:
            try:
                item = _reduce_q.get(timeout=15)
            except _queue_mod.Empty:
                continue
            if item is None:
                break
            yield item

    reduce_thread.join()

    if _reduce_error:
        if stream:
            yield {"type": "error", "content": f"Reduce phase error: {_reduce_error[0]}"}
        return

    if _cancelled():
        yield {"type": "error", "content": "Analysis cancelled"}
        return

    reduced_summaries = _reduce_result[0] if _reduce_result else batch_summaries

    reduce_prompt = _reduce_prompt(query, reduced_summaries, total_docs)

    if stream:
        if len(reduced_summaries) < len(batch_summaries):
            yield {"type": "progress", "content": {
                "stage": "reduce",
                "current": 0,
                "total": 1,
                "message": (
                    f"Condensed {len(batch_summaries)} batch summaries to "
                    f"{len(reduced_summaries)} — running final synthesis..."
                ),
            }}

        header = (
            f"**Analysis of {total_docs} documents "
            f"({total_chars:,} characters) in {num_batches} batches**\n\n---\n\n"
        )
        yield {"type": "token", "content": header}
        full_text = header

        try:
            resp = _llm_generate(
                reduce_prompt, stream=True,
                num_predict=LLM_REDUCE_PREDICT,
                num_ctx=LLM_REDUCE_CTX,
                timeout=1800,
            )
            for line in resp.iter_lines():
                if _cancelled():
                    yield {"type": "error", "content": "Analysis cancelled"}
                    return
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        token = data["response"]
                        full_text += token
                        yield {"type": "token", "content": token}
                    if data.get("done"):
                        break
            yield {"type": "done", "content": ""}
        except Exception as e:
            yield {"type": "error", "content": f"Reduce phase error: {e}"}
            return

        # Save to database: synthesis + all batch findings
        batch_section = "\n\n---\nBATCH FINDINGS\n---\n\n"
        batch_section += "\n\n".join(
            f"[Batch {i+1}]\n{s}" for i, s in enumerate(batch_summaries)
        )
        saved_text = full_text + batch_section

        try:
            result = AnalysisResult(
                query=query,
                document_ids=doc_ids,
                analysis_text=saved_text,
                model_used=OLLAMA_MODEL,
            )
            database.insert_analysis(result)
            logger.info("Analysis saved to database (%d chars)", len(saved_text))
        except Exception as e:
            logger.error("Failed to save analysis: %s", e)

        # Success — delete the checkpoint
        _delete_checkpoint(query, doc_ids)
        return

    # Non-streaming fallback
    reduced_summaries_nb = _hierarchical_reduce(
        query, batch_summaries, total_docs, cancel_flag=cancel_flag,
    )
    reduce_prompt = _reduce_prompt(query, reduced_summaries_nb, total_docs)
    try:
        resp = _llm_generate(
            reduce_prompt, stream=False,
            num_predict=LLM_REDUCE_PREDICT,
            num_ctx=LLM_REDUCE_CTX,
            timeout=1800,
        )
        analysis_text = (resp.json().get("response", "")
                         if resp.status_code == 200
                         else f"Ollama error: {resp.status_code}")
    except Exception as e:
        analysis_text = f"Error: {e}"

    result = AnalysisResult(
        query=query,
        document_ids=doc_ids,
        analysis_text=analysis_text,
        model_used=OLLAMA_MODEL,
    )
    database.insert_analysis(result)
    _delete_checkpoint(query, doc_ids)
    return result


# ---------------------------------------------------------------------------
# Correlation analysis (uses chunk-based approach for co-occurrences)
# ---------------------------------------------------------------------------

def _collect_correlation_chunks(names: list[str], doc_ids: list[int],
                                top_k: int = 20) -> tuple[list[dict], dict]:
    """Collect and rank text chunks for correlation analysis.

    Uses a larger chunk window (1500 chars) so co-occurrences that span
    nearby paragraphs aren't missed.
    """
    from . import database

    CORR_CHUNK_SIZE = 1500
    CORR_CHUNK_OVERLAP = 300

    all_chunks: list[dict] = []
    for doc_id in doc_ids:
        doc = database.get_document_by_id(doc_id)
        if not doc or not doc.get("text_path"):
            continue
        text_path = TEXT_DIR / doc["text_path"]
        if not text_path.exists():
            continue
        content = text_path.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text(content, chunk_size=CORR_CHUNK_SIZE,
                             overlap=CORR_CHUNK_OVERLAP)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "filename": doc["filename"],
                "doc_id": doc_id,
                "chunk_idx": i,
                "query": doc.get("found_via_query", ""),
            })

    if not all_chunks:
        return [], {}

    name_lowers = [n.lower() for n in names]
    last_names = []
    for n in names:
        parts = n.strip().split()
        last_names.append(parts[-1].lower() if len(parts) >= 2 else n.lower())

    per_name_docs: dict[str, set] = {n: set() for n in names}
    multi_mention_chunks: list[dict] = []
    single_mention_chunks: list[dict] = []

    for chunk in all_chunks:
        text_lower = chunk["text"].lower()
        matched_names = []
        for i, (full, last) in enumerate(zip(name_lowers, last_names)):
            if full in text_lower or last in text_lower:
                matched_names.append(names[i])
                per_name_docs[names[i]].add(chunk["filename"])

        chunk["matched_names"] = matched_names
        chunk["match_count"] = len(matched_names)

        if len(matched_names) >= 2:
            multi_mention_chunks.append(chunk)
        elif len(matched_names) == 1:
            single_mention_chunks.append(chunk)

    scored: list[dict] = []

    for chunk in multi_mention_chunks:
        chunk["score"] = 1.0 + (chunk["match_count"] * 0.1)
        scored.append(chunk)

    multi_count = len(scored)
    if multi_count < 5:
        remaining_slots = min(5, top_k - multi_count)
        for chunk in single_mention_chunks[:remaining_slots]:
            chunk["score"] = 0.5
            scored.append(chunk)

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate: 1 chunk per document
    deduped: list[dict] = []
    doc_seen: set[str] = set()
    for chunk in scored:
        fn = chunk["filename"]
        if fn not in doc_seen:
            deduped.append(chunk)
            doc_seen.add(fn)
    scored = deduped

    overlap_docs = set()
    for n1 in names:
        for n2 in names:
            if n1 != n2:
                overlap_docs |= per_name_docs[n1] & per_name_docs[n2]

    stats = {
        "per_name_docs": {n: len(s) for n, s in per_name_docs.items()},
        "overlap_doc_count": len(overlap_docs),
        "multi_mention_chunks": len(multi_mention_chunks),
        "total_chunks_scanned": len(all_chunks),
    }

    return scored[:top_k], stats


SUMMARIZE_WORKERS = 4  # concurrent Ollama requests for summarization


def _summarize_one(doc: dict) -> tuple[int, str | None]:
    """Summarize a single document. Returns (doc_id, snippet_or_None)."""
    text_path = TEXT_DIR / doc["text_path"]
    if not text_path.exists():
        return doc["id"], None

    text = text_path.read_text(encoding="utf-8", errors="replace")
    sample = text[:800].strip()
    if not sample:
        return doc["id"], None

    prompt = (
        "Below is the beginning of a document from the DOJ Epstein Files. "
        "Respond with ONLY a single short description (max 60 characters) "
        "that says what type of document this is and its main subject. "
        "Examples: \"Email re: dinner plans with Leon Black\", "
        "\"Financial statement for Epstein foundation\", "
        "\"Deposition transcript - Ghislaine Maxwell\". "
        "Do NOT include the filename. Just the description.\n\n"
        f"FILENAME: {doc['filename']}\n\n{sample}\n\nDESCRIPTION:"
    )

    try:
        resp = _llm_generate(
            prompt, stream=False,
            num_predict=40,
            num_ctx=2048,
            timeout=30,
        )
        if resp.status_code == 200:
            snippet = resp.json().get("response", "").strip()
            snippet = snippet.split("\n")[0].strip().strip('"').strip("'")
            if len(snippet) > 120:
                snippet = snippet[:120].rsplit(" ", 1)[0] + "..."
            return doc["id"], snippet or None
    except Exception as e:
        logger.warning("Summarize failed for %s: %s", doc["filename"], e)

    return doc["id"], None


def summarize_documents(doc_ids: list[int] = None,
                        stream: bool = False,
                        cancel_flag=None):
    """Generate short LLM summaries for documents, running multiple
    Ollama requests in parallel via a thread pool.

    Yields progress updates if stream=True.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from . import database

    def _cancelled():
        return cancel_flag and cancel_flag.is_set()

    if doc_ids:
        docs = [database.get_document_by_id(did) for did in doc_ids]
        docs = [d for d in docs if d]
    else:
        docs = database.get_extracted_documents()

    docs = [d for d in docs if not d.get("snippet")]

    if not docs:
        if stream:
            yield {"type": "done", "content": "All documents already have summaries."}
        return

    total = len(docs)
    done = 0

    if stream:
        yield {"type": "estimate", "content": {
            "doc_count": total,
            "est_seconds": total * 3 // SUMMARIZE_WORKERS,
        }}

    with ThreadPoolExecutor(max_workers=SUMMARIZE_WORKERS) as pool:
        futures = {}
        doc_iter = iter(docs)

        # Seed the pool
        for _ in range(min(SUMMARIZE_WORKERS * 2, total)):
            doc = next(doc_iter, None)
            if doc is None:
                break
            futures[pool.submit(_summarize_one, doc)] = doc

        while futures:
            if _cancelled():
                for f in futures:
                    f.cancel()
                if stream:
                    yield {"type": "error", "content": "Summarization cancelled"}
                return

            # Wait for the next completion
            completed = next(as_completed(futures))
            del futures[completed]

            try:
                doc_id, snippet = completed.result()
                if snippet:
                    database.update_document_snippet(doc_id, snippet)
            except Exception as e:
                logger.warning("Summarize future error: %s", e)

            done += 1

            # Feed more work into the pool
            doc = next(doc_iter, None)
            if doc is not None:
                futures[pool.submit(_summarize_one, doc)] = doc

            if stream and done % 10 == 0:
                yield {"type": "progress", "content": {
                    "stage": "summarizing",
                    "current": done,
                    "total": total,
                    "message": f"Summarized {done}/{total} documents",
                }}

    if stream:
        yield {"type": "progress", "content": {
            "stage": "summarizing",
            "current": total,
            "total": total,
            "message": f"Done — {total} documents summarized",
        }}
        yield {"type": "done", "content": ""}

    logger.info("Summarized %d documents with %d workers",
                done, SUMMARIZE_WORKERS)


def analyze_correlation(names: list[str], doc_ids: list[int],
                        stream: bool = False,
                        cancel_flag=None):
    """Analyze cross-connections between multiple people."""
    from . import database

    def _cancelled():
        return cancel_flag and cancel_flag.is_set()

    if not doc_ids:
        for name in names:
            docs = database.get_documents_by_query(name)
            doc_ids.extend(
                d["id"] for d in docs if d["status"] == "extracted"
            )
        doc_ids = list(set(doc_ids))

    if not doc_ids:
        msg = ("No extracted documents found for any of the queried names. "
               "Search and download documents for each person first.")
        if stream:
            yield {"type": "error", "content": msg}
            return
        return msg

    names_str = ", ".join(names)
    logger.info("Correlation analysis: %s across %d documents",
                names_str, len(doc_ids))

    if stream:
        yield {"type": "token",
               "content": f"Scanning {len(doc_ids)} documents for "
                          f"cross-connections between {names_str}...\n\n"}

    ranked_chunks, stats = _collect_correlation_chunks(names, doc_ids, top_k=20)

    if stream:
        stat_msg = (
            f"**Scan results:** {stats['total_chunks_scanned']} text segments "
            f"scanned across {len(doc_ids)} documents\n"
        )
        for name in names:
            count = stats["per_name_docs"].get(name, 0)
            stat_msg += f"- **{name}**: mentioned in {count} documents\n"
        stat_msg += (
            f"- **Documents mentioning multiple names**: "
            f"{stats['overlap_doc_count']}\n"
            f"- **Text segments with co-occurrences**: "
            f"{stats['multi_mention_chunks']}\n\n"
        )
        yield {"type": "token", "content": stat_msg}

    if not ranked_chunks:
        msg = "No relevant text chunks found across the documents."
        if stream:
            yield {"type": "error", "content": msg}
            return
        return msg

    if stream:
        seen = set()
        source_list = []
        for c in ranked_chunks:
            key = (c["filename"], c["doc_id"])
            if key not in seen:
                seen.add(key)
                source_list.append({
                    "filename": c["filename"],
                    "doc_id": c["doc_id"],
                    "score": c.get("score", 0),
                })
        yield {"type": "sources", "content": source_list}

    multi_chunks = [c for c in ranked_chunks if c.get("match_count", 0) >= 2]
    single_chunks = [c for c in ranked_chunks if c.get("match_count", 0) < 2]

    context_parts = []
    for chunk in multi_chunks:
        matched = ", ".join(chunk.get("matched_names", []))
        label = f"[Document: {chunk['filename']} | Names found: {matched}]"
        context_parts.append(f"{label}\n{chunk['text']}")

    if len(multi_chunks) < 5:
        context_parts.append(
            "\n--- Additional context (single subject mentions) ---"
        )
        for chunk in single_chunks[:max(3, 10 - len(multi_chunks))]:
            matched = ", ".join(chunk.get("matched_names", []))
            label = f"[Document: {chunk['filename']} | Names found: {matched}]"
            context_parts.append(f"{label}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    names_list = "\n".join(f"- {n}" for n in names)
    prompt = f"""Below are excerpts from DOJ Epstein Files where BOTH of these people are mentioned:
{names_list}

For EACH document excerpt, answer:
- What type of document is it (email, schedule, financial record, legal filing)?
- Why do both names appear?
- What specific connection does it reveal?
- What dates are mentioned?
- What other people are mentioned?

After analyzing each document, provide:
- A TIMELINE of events connecting these individuals
- The most significant finding

DOCUMENT EXCERPTS:

{context}

DOCUMENT-BY-DOCUMENT ANALYSIS:"""

    if stream:
        yield {"type": "progress", "content": {
            "stage": "analyzing", "current": 0, "total": 1,
            "message": "Running LLM cross-connection analysis...",
        }}

    try:
        resp = _llm_generate(
            prompt, stream=False,
            num_predict=LLM_REDUCE_PREDICT,
            num_ctx=LLM_REDUCE_CTX,
        )
        if _cancelled():
            if stream:
                yield {"type": "error", "content": "Analysis cancelled"}
            return
        result = (resp.json().get("response", "")
                  if resp.status_code == 200 else "LLM returned an error")
    except Exception as e:
        if stream:
            yield {"type": "error", "content": str(e)}
        else:
            return f"Error: {e}"
        return

    if stream:
        yield {"type": "token", "content": result}
        yield {"type": "done", "content": ""}
    else:
        return result
