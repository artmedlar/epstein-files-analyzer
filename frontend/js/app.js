/**
 * Epstein Files Analyzer - Main Application
 */

const API_BASE = '';
const WS_BASE = `ws://${window.location.host}`;

// State
let currentView = 'search';
let searchResults = [];
let selectedUrls = new Set();

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initSearch();
    initAnalysis();
    initSettings();
    initModal();
    checkStatus();
    loadHistory();
});

function initNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            switchView(view);
        });
    });
}

function switchView(view) {
    currentView = view;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelector(`.nav-btn[data-view="${view}"]`).classList.add('active');
    document.getElementById(`view-${view}`).classList.add('active');

    if (view === 'documents') loadDocuments();
    if (view === 'settings') loadPrompts();
    if (view === 'analysis') {
        const q = document.getElementById('analysis-query').value.trim();
        loadTimeline(q || null);
    }
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async function checkStatus() {
    try {
        const resp = await fetch(`${API_BASE}/api/status`);
        const data = await resp.json();
        const dot = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');
        const count = document.getElementById('doc-count');

        count.textContent = `${data.document_count} docs`;

        if (data.ollama.available && data.ollama.has_llm) {
            dot.className = 'status-dot online';
            text.textContent = 'Ollama connected';
        } else if (data.ollama.available) {
            dot.className = 'status-dot partial';
            text.textContent = 'Ollama: missing model';
        } else {
            dot.className = 'status-dot offline';
            text.textContent = 'Ollama offline';
        }
    } catch (e) {
        document.getElementById('status-indicator').className = 'status-dot offline';
        document.getElementById('status-text').textContent = 'Server error';
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

function initSearch() {
    const input = document.getElementById('search-input');
    const btn = document.getElementById('search-btn');
    const localBtn = document.getElementById('search-local-btn');
    const downloadAllBtn = document.getElementById('download-all-btn');
    const downloadSelBtn = document.getElementById('download-selected-btn');
    const selectAllBtn = document.getElementById('select-all-btn');

    btn.addEventListener('click', () => doSearch(input.value.trim()));
    input.addEventListener('keydown', e => {
        if (e.key === 'Enter') doSearch(input.value.trim());
    });
    localBtn.addEventListener('click', () => doLocalSearch(input.value.trim()));
    downloadAllBtn.addEventListener('click', () => downloadDocuments([...searchResults.map(r => r.url)]));
    downloadSelBtn.addEventListener('click', () => downloadDocuments([...selectedUrls]));
    selectAllBtn.addEventListener('click', toggleSelectAll);
}

function doSearch(query) {
    if (!query) return;

    const progressArea = document.getElementById('search-progress');
    const resultsArea = document.getElementById('search-results');
    const progressBar = document.getElementById('search-progress-bar');
    const progressText = document.getElementById('search-progress-text');

    progressArea.classList.remove('hidden');
    resultsArea.classList.add('hidden');
    progressBar.style.width = '0%';
    progressBar.classList.add('indeterminate');
    progressText.textContent = 'Connecting to DOJ search...';

    searchResults = [];
    selectedUrls.clear();

    const ws = new WebSocket(`${WS_BASE}/ws/search`);

    ws.onopen = () => {
        ws.send(JSON.stringify({
            query: query,
            max_pages: parseInt(document.getElementById('search-max-pages').value) || 100,
        }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'progress') {
            const p = msg.data;
            progressText.textContent = p.message;
            if (p.total > 0) {
                progressBar.classList.remove('indeterminate');
                progressBar.style.width = `${(p.current / p.total) * 100}%`;
            }
        } else if (msg.type === 'results') {
            searchResults = msg.data.documents;
            progressBar.classList.remove('indeterminate');
            progressBar.style.width = '100%';
            progressText.textContent = `Found ${msg.data.count} documents`;
            renderSearchResults(msg.data);
            checkStatus();
        } else if (msg.type === 'error') {
            progressBar.classList.remove('indeterminate');
            progressText.textContent = `Error: ${msg.message}`;
        }
    };

    ws.onerror = () => {
        progressBar.classList.remove('indeterminate');
        progressText.textContent = 'WebSocket connection error';
    };
}

async function doLocalSearch(query) {
    if (!query) return;
    try {
        const resp = await fetch(`${API_BASE}/api/search/local?q=${encodeURIComponent(query)}&limit=100`);
        const data = await resp.json();
        document.getElementById('search-progress').classList.add('hidden');

        const resultsArea = document.getElementById('search-results');
        resultsArea.classList.remove('hidden');
        document.getElementById('results-title').textContent =
            `Local results for "${query}" (${data.count})`;

        const list = document.getElementById('results-list');
        list.innerHTML = '';

        if (data.results.length === 0) {
            list.innerHTML = '<p class="empty-state">No local results found. Try searching DOJ first.</p>';
            return;
        }

        data.results.forEach(r => {
            const div = document.createElement('div');
            div.className = 'result-item';
            div.innerHTML = `
                <div class="result-info">
                    <div class="result-filename">${escapeHtml(r.filename)}</div>
                    <div class="result-meta">${escapeHtml(r.preview?.substring(0, 150) || '')}...</div>
                </div>
                <span class="result-status extracted">extracted</span>
            `;
            div.addEventListener('click', () => viewDocument(r.doc_id));
            list.appendChild(div);
        });
    } catch (e) {
        console.error('Local search error:', e);
    }
}

function renderSearchResults(data) {
    const resultsArea = document.getElementById('search-results');
    const list = document.getElementById('results-list');
    const title = document.getElementById('results-title');

    resultsArea.classList.remove('hidden');
    title.textContent = `"${data.query}" - ${data.count} documents found`;

    list.innerHTML = '';
    document.getElementById('download-all-btn').disabled = data.count === 0;

    data.documents.forEach(doc => {
        const div = document.createElement('div');
        div.className = 'result-item';
        div.innerHTML = `
            <input type="checkbox" data-url="${escapeHtml(doc.url)}">
            <div class="result-info">
                <div class="result-filename">${escapeHtml(doc.filename)}</div>
                <div class="result-meta">${escapeHtml(doc.data_set || 'Unknown dataset')} &middot; <a href="${escapeHtml(doc.url)}" target="_blank" onclick="event.stopPropagation()">DOJ link</a></div>
            </div>
        `;
        const checkbox = div.querySelector('input[type="checkbox"]');
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) selectedUrls.add(doc.url);
            else selectedUrls.delete(doc.url);
            document.getElementById('download-selected-btn').disabled = selectedUrls.size === 0;
        });
        list.appendChild(div);
    });
}

function toggleSelectAll() {
    const checkboxes = document.querySelectorAll('#results-list input[type="checkbox"]');
    const allChecked = [...checkboxes].every(cb => cb.checked);
    checkboxes.forEach(cb => {
        cb.checked = !allChecked;
        const url = cb.dataset.url;
        if (cb.checked) selectedUrls.add(url);
        else selectedUrls.delete(url);
    });
    document.getElementById('download-selected-btn').disabled = selectedUrls.size === 0;
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------

function downloadDocuments(urls) {
    if (!urls.length) return;

    const progressArea = document.getElementById('search-progress');
    const progressBar = document.getElementById('search-progress-bar');
    const progressText = document.getElementById('search-progress-text');

    progressArea.classList.remove('hidden');
    progressBar.style.width = '0%';
    progressBar.classList.remove('indeterminate');

    const ws = new WebSocket(`${WS_BASE}/ws/download`);

    ws.onopen = () => {
        ws.send(JSON.stringify({ urls: urls }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'progress') {
            const p = msg.data;
            progressText.textContent = p.message;
            if (p.total > 0) {
                progressBar.style.width = `${(p.current / p.total) * 100}%`;
            }
        } else if (msg.type === 'complete') {
            progressBar.style.width = '100%';
            const d = msg.data;
            progressText.textContent =
                `Done! ${d.downloaded} downloaded, ${d.extracted} extracted, ${d.failed} failed`;
            checkStatus();
        } else if (msg.type === 'error') {
            progressText.textContent = `Error: ${msg.message}`;
        }
    };

    ws.onerror = () => {
        progressText.textContent = 'Download connection error';
    };
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

async function loadDocuments() {
    try {
        const resp = await fetch(`${API_BASE}/api/documents`);
        const data = await resp.json();
        const list = document.getElementById('documents-list');

        if (data.documents.length === 0) {
            list.innerHTML = '<p class="empty-state">No documents downloaded yet. Use Search to find and download documents.</p>';
            return;
        }

        // Group by query (a document can belong to multiple queries)
        const groups = {};
        data.documents.forEach(doc => {
            const queries = doc.queries && doc.queries.length
                ? doc.queries
                : [doc.found_via_query || '(no query)'];
            queries.forEach(q => {
                if (!groups[q]) groups[q] = [];
                groups[q].push(doc);
            });
        });

        list.innerHTML = '';
        Object.entries(groups).forEach(([query, docs]) => {
            const group = document.createElement('div');
            group.className = 'doc-group';

            const header = document.createElement('div');
            header.className = 'doc-group-header';
            header.innerHTML = `
                <div class="doc-group-title">
                    <strong>"${escapeHtml(query)}"</strong>
                    <span class="doc-group-count">${docs.length} documents</span>
                </div>
                <button class="btn-small btn-delete" title="Remove this search set">Remove</button>
            `;
            header.querySelector('.btn-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteSearchSet(query);
            });
            group.appendChild(header);

            docs.forEach(doc => {
                const div = document.createElement('div');
                div.className = 'doc-item';
                const statusClass = doc.status === 'extracted' ? 'extracted' :
                                   doc.status === 'failed' ? 'failed' : 'found';
                div.innerHTML = `
                    <div>
                        <div class="result-filename">${escapeHtml(doc.filename)}</div>
                        <div class="result-meta">
                            ${escapeHtml(doc.data_set || '')}
                            ${doc.page_count ? ` &middot; ${doc.page_count} pages` : ''}
                            ${doc.ocr_needed ? ' &middot; OCR' : ''}
                        </div>
                    </div>
                    <span class="result-status ${statusClass}">${doc.status}</span>
                `;
                if (doc.status === 'extracted') {
                    div.addEventListener('click', () => viewDocument(doc.id));
                }
                group.appendChild(div);
            });

            list.appendChild(group);
        });
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

async function deleteSearchSet(query) {
    if (!confirm(`Remove the "${query}" search set?\n\nDocuments shared with other search sets will be kept. Documents unique to this set will be deleted from disk.\n\nThis cannot be undone.`)) {
        return;
    }
    try {
        const resp = await fetch(
            `${API_BASE}/api/documents/query/${encodeURIComponent(query)}`,
            { method: 'DELETE' }
        );
        const data = await resp.json();
        if (resp.ok) {
            loadDocuments();
            checkStatus();
        } else {
            alert(data.error || 'Delete failed');
        }
    } catch (e) {
        console.error('Delete failed:', e);
        alert('Failed to delete documents');
    }
}

async function viewDocument(docId) {
    try {
        const resp = await fetch(`${API_BASE}/api/documents/${docId}/text`);
        const data = await resp.json();

        const docResp = await fetch(`${API_BASE}/api/documents/${docId}`);
        const doc = await docResp.json();

        document.getElementById('viewer-title').textContent = data.filename;
        document.getElementById('viewer-text').textContent = data.text;
        document.getElementById('viewer-doj-link').href = doc.url;
        document.getElementById('viewer-meta').innerHTML = `
            Pages: ${doc.page_count || '?'} |
            OCR: ${doc.ocr_needed ? 'Yes' : 'No'} |
            Confidence: ${doc.ocr_confidence ? doc.ocr_confidence + '%' : 'N/A'} |
            Query: ${escapeHtml(doc.found_via_query || 'N/A')}
        `;
        document.getElementById('doc-viewer-modal').classList.remove('hidden');
    } catch (e) {
        console.error('Failed to load document:', e);
    }
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

function initAnalysis() {
    document.getElementById('analyze-btn').addEventListener('click', () => {
        const query = document.getElementById('analysis-query').value.trim();
        if (query) runAnalysis(query);
    });
    document.getElementById('correlate-btn').addEventListener('click', () => {
        const names = document.getElementById('correlation-names').value
            .split(',').map(n => n.trim()).filter(n => n);
        if (names.length >= 2) runCorrelation(names);
    });
    document.getElementById('summarize-btn').addEventListener('click', runSummarize);
}

function renderMarkdown(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')
        .replace(/^# (.+)$/gm, '<h2>$1</h2>')
        .replace(/^---$/gm, '<hr>')
        .replace(/^\* (.+)$/gm, '<li>$1</li>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/^\t\* (.+)$/gm, '<li class="indent">$1</li>')
        .replace(/^\t\+ (.+)$/gm, '<li class="indent">$1</li>')
        .replace(/^\t- (.+)$/gm, '<li class="indent">$1</li>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
}

let activeAnalysisWs = null;

function cancelAnalysis() {
    if (activeAnalysisWs && activeAnalysisWs.readyState === WebSocket.OPEN) {
        activeAnalysisWs.send(JSON.stringify({ type: 'cancel' }));
        activeAnalysisWs.close();
        activeAnalysisWs = null;
    }
    const btn = document.getElementById('cancel-analysis-btn');
    if (btn) btn.classList.add('hidden');
}

function showAnalysisProgress(stage, current, total, message) {
    let bar = document.getElementById('analysis-progress-bar');
    let text = document.getElementById('analysis-progress-text');
    let container = document.getElementById('analysis-progress');

    if (!container) return;
    container.classList.remove('hidden');

    const pct = total > 0 ? Math.round((current / total) * 100) : 0;
    bar.style.width = pct + '%';
    bar.textContent = pct + '%';
    text.textContent = message;
}

function hideAnalysisProgress() {
    const container = document.getElementById('analysis-progress');
    if (container) container.classList.add('hidden');
}

function formatTime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function runAnalysis(query, docIds = []) {
    const output = document.getElementById('analysis-output');
    const sourcesDiv = document.getElementById('analysis-sources');
    const cancelBtn = document.getElementById('cancel-analysis-btn');

    const analyzeBtn = document.getElementById('analyze-btn');
    const correlateBtn = document.getElementById('correlate-btn');

    output.innerHTML = '<div class="analysis-loading">Connecting...</div>';
    sourcesDiv.classList.add('hidden');
    cancelBtn.classList.remove('hidden');
    analyzeBtn.disabled = true;
    correlateBtn.disabled = true;
    hideAnalysisProgress();

    let text = '';
    let gotFirstToken = false;
    let batchFindings = [];
    let analysisDone = false;
    let userCancelled = false;

    function enableButtons() {
        analyzeBtn.disabled = false;
        correlateBtn.disabled = false;
    }

    const ws = new WebSocket(`${WS_BASE}/ws/analyze`);
    activeAnalysisWs = ws;

    ws.onopen = () => {
        output.innerHTML = '<div class="analysis-loading">Gathering documents...</div>';
        ws.send(JSON.stringify({ query: query, doc_ids: docIds }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'estimate') {
            const e = msg.content;
            const resumed = e.resumed_from || 0;
            const resumeNote = resumed > 0
                ? `<br><strong>Resuming from checkpoint — ${resumed}/${e.batch_count} batches already done</strong>`
                : '';
            output.innerHTML = `<div class="analysis-loading">
                <strong>${e.doc_count} documents</strong> to analyze
                (${(e.total_chars || 0).toLocaleString()} characters)<br>
                Processing in <strong>${e.batch_count} batches</strong><br>
                Estimated time: <strong>~${formatTime(e.est_seconds)}</strong>
                ${resumeNote}
            </div>`;

        } else if (msg.type === 'progress') {
            const p = msg.content;
            showAnalysisProgress(p.stage, p.current, p.total, p.message);

        } else if (msg.type === 'batch_result') {
            const b = msg.content;
            batchFindings.push(b);
            let batchHtml = '<div class="batch-findings-header">' +
                `<strong>Map phase:</strong> ${b.docs_processed}/${b.total_docs} documents processed ` +
                `(batch ${b.batch}/${b.total_batches})</div>`;
            batchHtml += '<div class="batch-findings-list">';
            batchFindings.forEach((bf, i) => {
                batchHtml += `<details${i === batchFindings.length - 1 ? ' open' : ''}>` +
                    `<summary>Batch ${bf.batch}: ${bf.docs_in_batch} docs ` +
                    `(${bf.filenames.slice(0, 3).join(', ')}${bf.filenames.length > 3 ? '...' : ''})</summary>` +
                    `<div class="batch-summary">${renderMarkdown(escapeHtml(bf.summary))}</div></details>`;
            });
            batchHtml += '</div>';
            output.innerHTML = '<div class="analysis-text">' + batchHtml + '</div>';
            output.scrollTop = output.scrollHeight;

        } else if (msg.type === 'time_update') {
            const t = msg.content;
            const timeText = document.getElementById('analysis-progress-text');
            if (timeText) {
                timeText.textContent += ` | Elapsed: ${formatTime(t.elapsed)}, ~${formatTime(t.est_remaining)} remaining`;
            }

        } else if (msg.type === 'token') {
            if (!gotFirstToken && msg.content) {
                gotFirstToken = true;
                hideAnalysisProgress();
                let preamble = '';
                if (batchFindings.length > 0) {
                    preamble = `<details class="map-phase-details"><summary>Map phase: ${batchFindings.length} batches processed (click to expand)</summary>`;
                    preamble += '<div class="batch-findings-list">';
                    batchFindings.forEach(bf => {
                        preamble += `<details><summary>Batch ${bf.batch}: ${bf.docs_in_batch} docs</summary>` +
                            `<div class="batch-summary">${renderMarkdown(escapeHtml(bf.summary))}</div></details>`;
                    });
                    preamble += '</div></details><hr>';
                }
                output.innerHTML = preamble;
            }
            text += msg.content;
            if (text) {
                let existing = output.querySelector('.final-synthesis');
                if (!existing) {
                    const div = document.createElement('div');
                    div.className = 'analysis-text final-synthesis';
                    output.appendChild(div);
                    existing = div;
                }
                existing.innerHTML = renderMarkdown(escapeHtml(text));
                output.scrollTop = output.scrollHeight;
            }

        } else if (msg.type === 'sources') {
            sourcesDiv.classList.remove('hidden');
            const list = document.getElementById('sources-list');
            const sources = msg.content;
            list.innerHTML = `<div class="source-count">${sources.length} documents analyzed</div>` +
                sources.slice(0, 50).map(s =>
                    `<div class="source-item">${escapeHtml(s.filename)}</div>`
                ).join('') +
                (sources.length > 50 ? `<div class="source-item">...and ${sources.length - 50} more</div>` : '');

        } else if (msg.type === 'error') {
            hideAnalysisProgress();
            output.innerHTML = `<div class="analysis-error">Error: ${escapeHtml(msg.content)}</div>`;
            cancelBtn.classList.add('hidden');
            analysisDone = true;
            enableButtons();

        } else if (msg.type === 'done') {
            hideAnalysisProgress();
            cancelBtn.classList.add('hidden');
            analysisDone = true;
            enableButtons();
            loadTimeline(query);

        } else if (msg.type === 'heartbeat') {
            // keep-alive
        }
    };

    ws.onclose = () => {
        activeAnalysisWs = null;
        cancelBtn.classList.add('hidden');
        enableButtons();
        if (!analysisDone && !userCancelled) {
            hideAnalysisProgress();
            output.innerHTML = '<div class="analysis-error">Connection lost. ' +
                'Your progress is checkpointed — click Analyze again to resume.</div>';
        }
    };

    ws.onerror = () => {
        // onclose will fire after this
    };

    cancelBtn.onclick = () => {
        userCancelled = true;
        if (activeAnalysisWs && activeAnalysisWs.readyState === WebSocket.OPEN) {
            activeAnalysisWs.send(JSON.stringify({ type: 'cancel' }));
            activeAnalysisWs.close();
        }
        cancelBtn.classList.add('hidden');
        enableButtons();
    };
}

function runCorrelation(names) {
    const output = document.getElementById('analysis-output');
    const sourcesDiv = document.getElementById('analysis-sources');
    const cancelBtn = document.getElementById('cancel-analysis-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const correlateBtn = document.getElementById('correlate-btn');

    output.innerHTML = '<div class="analysis-loading">Connecting...</div>';
    sourcesDiv.classList.add('hidden');
    cancelBtn.classList.remove('hidden');
    analyzeBtn.disabled = true;
    correlateBtn.disabled = true;
    hideAnalysisProgress();

    function enableButtons() {
        analyzeBtn.disabled = false;
        correlateBtn.disabled = false;
    }

    const ws = new WebSocket(`${WS_BASE}/ws/analyze`);
    activeAnalysisWs = ws;

    ws.onopen = () => {
        output.innerHTML = `<div class="analysis-loading">
            <div class="correlation-header">Cross-Connection Analysis</div>
            <div>Gathering documents for: <strong>${names.map(escapeHtml).join(' &harr; ')}</strong></div>
        </div>`;
        ws.send(JSON.stringify({
            query: names.join(' '),
            correlation_names: names,
            doc_ids: [],
        }));
    };

    let text = '';
    let gotFirstToken = false;

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'estimate') {
            const e = msg.content;
            output.innerHTML = `<div class="analysis-loading">
                <div class="correlation-header">Cross-Connection Analysis</div>
                <strong>${e.doc_count} documents</strong> to scan<br>
                Estimated time: <strong>~${e.est_seconds}s</strong>
            </div>`;

        } else if (msg.type === 'progress') {
            const p = msg.content;
            showAnalysisProgress(p.stage, p.current, p.total, p.message);

        } else if (msg.type === 'token') {
            if (!gotFirstToken && msg.content) {
                gotFirstToken = true;
                hideAnalysisProgress();
                output.innerHTML = '';
            }
            text += msg.content;
            if (text) {
                output.innerHTML = '<div class="analysis-text correlation-results">' +
                    renderMarkdown(escapeHtml(text)) + '</div>';
                output.scrollTop = output.scrollHeight;
            }

        } else if (msg.type === 'sources') {
            sourcesDiv.classList.remove('hidden');
            const list = document.getElementById('sources-list');
            list.innerHTML = msg.content.map(s =>
                `<div class="source-item">${escapeHtml(s.filename)} <span class="relevance">${(s.score * 100).toFixed(0)}%</span></div>`
            ).join('');

        } else if (msg.type === 'error') {
            hideAnalysisProgress();
            output.innerHTML = `<div class="analysis-error">Error: ${escapeHtml(msg.content)}</div>`;
            cancelBtn.classList.add('hidden');
            enableButtons();

        } else if (msg.type === 'done') {
            hideAnalysisProgress();
            cancelBtn.classList.add('hidden');
            enableButtons();
            if (text) {
                output.innerHTML = '<div class="analysis-text correlation-results">' +
                    renderMarkdown(escapeHtml(text)) + '</div>';
            }
            loadTimeline();

        } else if (msg.type === 'heartbeat') {
            // keep-alive, ignore
        }
    };

    ws.onclose = () => {
        cancelBtn.classList.add('hidden');
        activeAnalysisWs = null;
        enableButtons();
    };
    ws.onerror = () => {
        hideAnalysisProgress();
        output.innerHTML = '<div class="analysis-error">Connection error</div>';
        cancelBtn.classList.add('hidden');
        enableButtons();
    };
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

function initModal() {
    document.getElementById('viewer-close').addEventListener('click', closeModal);
    document.getElementById('doc-viewer-modal').addEventListener('click', (e) => {
        if (e.target.id === 'doc-viewer-modal') closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

function closeModal() {
    document.getElementById('doc-viewer-modal').classList.add('hidden');
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

async function loadHistory() {
    try {
        const resp = await fetch(`${API_BASE}/api/history`);
        const data = await resp.json();
        const container = document.getElementById('search-history');
        container.innerHTML = '';

        const seen = new Set();
        data.forEach(h => {
            if (seen.has(h.query)) return;
            seen.add(h.query);
            const chip = document.createElement('span');
            chip.className = 'history-chip';
            chip.textContent = `${h.query} (${h.results_count})`;
            chip.addEventListener('click', () => {
                document.getElementById('search-input').value = h.query;
            });
            container.appendChild(chip);
        });
    } catch (e) {
        // History not critical
    }
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

async function loadTimeline(query) {
    const container = document.getElementById('timeline-container');
    const chart = document.getElementById('timeline-chart');
    const note = document.getElementById('timeline-note');

    try {
        const url = query
            ? `${API_BASE}/api/timeline?query=${encodeURIComponent(query)}`
            : `${API_BASE}/api/timeline`;
        const resp = await fetch(url);
        const data = await resp.json();

        if (data.total_dated < 2) {
            container.classList.add('hidden');
            return;
        }

        container.classList.remove('hidden');
        chart.innerHTML = '';

        // Build monthly histogram from dated documents
        const docs = data.documents;
        const monthBuckets = {};

        docs.forEach(d => {
            const key = d.document_date.substring(0, 7); // YYYY-MM
            if (!monthBuckets[key]) monthBuckets[key] = 0;
            monthBuckets[key]++;
        });

        // Determine date range
        const firstDate = docs[0].document_date;
        const lastDate = docs[docs.length - 1].document_date;
        const startYear = parseInt(firstDate.substring(0, 4));
        const startMonth = parseInt(firstDate.substring(5, 7));
        const endYear = parseInt(lastDate.substring(0, 4));
        const endMonth = parseInt(lastDate.substring(5, 7));

        // Generate all months in range
        const allMonths = [];
        let y = startYear, m = startMonth;
        while (y < endYear || (y === endYear && m <= endMonth)) {
            const key = `${y}-${String(m).padStart(2, '0')}`;
            allMonths.push({ key, year: y, month: m, count: monthBuckets[key] || 0 });
            m++;
            if (m > 12) { m = 1; y++; }
        }

        if (allMonths.length === 0) {
            container.classList.add('hidden');
            return;
        }

        const maxCount = Math.max(...allMonths.map(b => b.count));

        // SVG dimensions
        const marginTop = 20;
        const marginBottom = 40;
        const marginLeft = 35;
        const marginRight = 10;
        const chartWidth = Math.max(allMonths.length * 6, 400);
        const svgWidth = chartWidth + marginLeft + marginRight;
        const chartHeight = 120;
        const svgHeight = chartHeight + marginTop + marginBottom;
        const barWidth = Math.max(Math.floor(chartWidth / allMonths.length) - 1, 2);

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', svgHeight);
        svg.style.display = 'block';

        // Y-axis gridlines
        const ySteps = niceSteps(maxCount);
        ySteps.forEach(val => {
            const yPos = marginTop + chartHeight - (val / maxCount) * chartHeight;
            // gridline
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', marginLeft);
            line.setAttribute('x2', svgWidth - marginRight);
            line.setAttribute('y1', yPos);
            line.setAttribute('y2', yPos);
            line.setAttribute('stroke', '#2d3244');
            line.setAttribute('stroke-width', '1');
            svg.appendChild(line);
            // label
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', marginLeft - 5);
            label.setAttribute('y', yPos + 4);
            label.setAttribute('text-anchor', 'end');
            label.setAttribute('fill', '#6b7280');
            label.setAttribute('font-size', '10');
            label.textContent = val;
            svg.appendChild(label);
        });

        // Index docs by month for click lookup
        const docsByMonth = {};
        docs.forEach(d => {
            const key = d.document_date.substring(0, 7);
            if (!docsByMonth[key]) docsByMonth[key] = [];
            docsByMonth[key].push(d);
        });

        // Bars
        allMonths.forEach((bucket, i) => {
            const barHeight = maxCount > 0
                ? (bucket.count / maxCount) * chartHeight
                : 0;
            const x = marginLeft + i * (chartWidth / allMonths.length);
            const y = marginTop + chartHeight - barHeight;

            if (bucket.count > 0) {
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', x);
                rect.setAttribute('y', y);
                rect.setAttribute('width', barWidth);
                rect.setAttribute('height', barHeight);
                rect.setAttribute('fill', '#3b82f6');
                rect.setAttribute('rx', '1');
                rect.style.cursor = 'pointer';

                // Tooltip via title element
                const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
                const monthName = new Date(bucket.year, bucket.month - 1).toLocaleString('default', { month: 'short' });
                title.textContent = `${monthName} ${bucket.year}: ${bucket.count} document${bucket.count > 1 ? 's' : ''} — click to view`;
                rect.appendChild(title);

                rect.addEventListener('click', () => {
                    showTimelineDetail(bucket.key, monthName, bucket.year, docsByMonth[bucket.key] || []);
                });

                svg.appendChild(rect);
            }
        });

        // X-axis: year labels at January of each year
        const seenYears = new Set();
        allMonths.forEach((bucket, i) => {
            if (bucket.month === 1 || (i === 0 && !seenYears.has(bucket.year))) {
                if (seenYears.has(bucket.year)) return;
                seenYears.add(bucket.year);
                const x = marginLeft + i * (chartWidth / allMonths.length);

                // tick mark
                const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tick.setAttribute('x1', x);
                tick.setAttribute('x2', x);
                tick.setAttribute('y1', marginTop + chartHeight);
                tick.setAttribute('y2', marginTop + chartHeight + 6);
                tick.setAttribute('stroke', '#6b7280');
                tick.setAttribute('stroke-width', '1');
                svg.appendChild(tick);

                // year label
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', x);
                label.setAttribute('y', marginTop + chartHeight + 20);
                label.setAttribute('text-anchor', 'middle');
                label.setAttribute('fill', '#9ca3af');
                label.setAttribute('font-size', '11');
                label.setAttribute('font-weight', bucket.year === 2008 ? '700' : '400');
                if (bucket.year === 2008) label.setAttribute('fill', '#f59e0b');
                label.textContent = bucket.year;
                svg.appendChild(label);
            }
        });

        // Baseline
        const baseline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        baseline.setAttribute('x1', marginLeft);
        baseline.setAttribute('x2', svgWidth - marginRight);
        baseline.setAttribute('y1', marginTop + chartHeight);
        baseline.setAttribute('y2', marginTop + chartHeight);
        baseline.setAttribute('stroke', '#4b5563');
        baseline.setAttribute('stroke-width', '1');
        svg.appendChild(baseline);

        chart.appendChild(svg);

        // Note below the chart
        const parts = [];
        parts.push(`${data.total} total documents`);
        parts.push(`${data.total_dated} on timeline (${data.high_confidence} high confidence, ${data.medium_confidence} moderate confidence)`);
        if (data.undated > 0) {
            parts.push(`${data.undated} with uncertain or no date, not charted`);
        }
        note.textContent = parts.join(' \u2014 ');

        // Scroll the timeline into view
        container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (e) {
        console.error('Failed to load timeline:', e);
        container.classList.add('hidden');
    }
}

function niceSteps(max) {
    if (max <= 0) return [0];
    const steps = [];
    let step = 1;
    if (max > 50) step = 10;
    else if (max > 20) step = 5;
    else if (max > 10) step = 2;
    for (let v = step; v <= max; v += step) {
        steps.push(v);
    }
    if (steps.length === 0 || steps[steps.length - 1] < max) steps.push(max);
    return steps;
}

function showTimelineDetail(monthKey, monthName, year, docs) {
    let detail = document.getElementById('timeline-detail');
    if (!detail) {
        detail = document.createElement('div');
        detail.id = 'timeline-detail';
        detail.className = 'timeline-detail';
        const container = document.getElementById('timeline-container');
        container.appendChild(detail);
    }

    let html = `<div class="timeline-detail-header">` +
        `<strong>${monthName} ${year}</strong> — ${docs.length} document${docs.length !== 1 ? 's' : ''}` +
        `<button class="timeline-detail-close" onclick="document.getElementById('timeline-detail').classList.add('hidden')">&times;</button>` +
        `</div>`;
    html += '<div class="timeline-detail-list">';
    docs.forEach(d => {
        const snippetHtml = d.snippet
            ? `<div class="timeline-detail-snippet">${escapeHtml(d.snippet)}</div>`
            : '';
        html += `<div class="timeline-detail-item" onclick="viewDocument(${d.id})">` +
            `<div class="timeline-detail-main">` +
            `<span class="timeline-detail-date">${d.document_date}</span>` +
            `<span class="timeline-detail-name">${escapeHtml(d.filename)}</span>` +
            `<span class="timeline-detail-confidence ${d.date_confidence}">${d.date_confidence}</span>` +
            `</div>` +
            snippetHtml +
            `</div>`;
    });
    html += '</div>';
    detail.innerHTML = html;
    detail.classList.remove('hidden');
    detail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---------------------------------------------------------------------------
// Settings — Prompt Editor
// ---------------------------------------------------------------------------

let promptsLoaded = false;

function initSettings() {
    document.getElementById('prompt-info-btn').addEventListener('click', () => {
        document.getElementById('prompt-info-panel').classList.toggle('hidden');
    });

    document.querySelectorAll('.prompt-editor').forEach(editor => {
        const key = editor.dataset.key;
        editor.querySelector('.prompt-save').addEventListener('click', () => savePrompt(key));
        editor.querySelector('.prompt-reset').addEventListener('click', () => resetPrompt(key));
    });
}

async function loadPrompts() {
    try {
        const resp = await fetch(`${API_BASE}/api/prompts`);
        const data = await resp.json();

        document.querySelectorAll('.prompt-editor').forEach(editor => {
            const key = editor.dataset.key;
            const info = data[key];
            if (!info) return;

            const textarea = editor.querySelector('.prompt-textarea');
            const status = editor.querySelector('.prompt-status');

            textarea.value = info.value;
            textarea.dataset.original = info.value;

            if (info.is_custom) {
                status.textContent = 'customized';
                status.className = 'prompt-status custom';
            } else {
                status.textContent = 'default';
                status.className = 'prompt-status';
            }
        });

        promptsLoaded = true;
    } catch (e) {
        console.error('Failed to load prompts:', e);
    }
}

async function savePrompt(key) {
    const editor = document.querySelector(`.prompt-editor[data-key="${key}"]`);
    const textarea = editor.querySelector('.prompt-textarea');
    const status = editor.querySelector('.prompt-status');
    const value = textarea.value.trim();

    if (!value) {
        alert('Prompt cannot be empty');
        return;
    }

    try {
        const resp = await fetch(`${API_BASE}/api/prompts/${key}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ value }),
        });
        if (resp.ok) {
            status.textContent = 'saved!';
            status.className = 'prompt-status saved';
            textarea.dataset.original = value;
            setTimeout(() => {
                status.textContent = 'customized';
                status.className = 'prompt-status custom';
            }, 2000);
        } else {
            const err = await resp.json();
            alert(err.error || 'Save failed');
        }
    } catch (e) {
        console.error('Save prompt failed:', e);
        alert('Failed to save prompt');
    }
}

async function resetPrompt(key) {
    if (!confirm('Reset this prompt to its default? Your customization will be lost.')) return;

    const editor = document.querySelector(`.prompt-editor[data-key="${key}"]`);
    const textarea = editor.querySelector('.prompt-textarea');
    const status = editor.querySelector('.prompt-status');

    try {
        const resp = await fetch(`${API_BASE}/api/prompts/${key}`, { method: 'DELETE' });
        if (resp.ok) {
            const data = await resp.json();
            textarea.value = data.default;
            textarea.dataset.original = data.default;
            status.textContent = 'default';
            status.className = 'prompt-status';
        }
    } catch (e) {
        console.error('Reset prompt failed:', e);
        alert('Failed to reset prompt');
    }
}

// ---------------------------------------------------------------------------
// LLM Document Summarization
// ---------------------------------------------------------------------------

function runSummarize() {
    const btn = document.getElementById('summarize-btn');
    const progress = document.getElementById('summarize-progress');
    btn.disabled = true;
    btn.textContent = 'Summarizing...';
    progress.classList.remove('hidden');
    progress.textContent = 'Connecting to LLM...';

    const ws = new WebSocket(`ws://${location.host}/ws/summarize`);

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
            case 'estimate':
                progress.textContent =
                    `${msg.content.doc_count} documents to summarize (~${Math.ceil(msg.content.est_seconds / 60)} min)`;
                break;
            case 'progress':
                progress.textContent = msg.content.message;
                break;
            case 'done':
                progress.textContent = msg.content || 'Summarization complete.';
                btn.textContent = 'Generate Document Summaries';
                btn.disabled = false;
                loadTimeline(null);
                break;
            case 'error':
                progress.textContent = 'Error: ' + msg.content;
                btn.textContent = 'Generate Document Summaries';
                btn.disabled = false;
                break;
        }
    };

    ws.onerror = () => {
        progress.textContent = 'WebSocket error';
        btn.textContent = 'Generate Document Summaries';
        btn.disabled = false;
    };

    ws.onclose = () => {
        btn.disabled = false;
        btn.textContent = 'Generate Document Summaries';
    };
}


// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
