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

        list.innerHTML = '';
        data.documents.forEach(doc => {
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
                        ${doc.found_via_query ? ` &middot; query: "${escapeHtml(doc.found_via_query)}"` : ''}
                    </div>
                </div>
                <span class="result-status ${statusClass}">${doc.status}</span>
            `;
            if (doc.status === 'extracted') {
                div.addEventListener('click', () => viewDocument(doc.id));
            }
            list.appendChild(div);
        });
    } catch (e) {
        console.error('Failed to load documents:', e);
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

    output.innerHTML = '<div class="analysis-loading">Connecting...</div>';
    sourcesDiv.classList.add('hidden');
    cancelBtn.classList.remove('hidden');
    hideAnalysisProgress();

    const ws = new WebSocket(`${WS_BASE}/ws/analyze`);
    activeAnalysisWs = ws;

    ws.onopen = () => {
        output.innerHTML = '<div class="analysis-loading">Gathering documents...</div>';
        ws.send(JSON.stringify({ query: query, doc_ids: docIds }));
    };

    let text = '';
    let gotFirstToken = false;
    let batchFindings = [];

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'estimate') {
            const e = msg.content;
            output.innerHTML = `<div class="analysis-loading">
                <strong>${e.doc_count} documents</strong> to analyze
                (${(e.total_chars || 0).toLocaleString()} characters)<br>
                Processing in <strong>${e.batch_count} batches</strong><br>
                Estimated time: <strong>~${formatTime(e.est_seconds)}</strong>
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

        } else if (msg.type === 'done') {
            hideAnalysisProgress();
            cancelBtn.classList.add('hidden');

        } else if (msg.type === 'heartbeat') {
            // keep-alive
        }
    };

    ws.onclose = () => {
        cancelBtn.classList.add('hidden');
        activeAnalysisWs = null;
    };
    ws.onerror = () => {
        hideAnalysisProgress();
        output.innerHTML = '<div class="analysis-error">Connection error</div>';
        cancelBtn.classList.add('hidden');
    };
}

function runCorrelation(names) {
    const output = document.getElementById('analysis-output');
    const sourcesDiv = document.getElementById('analysis-sources');
    const cancelBtn = document.getElementById('cancel-analysis-btn');

    output.innerHTML = '<div class="analysis-loading">Connecting...</div>';
    sourcesDiv.classList.add('hidden');
    cancelBtn.classList.remove('hidden');
    hideAnalysisProgress();

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

        } else if (msg.type === 'done') {
            hideAnalysisProgress();
            cancelBtn.classList.add('hidden');
            if (text) {
                output.innerHTML = '<div class="analysis-text correlation-results">' +
                    renderMarkdown(escapeHtml(text)) + '</div>';
            }

        } else if (msg.type === 'heartbeat') {
            // keep-alive, ignore
        }
    };

    ws.onclose = () => {
        cancelBtn.classList.add('hidden');
        activeAnalysisWs = null;
    };
    ws.onerror = () => {
        hideAnalysisProgress();
        output.innerHTML = '<div class="analysis-error">Connection error</div>';
        cancelBtn.classList.add('hidden');
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
// Utility
// ---------------------------------------------------------------------------

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
