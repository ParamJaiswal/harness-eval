/* ============================================================
   AI Evaluation Harness — Dashboard Application Logic
   ============================================================ */

const App = (() => {
  'use strict';

  // ─── State ──────────────────────────────────────────────────
  const state = {
    runs: [],
    models: [],
    benchmarks: [],
    tools: [],
    selectedRunId: null,
    selectedRunDetail: null,
    selectedRunMetrics: null,
    compareSelectedIds: new Set(),
    compareData: null,
    sortColumn: 'created_at',
    sortDirection: 'desc',
    taskSearchQuery: '',
    charts: {},
    pollTimer: null,
  };

  // ─── Chart.js Global Config ─────────────────────────────────
  function configureCharts() {
    if (typeof Chart === 'undefined') return;
    Chart.defaults.color = '#8888a0';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(18,18,26,0.95)';
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.1)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.titleFont = { weight: '600' };
  }

  // ─── API Helpers ────────────────────────────────────────────
  async function api(url, options = {}) {
    try {
      const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
      }
      const ct = res.headers.get('content-type') || '';
      if (ct.includes('application/json')) return res.json();
      return res;
    } catch (err) {
      console.error(`API error [${url}]:`, err);
      throw err;
    }
  }

  // ─── Toast System ───────────────────────────────────────────
  function toast(type, title, message = '') {
    const container = document.getElementById('toast-container');
    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.innerHTML = `
      <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
      <div class="toast-body">
        <div class="toast-title">${esc(title)}</div>
        ${message ? `<div class="toast-message">${esc(message)}</div>` : ''}
      </div>
      <button class="toast-close" onclick="this.closest('.toast').remove()">✕</button>
    `;
    container.appendChild(el);
    setTimeout(() => {
      el.classList.add('leaving');
      setTimeout(() => el.remove(), 300);
    }, 4500);
  }

  // ─── Utility ────────────────────────────────────────────────
  function esc(s) {
    if (s == null) return '';
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

  function shortId(id) {
    if (!id) return '—';
    return String(id).length > 12 ? String(id).substring(0, 8) + '…' : String(id);
  }

  function fmtDate(d) {
    if (!d) return '—';
    try {
      const dt = new Date(d);
      return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch { return String(d); }
  }

  function fmtScore(s) {
    if (s == null || s === '') return '—';
    const n = Number(s);
    if (isNaN(n)) return String(s);
    return (n * (n <= 1 ? 100 : 1)).toFixed(1) + '%';
  }

  function scoreClass(s) {
    if (s == null) return '';
    const n = Number(s);
    const v = n <= 1 ? n * 100 : n;
    if (v >= 75) return 'score-high';
    if (v >= 50) return 'score-mid';
    return 'score-low';
  }

  function statusBadge(status) {
    const s = (status || '').toLowerCase();
    const labels = { pending: 'Pending', running: 'Running', completed: 'Completed', failed: 'Failed' };
    return `<span class="badge badge-${s}"><span class="badge-dot"></span>${labels[s] || esc(status)}</span>`;
  }

  function evalTypeIcon(type) {
    const t = (type || '').toLowerCase();
    const icons = { llm: '🧠', agent: '🤖', rag: '📚' };
    return `<span class="eval-type-icon">${icons[t] || '📄'} ${esc(type?.toUpperCase?.() || type)}</span>`;
  }

  function adapterIcon(type) {
    const t = (type || '').toLowerCase();
    if (t.includes('llm')) return { icon: '🧠', cls: 'card-icon-llm' };
    if (t.includes('agent')) return { icon: '🤖', cls: 'card-icon-agent' };
    if (t.includes('rag')) return { icon: '📚', cls: 'card-icon-rag' };
    return { icon: '📄', cls: 'card-icon-llm' };
  }

  // ─── Tab Navigation ─────────────────────────────────────────
  function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === tabId);
      b.setAttribute('aria-selected', b.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.toggle('active', p.id === `tab-${tabId}`);
    });
    // Load data on tab switch
    if (tabId === 'run-eval') loadModelsAndBenchmarks();
    if (tabId === 'results') populateResultsDropdown();
    if (tabId === 'compare') populateCompareChips();
  }

  function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
  }

  // ─── Inner Tab Navigation (Results) ─────────────────────────
  function switchInnerTab(btn, panelId) {
    document.querySelectorAll('#result-inner-tabs .inner-tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('#results-detail .inner-panel').forEach(p => p.classList.remove('active'));
    const panel = document.getElementById(`inner-${panelId}`);
    if (panel) panel.classList.add('active');
  }

  // ============================================================
  //  DATA LOADING
  // ============================================================

  async function loadRuns() {
    try {
      state.runs = await api('/api/eval/runs');
      if (!Array.isArray(state.runs)) state.runs = [];
      renderRunsTable();
      updateHeroStats();
    } catch (err) {
      toast('error', 'Failed to load runs', err.message);
      // Show empty state
      document.getElementById('runs-skeleton').classList.add('hidden');
      document.getElementById('runs-empty').classList.remove('hidden');
    }
  }

  async function loadModelsAndBenchmarks() {
    try {
      const [models, benchmarks] = await Promise.all([
        api('/api/models').catch(() => []),
        api('/api/benchmarks').catch(() => []),
      ]);
      state.models = Array.isArray(models) ? models : [];
      state.benchmarks = Array.isArray(benchmarks) ? benchmarks : [];
      renderModelCards();
      renderBenchmarkCards();
    } catch (err) {
      toast('error', 'Failed to load models/benchmarks', err.message);
    }
  }

  async function loadRunDetail(runId) {
    if (!runId) {
      document.getElementById('results-empty').classList.remove('hidden');
      document.getElementById('results-detail').classList.add('hidden');
      document.getElementById('export-btn').disabled = true;
      return;
    }
    state.selectedRunId = runId;
    document.getElementById('results-empty').classList.add('hidden');
    document.getElementById('results-detail').classList.remove('hidden');
    document.getElementById('export-btn').disabled = false;

    try {
      const [detail, metrics] = await Promise.all([
        api(`/api/eval/runs/${runId}`),
        api(`/api/metrics/${runId}`).catch(() => null),
      ]);
      state.selectedRunDetail = detail;
      state.selectedRunMetrics = metrics;
      renderRunDetail();
    } catch (err) {
      toast('error', 'Failed to load run details', err.message);
    }
  }

  // ============================================================
  //  OVERVIEW TAB
  // ============================================================

  function updateHeroStats() {
    const runs = state.runs;
    document.getElementById('stat-total-runs').textContent = runs.length;

    const uniqueModels = new Set(runs.map(r => r.model_name)).size;
    document.getElementById('stat-models').textContent = uniqueModels;

    const completed = runs.filter(r => r.status === 'completed' && r.total_score != null);
    if (completed.length > 0) {
      const avg = completed.reduce((s, r) => {
        const v = Number(r.total_score);
        return s + (v <= 1 ? v * 100 : v);
      }, 0) / completed.length;
      document.getElementById('stat-avg-accuracy').textContent = avg.toFixed(1) + '%';
    } else {
      document.getElementById('stat-avg-accuracy').textContent = '—';
    }

    const totalTasks = runs.reduce((s, r) => s + (r.total_tasks || 0), 0);
    document.getElementById('stat-total-tasks').textContent = totalTasks.toLocaleString();
  }

  function renderRunsTable() {
    const skeleton = document.getElementById('runs-skeleton');
    const table = document.getElementById('runs-table');
    const empty = document.getElementById('runs-empty');
    const tbody = document.getElementById('runs-tbody');

    skeleton.classList.add('hidden');

    if (state.runs.length === 0) {
      table.classList.add('hidden');
      empty.classList.remove('hidden');
      return;
    }
    empty.classList.add('hidden');
    table.classList.remove('hidden');

    // Sort
    const sorted = [...state.runs].sort((a, b) => {
      const col = state.sortColumn;
      let va = a[col], vb = b[col];
      if (col === 'created_at') {
        va = new Date(va || 0).getTime();
        vb = new Date(vb || 0).getTime();
      }
      if (typeof va === 'string') va = va.toLowerCase();
      if (typeof vb === 'string') vb = vb.toLowerCase();
      if (va < vb) return state.sortDirection === 'asc' ? -1 : 1;
      if (va > vb) return state.sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    tbody.innerHTML = sorted.map(run => {
      const progress = run.total_tasks ? Math.round((run.completed_tasks || 0) / run.total_tasks * 100) : 0;
      return `
        <tr>
          <td><span class="run-id" onclick="App.viewRun('${esc(run.id)}')" title="${esc(run.id)}">${esc(shortId(run.id))}</span></td>
          <td>${esc(run.model_name)}</td>
          <td>${esc(run.benchmark_name)}</td>
          <td>${evalTypeIcon(run.eval_type)}</td>
          <td>${statusBadge(run.status)}</td>
          <td><span class="score-value ${scoreClass(run.total_score)}">${fmtScore(run.total_score)}</span></td>
          <td>
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <div style="flex:1;height:4px;background:var(--bg-tertiary);border-radius:2px;min-width:60px;">
                <div style="height:100%;width:${progress}%;background:var(--gradient-primary);border-radius:2px;transition:width 0.5s ease;"></div>
              </div>
              <span class="text-xs mono text-muted">${run.completed_tasks || 0}/${run.total_tasks || 0}</span>
            </div>
          </td>
          <td class="text-muted text-sm">${fmtDate(run.created_at)}</td>
          <td>
            <button class="btn btn-ghost btn-sm" onclick="App.viewRun('${esc(run.id)}')" title="View details">📋</button>
          </td>
        </tr>
      `;
    }).join('');

    // Header sort clicks
    document.querySelectorAll('#runs-table thead th[data-sort]').forEach(th => {
      th.onclick = () => {
        const col = th.dataset.sort;
        if (state.sortColumn === col) {
          state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
          state.sortColumn = col;
          state.sortDirection = 'asc';
        }
        renderRunsTable();
      };
    });
  }

  function viewRun(runId) {
    switchTab('results');
    const sel = document.getElementById('results-run-select');
    sel.value = runId;
    loadRunDetail(runId);
  }

  // ============================================================
  //  RUN EVALUATION TAB
  // ============================================================

  function renderModelCards() {
    const container = document.getElementById('model-cards-container');
    const select = document.getElementById('eval-model-select');

    if (state.models.length === 0) {
      container.innerHTML = `
        <div class="glass-card" style="padding:2rem;text-align:center;">
          <div style="font-size:1.5rem;margin-bottom:0.5rem;opacity:0.3;">🤖</div>
          <div class="text-sm text-muted">No models registered yet.<br>Add models via the API.</div>
        </div>
      `;
      select.innerHTML = '<option value="">No models available</option>';
      return;
    }

    select.innerHTML = '<option value="">Choose a model…</option>' +
      state.models.map(m => `<option value="${esc(m.name)}">${esc(m.name)}</option>`).join('');

    container.innerHTML = state.models.map(m => {
      const ai = adapterIcon(m.adapter_type);
      return `
        <div class="glass-card model-card" data-model="${esc(m.name)}" onclick="App.selectModelCard(this, '${esc(m.name)}')">
          <div class="card-header">
            <div class="card-icon ${ai.cls}">${ai.icon}</div>
            <div>
              <div class="card-title">${esc(m.name)}</div>
              <div class="card-subtitle">${esc(m.adapter_type || 'LLM')}</div>
            </div>
          </div>
          <div class="card-desc">${esc(m.description || 'No description available.')}</div>
        </div>
      `;
    }).join('');
  }

  function selectModelCard(cardEl, modelName) {
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
    cardEl.classList.add('selected');
    document.getElementById('eval-model-select').value = modelName;

    // Auto-detect eval type from adapter
    const model = state.models.find(m => m.name === modelName);
    if (model) {
      const t = (model.adapter_type || '').toLowerCase();
      if (t.includes('agent')) document.getElementById('eval-type-select').value = 'agent';
      else if (t.includes('rag')) document.getElementById('eval-type-select').value = 'rag';
      else document.getElementById('eval-type-select').value = 'llm';
    }
  }

  function renderBenchmarkCards() {
    const container = document.getElementById('benchmark-cards-container');
    const select = document.getElementById('eval-benchmark-select');

    if (state.benchmarks.length === 0) {
      container.innerHTML = `
        <div class="glass-card" style="padding:2rem;text-align:center;">
          <div style="font-size:1.5rem;margin-bottom:0.5rem;opacity:0.3;">📝</div>
          <div class="text-sm text-muted">No benchmarks registered yet.<br>Add benchmarks via the API.</div>
        </div>
      `;
      select.innerHTML = '<option value="">No benchmarks available</option>';
      return;
    }

    select.innerHTML = '<option value="">Choose a benchmark…</option>' +
      state.benchmarks.map(b => `<option value="${esc(b.name)}">${esc(b.name)}</option>`).join('') +
      '<option value="custom">Create Custom Benchmark...</option>';

    container.innerHTML = state.benchmarks.map(b => `
      <div class="glass-card benchmark-card" data-benchmark="${esc(b.name)}" onclick="App.selectBenchmarkCard(this, '${esc(b.name)}')">
        <div class="card-header">
          <div class="card-icon card-icon-llm">📝</div>
          <div>
            <div class="card-title">${esc(b.name)}</div>
            <div class="card-subtitle">${esc(b.category || 'General')}</div>
          </div>
        </div>
        <div class="card-desc">${esc(b.description || 'No description available.')}</div>
        <div class="card-meta">
          <span>📋 ${b.task_count || 0} tasks</span>
          <span>📂 ${esc(b.category || 'General')}</span>
        </div>
      </div>
    `).join('');
  }

  function selectBenchmarkCard(cardEl, benchName) {
    document.querySelectorAll('.benchmark-card').forEach(c => c.classList.remove('selected'));
    cardEl.classList.add('selected');
    document.getElementById('eval-benchmark-select').value = benchName;
    onBenchmarkChange(benchName);
  }

  async function startEvaluation() {
    const modelName = document.getElementById('eval-model-select').value;
    const benchmarkName = document.getElementById('eval-benchmark-select').value;
    const evalType = document.getElementById('eval-type-select').value;

    if (!modelName) { toast('warning', 'Select a model', 'Please choose a model to evaluate.'); return; }
    if (!benchmarkName) { toast('warning', 'Select a benchmark', 'Please choose a benchmark to run.'); return; }

    const btn = document.getElementById('run-eval-btn');
    const progress = document.getElementById('eval-progress');
    const statusText = document.getElementById('eval-status-text');

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Starting…';

    try {
      const openaiKey = localStorage.getItem('eval_openai_api_key') || '';
      const anthropicKey = localStorage.getItem('eval_anthropic_api_key') || '';
      const customKey = localStorage.getItem('eval_custom_api_key') || '';
      const customBaseUrl = localStorage.getItem('eval_custom_base_url') || '';

      const effectiveOpenaiKey = customBaseUrl ? (customKey || openaiKey) : openaiKey;

      const payload = {
        model_name: modelName,
        benchmark_name: benchmarkName,
        eval_type: evalType,
        openai_api_key: effectiveOpenaiKey || undefined,
        anthropic_api_key: anthropicKey || undefined,
        custom_base_url: customBaseUrl || undefined,
      };

      if (benchmarkName === 'custom') {
        const customTasks = getCustomTasks();
        if (customTasks.length === 0) {
          toast('warning', 'Empty Custom Benchmark', 'Please add at least one task with a prompt.');
          btn.disabled = false;
          btn.innerHTML = '⚡ Run Evaluation';
          return;
        }
        payload.custom_tasks = customTasks;
      }

      const result = await api('/api/eval/run', {
        method: 'POST',
        body: JSON.stringify(payload),
      });

      toast('success', 'Evaluation started', `Run ID: ${shortId(result.id)}`);
      statusText.textContent = `Run ${shortId(result.id)} in progress…`;
      progress.classList.remove('hidden');

      // Poll for progress
      pollEvalProgress(result.id);
    } catch (err) {
      toast('error', 'Failed to start evaluation', err.message);
      btn.disabled = false;
      btn.innerHTML = '⚡ Run Evaluation';
    }
  }

  function pollEvalProgress(runId) {
    if (state.pollTimer) clearInterval(state.pollTimer);

    state.pollTimer = setInterval(async () => {
      try {
        const run = await api(`/api/eval/runs/${runId}`);
        const total = run.total_tasks || 1;
        const completed = run.completed_tasks || 0;
        const pct = Math.round(completed / total * 100);

        document.getElementById('eval-progress-fill').style.width = pct + '%';
        document.getElementById('eval-progress-pct').textContent = pct + '%';
        document.getElementById('eval-progress-label').textContent =
          `Running evaluation… ${completed}/${total} tasks`;

        if (run.status === 'completed' || run.status === 'failed') {
          clearInterval(state.pollTimer);
          state.pollTimer = null;

          const btn = document.getElementById('run-eval-btn');
          btn.disabled = false;
          btn.innerHTML = '⚡ Run Evaluation';

          if (run.status === 'completed') {
            document.getElementById('eval-progress-fill').style.width = '100%';
            document.getElementById('eval-progress-pct').textContent = '100%';
            document.getElementById('eval-progress-label').textContent = 'Evaluation complete!';
            document.getElementById('eval-status-text').textContent = '';
            toast('success', 'Evaluation complete!', `Score: ${fmtScore(run.total_score)}`);
          } else {
            toast('error', 'Evaluation failed', 'The run did not complete successfully.');
          }
          loadRuns(); // Refresh overview
        }
      } catch (err) {
        // Network error, keep polling
      }
    }, 1500);
  }

  // ============================================================
  //  RESULTS TAB
  // ============================================================

  function populateResultsDropdown() {
    const sel = document.getElementById('results-run-select');
    const currentVal = sel.value;
    sel.innerHTML = '<option value="">Select a run to view results…</option>' +
      state.runs.map(r =>
        `<option value="${esc(r.id)}" ${r.id === currentVal ? 'selected' : ''}>` +
        `${esc(shortId(r.id))} — ${esc(r.model_name)} / ${esc(r.benchmark_name)} (${esc(r.status)})` +
        `</option>`
      ).join('');
  }

  function renderRunDetail() {
    const d = state.selectedRunDetail;
    const m = state.selectedRunMetrics;
    if (!d) return;

    document.getElementById('result-title').textContent = `${d.model_name} → ${d.benchmark_name}`;
    document.getElementById('result-meta').innerHTML = `
      <span>🆔 ${esc(shortId(d.id))}</span>
      <span>📅 ${fmtDate(d.created_at)}</span>
    `;

    const statusEl = document.getElementById('result-status-badge');
    statusEl.outerHTML = statusBadge(d.status);

    const typeEl = document.getElementById('result-type-badge');
    typeEl.innerHTML = evalTypeIcon(d.eval_type);

    // Metrics
    if (m) {
      const accVal = m.accuracy != null ? m.accuracy : (d.total_score != null ? d.total_score : null);
      const accDisplay = accVal != null ? fmtScore(accVal) : '—';
      document.getElementById('metric-accuracy').textContent = accDisplay;
      document.getElementById('metric-accuracy').className = `metric-value ${scoreClass(accVal)}`;

      document.getElementById('metric-latency').textContent =
        m.avg_latency_ms != null ? m.avg_latency_ms.toFixed(0) + 'ms' : '—';
      document.getElementById('metric-cost').textContent =
        m.total_cost_usd != null ? '$' + m.total_cost_usd.toFixed(4) : '—';
    } else {
      document.getElementById('metric-accuracy').textContent = fmtScore(d.total_score);
      document.getElementById('metric-accuracy').className = `metric-value ${scoreClass(d.total_score)}`;
      document.getElementById('metric-latency').textContent = '—';
      document.getElementById('metric-cost').textContent = '—';
    }

    // Show/hide trajectory and context tabs based on eval type
    const evalType = (d.eval_type || '').toLowerCase();
    const trajBtn = document.getElementById('trajectory-tab-btn');
    const ctxBtn = document.getElementById('context-tab-btn');
    trajBtn.classList.toggle('hidden', evalType !== 'agent');
    ctxBtn.classList.toggle('hidden', evalType !== 'rag');

    // Render tasks table
    renderTasksTable(d.results || []);

    // Render charts
    renderResultCharts(d, m);

    // Render trajectory/context if applicable
    if (evalType === 'agent') renderTrajectory(d.results || []);
    if (evalType === 'rag') renderContextCards(d.results || []);
  }

  function renderTasksTable(results) {
    const tbody = document.getElementById('tasks-tbody');
    state._allResults = results;

    const filtered = state.taskSearchQuery
      ? results.filter(r => JSON.stringify(r).toLowerCase().includes(state.taskSearchQuery.toLowerCase()))
      : results;

    if (filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:2rem;color:var(--text-secondary);">No task results available.</td></tr>`;
      return;
    }

    tbody.innerHTML = filtered.map((r, i) => `
      <tr class="expandable-toggle" onclick="App.toggleExpand(this, ${i})">
        <td><span class="expand-icon">▸</span></td>
        <td><span class="mono text-sm">${esc(r.task_id || `Task ${i + 1}`)}</span></td>
        <td><span class="score-value ${scoreClass(r.score)}">${r.score != null ? fmtScore(r.score) : '—'}</span></td>
        <td><span class="latency-value">${r.latency_ms != null ? r.latency_ms.toFixed(0) + 'ms' : '—'}</span></td>
        <td><span class="mono text-sm">${r.tokens_used != null ? r.tokens_used.toLocaleString() : '—'}</span></td>
        <td><span class="badge badge-type">${esc(r.scoring_method || '—')}</span></td>
      </tr>
      <tr class="expanded-content-row">
        <td colspan="6" style="padding:0;">
          <div class="expanded-content" id="expand-${i}">
            <div class="output-comparison">
              <div class="output-block">
                <div class="output-block-label">Raw Output</div>
                <pre>${esc(r.raw_output || 'No output recorded.')}</pre>
              </div>
              <div class="output-block">
                <div class="output-block-label">Expected Output</div>
                <pre>${esc(r.expected_output || 'No expected output defined.')}</pre>
              </div>
            </div>
            ${r.metadata_json ? `
              <div class="output-block mt-1">
                <div class="output-block-label">Metadata</div>
                <pre>${esc(typeof r.metadata_json === 'string' ? r.metadata_json : JSON.stringify(r.metadata_json, null, 2))}</pre>
              </div>
            ` : ''}
          </div>
        </td>
      </tr>
    `).join('');
  }

  function toggleExpand(row, idx) {
    row.classList.toggle('open');
    const content = document.getElementById(`expand-${idx}`);
    if (content) content.classList.toggle('show');
  }

  function filterTasks(query) {
    state.taskSearchQuery = query;
    if (state._allResults) renderTasksTable(state._allResults);
  }

  // ─── Charts ─────────────────────────────────────────────────
  function destroyChart(key) {
    if (state.charts[key]) {
      state.charts[key].destroy();
      delete state.charts[key];
    }
  }

  function renderResultCharts(detail, metrics) {
    const results = detail.results || [];
    if (results.length === 0) return;

    const accentColors = ['#4f8fff', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4'];

    // Per-task scores bar chart
    destroyChart('taskScores');
    const taskLabels = results.map((r, i) => r.task_id || `T${i + 1}`);
    const taskScores = results.map(r => {
      const s = Number(r.score);
      return isNaN(s) ? 0 : (s <= 1 ? s * 100 : s);
    });
    const taskColors = taskScores.map(s => s >= 75 ? '#22c55e' : s >= 50 ? '#f59e0b' : '#ef4444');

    const taskCtx = document.getElementById('chart-task-scores');
    if (taskCtx) {
      state.charts.taskScores = new Chart(taskCtx, {
        type: 'bar',
        data: {
          labels: taskLabels.length > 30 ? taskLabels.slice(0, 30) : taskLabels,
          datasets: [{
            label: 'Score (%)',
            data: taskScores.length > 30 ? taskScores.slice(0, 30) : taskScores,
            backgroundColor: (taskColors.length > 30 ? taskColors.slice(0, 30) : taskColors).map(c => c + '88'),
            borderColor: taskColors.length > 30 ? taskColors.slice(0, 30) : taskColors,
            borderWidth: 1,
            borderRadius: 4,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.04)' } },
            x: { grid: { display: false }, ticks: { maxRotation: 45 } },
          },
        },
      });
    }

    // Latency bar chart
    destroyChart('latency');
    const latencies = results.map(r => r.latency_ms || 0);
    const latCtx = document.getElementById('chart-latency');
    if (latCtx) {
      state.charts.latency = new Chart(latCtx, {
        type: 'bar',
        data: {
          labels: taskLabels.length > 30 ? taskLabels.slice(0, 30) : taskLabels,
          datasets: [{
            label: 'Latency (ms)',
            data: latencies.length > 30 ? latencies.slice(0, 30) : latencies,
            backgroundColor: 'rgba(79,143,255,0.4)',
            borderColor: '#4f8fff',
            borderWidth: 1,
            borderRadius: 4,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
            x: { grid: { display: false }, ticks: { maxRotation: 45 } },
          },
        },
      });
    }

    // Score distribution doughnut
    destroyChart('scoreDist');
    const high = taskScores.filter(s => s >= 75).length;
    const mid = taskScores.filter(s => s >= 50 && s < 75).length;
    const low = taskScores.filter(s => s < 50).length;
    const distCtx = document.getElementById('chart-score-dist');
    if (distCtx) {
      state.charts.scoreDist = new Chart(distCtx, {
        type: 'doughnut',
        data: {
          labels: ['High (≥75%)', 'Medium (50-75%)', 'Low (<50%)'],
          datasets: [{
            data: [high, mid, low],
            backgroundColor: ['rgba(34,197,94,0.6)', 'rgba(245,158,11,0.6)', 'rgba(239,68,68,0.6)'],
            borderColor: ['#22c55e', '#f59e0b', '#ef4444'],
            borderWidth: 2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '60%',
          plugins: {
            legend: { position: 'bottom', labels: { padding: 16 } },
          },
        },
      });
    }

    // Latency percentiles horizontal bar
    destroyChart('latencyPct');
    if (metrics && metrics.p50_latency != null) {
      const pctCtx = document.getElementById('chart-latency-pct');
      if (pctCtx) {
        state.charts.latencyPct = new Chart(pctCtx, {
          type: 'bar',
          data: {
            labels: ['p50', 'p95', 'p99', 'Average'],
            datasets: [{
              label: 'Latency (ms)',
              data: [
                metrics.p50_latency || 0,
                metrics.p95_latency || 0,
                metrics.p99_latency || 0,
                metrics.avg_latency_ms || 0,
              ],
              backgroundColor: [
                'rgba(79,143,255,0.5)',
                'rgba(139,92,246,0.5)',
                'rgba(245,158,11,0.5)',
                'rgba(34,197,94,0.5)',
              ],
              borderColor: ['#4f8fff', '#8b5cf6', '#f59e0b', '#22c55e'],
              borderWidth: 1,
              borderRadius: 4,
            }],
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
              y: { grid: { display: false } },
            },
          },
        });
      }
    }
  }

  // ─── Trajectory (Agent Runs) ────────────────────────────────
  function renderTrajectory(results) {
    const container = document.getElementById('trajectory-container');
    // Try to extract trajectory from metadata
    const allSteps = [];
    results.forEach(r => {
      let meta = r.metadata_json;
      if (typeof meta === 'string') {
        try { meta = JSON.parse(meta); } catch {}
      }
      if (meta && Array.isArray(meta.trajectory)) {
        meta.trajectory.forEach(step => allSteps.push(step));
      } else if (meta && meta.steps) {
        (Array.isArray(meta.steps) ? meta.steps : []).forEach(step => allSteps.push(step));
      }
    });

    if (allSteps.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">🛤️</div>
          <div class="empty-state-title">No trajectory data available</div>
          <div class="empty-state-desc">Agent trajectory steps will appear here when the evaluation includes step-by-step agent traces with tool calls.</div>
        </div>
      `;
      return;
    }

    container.innerHTML = allSteps.map((step, i) => {
      const type = (step.type || step.action || 'thought').toLowerCase();
      let iconCls = 'timeline-icon-thought';
      let icon = '💭';
      if (type.includes('tool') || type.includes('action')) { iconCls = 'timeline-icon-tool'; icon = '🔧'; }
      else if (type.includes('result') || type.includes('observation')) { iconCls = 'timeline-icon-result'; icon = '✅'; }
      else if (type.includes('error')) { iconCls = 'timeline-icon-error'; icon = '❌'; }

      return `
        <div class="timeline-step">
          <div class="timeline-icon ${iconCls}">${icon}</div>
          <div class="timeline-body">
            <div class="timeline-title">${esc(step.type || step.action || `Step ${i + 1}`)}</div>
            ${step.tool ? `<span class="badge badge-type" style="margin-bottom:0.25rem;">🔧 ${esc(step.tool)}</span>` : ''}
            <div class="timeline-detail">${esc(step.content || step.output || step.input || JSON.stringify(step, null, 2))}</div>
          </div>
        </div>
      `;
    }).join('');
  }

  // ─── Context Cards (RAG Runs) ───────────────────────────────
  function renderContextCards(results) {
    const container = document.getElementById('context-container');
    const allContexts = [];
    results.forEach(r => {
      let meta = r.metadata_json;
      if (typeof meta === 'string') {
        try { meta = JSON.parse(meta); } catch {}
      }
      if (meta && Array.isArray(meta.contexts)) {
        meta.contexts.forEach((ctx, i) => allContexts.push({ ...ctx, rank: i + 1 }));
      } else if (meta && Array.isArray(meta.retrieved_passages)) {
        meta.retrieved_passages.forEach((ctx, i) => allContexts.push({ ...ctx, rank: i + 1 }));
      }
    });

    if (allContexts.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📚</div>
          <div class="empty-state-title">No context data available</div>
          <div class="empty-state-desc">Retrieved passages and relevance scores will appear here when the evaluation includes RAG retrieval data.</div>
        </div>
      `;
      return;
    }

    container.innerHTML = allContexts.map(ctx => `
      <div class="glass-card context-card">
        <div class="context-rank">${ctx.rank}</div>
        <div class="context-body">
          <div class="context-text">${esc(ctx.text || ctx.passage || ctx.content || '')}</div>
          <div class="context-meta">
            ${ctx.relevance_score != null ? `<span>Relevance: <span class="relevance-score">${(ctx.relevance_score * 100).toFixed(1)}%</span></span>` : ''}
            ${ctx.source ? `<span>Source: ${esc(ctx.source)}</span>` : ''}
          </div>
        </div>
      </div>
    `).join('');
  }

  // ─── Export ─────────────────────────────────────────────────
  function exportRun() {
    if (!state.selectedRunId) return;
    window.open(`/api/export/${state.selectedRunId}?format=json`, '_blank');
    toast('info', 'Export started', 'JSON file download initiated.');
  }

  // ============================================================
  //  COMPARE TAB
  // ============================================================

  function populateCompareChips() {
    const container = document.getElementById('compare-chips');
    const completedRuns = state.runs.filter(r => r.status === 'completed');

    if (completedRuns.length === 0) {
      container.innerHTML = '<span class="text-sm text-muted">No completed runs available for comparison.</span>';
      return;
    }

    container.innerHTML = completedRuns.map(r => `
      <div class="compare-chip ${state.compareSelectedIds.has(r.id) ? 'selected' : ''}"
           data-run-id="${esc(r.id)}"
           onclick="App.toggleCompareChip(this, '${esc(r.id)}')">
        <span class="check-icon">✓</span>
        <span>${esc(r.model_name)} / ${esc(r.benchmark_name)}</span>
        <span class="text-xs mono">${esc(shortId(r.id))}</span>
      </div>
    `).join('');

    updateCompareButton();
  }

  function toggleCompareChip(el, runId) {
    if (state.compareSelectedIds.has(runId)) {
      state.compareSelectedIds.delete(runId);
      el.classList.remove('selected');
    } else {
      state.compareSelectedIds.add(runId);
      el.classList.add('selected');
    }
    updateCompareButton();
  }

  function updateCompareButton() {
    const btn = document.getElementById('compare-btn');
    const count = state.compareSelectedIds.size;
    btn.disabled = count < 2;
    btn.textContent = count < 2
      ? `⚖️ Select ${2 - count} more run${2 - count > 1 ? 's' : ''}`
      : `⚖️ Compare ${count} Runs`;
  }

  async function runComparison() {
    const ids = Array.from(state.compareSelectedIds);
    if (ids.length < 2) return;

    const btn = document.getElementById('compare-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Comparing…';

    try {
      const data = await api(`/api/eval/compare?run_ids=${ids.join(',')}`);
      state.compareData = data;
      renderComparison(data);
      document.getElementById('compare-empty').classList.add('hidden');
      document.getElementById('compare-results').classList.remove('hidden');
    } catch (err) {
      toast('error', 'Comparison failed', err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `⚖️ Compare ${ids.length} Runs`;
    }
  }

  function renderComparison(data) {
    const metrics = data.metrics || [];
    const dims = data.dimension_comparison || {};

    if (metrics.length === 0) {
      toast('warning', 'No metrics data', 'The comparison returned empty metrics.');
      return;
    }

    // Winner determination
    const best = metrics.reduce((a, b) => {
      const accA = a.accuracy != null ? (a.accuracy <= 1 ? a.accuracy : a.accuracy / 100) : 0;
      const accB = b.accuracy != null ? (b.accuracy <= 1 ? b.accuracy : b.accuracy / 100) : 0;
      return accA >= accB ? a : b;
    });

    const bestRun = state.runs.find(r => r.id === best.run_id);
    const bestAcc = best.accuracy != null ? fmtScore(best.accuracy) : '—';
    document.getElementById('winner-name').textContent = bestRun
      ? `${bestRun.model_name} — ${bestRun.benchmark_name}`
      : shortId(best.run_id);
    document.getElementById('winner-detail').textContent =
      `Highest accuracy: ${bestAcc} | Avg Latency: ${best.avg_latency_ms ? best.avg_latency_ms.toFixed(0) + 'ms' : '—'}`;

    // Colors for runs
    const runColors = ['#4f8fff', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4'];

    // Radar chart
    destroyChart('compareRadar');
    const radarLabels = Object.keys(dims).length > 0
      ? Object.keys(dims)
      : ['Accuracy', 'Latency', 'Cost', 'Consistency'];

    const radarDatasets = metrics.map((m, idx) => {
      const run = state.runs.find(r => r.id === m.run_id);
      const label = run ? run.model_name : shortId(m.run_id);
      let dataPoints;
      if (Object.keys(dims).length > 0 && dims[radarLabels[0]]) {
        dataPoints = radarLabels.map(dim => {
          const dimData = dims[dim];
          if (dimData && dimData[m.run_id] != null) {
            const v = dimData[m.run_id];
            return v <= 1 ? v * 100 : v;
          }
          return 0;
        });
      } else {
        // Synthesize from metrics
        const acc = m.accuracy != null ? (m.accuracy <= 1 ? m.accuracy * 100 : m.accuracy) : 0;
        const maxLat = Math.max(...metrics.map(x => x.avg_latency_ms || 1));
        const latScore = maxLat > 0 ? (1 - (m.avg_latency_ms || 0) / maxLat) * 100 : 50;
        const maxCost = Math.max(...metrics.map(x => x.total_cost_usd || 0.001));
        const costScore = maxCost > 0 ? (1 - (m.total_cost_usd || 0) / maxCost) * 100 : 50;
        dataPoints = [acc, Math.max(0, latScore), Math.max(0, costScore), acc * 0.95];
      }

      const color = runColors[idx % runColors.length];
      return {
        label,
        data: dataPoints,
        borderColor: color,
        backgroundColor: color + '22',
        pointBackgroundColor: color,
        pointBorderColor: '#fff',
        pointBorderWidth: 1,
        borderWidth: 2,
      };
    });

    const radarCtx = document.getElementById('chart-compare-radar');
    if (radarCtx) {
      state.charts.compareRadar = new Chart(radarCtx, {
        type: 'radar',
        data: { labels: radarLabels, datasets: radarDatasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            r: {
              beginAtZero: true,
              max: 100,
              grid: { color: 'rgba(255,255,255,0.06)' },
              angleLines: { color: 'rgba(255,255,255,0.06)' },
              pointLabels: { font: { size: 11 }, color: '#8888a0' },
              ticks: { display: false },
            },
          },
          plugins: {
            legend: { position: 'bottom', labels: { padding: 16 } },
          },
        },
      });
    }

    // Grouped bar chart
    destroyChart('compareBar');
    const barLabels = ['Accuracy (%)', 'Avg Latency (ms)', 'Total Tokens', 'Cost ($)'];
    const barDatasets = metrics.map((m, idx) => {
      const run = state.runs.find(r => r.id === m.run_id);
      const label = run ? run.model_name : shortId(m.run_id);
      const color = runColors[idx % runColors.length];
      return {
        label,
        data: [
          m.accuracy != null ? (m.accuracy <= 1 ? m.accuracy * 100 : m.accuracy) : 0,
          m.avg_latency_ms || 0,
          (m.total_tokens || 0) / 1000, // Show in thousands
          (m.total_cost_usd || 0) * 100, // Show in cents for visibility
        ],
        backgroundColor: color + '66',
        borderColor: color,
        borderWidth: 1,
        borderRadius: 4,
      };
    });

    const barCtx = document.getElementById('chart-compare-bar');
    if (barCtx) {
      state.charts.compareBar = new Chart(barCtx, {
        type: 'bar',
        data: { labels: barLabels, datasets: barDatasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'bottom', labels: { padding: 16 } },
          },
          scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
            x: { grid: { display: false } },
          },
        },
      });
    }

    // Detailed comparison table
    const thead = document.getElementById('compare-table-header');
    const tbody = document.getElementById('compare-table-body');
    thead.innerHTML = '<th>Metric</th>' + metrics.map((m, idx) => {
      const run = state.runs.find(r => r.id === m.run_id);
      return `<th>${esc(run ? run.model_name : shortId(m.run_id))}</th>`;
    }).join('');

    const metricRows = [
      { key: 'accuracy', label: 'Accuracy', fmt: v => fmtScore(v) },
      { key: 'avg_latency_ms', label: 'Avg Latency', fmt: v => v != null ? v.toFixed(1) + 'ms' : '—' },
      { key: 'p50_latency', label: 'P50 Latency', fmt: v => v != null ? v.toFixed(1) + 'ms' : '—' },
      { key: 'p95_latency', label: 'P95 Latency', fmt: v => v != null ? v.toFixed(1) + 'ms' : '—' },
      { key: 'total_tokens', label: 'Total Tokens', fmt: v => v != null ? v.toLocaleString() : '—' },
      { key: 'total_cost_usd', label: 'Total Cost', fmt: v => v != null ? '$' + v.toFixed(4) : '—' },
    ];

    tbody.innerHTML = metricRows.map(mr => {
      const vals = metrics.map(m => m[mr.key]);
      const best = mr.key === 'accuracy'
        ? Math.max(...vals.filter(v => v != null))
        : Math.min(...vals.filter(v => v != null && v > 0));

      return `<tr>
        <td class="text-sm" style="font-weight:500;">${mr.label}</td>
        ${metrics.map(m => {
          const v = m[mr.key];
          const isBest = v === best && v != null;
          return `<td class="mono text-sm" style="${isBest ? 'color:var(--accent-green);font-weight:600;' : ''}">${mr.fmt(v)}${isBest ? ' 🏆' : ''}</td>`;
        }).join('')}
      </tr>`;
    }).join('');
  }

  // ============================================================
  //  CUSTOMER ENDPOINT FUNCTIONS
  // ============================================================

  let _evalMode = 'demo'; // 'demo' | 'custom'

  function setEvalMode(mode) {
    _evalMode = mode;
    document.getElementById('mode-demo-btn').classList.toggle('active', mode === 'demo');
    document.getElementById('mode-custom-btn').classList.toggle('active', mode === 'custom');
    document.getElementById('eval-mode-demo').classList.toggle('hidden', mode !== 'demo');
    document.getElementById('eval-mode-custom').classList.toggle('hidden', mode !== 'custom');
    document.getElementById('eval-type-group').classList.toggle('hidden', mode !== 'demo');

    // Sync benchmark dropdown in custom mode
    if (mode === 'custom') {
      const src = document.getElementById('eval-benchmark-select');
      const dst = document.getElementById('custom-benchmark-select');
      if (dst) dst.innerHTML = src.innerHTML;
    }
  }

  function selectProvider(cardEl) {
    document.querySelectorAll('.provider-card').forEach(c => c.classList.remove('selected'));
    cardEl.classList.add('selected');
    updateEndpointPlaceholder();

    const provider = cardEl.dataset.provider;
    // Auto-set eval type to match provider
    const evalTypeEl = document.getElementById('custom-eval-type-select');
    if (provider === 'rag_webhook') evalTypeEl.value = 'rag';
    else if (provider === 'agent_webhook') evalTypeEl.value = 'agent';
    else evalTypeEl.value = 'llm';

    // Hide model ID for RAG/agent webhooks (they don't need it)
    const modelIdGroup = document.getElementById('custom-model-id-group');
    if (modelIdGroup) {
      modelIdGroup.style.display = (provider === 'rag_webhook' || provider === 'agent_webhook') ? 'none' : '';
    }
  }

  function updateEndpointPlaceholder() {
    const selected = document.querySelector('.provider-card.selected');
    const provider = selected ? selected.dataset.provider : 'openai_compatible';
    const hint = document.getElementById('endpoint-hint');
    const urlInput = document.getElementById('custom-endpoint-url');

    const hints = {
      openai_compatible: 'Base URL like <code>https://api.groq.com/v1</code> — <code>/chat/completions</code> is auto-appended',
      anthropic: 'Uses <code>https://api.anthropic.com/v1/messages</code> — leave blank unless using a proxy',
      custom_llm: 'Your endpoint URL — must accept <code>{"prompt": "..."}</code> and return <code>{"response": "..."}</code>',
      rag_webhook: 'Your RAG endpoint — accepts <code>{"query": "..."}</code>, returns <code>{"answer": "...", "contexts": [...]}</code>',
      agent_webhook: 'Your agent endpoint — accepts <code>{"task": "...", "tools": [...]}</code>, returns <code>{"answer": "...", "steps": [...]}</code>',
    };
    const placeholders = {
      openai_compatible: 'https://api.groq.com/v1',
      anthropic: 'https://api.anthropic.com (or leave blank)',
      custom_llm: 'https://your-api.com/v1/generate',
      rag_webhook: 'https://your-rag-api.com/query',
      agent_webhook: 'https://your-agent-api.com/run',
    };

    if (hint) hint.innerHTML = hints[provider] || '';
    if (urlInput) urlInput.placeholder = placeholders[provider] || '';
  }

  async function testEndpointConnection() {
    const url = document.getElementById('custom-endpoint-url').value.trim();
    const resultEl = document.getElementById('endpoint-test-result');
    const btn = document.getElementById('test-endpoint-btn');

    if (!url) {
      resultEl.textContent = '⚠️ Enter an endpoint URL first';
      resultEl.className = 'text-sm test-pending';
      return;
    }

    btn.disabled = true;
    resultEl.textContent = '⏳ Testing…';
    resultEl.className = 'text-sm test-pending';

    // Try calling /health on the base domain, then fall back to a simple HEAD
    try {
      const baseUrl = new URL(url);
      const healthUrl = `${baseUrl.protocol}//${baseUrl.host}/health`;
      const r = await fetch(healthUrl, { method: 'GET', signal: AbortSignal.timeout(5000) });
      if (r.ok) {
        resultEl.textContent = `✅ Connected — ${baseUrl.host} responded ${r.status}`;
        resultEl.className = 'text-sm test-ok';
      } else {
        resultEl.textContent = `⚠️ Host reachable (HTTP ${r.status}) — endpoint may still work`;
        resultEl.className = 'text-sm test-pending';
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        resultEl.textContent = '❌ Connection timed out (>5s)';
      } else {
        resultEl.textContent = '⚠️ Could not reach host — check URL and CORS. Evaluation may still work server-side.';
      }
      resultEl.className = 'text-sm test-err';
    } finally {
      btn.disabled = false;
    }
  }
  // ─── Settings Modal Functions ──────────────────────────────
  function openSettingsModal() {
    const modal = document.getElementById('settings-modal');
    if (!modal) return;
    document.getElementById('settings-openai-key').value = localStorage.getItem('eval_openai_api_key') || '';
    document.getElementById('settings-anthropic-key').value = localStorage.getItem('eval_anthropic_api_key') || '';
    document.getElementById('settings-custom-key').value = localStorage.getItem('eval_custom_api_key') || '';
    document.getElementById('settings-custom-base-url').value = localStorage.getItem('eval_custom_base_url') || '';
    modal.classList.remove('hidden');
  }

  function closeSettingsModal() {
    const modal = document.getElementById('settings-modal');
    if (modal) modal.classList.add('hidden');
  }

  function saveSettings() {
    const openaiKey = document.getElementById('settings-openai-key').value.trim();
    const anthropicKey = document.getElementById('settings-anthropic-key').value.trim();
    const customKey = document.getElementById('settings-custom-key').value.trim();
    const customBaseUrl = document.getElementById('settings-custom-base-url').value.trim();
    
    localStorage.setItem('eval_openai_api_key', openaiKey);
    localStorage.setItem('eval_anthropic_api_key', anthropicKey);
    localStorage.setItem('eval_custom_api_key', customKey);
    localStorage.setItem('eval_custom_base_url', customBaseUrl);
    
    toast('success', 'Settings Saved', 'API keys and base URL updated successfully.');
    closeSettingsModal();
  }

  // ─── Custom Benchmark Builder Functions ────────────────────
  let taskCounter = 0;

  function onBenchmarkChange(value) {
    const builder = document.getElementById('custom-benchmark-builder');
    if (!builder) return;
    
    if (value === 'custom') {
      builder.classList.remove('hidden');
      const container = document.getElementById('custom-tasks-list');
      if (container && container.children.length === 0) {
        addCustomTask();
      }
    } else {
      builder.classList.add('hidden');
    }
    
    const evalSelect = document.getElementById('eval-benchmark-select');
    const customSelect = document.getElementById('custom-benchmark-select');
    if (evalSelect && evalSelect.value !== value) evalSelect.value = value;
    if (customSelect && customSelect.value !== value) customSelect.value = value;
  }

  function addCustomTask() {
    const container = document.getElementById('custom-tasks-list');
    if (!container) return;
    
    taskCounter++;
    const taskId = `task_${taskCounter}`;
    
    const card = document.createElement('div');
    card.className = 'custom-task-card';
    card.id = `custom-task-card-${taskId}`;
    card.innerHTML = `
      <button class="remove-task-btn" onclick="App.removeCustomTask('${taskId}')" type="button" title="Remove task">✕</button>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div class="form-group" style="margin: 0;">
          <label class="form-label" style="font-size: 0.75rem;">Task ID</label>
          <input class="form-input custom-task-id" type="text" value="${taskId}" placeholder="e.g. task_1" />
        </div>
        <div class="form-group" style="margin: 0;">
          <label class="form-label" style="font-size: 0.75rem;">Scoring Method</label>
          <select class="form-select custom-task-scoring" style="padding: 0.55rem 0.75rem; font-size: 0.85rem;">
            <option value="exact_match">Exact Match</option>
            <option value="contains">Contains</option>
            <option value="fuzzy_match">Fuzzy Match</option>
            <option value="llm_judge">LLM Judge</option>
            <option value="regex">Regex Match</option>
          </select>
        </div>
        <div class="form-group" style="grid-column: 1 / -1; margin: 0;">
          <label class="form-label" style="font-size: 0.75rem;">Prompt / Input <span style="color:var(--accent-red);">*</span></label>
          <textarea class="form-input custom-task-prompt" rows="2" placeholder="Enter prompt here..." style="font-family: inherit; resize: vertical; min-height: 48px;"></textarea>
        </div>
        <div class="form-group" style="grid-column: 1 / -1; margin: 0;">
          <label class="form-label" style="font-size: 0.75rem;">Expected Output</label>
          <textarea class="form-input custom-task-expected" rows="2" placeholder="Enter expected response (optional)..." style="font-family: inherit; resize: vertical; min-height: 48px;"></textarea>
        </div>
      </div>
    `;
    container.appendChild(card);
    updateCustomTaskCount();
  }

  function removeCustomTask(taskId) {
    const card = document.getElementById(`custom-task-card-${taskId}`);
    if (card) {
      card.remove();
      updateCustomTaskCount();
    }
  }

  function updateCustomTaskCount() {
    const container = document.getElementById('custom-tasks-list');
    const badge = document.getElementById('custom-task-count');
    if (container && badge) {
      badge.textContent = `${container.children.length} Tasks`;
    }
  }

  function getCustomTasks() {
    const container = document.getElementById('custom-tasks-list');
    if (!container) return [];
    
    const tasks = [];
    const cards = container.getElementsByClassName('custom-task-card');
    for (const card of cards) {
      const id = card.querySelector('.custom-task-id').value.trim();
      const prompt = card.querySelector('.custom-task-prompt').value.trim();
      const expected = card.querySelector('.custom-task-expected').value.trim();
      const scoring = card.querySelector('.custom-task-scoring').value;
      
      if (prompt) {
        tasks.push({
          id: id || `task_${tasks.length + 1}`,
          prompt: prompt,
          expected_output: expected,
          scoring: scoring,
          type: "llm_task"
        });
      }
    }
    return tasks;
  }
  // ============================================================
  //  REFRESH & INIT  (extended to support custom mode)
  // ============================================================

  async function refreshAll() {
    toast('info', 'Refreshing data…');
    await loadRuns();
    await loadModelsAndBenchmarks();
    populateResultsDropdown();
    populateCompareChips();
    toast('success', 'Data refreshed');
  }

  // Override startEvaluation to handle both demo and custom modes
  const _startEvaluationDemo = startEvaluation;

  async function startEvaluationAll() {
    if (_evalMode === 'demo') {
      await startEvaluation();
      return;
    }

    // --- CUSTOM ENDPOINT MODE ---
    const endpointUrl = document.getElementById('custom-endpoint-url').value.trim();
    const apiKey = document.getElementById('custom-api-key').value.trim();
    const modelId = document.getElementById('custom-model-id').value.trim();
    const displayName = document.getElementById('custom-display-name').value.trim();
    const benchmarkName = document.getElementById('custom-benchmark-select').value;
    const evalType = document.getElementById('custom-eval-type-select').value;
    const selectedProvider = document.querySelector('.provider-card.selected');
    const providerType = selectedProvider ? selectedProvider.dataset.provider : 'openai_compatible';

    if (!endpointUrl) {
      toast('warning', 'Missing endpoint URL', 'Please enter your API endpoint URL.');
      return;
    }
    if (!benchmarkName) {
      toast('warning', 'Select a benchmark', 'Please choose a benchmark to run.');
      return;
    }

    const modelName = displayName || modelId || 'custom-model';
    const btn = document.getElementById('run-eval-btn');
    const progress = document.getElementById('eval-progress');
    const statusText = document.getElementById('eval-status-text');

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Starting…';

    try {
      const payload = {
        model_name: modelName,
        benchmark_name: benchmarkName,
        eval_type: evalType,
        endpoint_url: endpointUrl,
        api_key: apiKey,
        provider_type: providerType,
        model_id: modelId,
        display_name: displayName || modelId || modelName,
      };

      if (benchmarkName === 'custom') {
        const customTasks = getCustomTasks();
        if (customTasks.length === 0) {
          toast('warning', 'Empty Custom Benchmark', 'Please add at least one task with a prompt.');
          btn.disabled = false;
          btn.innerHTML = '⚡ Run Evaluation';
          return;
        }
        payload.custom_tasks = customTasks;
      }

      const result = await api('/api/eval/run', {
        method: 'POST',
        body: JSON.stringify(payload),
      });

      toast('success', 'Evaluation started! 🚀', `Evaluating your ${providerType} endpoint — Run ID: ${shortId(result.id)}`);
      statusText.textContent = `Run ${shortId(result.id)} in progress…`;
      progress.classList.remove('hidden');
      pollEvalProgress(result.id);
    } catch (err) {
      toast('error', 'Failed to start evaluation', err.message);
      btn.disabled = false;
      btn.innerHTML = '⚡ Run Evaluation';
    }
  }

  function init() {
    configureCharts();
    initTabs();
    loadRuns();
    loadModelsAndBenchmarks();
    updateEndpointPlaceholder(); // Set initial hints

    // Wire up the run button to the unified handler
    const runBtn = document.getElementById('run-eval-btn');
    if (runBtn) runBtn.onclick = startEvaluationAll;

    // Wire up benchmark change listeners to show/hide benchmark builder
    const evalSelect = document.getElementById('eval-benchmark-select');
    if (evalSelect) evalSelect.onchange = (e) => onBenchmarkChange(e.target.value);
    const customSelect = document.getElementById('custom-benchmark-select');
    if (customSelect) customSelect.onchange = (e) => onBenchmarkChange(e.target.value);
  }

  // Boot
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Public API
  return {
    switchTab,
    switchInnerTab,
    startEvaluation: startEvaluationAll,
    loadRunDetail,
    viewRun,
    selectModelCard,
    selectBenchmarkCard,
    toggleExpand,
    filterTasks,
    exportRun,
    toggleCompareChip,
    runComparison,
    refreshAll,
    // Settings modal
    openSettingsModal,
    closeSettingsModal,
    saveSettings,
    // Custom benchmark builder
    onBenchmarkChange,
    addCustomTask,
    removeCustomTask,
  };
})();
