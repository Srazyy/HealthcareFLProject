/**
 * FL Healthcare Benchmark Interactive Comparison Dashboard
 * Handles reactive state, data parsing, Chart.js visualizations, and side-by-side comparison.
 */

(function () {
  'use strict';

  // ── Global State ─────────────────────────────────────────────────────────────
  let rawData = [];
  let currentDataset = [];
  let versusRadarChart = null;
  let convergenceLineChart = null;
  let tradeoffBarChart = null;

  let state = {
    metric: 'f1',
    roundView: 'final', // 'final' | 'all'
    alphaFilter: 'all',  // 'all' | 0.1 | 1.0 | 100.0
    epsFilter: 'all',    // 'all' | 'dp' | 'none'
    configA: null,
    configB: null,
    sortCol: 'f1',
    sortAsc: false,
    searchQuery: '',
  };

  const METRIC_LABELS = {
    f1: 'F1-Score',
    mcc: 'Matthews Corr (MCC)',
    auc_roc: 'AUC-ROC',
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    loss: 'Aggregated Loss',
  };

  // ── Initialization ───────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadInitialData();
  });

  function initEventListeners() {
    // Primary Metric Dropdown
    const metricSelect = document.getElementById('metric-select');
    if (metricSelect) {
      metricSelect.addEventListener('change', (e) => {
        state.metric = e.target.value;
        renderAll();
      });
    }

    // Round View Dropdown
    const roundSelect = document.getElementById('round-view-select');
    if (roundSelect) {
      roundSelect.addEventListener('change', (e) => {
        state.roundView = e.target.value;
        renderAll();
      });
    }

    // Alpha Filter Pills
    document.querySelectorAll('#alpha-pill-group .pill-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#alpha-pill-group .pill-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state.alphaFilter = btn.dataset.alpha;
        renderAll();
      });
    });

    // Privacy Filter Pills
    document.querySelectorAll('#eps-pill-group .pill-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#eps-pill-group .pill-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state.epsFilter = btn.dataset.eps;
        renderAll();
      });
    });

    // Config A & B Selectors
    const selA = document.getElementById('select-config-a');
    const selB = document.getElementById('select-config-b');
    if (selA) selA.addEventListener('change', (e) => { state.configA = e.target.value; updateComparator(); });
    if (selB) selB.addEventListener('change', (e) => { state.configB = e.target.value; updateComparator(); });

    // Table Search
    const searchInput = document.getElementById('table-search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        state.searchQuery = e.target.value.toLowerCase().trim();
        renderTable();
      });
    }

    // Table Column Sorting
    document.querySelectorAll('.data-table th.sortable').forEach((th) => {
      th.addEventListener('click', () => {
        const col = th.dataset.col;
        if (state.sortCol === col) {
          state.sortAsc = !state.sortAsc;
        } else {
          state.sortCol = col;
          state.sortAsc = false;
        }
        document.querySelectorAll('.data-table th').forEach((h) => {
          h.classList.remove('sorted-asc', 'sorted-desc');
        });
        th.classList.add(state.sortAsc ? 'sorted-asc' : 'sorted-desc');
        renderTable();
      });
    });

    // File Upload
    const fileInput = document.getElementById('file-upload-input');
    if (fileInput) {
      fileInput.addEventListener('change', handleFileUpload);
    }

    // Reload Default Data
    const reloadBtn = document.getElementById('btn-reload-data');
    if (reloadBtn) {
      reloadBtn.addEventListener('click', () => {
        loadInitialData(true);
      });
    }

    // Export CSV
    const exportBtn = document.getElementById('btn-export-csv');
    if (exportBtn) {
      exportBtn.addEventListener('click', exportTableToCSV);
    }

    // Modal Guide
    const sweepGuideBtn = document.getElementById('btn-sweep-guide');
    const modalOverlay = document.getElementById('sweep-modal-overlay');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalOkBtn = document.getElementById('modal-ok-btn');

    if (sweepGuideBtn && modalOverlay) {
      sweepGuideBtn.addEventListener('click', () => modalOverlay.classList.add('active'));
    }
    const closeModal = () => modalOverlay && modalOverlay.classList.remove('active');
    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeModal);
    if (modalOkBtn) modalOkBtn.addEventListener('click', closeModal);
    if (modalOverlay) {
      modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
      });
    }
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal();
    });
  }

  // ── Data Loading & Ingestion ─────────────────────────────────────────────────
  async function loadInitialData(forceReload = false) {
    let loaded = false;

    // First attempt to fetch the live sweep_results.csv from results/ directory if served via HTTP
    try {
      const res = await fetch('../results/sweep_results.csv', { cache: 'no-cache' });
      if (res.ok) {
        const text = await res.text();
        if (text && text.length > 50) {
          rawData = parseCSV(text);
          loaded = true;
          setLiveStatus('Loaded from results/sweep_results.csv');
        }
      }
    } catch (_) {
      // CORS or file protocol fallback
    }

    // Fallback to window.INITIAL_SWEEP_DATA (embedded data from data.js)
    if (!loaded) {
      if (window.INITIAL_SWEEP_DATA && Array.isArray(window.INITIAL_SWEEP_DATA)) {
        rawData = sanitizeRecords(window.INITIAL_SWEEP_DATA);
        loaded = true;
        setLiveStatus('Benchmark Loaded');
      }
    }

    if (!loaded) {
      alert('Could not find benchmark data. Please upload a results CSV.');
      return;
    }

    // Initialize Comparator Dropdowns
    populateConfigSelectors();
    renderAll();
  }

  function setLiveStatus(text) {
    const badge = document.getElementById('live-status-badge');
    if (badge) badge.textContent = text;
  }

  function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const content = event.target.result;
        if (file.name.endsWith('.json')) {
          const parsed = JSON.parse(content);
          rawData = sanitizeRecords(Array.isArray(parsed) ? parsed : (parsed.rows || []));
        } else {
          rawData = parseCSV(content);
        }
        setLiveStatus(`File: ${file.name}`);
        populateConfigSelectors();
        renderAll();
      } catch (err) {
        alert('Failed to parse uploaded file: ' + err.message);
      }
    };
    reader.readAsText(file);
  }

  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) return [];

    const headers = lines[0].split(',').map((h) => h.trim());
    const records = [];

    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(',').map((v) => v.trim());
      if (vals.length !== headers.length) continue;
      const row = {};
      headers.forEach((h, idx) => {
        row[h] = vals[idx];
      });
      records.push(row);
    }
    return sanitizeRecords(records);
  }

  function sanitizeRecords(records) {
    return records.map((r) => {
      const rank = parseInt(r.rank, 10);
      const alpha = parseFloat(r.alpha);
      const round = parseInt(r.round, 10);
      let eps = r.epsilon;
      if (eps === null || eps === undefined || String(eps).toLowerCase() === 'none' || String(eps).toLowerCase() === 'null') {
        eps = 'none';
      } else {
        eps = String(parseFloat(eps).toFixed(1));
      }

      return {
        rank,
        epsilon: eps,
        alpha,
        round,
        accuracy: parseFloat(r.accuracy) || 0,
        precision: parseFloat(r.precision) || 0,
        recall: parseFloat(r.recall) || 0,
        f1: parseFloat(r.f1) || 0,
        mcc: parseFloat(r.mcc) || 0,
        auc_roc: parseFloat(r.auc_roc) || 0,
        loss: parseFloat(r.loss) || 0,
      };
    });
  }

  // ── Filtering & Processing ───────────────────────────────────────────────────
  function getFilteredDataset() {
    let filtered = [...rawData];

    // Round View
    if (state.roundView === 'final') {
      const maxRound = Math.max(...rawData.map((d) => d.round), 1);
      filtered = filtered.filter((d) => d.round === maxRound);
    }

    // Alpha Filter
    if (state.alphaFilter !== 'all') {
      const targetAlpha = parseFloat(state.alphaFilter);
      filtered = filtered.filter((d) => Math.abs(d.alpha - targetAlpha) < 0.001);
    }

    // Privacy Filter
    if (state.epsFilter === 'dp') {
      filtered = filtered.filter((d) => d.epsilon !== 'none');
    } else if (state.epsFilter === 'none') {
      filtered = filtered.filter((d) => d.epsilon === 'none');
    }

    return filtered;
  }

  function getConfigKey(d) {
    const epsLabel = d.epsilon === 'none' ? 'No DP' : `ε=${d.epsilon}`;
    return `r=${d.rank} | ${epsLabel} | α=${d.alpha}`;
  }

  function populateConfigSelectors() {
    const selA = document.getElementById('select-config-a');
    const selB = document.getElementById('select-config-b');
    if (!selA || !selB) return;

    // Get unique configs from final round
    const maxRound = Math.max(...rawData.map((d) => d.round), 1);
    const finalRuns = rawData.filter((d) => d.round === maxRound);

    const keys = Array.from(new Set(finalRuns.map(getConfigKey))).sort();

    selA.innerHTML = '';
    selB.innerHTML = '';

    keys.forEach((k) => {
      const optA = document.createElement('option');
      optA.value = k;
      optA.textContent = k;
      selA.appendChild(optA);

      const optB = document.createElement('option');
      optB.value = k;
      optB.textContent = k;
      selB.appendChild(optB);
    });

    if (keys.length > 0) {
      state.configA = keys[0];
      state.configB = keys[keys.length > 1 ? 1 : 0];
      selA.value = state.configA;
      selB.value = state.configB;
    }
  }

  // ── Render Master ────────────────────────────────────────────────────────────
  function renderAll() {
    currentDataset = getFilteredDataset();
    updateKPICards();
    updateComparator();
    renderHeatmaps();
    renderConvergenceChart();
    renderTradeoffBarChart();
    renderTable();
  }

  // ── KPI Overview Cards ───────────────────────────────────────────────────────
  function updateKPICards() {
    const maxRound = Math.max(...rawData.map((d) => d.round), 1);
    const finalData = rawData.filter((d) => d.round === maxRound);

    if (finalData.length === 0) return;

    // 1. Top Performing Config
    let best = finalData[0];
    finalData.forEach((d) => {
      if (d[state.metric] > best[state.metric]) best = d;
    });

    const bestValEl = document.getElementById('kpi-best-value');
    const bestSubEl = document.getElementById('kpi-best-subtext');
    if (bestValEl && bestSubEl) {
      bestValEl.textContent = best[state.metric].toFixed(4);
      const epsTxt = best.epsilon === 'none' ? 'No DP' : `ε=${best.epsilon}`;
      bestSubEl.innerHTML = `<span class="kpi-tag">r=${best.rank}</span><span class="kpi-tag">${epsTxt}</span><span class="kpi-tag">α=${best.alpha}</span>`;
    }

    // 2. Privacy Budget Penalty: Mean F1 for DP vs No DP
    const dpRows = finalData.filter((d) => d.epsilon !== 'none');
    const noDpRows = finalData.filter((d) => d.epsilon === 'none');
    const avgDp = dpRows.length ? dpRows.reduce((a, b) => a + b[state.metric], 0) / dpRows.length : 0;
    const avgNoDp = noDpRows.length ? noDpRows.reduce((a, b) => a + b[state.metric], 0) / noDpRows.length : 0;
    const penalty = avgNoDp > 0 ? ((avgDp - avgNoDp) / avgNoDp) * 100 : 0;

    const dpValEl = document.getElementById('kpi-dp-value');
    const dpSubEl = document.getElementById('kpi-dp-subtext');
    if (dpValEl && dpSubEl) {
      dpValEl.textContent = `${penalty > 0 ? '+' : ''}${penalty.toFixed(1)}%`;
      dpValEl.style.color = penalty >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';
      dpSubEl.textContent = `Mean ${METRIC_LABELS[state.metric]}: DP ${avgDp.toFixed(3)} vs No-DP ${avgNoDp.toFixed(3)}`;
    }

    // 3. Heterogeneity Drop: α=0.1 vs α=100.0
    const alpha01Rows = finalData.filter((d) => Math.abs(d.alpha - 0.1) < 0.01);
    const alpha100Rows = finalData.filter((d) => Math.abs(d.alpha - 100.0) < 0.01);
    const avg01 = alpha01Rows.length ? alpha01Rows.reduce((a, b) => a + b[state.metric], 0) / alpha01Rows.length : 0;
    const avg100 = alpha100Rows.length ? alpha100Rows.reduce((a, b) => a + b[state.metric], 0) / alpha100Rows.length : 0;
    const skewDrop = avg100 > 0 ? ((avg01 - avg100) / avg100) * 100 : 0;

    const skewValEl = document.getElementById('kpi-skew-value');
    const skewSubEl = document.getElementById('kpi-skew-subtext');
    if (skewValEl && skewSubEl) {
      skewValEl.textContent = `${skewDrop > 0 ? '+' : ''}${skewDrop.toFixed(1)}%`;
      skewValEl.style.color = skewDrop >= 0 ? 'var(--accent-emerald)' : 'var(--accent-amber)';
      skewSubEl.textContent = `Skewed (α=0.1): ${avg01.toFixed(3)} | IID (α=100): ${avg100.toFixed(3)}`;
    }

    // 4. LoRA Rank Gain: r=8 vs r=4
    const r4Rows = finalData.filter((d) => d.rank === 4);
    const r8Rows = finalData.filter((d) => d.rank === 8);
    const avgR4 = r4Rows.length ? r4Rows.reduce((a, b) => a + b[state.metric], 0) / r4Rows.length : 0;
    const avgR8 = r8Rows.length ? r8Rows.reduce((a, b) => a + b[state.metric], 0) / r8Rows.length : 0;
    const loraGain = avgR4 > 0 ? ((avgR8 - avgR4) / avgR4) * 100 : 0;

    const loraValEl = document.getElementById('kpi-lora-value');
    const loraSubEl = document.getElementById('kpi-lora-subtext');
    if (loraValEl && loraSubEl) {
      loraValEl.textContent = `${loraGain >= 0 ? '+' : ''}${loraGain.toFixed(1)}%`;
      loraValEl.style.color = loraGain >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';
      loraSubEl.textContent = `Rank r=8 (${avgR8.toFixed(3)}) vs Rank r=4 (${avgR4.toFixed(3)})`;
    }
  }

  // ── Comparator ("Versus Tool") ───────────────────────────────────────────────
  function updateComparator() {
    const maxRound = Math.max(...rawData.map((d) => d.round), 1);
    const finalData = rawData.filter((d) => d.round === maxRound);

    const recordA = finalData.find((d) => getConfigKey(d) === state.configA);
    const recordB = finalData.find((d) => getConfigKey(d) === state.configB);

    if (!recordA || !recordB) return;

    const listA = document.getElementById('metrics-list-a');
    const listB = document.getElementById('metrics-list-b');
    if (!listA || !listB) return;

    listA.innerHTML = '';
    listB.innerHTML = '';

    const metricsToCompare = ['f1', 'mcc', 'auc_roc', 'accuracy', 'precision', 'recall', 'loss'];

    metricsToCompare.forEach((m) => {
      const valA = recordA[m];
      const valB = recordB[m];
      const diff = valB - valA;
      const isLoss = m === 'loss';
      // For loss, lower is better. For others, higher is better.
      const isPositive = isLoss ? diff < 0 : diff > 0;
      const isZero = Math.abs(diff) < 0.0001;

      // Row A
      const rowA = document.createElement('div');
      rowA.className = 'metric-row';
      rowA.innerHTML = `
        <span class="metric-name">${METRIC_LABELS[m]}</span>
        <div class="metric-vals">
          <span class="metric-val-num">${valA.toFixed(4)}</span>
        </div>
      `;
      listA.appendChild(rowA);

      // Row B with Delta
      const deltaClass = isZero ? 'delta-neutral' : isPositive ? 'delta-positive' : 'delta-negative';
      const arrow = isZero ? '—' : isPositive ? '▲' : '▼';
      const sign = diff > 0 ? '+' : '';
      const rowB = document.createElement('div');
      rowB.className = 'metric-row';
      rowB.innerHTML = `
        <span class="metric-name">${METRIC_LABELS[m]}</span>
        <div class="metric-vals">
          <span class="metric-val-num">${valB.toFixed(4)}</span>
          <span class="delta-badge ${deltaClass}">
            ${arrow} ${sign}${diff.toFixed(4)}
          </span>
        </div>
      `;
      listB.appendChild(rowB);
    });

    renderRadarChart(recordA, recordB);
  }

  function renderRadarChart(recA, recB) {
    const ctx = document.getElementById('versus-radar-chart');
    if (!ctx) return;

    const radarMetrics = ['f1', 'mcc', 'auc_roc', 'accuracy', 'precision', 'recall'];
    const labels = radarMetrics.map((m) => METRIC_LABELS[m]);
    const dataA = radarMetrics.map((m) => recA[m]);
    const dataB = radarMetrics.map((m) => recB[m]);

    if (versusRadarChart) versusRadarChart.destroy();

    versusRadarChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: labels,
        datasets: [
          {
            label: `Config A (${state.configA})`,
            data: dataA,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.25)',
            pointBackgroundColor: '#06b6d4',
            borderWidth: 2,
          },
          {
            label: `Config B (${state.configB})`,
            data: dataB,
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.25)',
            pointBackgroundColor: '#8b5cf6',
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            pointLabels: {
              color: '#9ca3af',
              font: { family: 'Inter', size: 11, weight: 600 },
            },
            ticks: {
              backdropColor: 'transparent',
              color: '#6b7280',
              font: { family: 'JetBrains Mono', size: 9 },
            },
            min: 0,
            max: 1.0,
          },
        },
        plugins: {
          legend: {
            labels: { color: '#f3f4f6', font: { family: 'Inter', size: 12 } },
          },
        },
      },
    });
  }

  // ── 2D Heatmaps ──────────────────────────────────────────────────────────────
  function renderHeatmaps() {
    const container = document.getElementById('heatmaps-container');
    const metricIndicator = document.getElementById('heatmap-metric-indicator');
    if (!container) return;

    if (metricIndicator) {
      metricIndicator.textContent = `Metric: ${METRIC_LABELS[state.metric]}`;
    }

    container.innerHTML = '';

    const maxRound = Math.max(...rawData.map((d) => d.round), 1);
    const finalData = rawData.filter((d) => d.round === maxRound);

    const alphas = [0.1, 1.0, 100.0];
    const alphaLabels = {
      0.1: 'Extreme Skew (Hospital Non-IID)',
      1.0: 'Moderate Heterogeneity',
      100.0: 'Homogeneous (~IID Baseline)',
    };

    alphas.forEach((alpha) => {
      const alphaSubset = finalData.filter((d) => Math.abs(d.alpha - alpha) < 0.01);
      const card = document.createElement('div');
      card.className = 'heatmap-card';

      const ranks = Array.from(new Set(alphaSubset.map((d) => d.rank))).sort((a, b) => a - b);
      const epsilons = ['none', '8.0']; // Can extend dynamically

      let tableHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <h4 style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">α = ${alpha}</h4>
          <span style="font-size: 0.75rem; color: var(--accent-amber);">${alphaLabels[alpha]}</span>
        </div>
        <table class="heatmap-table">
          <thead>
            <tr>
              <th class="axis-col">Privacy (ε)</th>
              ${ranks.map((r) => `<th>r = ${r}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
      `;

      epsilons.forEach((eps) => {
        const epsLabel = eps === 'none' ? 'No DP' : `ε = ${eps}`;
        tableHTML += `<tr><td style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);">${epsLabel}</td>`;

        ranks.forEach((r) => {
          const match = alphaSubset.find((d) => d.rank === r && d.epsilon === eps);
          if (match) {
            const val = match[state.metric];
            const color = getHeatmapCellColor(val, state.metric);
            tableHTML += `
              <td class="heatmap-cell" style="background: ${color};" title="${getConfigKey(match)}: ${val.toFixed(4)}" onclick="window.selectHeatmapConfig('${getConfigKey(match)}')">
                ${val.toFixed(3)}
              </td>
            `;
          } else {
            tableHTML += `<td class="heatmap-cell" style="background: rgba(255,255,255,0.03); color: #6b7280;">—</td>`;
          }
        });

        tableHTML += `</tr>`;
      });

      tableHTML += `</tbody></table>`;
      card.innerHTML = tableHTML;
      container.appendChild(card);
    });
  }

  function getHeatmapCellColor(val, metric) {
    // Loss: 0.0 is best (green), 1.0 is worst (dark red)
    if (metric === 'loss') {
      const normalized = Math.max(0, Math.min(1, val));
      const hue = (1 - normalized) * 120; // 120=green, 0=red
      return `hsla(${hue}, 70%, 35%, 0.75)`;
    }
    // Performance metrics: 1.0 is best
    const normalized = Math.max(0, Math.min(1, val));
    // Gradient from deep blue/indigo (230) to teal (170) to emerald (140)
    const hue = 220 - normalized * 80;
    const lightness = 25 + normalized * 25;
    return `hsla(${hue}, 80%, ${lightness}%, 0.85)`;
  }

  window.selectHeatmapConfig = function (cfgKey) {
    const selB = document.getElementById('select-config-b');
    if (selB) {
      selB.value = cfgKey;
      state.configB = cfgKey;
      updateComparator();
      // Scroll smoothly to comparator section
      document.getElementById('comparator-section')?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // ── Convergence Trajectory Chart ─────────────────────────────────────────────
  function renderConvergenceChart() {
    const ctx = document.getElementById('convergence-line-chart');
    if (!ctx) return;

    const rounds = Array.from(new Set(rawData.map((d) => d.round))).sort((a, b) => a - b);
    const groups = {};

    currentDataset.forEach((d) => {
      const key = getConfigKey(d);
      if (!groups[key]) groups[key] = [];
      groups[key].push(d);
    });

    // Palette generator
    const colorPalette = [
      '#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6',
      '#14b8a6', '#f43f5e', '#3b82f6', '#84cc16', '#a855f7', '#0ea5e9'
    ];

    const datasets = Object.keys(groups).map((key, idx) => {
      const records = groups[key].sort((a, b) => a.round - b.round);
      const dataMap = {};
      records.forEach((r) => { dataMap[r.round] = r[state.metric]; });
      const seriesData = rounds.map((rd) => (dataMap[rd] !== undefined ? dataMap[rd] : null));

      const color = colorPalette[idx % colorPalette.length];
      return {
        label: key,
        data: seriesData,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 4,
        pointHoverRadius: 7,
      };
    });

    if (convergenceLineChart) convergenceLineChart.destroy();

    convergenceLineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: rounds.map((r) => `Round ${r}`),
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af', font: { family: 'Inter', size: 11 } },
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono', size: 10 } },
            title: {
              display: true,
              text: METRIC_LABELS[state.metric],
              color: '#9ca3af',
              font: { family: 'Inter', size: 11, weight: 600 },
            },
          },
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 12,
              color: '#9ca3af',
              font: { family: 'JetBrains Mono', size: 9 },
            },
          },
          tooltip: {
            callbacks: {
              label: (item) => `${item.dataset.label}: ${item.raw !== null ? item.raw.toFixed(4) : 'N/A'}`,
            },
          },
        },
      },
    });
  }

  // ── Tradeoff Bar Chart ───────────────────────────────────────────────────────
  function renderTradeoffBarChart() {
    const ctx = document.getElementById('tradeoff-bar-chart');
    if (!ctx) return;

    const maxRound = Math.max(...rawData.map((d) => d.round), 1);
    const finalData = rawData.filter((d) => d.round === maxRound);

    const alphas = [0.1, 1.0, 100.0];
    const labels = alphas.map((a) => `α = ${a}`);

    // Series: r=4 (DP), r=4 (No DP), r=8 (DP), r=8 (No DP)
    const seriesConfigs = [
      { rank: 4, eps: '8.0', label: 'r=4 | DP (ε=8)', color: '#06b6d4' },
      { rank: 4, eps: 'none', label: 'r=4 | No DP', color: '#10b981' },
      { rank: 8, eps: '8.0', label: 'r=8 | DP (ε=8)', color: '#8b5cf6' },
      { rank: 8, eps: 'none', label: 'r=8 | No DP', color: '#6366f1' },
    ];

    const datasets = seriesConfigs.map((cfg) => {
      const data = alphas.map((alpha) => {
        const item = finalData.find((d) => d.rank === cfg.rank && d.epsilon === cfg.eps && Math.abs(d.alpha - alpha) < 0.01);
        return item ? item[state.metric] : 0;
      });
      return {
        label: cfg.label,
        data: data,
        backgroundColor: cfg.color,
        borderRadius: 4,
      };
    });

    if (tradeoffBarChart) tradeoffBarChart.destroy();

    tradeoffBarChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#9ca3af', font: { family: 'Inter', size: 11, weight: 600 } },
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono', size: 10 } },
            title: {
              display: true,
              text: METRIC_LABELS[state.metric],
              color: '#9ca3af',
              font: { family: 'Inter', size: 11, weight: 600 },
            },
            min: 0,
            max: state.metric === 'loss' ? undefined : 1.0,
          },
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 12,
              color: '#9ca3af',
              font: { family: 'Inter', size: 10 },
            },
          },
        },
      },
    });
  }

  // ── Tabular Ledger & CSV Export ──────────────────────────────────────────────
  function renderTable() {
    const tbody = document.getElementById('table-body');
    if (!tbody) return;

    let data = [...currentDataset];

    // Search filter
    if (state.searchQuery) {
      data = data.filter((d) => {
        const str = `rank:${d.rank} eps:${d.epsilon} alpha:${d.alpha} round:${d.round}`.toLowerCase();
        return str.includes(state.searchQuery);
      });
    }

    // Sorting
    data.sort((a, b) => {
      let va = a[state.sortCol];
      let vb = b[state.sortCol];
      if (va < vb) return state.sortAsc ? -1 : 1;
      if (va > vb) return state.sortAsc ? 1 : -1;
      return 0;
    });

    tbody.innerHTML = '';

    if (data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="11" style="text-align: center; color: var(--text-muted); padding: 24px;">No matching benchmark records found.</td></tr>`;
      return;
    }

    data.forEach((r) => {
      const tr = document.createElement('tr');
      const epsBadgeClass = r.epsilon === 'none' ? 'badge-eps-none' : 'badge-eps-dp';
      const epsText = r.epsilon === 'none' ? 'No DP' : `ε = ${r.epsilon}`;

      tr.innerHTML = `
        <td><span class="badge badge-rank">r = ${r.rank}</span></td>
        <td><span class="badge ${epsBadgeClass}">${epsText}</span></td>
        <td><span class="badge badge-alpha">α = ${r.alpha}</span></td>
        <td style="font-family: 'JetBrains Mono';">${r.round}</td>
        <td style="font-family: 'JetBrains Mono'; font-weight: 700; color: #ffffff;">${r.f1.toFixed(4)}</td>
        <td style="font-family: 'JetBrains Mono';">${r.mcc.toFixed(4)}</td>
        <td style="font-family: 'JetBrains Mono';">${r.auc_roc.toFixed(4)}</td>
        <td style="font-family: 'JetBrains Mono';">${r.accuracy.toFixed(4)}</td>
        <td style="font-family: 'JetBrains Mono';">${r.precision.toFixed(4)}</td>
        <td style="font-family: 'JetBrains Mono';">${r.recall.toFixed(4)}</td>
        <td style="font-family: 'JetBrains Mono'; color: var(--accent-rose);">${r.loss.toFixed(4)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function exportTableToCSV() {
    if (currentDataset.length === 0) return;

    const headers = ['rank', 'epsilon', 'alpha', 'round', 'f1', 'mcc', 'auc_roc', 'accuracy', 'precision', 'recall', 'loss'];
    const rows = [headers.join(',')];

    currentDataset.forEach((d) => {
      const row = headers.map((h) => d[h]);
      rows.push(row.join(','));
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + encodeURIComponent(rows.join('\n'));
    const link = document.createElement('a');
    link.setAttribute('href', csvContent);
    link.setAttribute('download', `fl_healthcare_benchmark_${state.metric}_view.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

})();
