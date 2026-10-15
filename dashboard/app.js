/**
 * Healthcare FL Benchmark - Application Logic
 * Supports Multi-Page Views: Hospital Silos (index.html), Clinical Simulator (cases.html),
 * Parameter Tradeoffs (tradeoffs.html), and Model Comparator (comparator.html).
 */

(function () {
  'use strict';

  if (window.Chart) {
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.color = '#78716c';
  }

  // ── Global State ─────────────────────────────────────────────────────────────
  let rawData = [];
  let currentDataset = [];
  let versusRadarChart = null;
  let convergenceLineChart = null;
  let tradeoffBarChart = null;
  let siloRadarChart = null;
  let siloSensitivityChart = null;
  let activeCaseId = 'case-1';

  const ALPHA_STEPS = [0.1, 1.0, 100.0];
  const RANK_STEPS = [2, 4, 8, 16];
  const EPS_STEPS = ['none', '8.0', '3.0', '1.0'];

  let siloState = {
    alpha: '0.1',
    rank: '8',
    eps: 'none'
  };

  let compState = {
    rankA: 8,
    epsA: 'none',
    alphaA: 100.0,
    rankB: 4,
    epsB: '8.0',
    alphaB: 0.1,
  };

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
    initDiscreteSliders();
    loadInitialData();
  });

  function initDiscreteSliders() {
    // 1. Alpha Slider
    const sliderAlpha = document.getElementById('slider-alpha');
    const badgeAlpha = document.getElementById('badge-slider-alpha');
    if (sliderAlpha) {
      const updateAlpha = (idx) => {
        sliderAlpha.value = idx;
        siloState.alpha = String(ALPHA_STEPS[idx]);
        if (badgeAlpha) badgeAlpha.textContent = `α = ${siloState.alpha}`;
        document.querySelectorAll('[data-slider="alpha"]').forEach((el) => {
          el.classList.toggle('active', el.dataset.val === String(idx));
        });
        renderHospitalSiloModule();
        renderClinicalCaseTester();
      };
      sliderAlpha.addEventListener('input', (e) => updateAlpha(parseInt(e.target.value, 10)));
      document.querySelectorAll('[data-slider="alpha"]').forEach((el) => {
        el.addEventListener('click', () => updateAlpha(parseInt(el.dataset.val, 10)));
      });
    }

    // 2. Rank Slider
    const sliderRank = document.getElementById('slider-rank');
    const badgeRank = document.getElementById('badge-slider-rank');
    if (sliderRank) {
      const updateRank = (idx) => {
        sliderRank.value = idx;
        siloState.rank = String(RANK_STEPS[idx]);
        if (badgeRank) badgeRank.textContent = `r = ${siloState.rank}`;
        document.querySelectorAll('[data-slider="rank"]').forEach((el) => {
          el.classList.toggle('active', el.dataset.val === String(idx));
        });
        renderHospitalSiloModule();
        renderClinicalCaseTester();
      };
      sliderRank.addEventListener('input', (e) => updateRank(parseInt(e.target.value, 10)));
      document.querySelectorAll('[data-slider="rank"]').forEach((el) => {
        el.addEventListener('click', () => updateRank(parseInt(el.dataset.val, 10)));
      });
    }

    // 3. Epsilon Slider
    const sliderEps = document.getElementById('slider-eps');
    const badgeEps = document.getElementById('badge-slider-eps');
    if (sliderEps) {
      const updateEps = (idx) => {
        sliderEps.value = idx;
        siloState.eps = EPS_STEPS[idx];
        if (badgeEps) {
          badgeEps.textContent = siloState.eps === 'none' ? 'ε = ∞ (No DP)' : `ε = ${siloState.eps}`;
        }
        document.querySelectorAll('[data-slider="eps"]').forEach((el) => {
          el.classList.toggle('active', el.dataset.val === String(idx));
        });
        renderHospitalSiloModule();
        renderClinicalCaseTester();
      };
      sliderEps.addEventListener('input', (e) => updateEps(parseInt(e.target.value, 10)));
      document.querySelectorAll('[data-slider="eps"]').forEach((el) => {
        el.addEventListener('click', () => updateEps(parseInt(el.dataset.val, 10)));
      });
    }

    // ── Comparator Sliders for Model A ──
    const sliderCompRankA = document.getElementById('slider-comp-rank-a');
    const badgeCompRankA = document.getElementById('badge-comp-rank-a');
    const badgeCompSummaryA = document.getElementById('badge-comp-summary-a');
    if (sliderCompRankA) {
      const updateCompRankA = (idx) => {
        sliderCompRankA.value = idx;
        compState.rankA = RANK_STEPS[idx];
        if (badgeCompRankA) badgeCompRankA.textContent = `r = ${compState.rankA}`;
        if (badgeCompSummaryA) badgeCompSummaryA.textContent = `r=${compState.rankA} | ${compState.epsA === 'none' ? 'No DP' : 'ε=' + compState.epsA} | α=${compState.alphaA}`;
        document.querySelectorAll('[data-slider="comp-rank-a"]').forEach((el) => {
          el.classList.toggle('active', el.dataset.val === String(idx));
        });
        updateComparator();
      };
      sliderCompRankA.addEventListener('input', (e) => updateCompRankA(parseInt(e.target.value, 10)));
      document.querySelectorAll('[data-slider="comp-rank-a"]').forEach((el) => {
        el.addEventListener('click', () => updateCompRankA(parseInt(el.dataset.val, 10)));
      });
    }

    const sliderCompEpsA = document.getElementById('slider-comp-eps-a');
    const badgeCompEpsA = document.getElementById('badge-comp-eps-a');
    if (sliderCompEpsA) {
      const updateCompEpsA = (idx) => {
        sliderCompEpsA.value = idx;
        compState.epsA = EPS_STEPS[idx];
        if (badgeCompEpsA) badgeCompEpsA.textContent = compState.epsA === 'none' ? 'No DP' : `ε = ${compState.epsA}`;
        if (badgeCompSummaryA) badgeCompSummaryA.textContent = `r=${compState.rankA} | ${compState.epsA === 'none' ? 'No DP' : 'ε=' + compState.epsA} | α=${compState.alphaA}`;
        document.querySelectorAll('[data-slider="comp-eps-a"]').forEach((el) => {
          el.classList.toggle('active', el.dataset.val === String(idx));
        });
        updateComparator();
      };
      sliderCompEpsA.addEventListener('input', (e) => updateCompEpsA(parseInt(e.target.value, 10)));
      document.querySelectorAll('[data-slider="comp-eps-a"]').forEach((el) => {
        el.addEventListener('click', () => updateCompEpsA(parseInt(el.dataset.val, 10)));
      });
    }

    const sliderCompAlphaA = document.getElementById('slider-comp-alpha-a');
    const badgeCompAlphaA = document.getElementById('badge-comp-alpha-a');
    if (sliderCompAlphaA) {
      const updateCompAlphaA = (idx) => {
        sliderCompAlphaA.value = idx;
        compState.alphaA = ALPHA_STEPS[idx];
        if (badgeCompAlphaA) badgeCompAlphaA.textContent = `α = ${compState.alphaA}`;
        if (badgeCompSummaryA) badgeCompSummaryA.textContent = `r=${compState.rankA} | ${compState.epsA === 'none' ? 'No DP' : 'ε=' + compState.epsA} | α=${compState.alphaA}`;
        document.querySelectorAll('[data-slider="comp-alpha-a"]').forEach((el) => {
          el.classList.toggle('active', el.dataset.val === String(idx));
        });
        updateComparator();
      };
      sliderCompAlphaA.addEventListener('input', (e) => updateCompAlphaA(parseInt(e.target.value, 10)));
      document.querySelectorAll('[data-slider="comp-alpha-a"]').forEach((el) => {
        el.addEventListener('click', () => updateCompAlphaA(parseInt(el.dataset.val, 10)));
      });
    }

    // ── Comparator Sliders for Model B ──
    const sliderCompRankB = document.getElementById('slider-comp-rank-b');
    const badgeCompRankB = document.getElementById('badge-comp-rank-b');
    const badgeCompSummaryB = document.getElementById('badge-comp-summary-b');
    if (sliderCompRankB) {
      const updateCompRankB = (idx) => {
        sliderCompRankB.value = idx;
        compState.rankB = RANK_STEPS[idx];
        if (badgeCompRankB) badgeCompRankB.textContent = `r = ${compState.rankB}`;
        if (badgeCompSummaryB) badgeCompSummaryB.textContent = `r=${compState.rankB} | ${compState.epsB === 'none' ? 'No DP' : 'ε=' + compState.epsB} | α=${compState.alphaB}`;
        document.querySelectorAll('[data-slider="comp-rank-b"]').forEach((el) => {
          el.classList.toggle('active', el.dataset.val === String(idx));
        });
        updateComparator();
      };
      sliderCompRankB.addEventListener('input', (e) => updateCompRankB(parseInt(e.target.value, 10)));
      document.querySelectorAll('[data-slider="comp-rank-b"]').forEach((el) => {
        el.addEventListener('click', () => updateCompRankB(parseInt(el.dataset.val, 10)));
      });
    }

    const sliderCompEpsB = document.getElementById('slider-comp-eps-b');
    const badgeCompEpsB = document.getElementById('badge-comp-eps-b');
    if (sliderCompEpsB) {
      const updateCompEpsB = (idx) => {
        sliderCompEpsB.value = idx;
        compState.epsB = EPS_STEPS[idx];
        if (badgeCompEpsB) badgeCompEpsB.textContent = compState.epsB === 'none' ? 'No DP' : `ε = ${compState.epsB}`;
        if (badgeCompSummaryB) badgeCompSummaryB.textContent = `r=${compState.rankB} | ${compState.epsB === 'none' ? 'No DP' : 'ε=' + compState.epsB} | α=${compState.alphaB}`;
        document.querySelectorAll('[data-slider="comp-eps-b"]').forEach((el) => {
          el.classList.toggle('active', el.dataset.val === String(idx));
        });
        updateComparator();
      };
      sliderCompEpsB.addEventListener('input', (e) => updateCompEpsB(parseInt(e.target.value, 10)));
      document.querySelectorAll('[data-slider="comp-eps-b"]').forEach((el) => {
        el.addEventListener('click', () => updateCompEpsB(parseInt(el.dataset.val, 10)));
      });
    }

    const sliderCompAlphaB = document.getElementById('slider-comp-alpha-b');
    const badgeCompAlphaB = document.getElementById('badge-comp-alpha-b');
    if (sliderCompAlphaB) {
      const updateCompAlphaB = (idx) => {
        sliderCompAlphaB.value = idx;
        compState.alphaB = ALPHA_STEPS[idx];
        if (badgeCompAlphaB) badgeCompAlphaB.textContent = `α = ${compState.alphaB}`;
        if (badgeCompSummaryB) badgeCompSummaryB.textContent = `r=${compState.rankB} | ${compState.epsB === 'none' ? 'No DP' : 'ε=' + compState.epsB} | α=${compState.alphaB}`;
        document.querySelectorAll('[data-slider="comp-alpha-b"]').forEach((el) => {
          el.classList.toggle('active', el.dataset.val === String(idx));
        });
        updateComparator();
      };
      sliderCompAlphaB.addEventListener('input', (e) => updateCompAlphaB(parseInt(e.target.value, 10)));
      document.querySelectorAll('[data-slider="comp-alpha-b"]').forEach((el) => {
        el.addEventListener('click', () => updateCompAlphaB(parseInt(el.dataset.val, 10)));
      });
    }
  }

  function initEventListeners() {
    // Metric Selector
    const metricSelect = document.getElementById('metric-select');
    if (metricSelect) {
      metricSelect.addEventListener('change', (e) => {
        state.metric = e.target.value;
        renderAll();
      });
    }

    // Round Selector
    const roundSelect = document.getElementById('round-view-select');
    if (roundSelect) {
      roundSelect.addEventListener('change', (e) => {
        state.roundView = e.target.value;
        renderAll();
      });
    }

    // Config A & B Selectors for Comparator
    const selA = document.getElementById('select-config-a');
    const selB = document.getElementById('select-config-b');
    if (selA) selA.addEventListener('change', (e) => { state.configA = e.target.value; updateComparator(); });
    if (selB) selB.addEventListener('change', (e) => { state.configB = e.target.value; updateComparator(); });
  }

  // ── Data Loading ─────────────────────────────────────────────────────────────
  function loadInitialData() {
    if (window.INITIAL_SWEEP_DATA && window.INITIAL_SWEEP_DATA.length > 0) {
      rawData = sanitizeRecords(window.INITIAL_SWEEP_DATA);
    }
    populateConfigSelectors();
    renderAll();
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
        f1: parseFloat(r.f1) || 0,
        mcc: parseFloat(r.mcc) || 0,
        auc_roc: parseFloat(r.auc_roc) || 0,
        accuracy: parseFloat(r.accuracy) || 0,
        precision: parseFloat(r.precision) || 0,
        recall: parseFloat(r.recall) || 0,
        loss: parseFloat(r.loss) || 0,
      };
    });
  }

  function getFilteredDataset() {
    let filtered = [...rawData];

    if (state.roundView === 'final') {
      const maxRound = Math.max(...rawData.map((d) => d.round), 1);
      filtered = filtered.filter((d) => d.round === maxRound);
    }

    if (state.alphaFilter !== 'all') {
      const targetAlpha = parseFloat(state.alphaFilter);
      filtered = filtered.filter((d) => Math.abs(d.alpha - targetAlpha) < 0.001);
    }

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
    if (!selA || !selB || rawData.length === 0) return;

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

    if (document.getElementById('hospital-cards-container')) {
      renderHospitalSiloModule();
    }
    if (document.getElementById('case-tabs-container')) {
      renderClinicalCaseTester();
    }
    if (document.getElementById('kpi-best-value')) {
      updateKPICards();
    }
    if (document.getElementById('select-config-a') || document.getElementById('metrics-list-a')) {
      updateComparator();
    }
    if (document.getElementById('heatmaps-container')) {
      renderHeatmaps();
    }
    if (document.getElementById('convergence-line-chart')) {
      renderConvergenceChart();
    }
    if (document.getElementById('tradeoff-bar-chart')) {
      renderTradeoffBarChart();
    }
  }

  // ── KPI Overview Cards ───────────────────────────────────────────────────────
  function updateKPICards() {
    if (rawData.length === 0) return;
    const maxRound = Math.max(...rawData.map((d) => d.round), 1);
    const finalData = rawData.filter((d) => d.round === maxRound);
    if (finalData.length === 0) return;

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
      dpSubEl.textContent = `DP: ${avgDp.toFixed(3)} vs Baseline: ${avgNoDp.toFixed(3)}`;
    }

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
      skewSubEl.textContent = `α=0.1: ${avg01.toFixed(3)} | α=100: ${avg100.toFixed(3)}`;
    }

    const r4Rows = finalData.filter((d) => d.rank === 4);
    const r8Rows = finalData.filter((d) => d.rank === 8);
    const avgR4 = r4Rows.length ? r4Rows.reduce((a, b) => a + b[state.metric], 0) / r4Rows.length : 0;
    const avgR8 = r8Rows.length ? r8Rows.reduce((a, b) => a + b[state.metric], 0) / r8Rows.length : 0;
    const rankDelta = avgR4 > 0 ? ((avgR8 - avgR4) / avgR4) * 100 : 0;

    const loraValEl = document.getElementById('kpi-lora-value');
    const loraSubEl = document.getElementById('kpi-lora-subtext');
    if (loraValEl && loraSubEl) {
      loraValEl.textContent = `${rankDelta > 0 ? '+' : ''}${rankDelta.toFixed(1)}%`;
      loraValEl.style.color = rankDelta >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';
      lorSubEl.textContent = `r=8: ${avgR8.toFixed(3)} vs r=4: ${avgR4.toFixed(3)}`;
    }
  }

  // ── Hospital Silo Comparison Module ──────────────────────────────────────────
  function getSiloKey() {
    return `r${siloState.rank}_eps${siloState.eps}_alpha${siloState.alpha}`;
  }

  function renderHospitalSiloModule() {
    const container = document.getElementById('hospital-cards-container');
    if (!container || !window.HOSPITAL_COMPARISON_DATA) return;

    const key = getSiloKey();
    const data = window.HOSPITAL_COMPARISON_DATA[key] || Object.values(window.HOSPITAL_COMPARISON_DATA)[0];
    if (!data) return;

    const profiles = window.HOSPITAL_PROFILES || {
      A: { name: 'Hospital A (Surgical Center)', icon: 'A', description: 'Overrepresents surgical cases.', badge_color: '#b45309' },
      B: { name: 'Hospital B (Community Clinic)', icon: 'B', description: 'Overrepresents medical consults.', badge_color: '#0f766e' },
      C: { name: 'Hospital C (Regional Hospital)', icon: 'C', description: 'Mixed intake, localized sample.', badge_color: '#78716c' },
      FED: { name: 'Global Federated Model', icon: 'FL', description: 'Aggregated intelligence with zero raw data sharing.', badge_color: '#1c1917' }
    };

    const entities = [
      { id: 'hospital_A', profile: profiles.A, stats: data.hospital_A, isFed: false },
      { id: 'hospital_B', profile: profiles.B, stats: data.hospital_B, isFed: false },
      { id: 'hospital_C', profile: profiles.C, stats: data.hospital_C, isFed: false },
      { id: 'federated_global', profile: profiles.FED, stats: data.federated_global, isFed: true },
    ];

    container.innerHTML = '';

    entities.forEach((ent) => {
      const s = ent.stats;
      const card = document.createElement('div');
      card.className = `hosp-card ${ent.isFed ? 'highlight' : ''}`;

      const f1Good = s.f1 >= 0.75;
      const accGood = s.accuracy >= 0.75;
      const verdictClass = ent.isFed ? 'success' : (s.f1 < 0.65 ? 'warning' : 'success');
      const badgeText = ent.isFed ? 'Federated Global' : 'Local Isolated Model';
      const badgeStyle = ent.isFed ? 'background: #ccfbf1; color: #0f766e; border: 1px solid rgba(15, 118, 110, 0.3); font-weight: 700;' : 'background: #f5f3ef; color: #78716c; border: 1px solid #e7e5e4;';

      card.innerHTML = `
        <div class="hosp-header">
          <div class="hosp-title-group">
            <span class="badge" style="font-weight: 700; font-size: 0.8rem; background: ${ent.profile.badge_color}; color: #ffffff; padding: 4px 8px;">${ent.profile.icon}</span>
            <div>
              <div class="hosp-name">${ent.profile.name}</div>
              <span class="badge" style="font-size: 0.68rem; margin-top: 3px; display: inline-block; ${badgeStyle}">${badgeText}</span>
            </div>
          </div>
        </div>

        <div class="hosp-desc">${ent.profile.description}</div>

        <div class="hosp-dist-bar-wrapper">
          <div class="hosp-dist-labels">
            <span>Surgical: ${s.surgical_pct.toFixed(1)}%</span>
            <span>Medical: ${s.medical_pct.toFixed(1)}%</span>
          </div>
          <div class="hosp-dist-bar">
            <div class="hosp-dist-surg" style="width: ${s.surgical_pct}%;"></div>
            <div class="hosp-dist-med" style="width: ${s.medical_pct}%;"></div>
          </div>
        </div>

        <div class="hosp-metrics-grid">
          <div class="hosp-metric-box">
            <span class="hosp-metric-title">Macro F1</span>
            <span class="hosp-metric-val ${f1Good ? 'good' : 'bad'}">${(s.f1 * 100).toFixed(1)}%</span>
          </div>
          <div class="hosp-metric-box">
            <span class="hosp-metric-title">Accuracy</span>
            <span class="hosp-metric-val ${accGood ? 'good' : 'bad'}">${(s.accuracy * 100).toFixed(1)}%</span>
          </div>
          <div class="hosp-metric-box">
            <span class="hosp-metric-title">Surg Recall</span>
            <span class="hosp-metric-val ${s.surgical_recall >= 0.7 ? 'good' : 'bad'}">${(s.surgical_recall * 100).toFixed(1)}%</span>
          </div>
          <div class="hosp-metric-box">
            <span class="hosp-metric-title">Med Recall</span>
            <span class="hosp-metric-val ${s.medical_recall >= 0.7 ? 'good' : 'bad'}">${(s.medical_recall * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div class="hosp-verdict ${verdictClass}">
          ${s.bias_verdict}
        </div>
      `;

      container.appendChild(card);
    });

    updateSiloCharts(data);
  }

  function updateSiloCharts(data) {
    if (!data) return;

    // 1. Radar Chart
    const ctxRadar = document.getElementById('silo-radar-chart');
    if (ctxRadar) {
      const radarLabels = ['Macro F1', 'Accuracy', 'AUC-ROC', 'Surgical Recall', 'Medical Recall'];
      const toPct = (val) => Math.max(0, Math.min(100, Math.round(val * 100)));

      const datasets = [
        {
          label: 'Hospital A (Surgical Skew)',
          data: [
            toPct(data.hospital_A.f1),
            toPct(data.hospital_A.accuracy),
            toPct(data.hospital_A.auc_roc),
            toPct(data.hospital_A.surgical_recall),
            toPct(data.hospital_A.medical_recall),
          ],
          borderColor: '#b45309',
          backgroundColor: 'rgba(180, 83, 9, 0.12)',
          pointBackgroundColor: '#b45309',
        },
        {
          label: 'Hospital B (Medical Skew)',
          data: [
            toPct(data.hospital_B.f1),
            toPct(data.hospital_B.accuracy),
            toPct(data.hospital_B.auc_roc),
            toPct(data.hospital_B.surgical_recall),
            toPct(data.hospital_B.medical_recall),
          ],
          borderColor: '#0f766e',
          backgroundColor: 'rgba(15, 118, 110, 0.12)',
          pointBackgroundColor: '#0f766e',
        },
        {
          label: 'Hospital C (Regional Skew)',
          data: [
            toPct(data.hospital_C.f1),
            toPct(data.hospital_C.accuracy),
            toPct(data.hospital_C.auc_roc),
            toPct(data.hospital_C.surgical_recall),
            toPct(data.hospital_C.medical_recall),
          ],
          borderColor: '#78716c',
          backgroundColor: 'rgba(120, 113, 108, 0.12)',
          pointBackgroundColor: '#78716c',
        },
        {
          label: 'Federated Global Model',
          data: [
            toPct(data.federated_global.f1),
            toPct(data.federated_global.accuracy),
            toPct(data.federated_global.auc_roc),
            toPct(data.federated_global.surgical_recall),
            toPct(data.federated_global.medical_recall),
          ],
          borderColor: '#1c1917',
          backgroundColor: 'rgba(28, 25, 23, 0.18)',
          pointBackgroundColor: '#1c1917',
          borderWidth: 2.5,
        },
      ];

      if (siloRadarChart) {
        siloRadarChart.data.datasets = datasets;
        siloRadarChart.update();
      } else {
        siloRadarChart = new Chart(ctxRadar, {
          type: 'radar',
          data: { labels: radarLabels, datasets },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              r: {
                min: 0,
                max: 100,
                ticks: { stepSize: 20, color: '#78716c', backdropColor: 'transparent', font: { size: 10 } },
                grid: { color: '#e7e5e4' },
                angleLines: { color: '#e7e5e4' },
                pointLabels: { color: '#1c1917', font: { size: 11, weight: '700' } }
              }
            },
            plugins: {
              legend: { position: 'bottom', labels: { color: '#1c1917', font: { size: 11, weight: '600' }, boxWidth: 12 } },
            }
          }
        });
      }
    }

    // 2. Class Sensitivity Bar Chart
    const ctxSens = document.getElementById('silo-sensitivity-chart');
    if (ctxSens) {
      const labels = ['Hospital A', 'Hospital B', 'Hospital C', 'Federated Global'];
      const surgRecalls = [
        Math.round(data.hospital_A.surgical_recall * 100),
        Math.round(data.hospital_B.surgical_recall * 100),
        Math.round(data.hospital_C.surgical_recall * 100),
        Math.round(data.federated_global.surgical_recall * 100),
      ];
      const medRecalls = [
        Math.round(data.hospital_A.medical_recall * 100),
        Math.round(data.hospital_B.medical_recall * 100),
        Math.round(data.hospital_C.medical_recall * 100),
        Math.round(data.federated_global.medical_recall * 100),
      ];

      const barDatasets = [
        {
          label: 'Surgical Sensitivity (%)',
          data: surgRecalls,
          backgroundColor: '#b45309',
          borderRadius: 0,
        },
        {
          label: 'Medical Sensitivity (%)',
          data: medRecalls,
          backgroundColor: '#0f766e',
          borderRadius: 0,
        }
      ];

      if (siloSensitivityChart) {
        siloSensitivityChart.data.datasets = barDatasets;
        siloSensitivityChart.update();
      } else {
        siloSensitivityChart = new Chart(ctxSens, {
          type: 'bar',
          data: { labels, datasets: barDatasets },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                min: 0,
                max: 100,
                ticks: { color: '#78716c', stepSize: 20, callback: (v) => v + '%' },
                grid: { color: '#e7e5e4' },
              },
              x: {
                ticks: { color: '#1c1917', font: { weight: '600' } },
                grid: { display: false }
              }
            },
            plugins: {
              legend: { position: 'bottom', labels: { color: '#1c1917', font: { size: 11, weight: '600' }, boxWidth: 12 } },
            }
          }
        });
      }
    }
  }

  // ── Clinical Case Tester Module ──────────────────────────────────────────────
  function renderClinicalCaseTester() {
    const tabsContainer = document.getElementById('case-tabs-container');
    if (!tabsContainer || !window.CLINICAL_TEST_CASES) return;

    const key = getSiloKey();
    const cases = window.CLINICAL_TEST_CASES[key] || Object.values(window.CLINICAL_TEST_CASES)[0];
    if (!cases || cases.length === 0) return;

    let activeCase = cases.find((c) => c.id === activeCaseId) || cases[0];
    activeCaseId = activeCase.id;

    tabsContainer.innerHTML = '';
    cases.forEach((c) => {
      const btn = document.createElement('button');
      btn.className = `case-tab-btn ${c.id === activeCaseId ? 'active' : ''}`;
      btn.textContent = `${c.title} (${c.category})`;
      btn.addEventListener('click', () => {
        activeCaseId = c.id;
        renderClinicalCaseTester();
      });
      tabsContainer.appendChild(btn);
    });

    const titleEl = document.getElementById('case-active-title');
    const badgeEl = document.getElementById('case-active-badge');
    const textEl = document.getElementById('case-active-text');
    const noteEl = document.getElementById('case-active-note');

    if (titleEl) titleEl.textContent = `${activeCase.title}`;
    if (badgeEl) {
      badgeEl.textContent = activeCase.true_name === 'surgical_procedural' ? 'True: Surgical' : 'True: Medical';
      badgeEl.className = `case-badge-actual ${activeCase.true_name === 'surgical_procedural' ? 'surgical' : 'medical'}`;
    }
    if (textEl) textEl.textContent = `"${activeCase.snippet}"`;
    if (noteEl) noteEl.textContent = `Diagnosis Note: ${activeCase.clinical_note}`;

    const predContainer = document.getElementById('case-predictions-container');
    if (predContainer && activeCase.predictions) {
      predContainer.innerHTML = '';

      const modelKeys = [
        { key: 'hospital_A', title: 'Hospital A (Surgical Skew)', isFed: false },
        { key: 'hospital_B', title: 'Hospital B (Medical Skew)', isFed: false },
        { key: 'hospital_C', title: 'Hospital C (Regional Skew)', isFed: false },
        { key: 'federated_global', title: 'Global Federated Model', isFed: true },
      ];

      modelKeys.forEach((m) => {
        const p = activeCase.predictions[m.key];
        if (!p) return;

        const card = document.createElement('div');
        const correctClass = p.is_correct ? 'correct' : 'incorrect';
        const fedClass = m.isFed ? 'fed-highlight' : '';
        card.className = `model-pred-card ${correctClass} ${fedClass}`;

        const isSurg = p.pred_class === 1;
        const statusText = p.is_correct ? 'Correct' : 'Misdiagnosed';
        const statusStyle = p.is_correct 
          ? 'background: #ccfbf1; color: #0f766e; border: 1px solid rgba(15, 118, 110, 0.3); font-weight: 700;' 
          : 'background: #fef3c7; color: #b45309; border: 1px solid rgba(180, 83, 9, 0.3); font-weight: 700;';

        card.innerHTML = `
          <div class="pred-model-title">${m.title}</div>
          <div class="pred-verdict-row">
            <span class="pred-class-text ${isSurg ? 'surgical' : 'medical'}">
              ${isSurg ? 'Surgical Intervention' : 'Conservative Medical'}
            </span>
            <span class="badge" style="font-size: 0.72rem; padding: 2px 6px; ${statusStyle}">${statusText}</span>
          </div>

          <div class="pred-prob-bar-container">
            <div class="pred-prob-labels">
              <span>Confidence: ${p.confidence}%</span>
              <span>Surg: ${p.prob_surgical}% | Med: ${p.prob_medical}%</span>
            </div>
            <div class="pred-prob-bar-track">
              <div style="background: #b45309; width: ${p.prob_surgical}%; height: 100%;"></div>
              <div style="background: #0f766e; width: ${p.prob_medical}%; height: 100%;"></div>
            </div>
          </div>
        `;

        predContainer.appendChild(card);
      });
    }
  }

  // ── Comparator Module ────────────────────────────────────────────────────────
  function updateComparator() {
    const listA = document.getElementById('metrics-list-a');
    const listB = document.getElementById('metrics-list-b');
    if (!listA || !listB || rawData.length === 0) return;

    const maxRound = Math.max(...rawData.map((d) => d.round), 1);
    const finalRuns = rawData.filter((d) => d.round === maxRound);

    const runA = finalRuns.find((d) => d.rank === compState.rankA && d.epsilon === compState.epsA && Math.abs(d.alpha - compState.alphaA) < 0.01) || finalRuns[0];
    const runB = finalRuns.find((d) => d.rank === compState.rankB && d.epsilon === compState.epsB && Math.abs(d.alpha - compState.alphaB) < 0.01) || finalRuns[finalRuns.length - 1];

    if (!runA || !runB) return;

    const metrics = ['f1', 'mcc', 'auc_roc', 'accuracy', 'precision', 'recall', 'loss'];

    listA.innerHTML = '';
    listB.innerHTML = '';

    metrics.forEach((m) => {
      const valA = runA[m];
      const valB = runB[m];
      const isHigherBetter = m !== 'loss';

      let diff = valB - valA;
      let pctDiff = valA !== 0 ? (diff / valA) * 100 : 0;
      let isGood = isHigherBetter ? diff > 0 : diff < 0;

      const itemA = document.createElement('div');
      itemA.className = 'metric-item';
      itemA.innerHTML = `
        <span class="metric-name">${METRIC_LABELS[m]}</span>
        <span class="metric-val">${valA.toFixed(4)}</span>
      `;
      listA.appendChild(itemA);

      const deltaClass = Math.abs(diff) < 0.0001 ? 'delta-neutral' : (isGood ? 'delta-positive' : 'delta-negative');
      const deltaSign = diff > 0 ? '+' : '';

      const itemB = document.createElement('div');
      itemB.className = 'metric-item';
      itemB.innerHTML = `
        <span class="metric-name">${METRIC_LABELS[m]}</span>
        <div style="display: flex; align-items: baseline; gap: 8px;">
          <span class="metric-val">${valB.toFixed(4)}</span>
          <span class="metric-delta ${deltaClass}">${deltaSign}${pctDiff.toFixed(1)}%</span>
        </div>
      `;
      listB.appendChild(itemB);
    });

    const labelA = `Model A (r=${compState.rankA}, ${compState.epsA === 'none' ? 'No DP' : 'ε=' + compState.epsA}, α=${compState.alphaA})`;
    const labelB = `Model B (r=${compState.rankB}, ${compState.epsB === 'none' ? 'No DP' : 'ε=' + compState.epsB}, α=${compState.alphaB})`;
    renderVersusRadar(runA, runB, labelA, labelB);
  }

  function renderVersusRadar(runA, runB, labelA, labelB) {
    const ctx = document.getElementById('versus-radar-chart');
    if (!ctx) return;

    const labels = ['F1-Score', 'MCC', 'AUC-ROC', 'Accuracy', 'Precision', 'Recall'];
    const dataA = [runA.f1, Math.max(0, runA.mcc), runA.auc_roc, runA.accuracy, runA.precision, runA.recall].map((v) => Math.round(v * 100));
    const dataB = [runB.f1, Math.max(0, runB.mcc), runB.auc_roc, runB.accuracy, runB.precision, runB.recall].map((v) => Math.round(v * 100));

    const datasets = [
      {
        label: labelA || 'Model A',
        data: dataA,
        borderColor: '#78716c',
        backgroundColor: 'rgba(120, 113, 108, 0.15)',
        pointBackgroundColor: '#78716c',
      },
      {
        label: labelB || 'Model B',
        data: dataB,
        borderColor: '#0f766e',
        backgroundColor: 'rgba(15, 118, 110, 0.2)',
        pointBackgroundColor: '#0f766e',
      },
    ];

    if (versusRadarChart) {
      versusRadarChart.data.datasets = datasets;
      versusRadarChart.update();
    } else {
      versusRadarChart = new Chart(ctx, {
        type: 'radar',
        data: { labels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            r: {
              min: 0,
              max: 100,
              ticks: { stepSize: 20, color: '#78716c', backdropColor: 'transparent', font: { size: 10 } },
              grid: { color: '#e7e5e4' },
              angleLines: { color: '#e7e5e4' },
              pointLabels: { color: '#1c1917', font: { size: 11, weight: '700' } }
            }
          },
          plugins: {
            legend: { position: 'bottom', labels: { color: '#1c1917', font: { size: 11, weight: '600' }, boxWidth: 12 } },
          }
        }
      });
    }
  }

  // ── Heatmaps Module ──────────────────────────────────────────────────────────
  function renderHeatmaps() {
    const container = document.getElementById('heatmaps-container');
    if (!container) return;

    const maxRound = Math.max(...rawData.map((d) => d.round), 1);
    const data = rawData.filter((d) => d.round === maxRound);

    container.innerHTML = '';
    const alphas = [0.1, 1.0, 100.0];
    const alphaLabels = {
      0.1: 'Extreme Skew (α = 0.1)',
      1.0: 'Moderate Skew (α = 1.0)',
      100.0: 'Uniform IID (α = 100.0)',
    };

    const ranks = [2, 4, 8, 16];
    const epsilons = ['none', '8.0', '3.0', '1.0'];

    const allVals = data.map((d) => d[state.metric]);
    const minVal = Math.min(...allVals);
    const maxVal = Math.max(...allVals);

    alphas.forEach((alpha) => {
      const alphaRuns = data.filter((d) => Math.abs(d.alpha - alpha) < 0.001);
      const card = document.createElement('div');
      card.className = 'heatmap-card';

      let tableHtml = `
        <div class="heatmap-header">
          <span class="heatmap-title">${alphaLabels[alpha]}</span>
        </div>
        <table class="heatmap-table">
          <thead>
            <tr>
              <th style="font-size: 0.72rem; color: var(--text-muted);">Rank \\ ε</th>
              ${epsilons.map((e) => `<th>${e === 'none' ? 'No DP' : 'ε=' + e}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
      `;

      ranks.forEach((r) => {
        tableHtml += `<tr><td style="font-weight: 700; color: var(--text-secondary);">r = ${r}</td>`;
        epsilons.forEach((e) => {
          const run = alphaRuns.find((d) => d.rank === r && d.epsilon === e);
          if (run) {
            const val = run[state.metric];
            const ratio = maxVal > minVal ? (val - minVal) / (maxVal - minVal) : 0.5;
            const bg = getHeatmapColor(ratio, state.metric !== 'loss');
            tableHtml += `<td class="heatmap-cell" style="background-color: ${bg};" title="r=${r}, ε=${e}, α=${alpha}: ${val.toFixed(4)}">${val.toFixed(3)}</td>`;
          } else {
            tableHtml += `<td class="heatmap-cell" style="color: var(--text-muted);">-</td>`;
          }
        });
        tableHtml += `</tr>`;
      });

      tableHtml += `</tbody></table>`;
      card.innerHTML = tableHtml;
      container.appendChild(card);
    });
  }

  function getHeatmapColor(ratio, isHigherBetter) {
    const rNorm = isHigherBetter ? ratio : 1 - ratio;
    if (rNorm >= 0.65) {
      const alpha = 0.12 + ((rNorm - 0.65) / 0.35) * 0.38;
      return `rgba(15, 118, 110, ${alpha.toFixed(2)})`;
    } else if (rNorm >= 0.35) {
      const alpha = 0.08 + ((0.65 - rNorm) / 0.3) * 0.25;
      return `rgba(180, 83, 9, ${alpha.toFixed(2)})`;
    } else {
      const alpha = 0.12 + ((0.35 - rNorm) / 0.35) * 0.25;
      return `rgba(120, 113, 108, ${alpha.toFixed(2)})`;
    }
  }

  // ── Convergence & Tradeoffs Charts ───────────────────────────────────────────
  function renderConvergenceChart() {
    const ctx = document.getElementById('convergence-line-chart');
    if (!ctx || rawData.length === 0) return;

    const sampleConfigs = [
      { rank: 8, epsilon: 'none', alpha: 100.0, color: '#0f766e', label: 'r=8, No DP, α=100' },
      { rank: 8, epsilon: 'none', alpha: 0.1, color: '#b45309', label: 'r=8, No DP, α=0.1 (Skewed)' },
      { rank: 8, epsilon: '8.0', alpha: 100.0, color: '#1c1917', label: 'r=8, ε=8.0, α=100' },
      { rank: 4, epsilon: 'none', alpha: 100.0, color: '#78716c', label: 'r=4, No DP, α=100' },
      { rank: 2, epsilon: '1.0', alpha: 0.1, color: '#d97706', label: 'r=2, ε=1.0, α=0.1 (Strict)' },
    ];

    const rounds = [1, 2, 3, 4, 5];
    const datasets = sampleConfigs.map((cfg) => {
      const runs = rawData.filter((d) => d.rank === cfg.rank && d.epsilon === cfg.epsilon && Math.abs(d.alpha - cfg.alpha) < 0.01);
      runs.sort((a, b) => a.round - b.round);
      return {
        label: cfg.label,
        data: runs.map((d) => d[state.metric]),
        borderColor: cfg.color,
        backgroundColor: cfg.color,
        tension: 0.2,
        borderWidth: 2,
        pointRadius: 3,
      };
    });

    if (convergenceLineChart) {
      convergenceLineChart.data.datasets = datasets;
      convergenceLineChart.update();
    } else {
      convergenceLineChart = new Chart(ctx, {
        type: 'line',
        data: { labels: rounds.map((r) => `Round ${r}`), datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              ticks: { color: '#78716c' },
              grid: { color: '#e7e5e4' },
            },
            x: {
              ticks: { color: '#78716c' },
              grid: { color: '#e7e5e4' },
            }
          },
          plugins: {
            legend: { position: 'bottom', labels: { color: '#1c1917', font: { size: 11, weight: '600' }, boxWidth: 12 } }
          }
        }
      });
    }
  }

  function renderTradeoffBarChart() {
    const ctx = document.getElementById('tradeoff-bar-chart');
    if (!ctx || rawData.length === 0) return;

    const maxRound = Math.max(...rawData.map((d) => d.round), 1);
    const data = rawData.filter((d) => d.round === maxRound && Math.abs(d.alpha - 1.0) < 0.01);

    const ranks = [2, 4, 8, 16];
    const epsilons = ['none', '8.0', '3.0', '1.0'];
    const colors = {
      'none': '#0f766e',
      '8.0': '#14b8a6',
      '3.0': '#d97706',
      '1.0': '#b45309',
    };

    const datasets = epsilons.map((eps) => {
      const vals = ranks.map((r) => {
        const item = data.find((d) => d.rank === r && d.epsilon === eps);
        return item ? item[state.metric] : 0;
      });
      return {
        label: eps === 'none' ? 'No DP' : `ε = ${eps}`,
        data: vals,
        backgroundColor: colors[eps],
        borderRadius: 0,
      };
    });

    if (tradeoffBarChart) {
      tradeoffBarChart.data.datasets = datasets;
      tradeoffBarChart.update();
    } else {
      tradeoffBarChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: ranks.map((r) => `Rank r=${r}`), datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              ticks: { color: '#78716c' },
              grid: { color: '#e7e5e4' }
            },
            x: {
              ticks: { color: '#1c1917', font: { weight: '600' } },
              grid: { display: false }
            }
          },
          plugins: {
            legend: { position: 'bottom', labels: { color: '#1c1917', font: { size: 11, weight: '600' }, boxWidth: 12 } }
          }
        }
      });
    }
  }

})();
