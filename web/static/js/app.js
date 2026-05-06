// ── GNN-Based IDS Dashboard — app.js ──────────────────────────────────────
// Dark-mode-first Chart.js config to match the redesigned CSS theme.

let gnnMetrics = null, baselineMetrics = null, classificationReport = null;
const charts = {};

// ── Palette (matches CSS vars) ─────────────────────────────────────────────
const C = {
  text:    '#e8eaf2',
  muted:   '#8b90a8',
  border:  'rgba(255,255,255,0.07)',
  accent:  '#4f6ef7',
  green:   '#22c55e',
  blue:    '#3b82f6',
  amber:   '#f59e0b',
  red:     '#ef4444',
  teal:    '#14b8a6',
  purple:  '#a78bfa',
  surface: '#22263a',
};

// ── Global Chart.js defaults ───────────────────────────────────────────────
Chart.defaults.color          = C.muted;
Chart.defaults.borderColor    = C.border;
Chart.defaults.font.family    = "'Inter','Segoe UI',system-ui,sans-serif";
Chart.defaults.font.size      = 12;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding  = 16;
Chart.defaults.plugins.tooltip.backgroundColor = C.surface;
Chart.defaults.plugins.tooltip.titleColor      = C.text;
Chart.defaults.plugins.tooltip.bodyColor       = C.muted;
Chart.defaults.plugins.tooltip.borderColor     = 'rgba(255,255,255,0.13)';
Chart.defaults.plugins.tooltip.borderWidth     = 1;
Chart.defaults.plugins.tooltip.padding         = 10;

// shared scale helpers
const pctTicks = v => (v * 100).toFixed(1) + '%';
const yScale = (max = 1) => ({
  beginAtZero: true, max,
  ticks: { callback: pctTicks, color: C.muted },
  grid: { color: C.border },
});
const xScale = () => ({ ticks: { color: C.muted }, grid: { color: C.border } });

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', loadAllData);

async function loadAllData() {
  try {
    const [r1, r2, r3] = await Promise.all([
      fetch('/api/metrics/gnn'),
      fetch('/api/metrics/baseline'),
      fetch('/api/classification-report'),
    ]);
    gnnMetrics            = await r1.json();
    baselineMetrics       = await r2.json();
    classificationReport  = await r3.json();

    document.getElementById('loadingSpinner').style.display = 'none';
    document.getElementById('mainContent').style.display    = 'block';

    populateOverview();
    populatePerformance();
    populateTrainingHistory();
    populateComparison();
    populateAnalysis();
    initCharts();
    initSidebarHighlight();
  } catch (e) {
    console.error(e);
    document.getElementById('loadingSpinner').innerHTML =
      `<div style="color:#ef4444;text-align:center"><i class="bi bi-exclamation-triangle" style="font-size:32px"></i><p style="margin-top:12px">Failed to load metrics data</p></div>`;
  }
}

// ── Populate helpers ───────────────────────────────────────────────────────
function set(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? '—';
}
const pct  = v => v != null ? (v * 100).toFixed(2) + '%' : '—';
const num  = v => v != null ? Number(v).toLocaleString() : '—';
const diff = (a, b) => {
  const d = a - b;
  const s = d > 0 ? '+' : '';
  return s + (d * 100).toFixed(2) + '%';
};

function populateOverview() {
  const m = gnnMetrics;
  set('overviewBestF1',       pct(m.best_val_f1));
  set('overviewAccuracy',     pct(m.test.accuracy));
  set('overviewPrecision',    pct(m.test.precision));
  set('overviewRecall',       pct(m.test.recall));
  set('overviewBestEpoch',    m.best_epoch);
  set('overviewThreshold',    m.decision_threshold.toFixed(2));
  set('overviewBatchSize',    m.batch_size);
  set('overviewBestLoss',     m.best_val_loss.toFixed(6));
  set('overviewNormalCount',  num(m.train_class_counts[0]));
  set('overviewAttackCount',  num(m.train_class_counts[1]));
  set('overviewNormalWeight', m.class_weights[0].toFixed(4));
  set('overviewAttackWeight', m.class_weights[1].toFixed(4));
}

function populatePerformance() {
  const t = gnnMetrics.test, a = gnnMetrics.argmax_test;
  set('tableAccuracyTuned',  pct(t.accuracy));  set('tableAccuracyArgmax',  pct(a.accuracy));  set('tableAccuracyDiff',  diff(t.accuracy,  a.accuracy));
  set('tablePrecisionTuned', pct(t.precision)); set('tablePrecisionArgmax', pct(a.precision)); set('tablePrecisionDiff', diff(t.precision, a.precision));
  set('tableRecallTuned',    pct(t.recall));    set('tableRecallArgmax',    pct(a.recall));    set('tableRecallDiff',    diff(t.recall,    a.recall));
  set('tableF1Tuned',        pct(t.f1));        set('tableF1Argmax',        pct(a.f1));        set('tableF1Diff',        diff(t.f1,        a.f1));
}

function populateTrainingHistory() {
  const m = gnnMetrics, h0 = m.history[0];
  const improvement = ((h0.train_loss - m.best_val_loss) / h0.train_loss * 100).toFixed(2) + '%';
  set('trainingTotalEpochs',     m.history.length);
  set('trainingBestEpoch',       m.best_epoch);
  set('trainingInitialLoss',     h0.train_loss.toFixed(6));
  set('trainingBestValLoss',     m.best_val_loss.toFixed(6));
  set('trainingImprovement',     improvement);
  set('trainingBestEpochNumber', m.best_epoch);
  set('trainingBestF1',          pct(m.best_val_f1));
  set('trainingBestLoss',        m.best_val_loss.toFixed(6));
  set('trainingBestAccuracy',    pct(m.test.accuracy));
}

function populateComparison() {
  const m = gnnMetrics.test, rf = baselineMetrics.random_forest, xgb = baselineMetrics.xgboost;
  set('comparisonGnnAccuracy',  pct(m.accuracy));
  set('comparisonGnnPrecision', pct(m.precision));
  set('comparisonGnnRecall',    pct(m.recall));
  set('comparisonGnnF1',        pct(m.f1));
  set('comparisonRfAccuracy',   pct(rf.accuracy));  set('comparisonRfPrecision', pct(rf.precision));
  set('comparisonRfRecall',     pct(rf.recall));    set('comparisonRfF1',        pct(rf.f1));  set('comparisonRfRocAuc', pct(rf.roc_auc));
  set('comparisonXgbAccuracy',  pct(xgb.accuracy)); set('comparisonXgbPrecision', pct(xgb.precision));
  set('comparisonXgbRecall',    pct(xgb.recall));   set('comparisonXgbF1',        pct(xgb.f1)); set('comparisonXgbRocAuc', pct(xgb.roc_auc));
}

function populateAnalysis() {
  const rep = classificationReport;
  const cl  = (key, prefix) => {
    set(prefix + 'Precision', pct(rep[key].precision));
    set(prefix + 'Recall',    pct(rep[key].recall));
    set(prefix + 'F1',        pct(rep[key]['f1-score']));
    set(prefix + 'Support',   num(rep[key].support));
  };
  cl('0',            'analysisPrecisionClass0'.replace('Precision','').concat('Precision') && 'analysisPrecisionClass');
  // simpler direct calls:
  set('analysisPrecisionClass0', pct(rep['0'].precision)); set('analysisRecallClass0', pct(rep['0'].recall)); set('analysisF1Class0', pct(rep['0']['f1-score'])); set('analysisSupportClass0', num(rep['0'].support));
  set('analysisPrecisionClass1', pct(rep['1'].precision)); set('analysisRecallClass1', pct(rep['1'].recall)); set('analysisF1Class1', pct(rep['1']['f1-score'])); set('analysisSupportClass1', num(rep['1'].support));
  set('analysisMacroPrecision',  pct(rep['macro avg'].precision));    set('analysisMacroRecall', pct(rep['macro avg'].recall));    set('analysisMacroF1', pct(rep['macro avg']['f1-score']));    set('analysisMacroSupport', num(rep['macro avg'].support));
  set('analysisWeightedPrecision', pct(rep['weighted avg'].precision)); set('analysisWeightedRecall', pct(rep['weighted avg'].recall)); set('analysisWeightedF1', pct(rep['weighted avg']['f1-score'])); set('analysisWeightedSupport', num(rep['weighted avg'].support));
}

// ── Charts ─────────────────────────────────────────────────────────────────
function initCharts() {
  chartTestMetrics();
  chartThreshold();
  chartClassDoughnut('normalClassMetricsChart', gnnMetrics.test.class_0, [C.green, C.teal, '#a3e635']);
  chartClassDoughnut('attackClassMetricsChart',  gnnMetrics.test.class_1, [C.red,   C.amber, '#fb923c']);
  chartLossHistory();
  chartLineHistory('accuracyHistoryChart', 'Accuracy',          h => h.threshold_tuned.accuracy,  C.green);
  chartLineHistory('f1HistoryChart',       'F1 Score',          h => h.threshold_tuned.f1,         C.amber);
  chartPrecisionRecall();
  chartBarComparison('accuracyComparisonChart', 'Accuracy', m => m.accuracy);
  chartBarComparison('f1ComparisonChart',       'F1 Score', m => m.f1);
  chartRocAuc();
  chartPie('classDistributionChart', ['Normal Traffic','Attack Traffic'], [classificationReport['0'].support, classificationReport['1'].support], [C.green, C.red]);
  chartPie('predictionDistributionChart', ['Predicted Normal','Predicted Attack'], [gnnMetrics.test.prediction_counts[0], gnnMetrics.test.prediction_counts[1]], [C.teal, '#fb923c']);
  chartRadar();
  chartMacroWeighted();
}

function barDataset(label, data, color) {
  return { label, data, backgroundColor: color, borderRadius: 6, borderSkipped: false };
}

function chartTestMetrics() {
  const t = gnnMetrics.test;
  new Chart(document.getElementById('testMetricsChart'), {
    type: 'bar',
    data: {
      labels: ['Accuracy','Precision','Recall','F1 Score'],
      datasets: [barDataset('Test Metrics', [t.accuracy, t.precision, t.recall, t.f1], [C.blue, C.green, C.amber, C.red])],
    },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: yScale(), x: xScale() } },
  });
}

function chartThreshold() {
  const t = gnnMetrics.test, a = gnnMetrics.argmax_test;
  new Chart(document.getElementById('thresholdComparisonChart'), {
    type: 'bar',
    data: {
      labels: ['Accuracy','Precision','Recall','F1'],
      datasets: [
        barDataset('Threshold-Tuned (0.73)', [t.accuracy, t.precision, t.recall, t.f1], C.accent),
        barDataset('Argmax (0.5)',            [a.accuracy, a.precision, a.recall, a.f1], C.muted),
      ],
    },
    options: { responsive: true, scales: { y: yScale(), x: xScale() } },
  });
}

function chartClassDoughnut(id, cls, colors) {
  new Chart(document.getElementById(id), {
    type: 'doughnut',
    data: {
      labels: ['Precision','Recall','F1 Score'],
      datasets: [{ data: [cls.precision, cls.recall, cls.f1], backgroundColor: colors, borderColor: '#1a1d27', borderWidth: 3 }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom', labels: { color: C.muted } },
        tooltip: { callbacks: { label: ctx => (ctx.parsed * 100).toFixed(2) + '%' } },
      },
    },
  });
}

function chartLossHistory() {
  const epochs    = gnnMetrics.history.map((_, i) => i + 1);
  const trainLoss = gnnMetrics.history.map(h => h.train_loss);
  const valLoss   = gnnMetrics.history.map(h => h.argmax.loss);
  new Chart(document.getElementById('lossHistoryChart'), {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        { label: 'Train Loss', data: trainLoss, borderColor: C.blue,   backgroundColor: 'rgba(59,130,246,0.08)', borderWidth: 2, tension: 0.3, fill: true, pointRadius: 0 },
        { label: 'Val Loss',   data: valLoss,   borderColor: C.red,    backgroundColor: 'rgba(239,68,68,0.08)',  borderWidth: 2, tension: 0.3, fill: true, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { title: { display: true, text: 'Loss', color: C.muted }, ticks: { color: C.muted }, grid: { color: C.border } },
        x: { title: { display: true, text: 'Epoch', color: C.muted }, ticks: { color: C.muted }, grid: { color: C.border } },
      },
    },
  });
}

function chartLineHistory(id, label, accessor, color) {
  const epochs = gnnMetrics.history.map((_, i) => i + 1);
  new Chart(document.getElementById(id), {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [{ label, data: gnnMetrics.history.map(accessor), borderColor: color, backgroundColor: color + '14', borderWidth: 2, tension: 0.3, fill: true, pointRadius: 0 }],
    },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: yScale(), x: xScale() } },
  });
}

function chartPrecisionRecall() {
  const epochs = gnnMetrics.history.map((_, i) => i + 1);
  new Chart(document.getElementById('precisionRecallHistoryChart'), {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        { label: 'Precision', data: gnnMetrics.history.map(h => h.threshold_tuned.precision), borderColor: C.teal,  borderWidth: 2, tension: 0.3, fill: false, pointRadius: 0 },
        { label: 'Recall',    data: gnnMetrics.history.map(h => h.threshold_tuned.recall),    borderColor: C.amber, borderWidth: 2, tension: 0.3, fill: false, pointRadius: 0 },
      ],
    },
    options: { responsive: true, scales: { y: yScale(), x: xScale() } },
  });
}

function chartBarComparison(id, label, accessor) {
  const gnn = accessor(gnnMetrics.test);
  const rf  = accessor(baselineMetrics.random_forest);
  const xgb = accessor(baselineMetrics.xgboost);
  new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels: ['GNN','Random Forest','XGBoost'],
      datasets: [barDataset(label, [gnn, rf, xgb], [C.accent, C.muted, C.muted])],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: yScale(), x: xScale() },
    },
  });
}

function chartRocAuc() {
  new Chart(document.getElementById('rocAucComparisonChart'), {
    type: 'bar',
    data: {
      labels: ['GNN','Random Forest','XGBoost'],
      datasets: [barDataset('ROC-AUC', [0.9999, baselineMetrics.random_forest.roc_auc, baselineMetrics.xgboost.roc_auc], [C.accent, C.muted, C.muted])],
    },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: yScale(), x: xScale() } },
  });
}

function chartPie(id, labels, data, colors) {
  const total = data.reduce((a, b) => a + b, 0);
  new Chart(document.getElementById(id), {
    type: 'pie',
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors, borderColor: '#1a1d27', borderWidth: 3 }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom', labels: { color: C.muted } },
        tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.parsed.toLocaleString()} (${(ctx.parsed/total*100).toFixed(1)}%)` } },
      },
    },
  });
}

function chartRadar() {
  const c0 = classificationReport['0'], c1 = classificationReport['1'];
  new Chart(document.getElementById('radarChart'), {
    type: 'radar',
    data: {
      labels: ['Precision','Recall','F1 Score'],
      datasets: [
        { label: 'Normal (Class 0)', data: [c0.precision, c0.recall, c0['f1-score']], borderColor: C.green,  backgroundColor: 'rgba(34,197,94,0.15)',  borderWidth: 2 },
        { label: 'Attack (Class 1)', data: [c1.precision, c1.recall, c1['f1-score']], borderColor: C.red,    backgroundColor: 'rgba(239,68,68,0.15)',   borderWidth: 2 },
      ],
    },
    options: {
      responsive: true,
      scales: { r: { beginAtZero: true, max: 1, ticks: { callback: v => (v*100).toFixed(0)+'%', backdropColor: 'transparent', color: C.muted }, grid: { color: C.border }, angleLines: { color: C.border }, pointLabels: { color: C.muted } } },
      plugins: { legend: { position: 'bottom', labels: { color: C.muted } } },
    },
  });
}

function chartMacroWeighted() {
  const ma = classificationReport['macro avg'], wa = classificationReport['weighted avg'];
  new Chart(document.getElementById('macroWeightedChart'), {
    type: 'bar',
    data: {
      labels: ['Precision','Recall','F1 Score'],
      datasets: [
        barDataset('Macro Avg',    [ma.precision, ma.recall, ma['f1-score']], C.teal),
        barDataset('Weighted Avg', [wa.precision, wa.recall, wa['f1-score']], C.purple),
      ],
    },
    options: { responsive: true, scales: { y: yScale(), x: xScale() } },
  });
}

// ── Sidebar active link on scroll ──────────────────────────────────────────
function initSidebarHighlight() {
  const sections = document.querySelectorAll('.ids-section');
  const links    = document.querySelectorAll('.ids-nav-link');

  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const active = document.querySelector(`.ids-nav-link[data-section="${e.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });

  sections.forEach(s => observer.observe(s));

  links.forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const target = document.getElementById(link.dataset.section);
      if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
  });
}