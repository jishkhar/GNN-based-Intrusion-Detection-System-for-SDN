// Feature selection / reduction comparison charts (data: /api/feature-selection).
// Self-contained: the rest of the dashboard works without it.
(function () {
  const COLORS = ['#4f6ef7', '#22c55e', '#f59e0b', '#ef4444', '#14b8a6', '#a78bfa', '#ec4899', '#84cc16'];
  const MUTED = '#8b90a8', GRID = 'rgba(255,255,255,0.07)';

  function chartScale(values) {
    const lo = Math.min(...values), hi = Math.max(...values);
    const pad = Math.max(0.005, (hi - lo) * 0.1);
    return { min: Math.max(0, Math.floor((lo - pad) * 100) / 100), max: Math.min(1, hi + pad) };
  }

  function lineChart(id, table, labels) {
    const ks = [...new Set(table.results.filter(r => r.method !== 'all_features').map(r => r.k))].sort((a, b) => a - b);
    const base = table.results.find(r => r.method === 'all_features');
    const methods = [...new Set(table.results.map(r => r.method))].filter(m => m !== 'all_features');
    const datasets = methods.map((m, i) => ({
      label: labels[m] || m,
      data: ks.map(k => { const r = table.results.find(x => x.method === m && x.k === k); return r ? r.val_f1 : null; }),
      borderColor: COLORS[i % COLORS.length], backgroundColor: COLORS[i % COLORS.length],
      borderDash: ['pca', 'truncated_svd', 'fast_ica'].includes(m) ? [6, 3] : [],
      tension: 0.25, pointRadius: 3, spanGaps: true,
    }));
    datasets.push({
      label: 'All ' + table.n_features + ' features', data: ks.map(() => base.val_f1),
      borderColor: '#e8eaf2', borderDash: [2, 4], pointRadius: 0, borderWidth: 1.5,
    });
    const all = datasets.flatMap(d => d.data).filter(Number.isFinite);
    new Chart(document.getElementById(id), {
      type: 'line', data: { labels: ks, datasets },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'features / components kept', color: MUTED }, ticks: { color: MUTED }, grid: { color: GRID } },
          y: { ...chartScale(all), title: { display: true, text: 'validation F1 (axis zoomed)', color: MUTED },
               ticks: { color: MUTED, callback: v => (v * 100).toFixed(1) + '%' }, grid: { color: GRID } },
        },
        plugins: { tooltip: { callbacks: { label: c => `${c.dataset.label}: ${(c.parsed.y * 100).toFixed(2)}%` } } },
      },
    });
  }

  function barChart(id, table, labels) {
    // Best (by validation F1) k for each method.
    const best = {};
    table.results.forEach(r => { if (!best[r.method] || r.val_f1 > best[r.method].val_f1) best[r.method] = r; });
    const rows = Object.values(best).sort((a, b) => b.val_f1 - a.val_f1);
    const win = table.best.method;
    const all = rows.flatMap(r => [r.val_f1, r.test_f1]);
    new Chart(document.getElementById(id), {
      type: 'bar',
      data: {
        labels: rows.map(r => `${labels[r.method] || r.method}${r.method === 'all_features' ? '' : ' (' + r.k + ')'}`),
        datasets: [
          { label: 'Validation F1', data: rows.map(r => r.val_f1), borderRadius: 4,
            backgroundColor: rows.map(r => r.method === win ? '#22c55e' : '#4f6ef7') },
          { label: 'Test F1', data: rows.map(r => r.test_f1), borderRadius: 4, backgroundColor: 'rgba(245,158,11,0.75)' },
        ],
      },
      options: {
        indexAxis: 'y', responsive: true,
        scales: {
          x: { ...chartScale(all), ticks: { color: MUTED, callback: v => (v * 100).toFixed(1) + '%' }, grid: { color: GRID } },
          y: { ticks: { color: MUTED }, grid: { display: false } },
        },
        plugins: { tooltip: { callbacks: { label: c => `${c.dataset.label}: ${(c.parsed.x * 100).toFixed(2)}%` } } },
      },
    });
  }

  function bestCard(id, table, labels) {
    const b = table.best;
    const kept = b.type === 'reduction'
      ? `${b.k} components (from ${table.n_features})`
      : `${b.k} of ${table.n_features} features`;
    const items = [
      ['Method', `${labels[b.method] || b.method} <span class="text-secondary">(${b.type})</span>`],
      ['Keeps', kept],
      ['Validation F1', (b.val_f1 * 100).toFixed(2) + '%'],
      ['Test F1', (b.test_f1 * 100).toFixed(2) + '%'],
    ];
    if (b.feature_names) items.push(['Features', b.feature_names.join(', ')]);
    const ap = table.applied;
    if (ap) items.push(['Used by the GNN', `${labels[ap.method] || ap.method}: ${ap.feature_names.join(', ')}`]);
    document.getElementById(id).innerHTML = items.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  }

  async function init() {
    let data;
    try { data = await (await fetch('/api/feature-selection')).json(); } catch (e) { data = { error: String(e) }; }
    if (!data || data.error || !data.tables) { document.getElementById('fsEmpty').hidden = false; return; }
    document.getElementById('fsBody').hidden = false;
    document.getElementById('fsTol').textContent = (data.tolerance * 100).toFixed(1);
    const { edge, node } = data.tables;
    bestCard('fsBestEdge', edge, data.labels);
    bestCard('fsBestNode', node, data.labels);
    lineChart('fsEdgeLine', edge, data.labels);
    lineChart('fsNodeLine', node, data.labels);
    barChart('fsEdgeBar', edge, data.labels);
    barChart('fsNodeBar', node, data.labels);
  }

  // Charts need a visible container to size themselves: wait for the main dashboard to finish loading.
  document.addEventListener('DOMContentLoaded', () => {
    const main = document.getElementById('mainContent');
    const t = setInterval(() => {
      if (!main || main.style.display !== 'none') { clearInterval(t); init(); }
    }, 100);
    setTimeout(() => clearInterval(t), 15000);
  });
})();
