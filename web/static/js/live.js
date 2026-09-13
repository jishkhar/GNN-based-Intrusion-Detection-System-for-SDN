// ── GNN-IDS Live Monitor ─────────────────────────────────────────────────
// Status/topology/rules are polled; alerts, mitigations and per-window
// summaries stream in over Server-Sent Events (/api/live/stream).

const C = { text: '#e8eaf2', muted: '#8b90a8', border: 'rgba(255,255,255,0.07)', blue: '#3b82f6',
            amber: '#f59e0b', red: '#ef4444', green: '#22c55e', accent: '#4f6ef7', surface: '#22263a' };
Chart.defaults.color = C.muted;
Chart.defaults.borderColor = C.border;
Chart.defaults.font.family = "'Inter','Segoe UI',system-ui,sans-serif";

const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
const fmtTime = ts => new Date(ts * 1000).toLocaleTimeString();
const pct = v => (v * 100).toFixed(1) + '%';
const MAX_POINTS = 120;

try { $('apiKey').value = localStorage.getItem('idsApiKey') || ''; } catch (e) { /* storage blocked */ }

// ── timeline chart ─────────────────────────────────────────────────────────
const timeline = new Chart($('timelineChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Attack confidence', data: [], borderColor: C.red, backgroundColor: 'rgba(239,68,68,0.1)', yAxisID: 'y', tension: 0.25, pointRadius: 0, fill: true },
    { label: 'Active flows', data: [], borderColor: C.blue, yAxisID: 'y1', tension: 0.25, pointRadius: 0 },
  ] },
  options: {
    animation: false, responsive: true, interaction: { mode: 'index', intersect: false },
    scales: {
      y: { min: 0, max: 1, ticks: { callback: v => (v * 100) + '%' }, title: { display: true, text: 'confidence' } },
      y1: { position: 'right', beginAtZero: true, grid: { drawOnChartArea: false }, title: { display: true, text: 'flows' } },
      x: { ticks: { maxTicksLimit: 8 } },
    },
  },
});

function pushWindow(ev) {
  const d = timeline.data;
  d.labels.push(fmtTime(ev.timestamp));
  d.datasets[0].data.push(ev.confidence);
  d.datasets[1].data.push(ev.active_flows);
  if (d.labels.length > MAX_POINTS) { d.labels.shift(); d.datasets.forEach(s => s.data.shift()); }
  timeline.update('none');
}

// ── topology (d3 force layout) ─────────────────────────────────────────────
const svg = d3.select('#topology');
const gLinks = svg.append('g'), gNodes = svg.append('g'), gLabels = svg.append('g');
const sim = d3.forceSimulation().force('charge', d3.forceManyBody().strength(-60))
  .force('link', d3.forceLink().id(d => d.id).distance(45)).force('collide', d3.forceCollide(9)).alphaDecay(0.05);
const roleColor = { attacker: C.red, victim: C.amber, host: C.blue };
let nodeState = new Map();

function drawTopology(topo) {
  const { width, height } = svg.node().getBoundingClientRect();
  sim.force('center', d3.forceCenter(width / 2, height / 2)).force('x', d3.forceX(width / 2).strength(0.05)).force('y', d3.forceY(height / 2).strength(0.07));
  // Keep previous positions so the layout doesn't jump every refresh.
  const nodes = topo.nodes.slice(0, 250).map(n => Object.assign(nodeState.get(n.id) || {}, n));
  const ids = new Set(nodes.map(n => n.id));
  const links = topo.links.filter(l => ids.has(l.source) && ids.has(l.target)).map(l => ({ ...l }));
  nodeState = new Map(nodes.map(n => [n.id, n]));

  gLinks.selectAll('line').data(links).join('line').attr('stroke-width', l => Math.min(4, 0.5 + Math.log10(1 + l.packets)));
  gNodes.selectAll('circle').data(nodes, d => d.id).join('circle')
    .attr('r', d => d.role === 'host' ? 5 : 8).attr('fill', d => roleColor[d.role] || C.blue)
    .attr('stroke', '#0f1117').attr('stroke-width', 1.5)
    .selectAll('title').data(d => [d]).join('title').text(d => `${d.id}\nattacker score ${pct(d.score)}`);
  const labelled = nodes.filter(n => n.role !== 'host' || nodes.length <= 30);
  gLabels.selectAll('text').data(labelled, d => d.id).join('text').text(d => d.id);

  sim.nodes(nodes).on('tick', () => {
    gLinks.selectAll('line').attr('x1', l => l.source.x).attr('y1', l => l.source.y).attr('x2', l => l.target.x).attr('y2', l => l.target.y);
    gNodes.selectAll('circle').attr('cx', d => d.x).attr('cy', d => d.y);
    gLabels.selectAll('text').attr('x', d => d.x + 9).attr('y', d => d.y + 3);
  });
  sim.force('link').links(links);
  sim.alpha(0.4).restart();
  const shown = topo.nodes.length > 250 ? ` (showing 250 of ${topo.nodes.length})` : '';
  $('topologyCaption').textContent = `${topo.nodes.length} hosts, ${topo.links.length} host pairs${shown}. Node colour = GNN role, hover for attacker score.`;
}

// ── status / tables ────────────────────────────────────────────────────────
function setConnected(ok, text) {
  $('connBadge').classList.toggle('is-offline', !ok);
  $('connText').textContent = text;
}

function renderStatus(s) {
  const p = s.last_prediction;
  const level = p ? p.decision.level : '—';
  const kpi = $('levelKpi');
  kpi.className = 'ids-kpi ids-kpi--level-' + String(level).toLowerCase();
  $('kpiLevel').textContent = level;
  $('kpiLevelSub').textContent = p ? `confidence ${pct(p.confidence)}` : 'waiting for traffic';
  $('kpiFlows').textContent = p ? p.num_flows : 0;
  $('kpiHosts').textContent = p ? p.num_hosts : 0;
  $('kpiRules').textContent = s.active_rules;
  $('kpiAlerts').textContent = s.alerts;
  $('predType').textContent = p ? p.attack_type : '—';
  $('thrAttack').textContent = `${pct(s.thresholds.attack)} (model-tuned ${pct(s.thresholds.model_window)})`;
  $('predAttackers').textContent = p && p.decision.attackers.length
    ? `${p.decision.attackers.slice(0, 5).join(', ')}${p.decision.attackers.length > 5 ? ` +${p.decision.attackers.length - 5}` : ''}` : 'none';
  $('uploads').textContent = `${s.uploads} (${s.flow_entries_per_s.toFixed(1)} entries/s)`;
  setConnected(true, s.mitigation_enabled ? 'IDS live · mitigation on' : 'IDS live · mitigation off');
}

const LEVEL_BADGE = { ATTACK: 'ids-badge--red', SUSPICIOUS: 'ids-badge--amber', BENIGN: 'ids-badge--green' };

function alertRow(a, fresh) {
  const attackers = a.num_attackers > a.attackers.length ? `${a.attackers.slice(0, 5).join(', ')} +${a.num_attackers - 5}` : a.attackers.slice(0, 5).join(', ');
  return `<tr class="${fresh ? 'live-new' : ''}"><td>${fmtTime(a.timestamp)}</td>
    <td><span class="ids-badge ${LEVEL_BADGE[a.level] || ''}">${esc(a.level)}</span></td><td>${esc(a.attack_type)}</td>
    <td>${pct(a.confidence)}</td><td>${esc(attackers) || '—'}</td><td>${esc(a.victims.join(', ')) || '—'}</td>
    <td>${esc((a.actions || []).join(', ')) || '—'}</td></tr>`;
}

let alertsHtml = [];
function addAlert(a, fresh = true) {
  alertsHtml.unshift(alertRow(a, fresh));
  alertsHtml = alertsHtml.slice(0, 100);
  $('alertRows').innerHTML = alertsHtml.join('');
}

function renderRules(rules) {
  if (!rules.length) { $('ruleRows').innerHTML = '<tr class="ids-row--muted"><td colspan="7">No active rules</td></tr>'; return; }
  const now = Date.now() / 1000;
  $('ruleRows').innerHTML = rules.map(r => `<tr><td>${esc(r.rule_id)}</td><td>${esc(r.action)}</td>
    <td>${esc(Object.entries(r.match).map(([k, v]) => `${k}=${v}`).join(' '))}</td><td>${esc(r.attack_type)}</td><td>${esc(r.source)}</td>
    <td>${r.expires_at ? Math.max(0, Math.round(r.expires_at - now)) + ' s' : '—'}</td>
    <td><button class="btn btn-sm btn-outline-light" data-unblock="${esc(r.rule_id)}">Unblock</button></td></tr>`).join('');
}

function renderLog(entries) {
  if (!entries.length) return;
  $('logRows').innerHTML = entries.slice(0, 100).map(e => `<tr><td>${fmtTime(e.ts)}</td><td>${esc(e.event)}</td><td>${esc(e.rule_id || '')}</td>
    <td>${esc(e.match ? Object.entries(e.match).map(([k, v]) => `${k}=${v}`).join(' ') : '')}</td><td>${esc(e.reason || '')}</td></tr>`).join('');
}

// ── data loading ───────────────────────────────────────────────────────────
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(body.detail || r.statusText), { status: r.status });
  return body;
}

async function refresh() {
  try {
    const [status, topo, rules, log, latency] = await Promise.all([
      getJSON('/api/live/status'), getJSON('/api/topology'), getJSON('/api/mitigations'),
      getJSON('/api/mitigations/log?limit=100'), getJSON('/api/live/latency'),
    ]);
    $('offlineNotice').hidden = true;
    renderStatus(status);
    drawTopology(topo);
    renderRules(rules);
    renderLog(log);
    $('kpiLatency').textContent = latency.count ? latency.p90_ms.toFixed(1) + ' ms' : '—';
  } catch (err) {
    if (err.status === 503) {
      $('offlineNotice').hidden = false;
      $('offlineReason').textContent = err.message;
      setConnected(false, 'Live IDS offline');
    } else {
      setConnected(false, 'API unreachable');
    }
  }
}

async function loadAlerts() {
  try {
    const alerts = await getJSON('/api/alerts?limit=100');
    alertsHtml = alerts.map(a => alertRow(a, false));
    if (alertsHtml.length) $('alertRows').innerHTML = alertsHtml.join('');
  } catch (e) { /* offline notice handled in refresh() */ }
}

function connectStream() {
  const es = new EventSource('/api/live/stream');
  es.onmessage = msg => {
    const ev = JSON.parse(msg.data);
    if (ev.type === 'window') pushWindow(ev);
    else if (ev.type === 'alert') addAlert(ev);
    else if (ev.type === 'mitigation') refresh();
  };
  es.onerror = () => { es.close(); setTimeout(connectStream, 3000); };
}

// ── manual block / unblock ─────────────────────────────────────────────────
function authHeaders() {
  const key = $('apiKey').value.trim();
  try { localStorage.setItem('idsApiKey', key); } catch (e) { /* ignore */ }
  return key ? { 'Content-Type': 'application/json', 'X-API-Key': key } : { 'Content-Type': 'application/json' };
}

$('blockForm').addEventListener('submit', async e => {
  e.preventDefault();
  try {
    const rule = await getJSON('/api/mitigations/block', { method: 'POST', headers: authHeaders(),
      body: JSON.stringify({ ip: $('blockIp').value.trim(), action: $('blockAction').value }) });
    $('formMsg').textContent = `Queued ${rule.rule_id}; installed on the switches at the next controller poll.`;
    refresh();
  } catch (err) { $('formMsg').textContent = `Failed: ${err.message}`; }
});

$('ruleRows').addEventListener('click', async e => {
  const id = e.target.dataset.unblock;
  if (!id) return;
  try {
    await getJSON('/api/mitigations/unblock', { method: 'POST', headers: authHeaders(), body: JSON.stringify({ rule_id: id }) });
    refresh();
  } catch (err) { $('formMsg').textContent = `Unblock failed: ${err.message}`; }
});

refresh();
loadAlerts();
connectStream();
setInterval(refresh, 3000);
