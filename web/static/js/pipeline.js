// ── GNN-IDS Pipeline progress ─────────────────────────────────────────────
// Polls /api/pipeline (written by scripts/run_all.sh through scripts/pipeline_status.py).
// The dashboard restarts a few times during a run (after training, before the live
// demo), so failed polls show "Reconnecting…" and retry instead of giving up.

const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
const POLL_MS = 2000;
const LIVE_STAGES = { replay: 'Replayed InSDN attacks are flowing through the live IDS.', live_demo: 'Mininet attacks are running against the live IDS.' };
const ICONS = { pending: 'bi-circle', running: 'bi-arrow-repeat pl-spin', done: 'bi-check-circle-fill', failed: 'bi-x-circle-fill', skipped: 'bi-dash-circle' };
const STATE_TEXT = { running: 'Running', done: 'Completed', failed: 'Failed', interrupted: 'Interrupted', none: 'No run yet' };

let pinnedStage = null;   // stage whose log the user picked; null = follow the current stage
let clockOffset = 0;      // server_time - local time, for live timers
let last = null;

function fmtDuration(seconds) {
  if (seconds == null || seconds < 0) return '—';
  const s = Math.round(seconds), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${String(m).padStart(2, '0')}m` : m ? `${m}m ${String(sec).padStart(2, '0')}s` : `${sec}s`;
}
const now = () => Date.now() / 1000 + clockOffset;

function setBadge(state, text) {
  const badge = $('runBadge');
  badge.className = 'ids-status-badge pl-badge is-' + state;
  $('runState').textContent = text;
}

function renderStages(st) {
  const list = $('stageList');
  const selected = pinnedStage || st.log_stage;
  list.innerHTML = st.stages.map(s => {
    const took = s.started ? fmtDuration((s.finished || now()) - s.started) : '';
    const outputs = (s.outputs || []).map(o => {
      const cls = !o.exists ? 'is-missing' : (o.mtime >= st.started ? 'is-new' : '');
      const title = !o.exists ? 'not produced yet' : (o.mtime >= st.started ? 'updated in this run' : 'from an earlier run');
      return `<span class="pl-file ${cls}" title="${title}">${esc(o.path)}</span>`;
    }).join('');
    return `<li class="pl-stage is-${esc(s.state)} ${s.id === selected ? 'is-selected' : ''}" data-id="${esc(s.id)}" tabindex="0" role="button" aria-label="Show log of ${esc(s.title)}">
      <i class="bi ${ICONS[s.state] || 'bi-circle'} pl-icon"></i>
      <span class="pl-title">${esc(s.title)}</span>
      <span class="pl-time" data-started="${s.started || ''}" data-finished="${s.finished || ''}">${took}</span>
      <span class="pl-desc">${esc(s.description)}</span>
      ${s.note ? `<span class="pl-note">${s.state === 'skipped' ? 'Skipped · ' : ''}${esc(s.note)}</span>` : ''}
      ${outputs ? `<span class="pl-outputs">${outputs}</span>` : ''}
    </li>`;
  }).join('');
}

function renderLog(st) {
  const stage = st.stages.find(s => s.id === st.log_stage);
  $('logTitle').textContent = stage ? `Log · ${stage.title}` : 'Log';
  $('followBtn').hidden = !pinnedStage;
  const box = $('logBox');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  const text = (st.log_tail || []).join('\n');
  if (box.textContent !== text) {
    box.textContent = text || (stage && stage.state === 'pending' ? 'Not started yet.' : 'No output yet.');
    if (atBottom || !pinnedStage) box.scrollTop = box.scrollHeight;
  }
}

function renderResults(st) {
  const results = st.results || [];
  $('results').innerHTML = results.length ? results.map(r => {
    const isNew = st.started && r.mtime >= st.started;
    return `<div class="pl-result" title="${esc(r.file)}">
      <div class="pl-result-label">${isNew ? '<span class="pl-new-dot" title="updated in this run"></span>' : ''}${esc(r.label)}</div>
      <div class="pl-result-val">${esc(r.value)}</div>
      <div class="pl-result-detail">${esc(r.detail)}</div>
    </div>`;
  }).join('') : '<div class="pl-empty">No result files yet.</div>';
}

function renderOverview(st) {
  const stages = st.stages;
  const finished = stages.filter(s => ['done', 'failed', 'skipped'].includes(s.state)).length;
  const failed = stages.filter(s => s.state === 'failed').length;
  const current = stages.find(s => s.id === st.current);
  $('kpiStages').textContent = `${finished} / ${stages.length}`;
  $('kpiCurrent').textContent = current ? current.title : (st.state === 'running' ? 'Starting…' : '—');
  $('kpiFailed').textContent = failed;
  $('progressBar').style.width = `${(100 * finished) / stages.length}%`;
  $('kpiElapsed').dataset.started = st.started || '';
  $('kpiElapsed').dataset.finished = st.finished || '';
  $('runInfo').textContent = `Run ${st.run_id}` + (st.options ? ` · options: ${st.options}` : '') + (st.note ? ` · ${st.note}` : '');
  const live = st.state === 'running' && LIVE_STAGES[st.current];
  $('liveCallout').hidden = !live;
  if (live) $('liveCalloutTitle').textContent = LIVE_STAGES[st.current];
  setBadge(st.state, STATE_TEXT[st.state] || st.state);
}

function tickTimers() {
  const el = $('kpiElapsed');
  if (el.dataset.started) el.textContent = fmtDuration((Number(el.dataset.finished) || now()) - Number(el.dataset.started));
  document.querySelectorAll('.pl-time[data-started]').forEach(t => {
    if (t.dataset.started && !t.dataset.finished) t.textContent = fmtDuration(now() - Number(t.dataset.started));
  });
}

async function poll() {
  try {
    const url = '/api/pipeline' + (pinnedStage ? `?stage=${encodeURIComponent(pinnedStage)}` : '');
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(res.status);
    const st = await res.json();
    clockOffset = st.server_time - Date.now() / 1000;
    last = st;
    if (st.state === 'none') {
      $('noRun').hidden = false;
      $('overview').hidden = true;
      $('stageList').innerHTML = '';
      setBadge('none', STATE_TEXT.none);
      renderResults(st);
    } else {
      $('noRun').hidden = true;
      $('overview').hidden = false;
      renderOverview(st);
      renderStages(st);
      renderLog(st);
      renderResults(st);
    }
  } catch (e) {
    setBadge('offline', last && last.state === 'running' ? 'Reconnecting… (dashboard restarting)' : 'Dashboard offline');
  } finally {
    setTimeout(poll, POLL_MS);
  }
}

function pick(target) {
  const item = target.closest('.pl-stage');
  if (!item) return;
  pinnedStage = item.dataset.id === (last && last.current) ? null : item.dataset.id;
  if (last) { last.log_stage = item.dataset.id; renderStages(last); }
  $('logBox').textContent = 'Loading…';
  poll.now();
}
$('stageList').addEventListener('click', e => pick(e.target));
$('stageList').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(e.target); } });
$('followBtn').addEventListener('click', () => { pinnedStage = null; poll.now(); });

// Immediate refresh after a click, without starting a second polling loop.
let pending = null;
poll.now = () => {
  if (pending) return;
  pending = fetch('/api/pipeline' + (pinnedStage ? `?stage=${encodeURIComponent(pinnedStage)}` : ''), { cache: 'no-store' })
    .then(r => r.json()).then(st => { if (st.stages) { last = st; renderStages(st); renderLog(st); } })
    .catch(() => {}).finally(() => { pending = null; });
};

setInterval(tickTimers, 1000);
poll();
