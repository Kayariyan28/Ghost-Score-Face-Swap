const $ = (id) => document.getElementById(id);

const state = { source: null, target: null };

function bindUpload(inputId, previewId, key) {
  const input = $(inputId);
  const preview = $(previewId);
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    if (!file) return;
    state[key] = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.innerHTML = `<img src="${e.target.result}" alt="">`;
    };
    reader.readAsDataURL(file);
  });
}

bindUpload('sourceInput', 'sourcePreview', 'source');
bindUpload('targetInput', 'targetPreview', 'target');

function bindSlider(inputId, valueId) {
  const input = $(inputId);
  const out = $(valueId);
  const sync = () => { out.textContent = Number(input.value).toFixed(2); };
  input.addEventListener('input', sync);
  sync();
  return input;
}
const fidelityInput = bindSlider('fidelity', 'fidelityValue');
const sharpenInput = bindSlider('sharpen', 'sharpenValue');
const detailInput = bindSlider('detail', 'detailValue');

function currentMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : 'photoshop';
}
function syncModeSliders() {
  const mode = currentMode();
  document.querySelectorAll('.slider-opt[data-mode]').forEach((el) => {
    el.classList.toggle('active', el.dataset.mode === mode);
  });
}
document.querySelectorAll('input[name="mode"]').forEach((r) =>
  r.addEventListener('change', syncModeSliders),
);
syncModeSliders();

const statusEl = $('status');
function setStatus(msg, kind = 'info', spinner = false) {
  statusEl.className = `status show ${kind}`;
  statusEl.innerHTML = (spinner ? '<span class="spinner"></span>' : '') + `<span>${msg}</span>`;
}
function clearStatus() {
  statusEl.className = 'status';
  statusEl.innerHTML = '';
}

function classify(value, ranges) {
  // ranges = { good: [lo, hi], warn: [lo, hi] } — value lands in 'bad' if neither.
  if (value == null || isNaN(value)) return '';
  if (value >= ranges.good[0] && value <= ranges.good[1]) return 'good';
  if (value >= ranges.warn[0] && value <= ranges.warn[1]) return 'warn';
  return 'bad';
}

function fmt(value, digits = 2, suffix = '') {
  if (value == null || isNaN(value)) return '—';
  if (value === Infinity) return '∞';
  return Number(value).toFixed(digits) + suffix;
}

function fmtPose(p) {
  if (!p) return '—';
  const s = (v) => (v >= 0 ? '+' : '') + v.toFixed(1) + '°';
  return `${s(p.yaw)} / ${s(p.pitch)} / ${s(p.roll)}`;
}

function renderPro(pro) {
  const panel = $('proPanel');
  panel.classList.remove('hidden');
  $('proRouteBadge').textContent = pro.chosen_route || '—';
  $('proSrcPose').textContent = fmtPose(pro.source_pose);
  $('proTgtPose').textContent = fmtPose(pro.target_pose);
  $('proPoseDiff').textContent = pro.pose_diff_deg != null
    ? pro.pose_diff_deg.toFixed(1) + '°'
    : '—';

  // Ghost-detector metrics (highest-signal block on this panel).
  const g = pro.ghost || {};
  const setG = (id, value, cls) => {
    const el = $(id);
    if (!el) return;
    el.textContent = value;
    el.classList.remove('good', 'warn', 'bad');
    if (cls) el.classList.add(cls);
  };
  const ghostScoreCls = (v) =>
    v == null ? '' : (v < -0.15 ? 'good' : (v < 0 ? 'warn' : 'bad'));
  setG('proGhostScore', g.ghost_score == null ? '—' : g.ghost_score.toFixed(3),
       ghostScoreCls(g.ghost_score));
  setG('proArcSrc', g.arc_to_source == null ? '—' : g.arc_to_source.toFixed(3),
       g.arc_to_source != null && g.arc_to_source > 0.6 ? 'good'
       : g.arc_to_source != null && g.arc_to_source > 0.4 ? 'warn' : '');
  setG('proArcTgt', g.arc_to_target == null ? '—' : g.arc_to_target.toFixed(3));
  setG('proSeam', g.seam_gradient_energy == null
       ? '—' : g.seam_gradient_energy.toFixed(1),
       g.seam_gradient_energy != null && g.seam_gradient_energy < 28 ? 'good'
       : g.seam_gradient_energy != null && g.seam_gradient_energy < 40 ? 'warn' : '');
  setG('proDouble', g.double_edge_score == null
       ? '—' : g.double_edge_score.toFixed(3),
       g.double_edge_score != null && g.double_edge_score < 0.2 ? 'good'
       : g.double_edge_score != null && g.double_edge_score < 0.35 ? 'warn' : '');
  setG('proRoutingAdvice', (g.routing || '—').replace('_', ' '),
       (g.routing === 'ok') ? 'good' : (g.routing ? 'warn' : ''));

  const cand = $('proCandidates');
  cand.innerHTML = '';
  (pro.candidates || []).forEach((c, idx) => {
    const s = c.scores || {};
    const row = document.createElement('div');
    row.className = 'candidate-row' + (idx === 0 ? ' winner' : '');
    row.innerHTML = `
      <span class="route">${c.route}${idx === 0 ? ' ✓' : ''}</span>
      <span class="stat">ArcFace ${(s.arcface_cosine == null) ? '—' : Number(s.arcface_cosine).toFixed(3)}</span>
      <span class="stat">SSIM ${(s.ssim == null) ? '—' : Number(s.ssim).toFixed(3)}</span>
      <span class="stat">${s.elapsed_ms == null ? '—' : s.elapsed_ms + ' ms'}</span>
    `;
    cand.appendChild(row);
  });
}

function renderMetrics(m, retried) {
  const arc = m.arcface_cosine;
  const ssim = m.ssim;
  const de = m.delta_e_mean;
  const lm = m.landmark_rmse_norm;
  const psnr = m.psnr;
  const ms = m.elapsed_ms;

  const setM = (id, value, cls) => {
    const el = $(id);
    el.textContent = value;
    el.classList.remove('good', 'warn', 'bad');
    if (cls) el.classList.add(cls);
  };

  setM('m-arc',  fmt(arc, 3), classify(arc,  { good: [0.6, 1.1], warn: [0.4, 0.6] }));
  setM('m-ssim', fmt(ssim, 3), classify(ssim, { good: [0.85, 1.1], warn: [0.7, 0.85] }));
  // ΔE00: lower is better, so we *invert* the range logic.
  let deCls = '';
  if (!isNaN(de)) deCls = (de <= 5) ? 'good' : (de <= 10 ? 'warn' : 'bad');
  setM('m-de', fmt(de, 2), deCls);
  // landmark RMSE: lower is better.
  let lmCls = '';
  if (!isNaN(lm)) lmCls = (lm <= 0.02) ? 'good' : (lm <= 0.05 ? 'warn' : 'bad');
  setM('m-lm', fmt(lm, 4), lmCls);
  setM('m-psnr', fmt(psnr, 1), classify(psnr, { good: [25, 99], warn: [20, 25] }));
  setM('m-time', ms != null ? String(ms) : '—');

  // ArcFace verdict hint
  const hint = m.arcface_verdict === 'match' ? '✓ same identity'
            : m.arcface_verdict === 'partial' ? '~ partial match'
            : m.arcface_verdict === 'drift' ? '✗ identity drift'
            : 'unknown';
  $('m-arc-hint').textContent = hint;

  $('retriedBadge').classList.toggle('hidden', !retried);
}

$('swapBtn').addEventListener('click', async () => {
  if (!state.source || !state.target) {
    setStatus('Please upload both a base face and a target body image.', 'error');
    return;
  }

  const fd = new FormData();
  fd.append('source', state.source);
  fd.append('target', state.target);
  fd.append('mode', currentMode());
  fd.append('enhance', $('enhance').checked ? 'true' : 'false');
  fd.append('swap_all', $('swapAll').checked ? 'true' : 'false');
  fd.append('hd', $('hd').checked ? 'true' : 'false');
  fd.append('preserve_hair', $('preserveHair').checked ? 'true' : 'false');
  fd.append('preserve_source_tone', $('preserveSourceTone').checked ? 'true' : 'false');
  fd.append('fidelity', fidelityInput.value);
  fd.append('sharpen', sharpenInput.value);
  fd.append('detail', detailInput.value);

  const btn = $('swapBtn');
  btn.disabled = true;
  btn.textContent = 'Processing…';
  setStatus('Detecting faces, swapping, and enhancing…', 'info', true);
  $('resultPanel').classList.add('hidden');

  const t0 = performance.now();
  try {
    const res = await fetch('/api/swap', { method: 'POST', body: fd });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      setStatus(data.detail || `Request failed (${res.status})`, 'error');
      return;
    }

    const url = data.url + '?t=' + Date.now();
    $('resultImg').src = url;
    $('downloadBtn').href = url;
    $('resultPanel').classList.remove('hidden');
    const sec = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Done in ${sec}s.`, 'success');

    if (data.metrics) renderMetrics(data.metrics, data.retried);
    if (data.pro) renderPro(data.pro);
    else $('proPanel').classList.add('hidden');
    $('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    setStatus('Network error: ' + (err && err.message ? err.message : err), 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Swap Face';
  }
});
