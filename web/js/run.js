// Steps 2 (style) & 3 (run): prompt settings, hardware choice, starting a job, job progress.

import { $, $$, esc, api, toast, store, on, emit, formatDuration } from './ui.js';
import { getChapter } from './chapter.js';

const PRESETS = {
  default: 'masterpiece, best quality, anime style, highly detailed coloring, vibrant color harmony, crisp lineart',
  shonen: 'masterpiece, best quality, dynamic action anime, highly saturated vibrant shonen colors, crisp bold lineart, detailed illumination, colorful dramatic shading',
  pastel: 'masterpiece, best quality, soft pastel anime colors, shoujo aesthetic, gentle warm lighting, dreamy atmosphere, subtle watercolor shading, clean lineart',
  cinematic: 'masterpiece, best quality, cinematic dark atmosphere, dramatic moody lighting, deep contrast shadows, seinen aesthetic, hyperdetailed color depth, clean crisp lineart',
};
const DEFAULT_NEGATIVE = 'worst quality, low quality, blurry, artifact, oversaturated, deformed, greyscale, monochrome';

const HARDWARE = {
  gpu_t4: { label: 'GPU T4×2', note: 'Usually starts within minutes. About 1 min per page with both GPUs at 1024 px. Spreads are colorized whole.' },
  gpu_p100: { label: 'GPU P100', note: 'Usually starts within minutes. One GPU, so roughly half the speed of T4×2.' },
  tpu: { label: 'TPU v5e-8', note: '8 cores: fastest on long chapters, but the TPU queue can take hours. Spreads are split in two.' },
};

const ACTIVE = ['preparing', 'pushing', 'queued', 'running', 'downloading'];
const STEP_OF = { preparing: 'upload', pushing: 'queue', queued: 'queue', running: 'run', downloading: 'download' };
const STEPS = ['upload', 'queue', 'run', 'download'];

let quota = {};
let pollTimer = null;
let lastStep = 'upload';
let jobActive = false;
let lastStatus = null;

// ------------------------------------------------------------------ style settings

const SETTING_IDS = ['promptInput', 'negativeInput', 'cfgSlider', 'resolutionSelect', 'stepsSelect'];

function saveSettings() {
  const values = Object.fromEntries(SETTING_IDS.map(id => [id, $(`#${id}`).value]));
  store.set('mcs.style', JSON.stringify(values));
}

function loadSettings() {
  $('#promptInput').value = PRESETS.default;
  $('#negativeInput').value = DEFAULT_NEGATIVE;
  try {
    const saved = JSON.parse(store.get('mcs.style', '{}'));
    for (const id of SETTING_IDS) if (saved[id] != null) $(`#${id}`).value = saved[id];
  } catch { /* ignore broken storage */ }
  $('#cfgValue').textContent = Number($('#cfgSlider').value).toFixed(1);
  markPreset();
}

function markPreset() {
  const prompt = $('#promptInput').value.trim();
  $$('#presetChips [data-preset]').forEach(c => c.classList.toggle('active', prompt.startsWith(PRESETS[c.dataset.preset])));
  $$('#presetChips [data-append]').forEach(c => c.classList.toggle('active', prompt.includes(c.dataset.append)));
}

function bindStyle() {
  $('#presetChips').addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    const input = $('#promptInput');
    if (chip.dataset.preset) {
      // keep any appended extras (e.g. character tags) when switching the base preset
      const extras = $$('#presetChips [data-append]').map(c => c.dataset.append).filter(a => input.value.includes(a));
      input.value = [PRESETS[chip.dataset.preset], ...extras].join(', ');
    } else {
      const add = chip.dataset.append;
      input.value = input.value.includes(add)
        ? input.value.replace(`, ${add}`, '').replace(add, '').trim()
        : `${input.value.trim().replace(/,$/, '')}, ${add}`;
    }
    markPreset();
    saveSettings();
  });
  $('#promptInput').addEventListener('input', markPreset);
  $('#cfgSlider').addEventListener('input', e => { $('#cfgValue').textContent = Number(e.target.value).toFixed(1); });
  SETTING_IDS.forEach(id => $(`#${id}`).addEventListener('change', saveSettings));
  $('#resetStyleBtn').addEventListener('click', () => {
    $('#promptInput').value = PRESETS.default;
    $('#negativeInput').value = DEFAULT_NEGATIVE;
    $('#cfgSlider').value = 1;
    $('#cfgValue').textContent = '1.0';
    $('#resolutionSelect').value = '1024';
    $('#stepsSelect').value = '8';
    markPreset();
    saveSettings();
  });
}

// ------------------------------------------------------------------ hardware

const hardware = () => store.get('mcs.accelerator', 'gpu_t4') in HARDWARE ? store.get('mcs.accelerator', 'gpu_t4') : 'gpu_t4';

function renderHardware() {
  const acc = hardware();
  $$('#hardwareSeg button').forEach(b => {
    const selected = b.dataset.acc === acc;
    b.classList.toggle('active', selected);
    b.setAttribute('aria-checked', selected);
  });
  const q = quota[acc];
  const left = q ? ` ${q.remaining_hours.toFixed(1)} of ${Math.round(q.total_hours)} ${q.type} hours left this week.` : '';
  $('#hardwareNote').textContent = HARDWARE[acc].note + left;
}

async function loadQuota() {
  try {
    const data = await api('/api/accelerators');
    quota = Object.fromEntries((data.accelerators || []).filter(a => a.quota).map(a => [a.id, { ...a.quota, type: a.quota_type }]));
  } catch { /* quota is informational only */ }
  renderHardware();
}

// ------------------------------------------------------------------ encryption

// The switch works only once a key is saved under Accounts & tokens; the choice is remembered.
async function loadEncryption() {
  let ready = false;
  try {
    ready = !!(await api('/api/settings')).encryption_key_configured;
  } catch { /* server not reachable: leave it off */ }
  const toggle = $('#encryptToggle');
  toggle.disabled = !ready;
  toggle.checked = ready && store.get('mcs.encrypt', 'true') !== 'false';
  $('#encryptRow').title = ready
    ? 'Scans are encrypted before upload and results stay encrypted on Kaggle'
    : 'Set an encryption key under Accounts & tokens to enable this';
}

// ------------------------------------------------------------------ starting jobs

export function buildPayload(overrides = {}) {
  return {
    chapter_name: getChapter().name,
    max_dim: parseInt($('#resolutionSelect').value, 10),
    num_steps: parseInt($('#stepsSelect').value, 10),
    guidance_scale: parseFloat($('#cfgSlider').value),
    prompt: $('#promptInput').value.trim(),
    negative_prompt: $('#negativeInput').value.trim(),
    accelerator: hardware(),
    encrypt: $('#encryptToggle').checked && !$('#encryptToggle').disabled,
    ...overrides,
  };
}

export async function startJob(payload) {
  try {
    await api('/api/start', { method: 'POST', json: payload });
    jobActive = true;
    updateColorizeButton();
    showJobBar();
    poll();
    return true;
  } catch (err) {
    toast(err.message, 'bad');
    return false;
  }
}

export const isJobActive = () => jobActive;

function updateColorizeButton() {
  const btn = $('#colorizeBtn');
  const pages = getChapter().pages.length;
  btn.disabled = jobActive || !pages;
  btn.textContent = jobActive ? 'Job running…' : pages ? `Colorize ${pages} page${pages === 1 ? '' : 's'}` : 'Load a chapter first';
  emit('job:active', jobActive);
}

// ------------------------------------------------------------------ job progress

function showJobBar() {
  const bar = $('#jobBar');
  bar.hidden = false;
  bar.classList.remove('done', 'failed');
  $('#jobDismiss').hidden = true;
}

function renderJob(s) {
  const bar = $('#jobBar');
  const active = ACTIVE.includes(s.status);
  if (STEP_OF[s.status]) lastStep = STEP_OF[s.status];
  const current = s.status === 'completed' ? null : lastStep;

  const titles = {
    preparing: 'Uploading scans to Kaggle',
    pushing: 'Starting the Kaggle notebook',
    queued: 'Waiting for hardware',
    running: 'Colorizing',
    downloading: 'Downloading results',
    completed: `Done · ${s.completed_images} page${s.completed_images === 1 ? '' : 's'} colorized`,
    error: 'Failed',
  };
  $('#jobTitle').textContent = titles[s.status] || s.status;
  $('#jobMessage').textContent = s.status === 'error' ? (s.error || '') : (s.phase_message || '');
  $('#jobTime').textContent = formatDuration(s.elapsed_seconds);
  bar.classList.toggle('done', s.status === 'completed');
  bar.classList.toggle('failed', s.status === 'error');
  $('#jobDismiss').hidden = active;

  const idx = current ? STEPS.indexOf(current) : STEPS.length;
  $$('#jobSteps li').forEach((li, i) => {
    li.classList.toggle('done', i < idx);
    li.classList.toggle('current', i === idx);
  });

  const log = $('#jobLog');
  if (!log.hidden) {
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    log.textContent = (s.logs || []).join('\n');
    if (atBottom) log.scrollTop = log.scrollHeight;
  }
}

async function poll() {
  clearTimeout(pollTimer);
  let s;
  try {
    s = await api('/api/status');
  } catch {
    pollTimer = setTimeout(poll, 5000);
    return;
  }
  lastStatus = s;
  renderJob(s);
  if (ACTIVE.includes(s.status)) {
    pollTimer = setTimeout(poll, 3000);
    return;
  }
  if (jobActive) {
    jobActive = false;
    updateColorizeButton();
    if (s.status === 'completed') {
      toast(`Chapter colorized: ${s.completed_images} pages.`, 'ok');
      emit('job:done', s.chapter_name);
    } else if (s.status === 'error') {
      toast(`Colorization failed: ${s.error || 'see the log'}`, 'bad');
    }
  }
}

function bindJobBar() {
  $('#jobLogToggle').addEventListener('click', () => {
    const log = $('#jobLog');
    log.hidden = !log.hidden;
    if (!log.hidden && lastStatus) {
      log.textContent = (lastStatus.logs || []).join('\n');
      log.scrollTop = log.scrollHeight;
    }
  });
  $('#jobDismiss').addEventListener('click', () => { $('#jobBar').hidden = true; });
}

// ------------------------------------------------------------------ init

export async function initRun() {
  loadSettings();
  bindStyle();
  bindJobBar();

  $('#hardwareSeg').addEventListener('click', e => {
    const acc = e.target.closest('[data-acc]')?.dataset.acc;
    if (!acc) return;
    store.set('mcs.accelerator', acc);
    renderHardware();
  });
  renderHardware();
  loadQuota();

  loadEncryption();
  $('#encryptToggle').addEventListener('change', e => store.set('mcs.encrypt', String(e.target.checked)));
  on('account:changed', loadEncryption);

  $('#colorizeBtn').addEventListener('click', () => {
    const { pages } = getChapter();
    if (pages.length) startJob(buildPayload());
  });
  on('chapter:changed', updateColorizeButton);

  // Resume tracking a job that was started before this page was (re)loaded.
  try {
    const s = await api('/api/status');
    lastStatus = s;
    if (ACTIVE.includes(s.status)) {
      jobActive = true;
      showJobBar();
      renderJob(s);
      poll();
    }
  } catch { /* server not reachable yet */ }
  updateColorizeButton();
}
