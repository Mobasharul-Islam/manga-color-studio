// Models & LoRAs: the Studio's model/LoRA controls and the Library page.
// Every item can be cached in its own private Kaggle dataset (see app/library.py).

import { $, $$, esc, linkify, formatBytes, api, toast, confirmDialog, dropTarget } from './ui.js';

let lib = null;           // last /api/library snapshot
let pollTimer = null;

const ICON_TRASH = '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>';

function statusBadge(item) {
  if (item.status === 'caching') return '<span class="badge busy">Caching…</span>';
  if (item.status === 'failed') return '<span class="badge bad">Caching failed</span>';
  if (item.status === 'cached') {
    return item.legacy
      ? `<span class="badge ok" title="In the older shared dataset ${esc(item.dataset)}">Cached (old)</span>`
      : `<span class="badge ok" title="${esc(item.dataset)}">Cached</span>`;
  }
  return item.usable
    ? '<span class="badge" title="Downloaded from Hugging Face during each run">Not cached</span>'
    : '<span class="badge warn" title="Must be cached on Kaggle before it can be used">Needs caching</span>';
}

const canCache = item => item.status !== 'caching' && (item.status !== 'cached' || item.legacy);
const isBusy = d => !!d && (!!d.tasks.current || d.tasks.pending > 0 || [...d.models, ...d.loras].some(i => i.status === 'caching'));

export async function loadLibrary(fromKaggle = false) {
  try {
    lib = fromKaggle
      ? await api('/api/library/refresh', { method: 'POST' })
      : await api('/api/library');
  } catch (err) {
    if (fromKaggle) return loadLibrary(false);  // not connected yet: show the local library
    return;
  }
  renderStudio();
  renderLibraryPage();
  clearTimeout(pollTimer);
  if (isBusy(lib)) pollTimer = setTimeout(() => loadLibrary(false), 4000);
}

// ------------------------------------------------------------------ Studio sidebar

function renderStudio() {
  const sel = $('#modelSelect');
  sel.innerHTML = lib.models.map(m =>
    `<option value="${esc(m.id)}" ${m.id === lib.active_model ? 'selected' : ''}>${esc(m.name)}</option>`).join('');
  const active = lib.models.find(m => m.id === lib.active_model);
  $('#modelStatus').innerHTML = active ? statusBadge(active) : '';

  const list = $('#loraList');
  if (!lib.loras.length) {
    list.innerHTML = '<div class="lora-empty">No LoRAs yet. Add some in the Library.</div>';
    return;
  }
  list.innerHTML = lib.loras.map(l => {
    const blocked = !l.usable && !l.enabled;
    return `
      <div class="lora-row ${l.enabled ? '' : 'off'}" data-id="${esc(l.id)}">
        <input type="checkbox" class="switch" data-act="toggle" ${l.enabled ? 'checked' : ''} ${blocked ? 'disabled title="Cache it in the Library first"' : ''}>
        <span class="lora-name" title="${esc(l.source)}">${esc(l.name)}</span>
        ${l.usable ? (l.is_turbo ? '<span class="badge">Turbo</span>' : '<span></span>') : statusBadge(l)}
        <div class="lora-weight">
          <input type="range" class="range" data-act="scale" min="0" max="1.5" step="0.05" value="${l.scale}">
          <span class="value mono">${Number(l.scale).toFixed(2)}</span>
        </div>
      </div>`;
  }).join('');
}

function bindStudio() {
  $('#modelSelect').addEventListener('change', async e => {
    try {
      lib = await api('/api/library/models/active', { json: { id: e.target.value }, method: 'POST' });
      renderStudio();
      renderLibraryPage();
    } catch (err) {
      toast(err.message, 'bad');
    }
  });

  const list = $('#loraList');
  list.addEventListener('change', async e => {
    const row = e.target.closest('.lora-row');
    if (!row) return;
    const lora = lib.loras.find(l => l.id === row.dataset.id);
    if (e.target.dataset.act === 'toggle') {
      lora.enabled = e.target.checked;
      row.classList.toggle('off', !lora.enabled);
      api('/api/loras/update', { method: 'POST', json: { id: lora.id, enabled: lora.enabled } }).catch(err => toast(err.message, 'bad'));
    } else if (e.target.dataset.act === 'scale') {
      api('/api/loras/update', { method: 'POST', json: { id: lora.id, scale: lora.scale } }).catch(err => toast(err.message, 'bad'));
    }
  });
  list.addEventListener('input', e => {
    if (e.target.dataset.act !== 'scale') return;
    const row = e.target.closest('.lora-row');
    const lora = lib.loras.find(l => l.id === row.dataset.id);
    lora.scale = parseFloat(e.target.value);
    $('.value', row).textContent = lora.scale.toFixed(2);
  });
}

// ------------------------------------------------------------------ Library page

function itemRow(item, kind) {
  const isActiveModel = kind === 'model' && item.id === lib.active_model;
  const actions = [];
  if (kind === 'model' && !isActiveModel) actions.push(`<button class="btn btn-ghost btn-xs" data-act="use">Use</button>`);
  if (canCache(item)) actions.push(`<button class="btn btn-secondary btn-xs" data-act="cache">${item.status === 'failed' ? 'Retry' : 'Cache'}</button>`);
  const lastModel = kind === 'model' && lib.models.length < 2;
  actions.push(`<button class="icon-btn danger" data-act="delete" title="Delete" ${item.status === 'caching' || lastModel ? 'disabled' : ''}>${ICON_TRASH}</button>`);

  const sub = [isActiveModel ? 'In use' : '', item.source].filter(Boolean).join(' · ');
  const error = item.error && item.status !== 'caching'
    ? `<div class="row-error">${item.status === 'failed' ? '' : 'Last caching attempt failed: '}${linkify(item.error)}</div>` : '';
  return `
    <div class="row" data-id="${esc(item.id)}">
      <div class="row-title"><strong>${esc(item.name)}</strong><span title="${esc(item.source)}">${esc(sub)}</span></div>
      <span class="row-size">${formatBytes(item.size_bytes)}</span>
      ${statusBadge(item)}
      <div class="row-actions">${actions.join('')}</div>
      ${error}
    </div>`;
}

function renderLibraryPage() {
  $('#modelTable').innerHTML = lib.models.map(m => itemRow(m, 'model')).join('');
  $('#loraTable').innerHTML = lib.loras.length
    ? lib.loras.map(l => itemRow(l, 'lora')).join('')
    : '<div class="table-empty">No LoRAs yet.</div>';

  const legacy = lib.legacy || [];
  $('#legacyPanel').hidden = !legacy.length;
  $('#legacyTable').innerHTML = legacy.map(d => `
    <div class="row" data-ref="${esc(d.ref)}" data-in-use="${d.in_use ? '1' : ''}">
      <div class="row-title"><strong>${esc(d.ref)}</strong><span>${esc(d.note)}${d.in_use ? ' · still used until re-cached' : ''}</span></div>
      <span class="row-size">${formatBytes(d.size_bytes)}</span>
      <span></span>
      <div class="row-actions"><button class="icon-btn danger" data-act="delete-legacy" title="Delete">${ICON_TRASH}</button></div>
    </div>`).join('');

  const act = $('#libraryActivity');
  const t = lib.tasks;
  const lastError = (t.history || [])[0] && !t.history[0].ok ? t.history[0] : null;
  if (t.current) {
    const secs = Math.round(Date.now() / 1000 - t.current.started);
    act.className = 'activity';
    act.innerHTML = `<span class="spinner"></span><span>${esc(t.current.label)} · ${Math.floor(secs / 60)}m ${secs % 60}s${t.pending ? ` · ${t.pending} more queued` : ''}</span>`;
    act.hidden = false;
  } else if (lastError) {
    act.className = 'activity error';
    act.innerHTML = `<span>${esc(lastError.label)} failed: ${linkify(lastError.error || '')}</span>`;
    act.hidden = false;
  } else {
    act.hidden = true;
  }
}

async function onRowAction(e) {
  const btn = e.target.closest('[data-act]');
  if (!btn) return;
  const row = btn.closest('.row');
  try {
    if (btn.dataset.act === 'delete-legacy') {
      const ref = row.dataset.ref;
      const extra = row.dataset.inUse ? '\n\nSome items still use it; until you cache them again, runs download them from Hugging Face.' : '';
      if (!await confirmDialog('Delete dataset?', `This permanently deletes ${ref} from your Kaggle account.${extra}`, 'Delete', true)) return;
      const [owner, slug] = ref.split('/');
      await api(`/api/library/legacy/${encodeURIComponent(owner)}/${encodeURIComponent(slug)}`, { method: 'DELETE' });
      toast(`Deleting ${ref}…`);
      return loadLibrary();
    }

    const item = [...lib.models, ...lib.loras].find(i => i.id === row.dataset.id);
    if (btn.dataset.act === 'use') {
      lib = await api('/api/library/models/active', { method: 'POST', json: { id: item.id } });
      renderStudio();
      renderLibraryPage();
    } else if (btn.dataset.act === 'cache') {
      await api(`/api/library/items/${encodeURIComponent(item.id)}/cache`, { method: 'POST' });
      toast(`Caching ${item.name}…`);
      loadLibrary();
    } else if (btn.dataset.act === 'delete') {
      const detail = item.dataset && !item.legacy
        ? `This also permanently deletes its Kaggle dataset ${item.dataset}${item.size_bytes ? ` (${formatBytes(item.size_bytes)})` : ''}.`
        : item.legacy ? `The older shared dataset ${item.dataset} is kept.` : 'It is not cached on Kaggle, so only the entry is removed.';
      if (!await confirmDialog(`Delete ${item.name}?`, detail, 'Delete', true)) return;
      await api(`/api/library/items/${encodeURIComponent(item.id)}`, { method: 'DELETE' });
      loadLibrary();
    }
  } catch (err) {
    toast(err.message, 'bad');
  }
}

function bindLibraryPage() {
  $('#refreshLibraryBtn').addEventListener('click', async e => {
    e.target.disabled = true;
    await loadLibrary(true);
    e.target.disabled = false;
  });
  ['#modelTable', '#loraTable', '#legacyTable'].forEach(sel => $(sel).addEventListener('click', onRowAction));

  // generic "Cancel" buttons inside inline forms
  $$('[data-close]').forEach(b => b.addEventListener('click', () => { $(`#${b.dataset.close}`).hidden = true; }));

  // Add model
  $('#addModelBtn').addEventListener('click', () => {
    $('#addModelForm').hidden = false;
    $('#newModelUrl').focus();
  });
  $('#addModelForm').addEventListener('submit', async e => {
    e.preventDefault();
    const submit = $('button[type=submit]', e.target);
    submit.disabled = true;
    submit.textContent = 'Checking…';
    try {
      const data = await api('/api/library/models/add', { method: 'POST', json: { url: $('#newModelUrl').value.trim(), name: $('#newModelName').value.trim() } });
      await api('/api/library/models/active', { method: 'POST', json: { id: data.item.id } });
      e.target.reset();
      e.target.hidden = true;
      toast(`Added ${data.item.name}; caching on Kaggle…`, 'ok');
      loadLibrary();
    } catch (err) {
      toast(err.message, 'bad');
    } finally {
      submit.disabled = false;
      submit.textContent = 'Add & cache';
    }
  });

  // Add LoRA
  let loraFile = null;
  const form = $('#addLoraForm');
  const setSource = src => {
    $$('#loraSourceSeg button').forEach(b => b.classList.toggle('active', b.dataset.src === src));
    $$('[data-pane]', form).forEach(p => { p.hidden = p.dataset.pane !== src; });
    form.dataset.src = src;
  };
  setSource('link');
  $('#loraSourceSeg').addEventListener('click', e => { if (e.target.dataset.src) setSource(e.target.dataset.src); });
  $('#addLoraBtn').addEventListener('click', () => { form.hidden = false; $('#newLoraUrl').focus(); });

  const pickFile = f => {
    if (!f || !f.name.endsWith('.safetensors')) return toast('Choose a .safetensors file.', 'bad');
    loraFile = f;
    $('#loraFileLabel').textContent = `${f.name} · ${formatBytes(f.size)}`;
  };
  $('#loraFileInput').addEventListener('change', e => pickFile(e.target.files[0]));
  dropTarget($('#loraFilePick'), dt => pickFile(dt.files[0]));

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const submit = $('button[type=submit]', form);
    submit.disabled = true;
    try {
      const name = $('#newLoraName').value.trim();
      const scale = parseFloat($('#newLoraScale').value) || 1.0;
      const isTurbo = $('#newLoraTurbo').checked;
      if (form.dataset.src === 'file') {
        if (!loraFile) throw new Error('Choose a .safetensors file first.');
        submit.textContent = 'Uploading…';
        const body = new FormData();
        body.append('file', loraFile);
        body.append('name', name);
        body.append('scale', scale);
        body.append('is_turbo', isTurbo);
        await api('/api/loras/upload', { method: 'POST', body });
      } else {
        const url = $('#newLoraUrl').value.trim();
        if (!url) throw new Error('Enter a LoRA link or Hugging Face path.');
        submit.textContent = 'Adding…';
        await api('/api/loras/add', { method: 'POST', json: { url, name, scale, is_turbo: isTurbo, civitai_token: $('#newLoraCivitaiKey').value.trim() } });
      }
      form.reset();
      loraFile = null;
      $('#loraFileLabel').textContent = 'Choose a .safetensors file';
      form.hidden = true;
      toast('LoRA added; caching on Kaggle…', 'ok');
      loadLibrary();
    } catch (err) {
      toast(err.message, 'bad');
    } finally {
      submit.disabled = false;
      submit.textContent = 'Add & cache';
    }
  });
}

export function initLibrary() {
  bindStudio();
  bindLibraryPage();
  // Local state first (instant), then what is actually cached on Kaggle (a few seconds).
  return loadLibrary(false).then(() => loadLibrary(true));
}
