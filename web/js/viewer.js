// The page viewer: colorized results, comparison modes, post-processing and page re-runs.

import { $, $$, esc, api, toast, confirmDialog, store, on } from './ui.js';
import { startJob, buildPayload, isJobActive } from './run.js';

const state = {
  chapter: null,
  pairs: [],
  index: 0,
  mode: store.get('mcs.viewMode', 'compare'),
  canPost: false,
  postDirty: false,
  selected: new Set(),   // original scan names ticked in the grid
};

// ------------------------------------------------------------------ loading results

export async function loadResults(chapter = null, keepPage = false) {
  let data;
  try {
    data = await api(chapter ? `/api/results?chapter=${encodeURIComponent(chapter)}` : '/api/results');
  } catch (err) {
    toast(err.message, 'bad');
    return;
  }
  const changed = data.chapter_name !== state.chapter;
  state.chapter = data.chapter_name;
  state.pairs = data.pairs || [];
  state.canPost = !!data.can_postprocess;
  state.postDirty = false;
  if (changed) state.selected.clear();
  state.index = keepPage && !changed ? Math.min(state.index, Math.max(0, state.pairs.length - 1)) : 0;

  const sel = $('#resultChapterSelect');
  sel.innerHTML = (data.available_chapters || []).map(ch =>
    `<option value="${esc(ch)}" ${ch === data.chapter_name ? 'selected' : ''}>${esc(ch.replace(/_/g, ' '))}</option>`).join('');
  sel.hidden = !(data.available_chapters || []).length;

  if (data.postprocess) {
    $('#lineartSlider').value = data.postprocess.lineart_strength;
    $('#lineartValue').textContent = Number(data.postprocess.lineart_strength).toFixed(2);
    $('#bubblesToggle').checked = !!data.postprocess.preserve_bubbles;
  }
  render();
}

// ------------------------------------------------------------------ rendering

function colorUrl(pair) {
  if (!state.postDirty || !state.canPost || !pair.has_original) return pair.color_url;
  const params = new URLSearchParams({ lineart: $('#lineartSlider').value, bubbles: $('#bubblesToggle').checked });
  return `/api/image/preview/${encodeURIComponent(state.chapter)}/${encodeURIComponent(pair.color_name)}?${params}`;
}

function render() {
  const has = state.pairs.length > 0;
  $('#emptyState').hidden = has;
  $('#compareView').hidden = !has || state.mode !== 'compare';
  $('#sideView').hidden = !has || state.mode !== 'side';
  $('#gridView').hidden = !has || state.mode !== 'grid';
  $$('#viewModeSeg button').forEach(b => b.classList.toggle('active', b.dataset.mode === state.mode));
  $('#pageStrip').innerHTML = has && state.mode !== 'grid'
    ? state.pairs.map((p, i) => `<button class="page-thumb ${i === state.index ? 'active' : ''}" data-index="${i}" title="${esc(p.id)}"><img src="${esc(p.thumb_url || p.color_url)}" alt="" loading="lazy"><span>${i + 1}</span></button>`).join('')
    : '';
  if (state.mode === 'grid') renderGrid();
  showPage(state.index);
  layoutCompare();  // the image may already be loaded, so no load event will come
  renderTools();
}

function showPage(i) {
  const pair = state.pairs[i];
  if (!pair) return;
  state.index = i;
  const color = colorUrl(pair);
  const original = pair.input_url || pair.color_url;
  $('#compareColor').src = color;
  $('#compareOriginal').src = original;
  $('#sideColor').src = color;
  $('#sideOriginal').src = original;
  $('#downloadPageBtn').href = pair.color_url;
  $('#downloadPageBtn').download = pair.color_name;
  $$('#pageStrip .page-thumb').forEach((t, n) => t.classList.toggle('active', n === i));
  $('#pageStrip .page-thumb.active')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  renderTools();
}

function renderGrid() {
  $('#gridView').innerHTML = state.pairs.map((p, i) => {
    const picked = p.page && state.selected.has(p.page);
    return `<div class="grid-card ${picked ? 'selected' : ''}" data-index="${i}">
        <img src="${esc(p.thumb_url || p.color_url)}" alt="Page ${i + 1}" loading="lazy">
        <span class="num">${i + 1}</span>
        ${p.page ? `<input type="checkbox" class="pick" data-page="${esc(p.page)}" ${picked ? 'checked' : ''} title="Select for re-run">` : ''}
      </div>`;
  }).join('');
}

function renderTools() {
  const has = state.pairs.length > 0;
  const pair = state.pairs[state.index];
  const busy = isJobActive();
  $('#viewModeSeg').hidden = !has;
  $('#downloadPageBtn').hidden = !has || state.mode === 'grid';
  $('#rerunPageBtn').hidden = !has || state.mode === 'grid';
  $('#rerunPageBtn').disabled = busy || !pair?.page;
  $('#rerunPageBtn').title = pair?.page ? 'Colorize this page again with the current settings'
    : pair?.has_original ? 'This page was not made by the app (its file name differs), so it cannot be re-run'
    : 'No original scan for this page';
  const n = state.selected.size;
  $('#rerunSelectedBtn').hidden = !n;
  $('#rerunSelectedBtn').disabled = busy;
  $('#rerunSelectedBtn').textContent = `Re-run ${n} selected`;

  const post = $('#postBar');
  post.hidden = !has;
  post.classList.toggle('disabled', !state.canPost);
  $('#applyPostBtn').disabled = !state.canPost || !state.postDirty;
  $('#applyPostBtn').classList.toggle('dirty', state.postDirty);
  $('#postHint').textContent = state.canPost
    ? (state.postDirty ? 'Previewing this page. Apply to update the whole chapter.' : '')
    : 'No original scans found for these pages, so lineart and bubble restore are off.';
}

// ------------------------------------------------------------------ compare slider

// Sizes the stage to exactly the page as displayed (fit inside the viewer), so slider
// percentages map onto the picture and both images share one box.
function layoutCompare() {
  const img = $('#compareColor');
  const view = $('#compareView');
  if (view.hidden || !img.naturalWidth) return;
  const pad = 32;
  const scale = Math.min((view.clientWidth - pad) / img.naturalWidth, (view.clientHeight - pad) / img.naturalHeight);
  const stage = $('#compareStage');
  stage.style.width = `${Math.max(1, Math.floor(img.naturalWidth * scale))}px`;
  stage.style.height = `${Math.max(1, Math.floor(img.naturalHeight * scale))}px`;
}

function bindCompare() {
  const stage = $('#compareStage');
  stage.tabIndex = 0;
  stage.setAttribute('role', 'slider');
  stage.setAttribute('aria-label', 'Original / color split');
  let pos = 50;
  const setPct = pct => {
    pos = Math.min(100, Math.max(0, pct));
    stage.style.setProperty('--pos', `${pos}%`);
    stage.setAttribute('aria-valuenow', Math.round(pos));
  };
  const fromPointer = e => {
    const r = stage.getBoundingClientRect();
    setPct(((e.clientX - r.left) / r.width) * 100);
  };

  stage.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    e.preventDefault();               // no text selection / native image drag
    stage.setPointerCapture(e.pointerId);
    stage.classList.add('dragging');
    fromPointer(e);
  });
  stage.addEventListener('pointermove', e => { if (stage.hasPointerCapture(e.pointerId)) fromPointer(e); });
  const stop = e => {
    if (stage.hasPointerCapture(e.pointerId)) stage.releasePointerCapture(e.pointerId);
    stage.classList.remove('dragging');
  };
  stage.addEventListener('pointerup', stop);
  stage.addEventListener('pointercancel', stop);
  stage.addEventListener('dragstart', e => e.preventDefault());
  stage.addEventListener('keydown', e => {
    // With the slider focused, arrows move the split; otherwise they change pages.
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    e.stopPropagation();
    setPct(pos + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 10 : 2));
  });
  setPct(50);

  $('#compareColor').addEventListener('load', layoutCompare);
  new ResizeObserver(layoutCompare).observe($('#viewer'));
}

// ------------------------------------------------------------------ post-processing

let previewTimer = null;

function onPostChange() {
  $('#lineartValue').textContent = Number($('#lineartSlider').value).toFixed(2);
  state.postDirty = true;
  renderTools();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(() => showPage(state.index), 250);
}

async function applyPost() {
  const btn = $('#applyPostBtn');
  btn.disabled = true;
  btn.textContent = 'Applying…';
  try {
    const res = await api('/api/postprocess', {
      method: 'POST',
      json: { chapter: state.chapter, lineart_strength: parseFloat($('#lineartSlider').value), preserve_bubbles: $('#bubblesToggle').checked },
    });
    toast(`Updated ${res.count} pages.`, 'ok');
    await loadResults(state.chapter, true);
  } catch (err) {
    toast(err.message, 'bad');
  } finally {
    btn.textContent = 'Apply to chapter';
    renderTools();
  }
}

// ------------------------------------------------------------------ re-runs

async function rerun(pages) {
  if (!pages.length || !state.chapter) return;
  const label = pages.length === 1 ? '1 page' : `${pages.length} pages`;
  const ok = await confirmDialog(`Re-run ${label}?`,
    `Colorizes ${label} of "${state.chapter.replace(/_/g, ' ')}" again with the current model, LoRAs, prompt and hardware. The current versions are kept in the chapter's history folder.`,
    'Re-run');
  if (!ok) return;
  if (await startJob(buildPayload({ chapter_name: state.chapter, pages }))) {
    state.selected.clear();
    render();
  }
}

// ------------------------------------------------------------------ init

export function initViewer() {
  bindCompare();

  $('#viewModeSeg').addEventListener('click', e => {
    const mode = e.target.dataset.mode;
    if (!mode) return;
    state.mode = mode;
    store.set('mcs.viewMode', mode);
    render();
  });
  $('#pageStrip').addEventListener('click', e => {
    const t = e.target.closest('.page-thumb');
    if (t) showPage(Number(t.dataset.index));
  });
  $('#gridView').addEventListener('click', e => {
    if (e.target.classList.contains('pick')) {
      const page = e.target.dataset.page;
      if (e.target.checked) state.selected.add(page); else state.selected.delete(page);
      e.target.closest('.grid-card').classList.toggle('selected', e.target.checked);
      renderTools();
      return;
    }
    const card = e.target.closest('.grid-card');
    if (!card) return;
    state.mode = 'compare';
    store.set('mcs.viewMode', 'compare');
    state.index = Number(card.dataset.index);
    render();
  });
  $('#resultChapterSelect').addEventListener('change', e => loadResults(e.target.value));

  $('#lineartSlider').addEventListener('input', onPostChange);
  $('#bubblesToggle').addEventListener('change', onPostChange);
  $('#applyPostBtn').addEventListener('click', applyPost);

  $('#rerunPageBtn').addEventListener('click', () => {
    const pair = state.pairs[state.index];
    if (pair?.page) rerun([pair.page]);
  });
  $('#rerunSelectedBtn').addEventListener('click', () => rerun([...state.selected]));

  document.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName) || $('#studioView').hidden) return;
    if (!state.pairs.length) return;
    if (e.key === 'ArrowRight') showPage((state.index + 1) % state.pairs.length);
    else if (e.key === 'ArrowLeft') showPage((state.index - 1 + state.pairs.length) % state.pairs.length);
    else if (['1', '2', '3'].includes(e.key)) {
      state.mode = ['compare', 'side', 'grid'][Number(e.key) - 1];
      store.set('mcs.viewMode', state.mode);
      render();
    }
  });

  on('job:done', chapter => loadResults(chapter, true));
  on('job:active', renderTools);
  return loadResults();
}

export const currentResultChapter = () => state.chapter;
