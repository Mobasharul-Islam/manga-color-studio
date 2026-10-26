// Step 1: loading the chapter's pages into the app.

import { $, esc, api, toast, store, dropTarget, emit } from './ui.js';
import { filesFromDrop, packZip, isArchive, isImage } from './files.js';

const state = { pages: [], name: '' };

export function getChapter() {
  return { name: ($('#chapterNameInput').value.trim() || state.name || 'chapter_01'), pages: state.pages };
}

function render() {
  const has = state.pages.length > 0;
  $('#dropzone').hidden = has;
  $('#chapterLoaded').hidden = !has;
  $('#clearPagesBtn').hidden = !has;
  if (!has) return;
  $('#chapterNameInput').value = state.name;
  $('#pageCount').textContent = `${state.pages.length} page${state.pages.length === 1 ? '' : 's'}`;
  $('#thumbStrip').innerHTML = state.pages
    .map(p => `<img src="${esc(p.thumb_url || p.url)}" alt="${esc(p.name)}" title="${esc(p.name)}" loading="lazy">`).join('');
}

function setLoaded(data, fallbackName) {
  state.pages = data.images || [];
  state.name = data.chapter_name || fallbackName || 'chapter_01';
  store.set('mcs.chapterName', state.name);
  render();
  emit('chapter:changed', getChapter());
}

function setBusy(busy, label) {
  const dz = $('#dropzone');
  dz.classList.toggle('busy', busy);
  $('.dropzone-title', dz).textContent = busy ? label : 'Drop a chapter folder, images or a .zip/.cbz';
  if (busy) { dz.hidden = false; $('#chapterLoaded').hidden = true; }
}

async function upload(files, folderName) {
  if (!files.length) return;
  try {
    let archive;
    if (files.length === 1 && isArchive(files[0])) {
      archive = files[0];
    } else {
      const images = files.filter(isImage);
      if (!images.length) throw new Error('No images (.jpg, .png, .webp, .bmp) or .zip/.cbz found.');
      setBusy(true, `Packing ${images.length} pages…`);
      const name = (folderName || 'chapter').replace(/[^\w\-.]/g, '_');
      archive = new File([await packZip(images)], `${name}.zip`, { type: 'application/zip' });
    }
    setBusy(true, `Uploading ${(archive.size / 1e6).toFixed(1)} MB…`);
    const form = new FormData();
    form.append('file', archive);
    setLoaded(await api('/api/upload-zip', { method: 'POST', body: form }), folderName);
    toast(`Loaded ${state.pages.length} pages.`, 'ok');
  } catch (err) {
    toast(err.message, 'bad');
    render();
  } finally {
    setBusy(false);
    render();
  }
}

export async function initChapter() {
  // Restore whatever the server already has loaded (survives page reloads).
  try {
    const data = await api('/api/files');
    state.pages = data.images || [];
    const saved = store.get('mcs.chapterName');
    state.name = (data.chapter_name && data.chapter_name !== 'chapter_01') ? data.chapter_name : (saved || 'chapter_01');
  } catch { /* server not ready: start empty */ }
  render();

  $('#selectFolderBtn').addEventListener('click', async () => {
    // Native folder picker on the server's machine (this PC): keeps full file names.
    const btn = $('#selectFolderBtn');
    btn.disabled = true;
    try {
      const data = await api('/api/select-folder', { method: 'POST' });
      if (data.status === 'success') {
        setLoaded(data);
        toast(`Loaded ${state.pages.length} pages.`, 'ok');
      }
    } catch (err) {
      toast(err.message, 'bad');
    } finally {
      btn.disabled = false;
    }
  });

  $('#selectFilesBtn').addEventListener('click', () => $('#fileInput').click());
  $('#fileInput').addEventListener('change', e => {
    upload([...e.target.files], '');
    e.target.value = '';
  });

  dropTarget($('#dropzone'), async dt => {
    const { files, folderName } = await filesFromDrop(dt);
    upload(files, folderName);
  });

  // Shows the drop area again; the current pages stay until new ones are loaded.
  $('#replacePagesBtn').addEventListener('click', () => { $('#dropzone').hidden = !$('#dropzone').hidden; });

  $('#clearPagesBtn').addEventListener('click', async () => {
    try {
      await api('/api/clear-files', { method: 'POST' });
      state.pages = [];
      render();
      emit('chapter:changed', getChapter());
    } catch (err) {
      toast(err.message, 'bad');
    }
  });

  $('#chapterNameInput').addEventListener('change', e => {
    state.name = e.target.value.trim() || 'chapter_01';
    store.set('mcs.chapterName', state.name);
  });
}
