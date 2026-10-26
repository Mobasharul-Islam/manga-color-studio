// Entry point: view switching and module start-up.

import { $, $$, api, toast, store, on } from './ui.js';
import { initChapter } from './chapter.js';
import { initLibrary, loadLibrary } from './library.js';
import { initRun } from './run.js';
import { initViewer, currentResultChapter } from './viewer.js';
import { initSettings } from './settings.js';

function showView(name) {
  $$('.nav-tab').forEach(t => t.classList.toggle('active', t.dataset.view === name));
  $('#studioView').hidden = name !== 'studio';
  $('#libraryView').hidden = name !== 'library';
  store.set('mcs.view', name);
  if (name === 'library') loadLibrary();
}

$$('.nav-tab').forEach(t => t.addEventListener('click', () => showView(t.dataset.view)));
$$('[data-goto]').forEach(b => b.addEventListener('click', () => showView(b.dataset.goto)));
showView(store.get('mcs.view', 'studio') === 'library' ? 'library' : 'studio');

$('#openFolderBtn').addEventListener('click', async () => {
  const chapter = currentResultChapter();
  try {
    await api(`/api/open-folder${chapter ? `?chapter_name=${encodeURIComponent(chapter)}` : ''}`, { method: 'POST' });
  } catch (err) {
    toast(err.message, 'bad');
  }
});

on('account:changed', () => loadLibrary(true));

initSettings();
initChapter().then(initRun);
initLibrary();
initViewer();
