// Accounts & tokens dialog: Kaggle credentials, the optional Hugging Face token and the
// encryption key for scans & results stored on Kaggle.

import { $, $$, esc, api, dropTarget, emit } from './ui.js';

function feedback(text, ok) {
  const el = $('#settingsFeedback');
  el.hidden = !text;
  el.textContent = text || '';
  el.className = `feedback ${ok ? 'ok' : 'bad'}`;
}

export async function refreshAccount() {
  let s;
  try {
    s = await api('/api/settings');
  } catch {
    return null;
  }
  const ok = s.kaggle_authenticated && s.kaggle_username;
  $('#accountDot').className = `dot ${ok ? 'ok' : 'bad'}`;
  $('#accountLabel').textContent = ok ? `@${s.kaggle_username}` : 'Connect Kaggle';
  $('#settingsStatus').innerHTML = `<span class="dot ${ok ? 'ok' : 'bad'}"></span><span>${ok ? `Connected to Kaggle as @${s.kaggle_username}` : 'Kaggle is not connected yet'}</span>`;
  $('#hfBadge').textContent = s.hf_token_configured ? (s.hf_token_preview || 'set') : 'not set';
  $('#hfBadge').className = `badge ${s.hf_token_configured ? 'ok' : ''}`;
  if (ok && !$('#kaggleUserInput').value) $('#kaggleUserInput').value = s.kaggle_username;

  $('#encBadge').textContent = s.encryption_key_configured ? `set · ${s.encryption_key_fingerprint}` : 'not set';
  $('#encBadge').className = `badge ${s.encryption_key_configured ? 'ok' : ''}`;
  $('#encBadge').title = s.encryption_key_configured
    ? 'Fingerprint of your key. Each run checks that the notebook received this same key.' : '';
  $('#encHelp').textContent = s.encryption_key_configured
    ? 'Nothing to set up on Kaggle: each encrypted run sends this key as a temporary private dataset that the app deletes when the run ends. Keep a copy of the key: without it, encrypted results cannot be decrypted.'
    : '';
  return s;
}

function setCredMode(mode) {
  $$('#credSeg button').forEach(b => b.classList.toggle('active', b.dataset.cred === mode));
  $$('[data-credpane]').forEach(p => { p.hidden = p.dataset.credpane !== mode; });
}

async function uploadKaggleJson(file) {
  if (!file || !file.name.endsWith('.json')) return feedback('Choose your kaggle.json file.', false);
  $('#kaggleJsonLabel').textContent = `Checking ${file.name}…`;
  const body = new FormData();
  body.append('file', file);
  try {
    const data = await api('/api/settings/upload-kaggle-json', { method: 'POST', body });
    $('#kaggleJsonLabel').textContent = `Connected as @${data.kaggle_username}`;
    feedback(data.message, true);
    await refreshAccount();
    emit('account:changed');
  } catch (err) {
    $('#kaggleJsonLabel').textContent = 'Drop kaggle.json here or click to choose';
    feedback(err.message, false);
  }
}

export function openSettings() {
  feedback('');
  refreshAccount();
  $('#settingsDialog').showModal();
}

export async function initSettings() {
  $('#accountBtn').addEventListener('click', openSettings);
  $('#credSeg').addEventListener('click', e => { if (e.target.dataset.cred) setCredMode(e.target.dataset.cred); });
  $('#kaggleJsonInput').addEventListener('change', e => uploadKaggleJson(e.target.files[0]));
  dropTarget($('#kaggleJsonPick'), dt => uploadKaggleJson(dt.files[0]));

  $('#saveSettingsBtn').addEventListener('click', async e => {
    const btn = e.target;
    btn.disabled = true;
    btn.textContent = 'Checking…';
    try {
      const data = await api('/api/settings/save', {
        method: 'POST',
        json: {
          kaggle_username: $('#kaggleUserInput').value.trim() || null,
          kaggle_key: $('#kaggleKeyInput').value.trim() || null,
          hf_token: $('#hfTokenInput').value.trim() || null,
          encryption_key: $('#encKeyInput').value.trim() || null,
        },
      });
      feedback(data.message, data.status === 'success');
      $('#kaggleKeyInput').value = '';
      $('#hfTokenInput').value = '';
      $('#encKeyInput').value = '';
      await refreshAccount();
      emit('account:changed');
    } catch (err) {
      feedback(err.message, false);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Save & test';
    }
  });

  const s = await refreshAccount();
  if (s && !s.kaggle_authenticated) openSettings();  // first run: ask for Kaggle right away
}
