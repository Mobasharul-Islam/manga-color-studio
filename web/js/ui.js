// Small shared helpers: DOM lookup, escaping, API calls, toasts, confirm dialog, events.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function esc(text) {
  return String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Escapes text and turns kaggle.com URLs into links (errors point at notebooks to fix).
export function linkify(text) {
  return esc(text).replace(/https:\/\/www\.kaggle\.com\/[^\s<,)]+/g,
    url => `<a href="${url}" target="_blank" rel="noopener">${url}</a>`);
}

export function formatBytes(bytes) {
  if (!bytes) return '';
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}

export function formatDuration(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = n => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}

// fetch wrapper: JSON in/out, throws Error(detail) on non-2xx.
export async function api(path, { method = 'GET', json, body } = {}) {
  const opts = { method };
  if (json !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(json);
  } else if (body !== undefined) {
    opts.body = body;
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || `Request failed (${res.status})`);
  return data;
}

export function toast(message, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('#toasts').append(el);
  setTimeout(() => el.remove(), kind === 'bad' ? 7000 : 4000);
}

// Promise-based confirm using the <dialog id="confirmDialog">.
export function confirmDialog(title, text, okLabel = 'OK', danger = false) {
  const dlg = $('#confirmDialog');
  $('#confirmTitle').textContent = title;
  $('#confirmText').textContent = text;
  const ok = $('#confirmOk');
  ok.textContent = okLabel;
  ok.className = `btn btn-sm ${danger ? 'btn-danger' : 'btn-primary'}`;
  dlg.returnValue = '';
  dlg.showModal();
  return new Promise(resolve => dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), { once: true }));
}

// Minimal event bus between modules.
const listeners = {};
export const on = (name, fn) => (listeners[name] ||= []).push(fn);
export const emit = (name, data) => (listeners[name] || []).forEach(fn => fn(data));

// localStorage that never throws (private windows, blocked storage).
export const store = {
  get(key, fallback = null) {
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* ignore */ }
  },
};

// Wires a drop target that accepts files.
export function dropTarget(el, onFiles) {
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('dragover'); });
  el.addEventListener('dragleave', () => el.classList.remove('dragover'));
  el.addEventListener('drop', e => {
    e.preventDefault();
    el.classList.remove('dragover');
    onFiles(e.dataTransfer);
  });
}
