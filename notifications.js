'use strict';

// Notifications for the administrator (16 Nov 2026; docs/operations.md, Notifications): what went wrong while nobody
// was looking, kept on the server and shown in the app - a bell with the unread count in the menu, and the unread
// list on the first page after signing in. There is no mail: this server has no mail server to send through.
//
// Raised by the schedule (jobs.js) and the workers (workers.js), never by a page:
//   job-failed       a job ended with a status other than 0, disappeared, or could not be launched
//   no-data          the lottery run got no draws from the API for any game (its retries are part of the same note)
//   no-data-ended    the retries of such a run ended: data arrived, or they were used up
//   blocked          the queue waited six hours for process.lock
//   worker-fallback  a control or a hyperopt did not start on the DL360 and runs on this server instead
//   worker-power     the DL360 did not come up, or is still on long after its last work
//   reading          the week's reading could not be written
//   slow             a job runs far longer than its recent runs
//
// The store is config/notifications.json (gitignored, like the schedule's state): the newest KEPT notes, and for each
// administrator the ids they have marked as read. A note is unread for an administrator until they mark it.
//
//   GET  /api/admin/notifications        -> { unread, of, kept, items: [{ id, at, kind, level, title, body, link, read }] }
//   POST /api/admin/notifications/read   { ids: [...] } or { all: true }

const fs = require('fs');
const path = require('path');

const config = require('./config');

const KEPT = 200;
const LEVELS = ['error', 'warning', 'info'];
const DEDUPE_MS = 24 * 3600 * 1000;
const ID = /^n\d{1,12}$/;

let FILE = path.join(config.CONFIG_DIR, 'notifications.json');
// test/notifications.test.js points the store at a temporary folder
function useFile(file) { FILE = file; }

function load() {
  try {
    const parsed = JSON.parse(fs.readFileSync(FILE, 'utf-8'));
    const s = parsed && typeof parsed === 'object' ? parsed : {};
    return {
      version: 1,
      seq: Number.isInteger(s.seq) && s.seq >= 0 ? s.seq : 0,
      items: Array.isArray(s.items) ? s.items.filter((n) => n && typeof n === 'object' && ID.test(String(n.id))) : [],
      read: s.read && typeof s.read === 'object' && !Array.isArray(s.read) ? s.read : {},
    };
  } catch (e) {
    if (e.code !== 'ENOENT') console.log(`notifications.json could not be read (${e.message}) - treating it as empty`);
    return { version: 1, seq: 0, items: [], read: {} };
  }
}

function save(state) {
  fs.mkdirSync(path.dirname(FILE), { recursive: true, mode: 0o700 });
  const tmp = `${FILE}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(state, null, 2), { mode: 0o600 });
  fs.renameSync(tmp, FILE);
}

const text = (value, max) => {
  const s = String(value == null ? '' : value).replace(/\s+/g, ' ').trim();
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
};

// Adds a note; answers it, or null when it was not added (a note with the same `dedupe` key inside dedupeMs, no
// title, or a store that cannot be written). Never throws: a note that cannot be kept must not cost the caller's job.
//   kind, level ('error' | 'warning' | 'info'), title, body (a string or lines), link ({href, label}; an app route),
//   dedupe (a key: one note per key per dedupeMs), at (ms).
function raise({ kind, level = 'warning', title, body = null, link = null, dedupe = null, dedupeMs = DEDUPE_MS, at = Date.now() } = {}) {
  try {
    const head = text(title, 200);
    if (!head) return null;
    const state = load();
    const key = dedupe ? text(dedupe, 200) : null;
    if (key && state.items.some((n) => n.dedupe === key && at - n.at < dedupeMs)) return null;
    state.seq += 1;
    const lines = [].concat(body == null ? [] : body).map((b) => text(b, 600)).filter(Boolean).slice(0, 6);
    const href = link && typeof link.href === 'string' && /^\/[A-Za-z0-9/_#?=&.-]*$/.test(link.href) ? link.href : null;
    const note = {
      id: `n${state.seq}`, at, kind: text(kind, 40) || 'note', level: LEVELS.includes(level) ? level : 'warning',
      title: head, body: lines, link: href ? { href, label: text(link.label, 60) || 'Open' } : null, dedupe: key,
    };
    state.items.unshift(note);
    state.items = state.items.slice(0, KEPT);
    const kept = new Set(state.items.map((n) => n.id));
    Object.keys(state.read).forEach((name) => {
      state.read[name] = [].concat(state.read[name] || []).filter((id) => kept.has(id));
    });
    save(state);
    return note;
  } catch (e) {
    console.log(`Could not keep a notification (${title}): ${e.message}`);
    return null;
  }
}

// What one administrator sees: every kept note, newest first, with their own read mark.
function view(user) {
  const state = load();
  const mine = new Set(user && !user.open ? [].concat(state.read[user.name] || []) : []);
  const items = state.items.map((n) => ({
    id: n.id, at: n.at, kind: n.kind, level: n.level, title: n.title, body: [].concat(n.body || []), link: n.link || null,
    read: mine.has(n.id),
  }));
  return { unread: items.filter((n) => !n.read).length, of: items.length, kept: KEPT, items };
}

// Marks notes as read for this administrator: the named ids, or all of them. Answers the ids newly marked.
function markRead(user, ids) {
  const state = load();
  const known = new Set(state.items.map((n) => n.id));
  const mine = new Set([].concat(state.read[user.name] || []).filter((id) => known.has(id)));
  const wanted = ids === 'all' ? [...known] : [...new Set([].concat(ids || []).map(String))].filter((id) => known.has(id));
  const marked = wanted.filter((id) => !mine.has(id));
  marked.forEach((id) => mine.add(id));
  state.read[user.name] = [...mine];
  if (marked.length) save(state);
  return marked;
}

function idsFrom(value) {
  return [].concat(value || []).map((v) => String(v).trim()).filter((v) => ID.test(v)).slice(0, KEPT);
}

// The routes (administrator only; system.js passes its guard, form token, JSON body parser and answers).
function install(app, { guard, jsonBody, csrf, safe, json }) {
  app.get('/api/admin/notifications', guard, safe((req, res) => json(res, view(req.user))));
  app.post('/api/admin/notifications/read', guard, jsonBody, csrf, safe((req, res) => {
    if (req.user.open) return json(res, { error: 'Authentication is off: there is no account to record it against.' }, 409);
    const all = (req.body || {}).all === true;
    const ids = all ? 'all' : idsFrom((req.body || {}).ids);
    if (!all && !ids.length) return json(res, { error: 'no notification was named' }, 400);
    const marked = markRead(req.user, ids);
    return json(res, { ok: true, message: `${marked.length} marked as read`, marked, unread: view(req.user).unread });
  }));
}

module.exports = { raise, view, markRead, idsFrom, install, KEPT, LEVELS, _internals: { useFile, load } };
