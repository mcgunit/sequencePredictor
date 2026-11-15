'use strict';

// The in-app wiki (Admin > Wiki): the project's own documentation, README.md and docs/*.md, served to administrators
// as markdown text. The app renders it (web/src/pages/admin/WikiPage.tsx); nothing is written here, and only the files
// this module lists can be asked for.
//
//   GET /api/admin/docs        -> { name, pages: [{ id, title, file, bytes, updatedAt }] }
//   GET /api/admin/docs/:page  -> { id, title, file, updatedAt, text }

const fs = require('fs');
const path = require('path');

const NAME = 'sequencePredictor Wiki';
const DIR = 'docs';
// the menu's order; a page that is not named here follows in alphabetical order, the changelog always last
const ORDER = ['models', 'scoring', 'tuning', 'research', 'markets', 'operations', 'roadmap'];
const LAST = ['changelog'];
const MAX_BYTES = 4 * 1024 * 1024;
const ID = /^[a-z0-9][a-z0-9-]{0,63}$/;

function titleOf(file, fallback) {
  try {
    const fd = fs.openSync(file, 'r');
    const buf = Buffer.alloc(2048);
    const n = fs.readSync(fd, buf, 0, 2048, 0);
    fs.closeSync(fd);
    const m = /^#\s+(.+?)\s*$/m.exec(buf.toString('utf-8', 0, n));
    return m ? m[1] : fallback;
  } catch (e) {
    return fallback;
  }
}

function entry(root, id, rel, fallback) {
  const file = path.join(root, rel);
  let st;
  try { st = fs.statSync(file); } catch (e) { return null; }
  if (!st.isFile() || st.size > MAX_BYTES) return null;
  return { id, title: titleOf(file, fallback), file: rel, bytes: st.size, updatedAt: st.mtime.toISOString() };
}

function list(root) {
  let names = [];
  try { names = fs.readdirSync(path.join(root, DIR)); } catch (e) { /* no docs folder */ }
  const ids = names.filter((n) => n.endsWith('.md')).map((n) => n.slice(0, -3)).filter((id) => ID.test(id) && id !== 'readme');
  const rank = (id) => (ORDER.includes(id) ? ORDER.indexOf(id) : LAST.includes(id) ? 1000 + LAST.indexOf(id) : 500);
  ids.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  const pages = [entry(root, 'readme', 'README.md', 'Overview'), ...ids.map((id) => entry(root, id, path.join(DIR, `${id}.md`), id))].filter(Boolean);
  const readme = pages.find((p) => p.id === 'readme');
  if (readme) readme.title = 'Overview';
  return { name: NAME, pages };
}

function page(root, id) {
  if (typeof id !== 'string' || !ID.test(id)) return null;
  const meta = list(root).pages.find((p) => p.id === id);
  if (!meta) return null;
  try {
    return { id: meta.id, title: meta.title, file: meta.file, updatedAt: meta.updatedAt, text: fs.readFileSync(path.join(root, meta.file), 'utf-8') };
  } catch (e) {
    return null;
  }
}

function install(app, { root, guard, safe, json }) {
  app.get('/api/admin/docs', guard, safe((req, res) => json(res, list(root))));
  app.get('/api/admin/docs/:page', guard, safe((req, res) => {
    const p = page(root, String(req.params.page || ''));
    return p ? json(res, p) : json(res, { error: 'no such page' }, 404);
  }));
}

module.exports = { NAME, list, page, install };
