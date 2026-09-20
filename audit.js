// Append-only record of who signed in, when, from where, and what changed
// about an account. The admin reads it on /admin/activity; a user sees their
// own recent sign-ins on /account.
//
// It lives in the gitignored config/ folder, like the accounts themselves:
// the daily pipeline commits everything under data/, and sign-in records must
// never be pushed to a git remote. One JSON object per line, so appending is
// a single write that cannot corrupt earlier lines, and reading the recent
// entries is a bounded tail read rather than parsing the whole history.
//
// Bounded by construction: at MAX_BYTES the file is rotated to .1 and a new
// one started, so the two files together are the hard ceiling.

const fs = require('fs');
const path = require('path');

const config = require('./config');

const FILE = path.join(config.CONFIG_DIR, 'audit.jsonl');
const MAX_BYTES = 2 * 1024 * 1024;   // ~10k sign-ins per file, two files kept
const MAX_DETAIL = 200;
const MAX_AGENT = 120;

// What each event means on the pages. Anything not listed is still recorded
// and shown by its raw name - a missing label must never hide an event.
const LABELS = {
  'login.ok': 'signed in',
  'login.failed': 'failed sign-in',
  'login.locked': 'blocked (too many failed attempts)',
  'logout': 'signed out',
  'password.self': 'changed their own password',
  'password.rejected': 'wrong current password',
  'password.admin': 'password set by the administrator',
  'user.added': 'account created',
  'user.deleted': 'account deleted',
  'session.rejected': 'session no longer valid',
};

function clientAddress(req) {
  // Behind the Tailscale funnel every visitor arrives through the local
  // proxy, so trust proxy (set in server.js) is what makes this the real
  // client rather than 127.0.0.1.
  return String((req && req.ip) || '').replace(/^::ffff:/, '') || 'unknown';
}

function userAgent(req) {
  const value = req && req.headers ? String(req.headers['user-agent'] || '') : '';
  return value.slice(0, MAX_AGENT);
}

function ensureDir() {
  fs.mkdirSync(config.CONFIG_DIR, { recursive: true, mode: 0o700 });
}

function rotateIfBig() {
  try {
    if (fs.statSync(FILE).size > MAX_BYTES) fs.renameSync(FILE, `${FILE}.1`);
  } catch (e) { /* no file yet */ }
}

// Never throws and never blocks a request: an audit trail that can break a
// sign-in is worse than one that occasionally misses a line, so failures are
// reported to the console and swallowed.
function record(event, req, user, detail) {
  try {
    ensureDir();
    rotateIfBig();
    const entry = {
      at: new Date().toISOString(),
      event,
      user: String(user || (req && req.user && req.user.name) || 'anonymous').slice(0, 64),
      ip: clientAddress(req),
      agent: userAgent(req),
    };
    if (detail) entry.detail = String(detail).slice(0, MAX_DETAIL);
    fs.appendFileSync(FILE, `${JSON.stringify(entry)}\n`, { mode: 0o600 });
  } catch (e) {
    console.log(`Could not write the audit log: ${e.message}`);
  }
}

// The most recent `limit` entries, newest first, optionally for one user.
// Reads the tail of the file (and the rotated one when needed) instead of the
// whole history.
function recent(limit = 100, forUser = null) {
  const wanted = Math.max(1, Math.min(limit, 1000));
  const entries = [];
  for (const file of [FILE, `${FILE}.1`]) {
    let text;
    try {
      const stat = fs.statSync(file);
      const bytes = Math.min(stat.size, 512 * 1024);
      const fd = fs.openSync(file, 'r');
      const buffer = Buffer.alloc(bytes);
      fs.readSync(fd, buffer, 0, bytes, stat.size - bytes);
      fs.closeSync(fd);
      text = buffer.toString('utf-8');
      // A tail read can start mid-line; drop that first partial line.
      if (stat.size > bytes) text = text.slice(text.indexOf('\n') + 1);
    } catch (e) {
      continue;
    }
    const lines = text.split('\n').filter(Boolean);
    for (let i = lines.length - 1; i >= 0; i -= 1) {
      let entry;
      try { entry = JSON.parse(lines[i]); } catch (e) { continue; }
      if (forUser && entry.user !== forUser) continue;
      entries.push(entry);
      if (entries.length >= wanted) return entries;
    }
  }
  return entries;
}

// Last successful sign-in per user name, from the recent window - used for
// the "Last seen" column on the Users page and for the administrator, who has
// no record in users.json to carry it.
function lastLogins(limit = 500) {
  const seen = {};
  recent(limit).forEach((entry) => {
    if (entry.event === 'login.ok' && !seen[entry.user]) seen[entry.user] = entry;
  });
  return seen;
}

function label(event) {
  return LABELS[event] || event;
}

module.exports = { record, recent, lastLogins, label, FILE };
