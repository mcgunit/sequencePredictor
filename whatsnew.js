// First-login introduction and the "what's new" notes: what each account has
// seen, and what it should be shown. The app reads and records it through
// system.js (/api/whats-new).
//
// What an account has already seen lives in config/user-state.json - not in
// users.json, because the administrator has no record there (the credential
// is the environment's), and password hashes should not sit next to churny
// interface state. The content lives in announcements.js, which is the only
// file to edit when a feature ships.

const fs = require('fs');
const path = require('path');

const config = require('./config');
const { ANNOUNCEMENTS, TOUR } = require('./announcements');

const STATE_FILE = path.join(config.CONFIG_DIR, 'user-state.json');

// --- per-account state ------------------------------------------------------
function loadState() {
  try {
    const parsed = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (e) {
    if (e.code !== 'ENOENT') console.log(`user-state.json could not be read (${e.message}) - treating it as empty`);
    return {};
  }
}

function saveState(state) {
  try {
    fs.mkdirSync(config.CONFIG_DIR, { recursive: true, mode: 0o700 });
    const tmp = `${STATE_FILE}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(state, null, 2), { mode: 0o600 });
    fs.renameSync(tmp, STATE_FILE);
  } catch (e) {
    // Losing this only means someone is shown the note twice.
    console.log(`Could not write user-state.json: ${e.message}`);
  }
}

function stateOf(user) {
  return loadState()[user.name] || {};
}

function markSeen(user, ids, tourDone) {
  const state = loadState();
  const mine = state[user.name] || {};
  const seen = new Set(mine.seen || []);
  ids.forEach((id) => seen.add(id));
  state[user.name] = {
    ...mine,
    seen: [...seen].slice(-200),        // ids are permanent; keep the recent ones
    tourDone: tourDone || mine.tourDone || false,
    updatedAt: new Date().toISOString(),
  };
  saveState(state);
}

// --- what this account should be shown --------------------------------------
function visible(user) {
  return ANNOUNCEMENTS.filter((a) => a.audience !== 'admin' || user.role === 'admin');
}

function unseen(user) {
  const seen = new Set(stateOf(user).seen || []);
  return visible(user).filter((a) => !seen.has(a.id));
}

// What to show right now: the introduction for an account that has never seen
// anything, otherwise the unseen 'major' notes. 'minor' ones only count as
// unseen. Never for the open-access pseudo user - there is no account to
// record it against.
function pending(user) {
  if (!user || user.open) return null;
  const state = stateOf(user);
  const items = unseen(user);
  if (!state.tourDone) return { kind: 'tour', items };
  const major = items.filter((a) => a.level === 'major');
  return major.length ? { kind: 'news', items: major } : null;
}

// The tour's steps this reader's audience may see.
function tourSteps(user) {
  return TOUR.steps.filter((s) => s.audience !== 'admin' || (user && user.role === 'admin'));
}

// --- what a reader sends back -------------------------------------------------
// A list of ids or a single one as a plain string. Comma-separated values are
// accepted too, so a hand-made request cannot silently store one nonsense id
// instead of four real ones.
function idsFrom(value) {
  return [].concat(value || [])
    .flatMap((v) => String(v).split(','))
    .map((v) => v.trim())
    .filter(Boolean)
    .slice(0, 50);
}

// "Show me the introduction again": the tour is pending again.
function replay(user) {
  const state = loadState();
  if (state[user.name]) { state[user.name].tourDone = false; saveState(state); }
}

// the app's JSON (system.js) reads and records through these
module.exports = { pending, STATE_FILE, visible, unseen, stateOf, markSeen, tourSteps, idsFrom, replay, TOUR };
