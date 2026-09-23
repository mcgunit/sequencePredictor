// Council chat sessions, per account, on disk.
//
// Until now a council conversation lived only in the browser tab that asked
// it: a reload lost everything, and api.py keeps just its last twenty jobs in
// memory. Sessions now live in the gitignored config/ folder next to the
// accounts and the audit log - never in data/, which the pipeline commits -
// one JSON file per session under a directory per account, so one user's
// sessions are unreachable from another's simply by construction: the
// directory is derived from the name of whoever is asking.
//
// A session exists only from its first question onward. There is no "create
// an empty session" operation at all - appendTurn() creates the file when the
// first turn is recorded - so a session with zero messages cannot be written,
// and list() refuses to show one even if a file were edited by hand.
//
// A turn is recorded when the question is SUBMITTED, as pending with its job
// id, and completed by the server's own poller (council.js) when api.py
// finishes - so closing the tab does not lose the answer, and a server
// restart resumes tracking from the pending turns it finds on disk.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const ID = /^[a-f0-9]{12}$/;
const TITLE_LENGTH = 70;
const MAX_TURNS = 200;          // per session; older turns are dropped, newest kept

function createStore(rootDir) {
  function userDir(userName) {
    // Readable prefix for a human looking at the folder, plus a hash so two
    // names that sanitise to the same prefix can never share a directory.
    const safe = String(userName).replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 40) || 'user';
    const hash = crypto.createHash('sha1').update(String(userName)).digest('hex').slice(0, 8);
    return path.join(rootDir, `${safe}-${hash}`);
  }

  function sessionPath(userName, id) {
    if (!ID.test(String(id))) return null;
    return path.join(userDir(userName), `${id}.json`);
  }

  function newId() {
    return crypto.randomBytes(6).toString('hex');
  }

  function readJson(file) {
    try {
      const parsed = JSON.parse(fs.readFileSync(file, 'utf-8'));
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch (e) {
      return null;
    }
  }

  function writeJson(file, value) {
    fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
    const tmp = `${file}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(value, null, 2), { mode: 0o600 });
    fs.renameSync(tmp, file);
  }

  function summary(session) {
    const last = session.turns[session.turns.length - 1];
    return {
      id: session.id,
      title: session.title,
      turns: session.turns.length,
      created: session.created,
      updated: session.updated,
      pending: session.turns.some((t) => t.state === 'pending'),
      lastState: last ? last.state : null,
    };
  }

  // The account's sessions, newest first. A session without turns is not a
  // session - it is never written, and never listed.
  function list(userName) {
    let names;
    try {
      names = fs.readdirSync(userDir(userName)).filter((n) => n.endsWith('.json'));
    } catch (e) {
      return [];
    }
    return names
      .map((n) => readJson(path.join(userDir(userName), n)))
      .filter((s) => s && Array.isArray(s.turns) && s.turns.length > 0 && ID.test(String(s.id)))
      .sort((a, b) => String(b.updated).localeCompare(String(a.updated)))
      .map(summary);
  }

  function load(userName, id) {
    const file = sessionPath(userName, id);
    if (!file) return null;
    const session = readJson(file);
    return session && Array.isArray(session.turns) && session.turns.length > 0 ? session : null;
  }

  // Record a question. `sessionId` null starts a new session; the first
  // question becomes its title. Returns { session, turn }.
  function appendTurn(userName, sessionId, { jobId, question, context }) {
    const now = new Date().toISOString();
    let session = sessionId ? load(userName, sessionId) : null;
    if (!session) {
      session = {
        id: newId(),
        owner: String(userName),
        title: String(question).replace(/\s+/g, ' ').trim().slice(0, TITLE_LENGTH),
        created: now,
        updated: now,
        turns: [],
      };
    }
    const turn = {
      id: newId(),
      jobId: jobId ? String(jobId) : null,
      question: String(question),
      context: context ? String(context) : null,
      state: 'pending',
      asked: now,
      finished: null,
      result: null,
      error: null,
    };
    session.turns.push(turn);
    if (session.turns.length > MAX_TURNS) session.turns = session.turns.slice(-MAX_TURNS);
    session.updated = now;
    writeJson(sessionPath(userName, session.id), session);
    return { session, turn };
  }

  function updateTurn(userName, sessionId, turnId, patch) {
    const session = load(userName, sessionId);
    if (!session) return null;
    const turn = session.turns.find((t) => t.id === turnId);
    if (!turn) return null;
    Object.assign(turn, patch);
    if (patch.state && patch.state !== 'pending' && !turn.finished) turn.finished = new Date().toISOString();
    session.updated = new Date().toISOString();
    writeJson(sessionPath(userName, session.id), session);
    return turn;
  }

  function remove(userName, id) {
    const file = sessionPath(userName, id);
    if (!file) return false;
    try {
      fs.unlinkSync(file);
      return true;
    } catch (e) {
      return false;
    }
  }

  // Every turn still waiting for its answer, across all accounts - what a
  // restarted server has to pick up polling for.
  function pendingTurns() {
    const found = [];
    let dirs;
    try {
      dirs = fs.readdirSync(rootDir);
    } catch (e) {
      return found;
    }
    dirs.forEach((dir) => {
      let files;
      try { files = fs.readdirSync(path.join(rootDir, dir)).filter((n) => n.endsWith('.json')); } catch (e) { return; }
      files.forEach((name) => {
        const session = readJson(path.join(rootDir, dir, name));
        if (!session || !Array.isArray(session.turns)) return;
        session.turns.forEach((turn) => {
          if (turn.state === 'pending') {
            found.push({ owner: session.owner, sessionId: session.id, turnId: turn.id, jobId: turn.jobId, asked: turn.asked });
          }
        });
      });
    });
    return found;
  }

  return { list, load, appendTurn, updateTurn, remove, pendingTurns, userDir, ID, rootDir };
}

const config = require('./config');

module.exports = Object.assign(
  createStore(path.join(config.CONFIG_DIR, 'council-sessions')),
  { createStore },
);
