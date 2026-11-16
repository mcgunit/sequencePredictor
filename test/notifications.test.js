'use strict';
// The administrator's notifications (notifications.js, 16 Nov 2026): the store, one read mark per administrator, the
// dedupe key, the cap, and the two routes on a stand-in app - all in a temporary folder.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const notifications = require('../notifications');

let checks = 0;
const ok = (cond, text) => { checks += 1; assert.ok(cond, text); };

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'notifications-'));
const file = path.join(tmp, 'config', 'notifications.json');
notifications._internals.useFile(file);
const quiet = console.log;
try {
  const hana = { name: 'hana', admin: true };
  const ivy = { name: 'ivy', admin: true };
  ok(notifications.view(hana).of === 0 && notifications.view(hana).unread === 0 && !fs.existsSync(file), 'no file: an empty list, nothing written');

  const t = Date.parse('2026-11-16T06:10:00Z');
  const a = notifications.raise({ kind: 'no-data', level: 'warning', title: 'No data  from the\nlottery API', body: 'Tried again at 07:10.', link: { href: '/admin/jobs', label: 'Jobs & services' }, at: t });
  ok(a && a.id === 'n1' && a.title === 'No data from the lottery API' && a.body.length === 1 && a.link.href === '/admin/jobs', 'a note is kept, its text on one line');
  ok((fs.statSync(file).mode & 0o777) === 0o600, 'the store is the server\'s own (0600)');
  const b = notifications.raise({ kind: 'job-failed', level: 'error', title: 'Crypto daily ended with exit code 1', body: ['one', '', 'two'], link: { href: 'https://elsewhere.example/x', label: 'x' }, at: t + 1000 });
  ok(b.id === 'n2' && b.body.length === 2 && b.link === null, 'empty lines dropped; a link outside the app is not kept');
  ok(notifications.raise({ kind: 'x', title: '   ' }) === null && notifications.raise() === null, 'no title: no note');
  ok(notifications.raise({ kind: 'x', level: 'loud', title: 'Odd level', at: t + 2000 }).level === 'warning', 'an unknown level is a warning');

  let v = notifications.view(hana);
  ok(v.of === 3 && v.unread === 3 && v.items[0].id === 'n3' && v.items[2].id === 'n1' && v.items.every((n) => n.read === false) && !('dedupe' in v.items[0]), 'newest first, all unread');

  // read marks are per administrator
  ok(JSON.stringify(notifications.markRead(hana, ['n1', 'n9', 'n1'])) === '["n1"]', 'marks the ones that exist, once');
  ok(notifications.view(hana).unread === 2 && notifications.view(ivy).unread === 3, 'read for one administrator, still unread for the other');
  ok(notifications.markRead(hana, ['n1']).length === 0, 'marking again changes nothing');
  ok(notifications.markRead(ivy, 'all').length === 3 && notifications.view(ivy).unread === 0 && notifications.view(hana).unread === 2, 'mark all');
  ok(notifications.view({ name: 'anyone', open: true }).unread === 3, 'open access: everything shows as unread');

  // one note per dedupe key per window
  ok(notifications.raise({ kind: 'worker-power', title: 'Still on', dedupe: 'still-on', dedupeMs: 3600 * 1000, at: t + 5000 }), 'a keyed note');
  ok(notifications.raise({ kind: 'worker-power', title: 'Still on', dedupe: 'still-on', dedupeMs: 3600 * 1000, at: t + 60000 }) === null, 'the same key inside its window: not again');
  ok(notifications.raise({ kind: 'worker-power', title: 'Still on', dedupe: 'still-on', dedupeMs: 3600 * 1000, at: t + 2 * 3600 * 1000 }), 'after the window: again');
  ok(notifications.view(ivy).unread === 2, 'a new note is unread for everyone');

  // the cap: the newest KEPT, the read marks of dropped notes go with them
  console.log = () => {};
  for (let i = 0; i < notifications.KEPT + 5; i += 1) notifications.raise({ kind: 'job-failed', title: `Run ${i}`, at: t + 3 * 3600 * 1000 + i });
  console.log = quiet;
  const state = notifications._internals.load();
  ok(state.items.length === notifications.KEPT && state.items[0].title === `Run ${notifications.KEPT + 4}` && state.read.ivy.length === 0 && state.read.hana.length === 0,
    `kept ${notifications.KEPT}, the marks of dropped notes gone`);
  ok(notifications.view(hana).kept === notifications.KEPT, 'the view says how many are kept');

  // a store that cannot be read or written never throws
  fs.writeFileSync(file, '{ not json');
  console.log = () => {};
  ok(notifications.view(hana).of === 0, 'a broken file reads as empty');
  ok(notifications.raise({ kind: 'x', title: 'After a broken file' }).id === 'n1', 'and is written anew');
  notifications._internals.useFile(path.join(tmp, 'config', 'notifications.json', 'under-a-file.json'));
  ok(notifications.raise({ kind: 'x', title: 'Cannot be kept' }) === null, 'a store that cannot be written: null, no throw');
  console.log = quiet;
  notifications._internals.useFile(file);

  ok(JSON.stringify(notifications.idsFrom(['n1', ' n22 ', 'x1', 5, '../n1', null])) === '["n1","n22"]' && notifications.idsFrom('n3')[0] === 'n3', 'ids: the store\'s own form only');

  // --- the routes, on a stand-in app ---
  const routes = {};
  const app = { get: (p, ...h) => { routes[`GET ${p}`] = h; }, post: (p, ...h) => { routes[`POST ${p}`] = h; } };
  const guard = 'guard'; const jsonBody = 'jsonBody'; const csrf = 'csrf';
  const answers = [];
  notifications.install(app, { guard, jsonBody, csrf, safe: (fn) => fn, json: (res, body, status = 200) => { answers.push({ body, status }); } });
  const get = routes['GET /api/admin/notifications'];
  const post = routes['POST /api/admin/notifications/read'];
  ok(get && get[0] === guard && post && post[0] === guard && post[1] === jsonBody && post[2] === csrf, 'both routes behind the admin guard; the write behind the form token');
  const call = (h, req) => { h[h.length - 1](req, {}); return answers.pop(); };
  notifications.raise({ kind: 'slow', title: 'Second', at: Date.now() });
  let r = call(get, { user: hana });
  ok(r.status === 200 && r.body.of === 2 && r.body.unread === 2, 'GET: the list for this administrator');
  r = call(post, { user: hana, body: {} });
  ok(r.status === 400 && /no notification/.test(r.body.error), 'POST without ids: refused');
  r = call(post, { user: hana, body: { ids: ['n1', 'bogus'] } });
  ok(r.status === 200 && r.body.ok && r.body.marked.length === 1 && r.body.unread === 1 && /1 marked as read/.test(r.body.message), 'POST ids: marked, the new count');
  r = call(post, { user: hana, body: { all: true } });
  ok(r.status === 200 && r.body.unread === 0, 'POST all');
  r = call(post, { user: { name: 'anyone', open: true }, body: { all: true } });
  ok(r.status === 409, 'open access: nothing to record it against');
} finally {
  console.log = quiet;
  fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(`notifications.js: ${checks} checks passed`);
