// Rules that keep announcements.js safe to edit (see the header there).
// Run with: npm test
const assert = require('assert');
const { ANNOUNCEMENTS, TOUR } = require('../announcements');

const ID = /^\d{4}-\d{2}-\d{2}-[a-z0-9-]+$/;
const seen = new Set();
let checks = 0;
const ok = (cond, what) => { assert(cond, what); checks += 1; };

ANNOUNCEMENTS.forEach((a, i) => {
  ok(ID.test(a.id), `announcement ${i}: id "${a.id}" must look like 2026-09-20-short-name`);
  ok(!seen.has(a.id), `announcement ${i}: id "${a.id}" is used twice - ids are permanent and unique`);
  seen.add(a.id);
  ok(!Number.isNaN(Date.parse(a.date)), `announcement ${a.id}: date does not parse`);
  ok(a.id.startsWith(a.date), `announcement ${a.id}: id must start with its date`);
  ok(['major', 'minor'].includes(a.level), `announcement ${a.id}: level must be major or minor`);
  ok(['all', 'admin'].includes(a.audience), `announcement ${a.id}: audience must be all or admin`);
  ok(typeof a.title === 'string' && a.title.length > 0 && a.title.length < 120, `announcement ${a.id}: title missing or too long`);
  ok(Array.isArray(a.body) && a.body.length > 0 && a.body.every((line) => typeof line === 'string'), `announcement ${a.id}: body must be an array of strings`);
  ok(!/[<>]/.test(a.title + a.body.join('')), `announcement ${a.id}: no HTML in the text - it is escaped, so tags would show as tags`);
  if (a.link) {
    ok(typeof a.link.href === 'string' && a.link.href.startsWith('/'), `announcement ${a.id}: link.href must be a path on this site`);
    ok(typeof a.link.label === 'string' && a.link.label.length > 0, `announcement ${a.id}: link.label missing`);
  }
});

const dates = ANNOUNCEMENTS.map((a) => Date.parse(a.date));
ok(dates.every((d, i) => i === 0 || d <= dates[i - 1]), 'announcements must be newest first');

ok(typeof TOUR.title === 'string' && typeof TOUR.intro === 'string', 'the tour needs a title and an intro');
ok(Array.isArray(TOUR.steps) && TOUR.steps.length > 0, 'the tour needs steps');
TOUR.steps.forEach((step, i) => {
  ok(['all', 'admin'].includes(step.audience), `tour step ${i}: audience must be all or admin`);
  ok(typeof step.title === 'string' && typeof step.body === 'string' && step.body.length > 0, `tour step ${i}: title and body must be text`);
  ok(!/[<>]/.test(step.title + step.body), `tour step ${i}: no HTML in the text`);
});
ok(TOUR.steps.some((s) => s.audience === 'all'), 'the tour must have something for a plain user');

console.log(`announcements.js: ${checks} checks passed (${ANNOUNCEMENTS.length} announcements, ${TOUR.steps.length} tour steps)`);
