'use strict';
// pipeline.js: is a pipeline job holding process.lock, and which one (moved out of server.js on 24 Oct 2026). The app's
// Home and Lottery (lottery.js `running`) read it; the old pages' banner that read it too was removed in cleanup batch
// 6b (26 Oct 2026). Here: no lock, an empty or unreadable one, a stale one (the process is gone), every job by its
// script's name, an unknown script, the real /proc for this test's own process, what a reader gets (the job's words and
// since when, never the pid or a path), and that lottery.js reads this module and no server file keeps a copy of its own.
//   node test/pipeline.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const pipeline = require('../pipeline');

const REPO = path.join(__dirname, '..');
let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pipeline-'));
const lock = path.join(tmp, 'process.lock');
const since = new Date('2026-10-24T03:15:00Z');
const writeLock = (text) => { fs.writeFileSync(lock, text); fs.utimesSync(lock, since, since); };
// a command line as /proc gives it: the arguments separated by NUL
const procOf = (cmdlines) => (pid) => {
  if (!(pid in cmdlines)) throw new Error(`ENOENT: /proc/${pid}/cmdline`);
  return cmdlines[pid];
};

try {
  ok(pipeline.PIPELINE_LOCK === path.join(REPO, 'process.lock'), 'the lock is process.lock in the repo root, the one every Python entry point takes');

  // --- nothing running ------------------------------------------------------------------------------------------------
  ok(pipeline.pipelineStatus(lock) === null && pipeline.runningNow(lock) === null, 'no lock file: nothing running');
  for (const text of ['', '\n', 'abc', '0']) {
    writeLock(text);
    ok(pipeline.pipelineStatus(lock, procOf({})) === null, `a lock without a pid (${JSON.stringify(text)}): nothing running`);
  }
  writeLock('4242\n');
  ok(pipeline.pipelineStatus(lock, procOf({})) === null && pipeline.runningNow(lock, procOf({})) === null,
    'a stale lock (its process is gone): nothing running');

  // --- every job by its script's name, an unknown one in general words ------------------------------------------------
  Object.entries(pipeline.JOB_NAMES).forEach(([script, what]) => {
    const s = pipeline.pipelineStatus(lock, procOf({ 4242: `python3\0${script}\0--game\0lotto\0` }));
    ok(s && s.pid === 4242 && s.since === since.getTime() && s.what === what, `${script}: "${what}" since the lock was written`);
  });
  ok(pipeline.JOB_NAMES['Predictor.py'] === 'Today\'s predictions are being computed' && Object.keys(pipeline.JOB_NAMES).length === 12,
    'the twelve jobs the old banner named, in its words');
  ok(pipeline.pipelineStatus(lock, procOf({ 4242: 'python3\0SomethingElse.py\0' })).what === 'A pipeline job is running', 'an unknown script: a pipeline job is running');
  ok(pipeline.pipelineStatus(lock, procOf({ 4242: '/usr/bin/python3\0/root/sequencePredictor/HyperoptBoost.py\0' })).what === 'Weekly tuning: boosting models',
    'a script started by its full path is still named');

  // --- the real /proc: this test's own process ------------------------------------------------------------------------
  writeLock(`${process.pid}\n`);
  const own = pipeline.pipelineStatus(lock);
  ok(own && own.pid === process.pid && own.what === 'A pipeline job is running', `the real /proc: this process holds the lock (${own && own.what})`);

  // --- what a reader gets: the job's words and since when, never the pid or a path -------------------------------------
  const r = pipeline.runningNow(lock);
  ok(r && Object.keys(r).join() === 'what,since' && r.what === 'A pipeline job is running' && r.since === '2026-10-24T03:15:00Z',
    `runningNow: {what, since} only, since in ISO to the second (${JSON.stringify(r)})`);
  writeLock('4242');
  ok(JSON.stringify(pipeline.runningNow(lock, procOf({ 4242: 'python3\0Predictor.py\0' }))) === '{"what":"Today\'s predictions are being computed","since":"2026-10-24T03:15:00Z"}',
    'runningNow names the job as the old banner did');

  // --- lottery.js reads this module; no server file keeps a copy of its own --------------------------------------------
  const source = (file) => fs.readFileSync(path.join(REPO, file), 'utf-8');
  ok(/const pipeline = require\('\.\/pipeline'\);/.test(source('lottery.js')) && /pipeline\.runningNow\(/.test(source('lottery.js')),
    'lottery.js requires pipeline.js and asks runningNow()');
  ['server.js', 'lottery.js'].forEach((file) => ok(!/function pipelineStatus\(|const JOB_NAMES|const PIPELINE_LOCK/.test(source(file)),
    `${file} keeps no copy of its own`));
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}

console.log(`pipeline: ${checks} checks passed`);
