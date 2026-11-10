'use strict';
// The GPU worker's queue (workers.js): its settings, the queue, a real launch through setsid of a stand-in for
// worker/gpu-backtest.sh (no ssh, no worker), the exit code read back, Stop (SIGTERM to the script alone), the kill
// switch and adoption after a restart - all in a temporary folder.
//   node test/workers.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const workers = require('../workers');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'workers-'));
const cfgDir = path.join(tmp, 'config');
const logDir = path.join(tmp, 'log');
fs.mkdirSync(cfgDir, { recursive: true });
// the stand-in: writes its arguments, waits (a file says how long), exits 0, or 130 on SIGTERM
const fake = path.join(tmp, 'fake-gpu-backtest.sh');
fs.writeFileSync(fake, [
  '#!/usr/bin/env bash',
  `echo "$1 $2 $WORKER_HOST $WORKER_DIR $WORKER_PYTHON $WORKER_GPU_MB" >> ${tmp}/calls`,
  'trap \'echo stopped >> ' + tmp + '/calls; exit 130\' TERM',
  `for i in $(seq 1 $(cat ${tmp}/wait 2>/dev/null || echo 0)); do sleep 0.1 & wait $!; done`,
  'exit 0',
].join('\n'));
workers._internals.useDirs(cfgDir, logDir, fake);
const W = workers._internals;
const writeCfg = (gpu) => fs.writeFileSync(path.join(cfgDir, 'workers.json'), JSON.stringify({ gpu }));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(cond, ms = 8000) {
  const end = Date.now() + ms;
  while (Date.now() < end) { W.poll(); if (cond()) return true; await sleep(100); }
  return false;
}

(async () => {
  // --- settings ---
  ok(workers.gpuConfig() === null, 'no file: no GPU worker');
  writeCfg({ host: 'root@10.0.0.9' });
  let c = workers.gpuConfig();
  ok(c && c.host === 'root@10.0.0.9' && c.dir === '/opt/sequencePredictor-worker' && c.python === '/opt/sequencePredictor-worker/venv/bin/python' && c.gpuMemoryMb === 6144, 'defaults');
  writeCfg({ host: 'root@10.0.0.9; rm -rf /' });
  ok(workers.gpuConfig() === null, 'a host with shell characters is refused');
  writeCfg({ host: 'root@10.0.0.9', enabled: false });
  ok(workers.gpuConfig() === null, 'enabled false: none');
  writeCfg({ host: 'root@10.0.0.9', dir: '/srv/w', gpuMemoryMb: 4096 });
  c = workers.gpuConfig();
  ok(c.python === '/srv/w/venv/bin/python' && c.gpuMemoryMb === 4096, 'own folder and cap');
  ok(workers.budgetFor(3) === 3600 && workers.budgetFor(30) === 27900, 'budget: 15 min + 15 min a draw');

  // --- a run, launched for real (setsid), its exit code read back ---
  W.reset();
  fs.writeFileSync(path.join(tmp, 'wait'), '3');
  let r = workers.queue('bt-1', 'alice', 3);
  ok(r.ok && r.started && W.state().current.id === 'bt-1', `started at once: ${r.message}`);
  r = workers.queue('bt-2', 'bob', 30);
  ok(r.ok && !r.started && r.place === 1 && /number 1/.test(r.message), `the second waits: ${r.message}`);
  ok(!workers.queue('bt-2', 'bob', 30).ok, 'not twice');
  const s = workers.backtestState();
  ok(s.configured && s.host === '10.0.0.9' && s.current.id === 'bt-1' && s.queued[0] === 'bt-2', 'the state backtests.js reads (host without the user)');
  ok(await until(() => W.state().history.length === 1), 'the first ends');
  ok(W.state().history[0].exit === 0 && W.state().history[0].id === 'bt-1', 'exit 0 recorded');
  const calls = fs.readFileSync(path.join(tmp, 'calls'), 'utf-8').trim().split('\n');
  ok(calls[0] === 'bt-1 3600 root@10.0.0.9 /srv/w /srv/w/venv/bin/python 4096', `the script's arguments and settings: ${calls[0]}`);
  // --- Stop ---
  fs.writeFileSync(path.join(tmp, 'wait'), '200');
  W.startNext();
  ok(W.state().current && W.state().current.id === 'bt-2', 'the next one starts');
  ok(await until(() => W.state().current && W.state().current.pid), 'its PID is known');
  const cur = W.state().current;
  ok(!workers.stopRun('nope', 'x').ok, 'Stop names the run');
  r = workers.stopRun(cur.runId, 'carol');
  ok(r.ok, `stop: ${r.message}`);
  ok(await until(() => !W.state().current), 'it ends');
  ok(W.state().history[0].exit === 130 && /stopped/.test(fs.readFileSync(path.join(tmp, 'calls'), 'utf-8')), 'the script got the SIGTERM (130)');

  // --- cancel, the kill switch, no settings ---
  W.reset();
  W.state().queue.push({ id: 'q-1', by: 'x', days: 3 });
  ok(workers.cancel('q-1', 'dora').ok && !W.state().queue.length && !workers.cancel('q-1', 'dora').ok, 'cancel a queued one, once');
  fs.writeFileSync(path.join(cfgDir, 'scheduler.disabled'), '');
  ok(/disabled/.test(workers.queue('q-2', 'x', 3).message), 'the kill switch refuses');
  fs.unlinkSync(path.join(cfgDir, 'scheduler.disabled'));
  fs.unlinkSync(path.join(cfgDir, 'workers.json'));
  r = workers.queue('q-3', 'x', 3);
  ok(!r.ok && r.status === 503 && /no GPU worker/.test(r.message), 'no settings: 503');

  // --- adoption after a restart: an alive run is kept, a gone one is closed ---
  writeCfg({ host: 'root@10.0.0.9' });
  fs.writeFileSync(path.join(tmp, 'wait'), '30');
  W.reset();
  workers.queue('a-1', 'x', 3);
  ok(await until(() => W.state().current && W.state().current.pid), 'running');
  workers.stop();
  workers.start();   // reads workers-state.json again, as after a restart
  workers.stop();
  ok(W.state().current && W.state().current.id === 'a-1', 'an alive run is adopted');
  ok(await until(() => !W.state().current), 'and read to its end');
  const st = JSON.parse(fs.readFileSync(path.join(cfgDir, 'workers-state.json'), 'utf-8'));
  st.current = { id: 'gone', runId: 'gpu-backtest-gone', pid: 999999, goneAt: Date.now() - 60000, pidFile: path.join(tmp, 'x.pid'), exitFile: path.join(tmp, 'x.exit'), startedAt: Date.now() - 60000, host: 'h' };
  st.queue = [];
  fs.writeFileSync(path.join(cfgDir, 'workers-state.json'), JSON.stringify(st));
  workers.start();
  workers.stop();
  ok(!W.state().current && /restarted/.test(W.state().history[0].note), 'a gone run is closed on start');

  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(`workers.test.js: ${checks} checks passed`);
})().catch((e) => { console.error(e); process.exit(1); });
