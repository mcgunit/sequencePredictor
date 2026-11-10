'use strict';
// The worker lanes (workers.js): their settings, the queue, a real launch through setsid of a stand-in for
// worker/remote-backtest.sh (no ssh, no worker), the exit code read back, Stop (SIGTERM to the script alone), the kill
// switch and adoption after a restart; the DL360's power with a stand-in for worker/power.sh - all in a temporary
// folder.
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
  `echo "$1 $2 $WORKER_HOST $WORKER_DIR $WORKER_PYTHON $WORKER_REMOTE_ENV" >> ${tmp}/calls`,
  'trap \'echo stopped >> ' + tmp + '/calls; exit 130\' TERM',
  `for i in $(seq 1 $(cat ${tmp}/wait 2>/dev/null || echo 0)); do sleep 0.1 & wait $!; done`,
  'exit 0',
].join('\n'));
// the power stand-in: answers status with a fixed line, records wake/off and its settings
const fakePower = path.join(tmp, 'fake-power.sh');
fs.writeFileSync(fakePower, [
  '#!/usr/bin/env bash',
  `echo "$1 $WORKER_HOST $WORKER_ILO_ENV $WORKER_IDLE_MINUTES" >> ${tmp}/power`,
  'case "$1" in status) echo \'{"power":"on","reachable":true,"woken":true,"wokenAt":1794330000,"lastBusy":1794330600,"idleMinutes":10}\' ;; wake) sleep 0.3 ;; esac',
].join('\n'));
workers._internals.useDirs(cfgDir, logDir, fake, fakePower);
const G = workers.lane('gpu');
const C = workers.lane('cpu');
const W = {
  state: () => { G.backtestState(); return G.state; }, poll: () => G.poll(), startNext: () => G.startNext(),
  reset: () => { G.state = { version: 1, queue: [], current: null, history: [] }; },
};
let cpuCfg = null;
const writeCfg = (gpu) => fs.writeFileSync(path.join(cfgDir, 'workers.json'), JSON.stringify({ gpu, ...(cpuCfg ? { cpu: cpuCfg } : {}) }));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(cond, ms = 8000) {
  const end = Date.now() + ms;
  while (Date.now() < end) { W.poll(); if (cond()) return true; await sleep(100); }
  return false;
}

(async () => {
  // --- settings ---
  ok(workers.configOf('gpu') === null, 'no file: no GPU worker');
  writeCfg({ host: 'root@10.0.0.9' });
  let c = workers.configOf('gpu');
  ok(c && c.host === 'root@10.0.0.9' && c.dir === '/opt/sequencePredictor-worker' && c.python === '/opt/sequencePredictor-worker/venv/bin/python' && c.gpuMemoryMb === 6144, 'defaults');
  writeCfg({ host: 'root@10.0.0.9; rm -rf /' });
  ok(workers.configOf('gpu') === null, 'a host with shell characters is refused');
  writeCfg({ host: 'root@10.0.0.9', enabled: false });
  ok(workers.configOf('gpu') === null, 'enabled false: none');
  writeCfg({ host: 'root@10.0.0.9', dir: '/srv/w', gpuMemoryMb: 4096 });
  c = workers.configOf('gpu');
  ok(c.python === '/srv/w/venv/bin/python' && c.gpuMemoryMb === 4096 && workers.configOf('cpu') === null, 'own folder and cap; no cpu entry, no CPU worker');
  ok(workers.KINDS.gpu.budget(3) === 3600 && workers.KINDS.gpu.budget(30) === 27900 && workers.KINDS.cpu.budget(1000) === 12 * 3600, 'budgets: GPU 15 min + 15 min a draw, CPU 12 h');

  // --- a run, launched for real (setsid), its exit code read back ---
  W.reset();
  fs.writeFileSync(path.join(tmp, 'wait'), '3');
  let r = G.queue('bt-1', 'alice', 3);
  ok(r.ok && r.started && W.state().current.id === 'bt-1', `started at once: ${r.message}`);
  r = G.queue('bt-2', 'bob', 30);
  ok(r.ok && !r.started && r.place === 1 && /number 1/.test(r.message), `the second waits: ${r.message}`);
  ok(!G.queue('bt-2', 'bob', 30).ok, 'not twice');
  const s = G.backtestState();
  ok(s.configured && s.host === '10.0.0.9' && s.current.id === 'bt-1' && s.queued[0] === 'bt-2', 'the state backtests.js reads (host without the user)');
  ok(await until(() => W.state().history.length === 1), 'the first ends');
  ok(W.state().history[0].exit === 0 && W.state().history[0].id === 'bt-1', 'exit 0 recorded');
  const calls = fs.readFileSync(path.join(tmp, 'calls'), 'utf-8').trim().split('\n');
  ok(calls[0] === 'bt-1 3600 root@10.0.0.9 /srv/w /srv/w/venv/bin/python CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=false SP_GPU_MEMORY_MB=4096', `the script's arguments and settings: ${calls[0]}`);
  // --- Stop ---
  fs.writeFileSync(path.join(tmp, 'wait'), '200');
  W.startNext();
  ok(W.state().current && W.state().current.id === 'bt-2', 'the next one starts');
  ok(await until(() => W.state().current && W.state().current.pid), 'its PID is known');
  const cur = W.state().current;
  ok(!G.stopRun('nope', 'x').ok, 'Stop names the run');
  r = G.stopRun(cur.runId, 'carol');
  ok(r.ok, `stop: ${r.message}`);
  ok(await until(() => !W.state().current), 'it ends');
  ok(W.state().history[0].exit === 130 && /stopped/.test(fs.readFileSync(path.join(tmp, 'calls'), 'utf-8')), 'the script got the SIGTERM (130)');

  // --- cancel, the kill switch, no settings ---
  W.reset();
  W.state().queue.push({ id: 'q-1', by: 'x', days: 3 });
  ok(G.cancel('q-1', 'dora').ok && !W.state().queue.length && !G.cancel('q-1', 'dora').ok, 'cancel a queued one, once');
  fs.writeFileSync(path.join(cfgDir, 'scheduler.disabled'), '');
  ok(/disabled/.test(G.queue('q-2', 'x', 3).message), 'the kill switch refuses');
  fs.unlinkSync(path.join(cfgDir, 'scheduler.disabled'));
  fs.unlinkSync(path.join(cfgDir, 'workers.json'));
  r = G.queue('q-3', 'x', 3);
  ok(!r.ok && r.status === 503 && /no GPU worker/.test(r.message), 'no settings: 503');

  // --- adoption after a restart: an alive run is kept, a gone one is closed ---
  writeCfg({ host: 'root@10.0.0.9' });
  fs.writeFileSync(path.join(tmp, 'wait'), '30');
  W.reset();
  G.queue('a-1', 'x', 3);
  ok(await until(() => W.state().current && W.state().current.pid), 'running');
  workers.stop();
  workers.start();   // reads workers-state.json again, as after a restart
  workers.stop();
  ok(W.state().current && W.state().current.id === 'a-1', 'an alive run is adopted');
  ok(await until(() => !W.state().current), 'and read to its end');
  const st = JSON.parse(fs.readFileSync(path.join(cfgDir, 'workers-state.json'), 'utf-8'));   // the GPU lane's file, as in production
  st.current = { id: 'gone', runId: 'gpu-backtest-gone', pid: 999999, goneAt: Date.now() - 60000, pidFile: path.join(tmp, 'x.pid'), exitFile: path.join(tmp, 'x.exit'), startedAt: Date.now() - 60000, host: 'h' };
  st.queue = [];
  fs.writeFileSync(path.join(cfgDir, 'workers-state.json'), JSON.stringify(st));
  workers.start();
  workers.stop();
  ok(!W.state().current && /restarted/.test(W.state().history[0].note), 'a gone run is closed on start');

  // --- the CPU lane (the DL360): its own queue and state file, the CPU environment, the iLO settings ---
  cpuCfg = { host: 'root@10.0.0.8', iloEnv: '/root/.config/x/ilo.env', idleMinutes: 15 };
  writeCfg({ host: 'root@10.0.0.9' });
  c = workers.configOf('cpu');
  ok(c && c.iloEnv === '/root/.config/x/ilo.env' && c.idleMinutes === 15 && c.python === '/opt/sequencePredictor-worker/venv/bin/python', 'the cpu settings');
  fs.writeFileSync(path.join(tmp, 'wait'), '2');
  r = C.queue('c-1', 'gus', 90);
  ok(r.ok && r.started && /CPU worker/.test(r.message) && !G.backtestState().current, `the CPU lane runs on its own: ${r.message}`);
  ok(await (async () => { const end = Date.now() + 8000; while (Date.now() < end) { C.poll(); if (!C.backtestState().current) return true; await sleep(100); } return false; })(), 'the CPU run ends');
  const cl = fs.readFileSync(path.join(tmp, 'calls'), 'utf-8').trim().split('\n').pop();
  ok(cl === 'c-1 43200 root@10.0.0.8 /opt/sequencePredictor-worker /opt/sequencePredictor-worker/venv/bin/python CUDA_VISIBLE_DEVICES=-1', `the CPU run's settings: ${cl}`);
  ok(fs.existsSync(path.join(cfgDir, 'workers-cpu-state.json')) && C.state.history[0].exit === 0, 'its own state file and history');

  // --- the DL360's power: status (read through the stand-in), on, off (refused while a backtest waits) ---
  workers._internals.resetPower();
  let p = await workers.powerStatus({ waitMs: 3000 });
  ok(p && p.power === 'on' && p.reachable === true && p.woken && p.idleMinutes === 15 && p.lastBusy === 1794330600 * 1000, 'the power status');
  ok(/root@10.0.0.8 \/root\/.config\/x\/ilo.env 15/.test(fs.readFileSync(path.join(tmp, 'power'), 'utf-8')), 'power.sh gets the host, the iLO file and the idle minutes');
  r = workers.powerOn('hana');
  ok(r.ok && /5 to 10 minutes/.test(r.message) && !workers.powerOn('hana').ok, 'switch on, once at a time');
  await sleep(800);
  ok(/wake/.test(fs.readFileSync(path.join(tmp, 'power'), 'utf-8')) && workers.powerOn('hana').ok, 'the wake ran; a second one may start after it');
  await sleep(800);
  C.state.queue.push({ id: 'c-2', by: 'x', days: 90 });
  ok(/running or queued/.test(workers.powerOff('ivy').message), 'off refused while a backtest waits');
  C.state.queue = [];
  r = workers.powerOff('ivy');
  ok(r.ok && /council/.test(r.message), 'off: the council goes down with it');
  await sleep(500);
  ok(/off/.test(fs.readFileSync(path.join(tmp, 'power'), 'utf-8')), 'power.sh off ran');
  cpuCfg = { host: 'root@10.0.0.8' };
  writeCfg({ host: 'root@10.0.0.9' });
  ok(!workers.powerOn('x').ok && /iLO/.test((await workers.powerStatus()).note), 'no iLO settings: never switched by this server');

  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(`workers.test.js: ${checks} checks passed`);
})().catch((e) => { console.error(e); process.exit(1); });
