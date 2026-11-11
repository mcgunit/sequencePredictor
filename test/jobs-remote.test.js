'use strict';
// The Sunday controls on the DL360 (jobs.js, the owner's decision of 11 Nov 2026): a control goes to the DL360's own
// queue when config/workers.json has a cpu entry, runs there beside this server's queue, is recorded in the
// scheduler's history with worker 'cpu', and - when the DL360 does not come up (exit 75) - runs here instead, with the
// controls behind it, in their order. On a copy of jobs.js in a temp dir, the schedule dry (only runs started by hand
// start), worker/remote-control.sh replaced by a stand-in that only prints and ends with the code it is given: nothing
// is copied anywhere and no control runs. process.lock is held by this test, so nothing queued here starts either.
// Run: TZ=Europe/Brussels node test/jobs-remote.test.js
process.env.TZ = 'Europe/Brussels';
delete process.env.SCHEDULER;
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

let checks = 0;
const ok = (cond, what) => { assert.ok(cond, what); checks += 1; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(cond, ms = 8000) {
  const end = Date.now() + ms;
  while (Date.now() < end) { if (cond()) return true; await wait(100); }
  return cond();
}

(async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'jobs-remote-'));
  fs.copyFileSync(path.join(__dirname, '..', 'jobs.js'), path.join(dir, 'jobs.js'));
  fs.writeFileSync(path.join(dir, 'config.js'), `module.exports = { CONFIG_DIR: ${JSON.stringify(path.join(dir, 'config'))} };`);
  fs.mkdirSync(path.join(dir, 'config'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'config', 'scheduler.json'), JSON.stringify({ version: 1, occurrences: {}, queue: [], history: [], current: null,
    baselinedAt: 'x', baselinedPlans: { weeklyTuning: 'x', weeklyControls: 'x' } }));   // a scheduler.json from before: no remote part
  fs.writeFileSync(path.join(dir, 'process.lock'), String(process.pid));                 // held: nothing queued here starts
  const code = path.join(dir, 'code');
  const fake = path.join(dir, 'fake-remote.sh');
  // the stand-in: its arguments (run id, snapshot, script, arguments) into its log, then the code in ./code
  fs.writeFileSync(fake, `#!/usr/bin/env bash\necho "fake $*"\necho "host=$WORKER_HOST"\nsleep 0.3\nexit "$(cat ${JSON.stringify(code)})"\n`);
  const quiet = console.log;
  console.log = () => {};
  let copy;
  let cfg = { kind: 'cpu', host: 'root@dl360.test', dir: '/opt/w', python: '/opt/w/venv/bin/python' };
  try {
    copy = require(path.join(dir, 'jobs.js'));
    const c = copy._internals;
    c.useWorker(() => cfg, (x) => ({ WORKER_HOST: x.host }), fake);
    copy.start();
    copy.stop();   // the test drives the ticks itself
    const st = () => c.state();
    ok(st().remote && st().remote.current === null && Array.isArray(st().remote.queue), 'an old scheduler.json gets an empty remote part');
    ok(c.toWorker('marketRows') && !c.toWorker('predictor') && !c.toWorker('statisticalTuning'), 'only the controls go to the DL360');

    // 1. a control by hand: queued for the DL360, started there, recorded with worker cpu
    fs.writeFileSync(code, '0');
    const r1 = copy.runNow('marketRows', 'tester');
    ok(r1.ok && /DL360/.test(r1.message), `Run now says the DL360: ${r1.message}`);
    ok(st().queue.length === 0, 'nothing in this server\'s queue');
    ok(st().remote.current && st().remote.current.key === 'marketRows', 'it started on the DL360 at once (by hand, in dry mode)');
    ok(copy.runRefusal('marketRows').status === 409 && /DL360/.test(copy.runRefusal('marketRows').message), 'a second Run now is refused, saying where it runs');
    const s1 = copy.status();
    ok(s1.remote && s1.remote.configured && s1.remote.host === 'dl360.test' && s1.remote.current.key === 'marketRows', 'status() shows it under remote, without the user');
    ok(s1.current === null, 'and this server runs nothing');
    ok(copy.remoteBusy() && copy.remoteBusy().running === s1.remote.current.name, 'workers.js sees it (no shutting the DL360 down meanwhile)');
    ok(await until(() => { c.pollRemote(); return !st().remote.current; }), 'it ends');
    const h1 = st().history[0];
    ok(h1.key === 'marketRows' && h1.exit === 0 && h1.worker === 'cpu' && h1.host === 'dl360.test', `recorded in the history with worker cpu: ${JSON.stringify(h1)}`);
    const log1 = fs.readFileSync(path.join(dir, 'log', 'marketRows.log'), 'utf-8');
    ok(/^=== scheduler: Weekly controls: market rows under a proper score started \d{4}-\d\d-\d\d \d\d:\d\d:\d\d( \S+)? UTC ===$/m.test(log1), 'the log has the scheduler\'s start marker (admin.js slices by it)');
    ok(/fake marketRows-dl360-\S+ marketRows-dl360-\S+ MarketRows\.py -g crypto,shares,cryptoweek,sharesweek -d 250/.test(log1), `the wrapper got the run id, a snapshot, the script and its arguments: ${log1}`);
    ok(/host=root@dl360\.test/.test(log1), 'and the worker\'s settings in its environment');
    ok(/=== scheduler: .* finished status=0 after \d+s ===/.test(log1), 'and the finish marker');
    ok(!fs.readdirSync(path.join(dir, 'config', 'runs')).length, 'its pid and exit files are cleaned up');

    // 2. the DL360 does not come up: this control and the one behind it are queued here, in their order
    fs.writeFileSync(code, '75');
    copy.runNow('randomnessDiscrimination', 'tester');
    copy.runNow('featureControl', 'tester');
    ok(st().remote.current.key === 'randomnessDiscrimination' && st().remote.queue.map((q) => q.key).join() === 'featureControl', 'one runs, one waits behind it');
    ok(copy.status().jobs.find((j) => j.key === 'featureControl').refusal !== null, 'the waiting one cannot be started twice');
    ok(await until(() => { c.pollRemote(); return !st().remote.current; }), 'the not-started run ends');
    const h2 = st().history[0];
    ok(h2.key === 'randomnessDiscrimination' && h2.exit === 75 && /run on this server instead/.test(h2.note), `recorded with why: ${h2.note}`);
    ok(st().queue.map((q) => q.key).join() === 'randomnessDiscrimination,featureControl', `both queued here, in order: ${st().queue.map((q) => q.key).join()}`);
    ok(st().queue.every((q) => q.manual && /DL360 did not come up/.test(q.reason)), 'still by hand, and the reason says why they are here');
    ok(st().remote.queue.length === 0 && !st().remote.current, 'nothing left for the DL360');
    c.startNextRemote();
    ok(!st().remote.current, 'and nothing goes back to it');
    ok(copy.cancelQueued('featureControl', 'tester').ok && copy.cancelQueued('randomnessDiscrimination', 'tester').ok, 'cancelled here');

    // 3. cancel and stop on the DL360's side
    fs.writeFileSync(code, '0');
    fs.writeFileSync(fake, `#!/usr/bin/env bash\ntrap 'echo stopped; exit 130' TERM\nsleep 30 & wait $!\n`);
    copy.runNow('randomnessDiscrimination', 'tester');
    copy.runNow('marketRows', 'tester');
    const cancel = copy.cancelQueued('marketRows', 'tester');
    ok(cancel.ok && st().remote.queue.length === 0, 'a control waiting for the DL360 is cancelled');
    ok(await until(() => { c.pollRemote(); return Boolean(st().remote.current.pid); }), 'the running one reports its PID');
    const kills = [];
    const stop = copy.stopRun(st().remote.current.runId, 'tester', { kill: (pid, sig) => kills.push([pid, sig]), isAlive: () => true });
    ok(stop.ok && /DL360/.test(stop.message) && kills.length === 1 && kills[0][0] > 0 && kills[0][1] === 'SIGTERM', `Stop signals the wrapper alone (a positive PID): ${JSON.stringify(kills)}`);
    process.kill(st().remote.current.pid, 'SIGTERM');
    ok(await until(() => { c.pollRemote(); return !st().remote.current; }), 'the stopped run ends');
    ok(st().history[0].exit === 130 && st().queue.length === 0, 'recorded as stopped (130), and not moved here');

    // 4. a restart adopts a run still going, and closes one that is gone
    fs.writeFileSync(fake, `#!/usr/bin/env bash\nsleep 2\nexit 0\n`);
    copy.runNow('marketRows', 'tester');
    ok(await until(() => { c.pollRemote(); return Boolean(st().remote.current.pid); }), 'started');
    c.adoptRemote();
    ok(st().remote.current && st().remote.current.key === 'marketRows', 'adopted while it runs');
    ok(await until(() => { c.pollRemote(); return !st().remote.current; }), 'and finished afterwards');
    ok(st().history[0].exit === 0, 'with its exit code');

    // 5. no cpu entry: the controls stay here; a control waiting for a DL360 that is taken out of the file runs here
    cfg = null;
    ok(!c.toWorker('marketRows'), 'without the DL360 a control is not sent there');
    copy.runNow('marketRows', 'tester');
    ok(st().queue.map((q) => q.key).join() === 'marketRows' && st().remote.queue.length === 0, 'Run now queues it here, as before');
    ok(copy.status().remote === null, 'and status() has no remote part');
    copy.cancelQueued('marketRows', 'tester');
    st().remote.queue.push({ key: 'featureControl', reason: 'test', queuedAt: Date.now(), manual: true, occurrence: null });
    c.startNextRemote();
    ok(st().queue.map((q) => q.key).join() === 'featureControl' && /no longer set up/.test(st().queue[0].reason), 'one left waiting for it moves here');

    // 6. a plan occurrence: the four controls of a first Sunday go to the DL360 with the occurrence (their one copy of the data)
    cfg = { kind: 'cpu', host: 'root@dl360.test', dir: '/opt/w', python: '/opt/w/venv/bin/python' };
    st().queue.length = 0;
    process.env.SCHEDULER = 'on';   // a plan queues nothing in dry mode
    fs.writeFileSync(fake, `#!/usr/bin/env bash\necho "fake $*"\nexit 0\n`);
    c.enqueuePlan('weeklyControls', 'weeklyControls@2026-W44', 'test', new Date('2026-11-01T10:30:00'));
    ok(st().queue.length === 0, 'nothing for this server');
    ok(st().remote.queue.map((q) => q.key).join() === 'nullControls,randomnessDiscrimination,featureControl,marketRows', 'all four for the DL360, the band first');
    ok(st().remote.queue.every((q) => q.occurrence === 'weeklyControls@2026-W44' && !q.manual), 'each with its occurrence, not by hand');
    c.startNextRemote();
    ok(st().remote.current && st().remote.current.key === 'nullControls' && st().remote.queue.length === 3, 'the band starts there, three wait');
    ok(await until(() => { c.pollRemote(); return !st().remote.current; }), 'it ends');
    ok(/fake nullControls-dl360-\S+ weeklyControls-2026-W44 NullControls\.py/.test(fs.readFileSync(path.join(dir, 'log', 'nullControls.log'), 'utf-8')),
      'its snapshot is the occurrence, so the four share one copy of the data');
    c.startNextRemote();
    ok(st().remote.current && st().remote.current.key === 'randomnessDiscrimination', 'the next one follows, in the plan\'s order');
    ok(await until(() => { c.pollRemote(); return !st().remote.current; }), 'and ends');
    ok(/fake randomnessDiscrimination-dl360-\S+ weeklyControls-2026-W44 /.test(fs.readFileSync(path.join(dir, 'log', 'randomnessDiscrimination.log'), 'utf-8')), 'with the same snapshot');
    st().remote.queue.length = 0;
    delete process.env.SCHEDULER;
  } finally {
    console.log = quiet;
    if (copy) copy.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
  console.log(`jobs.js, the controls on the DL360: ${checks} checks passed`);
})().catch((e) => { console.error(e); process.exit(1); });
