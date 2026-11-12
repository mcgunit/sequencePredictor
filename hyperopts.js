'use strict';
// Hyperopts started from the admin pages (owner, 11 Nov 2026; docs/operations.md, The workers). One tuner on one game
// (and, where the tuner has several rows, one row), run on a worker beside the schedule (no process.lock here):
//   deep learning  HyperoptDeepLearning.py, one model, on the GPU worker (workers.js gpu lane, the deep-learning
//                  backtests' queue) - since 11 Nov 2026
//   statistics, boosting, ensemble, quantum, RL  the other five tuners (owner, 12 Nov 2026), on the CPU worker (the
//                  DL360, workers.js cpu lane, the statistical and boosting backtests' queue) - or on this server in a
//                  folder of its own when the DL360 does not come up (the owner's fallback; worker/remote-hyperopt.sh)
//
// Record only (the owner's decision): the run writes nothing that is served. The worker's copy of
// bestParams_<game>.json is thrown away; the tuner's --result-file (src/TuningResult.py) comes back as
// config/hyperopts/<id>.result.json (per study: trials done, what is scored now, this run's best, the gate's verdict and
// what it would write), and the page shows it. Apply writes that - the gate's, or this run's best trial against the
// gate where the tuner offers it - with the gate's record into bestParams_<game>.json on this server; the next pipeline
// run commits it (its git add -A), as every tuner write. Apply is refused while a job holds process.lock (a tuner or
// the predictor may be writing the same file), and warns when a value it would replace was changed since the run
// started (the served file is snapshotted then).
//
// The variety line (owner, 12 Nov 2026): a setting - config/tuning.json {"varietyLine"} is what the weekly chain uses
// (src/TuningResult.load_variety_line), and every run from this page passes its own (--variety-line, the form's field,
// the setting by default).
//
// Files (all gitignored, under config/hyperopts/): <id>.json the request and its status (the server writes it,
// remote-hyperopt.sh marks running/done/failed/stopped), <id>.result.json the tuner's result, <id>.served.json the
// served file when the run was queued. The run's output: log/hyperopts/<id>.log.
const fs = require('fs');
const path = require('path');

const DIR = path.join('config', 'hyperopts');
const LOG_DIR = path.join('log', 'hyperopts');
const SETTINGS_FILE = path.join('config', 'tuning.json');
const KEPT = 100;
const GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'pick3', 'keno'];
const MARKETS = ['crypto', 'shares', 'cryptoweek', 'sharesweek'];
const POSITIONAL = ['pick3', 'jokerplus', ...MARKETS];
// HyperoptDeepLearning.MODEL_DISPLAY_NAMES (test/hyperopts.test.js checks it against the file)
const MODELS = {
  lstm_model: 'LSTM Base Model',
  tcn_model: 'TCN Base Model',
  unified_lstm_tcn_model: 'UnifiedLstmTcn Model',
  unified_lstm_gru_tcn_model: 'UnifiedLstmGruTcn Model',
  transformer_model: 'Transformer Model',
  gnn_model: 'GNN Model',
  autoencoder_model: 'Autoencoder Model',
};
// the variety line, as a percentage on the form (src/TuningResult.clamp_line: 50-100%, 95 by default)
const VARIETY = { default: 95, min: 50, max: 100, integer: false };
const common = (trials, days, trialTimeout, se, maxDays = 1000) => ({
  trials: { default: trials, min: 1, max: 200, integer: true },
  days: { default: days, min: 5, max: maxDays, integer: true },
  trialTimeout: { default: trialTimeout, min: 60, max: 6 * 3600, integer: true },
  gateSeMargin: { default: se, min: 0, max: 5, integer: false },
});
// the deep-learning form's fields: {default, min, max} - the defaults are the tuner's own (HyperoptDeepLearning.py,
// src/DeepLearningTuning.py)
const FIELDS = {
  trials: { default: 15, min: 1, max: 100, integer: true },
  days: { default: 3, min: 1, max: 30, integer: true },
  trainSeconds: { default: 240, min: 30, max: 3600, integer: true },
  pruneAfterDays: { default: 2, min: 0, max: 30, integer: true },
  trialTimeout: { default: 7200, min: 600, max: 6 * 3600, integer: true },
  gateSeMargin: { default: 1, min: 0, max: 5, integer: false },
};
const num = (v) => String(v);
const gateArgs = (p) => (p.gate ? ['--gate-se-margin', num(p.gateSeMargin)] : ['--no-gate']);
const varietyArg = (p) => ['--variety-line', num(Math.round(p.varietyLine * 10) / 1000)];
// The tuners (test/hyperopts.test.js checks the rows and games against the Python files). Per tuner: its script, its
// lane, its rows (CLI name -> the tuningGate row / display name; '' = every row the tuner has for the game), the games
// a row applies to (rowGames; every game of the tuner otherwise), the form's fields, the gate's reference count (for
// the budget), its arguments and its study keys in the result file.
const TUNERS = {
  deeplearning: {
    label: 'Deep learning', script: 'HyperoptDeepLearning.py', lane: 'gpu', rowWord: 'Model', rows: MODELS, games: GAMES, rowGames: {},
    fields: FIELDS, references: 2,
    args: (p) => ['-g', p.game, '-m', p.model, '-t', num(p.trials), '-d', num(p.days), '--train-seconds', num(p.trainSeconds),
      '--prune-after-days', num(p.pruneAfterDays), '--trial-timeout', num(p.trialTimeout), '--gate', p.gate ? 'true' : 'false',
      '--gate-se-margin', num(p.gateSeMargin), ...varietyArg(p), ...(p.fresh ? ['--fresh'] : [])],
    studyKeys: (p) => [`${p.game}-${p.model}-skill`],   // src/DeepLearningTuning.study_name
  },
  statistics: {
    label: 'Statistics', script: 'HyperoptStatistics.py', lane: 'cpu', rowWord: 'Strategy',
    rows: {
      Markov: 'Markov Model', MarkovMonteCarlo: 'MarkovMonteCarlo Model', MarkovBayesian: 'MarkovBayesian Model',
      MarkovBayesianEnhanced: 'MarkovBayesianEnhanched Model', PoissonMonteCarlo: 'PoissonMonteCarlo Model', PoissonMarkov: 'PoissonMarkov Model',
      LaPlaceMonteCarlo: 'LaplaceMonteCarlo Model', HybridStatistical: 'HybridStatisticalModel', KenoSubsetTuning: 'KenoSubsetTuning',
      Garch: 'GARCH Model', RegimeHmm: 'Regime HMM Model',
    },
    games: [...GAMES, ...MARKETS],
    rowGames: {
      KenoSubsetTuning: ['keno'], Garch: MARKETS, RegimeHmm: MARKETS,
      ...Object.fromEntries(['MarkovBayesian', 'MarkovBayesianEnhanced', 'PoissonMarkov', 'HybridStatistical'].map((r) => [r, [...GAMES, ...MARKETS].filter((g) => !POSITIONAL.includes(g))])),
    },
    fields: common(15, 90, 1200, 1), references: 2,
    args: (p) => ['-g', p.game, '-s', p.model, '-t', num(p.trials), '-d', num(p.days), '--trial-timeout', num(p.trialTimeout), ...gateArgs(p), ...varietyArg(p)],
    studyKeys: (p) => [`${p.game}_${p.model}`],
  },
  boost: {
    label: 'Boosting', script: 'HyperoptBoost.py', lane: 'cpu', rowWord: 'Model',
    rows: {
      XGBoost: 'XGBoost Model', XGBoostMultiLabel: 'XGBoostMultiLabel Model', LightGBM: 'LightGBM Model',
      LightGBMMultiLabel: 'LightGBMMultiLabel Model', CatBoost: 'CatBoost Model', CatBoostMultiLabel: 'CatBoostMultiLabel Model',
    },
    games: GAMES,
    rowGames: Object.fromEntries(['XGBoostMultiLabel', 'LightGBMMultiLabel', 'CatBoostMultiLabel'].map((r) => [r, GAMES.filter((g) => !POSITIONAL.includes(g))])),
    fields: common(15, 90, 1200, 1), references: 2,
    args: (p) => ['-g', p.game, '-s', p.model, '-t', num(p.trials), '-d', num(p.days), '--trial-timeout', num(p.trialTimeout), ...gateArgs(p), ...varietyArg(p)],
    studyKeys: (p) => [`${p.game}_${p.model}`],
  },
  ensemble: {
    label: 'Ensemble', script: 'HyperoptEnsemble.py', lane: 'cpu', rowWord: 'Row', rows: { '': 'SubsetEnsemble Model' }, games: GAMES, rowGames: {},
    // -t is the search's evaluations of row combinations (pure numpy over the stored predictions: minutes)
    fields: { trials: { default: 16384, min: 16, max: 100000, integer: true }, days: { default: 120, min: 10, max: 1000, integer: true },
      gateSeMargin: { default: 1, min: 0, max: 5, integer: false } },
    budget: () => 6 * 3600,
    args: (p) => ['-g', p.game, '-t', num(p.trials), '-d', num(p.days), ...gateArgs(p), ...varietyArg(p)],
    studyKeys: null,   // every study in the result file
  },
  quantum: {
    label: 'Quantum', script: 'HyperoptQuantum.py', lane: 'cpu', rowWord: 'Row',
    rows: { '': 'all three rows', quantum_kernel: 'QuantumMetaLearner Model', quantum_vqc: 'QuantumVQC Model', classical_svm: 'ClassicalSVM Model' },
    games: [...GAMES, 'crypto', 'shares'], rowGames: {},
    fields: common(15, 300, 1800, 2.5), references: 2,
    args: (p) => ['-g', p.game, ...(p.model ? ['-v', p.model] : []), '-t', num(p.trials), '-d', num(p.days), '--trial-timeout', num(p.trialTimeout), ...gateArgs(p), ...varietyArg(p)],
    studyKeys: null,
  },
  rlticket: {
    label: 'RL ticket and position', script: 'HyperoptRLTicket.py', lane: 'cpu', rowWord: 'Rows', rows: { '': 'RL Ticket (lottery) or RL Position (markets)' },
    games: [...GAMES, ...MARKETS], rowGames: {},
    // pure numpy, minutes not hours: no per-trial time limit of its own
    fields: { trials: { default: 20, min: 1, max: 500, integer: true }, days: { default: 40, min: 5, max: 1000, integer: true },
      gateSeMargin: { default: 1, min: 0, max: 5, integer: false } },
    budget: () => 6 * 3600,
    args: (p) => ['-g', p.game, '-t', num(p.trials), '-d', num(p.days), ...gateArgs(p), ...varietyArg(p)],
    studyKeys: null,
  },
};
const tunerOf = (rec) => TUNERS[(rec && rec.tuner) || 'deeplearning'] || TUNERS.deeplearning;
const BUDGET_MAX = 14 * 24 * 3600;
// the run's wall-clock budget: every trial at its timeout, the gate's references (full trials), an hour to start - and
// on the DL360 half an hour more to switch it on
const budgetOf = (p) => {
  const t = tunerOf(p);
  const wake = t.lane === 'cpu' ? 1800 : 0;
  if (t.budget) return Math.min(BUDGET_MAX, t.budget(p) + wake);
  return Math.min(BUDGET_MAX, (p.trials + (p.gate ? t.references : 0)) * p.trialTimeout + 3600 + wake);
};
const STATUSES = ['queued', 'running', 'done', 'failed', 'out of time', 'stopped', 'lost', 'cancelled', 'ended'];

const ID = /^[A-Za-z0-9_-]{1,80}$/;
const fileOf = (root, id, kind = '') => path.join(root, DIR, `${id}${kind ? `.${kind}` : ''}.json`);
const studyKey = (game, model) => `${game}-${model}-skill`;   // src/DeepLearningTuning.study_name
function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf-8')); } catch (e) { return null; }
}
function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(`${file}.tmp`, `${JSON.stringify(value, null, 1)}\n`);
  fs.renameSync(`${file}.tmp`, file);
}
const readRecord = (root, id) => (ID.test(String(id || '')) ? readJson(fileOf(root, id)) : null);

// The owner's variety line (config/tuning.json), as a percentage: VARIETY.default without one
function settings(root) {
  const v = Number((readJson(path.join(root, SETTINGS_FILE)) || {}).varietyLine);
  const pctv = Number.isFinite(v) ? (v > 1 ? v : v * 100) : VARIETY.default;
  return { varietyLine: pctv >= VARIETY.min && pctv <= VARIETY.max ? Math.round(pctv * 10) / 10 : VARIETY.default };
}
function saveSettings(root, body, user, now = new Date()) {
  const v = Number((body || {}).varietyLine);
  if (!Number.isFinite(v) || v < VARIETY.min || v > VARIETY.max) return { ok: false, status: 400, message: `the variety line is a percentage from ${VARIETY.min} to ${VARIETY.max}` };
  const before = settings(root).varietyLine;
  const file = path.join(root, SETTINGS_FILE);
  writeJson(file, { ...(readJson(file) || {}), varietyLine: Math.round(v * 10) / 1000, changedBy: String(user || 'admin').slice(0, 64), changedAt: now.toISOString() });
  return { ok: true, status: 200, message: `Variety line ${v}%: the weekly tuning uses it from its next run, and new hyperopts start with it`, detail: `${before}% -> ${v}%` };
}
function allRecords(root) {
  let names = [];
  try { names = fs.readdirSync(path.join(root, DIR)).filter((n) => /^[A-Za-z0-9_-]+\.json$/.test(n)); } catch (e) { return []; }
  return names.map((n) => readRecord(root, n.slice(0, -5))).filter((r) => r && r.id && r.game)
    .sort((a, b) => String(b.requestedAt || '').localeCompare(String(a.requestedAt || '')));
}

// The tuner's arguments for a record (remote-hyperopt.sh adds --save false and --result-file); records from before
// 12 Nov 2026 have no tuner and no variety line: the deep-learning tuner, the setting's default
function argsOf(rec) {
  return tunerOf(rec).args({ varietyLine: VARIETY.default, ...rec });
}

// The result file's studies of a record: the deep-learning tuner's one, or every study the tuner wrote
function studiesOf(root, rec) {
  const result = readJson(fileOf(root, rec.id, 'result'));
  const all = result && result.studies && typeof result.studies === 'object' ? result.studies : {};
  const keys = tunerOf(rec).studyKeys ? tunerOf(rec).studyKeys(rec) : Object.keys(all).sort();
  return keys.filter((k) => all[k]).map((k) => ({ key: k, ...all[k] }));
}

// What Apply writes for a study: {params, remove, scores, rows} - the gate's decision (the result file's params,
// remove, modelScores and rows; the deep-learning tuner's single record under the run's row), or this run's best trial
// (challengerParams) with the main row's record marked as applied against the gate
function writesOf(rec, study, what) {
  if (what === 'challenger') {
    const p = study.challengerParams;
    return p && typeof p === 'object' && Object.keys(p).length ? { params: p, remove: [], scores: {}, rows: {} } : null;
  }
  const params = study.params && typeof study.params === 'object' ? study.params : {};
  const remove = Array.isArray(study.remove) ? study.remove.filter((k) => typeof k === 'string') : [];
  const scores = study.modelScores && typeof study.modelScores === 'object' ? study.modelScores : {};
  if (!Object.keys(params).length && !remove.length) return null;
  const rows = study.rows && typeof study.rows === 'object' && Object.keys(study.rows).length ? study.rows
    : study.record ? { [study.row || rec.row]: study.record } : {};
  return { params, remove, scores, rows };
}
const mainRow = (rec, study) => study.row || rec.row;

// Apply's view of a finished study: what it would write, and which served values changed since the run started
function applyView(root, rec, study) {
  if (!study || study.status !== 'done') return null;
  const served = readJson(path.join(root, `bestParams_${rec.game}.json`)) || {};
  const before = readJson(fileOf(root, rec.id, 'served')) || {};
  const same = (x, y) => JSON.stringify(x) === JSON.stringify(y);
  const side = (w) => {
    if (!w) return null;
    const changes = [
      ...Object.keys(w.params).filter((k) => !same(served[k], w.params[k])).map((k) => ({ key: k, served: served[k] ?? null, value: w.params[k] })),
      ...w.remove.filter((k) => k in served).map((k) => ({ key: k, served: served[k], value: null, removed: true })),
      ...Object.keys(w.scores).filter((k) => !same((served.modelScores || {})[k], w.scores[k]))
        .map((k) => ({ key: `modelScores.${k}`, served: (served.modelScores || {})[k] ?? null, value: w.scores[k] ?? null, ...(w.scores[k] === null ? { removed: true } : {}) })),
    ];
    const keys = [...Object.keys(w.params), ...w.remove];
    return { changes, changedSince: keys.filter((k) => !same(served[k], before[k])), rows: Object.keys(w.rows) };
  };
  return { gate: side(writesOf(rec, study, 'gate')), challenger: study.decision === 'replaced' ? null : side(writesOf(rec, study, 'challenger')) };
}

// The record as the pages read it, with the lane's state (`ws`: workers.lane(<lane>).backtestState()) and the result
const LONG = ['summary', 'record', 'rows'];
function viewOf(root, rec, ws, { full = false } = {}) {
  const t = tunerOf(rec);
  const out = { ...rec, tuner: rec.tuner || 'deeplearning', tunerLabel: t.label, worker: rec.worker || t.lane };
  const studies = studiesOf(root, rec).map((st) => {
    const v = { ...st };
    if (!full) LONG.forEach((k) => delete v[k]);
    return v;
  });
  out.studies = studies;
  // the table's one line: the study running now, else the first (the deep-learning tuner has only one)
  out.study = studies.find((st) => st.status === 'running' || st.status === 'gate') || studies[0] || null;
  if (ws) {
    const place = ws.queued.indexOf(rec.id);
    const running = ws.current && ws.current.id === rec.id;
    if (rec.status === 'queued' && place >= 0) out.place = place + 1;
    if (running) {
      out.runId = ws.current.runId;
      if (rec.status === 'queued') out.status = 'running';
      out.where = ws.current.local ? 'this server' : out.worker === 'cpu' ? 'the DL360' : 'the GPU worker';
    }
    if ((rec.status === 'queued' && place < 0 && !running) || (rec.status === 'running' && !running)) {
      out.status = 'ended';
      out.error = rec.status === 'running' ? `the run ended without marking its record (killed, or the server lost it) - see log/hyperopts/${rec.id}.log`
        : `no longer in the ${out.worker === 'cpu' ? 'CPU' : 'GPU'} worker's queue`;
    }
  }
  if (full) {
    const applied = rec.appliedStudies || (rec.applied ? { [studies[0] ? studies[0].key : '']: rec.applied } : {});
    out.apply = Object.fromEntries(studies.filter((st) => !applied[st.key]).map((st) => [st.key, applyView(root, rec, st)]).filter(([, v]) => v));
    out.appliedStudies = applied;
  }
  return out;
}

// laneStates: {gpu, cpu} -> each lane's backtestState() (or null)
function list(root, laneStates) {
  const ls = laneStates && (laneStates.gpu !== undefined || laneStates.cpu !== undefined) ? laneStates : { gpu: laneStates || null, cpu: null };
  const lanes = Object.fromEntries(['gpu', 'cpu'].map((k) => {
    const ws = ls[k];
    return [k, ws ? { configured: Boolean(ws.configured), host: ws.host || null, running: ws.current ? ws.current.id : null, local: Boolean(ws.current && ws.current.local), queued: ws.queued.length }
      : { configured: false, host: null, running: null, local: false, queued: 0 }];
  }));
  return {
    hyperopts: allRecords(root).slice(0, KEPT).map((r) => viewOf(root, r, ls[tunerOf(r).lane] || null)),
    games: GAMES, models: MODELS, fields: FIELDS, budgetMaxSeconds: BUDGET_MAX,
    tuners: Object.fromEntries(Object.entries(TUNERS).map(([k, t]) => [k, {
      label: t.label, script: t.script, lane: t.lane, rowWord: t.rowWord, rows: t.rows, games: t.games,
      rowGames: t.rowGames, fields: { ...t.fields, ...(k === 'deeplearning' ? {} : {}) }, references: t.references, fixedBudget: t.budget ? t.budget({}) : null,
    }])),
    variety: { ...VARIETY, ...settings(root) },
    gpu: lanes.gpu, cpu: lanes.cpu,
  };
}

// The form's values, checked: {ok, params} or {ok:false, status, message}
function parse(body, root = null) {
  const b = body || {};
  const tuner = String(b.tuner || 'deeplearning');
  const t = TUNERS[tuner];
  if (!t) return { ok: false, status: 400, message: `${tuner} is not a tuner the page runs` };
  const game = String(b.game || '');
  const model = String(b.model ?? (Object.keys(t.rows).length === 1 ? Object.keys(t.rows)[0] : ''));
  if (!(model in t.rows)) {
    return { ok: false, status: 400, message: tuner === 'deeplearning' ? `${model || 'that model'} is not a deep-learning model the tuner knows` : `${model || 'that row'} is not a row the ${t.label.toLowerCase()} tuner has` };
  }
  if (!t.games.includes(game)) return { ok: false, status: 400, message: `${game || 'that game'} is not a game the ${t.label.toLowerCase()} tuner runs` };
  if (t.rowGames[model] && !t.rowGames[model].includes(game)) return { ok: false, status: 400, message: `${t.rows[model]} is not tuned on ${game}` };
  const p = { tuner, game, model };
  for (const [k, f] of Object.entries(t.fields)) {
    const raw = b[k];
    const v = raw === undefined || raw === null || raw === '' ? f.default : Number(raw);
    if (!Number.isFinite(v) || (f.integer && !Number.isInteger(v)) || v < f.min || v > f.max) {
      return { ok: false, status: 400, message: `${k} is ${f.integer ? 'a whole number' : 'a number'} from ${f.min} to ${f.max}` };
    }
    p[k] = v;
  }
  if (tuner === 'deeplearning' && p.pruneAfterDays > p.days) return { ok: false, status: 400, message: 'prune after days is at most the days per trial (0 = no pruning)' };
  const vraw = b.varietyLine;
  const vl = vraw === undefined || vraw === null || vraw === '' ? (root ? settings(root).varietyLine : VARIETY.default) : Number(vraw);
  if (!Number.isFinite(vl) || vl < VARIETY.min || vl > VARIETY.max) return { ok: false, status: 400, message: `the variety line is a percentage from ${VARIETY.min} to ${VARIETY.max}` };
  p.varietyLine = Math.round(vl * 10) / 10;
  p.gate = b.gate === undefined ? true : b.gate === true || b.gate === 'true';
  if (tuner === 'deeplearning') p.fresh = b.fresh === true || b.fresh === 'true';
  return { ok: true, params: p };
}

function start(root, body, user, workers, now = new Date()) {
  const parsed = parse(body, root);
  if (!parsed.ok) return parsed;
  const p = parsed.params;
  const t = TUNERS[p.tuner];
  const lane = workers && typeof workers.lane === 'function' ? workers.lane(t.lane) : null;
  // the CPU tuners run on this server when no DL360 is set up (the owner's fallback); the GPU one needs its worker
  const local = t.lane === 'cpu' && !(lane && lane.config());
  if (!lane || (!local && !lane.config())) return { ok: false, status: 503, message: 'no GPU worker is set up (config/workers.json): the deep-learning hyperopts run there' };
  const slug = `${p.game}-${p.tuner === 'deeplearning' ? p.model : `${p.tuner}${p.model ? `-${p.model}` : ''}`}`.replace(/_/g, '-').toLowerCase();
  const id = `${slug}-${now.toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}`;
  if (readRecord(root, id)) return { ok: false, status: 409, message: 'a hyperopt of this row was started this very second - try again' };
  const rec = { id, ...p, row: t.rows[p.model], budget: budgetOf(p), by: String(user || 'admin').slice(0, 64),
    requestedAt: now.toISOString(), status: 'queued', worker: t.lane };
  writeJson(fileOf(root, id), rec);
  writeJson(fileOf(root, id, 'served'), readJson(path.join(root, `bestParams_${p.game}.json`)) || {});
  const queued = lane.queue(id, rec.by, p.days, { hyperopt: { budget: rec.budget, args: [t.script, ...argsOf(rec)], ...(local ? { local: true } : {}) } });
  if (!queued.ok) {
    ['', 'served'].forEach((k) => { try { fs.unlinkSync(fileOf(root, id, k)); } catch (e) { /* gone */ } });
    return queued;
  }
  const what = p.tuner === 'deeplearning' ? MODELS[p.model].replace(/ Model$/, '') : `${t.label}${p.model ? ` (${t.rows[p.model]})` : ''}`;
  return { ok: true, status: 200, id,
    message: `${what} on ${p.game}, ${p.trials} trials of ${p.days} draws${p.fresh ? ', fresh' : ''}${local ? ' (on this server: no DL360 is set up)' : ''}: ${queued.message}`,
    detail: `${p.tuner} ${p.game} ${p.model || 'all'} t${p.trials} d${p.days}${p.trainSeconds ? ` s${p.trainSeconds}` : ''} gate ${p.gate ? p.gateSeMargin : 'off'} variety ${p.varietyLine}%${p.fresh ? ' fresh' : ''}` };
}

function cancel(root, id, user, workers) {
  const rec = readRecord(root, id);
  if (!rec) return { ok: false, status: 404, message: 'no such hyperopt' };
  const kind = tunerOf(rec).lane;
  const lane = workers && typeof workers.lane === 'function' ? workers.lane(kind) : null;
  if (!lane) return { ok: false, status: 503, message: `no ${kind === 'cpu' ? 'CPU' : 'GPU'} worker is available here` };
  const ws = lane.backtestState();
  if (ws.current && ws.current.id === id) {
    const r = lane.stopRun(ws.current.runId, user);
    return r.ok ? { ...r, message: 'Hyperopt stopping: its finished trials stay in the study' } : r;
  }
  if (rec.status !== 'queued') return { ok: false, status: 409, message: `that hyperopt is ${rec.status}, not queued` };
  const r = lane.cancel(id, user);
  writeJson(fileOf(root, id), { ...rec, status: 'cancelled', finishedAt: new Date().toISOString(), error: `cancelled by ${String(user || 'admin').slice(0, 64)}` });
  return r.ok ? { ...r, message: 'Hyperopt cancelled' } : { ok: true, status: 200, message: 'Hyperopt cancelled' };
}

// a job holds process.lock (a live PID in it): a tuner or the predictor may be writing bestParams_<game>.json
function lockHolder(root) {
  let pid;
  try { pid = Number(String(fs.readFileSync(path.join(root, 'process.lock'), 'utf-8')).trim()); } catch (e) { return null; }
  if (!Number.isInteger(pid) || pid <= 0) return null;
  try { process.kill(pid, 0); return pid; } catch (e) { return e.code === 'EPERM' ? pid : null; }
}

// Apply: body {what: 'gate' | 'challenger', study: <key> (default: the record's first study), anyway: bool}.
function apply(root, id, body, user, now = new Date(), deps = {}) {
  const rec = readRecord(root, id);
  if (!rec) return { ok: false, status: 404, message: 'no such hyperopt' };
  const studies = studiesOf(root, rec);
  const key = (body || {}).study ? String(body.study) : studies[0] ? studies[0].key : '';
  const applied = rec.appliedStudies || (rec.applied ? { [key]: rec.applied } : {});
  if (applied[key] || (rec.applied && !rec.appliedStudies)) {
    const a = applied[key] || rec.applied;
    return { ok: false, status: 409, message: `already applied by ${a.by} on ${a.at}` };
  }
  const study = studies.find((st) => st.key === key) || null;
  if (!study || study.status !== 'done') return { ok: false, status: 409, message: 'the study has not finished: there is nothing to apply yet' };
  const what = (body || {}).what === 'challenger' ? 'challenger' : 'gate';
  const w = writesOf(rec, study, what);
  if (!w) {
    return { ok: false, status: 409, message: what === 'gate' ? `the gate kept what is served (${study.reason || 'no reason given'}): nothing to write${study.challengerParams ? ' - "Apply this run\'s best" writes the best trial against it' : ''}`
      : 'this run has no scored trial to apply' };
  }
  if (what === 'challenger' && study.decision === 'replaced') return { ok: false, status: 409, message: 'the gate already chose this run\'s best: apply the gate\'s result' };
  const holder = (deps.lockHolder || lockHolder)(root);
  if (holder) return { ok: false, status: 409, message: `a job is running (process.lock, PID ${holder}) and may be writing bestParams_${rec.game}.json - apply when it has finished` };
  const file = path.join(root, `bestParams_${rec.game}.json`);
  const served = readJson(file);
  if (!served || typeof served !== 'object') return { ok: false, status: 500, message: `bestParams_${rec.game}.json does not read` };
  const before = readJson(fileOf(root, id, 'served')) || {};
  const changed = [...Object.keys(w.params), ...w.remove].filter((k) => JSON.stringify(served[k]) !== JSON.stringify(before[k]));
  if (changed.length && !(body || {}).anyway) {
    return { ok: false, status: 409, changed, message: `${changed.length} of these values changed since the run started (${changed.slice(0, 4).join(', ')}${changed.length > 4 ? ', ...' : ''}): the gate compared against the old ones - apply anyway?` };
  }
  const by = String(user || 'admin').slice(0, 64);
  const at = now.toISOString();
  const stamp = { by, at, what, hyperopt: id, ...(changed.length ? { changedSince: changed } : {}) };
  let rows = w.rows;
  if (what === 'challenger') {
    const record = { ...(study.record || (study.rows || {})[mainRow(rec, study)] || {}) };
    record.reason = `applied by hand by ${by} against the gate's verdict (${study.decision}: ${study.reason || record.reason || ''})`;
    record.decision = 'replaced';
    rows = { [mainRow(rec, study)]: record };
  }
  const next = { ...served, ...w.params };
  w.remove.forEach((k) => { delete next[k]; });
  if (Object.keys(w.scores).length) {
    next.modelScores = { ...(served.modelScores || {}) };
    Object.entries(w.scores).forEach(([k, v]) => { if (v === null) delete next.modelScores[k]; else next.modelScores[k] = v; });
  }
  next.tuningGate = { ...(served.tuningGate || {}), ...Object.fromEntries(Object.entries(rows).map(([k, r]) => [k, { ...r, applied: stamp }])) };
  writeJson(file, next);
  const n = Object.keys(w.params).length + w.remove.length + Object.keys(w.scores).length;
  const mark = { by, at, what, keys: n };
  const recNext = { ...rec, appliedStudies: { ...applied, [key]: mark } };
  if (tunerOf(rec) === TUNERS.deeplearning || studies.length === 1) recNext.applied = mark;
  writeJson(fileOf(root, id), recNext);
  return { ok: true, status: 200,
    message: `${n} values of ${Object.keys(rows).join(', ') || rec.row} written to bestParams_${rec.game}.json (${what === 'gate' ? `the gate's ${study.decision}` : "this run's best trial"}) - the next run serves them and its commit includes the file`,
    detail: `${id} ${key} ${what}${changed.length ? ` anyway (${changed.length} changed)` : ''}` };
}

// The run's log, its last `bytes` (64 KB by default)
function logTail(root, id, bytes = 64 * 1024) {
  if (!ID.test(String(id || ''))) return null;
  const file = path.join(root, LOG_DIR, `${id}.log`);
  let fd;
  try {
    fd = fs.openSync(file, 'r');
    const size = fs.fstatSync(fd).size;
    const n = Math.min(size, bytes);
    const buf = Buffer.alloc(n);
    fs.readSync(fd, buf, 0, n, size - n);
    // progress bars rewrite one line with carriage returns (Keras also with backspaces and colour codes): drop the
    // codes and the backspaces, keep the last state of each line
    const text = buf.toString('utf-8').replace(/\x1b\[[0-9;?]*[A-Za-z]/g, '').replace(/\x08/g, '').split('\n').map((l) => l.split('\r').filter(Boolean).pop() || '').join('\n');
    return { size, text, truncated: size > n };
  } catch (e) { return { size: 0, text: '', truncated: false }; } finally { if (fd !== undefined) fs.closeSync(fd); }
}

function install(app, { root, workers = null, guard, jsonBody, csrf, safe, json, reply, audit }) {
  const laneState = (k) => (workers && typeof workers.lane === 'function' && workers.lane(k) ? workers.lane(k).backtestState() : null);
  const lanes = () => ({ gpu: laneState('gpu'), cpu: laneState('cpu') });
  app.get('/api/admin/hyperopts', guard, safe((req, res) => json(res, list(root, lanes()))));
  app.post('/api/admin/hyperopts/settings', guard, jsonBody, csrf, safe((req, res) => {
    const r = saveSettings(root, req.body, req.user.name);
    if (r.ok) audit.record('hyperopt.settings', req, req.user.name, r.detail);
    reply(res, r);
  }));
  app.get('/api/admin/hyperopts/:id', guard, safe((req, res) => {
    const rec = readRecord(root, req.params.id);
    if (!rec) return json(res, { error: 'no such hyperopt' }, 404);
    return json(res, viewOf(root, rec, laneState(tunerOf(rec).lane), { full: true }));
  }));
  app.get('/api/admin/hyperopts/:id/log', guard, safe((req, res) => {
    const rec = readRecord(root, req.params.id);
    if (!rec) return json(res, { error: 'no such hyperopt' }, 404);
    return json(res, logTail(root, rec.id));
  }));
  app.post('/api/admin/hyperopts/start', guard, jsonBody, csrf, safe((req, res) => {
    const r = start(root, req.body, req.user.name, workers);
    if (r.ok) audit.record('hyperopt.started', req, req.user.name, r.detail);
    reply(res, r, r.ok ? { id: r.id } : {});
  }));
  app.post('/api/admin/hyperopts/:id/cancel', guard, jsonBody, csrf, safe((req, res) => {
    const r = cancel(root, String(req.params.id || ''), req.user.name, workers);
    if (r.ok) audit.record('hyperopt.cancelled', req, req.user.name, String(req.params.id));
    reply(res, r);
  }));
  app.post('/api/admin/hyperopts/:id/apply', guard, jsonBody, csrf, safe((req, res) => {
    const r = apply(root, String(req.params.id || ''), req.body, req.user.name);
    if (r.ok) audit.record('hyperopt.applied', req, req.user.name, r.detail);
    reply(res, r);
  }));
}

module.exports = { DIR, GAMES, MARKETS, MODELS, FIELDS, TUNERS, VARIETY, STATUSES, budgetOf, argsOf, parse, studyKey, studiesOf, readRecord, list, viewOf, start, cancel, apply, settings, saveSettings, logTail, lockHolder, install };
