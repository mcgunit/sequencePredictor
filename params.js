'use strict';
// Admin › Parameters (owner's roadmap step 2, 8 Nov 2026): every setting in a game's bestParams_<game>.json, one card
// per model, the settings several models read in a card of their own. Read-only: the manual overrides (step 3) come
// later and will write next to this, never into the tuners' file.
//
// Which model a key belongs to is the code that reads it (Predictor.py, the tuners), written down once here as
// FAMILIES. A key no family claims is shown under "Not assigned" rather than hidden, and a key the predictor stores
// but never reads (the four LSTM keys of an old design) is shown in a card of its own. Since 8 Nov 2026 every
// deep-learning row reads its own "<prefix>_yearsOfHistory" (Predictor.rowYearsOfHistory); the LSTM rows read the bare
// yearsOfHistory, which is also every other row's fallback. Values are the file's; a key the file does not have is
// "not set", and the model then runs on its code default, which this page does not repeat (it lives in the code, and
// a copy here would go stale).
const fs = require('fs');
const path = require('path');

const LOTTERY_GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'pick3', 'keno'];
const MARKET_GAMES = ['crypto', 'shares', 'cryptoweek', 'sharesweek'];
const GAMES = [...LOTTERY_GAMES, ...MARKET_GAMES];
// the games whose rows get a "· Sampled" twin: the number games ranked by probability (Predictor.probabilityOrder);
// Pick3 and Joker+ are positional (digits in drawn order), the markets are settled into paper books
const SAMPLED_GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'keno'];

// The bare keys of the two LSTM rows (Predictor.configureLstmModel). labelSmoothing is read by LSTM Smoothed only.
const LSTM_KEYS = ['batchSize', 'epochs', 'num_lstm_layers', 'lstm_units', 'num_bidirectional_layers', 'bidirectional_lstm_units',
  'optimizer_type', 'learningRate', 'dropout', 'l2Regularization', 'useFinalLSTMLayer', 'earlyStopPatience',
  'reduceLearningRatePatience', 'reduceLearningRateFactor', 'windowSize', 'labelSmoothing'];

// Settings read by more than one model, or by the run itself.
const SHARED = [
  { key: 'yearsOfHistory', note: 'years of draws the LSTM rows train on, and every other deep-learning row without its own <prefix>_yearsOfHistory' },
  { key: 'useMemoriseRows', note: 'the memorise rows on or off for this game (not set: on)', fallback: true },
  { key: 'memoriseWindowDraws', note: 'the newest draws a statistical memorise row reads (not set: 30)', fallback: 30 },
  { key: 'sampleTemperatures', note: 'per row, the temperature its sampled twin draws with (a row not listed: 1)', fallback: {} },
];
// The tuners' own bookkeeping, not settings: shown as the cards' "last tuned" and score lines.
const RECORDS = ['tuningGate', 'modelScores'];
// Stored by an older LSTM design; nothing in the predictor reads them (two are still read by the deep-learning tuner).
const UNREAD = {
  lstmMarkovAlpha: 'not read anywhere',
  useLstmMarkovPrediction: 'not read anywhere',
  useTopPrediction: 'read by HyperoptDeepLearning.py only, not by the predictor',
  outputActivation: 'read by HyperoptDeepLearning.py only, not by the predictor',
};

// One family per card. prefixes claim every key that starts with them (the longest matching prefix wins, so
// "markovMc" goes to MarkovMonteCarlo, not Markov); keys claims exact names; switch is the use-flag that turns the row
// on or off; rows are the row names the card covers (gate records and scores are looked up under them); games limits
// the family to lottery or market games; memorise says whether the row has a memorise twin ('dl', 'stat', 'boost',
// 'catboost' when only on with memoriseCatBoost).
const DL_PREFIXES = [
  ['tcn', 'TCN Base Model', 'useTcn'],
  ['unifiedLstmTcn', 'UnifiedLstmTcn Model', 'useUnifiedLstmTcn'],
  ['unifiedLstmGruTcn', 'UnifiedLstmGruTcn Model', 'useUnifiedLstmGruTcn'],
  ['transformer', 'Transformer Model', 'useTransformer'],
  ['gnn', 'GNN Model', 'useGnn'],
  ['autoencoder', 'Autoencoder Model', 'useAutoencoder'],
];
const boosting = (prefix, base, sw, mlSw) => [
  { group: 'Boosting', rows: [`${base} Model`], prefixes: [prefix], switches: [sw], memorise: prefix.startsWith('catBoost') ? 'catboost' : 'boost',
    ...(prefix.startsWith('catBoost') ? { keys: ['memoriseCatBoost'] } : {}) },
  { group: 'Boosting', rows: [`${base}MultiLabel Model`], prefixes: [`${prefix}Ml`], switches: [mlSw], memorise: prefix.startsWith('catBoost') ? 'catboost' : 'boost' },
];
const FAMILIES = [
  // statistical rows (Predictor.statisticalMethod; tuned by HyperoptStatistics.py)
  { group: 'Statistical', rows: ['Markov Model'], prefixes: ['markov'], switches: ['useMarkov'], memorise: 'stat' },
  { group: 'Statistical', rows: ['MarkovMonteCarlo Model'], prefixes: ['markovMc'], switches: ['useMarkovMonteCarlo'], memorise: 'stat' },
  { group: 'Statistical', rows: ['MarkovBayesian Model'], prefixes: ['markovBayesian'], switches: ['useMarkovBayesian'], memorise: 'stat' },
  { group: 'Statistical', rows: ['MarkovBayesianEnhanched Model'], prefixes: ['markovBayesianEnhanced'], switches: ['usevMarkovBayesianEnhanced'], memorise: 'stat' },
  { group: 'Statistical', rows: ['PoissonMonteCarlo Model'], prefixes: ['poissonMonteCarlo'], switches: ['usePoissonMonteCarlo'], memorise: 'stat' },
  { group: 'Statistical', rows: ['PoissonMarkov Model'], prefixes: ['poissonMarkov'], switches: ['usePoissonMarkov'], memorise: 'stat' },
  { group: 'Statistical', rows: ['LaplaceMonteCarlo Model'], prefixes: ['laplaceMonteCarlo'], switches: ['useLaplaceMonteCarlo'], memorise: 'stat' },
  { group: 'Statistical', rows: ['HybridStatisticalModel'], prefixes: ['hybridStatisticalModel'], switches: ['useHybridStatisticalModel'] },
  { group: 'Statistical', rows: ['ClassicalSVM Model'], prefixes: ['classicalSvm'] },
  // boosting rows (tuned by HyperoptBoost.py); the XGBoost row's switch is the old "useBoost"
  ...boosting('xgBoost', 'XGBoost', 'useBoost', 'useXgBoostMultiLabel'),
  ...boosting('lightGbm', 'LightGBM', 'useLightGbm', 'useLightGbmMultiLabel'),
  ...boosting('catBoost', 'CatBoost', 'useCatBoost', 'useCatBoostMultiLabel'),
  // deep learning (tuned by HyperoptDeepLearning.py); the two LSTM rows share their bare keys
  { group: 'Deep learning', rows: ['LSTM Base Model', 'LSTM Smoothed Model'], keys: LSTM_KEYS, switches: ['useLstm', 'useLstmSmoothed'], memorise: 'dl',
    notes: { labelSmoothing: 'read by LSTM Smoothed Model only' } },
  ...DL_PREFIXES.map(([prefix, row, sw]) => ({ group: 'Deep learning', rows: [row], prefixes: [`${prefix}_`], switches: [sw], memorise: 'dl',
    notes: { [`${prefix}_yearsOfHistory`]: 'years of draws this row trains on (not set: the shared yearsOfHistory)' } })),
  // quantum (tuned by HyperoptQuantum.py; the kernel's settings are read by TrainMetaLearner.py's quantum meta-learner)
  { group: 'Quantum', rows: ['QuantumVQC Model'], prefixes: ['quantumVqc'] },
  { group: 'Quantum', rows: ['QuantumMetaLearner Model'], prefixes: ['quantumMetaLearner', 'quantumKernel_'],
    notes: { 'quantumKernel_*': 'the quantum kernel the meta-learner is trained with (TrainMetaLearner.py)' } },
  // the rows built on other rows
  { group: 'Ensembles and meta-learners', rows: ['MetaLearner Model'], prefixes: ['metaLearner'] },
  { group: 'Ensembles and meta-learners', rows: ['MetaLearnerV2 Model'], prefixes: ['metaLearnerV2'] },
  { group: 'Ensembles and meta-learners', rows: ['WeightedEnsemble Model'], prefixes: ['weightedEnsemble'] },
  { group: 'Ensembles and meta-learners', rows: ['SubsetEnsemble Model'], prefixes: ['subsetEnsemble'],
    notes: { subsetEnsembleObjective: 'the tuner\'s score of the chosen subset', subsetEnsembleMean: 'the tuner\'s mean hits of the chosen subset',
      subsetEnsembleTunedOn: 'the window and search the subset was chosen on' } },
  { group: 'Ensembles and meta-learners', rows: ['RL Ticket Model'], prefixes: ['rlTicket'], switches: ['useRlTicket'], games: 'lottery' },
  { group: 'Keno sub-selections', rows: ['KenoSubsetTuning'], keys: ['use_5', 'use_6', 'use_7', 'use_8', 'use_9', 'use_10'], title: 'Keno sub-selections',
    notes: { use_5: 'play the 5-number selection', use_10: 'play the 10-number selection' }, games: ['keno'] },
  // market rows
  { group: 'Market rows', rows: ['GARCH Model'], prefixes: ['garch'], switches: ['useGarch'], games: 'market' },
  { group: 'Market rows', rows: ['Regime HMM Model', 'Regime HMM ZeroMean Model', 'Regime HMM Single Model'], prefixes: ['regimeHmm'], switches: ['useRegimeHmm'], games: 'market',
    title: 'Regime HMM Model (and its two ablations)' },
  { group: 'Market rows', rows: ['RL Position Model'], prefixes: ['rlPosition'], switches: ['useRlPosition'], games: 'market' },
  { group: 'Market rows', rows: ['Chronos Model'], switches: ['useChronos'], games: 'market' },
  { group: 'Market rows', rows: ['TimesFM Model'], switches: ['useTimesFm'], games: 'market' },
];
const GROUP_ORDER = ['Statistical', 'Boosting', 'Deep learning', 'Quantum', 'Ensembles and meta-learners', 'Keno sub-selections', 'Market rows'];

const isMarket = (game) => MARKET_GAMES.includes(game);
const appliesTo = (f, game) => !f.games || (f.games === 'market' ? isMarket(game) : f.games === 'lottery' ? !isMarket(game) : f.games.includes(game));

// The family that claims a key: an exact key or switch first, then the longest matching prefix.
function familyOf(key) {
  const exact = FAMILIES.find((f) => (f.keys || []).includes(key) || (f.switches || []).includes(key));
  if (exact) return exact;
  let best = null;
  let length = 0;
  FAMILIES.forEach((f) => (f.prefixes || []).forEach((p) => {
    if (key.startsWith(p) && p.length > length) { best = f; length = p.length; }
  }));
  return best;
}

const noteOf = (f, key) => {
  const notes = f.notes || {};
  if (notes[key]) return notes[key];
  const wild = Object.keys(notes).find((k) => k.endsWith('*') && key.startsWith(k.slice(0, -1)));
  return wild ? notes[wild] : null;
};

// A gate record, trimmed to what a card says: when, which tuner, what it decided and why.
function gateOf(rec) {
  if (!rec || typeof rec !== 'object') return null;
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  return {
    date: typeof rec.date === 'string' ? rec.date : null, tuner: typeof rec.tuner === 'string' ? rec.tuner : null,
    decision: typeof rec.decision === 'string' ? rec.decision : null, reason: typeof rec.reason === 'string' ? rec.reason : null,
    windowDays: num(rec.window_days), runTrials: num(rec.run_trials), studyTrials: num(rec.study_trials), trial: num(rec.trial),
  };
}

// What the variant rows of this card's model are for this game (src/RowVariants.py, Predictor's memorise rows).
function variantsOf(f, game, params) {
  const sampledGame = SAMPLED_GAMES.includes(game);
  const temps = params.sampleTemperatures && typeof params.sampleTemperatures === 'object' ? params.sampleTemperatures : {};
  const memoriseOn = !isMarket(game) && params.useMemoriseRows !== false;
  const memorise = (() => {
    if (!f.memorise || isMarket(game)) return { state: 'none', why: isMarket(game) ? 'no variant rows on the market games' : 'no memorise variant for this model' };
    if (!memoriseOn) return { state: 'off', why: 'useMemoriseRows is off for this game' };
    if (f.memorise === 'catboost' && !params.memoriseCatBoost) return { state: 'off', why: 'off unless memoriseCatBoost is set (CatBoost is slow)' };
    return { state: 'on', why: { dl: 'fresh shallow network, 120 s, no dropout, L2 or smoothing', stat: `the newest ${Number(params.memoriseWindowDraws) || 30} draws, no smoothing`,
      boost: 'deep unregularised trees', catboost: 'deep unregularised trees' }[f.memorise] };
  })();
  // the LSTM card covers two rows; LSTM Smoothed has no memorise twin (it would be LSTM Base's)
  const sampled = sampledGame
    ? f.rows.filter((r) => r !== 'KenoSubsetTuning').map((row) => ({ row, temperature: Object.prototype.hasOwnProperty.call(temps, row) ? temps[row] : 1, set: Object.prototype.hasOwnProperty.call(temps, row) }))
    : [];
  return { sampled, sampledWhy: sampledGame ? null : (isMarket(game) ? 'no variant rows on the market games' : 'no sampled rows on a positional game (digits in drawn order)'), memorise };
}

// --- manual overrides (owner's roadmap step 3, 8 Nov 2026) ---------------------------------------------------------
// config/paramOverrides.json, read by the predictor (src/ParamOverrides.py) and written here only:
//   {"version": 1, "games": {"<game>": {"overrides": {"<key>": {value, rows, tuned, by, at, note}}, "fresh": {"<row>": at}}}}
// An override wins over bestParams_<game>.json until it is released; the tuners keep writing their file underneath.
// A key "sampleTemperatures/<row>" sets that row's sample temperature. "fresh" asks a deep-learning row to train once
// from new weights; the predictor records each request it honoured in data/models/fresh-<game>.json.
const OVERRIDES_FILE = path.join('config', 'paramOverrides.json');
const TEMPERATURE = 'sampleTemperatures/';
const NOTE_MAX = 200;
const DL_ROWS = ['LSTM Base Model', 'LSTM Smoothed Model', ...DL_PREFIXES.map(([, row]) => row)];
const DL_PREFIX_OF = Object.fromEntries(DL_PREFIXES.map(([prefix, row]) => [row, prefix]));
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf-8')); } catch (e) {
    if (e.code === 'ENOENT') return fallback;
    throw e;
  }
}
function readOverrides(root) {
  const data = readJson(path.join(root, OVERRIDES_FILE), {});
  return data && typeof data === 'object' && data.games && typeof data.games === 'object' ? data : { version: 1, games: {} };
}
function gameOverrides(root, game) {
  const g = readOverrides(root).games[game] || {};
  return { overrides: g.overrides && typeof g.overrides === 'object' ? g.overrides : {}, fresh: g.fresh && typeof g.fresh === 'object' ? g.fresh : {} };
}
function writeOverrides(root, data) {
  const file = path.join(root, OVERRIDES_FILE);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, `${JSON.stringify(data, null, 1)}\n`);
  fs.renameSync(tmp, file);
}
// the fresh-retrain requests the predictor has honoured (data/models/fresh-<game>.json, written by Predictor.py)
function freshDone(root, game) {
  try { return readJson(path.join(root, 'data', 'models', `fresh-${game}.json`), {}) || {}; } catch (e) { return {}; }
}

// The file with every override in place: what the predictor serves (the same merge as ParamOverrides.apply).
function served(params, overrides) {
  const out = { ...(params || {}) };
  Object.keys(overrides || {}).sort().forEach((key) => {
    const rec = overrides[key];
    if (!rec || typeof rec !== 'object' || !has(rec, 'value')) return;
    if (key.startsWith(TEMPERATURE)) {
      const table = out.sampleTemperatures && typeof out.sampleTemperatures === 'object' ? out.sampleTemperatures : {};
      out.sampleTemperatures = { ...table, [key.slice(TEMPERATURE.length)]: rec.value };
    } else out[key] = rec.value;
  });
  return out;
}

// The settings shared by several rows that an override may set, with the rows each one moves.
const SHARED_TARGETS = {
  yearsOfHistory: { type: 'int', min: 1, max: 40, rows: (p) => DL_ROWS.filter((r) => !DL_PREFIX_OF[r] || !has(p, `${DL_PREFIX_OF[r]}_yearsOfHistory`)) },
  memoriseWindowDraws: { type: 'int', min: 2, max: 10000, rows: () => [] },
  useMemoriseRows: { type: 'bool', rows: () => [] },
};
const typeOf = (v) => (typeof v === 'boolean' ? 'bool' : typeof v === 'number' ? (Number.isInteger(v) ? 'int' : 'float') : typeof v === 'string' ? 'string' : null);

// What overriding `key` in this game means: { rows, type, tuned, dl (rows that retrain from new weights) } or
// { error }. Only a setting the page shows can be set: a key in the file, a model's switch, a shared setting, or a row's
// sample temperature on a game with sampled rows. Lists and objects are not editable here.
function targetOf(game, params, key) {
  const p = params || {};
  if (typeof key !== 'string' || !key || key.length > 120) return { error: 'name a setting' };
  if (key.startsWith(TEMPERATURE)) {
    const row = key.slice(TEMPERATURE.length);
    if (!SAMPLED_GAMES.includes(game)) return { error: `${game} has no sampled rows` };
    if (!FAMILIES.some((f) => appliesTo(f, game) && f.rows.includes(row)) || row === 'KenoSubsetTuning') return { error: `no row called ${row}` };
    const temps = p.sampleTemperatures && typeof p.sampleTemperatures === 'object' ? p.sampleTemperatures : {};
    return { rows: [row], type: 'float', min: 0, minOpen: true, max: 100, tuned: has(temps, row) ? temps[row] : null, dl: [] };
  }
  if (SHARED_TARGETS[key]) {
    const t = SHARED_TARGETS[key];
    const rows = t.rows(p);
    return { rows, type: t.type, min: t.min, max: t.max, tuned: has(p, key) ? p[key] : null, dl: key === 'yearsOfHistory' ? rows : [] };
  }
  if (RECORDS.includes(key) || UNREAD[key] || SHARED.some((x) => x.key === key)) return { error: `${key} cannot be overridden here` };
  const f = familyOf(key);
  if (!f || !appliesTo(f, game)) return { error: `${key} is not a setting of a model ${game} runs` };
  const isSwitch = (f.switches || []).includes(key);
  if (!isSwitch && !has(p, key)) return { error: `${key} is not in bestParams_${game}.json` };
  const type = isSwitch ? 'bool' : typeOf(p[key]);
  if (!type) return { error: `${key} holds a list or an object, which cannot be edited here` };
  const rows = key === 'labelSmoothing' ? ['LSTM Smoothed Model'] : f.rows;
  return { rows, type, min: type === 'int' || type === 'float' ? 0 : undefined, tuned: has(p, key) ? p[key] : null, dl: rows.filter((r) => DL_ROWS.includes(r)) };
}

// The value as the override stores it, or { error }: same kind as the tuned value; numbers finite and not negative.
function checkValue(t, value) {
  if (t.type === 'bool') return typeof value === 'boolean' ? { value } : { error: 'the value is true or false' };
  if (t.type === 'string') {
    return typeof value === 'string' && value.trim() && value.length <= 64 ? { value: value.trim() } : { error: 'the value is a word of at most 64 characters' };
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) return { error: 'the value is a number' };
  if (t.type === 'int' && !Number.isInteger(value)) return { error: 'the value is a whole number' };
  if (t.min !== undefined && (t.minOpen ? value <= t.min : value < t.min)) return { error: `the value is ${t.minOpen ? 'above' : 'at least'} ${t.min}` };
  if (t.max !== undefined && value > t.max) return { error: `the value is at most ${t.max}` };
  return { value };
}

// When the change is used: the game's next predictor run (jobs.js: crypto 00:05 UTC, the lottery games 00:30 UTC,
// the stocks 18:05 Brussels on weekdays).
const nextRun = (game) => (game.startsWith('crypto') ? 'the next crypto run (00:05 UTC)' : game.startsWith('shares') ? 'the next stocks run (18:05, Monday to Friday)' : 'the next lottery run (00:30 UTC)');
const retrainText = (game, rows) => (rows.length
  ? ` ${rows.map((r) => r.replace(/ Model$/, '')).join(', ')} ${rows.length === 1 ? 'trains' : 'train'} from new weights on ${nextRun(game)}, so that run takes longer than usual.`
  : '');
const shown = (v) => (typeof v === 'string' ? v : JSON.stringify(v));

// Sets one override; { ok, message, detail } or { ok: false, status, message }.
function setOverride(root, game, key, value, note, user, now = new Date()) {
  if (!GAMES.includes(game)) return { ok: false, status: 404, message: 'unknown game' };
  const { params } = readParams(root, game);
  const t = targetOf(game, params, key);
  if (t.error) return { ok: false, status: 400, message: t.error };
  const v = checkValue(t, value);
  if (v.error) return { ok: false, status: 400, message: `${key}: ${v.error}` };
  if (note !== undefined && note !== null && typeof note !== 'string') return { ok: false, status: 400, message: 'the note is text' };
  const at = now.toISOString();
  const data = readOverrides(root);
  const entry = data.games[game] || (data.games[game] = {});
  entry.overrides = entry.overrides || {};
  entry.overrides[key] = { value: v.value, rows: t.rows, tuned: t.tuned, by: String(user || 'admin').slice(0, 64), at, ...(note && note.trim() ? { note: note.trim().slice(0, NOTE_MAX) } : {}) };
  if (t.dl.length) {
    entry.fresh = entry.fresh || {};
    t.dl.forEach((row) => { entry.fresh[row] = at; });
  }
  data.version = 1;
  writeOverrides(root, data);
  return { ok: true, detail: `${game} ${key} = ${shown(v.value)}`,
    message: `${key} is set to ${shown(v.value)} for ${game}; it wins over the tuners until it is released. Used from ${nextRun(game)}.${retrainText(game, t.dl)}` };
}

// Releases one override: the tuned value is served again from the next run.
function releaseOverride(root, game, key, user, now = new Date()) {
  if (!GAMES.includes(game)) return { ok: false, status: 404, message: 'unknown game' };
  const data = readOverrides(root);
  const entry = data.games[game];
  if (!entry || !entry.overrides || !has(entry.overrides, key)) return { ok: false, status: 409, message: `${key} has no manual override for ${game} (released already?)` };
  const rec = entry.overrides[key];
  delete entry.overrides[key];
  // a deep-learning row trains from new weights again: its value goes back to the tuned one
  const { params } = readParams(root, game);
  const t = targetOf(game, params, key);
  const dl = t.error ? (rec.rows || []).filter((r) => DL_ROWS.includes(r)) : t.dl;
  if (dl.length) {
    entry.fresh = entry.fresh || {};
    dl.forEach((row) => { entry.fresh[row] = now.toISOString(); });
  }
  writeOverrides(root, data);
  return { ok: true, detail: `${game} ${key} released by ${String(user || 'admin').slice(0, 64)}`,
    message: `${key} is released for ${game}: the tuned value${t.tuned === null || t.tuned === undefined ? ' (or the code default)' : ` ${shown(t.tuned)}`} is used again from ${nextRun(game)}.${retrainText(game, dl)}` };
}

// One game's view. params: the parsed file (null when there is none); stat: its fs.Stats or null; manual: the game's
// overrides and fresh requests ({ overrides, fresh, done }).
function parametersView(game, params, stat = null, now = new Date(), manual = {}) {
  const p = params && typeof params === 'object' ? params : {};
  const overrides = manual.overrides || {};
  const fresh = manual.fresh || {};
  const done = manual.done || {};
  const live = served(p, overrides);
  const ov = (key) => {
    const r = overrides[key];
    return r && typeof r === 'object' && has(r, 'value') ? { value: r.value, by: r.by || null, at: r.at || null, note: r.note || null } : null;
  };
  const gates = p.tuningGate && typeof p.tuningGate === 'object' ? p.tuningGate : {};
  const scores = p.modelScores && typeof p.modelScores === 'object' ? p.modelScores : {};
  const families = FAMILIES.filter((f) => appliesTo(f, game));
  const byFamily = new Map(families.map((f) => [f, []]));
  const unassigned = [];
  const unread = [];
  Object.keys(p).forEach((key) => {
    if (RECORDS.includes(key) || SHARED.some((s) => s.key === key)) return;
    if (UNREAD[key]) { unread.push({ key, value: p[key], note: UNREAD[key] }); return; }
    const f = familyOf(key);
    if (f && byFamily.has(f)) byFamily.get(f).push(key);
    else unassigned.push({ key, value: p[key], note: f ? `belongs to ${f.rows[0]}, which this game does not run` : null });
  });
  const editable = (key) => !targetOf(game, p, key).error;
  const cards = families.map((f) => {
    const keys = byFamily.get(f);
    const switches = (f.switches || []).map((key) => ({ key, value: has(p, key) ? p[key] : null, set: has(p, key), override: ov(key), editable: editable(key) }));
    const switchKeys = new Set(switches.map((s) => s.key));
    const list = keys.filter((k) => !switchKeys.has(k)).map((key) => ({ key, value: p[key], read: true, note: noteOf(f, key), override: ov(key), editable: editable(key), type: typeOf(p[key]) }));
    const gate = f.rows.map((r) => ({ row: r, ...gateOf(gates[r]) })).filter((g) => g.date || g.decision);
    const score = f.rows.map((r) => ({ row: r, score: typeof scores[r] === 'number' ? scores[r] : null })).filter((s) => s.score !== null);
    const variants = variantsOf(f, game, live);
    const temps = p.sampleTemperatures && typeof p.sampleTemperatures === 'object' ? p.sampleTemperatures : {};
    variants.sampled = variants.sampled.map((s) => ({ ...s, temperature: has(temps, s.row) ? temps[s.row] : 1, set: has(temps, s.row), key: `${TEMPERATURE}${s.row}`, override: ov(`${TEMPERATURE}${s.row}`) }));
    const retrain = f.rows.filter((r) => DL_ROWS.includes(r) && fresh[r]).map((r) => ({ row: r, asked: fresh[r], pending: done[r] !== fresh[r] }));
    const manualCount = list.filter((l) => l.override).length + switches.filter((x) => x.override).length + variants.sampled.filter((x) => x.override).length;
    return {
      id: f.rows[0].toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''),
      title: f.title || f.rows.join(' and '), group: f.group, rows: f.rows, switches, params: list,
      stored: list.length + switches.filter((x) => x.set).length, gates: gate, scores: score, variants, retrain, manual: manualCount,
      deepLearning: f.rows.some((r) => DL_ROWS.includes(r)),
    };
  });
  const groups = GROUP_ORDER.map((label) => ({ label, cards: cards.filter((c) => c.group === label && (c.stored || c.gates.length || c.manual)) }))
    .filter((g) => g.cards.length);
  // the models this game can run that have nothing in the file: they run entirely on their code defaults
  const empty = cards.filter((c) => !c.stored && !c.gates.length && !c.manual).map((c) => c.title);
  const shared = SHARED.map((s) => ({ key: s.key, value: has(p, s.key) ? p[s.key] : null, set: has(p, s.key), note: s.note, override: ov(s.key),
    editable: !!SHARED_TARGETS[s.key], type: SHARED_TARGETS[s.key] ? SHARED_TARGETS[s.key].type : null, retrains: SHARED_TARGETS[s.key] ? targetOf(game, p, s.key).dl : [] }));
  // every override of the game in one list, including one whose setting is no longer shown (so it can be released)
  const active = Object.keys(overrides).filter((k) => ov(k)).sort().map((key) => {
    const t = targetOf(game, p, key);
    return { key, ...ov(key), tuned: t.error ? (overrides[key].tuned ?? null) : t.tuned, rows: overrides[key].rows || [], orphan: !!t.error,
      retrains: t.error ? (overrides[key].rows || []).filter((r) => DL_ROWS.includes(r)) : t.dl };
  });
  return {
    game, group: isMarket(game) ? 'Markets' : 'Lottery', builtAt: now.toISOString(),
    file: { name: `bestParams_${game}.json`, exists: !!params, modified: stat ? stat.mtime.toISOString() : null, keys: Object.keys(p).length },
    shared, groups, empty, unread, unassigned, overrides: active, nextRun: nextRun(game),
    variants: { sampled: SAMPLED_GAMES.includes(game), memorise: !isMarket(game) && live.useMemoriseRows !== false },
  };
}

// The file as the predictor reads it; null when the game has none yet (a game not tuned since a switch).
function readParams(root, game) {
  const file = path.join(root, `bestParams_${game}.json`);
  try {
    const stat = fs.statSync(file);
    return { params: JSON.parse(fs.readFileSync(file, 'utf-8')), stat };
  } catch (e) {
    if (e.code === 'ENOENT') return { params: null, stat: null };
    throw e;
  }
}

function gameView(root, game) {
  const { params, stat } = readParams(root, game);
  const g = gameOverrides(root, game);
  return parametersView(game, params, stat, new Date(), { ...g, done: freshDone(root, game) });
}

// GET /api/admin/parameters/:game (admin only: install is called inside admin.js's guard). The two POSTs (set and
// release an override) are installed by system.js, which has the form token, the JSON body and the audit log.
function install(app, { root, answer }) {
  app.get('/api/admin/parameters/:game', (req, res) => answer(res, () => {
    const game = String(req.params.game || '');
    if (!GAMES.includes(game)) return { error: 'unknown game', status: 404 };
    return gameView(root, game);
  }));
}

module.exports = { GAMES, FAMILIES, SHARED, UNREAD, RECORDS, OVERRIDES_FILE, familyOf, parametersView, readParams, readOverrides, gameOverrides,
  served, targetOf, checkValue, setOverride, releaseOverride, gameView, install };
