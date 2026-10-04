const express = require('express');
const path = require('path');
const fs = require('fs');

const config = require("./config");
const auth = require("./auth");
const council = require("./council");
const services = require("./services");
const jobs = require("./jobs");
const whatsnew = require("./whatsnew");

const app = express();

// Login and user management (auth.js, README roadmap item 3): form posts
// need the body parser; the middleware redirects anonymous requests to
// /login when WEB_USER / WEB_PASSWORD are set and is a no-op otherwise.
// Half-configured credentials would silently mean "open", so they stop the
// server instead. trust proxy makes req.ip - the login lockout key - the
// client's address when a reverse proxy on this machine forwards to us.
if (auth.misconfigured()) {
  console.error('Set both WEB_USER and WEB_PASSWORD to require a login, or neither for open access - refusing to start with only one of them.');
  process.exit(1);
}
app.set('trust proxy', config.TRUST_PROXY);
app.use(express.urlencoded({ extended: false, limit: '16kb' }));
app.use(auth.middleware);

// Init the LLM Council
council.install(app, {
  header: generateHeader,
  footer: generateFooter,
  escapeHtml: auth.escapeHtml
});

// Paths
const dataPath = path.join(__dirname, 'data', 'database');
// The weekly control experiments (NullControls.py, RandomnessDiscrimination.py)
// write here; controls.js reads them for the History page. Optional: the
// cards render without them.
const controlsPath = path.join(__dirname, 'data', 'controls');
const controls = require('./controls');
// The Crypto and Shares pages (README roadmap item 4): drawn from the
// settlement's exports under data/markets/, see markets.js.
const marketsPath = path.join(__dirname, 'data', 'markets');
const markets = require('./markets');
// modelsPath removed as it is no longer used

// --- GAME SHAPES ---
// Mirrors Predictor.py's SPECIAL_COLUMN_COUNTS: how many trailing values of a
// full result/ticket row are special numbers (euromillions stars, eurodreams
// dream number, vikinglotto viking, jokerplus zodiac sign code). A main-ball
// hit and a special-ball hit are different prize dimensions, so the UI must
// never pool them.
const SPECIAL_COLUMN_COUNTS = { euromillions: 2, eurodreams: 1, vikinglotto: 1, jokerplus: 1 };

// --- Lotto multi-pick (src/MultiPick.py, README "Lotto multi-pick") ---
// A lotto row carries, besides its six numbers (ordered by the model's own
// probability, highest first), the next three it did not play - row.multiPick
// - for a 7-, 8- or 9-number system play. The table shows them as three
// shaded cells; the hits column adds the hits among all nine.
const comb = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return r; };
// P(at least minHits of the drawn numbers among `picked`): the exact hypergeometric, as src/MultiPick.chance
const winChance = (picked, pool = 45, draw = 6, minHits = 3) => { let sum = 0; for (let h = minHits; h <= Math.min(draw, picked); h++) sum += comb(draw, h) * comb(pool - draw, picked - h); return sum / comb(pool, picked); };
const MULTI_PICK = { lotto: { extra: 3, grids: { 7: 7, 8: 28, 9: 84 }, gridPrice: 1.5, chance: { 6: winChance(6), 7: winChance(7), 8: winChance(8), 9: winChance(9) }, perExtra: 6 / 45 } };
const multiPickFor = (game) => MULTI_PICK[game] || null;
const multiHits = (ticketMains, extras, realMains) => new Set([...ticketMains, ...extras].map(Number).filter((n) => realMains.includes(n))).size;

// --- Which rows a reader sees by default (3 Oct 2026, owner's decision) ---
// A user sees the next prediction of the top five rows of the History ranking
// (modelPerformance.json, the same order as the Best model card) and can
// switch the rest on; the administrator sees every row, switch on. Nothing
// is dropped from the day files - it is display only.
const TOP_MODELS_SHOWN = 5;
// The whole ranking (names in order) and its metric; generateTable takes the
// first TOP_MODELS_SHOWN names that are actually in the table it draws.
function rankingFor(game) {
  try {
    const report = JSON.parse(fs.readFileSync(path.join(dataPath, 'modelPerformance.json'), 'utf-8'));
    const info = report.games && report.games[game];
    if (!info || !Array.isArray(info.models) || !info.models.length) return null;
    return { names: info.models.map((m) => String(m.name)), metric: info.metric === 'profit_per_bet' ? 'profit per bet' : 'average hits' };
  } catch (e) { return null; }
}
const seesAllModels = (user) => !user || user.open === true || user.role === 'admin';
const nextTableOptions = (game, user) => ({ ranking: rankingFor(game), showAll: seesAllModels(user) });

// --- the three sections (decided with the owner on 1 Oct 2026) ---
// The site is organised by what is predicted: the lottery games, crypto and
// shares. The lottery pages (new predictions, History) list the lottery
// folders only; the two markets have their own pages (markets.js), and their
// day files under data/database/ remain reachable as the "game view" - the
// same digits, scored the same way - from those pages and by URL.
const MARKET_GAMES = ['crypto', 'shares', 'cryptoweek', 'sharesweek'];   // the two markets and their week games (one draw per week)
const isMarketGame = (game) => MARKET_GAMES.includes(game);
const marketBase = (game) => String(game).replace(/week$/, '');
const isMarketFolder = (folder) => MARKET_GAMES.includes(gameFromFolder(folder));
// A route parameter must be a plain folder or file name. Express decodes %2F
// to '/', and path.join would normalise 'x/..' away, so an existsSync check
// alone lets 'lotto<img ...>/..' through to the page (found in review).
const isPlainName = (name) => typeof name === 'string' && /^[A-Za-z0-9._-]+$/.test(name) && name !== '.' && name !== '..';
function databaseFolders() {
  return fs.readdirSync(dataPath, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((dir) => dir.name);
}
function lotteryFolders() { return databaseFolders().filter((folder) => !isMarketFolder(folder)); }
function dayFiles(folder) {
  return fs.readdirSync(path.join(dataPath, folder)).filter((file) => file.endsWith('.json'))
    .sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));
}
// The instrument symbols of a market, in slot order, from its page record
// (data/markets/<market>.json) - so the game view can head its columns BTC,
// ETH, ... instead of Num 1, Num 2. Empty when the record is not there yet.
function marketSymbols(game) {
  const record = markets.loadMarket(marketsPath, game);
  if (!record) return [];
  return record.instruments.filter((i) => i && typeof i === 'object' && i.symbol !== undefined && i.symbol !== null)
    .sort((a, b) => (Number(a.position) || 0) - (Number(b.position) || 0)).map((i) => auth.escapeHtml(String(i.symbol)));
}
// The lottery section's own navigation: new predictions and the History.
function lotteryNav(active) {
  const item = (href, label, key) => `<a href="${href}" style="text-decoration:none; padding:6px 12px; border-radius:4px; color:${active === key ? 'white' : '#2c3e50'}; background:${active === key ? '#2c3e50' : '#e1e4e8'};">${label}</a>`;
  return `<div style="display:flex; gap:8px; margin:0 0 18px; flex-wrap:wrap; align-items:center;"><span style="color:#7f8c8d; font-size:0.9em; margin-right:4px;">Lottery games:</span>
    ${item('/lottery', 'New predictions', 'predictions')}${item('/database', 'History', 'history')}</div>`;
}
// The note on a market's game view: what the digits are, where the prices are.
function marketGameViewNote(game) {
  const base = marketBase(game);
  const title = markets.MARKETS[base] ? markets.MARKETS[base].title : base;
  const symbols = marketSymbols(game);
  const noun = base === 'crypto' ? 'coin' : 'share';
  if (game !== base) {
    return `<div style="background:#fef9e7; border:1px solid #f9e79f; border-radius:8px; padding:12px 16px; margin-bottom:20px; color:#7d6608;">
    <b>This is the game view of the ${base} market's week game.</b> Each week is one draw with one slot per ${noun}${symbols.length ? ` (${symbols.join(', ')}, in that order)` : ''}.
    The digit is the <b>bin</b> of the week's move - the close of the week's last ${base === 'crypto' ? 'day (Sunday, UTC)' : 'session (normally Friday)'} against the close of the week before, placed among the ${noun}'s own past weekly moves sorted
    from worst to best and cut into ten equal piles. A hit is the right bin in the right slot. The <a href="/markets/${base}" style="color:#7d6608; font-weight:bold;">${title} page</a> shows the coming week's calls in its <i>Week ahead</i> card.</div>`;
  }
  return `<div style="background:#fef9e7; border:1px solid #f9e79f; border-radius:8px; padding:12px 16px; margin-bottom:20px; color:#7d6608;">
    <b>This is the game view of the ${game} market.</b> Each day is one draw with one slot per ${game === 'crypto' ? 'coin' : 'share'}${symbols.length ? ` (${symbols.join(', ')}, in that order)` : ''}.
    The digit is the <b>bin</b> of that day's move - today's close against yesterday's close, placed among the ${game === 'crypto' ? 'coin' : 'share'}'s own past daily moves sorted
    from worst to best and cut into ten equal piles: 0 is a day among its worst tenth, 9 among its best, 4 and 5 barely moved. A hit is the right bin in the right slot.
    The <a href="/markets/${game}" style="color:#7d6608; font-weight:bold;">${title} page</a> explains this in three steps with the newest day as the example, and shows the same days as prices and moves.</div>`;
}

// --- JOKER+ ---
// The Python side stores the Joker+ zodiac sign as its 0..11 code (regulation
// order, see Helpers.ZODIAC_CANONICAL) so every model works on ints; the UI
// decodes it back to the spelling the National Lottery's CSV exports use
// (code 2 is 'Tweeling' there, not the regulation's 'Tweelingen') so the
// page reads like the official result listing.
const ZODIAC = ["Ram", "Stier", "Tweeling", "Kreeft", "Leeuw", "Maagd",
                "Weegschaal", "Schorpioen", "Boogschutter", "Steenbok", "Waterman", "Vissen"];
// Alternate spellings tolerated when a row carries a name instead of a code
// (hand-edited files, the draw API's English names) - a value that only
// differs in spelling must still count as the same sign.
const ZODIAC_ALIASES = {
  tweelingen: 2, aries: 0, taurus: 1, gemini: 2, cancer: 3, leo: 4, virgo: 5,
  libra: 6, scorpio: 7, sagittarius: 8, capricorn: 9, aquarius: 10, pisces: 11
};

// Sign code (0..11) of a ticket/result value, or null when it is absent or
// unrecognizable: an unknown sign simply cannot match, it must not break the
// page. Mirrors Helpers._zodiac_code_or_none.
function zodiacCode(value) {
  if (value === null || value === undefined || typeof value === 'boolean') return null;
  if (typeof value === 'number' || /^\s*\d+\s*$/.test(String(value))) {
    const code = Number(value);
    return Number.isInteger(code) && code >= 0 && code < ZODIAC.length ? code : null;
  }
  const key = String(value).trim().toLowerCase();
  const idx = ZODIAC.findIndex(name => name.toLowerCase() === key);
  if (idx >= 0) return idx;
  return Object.prototype.hasOwnProperty.call(ZODIAC_ALIASES, key) ? ZODIAC_ALIASES[key] : null;
}

// Display name of a sign value. Unknown values are shown verbatim rather than
// hidden so a bad code is visible in the UI instead of silently vanishing.
function zodiacName(value) {
  const code = zodiacCode(value);
  return code === null ? String(value) : ZODIAC[code];
}

// Joker+ rows are [d1..d6, zodiacCode]; for display the trailing code becomes
// its name. Other games and mains-only 6-digit rows are returned untouched.
function displayRow(row, game) {
  if (game !== 'jokerplus' || !Array.isArray(row) || row.length !== 7) return row;
  return row.slice(0, 6).concat([zodiacName(row[6])]);
}

// Leading/trailing runs of a Joker+ ticket against the drawn digits - the two
// quantities the game pays on (mirrors Helpers.jokerplus_runs). L = number of
// LEADING positions matching consecutively from the left end, R = number of
// TRAILING positions matching consecutively from the right end. Compared
// positionally in drawn order, never as sets: digits repeat within a draw,
// so membership tests are meaningless here. When every position matches the
// match is full and R is reported as 0 so the two runs never double-count
// the same positions; otherwise the mismatching position separates them and
// L + R <= 5. Only the leading min(len) positions are compared so a 6-digit
// mains-only ticket and a 7-value [digits + sign] row both work.
function jokerplusRuns(ticketDigits, realDigits) {
  const n = Math.min(ticketDigits.length, realDigits.length);
  let left = 0;
  while (left < n && Number(ticketDigits[left]) === Number(realDigits[left])) left += 1;
  if (left === n) return { left: n, right: 0 };
  let right = 0;
  while (right < n - left && Number(ticketDigits[n - 1 - right]) === Number(realDigits[n - 1 - right])) right += 1;
  return { left, right };
}

// Whether a Joker+ ticket's sign matches the drawn sign (0/1 for the "(Z)"
// part of the notation). A ticket without a sign cell cannot match.
function jokerplusSignHit(ticketSpecials, realSpecials) {
  if (ticketSpecials.length === 0 || realSpecials.length === 0) return 0;
  const ticketSign = zodiacCode(ticketSpecials[0]);
  return ticketSign !== null && ticketSign === zodiacCode(realSpecials[0]) ? 1 : 0;
}

// Official Joker+ prize structure (Reglement Joker+, Sept 2023), mirroring
// Helpers.PAYOUT_TABLE_JOKERPLUS. Left and right runs each pay per run length
// and cumulate; all six digits matching is its own tier (the fixed minimum
// jackpot when the sign matches too, which then replaces the sign refund
// rather than adding to it). A matching sign alone refunds the stake. The
// digits are system-generated; only the sign is chosen by the player.
const PAYOUT_TABLE_JOKERPLUS = {
  runs: { 0: 0, 1: 2, 2: 5, 3: 20, 4: 200, 5: 2000 },
  full: 20000,
  fullWithSign: 200000,
  sign: 1.5,
  betCost: 1.5
};

// Net profit of one Joker+ ticket (mirrors Helpers.jokerplus_ticket_profit).
// ticket/realResult are [d1..d6, zodiacCode]; a 6-value ticket or result is
// scored with the sign unknown (no sign match possible). Invalid shapes score
// 0 like the other games' invalid-shape branches in calculateProfit.
function jokerplusTicketProfit(ticket, realResult) {
  if (!Array.isArray(ticket) || !Array.isArray(realResult)) return 0;
  if (![6, 7].includes(ticket.length) || ![6, 7].includes(realResult.length)) return 0;
  const ticketDigits = ticket.slice(0, 6).map(Number);
  const realDigits = realResult.slice(0, 6).map(Number);
  if (ticketDigits.some(Number.isNaN) || realDigits.some(Number.isNaN)) return 0;
  const signMatch = jokerplusSignHit(ticket.slice(6), realResult.slice(6)) === 1;
  const { left, right } = jokerplusRuns(ticketDigits, realDigits);
  const table = PAYOUT_TABLE_JOKERPLUS;
  let payout;
  if (left === 6) {
    payout = signMatch ? table.fullWithSign : table.full;
  } else {
    payout = table.runs[left] + table.runs[right];
    if (signMatch) payout += table.sign;
  }
  return payout - table.betCost;
}

// Joker+ profits carry half-euro cents (1.50 stake/refund), so they are shown
// with two decimals; the other payout games keep their integer rendering.
function formatProfit(value, game) {
  return game === 'jokerplus' && typeof value === 'number' ? value.toFixed(2) : value;
}

// Frequency dict fed to the bar charts. The day JSON's numberFrequency pools
// every value of every predicted row; for Joker+ that would mix the zodiac
// code in with the digits. A code 3 is indistinguishable from a digit 3
// here, but codes 10 and 11 can only be signs, so the chart is restricted to
// the digit range 0..9 and its x-axis stays "digits". Other games are
// returned as-is (same object, so their charts render byte-identically).
function chartFrequency(freq, game) {
  if (game !== 'jokerplus' || !freq) return freq;
  const digitsOnly = {};
  Object.keys(freq).forEach((key) => { if (/^\d$/.test(String(key).trim())) digitsOnly[key] = freq[key]; });
  return digitsOnly;
}

// --- LOGIC: is the pipeline busy right now? ---
// Every Python entry point (Predictor.py, the five tuners, TrainMetaLearner)
// takes this one PID lock file in the repo root, so it is the cheapest honest
// answer to "is a run in progress" - no scheduler needed, and it stays true
// when a job is started by hand or by cron. /proc gives the command line, so
// the banner can say which job it is rather than just "busy".
const PIPELINE_LOCK = path.join(__dirname, 'process.lock');
const JOB_NAMES = {
  'Predictor.py': 'Today\'s predictions are being computed',
  'HyperoptStatistics.py': 'Weekly tuning: statistical models',
  'HyperoptBoost.py': 'Weekly tuning: boosting models',
  'HyperoptRLTicket.py': 'Weekly tuning: RL ticket model',
  'HyperoptEnsemble.py': 'Weekly tuning: ensemble subsets',
  'HyperoptQuantum.py': 'Weekly tuning: quantum meta-learners',
  'HyperoptDeepLearning.py': 'Tuning: deep learning models',
  'TrainMetaLearner.py': 'Retraining the meta-learner',
  'RandomnessDiscrimination.py': 'Weekly controls: randomness discrimination',
  'NullControls.py': 'Weekly controls: null histories',
  'IrrelevantFeatureControl.py': 'Weekly controls: irrelevant-feature control',
  'MarketsDaily.py': 'Refreshing the market games (bars and bins)',
};

function pipelineStatus() {
  let pid;
  let since = null;
  try {
    pid = Number(fs.readFileSync(PIPELINE_LOCK, 'utf-8').trim());
    since = fs.statSync(PIPELINE_LOCK).mtimeMs;
  } catch (e) {
    return null;                       // no lock: nothing running
  }
  if (!pid || Number.isNaN(pid)) return null;
  let cmdline;
  try {
    cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf-8').replace(/\0/g, ' ').trim();
  } catch (e) {
    return null;                       // stale lock of a dead run - the scripts clean it up themselves
  }
  const script = Object.keys(JOB_NAMES).find((name) => cmdline.includes(name));
  return { pid, since, what: script ? JOB_NAMES[script] : 'A pipeline job is running' };
}

function pipelineBanner() {
  const status = pipelineStatus();
  if (!status) return '';
  const minutes = Math.max(0, Math.round((Date.now() - status.since) / 60000));
  const running = minutes < 1 ? 'just started' : `running for ${minutes < 90 ? `${minutes} min` : `${(minutes / 60).toFixed(1)} h`}`;
  return `
    <div style="background:#eaf4fd; border:1px solid #aed6f1; color:#21618c; border-radius:8px; padding:12px 16px; margin-bottom:20px; display:flex; align-items:center; gap:12px;">
      <span style="display:inline-block; width:14px; height:14px; border:3px solid #aed6f1; border-top-color:#2980b9; border-radius:50%; animation:sp-spin 1s linear infinite;"></span>
      <span><b>${status.what}.</b> ${running} - this page updates when it finishes.</span>
    </div>
    <style>@keyframes sp-spin { to { transform: rotate(360deg); } }</style>
    <script>setTimeout(function () { location.reload(); }, 60000);</script>`;
}

// --- LOGIC: which draw a "new prediction" is for ---
// A day JSON is named after the draw it scored, and its "newPrediction" is
// the prediction for the NEXT draw of that game (checked against the data:
// file D's newPrediction is exactly what file D+1 carries as its scored
// currentPrediction). The schedule differs per game and has changed before -
// Keno/Pick3/Joker+ draw daily, Lotto Wed+Sat, Euromillions Tue+Fri,
// EuroDreams Mon+Thu, VikingLotto Wed - so it is read from the game's own
// recent draw dates instead of being hardcoded: the weekdays of the last
// SCHEDULE_WINDOW stored draws are the schedule, and the prediction applies
// to the first date after its file that falls on one of them.
const SCHEDULE_WINDOW = 40;
const WEEKDAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// "2026-9-16.json" (day files are not zero padded) -> local Date, or null
// for anything that isn't a day file.
function parseDayFileDate(file) {
  const parts = String(file).replace('.json', '').split('-').map(Number);
  if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) return null;
  const [year, month, day] = parts;
  const date = new Date(year, month - 1, day);
  return (date.getFullYear() === year && date.getMonth() === month - 1 && date.getDate() === day) ? date : null;
}

function formatDrawDate(date) {
  return `${WEEKDAY_NAMES[date.getDay()]} ${date.getDate()} ${MONTH_NAMES[date.getMonth()]} ${date.getFullYear()}`;
}

// First date after `anchor` (default: the newest stored draw) whose weekday
// is one the game actually draws on. null when there are no day files or no
// such date within two weeks (an unreadable schedule shows no date at all
// rather than a guessed one).
function nextDrawDate(fileDates, anchor) {
  const dates = fileDates.filter(Boolean).sort((a, b) => a - b);
  if (!dates.length) return null;
  const from = anchor || dates[dates.length - 1];
  // Only draws up to the anchor define its schedule, so a history page from
  // before a schedule change is read with the schedule of its own time.
  const upToAnchor = dates.filter((d) => d <= from);
  const weekdays = new Set((upToAnchor.length ? upToAnchor : dates).slice(-SCHEDULE_WINDOW).map((d) => d.getDay()));
  const next = new Date(from.getFullYear(), from.getMonth(), from.getDate());
  for (let step = 0; step < 14; step += 1) {
    next.setDate(next.getDate() + 1);
    if (weekdays.has(next.getDay())) return next;
  }
  return null;
}

// The "for the draw of ..." line next to a new-prediction table. On the home
// page (warnWhenPast) a date in the past means the newest stored prediction
// is for a draw that has already taken place - the predictor has not run
// since - which the reader must see; on a history page that is the normal
// state of every older day, so the date is shown plainly.
function drawDateMeta(fileDates, anchor, warnWhenPast) {
  const next = nextDrawDate(fileDates, anchor);
  if (!next) return '';
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const label = formatDrawDate(next);
  if (warnWhenPast && next < today) {
    return `<span class="card-meta" style="margin-left: 10px; color: #c0392b;" title="This is the newest stored prediction; the predictor has not produced one for a later draw yet.">for the draw of ${label} - already drawn, no newer run yet</span>`;
  }
  const when = next.getTime() === today.getTime() ? 'today, ' : '';
  return `<span class="card-meta" style="margin-left: 10px;">for the draw of ${when}${label}</span>`;
}

// Database folder names equal game names today, but the routes historically
// matched with includes() (e.g. a "keno_backup" folder still behaves as keno),
// so keep that tolerance. vikinglotto must be tested before lotto because
// "vikinglotto".includes("lotto") is true.
function gameFromFolder(folder) {
  const games = ["euromillions", "eurodreams", "vikinglotto", "lotto", "keno", "pick3", "jokerplus", "cryptoweek", "sharesweek", "crypto", "shares"];   // the week games before their base: "cryptoweek".includes("crypto")
  for (const g of games) if (folder.includes(g)) return g;
  return folder;
}

// Split a real-result row (a full CSV row) into main, special and bonus
// numbers. Lotto follows the real game's tiers: 6 played numbers score
// against the 6 drawn mains, and the 7th (bonus) value only supplements a
// partial match - "5 (1)" is a high tier, "6 (0)" the jackpot - so the bonus
// is a separate pool matched against the ticket ITSELF (a play has no bonus
// slot), mirroring Helpers.find_best_matching_prediction.
function splitRealResult(realResult, game) {
  if (!Array.isArray(realResult) || realResult.length === 0) return { mains: [], specials: [], bonus: [] };
  const s = SPECIAL_COLUMN_COUNTS[game] || 0;
  if (s > 0 && realResult.length > s) return { mains: realResult.slice(0, -s), specials: realResult.slice(-s), bonus: [] };
  if (game === "lotto") return { mains: realResult.slice(0, 6), specials: [], bonus: realResult.slice(6) };
  return { mains: realResult.slice(), specials: [], bonus: [] };
}

// Split one prediction row. For special-column games a row longer than the
// real main count carries its specials appended at the end; a row that is not
// longer is a mains-only ticket (RL Ticket Model rows, keno subset tickets),
// so it gets no special cells.
function splitTicket(row, realMains, specialCount) {
  if (specialCount > 0 && row.length > realMains.length) {
    return { mains: row.slice(0, -specialCount), specials: row.slice(-specialCount) };
  }
  return { mains: row, specials: [] };
}

// --- HELPER: Generate HTML Header ---
function generateHeader(title = "Sequence Predictor", user = null) {
  return `
  <!DOCTYPE html>
  <html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <style>
      /* GLOBAL RESET */
      * { box-sizing: border-box; }

      body { 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        margin: 0; 
        padding-top: 100px; /* Space for fixed header */
        background-color: #f0f2f5; 
        color: #333;
      }
      
      /* STICKY NAVBAR */
      .navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #2c3e50;
        color: white;
        padding: 15px 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        z-index: 1000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 80px;
      }
      
      .navbar a {
        color: #ecf0f1;
        text-decoration: none;
        margin-right: 20px;
        font-weight: 600;
        font-size: 1.1em;
        transition: color 0.2s;
      }
      .navbar a:hover { color: #3498db; }
      
      .nav-group { display: flex; align-items: center; }
      
      /* NAV BUTTON (e.g. the day page's "Back to History" link) */
      .nav-btn {
        background-color: #34495e; color: white; padding: 10px 15px;
        border: 1px solid #455a64; cursor: pointer; border-radius: 6px;
        font-size: 1em; transition: background 0.2s;
      }
      .nav-btn:hover { background-color: #2c3e50; }

      /* FIRST-LOGIN INTRODUCTION AND WHAT'S NEW (whatsnew.js) */
      ${whatsnew.DIALOG_CSS}

      /* LOGIN STATE + USER ADMIN FORMS (auth.js) */
      .nav-user { color: #bdc3c7; font-size: 0.9em; display: flex; align-items: center; gap: 12px; }
      .nav-user b { color: white; }
      .nav-user form { margin: 0; }
      .nav-user .nav-btn { padding: 6px 12px; font-size: 0.9em; }
      .auth-form label { display: block; font-weight: 600; margin: 10px 0 4px 0; max-width: 420px; }
      .auth-form input, .inline-form input[type=password] {
        padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;
      }
      .auth-form input { width: 100%; display: block; }
      .auth-form .nav-btn { margin-top: 12px; }
      .inline-form { display: inline-flex; gap: 6px; align-items: center; margin: 2px 6px 2px 0; }
      .inline-form .nav-btn { padding: 6px 10px; font-size: 0.9em; }
      
      /* LAYOUT */
      .container { padding: 20px; max-width: 1400px; margin: auto; }   /* wide enough for the lotto rows with their 7th-9th numbers and hits without a horizontal scroll (4 Oct 2026); the market pages used 1400px already */
      
      /* COLLAPSIBLE CARD STYLES */
      .card {
        background: white;
        margin-bottom: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e1e4e8;
        overflow: hidden;
      }
      
      .card-header {
        background-color: #fff;
        padding: 15px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
        transition: background-color 0.2s;
        border-bottom: 1px solid transparent;
      }
      .card-header:hover { background-color: #f8f9fa; }
      
      .card.expanded .card-header {
        background-color: #f1f3f5;
        border-bottom: 1px solid #e1e4e8;
      }

      .card-title { font-size: 1.2em; font-weight: bold; margin: 0; color: #2c3e50; }
      .card-meta { font-size: 0.9em; color: #7f8c8d; }
      
      .card-icon {
        transition: transform 0.3s ease;
        font-size: 1.2em;
        color: #7f8c8d;
      }
      .card.expanded .card-icon { transform: rotate(180deg); }

      .card-body {
        display: none; /* Hidden by default */
        padding: 20px;
        animation: fadeIn 0.3s ease-in-out;
      }
      /* Only show when expanded class is present */
      .card.expanded .card-body { display: block; }

      @keyframes fadeIn {
        from { opacity: 0; } to { opacity: 1; }
      }

      /* SCROLLABLE TABLES */
      .table-wrapper {
        width: 100%;
        overflow-x: auto; 
        margin-top: 15px;
        border: 1px solid #e1e4e8;
        border-radius: 4px;
      }

      table { width: 100%; border-collapse: collapse; background: white; font-size: 0.9em; min-width: 600px; }
      th, td { padding: 12px 15px; border: 1px solid #e1e4e8; text-align: center; white-space: nowrap; }
      th { background-color: #f8f9fa; color: #333; font-weight: bold; }
      tr:nth-child(even) { background-color: #f8f9fa; }
      
      /* FORMS & BUTTONS */
      button { cursor: pointer; }

    </style>
  </head>
  <body>
    <div class="navbar">
      <div class="nav-group">
        <a href="/" style="font-size: 1.3em;">📊 Predictor</a>
        <a href="/lottery">Lottery</a>
        <a href="/markets/crypto">Crypto</a>
        <a href="/markets/shares">Shares</a>
        <a href="/council">Council</a>
        ${user && user.role === 'admin' ? '<a href="/admin/users">Users</a><a href="/admin/jobs">Jobs</a>' : ''}
        ${whatsnew.navLink(user)}
      </div>
      ${user && !user.open ? `
      <div class="nav-user">
        <span>Signed in as <a href="/account" style="color:white; text-decoration:underline;">${auth.escapeHtml(user.name)}</a></span>
        <form method="post" action="/logout"><input type="hidden" name="_csrf" value="${auth.escapeHtml(user.csrf)}"><button type="submit" class="nav-btn">Logout</button></form>
      </div>` : ''}
    </div>

    <script>
      // Toggle Card Logic
      function toggleCard(header) {
        const card = header.parentElement;
        card.classList.toggle('expanded');
      }
      // "Show all models": the rows beyond the top of the ranking are in the
      // table, hidden; the box shows them (display only, nothing reloads)
      function toggleAllModels(box) {
        const wrap = box.closest('.table-wrapper');
        if (!wrap) return;
        wrap.querySelectorAll('tr.extra-model').forEach((row) => { row.style.display = box.checked ? '' : 'none'; });
      }
    </script>
    ${whatsnew.dialog(user)}
    <div class="container">
  `;
}

function generateFooter() {
  return `</div></body></html>`;
}

// --- LOGIC: Table Generation ---
// realResult is the full drawn row (mains + specials, or mains + lotto bonus);
// cells are highlighted index-aware so a predicted star only lights up against
// the drawn stars and a predicted main only against the drawn mains.
function generateTable(data, title = '', realResult = [], calcProfit = false, game = "", options = {}) {
  const modelRows = data || [];
  if (modelRows.length === 0) return `<p style="padding: 10px; color: #888;">No predictions.</p>`;

  const specialCount = SPECIAL_COLUMN_COUNTS[game] || 0;
  // the multi-pick chrome (7th-9th columns, "of 9", legend) only where a row
  // carries extras, so lotto pages scored before the feature stay six-column
  const multiCfg = multiPickFor(game);
  const multi = multiCfg && modelRows.some((m) => Array.isArray(m.multiPick) && m.multiPick.length) ? multiCfg : null;
  // the reader's default view: the first TOP_MODELS_SHOWN names of the
  // ranking that are in THIS table (a ranked row may be missing from a day);
  // with fewer than two present the table shows everything
  const allNames = [...new Set(modelRows.map((m) => m.name || 'not known'))];
  const ranking = options.ranking && Array.isArray(options.ranking.names) ? options.ranking : null;
  const ranked = ranking ? ranking.names.filter((n) => allNames.includes(n)).slice(0, options.limit || TOP_MODELS_SHOWN) : [];
  const topSet = ranked.length >= 2 && ranked.length < allNames.length ? new Set(ranked) : null;
  const showAll = options.showAll !== false;
  const hiddenNames = topSet ? allNames.filter((n) => !topSet.has(n)) : [];
  // Joker+ is positional: hits are leading/trailing runs, not membership, and
  // its 7th value is a zodiac sign code that must be shown as a name. The
  // market games (crypto, shares) are positional too: a return bin in the
  // right instrument's slot, so a cell lights only in its own slot.
  const isJoker = game === 'jokerplus';
  const isMarket = isMarketGame(game);
  const { mains: realMains, specials: realSpecials, bonus: realBonus } = splitRealResult(realResult, game);
  // No real result (next-draw / home tables) -> no highlighting and no Hits column.
  const hasReal = realMains.length > 0;

  let html = `<div class="table-wrapper">`;
  if (title) html += `<div style="padding: 10px; font-weight: bold; background: #f8f9fa; border-bottom: 1px solid #ddd;">${title}</div>`;
  if (hiddenNames.length) {
    html += `<label style="display:block; padding:8px 10px; font-size:0.85em; color:#555; background:#f8f9fa; border-bottom:1px solid #ddd; cursor:pointer;">
      <input type="checkbox" ${showAll ? 'checked' : ''} onchange="toggleAllModels(this)"> Show all ${allNames.length} models
      <span style="color:#7f8c8d;">- by default the top ${ranked.length} of the <a href="/database">History</a> ranking by ${ranking.metric} over every scored draw</span></label>`;
  }
  html += '<table border="1">';

  html += '<tr><th style="min-width: 150px;">Model</th><th style="width: 50px;">#</th>';
  if (modelRows.length > 0 && modelRows[0].predictions.length > 0) {
    // Joker+'s 7th column is the sign, not a seventh number; a market's
    // columns are its instruments (the slot is the coin or share).
    const symbols = isMarket ? marketSymbols(game) : [];
    Array.from({ length: modelRows[0].predictions[0].length }).forEach((_, i) => html += (isJoker && i === 6) ? '<th>Sign</th>'
      : (isMarket ? `<th title="the return bin of ${symbols[i] || `slot ${i + 1}`}">${symbols[i] || `Slot ${i + 1}`}</th>` : `<th>Num ${i + 1}</th>`));
    if (multi) {
      const base = modelRows[0].predictions[0].length;
      Array.from({ length: multi.extra }).forEach((_, i) => html += `<th style="background:#fdf2e9; color:#7d3c0f;" title="multi-pick: the model's next most probable number, for a system play of ${base + i + 1} numbers (${multi.grids[base + i + 1]} grids, ${(multi.grids[base + i + 1] * multi.gridPrice).toFixed(2).replace(/\.00$/, '')} EUR)">${base + i + 1}th</th>`);
    }
  }
  if(hasReal) html += `<th>Hits${multi ? ' <span style="color:#7d3c0f; font-weight:normal;">· of 9</span>' : ''}</th>`;
  if(calcProfit) html += '<th>Profit</th>';
  html += '</tr>';

  modelRows.forEach((model) => {
    model.predictions.forEach((row, rowIndex) => {
      const modelType = model.name || "not known";
      const { mains: ticketMains, specials: ticketSpecials } = splitTicket(row, realMains, specialCount);
      // Joker+ pays on positional runs (see jokerplusRuns) plus the sign, so
      // its cells are lit by run membership: the leading run green, the
      // trailing run blue, the sign amber - never by digit membership.
      const runs = (isJoker && hasReal) ? jokerplusRuns(ticketMains, realMains) : null;
      const signHit = runs ? jokerplusSignHit(ticketSpecials, realSpecials) : 0;
      const hiddenRow = topSet && !topSet.has(modelType);
      html += `<tr${hiddenRow ? ` class="extra-model"${showAll ? '' : ' style="display:none;"'}` : ''}>
        <td style="font-weight: bold; background: #f9f9f9;">${modelType}</td>
        <td style="font-weight: bold; background: #f9f9f9;">${rowIndex + 1}</td>`;
      row.forEach((cell, cellIndex) => {
        // Trailing cells past the ticket's main block are special columns and
        // only match against the drawn specials; everything else only against
        // the drawn mains (pick3 keeps its historical by-inclusion behavior,
        // keno subset rows and RL mains-only rows have no special cells).
        const isSpecialCell = ticketSpecials.length > 0 && cellIndex >= ticketMains.length;
        let cellStyle = '';
        let cellText = cell;
        if (isJoker) {
          if (isSpecialCell) cellText = zodiacName(cell);
          if (runs) {
            if (isSpecialCell) cellStyle = signHit ? 'background: #f39c12; color: white;' : '';
            else if (cellIndex < runs.left) cellStyle = 'background: #2ecc71; color: white;';
            else if (cellIndex >= ticketMains.length - runs.right) cellStyle = 'background: #3498db; color: white;';
          }
        } else {
          const isMatching = hasReal && (isSpecialCell ? realSpecials.includes(cell)
            : (isMarket ? realMains[cellIndex] === cell : realMains.includes(cell)));
          // Lotto bonus supplement: a played number equal to the bonus ball is
          // a tier-relevant hit ("5 (1)") but not a main hit - amber, not green.
          const isBonusMatch = hasReal && !isMatching && !isSpecialCell && realBonus.includes(cell);
          cellStyle = isMatching ? 'background: #2ecc71; color: white;'
            : (isBonusMatch ? 'background: #f39c12; color: white;' : '');
        }
        html += `<td style="text-align: center; ${cellStyle}">${cellText}</td>`;
      });
      // the multi-pick cells: the 7th-9th numbers of the main ticket only
      // (rowIndex 0); a row without a ranking shows them empty
      const extras = (multi && rowIndex === 0 && Array.isArray(model.multiPick)) ? model.multiPick.map(Number) : [];
      if (multi) {
        Array.from({ length: multi.extra }).forEach((_, i) => {
          const value = extras[i];
          if (value === undefined) { html += '<td style="text-align:center; background:#fdf2e9; color:#ccc;">-</td>'; return; }
          const hit = hasReal && realMains.includes(value);
          html += `<td style="text-align:center; border:1px dashed #e67e22; ${hit ? 'background:#2ecc71; color:white;' : 'background:#fdf2e9; color:#7d3c0f;'}">${value}</td>`;
        });
      }
      if(hasReal) {
        let hitDisplay;
        if (isJoker) {
          // "3/1 (1)" = leading run 3, trailing run 1, sign matched; a full
          // match reads "6/0 (Z)".
          hitDisplay = `${runs.left}/${runs.right} (${signHit})`;
        } else {
          const mainHits = isMarket ? ticketMains.filter((n, i) => realMains[i] === n).length
            : ticketMains.filter(n => realMains.includes(n)).length;
          const specialHits = ticketSpecials.filter(n => realSpecials.includes(n)).length
            + ticketMains.filter(n => realBonus.includes(n)).length;
          // "3 (1)" = 3 main hits, 1 special/bonus hit; games without a
          // special column or bonus just show the main count.
          hitDisplay = (specialCount > 0 || realBonus.length > 0) ? `${mainHits} (${specialHits})` : `${mainHits}`;
          if (multi && extras.length) {
            const nine = multiHits(ticketMains, extras, realMains);
            hitDisplay += ` <span style="color:#7d3c0f; font-weight:normal;" title="hits among all ${ticketMains.length + extras.length} numbers (a ${ticketMains.length + extras.length}-number system play)">· ${nine}</span>`;
          }
        }
        html += `<td style="font-weight: bold; background: #f9f9f9;">${hitDisplay}</td>`;
      }
      if(calcProfit) {
        const profit = calculateProfit(row, realResult, game, modelType);
        html += `<td style="background: #f9f9f9;">${formatProfit(profit, game)} €</td>`;
      }
      html += '</tr>';
    });
  });

  html += '</table></div>';
  if (multi) {
    const price = (size) => `${(multi.grids[size] * multi.gridPrice).toFixed(2).replace(/\.00$/, '')} EUR`;
    html += `<p style="color:#7d3c0f; font-size:0.85em; margin:8px 0 0; background:#fdf2e9; border:1px dashed #e67e22; border-radius:4px; padding:8px 10px;">
      <b>Multi-pick.</b> A row with shaded numbers lists its six in the model's own probability order, highest first - not small to large; the three shaded numbers are the model's next most
      probable ones, for a <b>system play</b>: 7 numbers = ${multi.grids[7]} grids = ${price(7)}, 8 numbers = ${multi.grids[8]} grids = ${price(8)}, 9 numbers = ${multi.grids[9]} grids = ${price(9)}
      (${multi.gridPrice.toFixed(2)} EUR a grid, against one grid for the six). More numbers win more by arithmetic alone: 6 numbers hit 3 of the six drawn - the smallest prize that needs no bonus number;
      ranks that use the bonus number are not counted here - ${(multi.chance[6] * 100).toFixed(1)}% of the time, 7 numbers ${(multi.chance[7] * 100).toFixed(1)}%, 8 numbers ${(multi.chance[8] * 100).toFixed(1)}%,
      9 numbers ${(multi.chance[9] * 100).toFixed(1)}%, and any extra number hits ${(multi.perExtra * 100).toFixed(1)}% of draws by luck. The <a href="/database">History</a> page tracks whether the model's
      extra numbers do better than that. A row without shaded numbers (a vote, the RL ticket, HybridStatisticalModel, or a row whose ranking failed that run) has no extras and keeps its six small to large.</p>`;
  }
  return html;
}

function calculateProfit(prediction, realResult, game, name) {
  const payoutTableKeno = {
    10: { 0: 3, 5: 1, 6: 4, 7: 10, 8: 200, 9: 2000, 10: 250000 },
    9: { 0: 3, 5: 2, 6: 5, 7: 50, 8: 500, 9: 50000 },
    8: { 0: 3, 5: 4, 6: 10, 7: 100, 8: 10000 },
    7: { 0: 3, 5: 3, 6: 30, 7: 3000 },
    6: { 3: 1, 4: 4, 5: 20, 6: 200 },
    5: { 3: 2, 4: 5, 5: 150 },
    4: { 2: 1, 3: 2, 4: 30 },
    3: { 2: 1, 3: 16 },
    2: { 2: 6.5 },
    "lost": -1
  };
  // Mirror of Helpers.PAYOUT_TABLE_PICK3 (Reglement Pick-3, juli 2024).
  const payoutTablePick3 = {
    straight: 500, straight_consolation: 1, box_with_doubles: 160, box_no_doubles: 80,
    front_pair: 50, back_pair: 50, bet_cost: 1
  };
  const played = prediction.length;

  switch (game) {
    case "keno": {
      // NEW LOGIC: Strictly ignore profit if prediction row > 10 numbers
      if (played > 10) return 0;

      const correctNumbers = prediction.filter(n => realResult.includes(n)).length;
      if (played >= 2 && played <= 10 && payoutTableKeno[played]) return payoutTableKeno[played][correctNumbers] ?? payoutTableKeno["lost"];
      return 0; 
    }
    case "pick3": {
      // Official cumulative model, the same as Helpers.pick3_ticket_profit
      // (which scores the backtests, the tuners and the performance report):
      // the tracked ticket plays every bet type at 1 EUR - straight, box,
      // front pair, back pair; a triple cannot play box, so its stake is 3 -
      // each bet is evaluated on its own, prizes cumulate, and the stake is
      // deducted. The previous version paid the first matching tier only and
      // never deducted the stake, so an exact [1,2,3] showed 500 here and
      // 676 in the backtest, and every pick3 row's History profit disagreed
      // with its tuning profit.
      if (played != 3 || realResult.length < 3) return 0;
      const pred = prediction.map(Number); const actual = realResult.slice(0, 3).map(Number);
      const distinct = new Set(pred).size;
      const isTriple = distinct === 1;
      const stake = (isTriple ? 3 : 4) * payoutTablePick3.bet_cost;
      let payout = 0;
      if (pred[0] === actual[0] && pred[1] === actual[1] && pred[2] === actual[2]) payout += payoutTablePick3.straight;
      else if (pred[2] === actual[2]) payout += payoutTablePick3.straight_consolation;
      if (!isTriple && [...pred].sort((a, b) => a - b).join(',') === [...actual].sort((a, b) => a - b).join(',')) {
        payout += distinct === 2 ? payoutTablePick3.box_with_doubles : payoutTablePick3.box_no_doubles;
      }
      if (pred[0] === actual[0] && pred[1] === actual[1]) payout += payoutTablePick3.front_pair;
      if (pred[1] === actual[1] && pred[2] === actual[2]) payout += payoutTablePick3.back_pair;
      return payout - stake;
    }
    case "jokerplus": {
      // Positional run tiers + sign, one 1.50 EUR stake per row (Z6).
      return jokerplusTicketProfit(prediction, realResult);
    }
    default: {
      // Unreachable today (calcProfit is only enabled for keno/pick3/jokerplus), but
      // kept split-aware per the main/special audit so a future caller cannot
      // reintroduce the pooled main+special count: hits are main-vs-main only.
      const { mains: realMains } = splitRealResult(realResult, game);
      const { mains: ticketMains } = splitTicket(prediction, realMains, SPECIAL_COLUMN_COUNTS[game] || 0);
      const correctNumbers = ticketMains.filter(n => realMains.includes(n)).length;
      return `${correctNumbers}/${ticketMains.length}`;
    }
  }
}

function generateList(data, title = '') {
  if(Array.isArray(data) && data.length > 0) {
    let html = '<div class="table-wrapper">';
    if (title) html += `<div style="padding: 10px; font-weight: bold; background: #f8f9fa;">${title}</div>`;
    html += '<table style="width: auto;"><tr>';
    data.forEach((item) => {
      html += `<td style="padding: 10px; background: #eee; font-size: 1.1em; font-weight: bold;">${item}</td>`;
    });
    html += '</tr></table></div>';
    return html;
  }
  return '';
}

// --- ROUTES ---

// --- LOGIC: Model performance summary (generated by Predictor.py after each
// prediction run - see Helpers.generate_model_performance_report) ---
function generatePerformanceSummary() {
  const reportPath = path.join(dataPath, 'modelPerformance.json');
  if (!fs.existsSync(reportPath)) return '';

  let report;
  try { report = JSON.parse(fs.readFileSync(reportPath, 'utf-8')); }
  catch (e) { return ''; }

  const metricLabel = { profit_per_bet: 'Profit / bet', avg_hits: 'Avg hits' };

  let rows = '';
  let anyBand = false;
  // The declaration is recorded once at report level, so the card can tell
  // "declared, no scored draw on or after the date yet" from "not declared".
  const declared = report.since || null;
  const anySince = Boolean(declared);
  Object.keys(report.games).sort().forEach((game) => {
    const info = report.games[game];
    const best = info.models[0];
    const value = best[info.metric];
    const valueColor = info.metric === 'profit_per_bet' ? (value > 0 ? '#27ae60' : '#c0392b') : '#2c3e50';
    const display = info.metric === 'profit_per_bet' ? `${value} €` : value;

    // Q0 (README "Null controls"): what the best of the rows scores on a
    // history with nothing in it. A hits-ranked row at or under that band
    // has not shown anything a leaderboard over noise would not show, and is
    // greyed like a row with too few draws. The payout games rank by profit,
    // which the control does not measure, so there the band is shown against
    // the rows' average hits but greys nothing.
    const band = controls.nullBand(controls.loadNullControls(controlsPath, game));
    if (band) anyBand = true;
    const bandGreys = Boolean(band) && info.metric === 'avg_hits';
    const bandTitle = !band ? '' : Object.keys(band.modes).map((mode) => {
      const b = band.modes[mode];
      return `${mode}: best of ${b.rows} rows ${b.best.toFixed(3)} ± ${b.sd.toFixed(3)} over ${b.seeds} histories of ${b.days} days (random ticket ${b.random})`;
    }).join(' · ') + ` · ${band.generatedAt || ''}`;
    const bandCell = !band ? '<span style="color: #aaa;" title="NullControls.py has not run for this game">not run</span>'
      : `<span title="${bandTitle}">≤ ${band.ceiling.toFixed(2)} ${band.metric.replace(' per draw', '/draw')}${bandGreys ? '' : ' <span style="color:#aaa;">(info)</span>'}</span>`;
    const inBand = (m) => bandGreys && controls.withinBand(m.avg_hits, band);

    // The forward record (since.json, src/Since.py): the same ranking over
    // the days on or after the declared date - the frozen design's own track
    // record, next to all history. Young there means young there.
    const since = info.since || null;
    const sinceByName = {};
    (since ? since.models : []).forEach((m) => { sinceByName[m.name] = m; });
    const sinceDisplay = (v) => (v === null || v === undefined ? '-' : (info.metric === 'profit_per_bet' ? `${v} €` : v));
    const sinceBest = since ? sinceByName[since.bestModel] : null;
    // The null band is not applied here: it is measured over long control
    // histories, and the best of thirty rows over a handful of draws clears
    // it by luck almost always.
    const sinceCell = since
      ? `<span title="${auth.escapeHtml(since.label)}: ${since.models.length} rows scored on ${sinceBest ? sinceBest.draws : 0} draws since ${since.date}"><b>${since.bestModel}</b> ${sinceDisplay(sinceBest ? sinceBest[info.metric] : null)} <span style="color:#7f8c8d;">(${sinceBest ? sinceBest.draws : 0} draws${sinceBest && sinceBest.draws < since.minDrawsForRanking ? ', too few to rank' : ''})</span></span>`
      : declared
        ? `<span style="color: #aaa;" title="${auth.escapeHtml(declared.label)}: no stored draw on or after ${declared.date} has been scored for this game yet">no scored draw since ${declared.date} yet</span>`
        : '<span style="color: #aaa;" title="No since.json declared: the forward record starts when you declare a date">not declared</span>';

    // Expandable full ranking per game
    const ranking = info.models.map((m, i) => {
      const v = m[info.metric];
      const mDisplay = v === null || v === undefined ? '-' : (info.metric === 'profit_per_bet' ? `${v} €` : v);
      const young = m.draws < info.minDrawsForRanking ? ' style="color: #aaa;" title="Too few scored draws to rank"'
        : (inBand(m) ? ` style="color: #999; font-style: italic;" title="Within the null band: the best of ${band.modes[band.from].rows} rows scores up to ${band.ceiling.toFixed(3)} on a ${band.from} history with nothing in it"` : '');
      const sm = sinceByName[m.name];
      const sinceYoung = sm && since && sm.draws < since.minDrawsForRanking;
      const sinceCells = !since ? '' : `<td${sinceYoung ? ' style="color:#aaa;" title="Too few scored draws since the date to rank"' : ''}>${sm ? sinceDisplay(sm[info.metric]) : '-'}</td><td${sinceYoung ? ' style="color:#aaa;"' : ''}>${sm ? sm.draws : '-'}</td>`;
      return `<tr${young}><td>${i + 1}</td><td style="text-align: left;">${m.name}</td><td>${mDisplay}</td><td>${m.avg_hits}${inBand(m) ? ' <span title="within the null band">∅</span>' : ''}</td><td>${m.best_hits}</td><td>${m.draws}</td>${sinceCells}</tr>`;
    }).join('');

    const bestNote = inBand(best) ? ' <span style="color: #999; font-weight: normal; font-size: 0.85em;" title="The best row is within the null band">∅ within band</span>' : '';

    rows += `
      <tr style="cursor: pointer;" onclick="const d = document.getElementById('rank-${game}'); d.style.display = d.style.display === 'none' ? 'table-row' : 'none';">
        <td style="font-weight: bold; text-align: left;">${game} <span style="color: #aaa; font-size: 0.85em;">▼</span></td>
        <td style="text-align: left;">${best.name}</td>
        <td>${metricLabel[info.metric] || info.metric}</td>
        <td style="font-weight: bold; color: ${valueColor};">${display}${bestNote}</td>
        <td>${best.draws}</td>
        <td>${bandCell}</td>
        <td style="text-align: left;">${sinceCell}</td>
      </tr>
      <tr id="rank-${game}" style="display: none;">
        <td colspan="7" style="padding: 0;">
          <table style="width: 100%; min-width: 0; margin: 0;">
            <tr><th>#</th><th style="text-align: left;">Model</th><th>${metricLabel[info.metric] || info.metric}</th><th>Avg hits</th><th>Best day</th><th>Scored draws</th>${since ? `<th title="${auth.escapeHtml(since.label)} - the null band is measured over long control histories and is not applied to this column">Since ${since.date}</th><th>Draws since</th>` : ''}</tr>
            ${ranking}
          </table>
        </td>
      </tr>`;
  });

  if (!rows) return '';

  return `
    <div class="card expanded" style="margin-top: 25px;">
      <div class="card-header" onclick="toggleCard(this)">
        <div>
          <span class="card-title">🏆 Best model per game</span>
          <span class="card-meta" style="margin-left: 10px;">all scored history · generated ${report.generatedAt || '?'}</span>
        </div>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body">
        <div class="table-wrapper">
          <table style="min-width: 0;">
            <tr><th style="text-align: left;">Game</th><th style="text-align: left;">Best model</th><th>Metric</th><th>Value</th><th>Scored draws</th><th title="Q0: what the best of the rows scores on a history with nothing in it">Null band</th><th style="text-align: left;" title="The forward record: the same ranking over the days since the declared date (since.json)">Best since the freeze</th></tr>
            ${rows}
          </table>
        </div>
        <p style="color: #7f8c8d; font-size: 0.85em; margin-bottom: 0;">
          Keno/Pick3/Joker+ rank by average profit per bet (real payout tables); other games by average hits of the main ticket.
          Click a game row for the full model ranking. Greyed models have fewer scored draws than the ranking minimum.
          ${anySince ? `<b>Best since the freeze</b> (${auth.escapeHtml(declared.label)}, from ${declared.date}): the same ranking counted only over the stored days on or after the declared date - the frozen design\'s own track record, every day of it predicted before its draw. Nothing is rebuilt; the date is a filter. A game reads "no scored draw since ... yet" until its first draw on or after the date is scored; rows with fewer than the ranking minimum of draws since the date are shown grey in that column until they have them. The null band is not applied to this column: it is measured over long control histories, and the best of thirty rows over a handful of draws clears it by luck almost always.` : 'The <b>Best since the freeze</b> column fills in once a date is declared in <code>since.json</code>: the same ranking counted only over the days from that date on.'}
          ${anyBand ? '<b>Null band</b> (weekly <code>NullControls.py</code>): the best of the tracked rows, run on histories with provably nothing in them - fair synthetic draws and the real draws shuffled - scores this much by selection alone (mean over the control histories plus two standard deviations). Rows marked ∅ sit within it: they have shown nothing a leaderboard over noise would not show. For the payout games the band is on average hits, not on the profit they rank by, so it is information only.' : 'The <b>null band</b> column fills in once the weekly <code>NullControls.py</code> job has run: what the best row scores on a history with nothing in it.'}
        </p>
      </div>
    </div>`;
}

// --- LOGIC: Best combination per game (README roadmap item 2, "portfolio of
// rows") - which SET of tracked rows is worth playing together, from the
// "combinations" section Helpers._build_combination_report writes into
// modelPerformance.json, with its shuffled-history control. ---
// Lotto multi-pick (README "Lotto multi-pick"): per row, the six numbers
// against the nine - average hits, the share of draws with the smallest
// prize (3 or more), and how often each extra number hit - next to the exact
// chance any set of that size has. Read from the same report; absent until a
// lotto day has been scored with extras.
function generateMultiPickSummary() {
  const reportPath = path.join(dataPath, 'modelPerformance.json');
  if (!fs.existsSync(reportPath)) return '';
  let report;
  try { report = JSON.parse(fs.readFileSync(reportPath, 'utf-8')); }
  catch (e) { return ''; }
  const esc = auth.escapeHtml;
  const pctOf = (x, digits = 1) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? '-' : `${(Number(x) * 100).toFixed(digits)}%`);
  let html = '';
  Object.keys(report.games || {}).sort().forEach((game) => {
    const info = report.games[game];
    const cfg = info.multi_pick;
    if (!cfg) return;
    const rows = (info.models || []).filter((m) => m.multi_pick && m.multi_pick.draws > 0);
    if (!rows.length) return;
    const six = cfg.chance[String(cfg.draw)] || {};
    const nine = cfg.chance[String(cfg.draw + cfg.extra)] || {};
    const labels = Array.from({ length: cfg.extra }, (_, i) => `${cfg.draw + i + 1}th`);
    const above = (rate, level) => (Number.isFinite(Number(rate)) && Number(rate) > level ? 'color:#27ae60; font-weight:bold;' : '');
    const body = rows.map((m) => {
      const mp = m.multi_pick;
      return `<tr><td style="text-align:left; font-weight:bold;">${esc(m.name)}</td><td>${mp.draws}</td>
        <td>${mp.base_avg_hits === undefined || mp.base_avg_hits === null ? '-' : mp.base_avg_hits}</td><td>${mp.avg_hits}</td>
        <td style="${above(mp.base_win_rate, six.win)}">${pctOf(mp.base_win_rate)}</td><td style="${above(mp.win_rate, nine.win)}">${pctOf(mp.win_rate)}</td>
        ${mp.extra_hit_rates.map((r) => `<td style="${above(r, cfg.per_extra_chance)}">${pctOf(r)}</td>`).join('')}</tr>`;
    }).join('');
    html += `<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">${esc(game)} multi-pick: six numbers against nine</span>
      <span class="card-meta" style="margin-left:10px;">${rows.length} rows with extras; chance of ${cfg.min_hits}+ main hits ${pctOf(six.win)} with ${cfg.draw} numbers, ${pctOf(nine.win)} with ${cfg.draw + cfg.extra}; an extra number hits ${pctOf(cfg.per_extra_chance)} by luck</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><p style="color:#7f8c8d; font-size:0.9em; margin-top:0;">Every lotto row plays six numbers and names three more - its next most probable - for a system play
      (${cfg.draw + cfg.extra} numbers = ${nine.grids} grids = ${Number(nine.price).toFixed(0)} EUR). More numbers win more by arithmetic: the chance of ${cfg.min_hits} or more of the six drawn numbers - the smallest prize
      that needs no bonus number; ranks with the bonus are not scored here - rises from ${pctOf(six.win)} to ${pctOf(nine.win)} for any ${cfg.draw + cfg.extra} numbers whatsoever. So the figures to watch are the
      model's rates <i>against those levels</i> (green when above): the share of draws with ${cfg.min_hits}+ main hits with six and with nine - both over the same draws, the ones with extras - and how often the
      7th, 8th and 9th number hit, each against ${pctOf(cfg.per_extra_chance)}. One row above a level is noise over a few draws; a row that stays above it
      for many draws is the finding, and the ground for rethinking the first six (README, roadmap item 10).</p>
      <div class="table-wrapper"><table><tr><th style="text-align:left;">Model</th><th>Draws</th><th title="average hits of the six over the draws with extras">Avg hits, 6</th><th title="average hits among all nine over the same draws">Avg hits, 9</th>
      <th title="share of draws with ${cfg.min_hits} or more main hits among the six; chance ${pctOf(six.win)}">${cfg.min_hits}+ hits, 6</th><th title="share of draws with ${cfg.min_hits} or more main hits among the nine; chance ${pctOf(nine.win)}">${cfg.min_hits}+ hits, 9</th>
      ${labels.map((l) => `<th title="how often this extra number was drawn; chance ${pctOf(cfg.per_extra_chance)}">${l} hit</th>`).join('')}</tr>${body}</table></div></div></div>`;
  });
  return html;
}

function generateCombinationSummary() {
  const reportPath = path.join(dataPath, 'modelPerformance.json');
  if (!fs.existsSync(reportPath)) return '';

  let report;
  try { report = JSON.parse(fs.readFileSync(reportPath, 'utf-8')); }
  catch (e) { return ''; }

  const metricLabel = { profit_per_draw: 'Profit / draw', avg_best_hits: 'Best-line avg hits' };
  const fmt = (v, metric) => (v === null || v === undefined) ? '-' : (metric === 'profit_per_draw' ? `${v} €` : v);
  const pct = (v) => (v === null || v === undefined) ? '-' : `${Math.round(v * 100)}%`;
  const members = (list) => list.join(' <span style="color:#aaa;">+</span> ');

  let rows = '';
  Object.keys(report.games).sort().forEach((game) => {
    const combo = report.games[game].combinations;
    if (!combo) return;
    const isProfit = combo.metric === 'profit_per_draw';
    const label = metricLabel[combo.metric] || combo.metric;

    if (!combo.best) {
      rows += `
      <tr>
        <td style="font-weight: bold; text-align: left;">${game}</td>
        <td colspan="5" style="text-align: left; color: #888;">${combo.note || 'no combination could be scored'} (${combo.candidateRows.length} established rows)</td>
      </tr>`;
      return;
    }

    const best = combo.best;
    const value = best.value;
    const valueColor = isProfit ? (value > 0 ? '#27ae60' : '#c0392b') : '#2c3e50';
    // "Beats the best single row" only means something where the metric is
    // additive (profit); for the hit games the best line of two rows is by
    // construction at least as good as either line alone.
    let single = '';
    if (combo.bestSingle) {
      const verdict = isProfit
        ? (combo.beatsBestSingle ? '<span style="color:#27ae60; font-weight:bold;">beats it</span>' : '<span style="color:#c0392b; font-weight:bold;">does not beat it</span>')
        : 'best single line';
      single = `${combo.bestSingle.name} ${fmt(combo.bestSingle.value, combo.metric)} <span style="color:#7f8c8d;">(${verdict})</span>`;
    }
    let control = `${combo.evaluated} combinations evaluated`;
    if (combo.control && combo.control.p_value !== undefined) {
      const c = combo.control;
      const weak = c.p_value > 0.05;
      control += ` · shuffled history: best-by-luck ${fmt(c.best_mean, combo.metric)} on average, ${fmt(c.best_p95, combo.metric)} at the 95th pct · `
        + `<span style="font-weight:bold; color:${weak ? '#c0392b' : '#27ae60'};" title="share of ${c.shuffles} shuffles whose best combination did at least as well">p = ${c.p_value}</span>`
        + (weak ? ' <span style="color:#7f8c8d;">(consistent with luck)</span>' : '');
    } else if (combo.control && combo.control.error) {
      control += ` · control not available (${combo.control.error})`;
    }

    const ranking = combo.ranking.map((r, i) => `
      <tr>
        <td>${i + 1}</td>
        <td style="text-align: left;">${members(r.members)}</td>
        <td>${r.how}</td>
        <td style="font-weight: bold;">${fmt(r.value, combo.metric)}</td>
        <td>${isProfit ? fmt(r.profit_per_bet, 'profit_per_draw') : r.avg_hits}</td>
        <td>${isProfit ? pct(r.win_day_rate) : r.best_hits}</td>
        <td>${r.draws}</td>
      </tr>`).join('');

    rows += `
      <tr style="cursor: pointer;" onclick="const d = document.getElementById('combo-${game}'); d.style.display = d.style.display === 'none' ? 'table-row' : 'none';">
        <td style="font-weight: bold; text-align: left;">${game} <span style="color: #aaa; font-size: 0.85em;">▼</span></td>
        <td style="text-align: left;">${members(best.members)}</td>
        <td>${label}</td>
        <td style="font-weight: bold; color: ${valueColor};">${fmt(value, combo.metric)}</td>
        <td>${best.draws}</td>
        <td style="text-align: left; font-size: 0.9em;">${single}</td>
      </tr>
      <tr id="combo-${game}" style="display: none;">
        <td colspan="6" style="padding: 0;">
          <p style="margin: 8px 12px; color: #7f8c8d; font-size: 0.85em;">${control}</p>
          <table style="width: 100%; min-width: 0; margin: 0;">
            <tr><th>#</th><th style="text-align: left;">Rows played together</th><th>Set</th><th>${label}</th><th>${isProfit ? 'Profit / bet' : 'Avg hits / line'}</th><th>${isProfit ? 'Winning draws' : 'Best day'}</th><th>Shared draws</th></tr>
            ${ranking}
          </table>
        </td>
      </tr>`;
  });

  if (!rows) return '';

  return `
    <div class="card" style="margin-top: 25px;">
      <div class="card-header" onclick="toggleCard(this)">
        <div>
          <span class="card-title">🧩 Best combination per game</span>
          <span class="card-meta" style="margin-left: 10px;">which rows to play together · all scored history</span>
        </div>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body">
        <div class="table-wrapper">
          <table style="min-width: 0;">
            <tr><th style="text-align: left;">Game</th><th style="text-align: left;">Rows played together</th><th>Metric</th><th>Value</th><th>Shared draws</th><th style="text-align: left;">Best single row on the same draws</th></tr>
            ${rows}
          </table>
        </div>
        <p style="color: #7f8c8d; font-size: 0.85em; margin-bottom: 0;">
          Every pair and triple of the ranked rows is scored on the draws all of its members were scored on.
          Keno/Pick3/Joker+ rank by <b>net profit per draw</b> of playing every member's tickets (profit adds up, so a set
          only beats its best row when at least two rows are positive on those draws; a greedy build-up beyond the best
          triple keeps adding rows while that improves). The other games rank by the <b>best line held per draw</b> -
          the average over draws of the best-scoring member's hits, which rewards rows whose good days do not coincide;
          it grows with every extra line, so pairs and triples are ranked separately. Click a game for the ranking and
          the control: the same search is re-run on shuffled history (each ticket keeps its day, the drawn result it is
          scored against is moved to another day). With hundreds of combinations a "best" one always exists by luck;
          <b>p</b> is the share of shuffles whose best combination did at least as well - read a combination as an edge
          only when p is small and stays small as draws accumulate.
        </p>
      </div>
    </div>`;
}

// --- LOGIC: Phase-shift (lag) analysis card - each predictor run scores its
// newPrediction against draws +1..+30 and keeps only the best peak of that
// run. The table shows this run's peak plus how often each lag has peaked
// across the persisted run history: a lag that keeps winning (e.g. pick3
// around +30 run after run) is evidence of a real shift, a peak that wanders
// every run is noise. ---
function generateLagAnalysis() {
  const reportPath = path.join(dataPath, 'modelPerformance.json');
  if (!fs.existsSync(reportPath)) return '';

  let report;
  try { report = JSON.parse(fs.readFileSync(reportPath, 'utf-8')); }
  catch (e) { return ''; }

  let gameCards = '';
  Object.keys(report.games).sort().forEach((game) => {
    const la = report.games[game].lagAnalysis;
    if (!la || Object.keys(la).length === 0) return;

    const modelRows = Object.keys(la).sort().map((name) => {
      const row = la[name];
      const peak = row.peak || {};
      const runs = row.runs || 0;
      const share = runs ? row.consensus_runs / runs : 0;
      // Only call a lag "tracked" once several runs agree on it - with one or
      // two runs the winning lag is whatever noise picked.
      const tracked = runs >= 3 && share >= 0.5;
      // Chronological peak trail (oldest → newest), run-length encoded so a
      // stable peak reads as +30×5 while a drift reads as +26 → +28 → +30.
      // The raw tally (lag_counts) can't tell those two apart.
      const segments = [];
      (row.history || []).forEach((r) => {
        const last = segments[segments.length - 1];
        if (last && last.lag === r.lag) { last.count += 1; last.to = r; }
        else segments.push({ lag: r.lag, count: 1, from: r, to: r });
      });
      const shown = segments.slice(-8);
      const trail = (segments.length > shown.length ? '… → ' : '') + shown.map((seg, i) => {
        const isNewest = i === shown.length - 1;
        const when = seg.count === 1 ? seg.from.run : `${seg.from.run} … ${seg.to.run}`;
        const tip = `${when} · avg hits ${seg.to.avg_hits} · z ${seg.to.z}`;
        const label = seg.count === 1 ? `+${seg.lag}` : `+${seg.lag}×${seg.count}`;
        return `<span title="${tip}" style="${isNewest ? 'font-weight:bold; color:#2c3e50;' : ''}">${label}</span>`;
      }).join(' → ');
      const z = peak.z === null || peak.z === undefined ? '-' : peak.z;
      return `<tr>
        <td style="text-align:left; font-weight:bold;">${name}</td>
        <td style="font-weight:bold;">+${peak.lag}</td>
        <td title="profile mean ${peak.profile_mean}">${peak.avg_hits}</td>
        <td>${z}</td>
        <td>${peak.n}</td>
        <td style="${tracked ? 'background:#2ecc71; color:white; font-weight:bold;' : ''}">+${row.consensus_lag} (${row.consensus_runs}/${runs})</td>
        <td style="text-align:left; color:#7f8c8d; white-space:nowrap;">${trail}</td>
      </tr>`;
    }).join('');
    if (!modelRows) return;

    gameCards += `
      <div class="card">
        <div class="card-header" onclick="toggleCard(this)">
          <span class="card-title" style="font-size: 1em;">${game}</span>
          <div class="card-icon">▼</div>
        </div>
        <div class="card-body">
          <div class="table-wrapper">
            <table>
              <tr>
                <th style="text-align:left;">Model</th>
                <th>Peak lag (this run)</th>
                <th>Avg hits</th>
                <th>z</th>
                <th>n</th>
                <th>Most frequent peak</th>
                <th style="text-align:left;">Peak trail (oldest → newest)</th>
              </tr>
              ${modelRows}
            </table>
          </div>
        </div>
      </div>`;
  });

  if (!gameCards) return '';

  return `
    <div class="card" style="margin-top: 25px;">
      <div class="card-header" onclick="toggleCard(this)">
        <div>
          <span class="card-title">📈 Phase-shift check (tracked peaks)</span>
          <span class="card-meta" style="margin-left: 10px;">best lag per predictor run, tracked across runs</span>
        </div>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body">
        <p style="color: #7f8c8d; font-size: 0.85em; margin-top: 0;">
          Each run scores every prediction against draws +1 .. +30 and keeps only its best lag (+1 is the draw the
          prediction was made for). <b>z</b> is how far that peak sticks out of the model's own lag profile - near 1
          means a flat profile, so the hits come from number-frequency structure rather than timing. The highlighted
          column is the lag that peaked in most runs: several runs agreeing on the same lag is the evidence a real
          phase shift exists; a peak that moves every run is noise. The trail shows the peaks in run order (hover a
          step for date and stats): a repeated <b>+30×5</b> means the peak is holding still, <b>+26 → +28 → +30</b>
          means it is drifting. Pick3 is scored positionally (digit in the right place), Joker+ as its left + right
          positional runs. One peak is recorded per run
          date, keeping the last 60 runs.
        </p>
        ${gameCards}
      </div>
    </div>`;
}

// --- LOGIC: Meta-learner feature control card (README "Null controls") -----
// Per game and served meta-learner variant: did the model give provably
// irrelevant columns stable importance, what did they cost it held-out, and
// which base models matter more than noise does? Read from the weekly
// IrrelevantFeatureControl.py records; the card is absent until one exists.
function generateFeatureControl() {
  const reportPath = path.join(dataPath, 'modelPerformance.json');
  if (!fs.existsSync(reportPath)) return '';
  let report;
  try { report = JSON.parse(fs.readFileSync(reportPath, 'utf-8')); }
  catch (e) { return ''; }

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (x, digits) => (x === null || x === undefined || !Number.isFinite(Number(x))) ? '-' : Number(x).toFixed(digits);
  let rows = '';
  let anyFits = false;
  let newest = '';
  Object.keys(report.games).sort().forEach((game) => {
    const record = controls.loadFeatureControl(controlsPath, game);
    if (!record) return;
    const d = controls.describeFeatureControl(record);
    if (d.anyFitsNoise) anyFits = true;
    if (d.generatedAt && String(d.generatedAt) > newest) newest = String(d.generatedAt);
    const gameCell = `<td rowspan="${d.variants.length || 1}" style="text-align:left; font-weight:bold; vertical-align:top;">${esc(game)}<br>`
      + `<small style="color:#7f8c8d; font-weight:normal;">${d.tableDays === null ? '?' : d.tableDays} table days${d.positional ? ' (positional)' : ''}`
      + `${d.lockboxDays ? `, ${d.lockboxDays} lockbox day(s) withheld` : ''}${d.behind ? `, ${d.behind} draw(s) behind the file` : ''} · ${d.repeats === null ? '?' : d.repeats} × ${d.noiseColumns === null ? '?' : d.noiseColumns} noise columns</small></td>`;
    if (!d.variants.length) {
      rows += `<tr>${gameCell}<td colspan="5" style="color:#aaa;">no variant was scored</td></tr>`;
      return;
    }
    d.variants.forEach((v, i) => {
      const verdict = v.error
        ? `<span style="color:#aaa;" title="${esc(v.error)}">failed</span>`
        : `<span style="${v.fitsNoise ? 'color:#e74c3c; font-weight:bold;' : 'color:#27ae60;'}" title="${esc(v.rule)}">${v.verdict}${v.fitsNoise ? ' ⚠' : ''}</span>`;
      rows += `<tr>
        ${i === 0 ? gameCell : ''}
        <td style="text-align:left;">${esc(v.label)}</td>
        <td>${verdict}</td>
        <td title="share of the model's positive training-side permutation importance that lands on the noise columns; a noise column matters ${fmt(v.noiseRatio, 2)}x an average real column (t ${fmt(v.noiseT, 1)})">${v.noiseShare === null ? '-' : fmt(v.noiseShare * 100, 1) + '%'}</td>
        <td title="held-out AUC of the fit without noise → mean over the fits with noise">${fmt(v.heldoutWithout, 4)} → ${fmt(v.heldoutWith, 4)}</td>
        <td style="text-align:left;" title="base models whose held-out importance exceeds the noise columns' mean + 2 sd (band ${fmt(v.band, 4)})">${v.above.length ? v.above.map(esc).join(', ') : '<span style="color:#aaa;">none</span>'}</td>
      </tr>`;
    });
  });
  if (!rows) return '';

  return `
    <div class="card" style="margin-top: 25px;">
      <div class="card-header" onclick="toggleCard(this)">
        <div>
          <span class="card-title">🧪 Meta-learner feature control (irrelevant column)</span>
          <span class="card-meta" style="margin-left: 10px;">does any meta-learner give a random column stable importance?${anyFits ? ' <b style="color:#e74c3c;">yes ⚠</b>' : ''}</span>
        </div>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body">
        <div class="table-wrapper">
          <table>
            <tr><th style="text-align:left;">Game</th><th style="text-align:left;">Meta-learner</th><th>Noise</th><th title="how much of the model's attribution went to the noise columns">Noise share</th><th title="held-out AUC without → with the noise columns">Held-out AUC</th><th style="text-align:left;" title="base models whose held-out importance clears the noise columns' band">Base models above the noise band</th></tr>
            ${rows}
          </table>
        </div>
        <p style="color: #7f8c8d; font-size: 0.85em;">
          Weekly <code>IrrelevantFeatureControl.py</code> (README "Null controls"): every meta-learner variant is refitted on its
          own training table with three <b>noise columns</b> appended - shuffled copies of real base-model scores, so they look
          exactly like a base model and mean nothing. <b>Noise share</b> is how much of the model's attribution (permutation
          importance on the training rows) lands on them: a model that cannot tell noise from signal gives three noise columns
          among eleven about 27%, a model that ignores them 0%. <b>Fits noise</b> (⚠) means a noise column matters in training
          at least a tenth as much as an average real column, consistently (pooled importance more than two standard errors
          above zero) - the model's weights on the real columns are then not evidence of anything either.
          <b>Held-out AUC</b> without → with noise is what the extra columns cost. The <b>base models above the noise band</b>
          are the columns whose held-out importance exceeds what the noise columns score (their mean + 2 sd): the ones that carry
          more than noise does. Importance is the ranking power lost when a column is scrambled (the drop in |AUC − 0.5|), so a
          model whose probabilities came out inverted still shows what it leans on. With eight base models tested against a
          two-sigma band, one clears it by chance about one week in five: a column counts when it stays above the band week after
          week, not when it appears once. The lockbox days leave the table first, exactly as in the trainer.
          ${newest ? `Newest record ${esc(newest.slice(0, 16).replace('T', ' '))}.` : ''}
        </p>
      </div>
    </div>`;
}

// --- LOGIC: Randomness watch card (README "Entropy & Divergence Analysis") -
// per game: KL(recent 60 draws || full history) for drift, KL(recent ||
// uniform) + normalized entropy for distance from a fair draw, a trend over
// checkpoint windows, and per-model KL(predicted || real) to expose models
// whose output distribution has departed from the actual process. ---
function generateRandomnessWatch() {
  const reportPath = path.join(dataPath, 'modelPerformance.json');
  if (!fs.existsSync(reportPath)) return '';

  let report;
  try { report = JSON.parse(fs.readFileSync(reportPath, 'utf-8')); }
  catch (e) { return ''; }

  let gameRows = '';
  let modelCards = '';
  let anyQ2 = false;
  Object.keys(report.games).sort().forEach((game) => {
    const rw = report.games[game].randomnessWatch;
    const aw = report.games[game].anomalyWatch;
    if (!rw && !aw) return;

    // Entropy meaningfully below 1 or KL drifting up is what the README's
    // security layer watches for. Thresholds are deliberately loose - this
    // is a tripwire, not a verdict.
    const entAlert = rw && rw.entropy_norm !== null && rw.entropy_norm < 0.95;
    const klAlert = rw && rw.kl_vs_history !== null && rw.kl_vs_history > 0.1;
    const aeAlert = aw && aw.alert;
    // Q2 (README "Null controls"): the controlled test of the same question
    // the tripwires above ask - a classifier suite against a null band and a
    // shuffled control, weekly. This is the verdict; the rest is the alarm.
    const q2Record = controls.loadDiscrimination(controlsPath, game);
    const q2 = q2Record ? controls.describeDiscrimination(q2Record) : null;
    if (q2) anyQ2 = true;
    const q2Alert = Boolean(q2 && q2.evidence);
    const q2Cell = !q2 ? '<span style="color:#aaa;" title="RandomnessDiscrimination.py has not run for this game">not run</span>'
      : `<span title="best AUC of the suite, mean over ${q2.repetitions} repetitions: real ${q2.real} ± ${q2.realSd}, null ${q2.nullMean} ± ${q2.nullSd} (band ≤ ${q2.threshold}), shuffled ${q2.shuffled} · ${q2.window}-draw windows over the newest ${q2.draws || 'all'} draws${q2.quantum ? ', quantum suite included' : ''} · ${q2.generatedAt || ''}" style="${q2.evidence ? 'color:#e74c3c; font-weight:bold;' : ''}">${q2.label}${q2.evidence ? ' ⚠' : ''}<br><small style="color:#7f8c8d;">AUC ${q2.real} vs band ≤ ${q2.threshold}</small></span>`;
    const status = (entAlert || klAlert || aeAlert || q2Alert)
      ? '<span style="background:#e67e22; color:white; padding:2px 8px; border-radius:3px; font-weight:bold;">watch</span>'
      : '<span style="background:#2ecc71; color:white; padding:2px 8px; border-radius:3px;">normal</span>';

    // Autoencoder predictability watch: strongly NEGATIVE z = the real
    // draw suddenly became easy to reconstruct = non-random structure.
    const anomaly = !aw ? '-' :
      `<span title="latest run ${aw.date} · latest z ${aw.latest_z}" style="${aw.alert ? 'color:#e74c3c; font-weight:bold;' : ''}">${aw.min_z_recent === null || aw.min_z_recent === undefined ? '-' : 'min z ' + aw.min_z_recent}${aw.alert ? ' ⚠' : ''}</span>`;
    const trend = ((rw && rw.trend) || []).map((t) =>
      `<span title="window ending ${t.end_date}: KL vs history ${t.kl_vs_history}, entropy ${t.entropy_norm}">${t.entropy_norm}</span>`
    ).join(' → ');

    gameRows += `<tr>
      <td style="text-align:left; font-weight:bold;">${game}</td>
      <td>${status}</td>
      <td>${!rw || rw.entropy_norm === null ? '-' : rw.entropy_norm}</td>
      <td>${!rw || rw.kl_vs_history === null ? '-' : rw.kl_vs_history}</td>
      <td>${!rw || rw.kl_vs_uniform === null ? '-' : rw.kl_vs_uniform}</td>
      <td>${anomaly}</td>
      <td>${q2Cell}</td>
      <td>${rw ? rw.draws_total : '-'}</td>
      <td style="text-align:left; color:#7f8c8d; font-size:0.85em;">${trend}</td>
    </tr>`;

    const models = (rw && rw.model_kl_vs_real) || {};
    const modelRows = Object.keys(models).sort((a, b) => models[a] - models[b]).map((m) =>
      `<tr><td style="text-align:left;">${m}</td><td>${models[m]}</td></tr>`
    ).join('');
    if (modelRows) {
      modelCards += `
        <div class="card">
          <div class="card-header" onclick="toggleCard(this)">
            <span class="card-title" style="font-size: 1em;">${game} - model KL(predicted || real)</span>
            <div class="card-icon">▼</div>
          </div>
          <div class="card-body">
            <div class="table-wrapper">
              <table style="min-width: 0;">
                <tr><th style="text-align:left;">Model</th><th>KL over last ${rw ? rw.window : '?'} draws</th></tr>
                ${modelRows}
              </table>
            </div>
          </div>
        </div>`;
    }
  });

  if (!gameRows) return '';

  return `
    <div class="card" style="margin-top: 25px;">
      <div class="card-header" onclick="toggleCard(this)">
        <div>
          <span class="card-title">🔬 Randomness watch (entropy & divergence)</span>
          <span class="card-meta" style="margin-left: 10px;">is the drawing process still indistinguishable from fair?</span>
        </div>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body">
        <div class="table-wrapper">
          <table>
            <tr><th style="text-align:left;">Game</th><th>Status</th><th>Entropy (norm.)</th><th>KL vs history</th><th>KL vs uniform</th><th>AE anomaly</th><th title="Q2: real draw windows against fair simulated ones, held to the null band of the same pipeline">Controlled test (Q2)</th><th>Draws</th><th style="text-align:left;">Entropy trend (oldest → newest)</th></tr>
            ${gameRows}
          </table>
        </div>
        <p style="color: #7f8c8d; font-size: 0.85em;">
          Computed over the last 60 scored draws (pick3 and Joker+ per digit position, averaged). Normalized entropy near 1 and
          KL near 0 mean the process looks fair and stationary; a sustained entropy drop or KL rise is a
          predictability signal worth investigating - <b>not</b> proof of manipulation (rule changes, data artifacts
          and small windows all move these numbers). Per-model KL shows how far each model's recent predictions sit
          from the real draw distribution. <b>AE anomaly</b> is the autoencoder security layer: the most negative
          rolling z of its reconstruction NLL over the last 30 real draws - a strongly negative value (⚠ below -3)
          means real draws suddenly became easy to reconstruct, i.e. a predictability spike.
          ${anyQ2 ? '<b>Controlled test</b> (weekly <code>RandomnessDiscrimination.py</code>, README "Null controls"): the verdict the tripwires cannot give. A suite of classifiers - logistic, SVM, random forest, gradient boosting, a small network, a quantum kernel and a VQC - is asked to tell windows of real draws from fair simulated ones, and its best AUC is held against the same suite\'s best on two fair histories (the null band, mean + 2 sd) and on the real draws shuffled (which keeps every frequency and kills only the order). Only above the band <i>and</i> above the shuffled control does the real process differ from a fair one in a way that uses time; and even then the README lists the boring causes to rule out first.' : 'The <b>controlled test</b> column fills in once the weekly <code>RandomnessDiscrimination.py</code> job has run.'}
        </p>
        ${modelCards}
      </div>
    </div>`;
}

// 1. Database Index
app.get('/database', (req, res) => {
  const folders = lotteryFolders();
  const marketFolders = databaseFolders().filter(isMarketFolder);
  let html = generateHeader("Lottery games - History", req.user);
  html += '<h1>History</h1>' + lotteryNav('history');
  html += `<p style="color: #7f8c8d; margin-top: -8px;">Every past draw of every lottery game with what each model predicted and what it hit; below, the cards that rank the
    rows and watch the draws (they include the two markets, which are tracked as games too).</p>`;
  html += '<div style="display: flex; gap: 10px; flex-wrap: wrap;">';
  folders.forEach((folder) => {
    html += `<form action="/database/${folder}" method="get">
      <button type="submit" style="padding: 15px 30px; font-size: 1.1em; cursor: pointer; background: white; border: 1px solid #ccc; border-radius: 5px;">${folder}</button>
    </form>`;
  });
  html += '</div>';
  if (marketFolders.length) {
    html += `<p style="color: #7f8c8d; font-size: 0.9em;">The markets' game view, the same digits scored the same way: ${marketFolders.map((f) => `<a href="/database/${f}">${f}</a>`).join(' · ')}
      - their own pages are <a href="/markets/crypto">Crypto</a> and <a href="/markets/shares">Shares</a>.</p>`;
  }
  html += generatePerformanceSummary();
  html += generateMultiPickSummary();
  html += generateCombinationSummary();
  html += generateLagAnalysis();
  html += generateRandomnessWatch();
  html += generateFeatureControl();
  html += generateFooter();
  res.send(html);
});

// 2. Folder View
app.get('/database/:folder', (req, res) => {
  const folder = req.params.folder;
  if (!isPlainName(folder)) return res.status(404).send('Folder not found');
  const folderPath = path.join(dataPath, folder);
  if (!fs.existsSync(folderPath)) return res.status(404).send('Folder not found');
  const game = gameFromFolder(folder);
  const calcProfit = game === "keno" || game === "pick3" || game === "jokerplus";
  // Joker+ has a payout table AND a positional notation worth seeing at a
  // glance, so its history shows both the profit and the best 'L/R (Z)'.
  const isJoker = game === 'jokerplus';
  const specialCount = SPECIAL_COLUMN_COUNTS[game] || 0;

  const files = fs.readdirSync(folderPath).filter((file) => file.endsWith('.json'));
  const filesByMonth = files.reduce((acc, file) => {
    const date = new Date(file.replace('.json', ''));
    if(!isNaN(date)) {
        const monthYear = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        if (!acc[monthYear]) acc[monthYear] = [];
        acc[monthYear].push(file);
    }
    return acc;
  }, {});

  const sortedMonths = Object.keys(filesByMonth).sort((a, b) => new Date(b) - new Date(a));

  let html = generateHeader(`${auth.escapeHtml(folder)} Predictions`, req.user);
  html += `<h1>${auth.escapeHtml(folder)}</h1>`;
  html += isMarketFolder(folder) ? marketGameViewNote(game) : lotteryNav('history');
  html += '<div>';

  sortedMonths.forEach((month, index) => {
    filesByMonth[month].sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));
    // For Joker+ 'mains' holds L+R and 'specials' the sign hit, so the same
    // (mains, then specials) comparison ranks by (L+R, Z); left/right keep
    // the two runs apart for the 'L/R (Z)' display.
    let monthProfit = 0; let monthBest = { mains: 0, specials: 0, left: 0, right: 0 };

    const fileListHtml = filesByMonth[month].map(file => {
        const filePath = path.join(folderPath, file);
        const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        let fileProfit = 0; let fileBest = { mains: 0, specials: 0, left: 0, right: 0 };
        let fileBestMulti = 0; let anyMulti = false;      // lotto multi-pick: the best hits among nine that day
        const validPredictions = jsonData.currentPrediction || [];

        if (validPredictions && validPredictions.length > 0) {
            if(calcProfit) {
                fileProfit = validPredictions.reduce((acc, predObj) => {
                    let pProfit = 0;
                    predObj.predictions.forEach(p => pProfit += calculateProfit(p, jsonData.realResult, game, predObj.name));
                    return acc + pProfit;
                }, 0);
            }
            if (!calcProfit || isJoker) {
                // Best row is recomputed from the predictions (main hits vs
                // real mains only, then special hits as tie-break) instead of
                // trusting jsonData.matchingNumbers: old day JSONs still carry
                // the pooled main+special shape in that field, newer ones the
                // split one, and recomputing renders both vintages the same
                // way.
                const { mains: realMains, specials: realSpecials, bonus: realBonus } = splitRealResult(jsonData.realResult, game);
                validPredictions.forEach(predObj => {
                    if (multiPickFor(game) && Array.isArray(predObj.multiPick) && predObj.predictions[0]) {
                        const { mains: mainsOnly } = splitTicket(predObj.predictions[0], realMains, specialCount);
                        fileBestMulti = Math.max(fileBestMulti, multiHits(mainsOnly, predObj.multiPick, realMains));
                        anyMulti = true;
                    }
                    predObj.predictions.forEach(p => {
                        const { mains: ticketMains, specials: ticketSpecials } = splitTicket(p, realMains, specialCount);
                        let candidate;
                        if (isJoker) {
                            // Positional runs, sign as tie-break - the same
                            // ranking Helpers.find_best_matching_prediction uses.
                            const runs = jokerplusRuns(ticketMains, realMains);
                            candidate = { mains: runs.left + runs.right, specials: jokerplusSignHit(ticketSpecials, realSpecials), left: runs.left, right: runs.right };
                        } else {
                            // a market game's hit is the bin in its own slot
                            const mainHits = isMarketGame(game)
                              ? ticketMains.filter((n, i) => realMains[i] === n).length
                              : ticketMains.filter(n => realMains.includes(n)).length;
                            // Lotto: the bonus supplements the tier ("5 (1)"),
                            // matched against the played numbers themselves.
                            const specialHits = ticketSpecials.filter(n => realSpecials.includes(n)).length
                              + ticketMains.filter(n => realBonus.includes(n)).length;
                            candidate = { mains: mainHits, specials: specialHits, left: 0, right: 0 };
                        }
                        if (candidate.mains > fileBest.mains || (candidate.mains === fileBest.mains && candidate.specials > fileBest.specials)) {
                            fileBest = candidate;
                        }
                    });
                });
            }
        }
        monthProfit += fileProfit;
        if (fileBest.mains > monthBest.mains || (fileBest.mains === monthBest.mains && fileBest.specials > monthBest.specials)) {
            monthBest = fileBest;
        }
        const color = fileProfit > 0 ? 'green' : (fileProfit < 0 ? 'red' : 'orange');
        // "Match: 3 (1)" = 3 main hits (1 special hit) for games that draw
        // special numbers; other games just show the main count. Joker+
        // reads "Match: L/R (Z)" next to its profit.
        const showSupplement = specialCount > 0 || game === 'lotto';
        const matchStat = isJoker
          ? `Match: ${fileBest.left}/${fileBest.right} (${fileBest.specials})`
          : `Match: ${fileBest.mains}${showSupplement ? ` (${fileBest.specials})` : ''}`;
        const displayStat = (calcProfit
          ? (isJoker ? `${matchStat} · ${formatProfit(fileProfit, game)} €` : `${fileProfit} €`)
          : matchStat) + (anyMulti ? ` <span style="color:#7d3c0f; font-weight:normal;" title="best hits among a row's nine numbers (multi-pick)">· of 9: ${fileBestMulti}</span>` : '');

        return `<li style="padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between;">
            <a href="/database/${folder}/${file}" style="text-decoration: none; color: #333;">📄 ${file}</a>
            <span style="font-weight: bold; color: ${color};">${displayStat}</span>
        </li>`;
    }).join('');

    const monthColor = monthProfit > 0 ? '#27ae60' : (monthProfit < 0 ? '#c0392b' : '#7f8c8d');
    const bestStat = isJoker
      ? `Best Match: ${monthBest.left}/${monthBest.right} (${monthBest.specials})`
      : `Best Match: ${monthBest.mains}${(specialCount > 0 || game === 'lotto') ? ` (${monthBest.specials})` : ''}`;
    const headerStat = calcProfit ? (isJoker ? `Total: ${formatProfit(monthProfit, game)} € · ${bestStat}` : `Total: ${monthProfit} €`) : bestStat;
    // Only expand if it is the first month (index === 0)
    const isExpanded = index === 0 ? 'expanded' : '';

    html += `
    <div class="card ${isExpanded}">
        <div class="card-header" onclick="toggleCard(this)">
            <div><span class="card-title">${month}</span><span style="margin-left: 10px; font-size: 0.9em; background: ${monthColor}; color: white; padding: 2px 8px; border-radius: 4px;">${headerStat}</span></div>
            <div class="card-icon">▼</div>
        </div>
        <div class="card-body">
            <ul style="list-style: none; padding: 0; margin: 0;">${fileListHtml}</ul>
        </div>
    </div>`;
  });

  html += '</div>';
  html += generateFooter();
  res.send(html);
});

// 3. File Detail View
app.get('/database/:folder/:file', (req, res) => {
  const folder = req.params.folder;
  const file = req.params.file;
  if (!isPlainName(folder) || !isPlainName(file)) return res.status(404).send('File not found');
  const filePath = path.join(dataPath, folder, file);
  if (!fs.existsSync(filePath)) return res.status(404).send('File not found');
  const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  const game = gameFromFolder(folder);
  const calculateProfitFlag = game === "keno" || game === "pick3" || game === "jokerplus";
  const specialCount = SPECIAL_COLUMN_COUNTS[game] || 0;
  // Mains of the drawn row, used for the frequency-chart bar coloring: the
  // charted frequencies are main-number frequencies, so a bar must not turn
  // green just because its number came out as a star/dream/viking/bonus. The
  // array is interpolated into the chart script server-side - the browser has
  // no jsonData object (the previous client-side jsonData.realResult lookup
  // threw a ReferenceError and the analysis chart never rendered).
  const { mains: realMains } = splitRealResult(jsonData.realResult, game);
  // Joker+ charts are restricted to the digit range (see chartFrequency);
  // other games get the same object back.
  const currentFrequency = chartFrequency(jsonData.currentNumberFrequency, game);
  const nextFrequency = chartFrequency(jsonData.numberFrequency, game);
  // The draw this day's "Next Draw Prediction" was made for: the next one
  // after this file, on the game's own schedule (see drawDateMeta).
  const nextDrawMeta = drawDateMeta(
    fs.readdirSync(path.join(dataPath, folder)).filter((f) => f.endsWith('.json')).map(parseDayFileDate),
    parseDayFileDate(file), false);

  let html = generateHeader(`${auth.escapeHtml(file)} Details`, req.user);
  html += `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0;">${auth.escapeHtml(file)}</h1>
        <a href="/database/${encodeURIComponent(folder)}" class="nav-btn" style="text-decoration: none;">Back to ${isMarketFolder(folder) ? `${auth.escapeHtml(folder)} game view` : 'History'}</a>
    </div>
    ${isMarketFolder(folder) ? marketGameViewNote(game) : ''}

    <div class="card expanded">
        <div class="card-header" onclick="toggleCard(this)">
            <span class="card-title">Real Result</span><div class="card-icon">▼</div>
        </div>
        <div class="card-body">${generateList(displayRow(jsonData.realResult, game))}</div>
    </div>

    <div class="card expanded">
        <div class="card-header" onclick="toggleCard(this)">
             <span class="card-title">Analysis of Prediction</span><div class="card-icon">▼</div>
        </div>
        <div class="card-body">
            ${generateTable(jsonData.currentPrediction, '', jsonData.realResult, calculateProfitFlag, game)}
            ${game === 'jokerplus'
              ? `<p style="color: #7f8c8d; font-size: 0.85em; margin: 10px 0 0;">Hits are shown as <b>L/R (Z)</b>: L = leading digits matching consecutively from the left (green cells), R = trailing digits matching consecutively from the right (blue cells); a full match reads 6/0. Z = 1 (amber cell) when the zodiac sign matches. Joker+ pays per run length from either end, so a right digit in the wrong position is worth nothing. Only the sign is player-selectable - the six digits are system-generated.</p>`
              : (specialCount > 0
              ? `<p style="color: #7f8c8d; font-size: 0.85em; margin: 10px 0 0;">Hits are shown as <b>N (M)</b>: N hits among the main numbers, M among the special numbers (euromillions stars / eurodreams dream / vikinglotto viking). Cells highlight green only within their own group.</p>`
              : (game === 'lotto'
                ? `<p style="color: #7f8c8d; font-size: 0.85em; margin: 10px 0 0;">Hits are shown as <b>N (M)</b>: N among the 6 drawn mains, M = 1 (amber cell) when a played number matches the bonus ball - "5 (1)" is a high tier, "6 (0)" the jackpot; a full main match makes a bonus match impossible.</p>`
                : (isMarketFolder(folder)
                  ? `<p style="color: #7f8c8d; font-size: 0.85em; margin: 10px 0 0;">Hits are the predicted bin equal to the actual bin <b>in the same slot</b> (green cells) - chance is 1 in 10 per slot. ${game === marketBase(game)
                    ? `The <a href="/markets/${game}">${game} page</a> shows the newest 30 settled days as returns, every model's money and the next day's calls as prices.`
                    : `This is the ${marketBase(game)} market's week game - one draw per week; the <a href="/markets/${marketBase(game)}">${marketBase(game)} page</a> shows the coming week's calls in its Week ahead card.`}</p>`
                  : '')))}

            ${currentFrequency && Object.keys(currentFrequency).length > 0 ? `
                <div style="margin-top: 20px; height: 200px; width: 100%;">
                    <canvas id="chart-analysis"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('chart-analysis').getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: ${JSON.stringify(Object.keys(currentFrequency))},
                        datasets: [{
                            label: 'Freq',
                            data: ${JSON.stringify(Object.values(currentFrequency))},
                            backgroundColor: ${JSON.stringify(Object.keys(currentFrequency))}.map(n => ${JSON.stringify(realMains)}.includes(Number(n)) ? 'rgba(46, 204, 113, 0.8)' : 'rgba(52, 152, 219, 0.6)')
                        }]
                    },
                    options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
                    });
                </script>
            ` : ''}
        </div>
    </div>

    <div class="card expanded">
        <div class="card-header" onclick="toggleCard(this)">
             <div><span class="card-title">Next Draw Prediction</span>${nextDrawMeta}</div><div class="card-icon">▼</div>
        </div>
        <div class="card-body">
            ${generateTable(jsonData.newPrediction, '', [], false, game, nextTableOptions(game, req.user))}

            ${nextFrequency ? `
                <div style="margin-top: 20px; height: 200px; width: 100%;">
                    <canvas id="chart-detail"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('chart-detail').getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: ${JSON.stringify(Object.keys(nextFrequency))},
                        datasets: [{ label: 'Freq', data: ${JSON.stringify(Object.values(nextFrequency))}, backgroundColor: 'rgba(52, 152, 219, 0.6)' }]
                    },
                    options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
                    });
                </script>
            ` : ''}
        </div>
    </div>
  `;
  html += generateFooter();
  res.send(html);
});

// 4. Home: the three sections
function sectionCard(title, text, lines, links) {
  return `<div class="card" style="display:flex; flex-direction:column;">
    <div style="padding: 18px 20px 6px;"><div class="card-title">${title}</div><p style="color:#555; margin:8px 0 0;">${text}</p></div>
    <div style="padding: 6px 20px 10px; flex:1;">${lines.length ? `<ul style="margin:0; padding-left:18px; color:#2c3e50; font-size:0.95em; line-height:1.6;">${lines.map((l) => `<li>${l}</li>`).join('')}</ul>` : ''}</div>
    <div style="padding: 0 20px 18px; display:flex; gap:8px; flex-wrap:wrap;">${links.map((l, i) => `<a href="${l.href}" class="nav-btn" style="text-decoration:none; ${i ? 'background:#7f8c8d;' : ''}">${l.label}</a>`).join('')}</div>
  </div>`;
}

function lotterySectionLines() {
  const lines = [];
  lotteryFolders().forEach((folder) => {
    const files = dayFiles(folder);
    if (!files.length) return;
    const next = nextDrawDate(files.map(parseDayFileDate), null);
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const when = !next ? '' : (next.getTime() === today.getTime() ? 'today' : (next < today ? `${formatDrawDate(next)} (no newer run yet)` : formatDrawDate(next)));
    let rows = 0;
    try { rows = (JSON.parse(fs.readFileSync(path.join(dataPath, folder, files[0]), 'utf-8')).newPrediction || []).length; } catch (e) { rows = 0; }
    lines.push(`<b>${folder}</b>${when ? ` - next draw ${when}` : ''}${rows ? `, ${rows} model rows` : ''}`);
  });
  return lines;
}

function marketSectionLines(game) {
  const record = markets.loadMarket(marketsPath, game);
  if (!record) return ['no record yet - it appears after the first daily run that includes this market'];
  const view = markets.describeMarket(record, { regimes: markets.loadRegimes(marketsPath, game) });
  const lines = [`<b>${view.instruments.map((i) => i.symbol).join(', ')}</b>`];
  if (view.newestGameDay) lines.push(`newest settled day ${view.newestGameDay}${view.madeOn ? `, predictions for the day after ${view.madeOn}` : ''}`);
  if (view.models.length) {
    const best = view.models[0];
    lines.push(`${view.models.length} model rows over ${view.scoredDays} day(s); best exact rate ${markets.pct(best.exact)} (${best.name}), chance ${markets.pct(view.chance.exact, 0)}`);
    const richest = view.models.filter((m) => m.pnlCash !== null).sort((a, b) => b.pnlCash - a.pnlCash)[0];
    const marketBook = view.trading && view.trading.benchmark.length ? view.trading.benchmark[view.trading.benchmark.length - 1].total : null;
    if (richest && view.trading) {
      lines.push(`best paper book ${markets.money(richest.pnlCash, 2)} ${view.trading.currency} (${richest.name})${marketBook === null ? '' : `, the market ${markets.money(marketBook, 2)}`}, ${view.trading.stake} per position`);
    }
  }
  const reading = view.regimes.find((r) => r.row === 'Regime HMM Model');
  if (reading && reading.label) lines.push(`regime reading: ${reading.label} (${markets.pct(reading.probability, 0)} sure)${reading.date ? `, after ${reading.date}` : ''}`);
  return lines;
}

app.get('/', (req, res) => {
  let html = generateHeader("Sequence Predictor", req.user);
  html += `<h1 style="margin-bottom: 8px;">Sequence Predictor</h1>
    <p style="color: #7f8c8d; margin-top: 0;">A research project that runs the same prediction models against three kinds of sequence and tracks, honestly, how each one does.
    Pick a section.</p>`;
  // Visible to every visitor: a run in progress explains why today's
  // predictions are not here yet, and is the one thing a reader cannot
  // otherwise tell.
  html += pipelineBanner();
  html += '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap:20px; align-items:stretch;">';
  html += sectionCard('Lottery games',
    'Lotto, EuroMillions, EuroDreams, VikingLotto, Keno, Pick3 and Joker+: every model\'s ticket for the next draw, and the record of every past draw.',
    lotterySectionLines(), [{ href: '/lottery', label: 'New predictions' }, { href: '/database', label: 'History' }]);
  html += sectionCard('Crypto',
    'Five coins against USDT, every day. Each coin\'s next-day return is cut into ten equally likely bins and every model predicts one bin per coin - drawn as a price on the chart, with the price band it stands for in the table beneath.',
    marketSectionLines('crypto'), [{ href: '/markets/crypto', label: 'Crypto page' }]);
  html += sectionCard('Shares',
    'Four shares on Nasdaq, every trading day, the same way: a bin per share per day, settled the morning after the close.',
    marketSectionLines('shares'), [{ href: '/markets/shares', label: 'Shares page' }]);
  html += '</div>';
  html += `<p style="color: #7f8c8d; font-size: 0.9em; margin-top: 10px;">Not advice. The models are not expected to beat any of the three; measuring whether they do, against controls that are known to carry nothing, is the point.
    The <a href="/council">Council</a> puts a question to several local language models at once.</p>`;
  html += generateFooter();
  res.send(html);
});

// 5. Lottery games: new predictions (the former home page, lottery folders only)
app.get('/lottery', (req, res) => {
  const folders = lotteryFolders();
  let html = generateHeader("Lottery games - New predictions", req.user);
  html += `<h1 style="margin-bottom: 20px;">New Predictions</h1>` + lotteryNav('predictions');
  html += `<p style="color: #7f8c8d; margin-top: -8px;">One card per game. Open a card for every model's ticket for the next draw -
       each row is one method, kept separate on purpose so its real-life record can be followed on the
       <a href="/database">History</a> page.</p>`;
  // Visible to every visitor: a run in progress explains why today's draw is
  // not here yet, and is the one thing a reader cannot otherwise tell.
  html += pipelineBanner();

  folders.forEach((folder) => {
    const folderPath = path.join(dataPath, folder);
    const files = dayFiles(folder);

    if (files.length > 0) {
      const latestFile = files[0];
      const jsonData = JSON.parse(fs.readFileSync(path.join(folderPath, latestFile), 'utf-8'));
      // Without a real result the game only steers display (Joker+ sign
      // name, digit-only chart); every other game renders exactly as before.
      const game = gameFromFolder(folder);
      const nextFrequency = chartFrequency(jsonData.numberFrequency, game);
      // Which draw these numbers are for: the next one after the newest
      // scored day file (see drawDateMeta).
      const drawMeta = drawDateMeta(files.map(parseDayFileDate), null, true);

      // Collapsed by default (No 'expanded' class)
      html += `
        <div class="card">
          <div class="card-header" onclick="toggleCard(this)">
            <div>
                <span class="card-title">${folder}</span>
                ${drawMeta}
                <span class="card-meta" style="margin-left: 10px;">${(jsonData.newPrediction || []).length} model rows</span>
            </div>
            <div class="card-icon">▼</div>
          </div>
          
          <div class="card-body">
            ${generateTable(jsonData.newPrediction, '', [], false, game, nextTableOptions(game, req.user))}

            ${nextFrequency ? `
                <div style="margin-top: 20px; height: 200px; width: 100%;">
                    <canvas id="chart-${folder}"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('chart-${folder}').getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: ${JSON.stringify(Object.keys(nextFrequency))},
                        datasets: [{ label: 'Freq', data: ${JSON.stringify(Object.values(nextFrequency))}, backgroundColor: 'rgba(52, 152, 219, 0.6)' }]
                    },
                    options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
                    });
                </script>
            ` : ''}
            
            <div style="margin-top: 15px; text-align: right;">
                <a href="/database/${folder}" style="color: #3498db; text-decoration: none; font-weight: bold;">View History →</a>
            </div>
          </div>
        </div>
      `;
    }
  });

  html += generateFooter();
  res.send(html);
});

// Login, logout and the admin's user page (auth.js). The Optuna dashboard is
// no longer started or linked from here - start it by hand when needed:
// optuna-dashboard sqlite:///db.sqlite3
auth.install(app, { header: generateHeader, footer: generateFooter });
markets.install(app, { header: generateHeader, footer: generateFooter, dataDir: marketsPath, controlsDir: controlsPath });
// The Jobs page: the scheduled pipeline jobs (jobs.js, README roadmap item
// 8) above the supervised services (services.js, today the Council API).
// services.js renders whatever the third argument returns, so it stays
// independent of the schedule.
services.install(app, { header: generateHeader, footer: generateFooter }, jobs.section);
jobs.install(app, { header: generateHeader, footer: generateFooter });
// First-login introduction and the what's-new note (whatsnew.js).
whatsnew.install(app, { header: generateHeader, footer: generateFooter }, auth);

// The UI is reachable from outside (a Tailscale funnel proxies to this
// port), pm2 restarts the process on exit, and this process now owns the
// daily pipeline schedule - so an unexpected throw must be logged with its
// stack instead of dying silently or being swallowed.
process.on('uncaughtException', (error) => {
  console.error(`[${new Date().toISOString()}] uncaught exception:`, error);
});
process.on('unhandledRejection', (reason) => {
  console.error(`[${new Date().toISOString()}] unhandled rejection:`, reason);
});

const server = app.listen(config.PORT, config.INTERFACE, () => {
  console.log(`Server running at http://${config.INTERFACE}:${config.PORT}` +
              (auth.enabled() ? ` - login required (admin: ${config.ADMIN_USER})` : ' - open access: set WEB_USER and WEB_PASSWORD to require a login'));
  services.startAll();
  // The schedule starts last and, by default, in dry mode: it records what
  // it would have run so it can be watched for a weekend next to the still
  // existing crontab entries (see jobs.js and README roadmap item 8).
  jobs.start();
});

// Services are ordinary children, so they must go down with the server
// rather than being left behind on a restart or a deploy.
['SIGTERM', 'SIGINT'].forEach((signal) => process.on(signal, () => {
  console.log(`${signal} received - stopping supervised services`);
  // Only the schedule's timer stops. A pipeline job launched with
  // setsid --fork keeps running on purpose and is adopted again at the next
  // start - that is what makes a deploy safe during a 12-hour tuning run.
  jobs.stop();
  services.stopAll();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 6000).unref();
}));
process.on('exit', () => services.stopAll());
server.on('error', (error) => {
  console.error(`Could not listen on ${config.INTERFACE}:${config.PORT}: ${error.message}`);
  process.exit(1);
});