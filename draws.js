// draws.js - one draw of a game, scored (22 Oct 2026, admin batch 4: the workbench). The rules that read a drawn row
// against the models' tickets moved here from server.js, so the admin workbench (workbench.js) and the users' Lottery
// (lottery.js) score a ticket with one set of rules; below them, the pure builders of the
// workbench's views: one ticket and one row scored, one day's figures (the old History day line), a game's months
// (the old /database/:folder page), one draw's table (the old /database/:folder/:file page) and the votes under it.
//
// Every builder takes a `keep` filter on the row names: the scope. The administrator's All is every row; Top is the
// five rows ranked first before that draw (rankhistory.js, never today's ranking on a past draw); a user's view will
// be built from the kept rows only, so a figure is always over the rows it says, and a hidden row is never sent.
//
// test/workbench.test.js checks the builders on hand-made days and, next to production, the views built from a copy of
// production's files.
'use strict';

const fs = require('fs');
const path = require('path');
const brussels = require('./brussels');

// --- GAME SHAPES ---
// Mirrors Predictor.py's SPECIAL_COLUMN_COUNTS: how many trailing values of a
// full result/ticket row are special numbers (euromillions stars, eurodreams
// dream number, vikinglotto viking, jokerplus zodiac sign code). A main-ball
// hit and a special-ball hit are different prize dimensions, so the UI must
// never pool them.
const SPECIAL_COLUMN_COUNTS = { euromillions: 2, eurodreams: 1, vikinglotto: 1, jokerplus: 1 };
// the special columns by name, as the tables head them (the old page said "Num 6, Num 7")
const SPECIAL_LABELS = { euromillions: ['Star 1', 'Star 2'], eurodreams: ['Dream'], vikinglotto: ['Viking'], jokerplus: ['Sign'] };
// the two markets and their week games (one draw per week): positional, a slot per instrument
const MARKET_GAMES = ['crypto', 'shares', 'cryptoweek', 'sharesweek'];
// the games with a payout table on the History pages: keno, pick3 and Joker+ (the others are pari-mutuel)
const MONEY_GAMES = ['keno', 'pick3', 'jokerplus'];
// the games scored by position, not by membership: their vote is drawn per slot
const POSITIONAL_GAMES = ['pick3', 'jokerplus', ...MARKET_GAMES];

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
// A Joker+ ticket the payout table can read: six digits, with or without the sign (jokerplusTicketProfit scores
// any other shape 0, and it then costs nothing either)
const jokerplusValid = (ticket, realResult) => Array.isArray(ticket) && Array.isArray(realResult) && [6, 7].includes(ticket.length) && [6, 7].includes(realResult.length)
  && !ticket.slice(0, 6).map(Number).some(Number.isNaN) && !realResult.slice(0, 6).map(Number).some(Number.isNaN);

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

// --- MONEY ---
// Mirror of Helpers.PAYOUT_TABLE_KENO: the gross prize of a 1 EUR ticket per size and hits; "lost" is already the
// net result of a losing ticket (-stake).
const PAYOUT_TABLE_KENO = {
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
// The keno ticket sizes with a tracked profit (Helpers.keno_ticket_profit): 5 to 10 numbers. The 20-number row is a
// row's ranking, not a ticket.
const KENO_SIZES = { min: 5, max: 10 };
const KENO_STAKE = -PAYOUT_TABLE_KENO.lost;
const kenoPlayed = (ticket) => Array.isArray(ticket) && ticket.length >= KENO_SIZES.min && ticket.length <= KENO_SIZES.max;
// Net profit of one keno ticket, as Helpers.keno_ticket_profit (the History ranking's profit_per_bet): the prize
// minus the 1 EUR stake on a win, -1 on a loss, hits counted once per number; a size without a tracked profit 0.
// Until 22 Oct 2026 the pages added the GROSS prize of a winning ticket (one euro more per winning ticket than the
// ranking counts, 843 euro over the 348 stored keno days of 22 Oct) and paid sizes 2 to 4 too, which no row plays.
function kenoTicketProfit(prediction, realResult) {
  if (!kenoPlayed(prediction)) return 0;
  const real = new Set((realResult || []).map(Number));
  const hits = new Set(prediction.map(Number).filter((n) => real.has(n))).size;
  const prize = PAYOUT_TABLE_KENO[prediction.length][hits];
  return prize === undefined ? PAYOUT_TABLE_KENO.lost : prize - KENO_STAKE;
}
// Mirror of Helpers.PAYOUT_TABLE_PICK3 (Reglement Pick-3, juli 2024).
const PAYOUT_TABLE_PICK3 = {
  straight: 500, straight_consolation: 1, box_with_doubles: 160, box_no_doubles: 80,
  front_pair: 50, back_pair: 50, bet_cost: 1
};
// One pick3 ticket's bets, each evaluated on its own (Helpers.pick3_ticket_profit): the tracked ticket plays every
// bet type at 1 EUR - straight, box, front pair, back pair; a triple cannot play box, so its stake is 3 - and the
// prizes cumulate. {stake, payout, bets: {straight, consolation, box, frontPair, backPair}} (each the prize won, 0
// when lost), or null for a row that is not three digits (or a result without three).
function pick3Bets(prediction, realResult) {
  if (!Array.isArray(prediction) || prediction.length != 3 || !Array.isArray(realResult) || realResult.length < 3) return null;
  const table = PAYOUT_TABLE_PICK3;
  const pred = prediction.map(Number); const actual = realResult.slice(0, 3).map(Number);
  const distinct = new Set(pred).size;
  const isTriple = distinct === 1;
  const bets = { straight: 0, consolation: 0, box: 0, frontPair: 0, backPair: 0 };
  if (pred[0] === actual[0] && pred[1] === actual[1] && pred[2] === actual[2]) bets.straight = table.straight;
  else if (pred[2] === actual[2]) bets.consolation = table.straight_consolation;
  if (!isTriple && [...pred].sort((a, b) => a - b).join(',') === [...actual].sort((a, b) => a - b).join(',')) {
    bets.box = distinct === 2 ? table.box_with_doubles : table.box_no_doubles;
  }
  if (pred[0] === actual[0] && pred[1] === actual[1]) bets.frontPair = table.front_pair;
  if (pred[1] === actual[1] && pred[2] === actual[2]) bets.backPair = table.back_pair;
  const payout = bets.straight + bets.consolation + bets.box + bets.frontPair + bets.backPair;
  return { stake: (isTriple ? 3 : 4) * table.bet_cost, payout, bets };
}

function calculateProfit(prediction, realResult, game, name) {
  switch (game) {
    case "keno": {
      // Net, as the History ranking counts it (kenoTicketProfit); a row of more than 10 numbers is a ranking, 0.
      return kenoTicketProfit(prediction, realResult);
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
      const b = pick3Bets(prediction, realResult);
      return b ? b.payout - b.stake : 0;
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
// What one ticket costs on the money games' pages: keno 1 EUR a played size, pick3 4 (3 for a triple), Joker+ 1.50;
// a ticket the payout table cannot read costs nothing and wins nothing. 0 for every other game (no payout table).
function ticketStake(prediction, realResult, game) {
  if (game === 'keno') return kenoPlayed(prediction) ? KENO_STAKE : 0;
  if (game === 'pick3') { const b = pick3Bets(prediction, realResult); return b ? b.stake : 0; }
  if (game === 'jokerplus') return jokerplusValid(prediction, realResult) ? PAYOUT_TABLE_JOKERPLUS.betCost : 0;
  return 0;
}

// --- DAY FILES ---
// A day file is named after the draw it scored, not zero padded: data/database/<game>/2026-9-16.json. Its
// "newPrediction" is the prediction for the NEXT draw of that game (checked against the data: file D's
// newPrediction is exactly what file D+1 carries as its scored currentPrediction). The schedule differs per game
// and has changed before - Keno/Pick3/Joker+ draw daily, Lotto Wed+Sat, Euromillions Tue+Fri, EuroDreams Mon+Thu,
// VikingLotto Wed - so it is read from the game's own recent draw dates instead of being hardcoded: the weekdays of
// the last SCHEDULE_WINDOW stored draws are the schedule, and the prediction applies to the first date after its
// file that falls on one of them.
const SCHEDULE_WINDOW = 40;
const DAY_FILE = /^(\d{4})-(\d{1,2})-(\d{1,2})\.json$/;

// "2026-9-16.json" (day files are not zero padded) -> local Date, or null
// for anything that isn't a day file.
function parseDayFileDate(file) {
  const parts = String(file).replace('.json', '').split('-').map(Number);
  if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) return null;
  const [year, month, day] = parts;
  const date = new Date(year, month - 1, day);
  return (date.getFullYear() === year && date.getMonth() === month - 1 && date.getDate() === day) ? date : null;
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

// The ISO day of a day file ("2026-9-5.json" -> "2026-09-05"), or null; and back.
const pad2 = (n) => String(n).padStart(2, '0');
function isoOfFile(file) {
  const m = DAY_FILE.exec(String(file));
  if (!m) return null;
  const iso = `${m[1]}-${pad2(m[2])}-${pad2(m[3])}`;
  const d = new Date(`${iso}T00:00:00Z`);
  return !Number.isNaN(d.getTime()) && d.toISOString().slice(0, 10) === iso ? iso : null;
}
function fileOfIso(day) {
  const m = /^(\d{4})-(\d{1,2})-(\d{1,2})$/.exec(String(day));
  return m ? `${Number(m[1])}-${Number(m[2])}-${Number(m[3])}.json` : null;
}
const isoOfDate = (date) => (date ? `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}` : null);
// The draw a newest day's next prediction is for (nextDrawDate on the game's own schedule), as an ISO day.
const nextDrawDay = (files, anchorFile) => isoOfDate(nextDrawDate(files.map(parseDayFileDate), parseDayFileDate(anchorFile)));
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const weekdayOf = (iso) => WEEKDAYS[new Date(`${iso}T12:00:00Z`).getUTCDay()];

// --- SALES CLOSE ---
// src/drawClose.json (27 Oct 2026): a lottery draw's sales close per game and weekday, Brussels wall time - the table
// src/Liveness.py judges a ticket by (saved after its draw's close, it could not be played and counts out of the live
// record) and rankhistory.js cuts a draw's point-in-time ranking at. Read once when this module loads.
const DRAW_CLOSE = (() => {
  try { return JSON.parse(fs.readFileSync(path.join(__dirname, 'src', 'drawClose.json'), 'utf-8')); } catch (e) { return null; }
})();
// The sales close of a game's draw on an ISO day: {time: '13:00' (Brussels), at (epoch ms), scheduled}; a weekday the
// table does not list gets the game's earliest close with scheduled false, as Liveness.close_instant. null for a game
// the table does not list (the markets: their close is markets.js's) or a day that is not one.
function salesClose(game, iso) {
  const table = DRAW_CLOSE && DRAW_CLOSE.games ? DRAW_CLOSE.games[game] : null;
  if (!table || !brussels.isDay(String(iso))) return null;
  const listed = table[weekdayOf(iso)];
  const time = listed || Object.values(table).sort()[0];
  if (!/^\d{2}:\d{2}$/.test(String(time))) return null;
  const [hour, minute] = time.split(':').map(Number);
  return { time, at: brussels.instant(iso, hour, minute), scheduled: Boolean(listed) };
}

// --- the workbench's builders ------------------------------------------------------------------------------
// How a game's tickets are read: 'lotto' (six mains, the bonus supplements), 'specials' (mains and specials apart:
// euromillions, eurodreams, vikinglotto), 'jokerplus' (runs from either end and the sign), 'keno' (a ranking of 20
// and one ticket per size), 'pick3' (three digits, by position), 'market' (a bin per instrument, by slot).
function familyOf(game) {
  if (game === 'lotto' || game === 'jokerplus' || game === 'keno' || game === 'pick3') return game;
  if (MARKET_GAMES.includes(game)) return 'market';
  return SPECIAL_COLUMN_COUNTS[game] ? 'specials' : 'numbers';
}
const round2 = (x) => (typeof x === 'number' && Number.isFinite(x) ? Math.round(x * 100) / 100 : null);
const rowsOf = (list) => (Array.isArray(list) ? list.filter((m) => m && typeof m === 'object' && Array.isArray(m.predictions)) : []);
const nameOf = (m) => (m && m.name) || 'not known';
const keepAll = () => true;

// One ticket, scored. mark: one letter per main cell - 'm' a hit (lotto, keno, the special games: the number is
// among the drawn mains; pick3 and the markets: the right value in the right slot), 'b' the lotto bonus ball (a
// played number equal to it: the "(1)" of "5 (1)", amber), 'l' / 'r' Joker+'s leading and trailing runs, '-' none.
// markSpecials: 's' a special that is among the drawn specials, 'z' Joker+'s sign. hits: lotto and the special
// games {mains, specials} - "N (M)", lotto's M the bonus -, the markets {mains} by slot, keno {mains}, pick3
// {positional, membership} (membership is the old page's Hits column), Joker+ {left, right, sign} - "L/R (Z)".
// Without a drawn row (the coming draw) only the cells; mainsRef, the drawn mains a row is split against (a row
// longer than them carries specials), is then the newest drawn row's - by default the drawn row itself.
function scoreTicket(game, row, realResult, split = splitRealResult(realResult, game), mainsRef = split.mains) {
  const family = familyOf(game);
  const cells = Array.isArray(row) ? row : [];
  const { mains: realMains, specials: realSpecials, bonus: realBonus } = split;
  const { mains, specials } = splitTicket(cells, mainsRef, SPECIAL_COLUMN_COUNTS[game] || 0);
  // specials only where the game has special columns ([] there: a mains-only ticket, e.g. an RL ticket row)
  const out = { mains: mains.slice() };
  if (SPECIAL_COLUMN_COUNTS[game]) out.specials = specials.slice();
  if (family === 'keno') out.kind = kenoPlayed(cells) ? 'ticket' : 'ranking';
  if (family === 'jokerplus' && specials.length) out.signName = zodiacName(specials[0]);
  if (!realMains.length) return out;
  if (family === 'jokerplus') {
    const runs = jokerplusRuns(mains, realMains);
    const sign = jokerplusSignHit(specials, realSpecials);
    out.mark = mains.map((_, i) => (i < runs.left ? 'l' : (i >= mains.length - runs.right ? 'r' : '-'))).join('');
    out.markSpecials = specials.map(() => (sign ? 'z' : '-')).join('');
    out.hits = { left: runs.left, right: runs.right, sign };
  } else {
    const bySlot = family === 'market' || family === 'pick3';
    out.mark = mains.map((n, i) => {
      if (bySlot ? realMains[i] === n : realMains.includes(n)) return 'm';
      return realBonus.includes(n) ? 'b' : '-';
    }).join('');
    if (specials.length) out.markSpecials = specials.map((n) => (realSpecials.includes(n) ? 's' : '-')).join('');
    const membership = mains.filter((n) => realMains.includes(n)).length;
    const positional = mains.filter((n, i) => realMains[i] === n).length;
    // lotto's supplement is the bonus ball matched against the played numbers themselves (splitRealResult)
    const extra = specials.filter((n) => realSpecials.includes(n)).length + mains.filter((n) => realBonus.includes(n)).length;
    if (family === 'pick3') {
      out.markMember = mains.map((n) => (realMains.includes(n) ? 'm' : '-')).join('');
      out.hits = { positional, membership };
    } else if (family === 'market') out.hits = { mains: positional };
    else if (family === 'keno') out.hits = { mains: membership };
    else out.hits = { mains: membership, specials: extra };
  }
  if (MONEY_GAMES.includes(game)) {
    const played = family !== 'keno' || out.kind === 'ticket';
    out.stake = played ? ticketStake(cells, realResult, game) : 0;
    out.profit = played ? round2(calculateProfit(cells, realResult, game)) : null;
    if (family === 'pick3') { const b = pick3Bets(cells, realResult); out.bets = b ? b.bets : null; }
  }
  return out;
}

// One model row of a table: its tickets scored, and for a lotto row with extras the multi-pick (the 7th to 9th
// numbers of its first ticket, nine: the hits among all nine).
function scoreRow(game, model, realResult, split = splitRealResult(realResult, game), mainsRef = split.mains) {
  const tickets = model.predictions.map((p) => scoreTicket(game, p, realResult, split, mainsRef));
  const out = { name: nameOf(model), tickets };
  if (multiPickFor(game) && Array.isArray(model.multiPick) && model.multiPick.length) {
    const extras = model.multiPick.map(Number);
    const mp = { numbers: extras };
    if (split.mains.length) {
      mp.mark = extras.map((n) => (split.mains.includes(n) ? 'm' : '-')).join('');
      if (tickets[0]) mp.nine = multiHits(tickets[0].mains, extras, split.mains);
    }
    out.multiPick = mp;
  }
  return out;
}

// One day's figures, the old History day line (server.js /database/:folder), over the kept rows: the money games'
// profit (keno, pick3, Joker+) summed row by row, and every other game's best ticket - the most main hits, then
// specials (Joker+: L + R, then the sign; a market: hits by slot) - with lotto's best of nine. rows: the model rows
// counted; tickets: their tickets (keno: the played sizes only, not the 20-number ranking); stake: what those tickets
// cost on a money game. best is null for keno and pick3 (the old page showed their profit only) and with no row.
function dayFigures(game, day, keep = keepAll) {
  const realResult = Array.isArray(day && day.realResult) ? day.realResult : [];
  const rows = rowsOf(day && day.currentPrediction).filter((m) => keep(nameOf(m)));
  const family = familyOf(game);
  const money = MONEY_GAMES.includes(game);
  const split = splitRealResult(realResult, game);
  const specialCount = SPECIAL_COLUMN_COUNTS[game] || 0;
  let profit = 0; let stake = 0; let tickets = 0;
  let best = null; let multi = null;
  rows.forEach((m) => {
    let rowProfit = 0;
    m.predictions.forEach((p) => {
      if (family === 'keno' ? kenoPlayed(p) : true) tickets += 1;
      if (money) { rowProfit += calculateProfit(p, realResult, game, nameOf(m)); stake += ticketStake(p, realResult, game); }
    });
    profit += rowProfit;
    if (money && family !== 'jokerplus') return;
    if (multiPickFor(game) && Array.isArray(m.multiPick) && m.predictions[0]) {
      const { mains: mainsOnly } = splitTicket(m.predictions[0], split.mains, specialCount);
      multi = Math.max(multi === null ? 0 : multi, multiHits(mainsOnly, m.multiPick, split.mains));
    }
    m.predictions.forEach((p) => {
      const { mains, specials } = splitTicket(p, split.mains, specialCount);
      let c;
      if (family === 'jokerplus') {
        const runs = jokerplusRuns(mains, split.mains);
        c = { mains: runs.left + runs.right, specials: jokerplusSignHit(specials, split.specials), left: runs.left, right: runs.right };
      } else {
        const mainHits = family === 'market' ? mains.filter((n, i) => split.mains[i] === n).length : mains.filter((n) => split.mains.includes(n)).length;
        c = { mains: mainHits, specials: specials.filter((n) => split.specials.includes(n)).length + mains.filter((n) => split.bonus.includes(n)).length };
      }
      if (!best || c.mains > best.mains || (c.mains === best.mains && c.specials > best.specials)) best = { ...c, by: nameOf(m) };
    });
  });
  const out = { rows: rows.length, tickets };
  if (money) { out.profit = round2(profit); out.stake = round2(stake); }
  out.best = best ? bestOut(family, best) : null;
  if (multi !== null) out.multi = multi;
  return out;
}
// the best ticket as the line reads it: "N (M)" (lotto: M the bonus; the special games: the specials), "N" (the
// markets, by slot), "L/R (Z)" (Joker+); by: the first row in the table's order that reached it
function bestOut(family, b) {
  if (family === 'jokerplus') return { left: b.left, right: b.right, sign: b.specials, by: b.by };
  if (family === 'market' || family === 'numbers') return { mains: b.mains, by: b.by };
  return { mains: b.mains, specials: b.specials, by: b.by };
}
// "better" as the old month header compares two best lines: more mains (Joker+: L + R), then more specials
const bestKey = (family, b) => (family === 'jokerplus' ? [b.left + b.right, b.sign] : [b.mains, b.specials || 0]);
function betterBest(family, a, b) {
  if (!b) return a;
  if (!a) return b;
  const [am, as] = bestKey(family, a); const [bm, bs] = bestKey(family, b);
  return bm > am || (bm === am && bs > as) ? b : a;
}

// The drawn row as the strip shows it: mains, then the specials or the lotto bonus as their own group; Joker+'s sign
// by its name.
function resultOf(game, realResult) {
  if (!Array.isArray(realResult) || !realResult.length) return null;
  const s = splitRealResult(realResult, game);
  const out = { mains: s.mains, specials: s.specials, bonus: s.bonus };
  if (game === 'jokerplus' && s.specials.length) out.signName = zodiacName(s.specials[0]);
  return out;
}

// A game's months, the old /database/:folder page: entries [{date (ISO), day (the parsed file), keep (the scope's
// filter, or null when the scope has no rows for that draw), ranking (what the scope says about it), liveMarks (the
// kept rows not live as stored, by status: the admin's markers), afterClose (the users' rule: a kept row's
// ticket was not out before the sales close)}], newest first. Months newest first, each with its draws newest first
// and its totals over the same rows: the profit and stake summed, the best line of the month; draws: every stored
// draw of the month, counted: those with rows in scope (with Top, the draws that have a ranking recorded before
// them) and not afterClose - every total is over the counted draws only; afterClose: the draws shown greyed with
// their figures, left out of the totals (27 Oct 2026).
function historyMonths(game, entries) {
  const family = familyOf(game);
  const money = MONEY_GAMES.includes(game);
  const months = [];
  let current = null;
  entries.forEach((e) => {
    const month = e.date.slice(0, 7);
    if (!current || current.month !== month) {
      // rows: summed over the counted draws (row-draws); rowsMin / rowsMax: the rows of one counted draw, the header's words
      current = { month, draws: 0, counted: 0, afterClose: 0, rows: 0, rowsMin: null, rowsMax: null, tickets: 0, ...(money ? { profit: 0, stake: 0 } : {}), best: null, multi: null, days: [] };
      months.push(current);
    }
    const line = { date: e.date, weekday: weekdayOf(e.date), result: resultOf(game, e.day && e.day.realResult) };
    current.draws += 1;
    if (e.liveMarks) line.liveMarks = e.liveMarks;
    if (e.keep && e.afterClose) {
      // shown with its figures, greyed, out of every total
      Object.assign(line, dayFigures(game, e.day, e.keep), { afterClose: true });
      current.afterClose += 1;
    } else if (e.keep) {
      const f = dayFigures(game, e.day, e.keep);
      Object.assign(line, f);
      current.counted += 1;
      current.rows += f.rows; current.tickets += f.tickets;
      current.rowsMin = current.rowsMin === null ? f.rows : Math.min(current.rowsMin, f.rows);
      current.rowsMax = current.rowsMax === null ? f.rows : Math.max(current.rowsMax, f.rows);
      if (money) { current.profit += f.profit; current.stake += f.stake; }
      current.best = betterBest(family, current.best, f.best);
      if (f.multi !== undefined) current.multi = Math.max(current.multi === null ? 0 : current.multi, f.multi);
    }
    if (e.ranking !== undefined) line.ranking = e.ranking;
    current.days.push(line);
  });
  months.forEach((m) => {
    if (money) { m.profit = round2(m.profit); m.stake = round2(m.stake); }
    if (m.multi === null) delete m.multi;
  });
  return months;
}

// The vote under a table: how many of the kept rows' first tickets carry each value. A positional game (pick3,
// Joker+, the markets) per slot, with the value drawn in that slot - the old pooled chart lit a bar for a digit drawn
// in another slot (owner's question 3) -, every other game per number with the drawn mains. stored: the day file's
// own vote of every row (Helpers.count_number_frequencies_from_new_prediction: weighted by the rows' scores, mains
// only), sent only when every row is kept; from: 'file', or 'previous file' when the file has none (35 old
// vikinglotto files; the previous file's numberFrequency is the same vote).
// mainsRef as in scoreTicket: the coming draw's rows are split against the newest drawn row.
function votesOf(game, rows, realResult, stored = null, from = 'file', mainsRef = null) {
  const split = splitRealResult(realResult, game);
  const ref = mainsRef || split.mains;
  const firsts = rows.map((m) => (m.predictions[0] ? splitTicket(m.predictions[0], ref, SPECIAL_COLUMN_COUNTS[game] || 0).mains : []));
  const count = (values) => {
    const n = new Map();
    values.forEach((v) => n.set(v, (n.get(v) || 0) + 1));
    return [...n.entries()].sort((a, b) => Number(a[0]) - Number(b[0]));
  };
  const out = { rows: rows.length };
  if (POSITIONAL_GAMES.includes(game)) {
    const width = Math.max(ref.length, ...firsts.map((t) => t.length), 0);
    out.kind = 'slots';
    out.slots = Array.from({ length: width }, (_, i) => ({ drawn: split.mains.length ? split.mains[i] : null,
      counts: count(firsts.map((t) => t[i]).filter((v) => v !== undefined && v !== null)) }));
  } else {
    out.kind = 'numbers';
    out.counts = count(firsts.flatMap((t) => [...new Set(t)]));
    out.drawn = split.mains;
  }
  const freq = chartFrequency(stored, game);
  if (freq && typeof freq === 'object' && Object.keys(freq).length) {
    out.stored = { from, values: Object.entries(freq).map(([k, v]) => [Number(k), Math.round(Number(v) * 10000) / 10000])
      .filter(([k, v]) => Number.isFinite(k) && Number.isFinite(v)).sort((a, b) => a[0] - b[0]) };
  }
  return out;
}

// The multi-pick legend's figures, once per table (B7): the system plays of 7 to 9 numbers, their grids and price,
// the chance of 3 of the six drawn among 6 to 9 numbers, and an extra number's chance of being drawn.
function multiPickLegend(game) {
  const cfg = multiPickFor(game);
  if (!cfg) return null;
  return { extra: cfg.extra, gridPrice: cfg.gridPrice, perExtra: Math.round(cfg.perExtra * 10000) / 10000, minHits: 3,
    sizes: [6, 7, 8, 9].map((size) => ({ size, grids: size === 6 ? 1 : cfg.grids[size], price: size === 6 ? cfg.gridPrice : cfg.grids[size] * cfg.gridPrice,
      chance: Math.round(cfg.chance[size] * 10000) / 10000 })) };
}

// The shape of a game's table: its main cells, its special columns by name, the lotto bonus, whether it is read by
// slot, whether it has money, and the price of a ticket.
function shapeOf(game, { slots = null } = {}) {
  const family = familyOf(game);
  const out = { family, specials: SPECIAL_LABELS[game] || [], bonus: game === 'lotto', positional: POSITIONAL_GAMES.includes(game), money: MONEY_GAMES.includes(game) };
  if (slots) out.slots = slots;
  if (game === 'keno') out.ticket = { currency: 'EUR', price: 1, sizes: [KENO_SIZES.min, KENO_SIZES.max], ranking: 'a row of more than 10 numbers is the row\'s ranking, not played' };
  if (game === 'pick3') out.ticket = { currency: 'EUR', price: 4, triple: 3, bets: ['straight', 'box', 'front pair', 'back pair'], note: 'every bet type at 1 EUR; a triple cannot play box' };
  if (game === 'jokerplus') out.ticket = { currency: 'EUR', price: PAYOUT_TABLE_JOKERPLUS.betCost };
  return out;
}

module.exports = {
  SPECIAL_COLUMN_COUNTS, SPECIAL_LABELS, MARKET_GAMES, MONEY_GAMES, POSITIONAL_GAMES, MULTI_PICK, multiPickFor, multiHits, winChance,
  ZODIAC, zodiacCode, zodiacName, jokerplusRuns, jokerplusSignHit, PAYOUT_TABLE_JOKERPLUS, jokerplusTicketProfit,
  chartFrequency, gameFromFolder, splitRealResult, splitTicket, PAYOUT_TABLE_KENO, KENO_SIZES, KENO_STAKE, kenoPlayed, kenoTicketProfit, PAYOUT_TABLE_PICK3, pick3Bets,
  calculateProfit, ticketStake, SCHEDULE_WINDOW, DAY_FILE, parseDayFileDate, nextDrawDate, isoOfFile, fileOfIso, nextDrawDay, weekdayOf, DRAW_CLOSE, salesClose,
  familyOf, scoreTicket, scoreRow, dayFigures, betterBest, resultOf, historyMonths, votesOf, multiPickLegend, shapeOf,
};
