const express = require('express');
const path = require('path');
const fs = require('fs');

const config = require("./config");
const auth = require("./auth");
const council = require("./council");

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
// modelsPath removed as it is no longer used

// --- GAME SHAPES ---
// Mirrors Predictor.py's SPECIAL_COLUMN_COUNTS: how many trailing values of a
// full result/ticket row are special numbers (euromillions stars, eurodreams
// dream number, vikinglotto viking, jokerplus zodiac sign code). A main-ball
// hit and a special-ball hit are different prize dimensions, so the UI must
// never pool them.
const SPECIAL_COLUMN_COUNTS = { euromillions: 2, eurodreams: 1, vikinglotto: 1, jokerplus: 1 };

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

// Database folder names equal game names today, but the routes historically
// matched with includes() (e.g. a "keno_backup" folder still behaves as keno),
// so keep that tolerance. vikinglotto must be tested before lotto because
// "vikinglotto".includes("lotto") is true.
function gameFromFolder(folder) {
  const games = ["euromillions", "eurodreams", "vikinglotto", "lotto", "keno", "pick3", "jokerplus"];
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
      .container { padding: 20px; max-width: 1000px; margin: auto; }
      
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
        <a href="/database">History</a>
        <a href="/council">Council</a>
        ${user && user.role === 'admin' ? '<a href="/admin/users">Users</a>' : ''}
      </div>
      ${user && !user.open ? `
      <div class="nav-user">
        <span>Signed in as <b>${auth.escapeHtml(user.name)}</b></span>
        <form method="post" action="/logout"><input type="hidden" name="_csrf" value="${auth.escapeHtml(user.csrf)}"><button type="submit" class="nav-btn">Logout</button></form>
      </div>` : ''}
    </div>

    <script>
      // Toggle Card Logic
      function toggleCard(header) {
        const card = header.parentElement;
        card.classList.toggle('expanded');
      }
    </script>
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
function generateTable(data, title = '', realResult = [], calcProfit = false, game = "") {
  const modelRows = data || [];
  if (modelRows.length === 0) return `<p style="padding: 10px; color: #888;">No predictions.</p>`;

  const specialCount = SPECIAL_COLUMN_COUNTS[game] || 0;
  // Joker+ is positional: hits are leading/trailing runs, not membership, and
  // its 7th value is a zodiac sign code that must be shown as a name.
  const isJoker = game === 'jokerplus';
  const { mains: realMains, specials: realSpecials, bonus: realBonus } = splitRealResult(realResult, game);
  // No real result (next-draw / home tables) -> no highlighting and no Hits column.
  const hasReal = realMains.length > 0;

  let html = `<div class="table-wrapper">`;
  if (title) html += `<div style="padding: 10px; font-weight: bold; background: #f8f9fa; border-bottom: 1px solid #ddd;">${title}</div>`;
  html += '<table border="1">';

  html += '<tr><th style="min-width: 150px;">Model</th><th style="width: 50px;">#</th>';
  if (modelRows.length > 0 && modelRows[0].predictions.length > 0) {
    // Joker+'s 7th column is the sign, not a seventh number.
    Array.from({ length: modelRows[0].predictions[0].length }).forEach((_, i) => html += (isJoker && i === 6) ? '<th>Sign</th>' : `<th>Num ${i + 1}</th>`);
  }
  if(hasReal) html += '<th>Hits</th>';
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
      html += `<tr>
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
          const isMatching = hasReal && (isSpecialCell ? realSpecials.includes(cell) : realMains.includes(cell));
          // Lotto bonus supplement: a played number equal to the bonus ball is
          // a tier-relevant hit ("5 (1)") but not a main hit - amber, not green.
          const isBonusMatch = hasReal && !isMatching && !isSpecialCell && realBonus.includes(cell);
          cellStyle = isMatching ? 'background: #2ecc71; color: white;'
            : (isBonusMatch ? 'background: #f39c12; color: white;' : '');
        }
        html += `<td style="text-align: center; ${cellStyle}">${cellText}</td>`;
      });
      if(hasReal) {
        let hitDisplay;
        if (isJoker) {
          // "3/1 (1)" = leading run 3, trailing run 1, sign matched; a full
          // match reads "6/0 (Z)".
          hitDisplay = `${runs.left}/${runs.right} (${signHit})`;
        } else {
          const mainHits = ticketMains.filter(n => realMains.includes(n)).length;
          const specialHits = ticketSpecials.filter(n => realSpecials.includes(n)).length
            + ticketMains.filter(n => realBonus.includes(n)).length;
          // "3 (1)" = 3 main hits, 1 special/bonus hit; games without a
          // special column or bonus just show the main count.
          hitDisplay = (specialCount > 0 || realBonus.length > 0) ? `${mainHits} (${specialHits})` : `${mainHits}`;
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
  Object.keys(report.games).sort().forEach((game) => {
    const info = report.games[game];
    const best = info.models[0];
    const value = best[info.metric];
    const valueColor = info.metric === 'profit_per_bet' ? (value > 0 ? '#27ae60' : '#c0392b') : '#2c3e50';
    const display = info.metric === 'profit_per_bet' ? `${value} €` : value;

    // Expandable full ranking per game
    const ranking = info.models.map((m, i) => {
      const v = m[info.metric];
      const mDisplay = v === null || v === undefined ? '-' : (info.metric === 'profit_per_bet' ? `${v} €` : v);
      const young = m.draws < info.minDrawsForRanking ? ' style="color: #aaa;" title="Too few scored draws to rank"' : '';
      return `<tr${young}><td>${i + 1}</td><td style="text-align: left;">${m.name}</td><td>${mDisplay}</td><td>${m.avg_hits}</td><td>${m.best_hits}</td><td>${m.draws}</td></tr>`;
    }).join('');

    rows += `
      <tr style="cursor: pointer;" onclick="const d = document.getElementById('rank-${game}'); d.style.display = d.style.display === 'none' ? 'table-row' : 'none';">
        <td style="font-weight: bold; text-align: left;">${game} <span style="color: #aaa; font-size: 0.85em;">▼</span></td>
        <td style="text-align: left;">${best.name}</td>
        <td>${metricLabel[info.metric] || info.metric}</td>
        <td style="font-weight: bold; color: ${valueColor};">${display}</td>
        <td>${best.draws}</td>
      </tr>
      <tr id="rank-${game}" style="display: none;">
        <td colspan="5" style="padding: 0;">
          <table style="width: 100%; min-width: 0; margin: 0;">
            <tr><th>#</th><th style="text-align: left;">Model</th><th>${metricLabel[info.metric] || info.metric}</th><th>Avg hits</th><th>Best day</th><th>Scored draws</th></tr>
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
            <tr><th style="text-align: left;">Game</th><th style="text-align: left;">Best model</th><th>Metric</th><th>Value</th><th>Scored draws</th></tr>
            ${rows}
          </table>
        </div>
        <p style="color: #7f8c8d; font-size: 0.85em; margin-bottom: 0;">
          Keno/Pick3/Joker+ rank by average profit per bet (real payout tables); other games by average hits of the main ticket.
          Click a game row for the full model ranking. Greyed models have fewer scored draws than the ranking minimum.
        </p>
      </div>
    </div>`;
}

// --- LOGIC: Best combination per game (README roadmap item 2, "portfolio of
// rows") - which SET of tracked rows is worth playing together, from the
// "combinations" section Helpers._build_combination_report writes into
// modelPerformance.json, with its shuffled-history control. ---
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
    const status = (entAlert || klAlert || aeAlert)
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
            <tr><th style="text-align:left;">Game</th><th>Status</th><th>Entropy (norm.)</th><th>KL vs history</th><th>KL vs uniform</th><th>AE anomaly</th><th>Draws</th><th style="text-align:left;">Entropy trend (oldest → newest)</th></tr>
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
        </p>
        ${modelCards}
      </div>
    </div>`;
}

// 1. Database Index
app.get('/database', (req, res) => {
  const folders = fs.readdirSync(dataPath, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((dir) => dir.name);
  let html = generateHeader("Database Folders", req.user);
  html += '<h1>Available Database Folders</h1><div style="display: flex; gap: 10px; flex-wrap: wrap;">';
  folders.forEach((folder) => {
    html += `<form action="/database/${folder}" method="get">
      <button type="submit" style="padding: 15px 30px; font-size: 1.1em; cursor: pointer; background: white; border: 1px solid #ccc; border-radius: 5px;">${folder}</button>
    </form>`;
  });
  html += '</div>';
  html += generatePerformanceSummary();
  html += generateCombinationSummary();
  html += generateLagAnalysis();
  html += generateRandomnessWatch();
  html += generateFooter();
  res.send(html);
});

// 2. Folder View
app.get('/database/:folder', (req, res) => {
  const folder = req.params.folder;
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

  let html = generateHeader(`${folder} Predictions`, req.user);
  html += `<h1>${folder}</h1><div>`;

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
                    predObj.predictions.forEach(p => {
                        const { mains: ticketMains, specials: ticketSpecials } = splitTicket(p, realMains, specialCount);
                        let candidate;
                        if (isJoker) {
                            // Positional runs, sign as tie-break - the same
                            // ranking Helpers.find_best_matching_prediction uses.
                            const runs = jokerplusRuns(ticketMains, realMains);
                            candidate = { mains: runs.left + runs.right, specials: jokerplusSignHit(ticketSpecials, realSpecials), left: runs.left, right: runs.right };
                        } else {
                            const mainHits = ticketMains.filter(n => realMains.includes(n)).length;
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
        const displayStat = calcProfit
          ? (isJoker ? `${matchStat} · ${formatProfit(fileProfit, game)} €` : `${fileProfit} €`)
          : matchStat;

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

  let html = generateHeader(`${file} Details`, req.user);
  html += `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0;">${file}</h1>
        <a href="/database/${folder}" class="nav-btn" style="text-decoration: none;">Back to History</a>
    </div>

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
                : ''))}

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
             <span class="card-title">Next Draw Prediction</span><div class="card-icon">▼</div>
        </div>
        <div class="card-body">
            ${generateTable(jsonData.newPrediction, '', [], false, game)}

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

// 5. Home Page
app.get('/', (req, res) => {
  const folders = fs.readdirSync(dataPath, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((dir) => dir.name);
  let html = generateHeader("Home - Dashboard", req.user);
  html += `<h1 style="margin-bottom: 20px;">New Predictions</h1>`;

  folders.forEach((folder) => {
    const folderPath = path.join(dataPath, folder);
    const files = fs.readdirSync(folderPath).filter((file) => file.endsWith('.json')).sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));

    if (files.length > 0) {
      const latestFile = files[0];
      const jsonData = JSON.parse(fs.readFileSync(path.join(folderPath, latestFile), 'utf-8'));
      // Without a real result the game only steers display (Joker+ sign
      // name, digit-only chart); every other game renders exactly as before.
      const game = gameFromFolder(folder);
      const nextFrequency = chartFrequency(jsonData.numberFrequency, game);

      // Collapsed by default (No 'expanded' class)
      html += `
        <div class="card">
          <div class="card-header" onclick="toggleCard(this)">
            <div>
                <span class="card-title">${folder}</span>
                <!--<span class="card-meta">(${latestFile})</span>-->
            </div>
            <div class="card-icon">▼</div>
          </div>
          
          <div class="card-body">
            ${generateTable(jsonData.newPrediction, '', [], false, game)}

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

app.listen(config.PORT, config.INTERFACE, () => {
  console.log(`Server running at http://${config.INTERFACE}:${config.PORT}` +
              (auth.enabled() ? ` - login required (admin: ${config.ADMIN_USER})` : ' - open access: set WEB_USER and WEB_PASSWORD to require a login'));
});