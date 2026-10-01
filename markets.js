// markets.js - the Crypto and Shares pages (README roadmap item 4, phase M2).
//
// Predictor.py tracks the two markets as positional games, and after each
// run src/MarketSettle.py settles every stored day against the real returns
// and writes data/markets/<market>.json: closes, the predicted course per
// model, the next day's predictions as prices, per-model accuracy against
// chance and the daily accuracy series. These pages draw that file and
// nothing else - no database, no Python at request time.
//
// Two more records join the page (phase M3): data/controls/markets/<market>-rows.json,
// the weekly MarketRows.py report that scores the market rows under a proper
// scoring rule against GARCH, and data/markets/<market>-regimes.json, the
// Regime HMM rows' reading of which regime the market is in.
//
// Pure functions (loadMarket, describeMarket, formatting) are exported for
// test/markets.test.js; install() adds the two routes.
'use strict';

const fs = require('fs');
const path = require('path');

const MARKETS = {
  crypto: { title: 'Crypto predictor', noun: 'coin', unit: 'USDT', calendar: 'every day (UTC)' },
  shares: { title: 'Shares predictor', noun: 'share', unit: 'USD', calendar: 'every trading day (New York)' },
};

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf-8')); } catch (e) { return null; }
}

function loadMarket(dir, market) {
  const record = readJson(path.join(dir, `${market}.json`));
  return record && record.market === market && Array.isArray(record.instruments) && Array.isArray(record.models) ? record : null;
}

function loadRows(controlsDir, market) {
  const record = readJson(path.join(controlsDir, 'markets', `${market}-rows.json`));
  return record && record.market === market && Array.isArray(record.rows) ? record : null;
}

function loadRegimes(dir, market) {
  const record = readJson(path.join(dir, `${market}-regimes.json`));
  return record && typeof record === 'object' ? record : null;
}

const num = (x) => {
  const v = Number(x);
  return x === null || x === undefined || !Number.isFinite(v) ? null : v;
};
const pct = (x, digits = 1) => (num(x) === null ? '-' : `${(num(x) * 100).toFixed(digits)}%`);
const money = (x, digits = 4) => (num(x) === null ? '-' : (num(x) >= 0 ? '+' : '') + num(x).toFixed(digits));
const price = (x) => {
  const v = num(x);
  if (v === null) return '-';
  return v >= 1000 ? v.toFixed(0) : (v >= 10 ? v.toFixed(2) : v.toFixed(4));
};

// The weekly report's rows, ready to print: log-scores, the interval and
// verdict against GARCH and against uniform, hit rates and the fixed rule's
// P&L with the real returns. Sorted best log-score first, rows without a
// probability (the baselines) last.
function describeRows(record) {
  if (!record) return null;
  const uniform = num(record.uniform_log_score);
  const interval = (x) => (x && num(x.mean) !== null ? { mean: num(x.mean), lo: num(x.lo), hi: num(x.hi), verdict: String(x.verdict || ''), days: num(x.days) } : null);
  const rows = (record.rows || []).map((r) => ({
    name: String(r.name), kind: String(r.kind || 'base'), days: num(r.days), scoredDays: num(r.scored_days),
    logScore: num(r.log_score), se: num(r.log_score_se),
    vsReference: interval(r.vs_reference), vsUniform: interval(r.vs_uniform),
    exact: num(r.exact_rate), adjacent: num(r.adjacent_rate), direction: num(r.direction_rate),
    trades: num(r.trades), pnl: num(r.pnl_return), pnlPerTrade: num(r.pnl_per_trade),
    isReference: String(r.name) === String(record.reference),
  }));
  rows.sort((a, b) => {
    if ((a.logScore === null) !== (b.logScore === null)) return a.logScore === null ? 1 : -1;
    if (a.logScore !== null && b.logScore !== null && a.logScore !== b.logScore) return b.logScore - a.logScore;
    return (b.exact || 0) - (a.exact || 0);
  });
  const reference = rows.find((r) => r.isReference) || null;
  return {
    market: record.market, generatedAt: record.generated_at || null, days: num(record.days_scored), requested: num(record.days_requested),
    firstDay: record.first_day || null, lastDay: record.last_day || null, uniform, reference: String(record.reference || ''),
    referenceRow: reference, floor: num(record.probability_floor), rows, lockboxWithheld: num(record.lockbox_days_withheld) || 0,
    betterThanReference: rows.filter((r) => r.vsReference && r.vsReference.verdict === 'better').map((r) => r.name),
    referenceAboveUniform: !!(reference && reference.vsUniform && reference.vsUniform.verdict === 'better'),
    errors: record.errors && typeof record.errors === 'object' ? record.errors : {},
  };
}

// The newest reading of each Regime HMM row: which regime, how sure, how many.
function describeRegimes(record, symbols) {
  if (!record) return [];
  return Object.keys(record).filter((k) => Array.isArray(record[k]) && record[k].length).sort().map((row) => {
    const latest = record[row][record[row].length - 1];
    return {
      row, date: latest.date || null, regimes: num(latest.regimes), label: String(latest.label || ''), template: latest.template === null || latest.template === undefined ? null : num(latest.template),
      probability: num(latest.probability), volatilityRank: num(latest.volatility_rank),
      expected: Array.isArray(latest.expected_return) ? latest.expected_return.map((v, i) => ({ symbol: symbols && symbols[i] ? symbols[i] : `#${i + 1}`, value: num(v) })) : [],
      volatility: Array.isArray(latest.volatility) ? latest.volatility.map((v) => num(v)) : [],
      history: record[row].length,
    };
  });
}

// One view model per market: models with their rates read against chance,
// instruments with what the charts need, the next-day table, the series.
// `extras` = { rows, regimes }: the weekly report and the regime log.
function describeMarket(record, extras) {
  const chance = record.chance || { exact: 0.1, adjacent: 0.28, direction: 0.5 };
  const models = (record.models || []).map((m) => {
    const exact = num(m.exact_rate); const adjacent = num(m.adjacent_rate); const direction = num(m.direction_rate);
    return {
      name: String(m.name), days: num(m.days), positions: num(m.positions), trades: num(m.trades),
      exact, adjacent, direction,
      aboveExact: exact !== null && exact > chance.exact,
      aboveAdjacent: adjacent !== null && adjacent > chance.adjacent,
      aboveDirection: direction !== null && direction > chance.direction,
      pnlTotal: num(m.pnl_total), pnlPerTrade: num(m.pnl_per_trade),
      firstDay: m.first_day || null, lastDay: m.last_day || null,
    };
  });
  const next = record.next || null;
  const instruments = (record.instruments || []).filter((i) => i && typeof i === 'object').map((i) => {
    const nextFor = next && next.instruments && next.instruments[i.symbol] ? next.instruments[i.symbol] : null;
    return {
      symbol: String(i.symbol), name: i.name || String(i.symbol), position: num(i.position), quote: i.quote || '',
      active: i.active !== false,
      lastClose: num(i.last_close), lastDate: i.last_date || null,
      closes: Array.isArray(i.closes) ? i.closes.filter((p) => Array.isArray(p) && p.length === 2 && num(p[1]) !== null) : [],
      course: i.predicted_course && typeof i.predicted_course === 'object' ? i.predicted_course : {},
      next: nextFor ? (nextFor.predictions || []).map((p) => ({
        model: String(p.model), bin: num(p.bin), direction: num(p.direction), price: num(p.price), low: num(p.low), high: num(p.high),
      })) : [],
    };
  });
  return {
    market: record.market, generatedAt: record.generated_at || null, k: num(record.k) || 10, fee: num(record.fee),
    chance, newestGameDay: record.newest_game_day || null,
    madeOn: next ? next.made_on || null : null,
    models, drawn: Array.isArray(record.drawn_models) ? record.drawn_models.map(String) : [],
    instruments, daily: Array.isArray(record.daily) ? record.daily : [],
    scoredDays: models.length ? Math.max(...models.map((m) => m.days || 0)) : 0,
    rows: describeRows(extras && extras.rows ? extras.rows : null),
    regimes: describeRegimes(extras && extras.regimes ? extras.regimes : null, instruments.map((i) => i.symbol)),
    days: describeDays(record.days),
  };
}

// The settled days in market terms (MarketSettle.day_records): per instrument
// the real return, its bin and the edges it was cut with; per model the bins
// it played and the day's hits. Malformed entries are dropped, never thrown.
function describeDays(days) {
  if (!Array.isArray(days)) return [];
  return days.filter((d) => d && typeof d.date === 'string' && Array.isArray(d.instruments)).map((d) => ({
    date: d.date,
    instruments: d.instruments.filter((i) => i && i.symbol !== undefined).map((i) => ({
      symbol: String(i.symbol), ret: num(i.return), bin: num(i.bin), edges: Array.isArray(i.edges) ? i.edges.map(num) : [],
    })),
    models: (Array.isArray(d.models) ? d.models : []).filter((m) => m && typeof m === 'object' && m.name !== undefined).map((m) => ({
      name: String(m.name), bins: Array.isArray(m.bins) ? m.bins.map((b) => (b === null ? null : num(b))) : [],
      exact: num(m.exact), adjacent: num(m.adjacent), direction: num(m.direction), positions: num(m.positions), pnl: num(m.pnl), trades: num(m.trades),
    })),
    best: d.best || null, exactMean: num(d.exact_mean),
  }));
}

// A bin's return interval under the day's edges: [low, high] with null at the
// open ends (MarketGame.bin_interval).
function binInterval(bin, edges) {
  if (bin === null || !edges.length) return [null, null];
  return [bin > 0 ? edges[bin - 1] : null, bin < edges.length ? edges[bin] : null];
}
const signedPct = (x, digits = 1) => {
  if (num(x) === null) return '-';
  const text = (num(x) * 100).toFixed(digits);
  if (Number(text) === 0) return `${Math.abs(Number(text)).toFixed(digits)}%`;   // never "-0.0%"
  return `${num(x) >= 0 ? '+' : ''}${text}%`;
};
function intervalText(bin, edges) {
  const [low, high] = binInterval(bin, edges);
  if (low === null && high === null) return '-';
  if (low === null) return `below ${signedPct(high)}`;
  if (high === null) return `above ${signedPct(low)}`;
  return `${signedPct(low)} to ${signedPct(high)}`;
}
// The day file behind a game day, as Predictor.py names it (no zero padding).
function gameViewLink(market, date) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(date));
  return m ? `/database/${market}/${m[1]}-${Number(m[2])}-${Number(m[3])}.json` : `/database/${market}`;
}

// --- page -------------------------------------------------------------------
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const COLOURS = ['#e67e22', '#8e44ad', '#16a085', '#c0392b', '#2980b9'];

function page(market, view, header, footer, user) {
  const meta = MARKETS[market];
  let html = header(meta.title, user);
  html += `<h1>${esc(meta.title)}</h1>`;
  if (!view) {
    html += `<p style="color:#7f8c8d;">No market record yet. It appears after the first daily run that includes the <code>${esc(market)}</code>
      game (the predictor fetches the bars, cuts the return bins, predicts, and the settlement writes <code>data/markets/${esc(market)}.json</code>).</p>`;
    return html + footer();
  }
  const c = view.chance;
  html += `<p style="color:#7f8c8d; margin-top:-12px;">Same models as the lottery games, same daily tracking, same controls - a predictor, not a trading bot.
    Each ${esc(meta.noun)}'s next-day return is cut into ${view.k} equiprobable bins fitted on its own past (see README, roadmap item 4), and every model
    predicts one bin per ${esc(meta.noun)}, ${esc(meta.calendar)}. A bin is a return interval, so it is drawn as a predicted price on the chart, with the price band it stands for in the table beneath.
    Chance is ${pct(c.exact, 0)} for the exact bin, ${pct(c.adjacent, 0)} for the adjacent bin and ${pct(c.direction, 0)} for the direction.
    <b>Paper P&amp;L</b> is the fixed rule - long when the predicted bin is in the upper half, flat otherwise, minus a fee of ${pct(view.fee, 2)} per position - in
    units of the ${esc(meta.unit)} price (0.01 = 1%). Results settle the morning after, when the day's bar has closed.</p>`;

  // how a day becomes a draw - the newest settled day as the worked example
  const example = view.days.length ? view.days[0] : null;
  const half = view.k / 2;
  let worked = '';
  if (example) {
    const drawText = example.instruments.map((i) => (i.bin === null ? '?' : i.bin)).join(' ');
    const best = example.models.find((m) => m.name === example.best) || example.models[0] || null;
    worked += `<p style="margin:10px 0 6px;"><b>Worked example - ${esc(example.date)}, the newest settled day.</b></p>
      <div class="table-wrapper" style="margin-top:0;"><table style="min-width:0;"><tr><th style="text-align:left;">${esc(meta.noun)}</th><th>Return, close to close</th><th>Bin</th><th>The bin's interval that day</th></tr>
      ${example.instruments.map((i) => `<tr><td style="text-align:left; font-weight:bold;">${esc(i.symbol)}</td><td style="color:${(i.ret || 0) >= 0 ? '#27ae60' : '#c0392b'};">${signedPct(i.ret, 2)}</td><td><b>${i.bin === null ? '-' : i.bin}</b></td><td>${esc(intervalText(i.bin, i.edges))}</td></tr>`).join('')}
      </table></div>
      <p style="margin:8px 0 0;">So the draw of ${esc(example.date)} reads <b style="letter-spacing:2px;">${esc(drawText)}</b> on the <a href="${gameViewLink(market, example.date)}">game view</a>.`;
    if (best) {
      const cells = best.bins.map((b, i) => {
        const actual = example.instruments[i] ? example.instruments[i].bin : null;
        const hit = b !== null && actual !== null && b === actual;
        const near = !hit && b !== null && actual !== null && Math.abs(b - actual) === 1;
        return `<span style="display:inline-block; min-width:1.4em; text-align:center; padding:1px 4px; margin-right:2px; border-radius:3px; ${hit ? 'background:#2ecc71; color:white;' : (near ? 'background:#d5f5e3;' : 'background:#eee;')}">${b === null ? '?' : b}</span>`;
      }).join('');
      const settled = best.positions === null ? example.instruments.length : best.positions;
      const neighbours = (best.adjacent === null || best.exact === null) ? null : best.adjacent - best.exact;
      worked += ` ${esc(best.name)} had played ${cells} after the previous close - ${best.exact === null ? '-' : best.exact} of ${settled} ${esc(meta.noun)}s in the right bin (green), ${neighbours === null ? '-' : neighbours} more in a neighbouring bin (pale green), ${best.direction === null ? '-' : best.direction} on the right side of zero.`;
    }
    worked += '</p>';
  }
  html += `<div class="card expanded"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">How a day becomes a draw</span>
    <span class="card-meta" style="margin-left:10px;">what the digits on the game view mean here</span></div><div class="card-icon">▼</div></div>
    <div class="card-body"><p style="margin-top:0;">${esc(meta.calendar.replace(/ \(.*\)$/, '').replace(/^./, (c) => c.toUpperCase()))} is one draw with one slot per ${esc(meta.noun)}. A ${esc(meta.noun)}'s "number" is the <b>bin</b> of its
    return from the previous close to this close: its past returns are cut into ${view.k} equally likely tenths, bin 0 is a day among the worst tenth it has ever had,
    bin ${view.k - 1} among the best tenth, and bins ${half - 1} and ${half} sit around zero. The edges are fitted on that ${esc(meta.noun)}'s returns before the day, so each bin held
    1 in ${view.k} of that past - the chance level a model's exact rate is read against, over many days and beyond the null band the controls give it, never from a handful of days.
    A model's <b>ticket</b> is one bin per ${esc(meta.noun)}, made in the morning after the previous close (for crypto a few hours into the UTC day it predicts, for shares before
    New York opens - a Monday's ticket is made on the Saturday); a <b>hit</b> is the right bin in the right slot (never a set: the same digit on another ${esc(meta.noun)} means nothing),
    and the <b>direction</b> is whether the predicted bin lies on the same side of zero as the real return. A predicted bin is a return interval, which is why this page can draw it
    as a price on the chart and list the price band it stands for under it.${worked}
    <p style="color:#7f8c8d; font-size:0.85em; margin-bottom:0;">The <a href="/database/${esc(market)}">game view</a> shows these same digits in the lottery layout, scored the same way (the right bin in the right slot).${view.days.length ? ' The <i>Day by day</i> card below is that history translated back into returns.' : ''}</p></div></div>`;

  // models
  html += `<div class="card expanded"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Models over ${view.scoredDays} scored day(s)</span>
    <span class="card-meta" style="margin-left:10px;">accuracy against chance, paper P&amp;L of the fixed rule</span></div><div class="card-icon">▼</div></div>
    <div class="card-body"><div class="table-wrapper"><table>
    <tr><th style="text-align:left;">Model</th><th>Days</th><th title="predicted bin equals the actual bin; chance ${pct(c.exact, 0)}">Exact</th>
    <th title="within one bin; chance ${pct(c.adjacent, 0)}">Adjacent</th><th title="sign of the predicted return equals the real one; chance ${pct(c.direction, 0)}">Direction</th>
    <th title="positions taken (predicted bin in the upper half)">Trades</th><th title="sum of real returns of the positions taken, minus the fee">P&amp;L</th><th>P&amp;L / trade</th></tr>`;
  if (!view.models.length) html += `<tr><td colspan="8" style="color:#aaa;">no settled day yet - the first predictions settle tomorrow morning</td></tr>`;
  view.models.forEach((m) => {
    const mark = (above, text) => `<span style="${above ? 'color:#27ae60; font-weight:bold;' : ''}">${text}</span>`;
    html += `<tr><td style="text-align:left; font-weight:bold;">${esc(m.name)}</td><td>${m.days === null ? '-' : m.days}</td>
      <td>${mark(m.aboveExact, pct(m.exact))}</td><td>${mark(m.aboveAdjacent, pct(m.adjacent))}</td><td>${mark(m.aboveDirection, pct(m.direction))}</td>
      <td>${m.trades === null ? '-' : m.trades}</td><td style="color:${(m.pnlTotal || 0) >= 0 ? '#27ae60' : '#c0392b'};">${money(m.pnlTotal)}</td><td>${money(m.pnlPerTrade, 5)}</td></tr>`;
  });
  html += `</table></div><p style="color:#7f8c8d; font-size:0.85em;">Green: above chance. With ${view.instruments.length} ${esc(meta.noun)}s a day and a
    handful of days, a rate above chance is noise more often than not; read the rows over months, and against the null band the controls give the lottery rows.</p></div></div>`;

  // daily accuracy
  if (view.daily.length) {
    html += `<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Accuracy per day</span>
      <span class="card-meta" style="margin-left:10px;">mean exact rate over every model, the best model's, and chance</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><div style="height:220px;"><canvas id="daily-${esc(market)}"></canvas></div>
      <script>new Chart(document.getElementById('daily-${esc(market)}').getContext('2d'), { type: 'line', data: {
        labels: ${JSON.stringify(view.daily.map((d) => d.date))},
        datasets: [
          { label: 'mean exact rate', data: ${JSON.stringify(view.daily.map((d) => num(d.exact_mean)))}, borderColor: '#2980b9', tension: 0.2, pointRadius: 2 },
          { label: 'best model that day', data: ${JSON.stringify(view.daily.map((d) => num(d.best_exact)))}, borderColor: '#27ae60', tension: 0.2, pointRadius: 2 },
          { label: 'chance', data: ${JSON.stringify(view.daily.map(() => c.exact))}, borderColor: '#95a5a6', borderDash: [4, 4], pointRadius: 0 }
        ] }, options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 1 } } } });</script></div></div>`;
  }

  // day by day: the settled days in market terms, each expandable to its models
  if (view.days.length) {
    const dayBlocks = view.days.map((d) => {
      const symbols = d.instruments.map((i) => i.symbol);     // a day's own slots: the instrument set can change between days
      const instrumentCells = d.instruments.map((i) => `<span style="display:inline-block; min-width:110px; margin-right:8px;"><b>${esc(i.symbol)}</b>
        <span style="color:${(i.ret || 0) >= 0 ? '#27ae60' : '#c0392b'};">${signedPct(i.ret)}</span> <span style="color:#7f8c8d;">bin ${i.bin === null ? '-' : i.bin}</span></span>`).join('');
      const modelRows = d.models.map((m) => {
        const cells = m.bins.map((b, i) => {
          const inst = d.instruments[i];
          const actual = inst ? inst.bin : null;
          const hit = b !== null && actual !== null && b === actual;
          const near = !hit && b !== null && actual !== null && Math.abs(b - actual) === 1;
          const arrow = b === null ? '' : (b >= half ? '<span style="color:#27ae60;">&#9650;</span>' : '<span style="color:#c0392b;">&#9660;</span>');
          return `<td style="${hit ? 'background:#2ecc71; color:white;' : (near ? 'background:#d5f5e3;' : '')}" title="${inst ? esc(intervalText(b, inst.edges)) : ''}${b === null ? '' : (b >= half ? ' - rule: long' : ' - rule: flat')}">${b === null ? '-' : b} ${arrow}</td>`;
        }).join('');
        return `<tr><td style="text-align:left; font-weight:bold;">${esc(m.name)}</td>${cells}<td>${m.exact === null ? '-' : m.exact}/${m.positions === null ? '-' : m.positions}</td>
          <td>${m.direction === null ? '-' : m.direction}/${m.positions === null ? '-' : m.positions}</td><td style="color:${(m.pnl || 0) >= 0 ? '#27ae60' : '#c0392b'};">${money(m.pnl)}</td></tr>`;
      }).join('');
      return `<details style="border-bottom:1px solid #eee; padding:8px 0;"><summary style="cursor:pointer; display:flex; flex-wrap:wrap; align-items:center; gap:6px;">
        <b style="min-width:100px;">${esc(d.date)}</b> ${instrumentCells}
        <span style="margin-left:auto; color:#7f8c8d; font-size:0.9em;">best ${esc(d.best || '-')} · mean exact ${pct(d.exactMean, 0)} · <a href="${gameViewLink(market, d.date)}">game view</a></span></summary>
        <div class="table-wrapper" style="margin-top:8px;"><table><tr><th style="text-align:left;">Model (ticket made after the previous close)</th>${symbols.map((sym) => `<th>${esc(sym)}</th>`).join('')}<th title="right bin in the right slot">Exact</th><th title="predicted bin on the same side of zero as the real return">Direction</th><th title="fixed rule: long when the bin is in the upper half, minus the fee">P&amp;L</th></tr>${modelRows}</table></div></details>`;
    }).join('');
    html += `<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Day by day</span>
      <span class="card-meta" style="margin-left:10px;">the newest ${view.days.length} settled day(s) as returns and bins; open a day for every model's ticket</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><p style="color:#7f8c8d; font-size:0.9em; margin-top:0;">Each line is one draw: per ${esc(meta.noun)} the real close-to-close return and the bin it fell in. Inside, every model's bin per ${esc(meta.noun)}
      with the position the fixed P&amp;L rule takes on it (&#9650; long, bin in the upper half; &#9660; flat, lower half) - the <i>Direction</i> column is scored on the sign of the bin's own return,
      which is read off the hover interval; green is the right bin, pale green one bin off. Chance is 1 in ${view.k} per ${esc(meta.noun)}.</p>${dayBlocks}</div></div>`;
  }

  // the rows under a proper score (the weekly MarketRows.py report)
  if (view.rows) {
    const rep = view.rows;
    const signed = (x, d = 3) => (num(x) === null ? '-' : (num(x) >= 0 ? '+' : '') + num(x).toFixed(d));
    const verdictMark = (iv) => {
      if (!iv) return '<span style="color:#aaa;">-</span>';
      const colour = iv.verdict === 'better' ? '#27ae60' : (iv.verdict === 'worse' ? '#c0392b' : '#7f8c8d');
      return `<span style="color:${colour};${iv.verdict === 'better' ? ' font-weight:bold;' : ''}" title="mean difference in log-score per day, 95% paired bootstrap interval over ${iv.days === null ? '?' : iv.days} days">${signed(iv.mean)} [${signed(iv.lo)}, ${signed(iv.hi)}] ${esc(iv.verdict)}</span>`;
    };
    const kindNote = { market: 'market row', base: 'lottery row', baseline: 'baseline' };
    let table = '';
    rep.rows.forEach((r) => {
      table += `<tr><td style="text-align:left; font-weight:bold;">${esc(r.name)}${r.isReference ? ' <span style="color:#7f8c8d; font-weight:normal;">(reference)</span>' : ''}<br><span style="color:#aaa; font-size:0.8em;">${esc(kindNote[r.kind] || r.kind)}</span></td>
        <td>${r.logScore === null ? '<span style="color:#aaa;">no probabilities</span>' : `<b>${r.logScore.toFixed(3)}</b>${r.se === null ? '' : ` <span style="color:#aaa;">± ${r.se.toFixed(3)}</span>`}`}</td>
        <td>${r.isReference ? '<span style="color:#7f8c8d;">reference</span>' : verdictMark(r.vsReference)}</td><td>${verdictMark(r.vsUniform)}</td>
        <td>${pct(r.exact)}</td><td>${pct(r.adjacent)}</td><td>${pct(r.direction)}</td><td>${r.trades === null ? '-' : r.trades}</td>
        <td style="color:${(r.pnl || 0) >= 0 ? '#27ae60' : '#c0392b'};">${money(r.pnl)}</td></tr>`;
    });
    const headline = rep.betterThanReference.length
      ? `<b style="color:#27ae60;">${esc(rep.betterThanReference.join(', '))}</b> carr${rep.betterThanReference.length === 1 ? 'ies' : 'y'} information beyond GARCH over this window (the interval's lower bound is above zero).`
      : `<b>No row carries information beyond GARCH</b> over this window: every interval against the reference includes zero or lies below it.`;
    const garchLine = rep.referenceRow
      ? (rep.referenceAboveUniform ? `GARCH itself is above the uniform forecast (${rep.referenceRow.logScore.toFixed(3)} against ${rep.uniform.toFixed(3)}), which is volatility clustering - the known predictable component.`
        : `GARCH itself is not distinguishable from the uniform forecast here (${rep.referenceRow.logScore === null ? '-' : rep.referenceRow.logScore.toFixed(3)} against ${rep.uniform.toFixed(3)}).`)
      : '';
    html += `<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Rows under a proper score</span>
      <span class="card-meta" style="margin-left:10px;">mean log-score of the probability each row gave the bin that happened, ${rep.days === null ? '?' : rep.days} game days${rep.firstDay ? ` ${esc(rep.firstDay)} to ${esc(rep.lastDay || '')}` : ''}, against GARCH</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><p style="color:#7f8c8d; font-size:0.9em;">${headline} ${garchLine} A row is judged here before its hit rates or paper P&amp;L are read: the log-score rewards honest
      probabilities and punishes overconfidence, so a row cannot win it by betting on the same bin every day. The uniform forecast scores ${rep.uniform === null ? '-' : rep.uniform.toFixed(3)}
      (log of 1/${view.k}); a probability under ${rep.floor === null ? '-' : rep.floor} is scored as ${rep.floor === null ? '-' : rep.floor}.${rep.lockboxWithheld ? ` ${rep.lockboxWithheld} lockbox day(s) withheld.` : ''}</p>
      <div class="table-wrapper"><table><tr><th style="text-align:left;">Row</th><th title="mean over days of the mean over instruments of log p(actual bin); higher is better">Log-score</th>
      <th title="difference to the GARCH row, 95% paired bootstrap interval over days">vs GARCH</th><th title="difference to the uniform forecast">vs uniform</th><th>Exact</th><th>Adjacent</th><th>Direction</th><th>Trades</th><th title="fixed rule with the real returns and the fee, in units of price (0.01 = 1%)">P&amp;L</th></tr>${table}</table></div>
      <p style="color:#7f8c8d; font-size:0.85em;">Walk-forward, every day refitted on the past only (MarketRows.py, Sundays). "Better" and "worse" are read off the interval, not the point estimate. The two ablation rows
      say what the Regime HMM's regimes are made of: if the full row is not better than <i>ZeroMean</i>, the regimes carry variance only; if it is not better than <i>Single</i>, there are no regimes worth the name.${Object.keys(rep.errors).length ? ` Rows that failed on some days: ${esc(Object.keys(rep.errors).join(', '))}.` : ''}
      Generated ${esc(rep.generatedAt || '-')}.</p></div></div>`;
  }

  // the regime reading (the Regime HMM rows' templates)
  if (view.regimes.length) {
    let table = '';
    view.regimes.forEach((r) => {
      table += `<tr><td style="text-align:left; font-weight:bold;">${esc(r.row)}</td><td>${esc(r.date || '-')}</td><td>${r.regimes === null ? '-' : r.regimes}</td>
        <td>${r.template === null ? '-' : `T${r.template}`}${r.label ? ` <span style="color:#7f8c8d;">(${esc(r.label)}${r.volatilityRank !== null && r.regimes !== null && r.regimes > 1 ? `, ${r.volatilityRank} of ${r.regimes} by volatility` : ''})</span>` : ''}</td>
        <td>${pct(r.probability, 0)}</td><td style="text-align:left; font-size:0.85em;">${r.expected.map((e) => `${esc(e.symbol)} ${e.value === null ? '-' : (e.value * 100).toFixed(2) + '%'}`).join(', ')}</td></tr>`;
    });
    html += `<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Regime reading</span>
      <span class="card-meta" style="margin-left:10px;">which regime the Regime HMM rows believe the market is in, after the newest game day</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><div class="table-wrapper"><table><tr><th style="text-align:left;">Row</th><th>After</th><th>Regimes</th><th title="a persistent template matched across the daily refits by the Wasserstein distance between the regime Gaussians; calm / normal / turbulent by the regime's volatility against the market's">Template</th><th>Probability</th><th style="text-align:left;">Expected next-day return</th></tr>${table}</table></div>
      <p style="color:#7f8c8d; font-size:0.85em;">A regime is a Gaussian over the ${esc(meta.noun)}s' returns and their recent volatility and momentum; the label is the volatility of the dominant regime against the market's, the template number keeps its identity across refits.
      The expected return is the mixture's mean per ${esc(meta.noun)} - a reading, not a recommendation.</p></div></div>`;
  }

  // instruments
  view.instruments.forEach((inst) => {
    const id = `chart-${esc(market)}-${esc(inst.symbol)}`;
    const labels = inst.closes.map((p) => p[0]);
    const nextLabel = 'next';
    const datasets = [{ label: `${inst.symbol} close`, data: inst.closes.map((p) => p[1]), borderColor: '#2c3e50', tension: 0.1, pointRadius: 0, borderWidth: 2 }];
    view.drawn.forEach((model, i) => {
      const points = inst.course[model] || [];
      const byDate = Object.fromEntries(points.map((p) => [p[0], p[1]]));
      const nextPoint = inst.next.find((p) => p.model === model);
      datasets.push({ label: `${model} predicted`, borderColor: COLOURS[i % COLOURS.length], tension: 0.1, pointRadius: 2, borderWidth: 1, borderDash: [3, 3],
        data: labels.map((d) => (byDate[d] === undefined ? null : byDate[d])).concat([nextPoint ? nextPoint.price : null]) });
    });
    const chartLabels = labels.concat([nextLabel]);
    datasets[0].data = datasets[0].data.concat([null]);
    const nextRows = inst.next.map((p) => `<tr><td style="text-align:left;">${esc(p.model)}</td><td>${p.bin === null ? '-' : p.bin}</td>
      <td>${p.direction === 1 ? '<span style="color:#27ae60;">up</span>' : (p.direction === -1 ? '<span style="color:#c0392b;">down</span>' : 'flat')}</td>
      <td>${price(p.price)}</td><td>${p.low === null ? 'below ' + price(p.high) : (p.high === null ? 'above ' + price(p.low) : `${price(p.low)} - ${price(p.high)}`)}</td></tr>`).join('');
    html += `<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">${esc(inst.symbol)} - ${esc(inst.name)}</span>
      <span class="card-meta" style="margin-left:10px;">last close ${price(inst.lastClose)} ${esc(inst.quote)} on ${esc(inst.lastDate || '-')}${inst.active ? '' : ' (inactive)'}</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><div style="height:260px;"><canvas id="${id}"></canvas></div>
      <script>new Chart(document.getElementById('${id}').getContext('2d'), { type: 'line', data: { labels: ${JSON.stringify(chartLabels)}, datasets: ${JSON.stringify(datasets)} },
        options: { maintainAspectRatio: false, spanGaps: true, plugins: { legend: { labels: { boxWidth: 12 } } }, scales: { x: { ticks: { maxTicksLimit: 8 } } } } });</script>
      <p style="color:#7f8c8d; font-size:0.85em;">The dashed lines are the price each model's predicted bin stood for, day by day (previous close moved by the bin's
        middle return); the last dashed point is the prediction for the next trading day${view.madeOn ? `, made after ${esc(view.madeOn)}` : ''}.</p>
      ${nextRows ? `<div class="table-wrapper"><table><tr><th style="text-align:left;">Next day, per model</th><th>Bin</th><th>Direction</th><th>Price it stands for</th><th>Interval</th></tr>${nextRows}</table></div>`
        : '<p style="color:#aaa;">no prediction for the next day yet</p>'}
      </div></div>`;
  });

  html += `<p style="color:#7f8c8d; font-size:0.85em; margin-top:20px;">Record generated ${esc(view.generatedAt || '-')}; newest game day ${esc(view.newestGameDay || '-')}.
    The same rows are tracked, slot by slot, on the <a href="/database/${esc(market)}">game view</a> like every lottery game.</p>`;
  return html + footer();
}

function install(app, { header, footer, dataDir, controlsDir }) {
  Object.keys(MARKETS).forEach((market) => {
    app.get(`/markets/${market}`, (req, res) => {
      const record = loadMarket(dataDir, market);
      const extras = { rows: controlsDir ? loadRows(controlsDir, market) : null, regimes: loadRegimes(dataDir, market) };
      res.send(page(market, record ? describeMarket(record, extras) : null, header, footer, req.user));
    });
  });
}

module.exports = { MARKETS, loadMarket, loadRows, loadRegimes, describeMarket, describeRows, describeRegimes, describeDays, binInterval, intervalText, gameViewLink,
  signedPct, page, install, pct, money, price };
