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
function intervalText(bin, edges, digits = 1) {
  const [low, high] = binInterval(bin, edges);
  if (low === null && high === null) return '-';
  if (low === null) return `below ${signedPct(high, digits)}`;
  if (high === null) return `above ${signedPct(low, digits)}`;
  return `${signedPct(low, digits)} to ${signedPct(high, digits)}`;
}
// The bin a move would fall in under a day's edges (MarketGame.bin_of).
const binOfMove = (move, edges) => edges.filter((e) => e <= move).length;
// The day file behind a game day, as Predictor.py names it (no zero padding).
function gameViewLink(market, date) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(date));
  return m ? `/database/${market}/${m[1]}-${Number(m[2])}-${Number(m[3])}.json` : `/database/${market}`;
}

// --- page -------------------------------------------------------------------
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const COLOURS = ['#e67e22', '#8e44ad', '#16a085', '#c0392b', '#2980b9'];

// The instrument charts zoom (wheel, pinch) and pan (drag) through
// chartjs-plugin-zoom, loaded on these pages only; hammer.js gives it touch
// gestures. Range buttons set the category axis' min/max by label, which
// needs no plugin, and "Reset" clears both. The market pages take a wider
// container than the lottery pages so the charts get room; the explainer's
// text keeps a readable line length of its own.
const CHART_SCRIPTS = `<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"></script>
<style>.container { max-width: 1400px; } .chart-tools { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:8px; font-size:0.85em; }
.chart-tools button { padding:4px 10px; border:1px solid #ccd1d6; border-radius:4px; background:white; color:#2c3e50; cursor:pointer; } .chart-tools button:hover { background:#f1f3f5; }
.chart-tools .hint { color:#7f8c8d; margin-left:auto; }</style>
<script>
  try { if (window.ChartZoom) Chart.register(window.ChartZoom); } catch (e) { /* zoom stays off, the chart still draws */ }
  window.marketCharts = window.marketCharts || {};
  // the newest \`days\` labels (the trailing 'next' label included); 0 = everything
  function marketRange(id, days) {
    const chart = window.marketCharts[id]; if (!chart) return;
    const labels = chart.data.labels; const x = chart.options.scales.x;
    if (typeof chart.resetZoom === 'function') chart.resetZoom('none');
    if (days > 0 && labels.length > days) { x.min = labels[labels.length - 1 - days]; x.max = labels[labels.length - 1]; } else { delete x.min; delete x.max; }
    chart.update();
  }
  function marketReset(id) { marketRange(id, 0); }
</script>`;
const ZOOM_OPTIONS = { zoom: { wheel: { enabled: true }, pinch: { enabled: true }, drag: { enabled: false }, mode: 'x' }, pan: { enabled: true, mode: 'x' } };
const RANGES = [['1M', 30], ['3M', 91], ['6M', 182], ['1Y', 365], ['All', 0]];

function page(market, view, header, footer, user) {
  const meta = MARKETS[market];
  let html = header(meta.title, user);
  html += CHART_SCRIPTS;
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

  // how a day becomes a draw - for a newcomer, in three steps, with the newest
  // settled day as the worked example and a strip per instrument showing what
  // a bin looks like. Wording reviewed by two reader personas on 1 Oct 2026.
  const example = view.days.length ? view.days[0] : null;
  const half = view.k / 2;
  const first = example && example.instruments.length ? example.instruments[0] : null;
  const closeBefore = (symbol, date) => {
    const inst = view.instruments.find((i) => i.symbol === symbol);
    if (!inst) return null;
    const idx = inst.closes.findIndex((p) => p[0] === date);
    return idx > 0 ? inst.closes[idx - 1][1] : null;
  };
  const closeOn = (symbol, date) => {
    const inst = view.instruments.find((i) => i.symbol === symbol);
    const point = inst ? inst.closes.find((p) => p[0] === date) : null;
    return point ? point[1] : null;
  };
  const candle = first ? { before: closeBefore(first.symbol, example.date), on: closeOn(first.symbol, example.date) } : null;
  const best = example ? (example.models.find((m) => m.name === example.best) || example.models[0] || null) : null;
  const dayWord = meta.calendar.replace(/ \(.*\)$/, '').replace(/^every /, '');      // "day" / "trading day"
  const nouns = `${esc(meta.noun)}s`;
  const closeNote = market === 'crypto'
    ? 'a crypto day runs midnight to midnight UTC, 02:00 Belgian time in summer and 01:00 in winter, so the ticket is placed seven to eight hours into the day it predicts; none of those hours is used, because the models read closed daily candles only, and the day\'s close is still unknown'
    : 'the New York session closes at 22:00 Belgian time in summer and 21:00 in winter, so a trading day\'s ticket is placed on its own morning, before New York opens, and a Monday\'s ticket is made on the Saturday';

  // step 1: two closes, one number
  let step1 = `<p><b>1. Two closes, one number.</b> On a price chart each ${esc(dayWord)} is drawn as one candle, and a candle is four prices: where the price opened, the highest
    and lowest it reached, and where it closed. Only the last one, the <b>close</b>, is used here. The ${esc(dayWord)}'s <b>move</b> is how much the close changed since the close before it, in percent`;
  if (candle && candle.before !== null && candle.on !== null) {
    const diff = candle.on - candle.before;
    // the difference in the close's own precision (price() would print 337 as 337.00)
    const diffText = candle.before >= 1000 ? Math.round(Math.abs(diff)).toString() : (candle.before >= 10 ? Math.abs(diff).toFixed(2) : Math.abs(diff).toFixed(4));
    step1 += `: ${esc(first.symbol)}'s previous close was ${price(candle.before)}; on ${esc(example.date)} it closed at ${price(candle.on)}, ${diffText} ${diff < 0 ? 'lower' : 'higher'}, a move of <b>${signedPct(first.ret, 2)}</b>`;
  }
  step1 += `. Nothing that happened within the ${esc(dayWord)} is used. Each new candle therefore adds exactly one number to the ${esc(meta.noun)}'s record: the move from the close before it to its own close.</p>`;

  // step 2: the number goes into one of ten bins
  let compare = '';
  if (example && example.instruments.length > 1) {
    const two = Math.log(1.02);
    const placed = example.instruments.filter((i) => i.edges.length === view.k - 1).map((i) => ({ symbol: i.symbol, bin: binOfMove(two, i.edges) }));
    if (placed.length > 1) {
      const top = placed.reduce((a, b) => (b.bin > a.bin ? b : a));
      const bottom = placed.reduce((a, b) => (b.bin < a.bin ? b : a));
      if (top.bin !== bottom.bin) {
        compare = ` Each ${esc(meta.noun)} has its own bins because the same move is a different event on each: on the strips below a +2% ${esc(dayWord)} lands in ${esc(top.symbol)}'s bin ${top.bin} but only in ${esc(bottom.symbol)}'s bin ${bottom.bin}.`;
      }
    }
  }
  if (!compare) compare = ` Each ${esc(meta.noun)} has its own bins, because the same move is a different event on each - ordinary for a volatile one, large for a calm one.`;
  const step2 = `<p><b>2. The number goes into one of ${view.k} bins.</b> A bin is a <b>range of daily moves</b>, and every ${esc(meta.noun)} has its own ${view.k} ranges, cut from its own past:
    take every daily move the ${esc(meta.noun)} has had on the days all the ${nouns} of this game traded side by side - years of them - sort them from the biggest fall to the biggest rise, and
    cut the sorted list into ${view.k} equal piles. Each pile is one bin: bin 0 holds the ${esc(meta.noun)}'s worst tenth of days, bin ${view.k - 1} its best tenth, bins ${half - 1} and ${half} the days that
    barely moved. The ${view.k - 1} cut points are the <b>bin edges</b>. They shift a little as days are added, and for any given day they are cut from the days before it only, so a day never
    helps draw the lines that put it in its bin. Every bin therefore holds about one day in ${view.k}, like one face of a ${view.k}-sided die: a model that picks a bin blindly lands in the
    right one about one time in ${view.k}, and that is the bar every model has to beat.${compare}</p>`;

  // the strips: what a bin looks like, per instrument, with the real move and the best ticket
  let strips = '';
  if (example) {
    const rows = example.instruments.map((inst, slot) => {
      const predicted = best && best.bins[slot] !== undefined ? best.bins[slot] : null;
      const cells = Array.from({ length: view.k }, (_, b) => {
        const actual = inst.bin === b;
        const played = predicted === b;
        const style = actual ? 'background:#2c3e50; color:white; font-weight:bold;' : (played ? 'background:#d5f5e3;' : 'background:white;');
        const border = played ? 'border:2px solid #27ae60;' : 'border:1px solid #ccd1d6;';
        const title = `bin ${b}: ${intervalText(b, inst.edges, 2)}${actual ? ' - the day fell here' : ''}${played ? ` - ${best.name} predicted this` : ''}`;
        return `<div style="flex:1; min-width:0; text-align:center; padding:3px 0; font-size:0.8em; ${style} ${border}" title="${esc(title)}">${b}</div>`;
      }).join('');
      const labels = Array.from({ length: view.k }, (_, b) => `<div style="flex:1; min-width:0; text-align:right; font-size:0.65em; color:#7f8c8d; transform:translateX(50%); white-space:nowrap;">${b < inst.edges.length ? esc(signedPct(inst.edges[b])) : ''}</div>`).join('');
      return `<div style="display:flex; align-items:flex-start; gap:10px; margin:6px 0 14px;">
        <div style="min-width:150px; font-size:0.9em; padding-top:3px;"><b>${esc(inst.symbol)}</b> <span style="color:${(inst.ret || 0) >= 0 ? '#27ae60' : '#c0392b'};">${signedPct(inst.ret, 2)}</span> &rarr; bin <b>${inst.bin === null ? '-' : inst.bin}</b></div>
        <div style="flex:1; min-width:0;"><div style="display:flex; gap:1px;">${cells}</div><div style="display:flex; gap:1px; margin-right:0;">${labels}</div></div></div>`;
    }).join('');
    const ends = first && first.edges.length === view.k - 1
      ? ` - ${esc(first.symbol)}'s bin ${half - 1} runs from ${esc(signedPct(first.edges[half - 2]))} to ${esc(signedPct(first.edges[half - 1]))}, while bin 0 is everything below ${esc(signedPct(first.edges[0]))} and bin ${view.k - 1} everything above ${esc(signedPct(first.edges[view.k - 2]))}, so the two end boxes have no outer edge`
      : '';
    strips = `<p style="margin:14px 0 2px;"><b>Worked example - ${esc(example.date)}, the newest settled day</b> (settled: its closes are in and every model's prediction for it has been scored against them).
      Each strip is one ${esc(meta.noun)}'s ${view.k} bins, with the cut point between two bins written under their boundary. The boxes are drawn the same width because each holds a tenth of
      the ${esc(meta.noun)}'s past days, not because the ranges are equal${ends}. The dark box is the bin the day's move fell in${best ? `; the green outline is the bin that ${esc(best.name)} - one of the
      models, the one with the most exact hits that day - had predicted beforehand. That prediction is its ticket, explained in step 3` : ''}.</p>${rows}`;
  } else {
    strips = `<p style="color:#7f8c8d; font-size:0.9em;">The worked example - the newest day's moves on the ${view.k} bins, one strip per ${esc(meta.noun)} - appears here after the first settled day.</p>`;
  }

  // step 3: the draw, the ticket, the hit
  let step3 = `<p><b>3. The bins side by side are the draw.</b> Every ${esc(dayWord)} is one draw with one slot per ${esc(meta.noun)}`;
  if (example) {
    const drawText = example.instruments.map((i) => (i.bin === null ? '?' : i.bin)).join(' ');
    step3 += `: the draw of ${esc(example.date)} reads <b style="letter-spacing:2px;">${esc(drawText)}</b>, which is what the <a href="${gameViewLink(market, example.date)}">game view</a> shows as digits`;
  }
  step3 += `. A model's <b>ticket</b> is one bin per ${esc(meta.noun)}, made at 09:00 Belgian time after the previous candle has closed (${closeNote})`;
  if (best) {
    const cells = best.bins.map((b, i) => {
      const actual = example.instruments[i] ? example.instruments[i].bin : null;
      const hit = b !== null && actual !== null && b === actual;
      const near = !hit && b !== null && actual !== null && Math.abs(b - actual) === 1;
      return `<span style="display:inline-block; min-width:1.4em; text-align:center; padding:1px 4px; margin-right:2px; border-radius:3px; ${hit ? 'background:#2ecc71; color:white;' : (near ? 'background:#d5f5e3;' : 'background:#eee;')}">${b === null ? '?' : b}</span>`;
    }).join('');
    const settled = best.positions === null ? example.instruments.length : best.positions;
    const neighbours = (best.adjacent === null || best.exact === null) ? null : best.adjacent - best.exact;
    step3 += `. ${esc(best.name)} had played ${cells} after the previous close - ${best.exact === null ? '-' : best.exact} of ${settled} ${nouns} in the right bin (green), ${neighbours === null ? '-' : neighbours} more in a neighbouring bin (pale green), ${best.direction === null ? '-' : best.direction} on the right side of zero`;
  }
  const two = example && example.instruments.length > 1 ? example.instruments.slice(0, 2).map((i) => i.symbol) : [`one ${esc(meta.noun)}`, 'another'];
  step3 += `. A <b>hit</b> is the right bin for the right ${esc(meta.noun)}, nothing else: a 7 played on ${esc(two[0])} is judged against ${esc(two[0])}'s own bin, and ${esc(two[1])} landing in 7 does not help it.
    One bin off counts as <b>adjacent</b> (nearly right). <b>Direction</b> is whether the middle of the predicted bin has the same sign as the real move - up or down: that matters for the bin
    that straddles zero (bin ${half - 1} on most days), which counts as down when its middle is below zero, and a day that closes exactly unchanged is a miss for every model.
    Chance is 1 in ${view.k} for the exact bin, ${pct(c.adjacent, 0)} for adjacent (a blind guess spread over all ${view.k} bins: an inner bin is within one by luck ${Math.round(300 / view.k)}% of the time, an end bin ${Math.round(200 / view.k)}%)
    and about ${pct(c.direction, 0)} for direction.</p>`;

  // why bins, and that a bin is still a price range
  let why = `<p><b>Why bins and not the price itself?</b> A price is never hit exactly, so "right" would need a tolerance, say &plusmn;1% - and with a fixed band a model that always says
    "unchanged" scores a hit on every day that moves less than the band. That score would measure how calm the ${esc(meta.noun)} was, not the model, and the same &plusmn;1% would mean one thing
    for a volatile ${esc(meta.noun)} and another for a calm one. With each ${esc(meta.noun)}'s own bins any answer - always bin ${half - 1}, always bin ${view.k - 1} - is right by luck about one time in ${view.k},
    because every bin has held one day in ${view.k} of its past. About, not exactly: the edges come from the past and the day being scored is a new one, so in a calm stretch the middle bins
    come up more often than one in ${view.k}, in a wild stretch the outer ones. That tilt - how big the move will be, not which way - is the one thing about a day's move known to be somewhat
    predictable, and the GARCH row, a model that forecasts only that, sits in the table to show how much of a score is just that tilt. So the question this project asks - do these models beat
    luck, and beat a plain volatility model? - has a clean answer, read against the same controls as the lottery games.</p>
    <p style="margin-bottom:0;"><b>A bin is still a price range</b>`;
  if (first && candle && candle.before !== null && first.bin !== null && first.edges.length) {
    const [lo, hi] = binInterval(first.bin, first.edges);
    const low = lo === null ? null : candle.before * Math.exp(lo);
    const high = hi === null ? null : candle.before * Math.exp(hi);
    const straddles = lo !== null && hi !== null && lo < 0 && hi > 0;
    const middle = straddles ? (lo + hi) / 2 : null;
    why += `: ${esc(first.symbol)}'s bin ${first.bin} on ${esc(example.date)} ran from ${esc(intervalText(first.bin, first.edges, 2))}${straddles ? ` (this bin straddles zero; its middle sits a hair ${middle < 0 ? 'below' : 'above'} zero, so for direction it counts as ${middle < 0 ? 'down' : 'up'})` : ''}, which from the close of ${price(candle.before)} means a close
      ${low === null ? `below about ${price(high)}` : (high === null ? `above about ${price(low)}` : `between about ${price(low)} and ${price(high)}`)}`;
  }
  why += `. That is why the charts below can draw a predicted bin as a price, and the table under each chart lists the band.</p>`;

  html += `<div class="card expanded"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">How a day becomes a draw</span>
    <span class="card-meta" style="margin-left:10px;">read this first: what a bin is, what a ticket is, what a hit is</span></div><div class="card-icon">▼</div></div>
    <div class="card-body" style="max-width:1080px;">${step1}${step2}${strips}${step3}${why}
    <p style="color:#7f8c8d; font-size:0.85em; margin:10px 0 0;">Strictly, every move and edge on this page is a log return - the kind that adds up day over day - rather than a chart's plain percentage change;
    for ordinary days the two are the same number, at the &plusmn;3% edges they differ by a few hundredths of a percent, at &plusmn;6% by about two tenths.
    The <a href="/database/${esc(market)}">game view</a> shows these same digits in the lottery layout, scored the same way (the right bin for the right ${esc(meta.noun)}).${view.days.length ? ' The <i>Day by day</i> card below is that history translated back into moves.' : ''}</p></div></div>`;

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
      <div class="card-body"><div class="chart-tools">${RANGES.map(([label, days]) => `<button type="button" onclick="marketRange('${id}', ${days})">${label}</button>`).join('')}
        <button type="button" onclick="marketReset('${id}')" style="margin-left:6px;">Reset zoom</button>
        <span class="hint">scroll or pinch to zoom, drag to pan</span></div>
      <div style="height:420px;"><canvas id="${id}"></canvas></div>
      <script>window.marketCharts['${id}'] = new Chart(document.getElementById('${id}').getContext('2d'), { type: 'line', data: { labels: ${JSON.stringify(chartLabels)}, datasets: ${JSON.stringify(datasets)} },
        options: { maintainAspectRatio: false, spanGaps: true, interaction: { mode: 'index', intersect: false },
          plugins: { legend: { labels: { boxWidth: 12 } }, zoom: ${JSON.stringify(ZOOM_OPTIONS)} }, scales: { x: { ticks: { maxTicksLimit: 10 } } } } });</script>
      <p style="color:#7f8c8d; font-size:0.85em;">A year of closes${view.drawn.length ? '; the dashed lines are the price each model\'s predicted bin stood for, day by day (previous close moved by the bin\'s middle return), and the last dashed point is the prediction for the next trading day' : ''}${view.madeOn ? `, made after ${esc(view.madeOn)}` : ''}.
        Scroll on the chart or pinch to zoom in on a period, drag to move along it, or use the buttons; the predicted lines start where the tracking started.</p>
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

module.exports = { MARKETS, loadMarket, loadRows, loadRegimes, describeMarket, describeRows, describeRegimes, describeDays, binInterval, intervalText, binOfMove, gameViewLink,
  signedPct, page, install, pct, money, price };
