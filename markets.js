// markets.js - the Crypto and Shares pages (README roadmap item 4, phase M2).
//
// Predictor.py tracks the two markets as positional games, and after each
// run src/MarketSettle.py settles every stored day against the real returns
// and writes data/markets/<market>.json: closes, the predicted course per
// model, the next day's predictions as prices, per-model accuracy against
// chance and the daily accuracy series. These pages draw that file and
// nothing else - no database, no Python at request time.
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

// One view model per market: models with their rates read against chance,
// instruments with what the charts need, the next-day table, the series.
function describeMarket(record) {
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
  const instruments = (record.instruments || []).map((i) => {
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
  };
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
    predicts one bin per ${esc(meta.noun)}, ${esc(meta.calendar)}. A bin is a return interval, so it is drawn as a predicted price with a band.
    Chance is ${pct(c.exact, 0)} for the exact bin, ${pct(c.adjacent, 0)} for the adjacent bin and ${pct(c.direction, 0)} for the direction.
    <b>Paper P&amp;L</b> is the fixed rule - long when the predicted bin is in the upper half, flat otherwise, minus a fee of ${pct(view.fee, 2)} per position - in
    units of the ${esc(meta.unit)} price (0.01 = 1%). Results settle the morning after, when the day's bar has closed.</p>`;

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
    The same rows are tracked on the <a href="/database/${esc(market)}">History</a> page like every game.</p>`;
  return html + footer();
}

function install(app, { header, footer, dataDir }) {
  Object.keys(MARKETS).forEach((market) => {
    app.get(`/markets/${market}`, (req, res) => {
      const record = loadMarket(dataDir, market);
      res.send(page(market, record ? describeMarket(record) : null, header, footer, req.user));
    });
  });
}

module.exports = { MARKETS, loadMarket, describeMarket, page, install, pct, money, price };
