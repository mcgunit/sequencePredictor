const express = require('express');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

const config = require("./config");

const app = express();

// Middleware to parse form data and JSON
app.use(express.urlencoded({ extended: true }));
app.use(express.json()); 

// Paths
const dataPath = path.join(__dirname, 'data', 'database');
// modelsPath removed as it is no longer used

// --- GLOBAL STATE ---
var selectedPlayedNumbers = [4, 5, 6, 7, 8, 9, 10]; // Default for Keno
var selectedModel = ["all"]; // Global filter for which models to show/calculate

// --- GAME SHAPES ---
// Mirrors Predictor.py's SPECIAL_COLUMN_COUNTS: how many trailing values of a
// full result/ticket row are special numbers (euromillions stars, eurodreams
// dream number, vikinglotto viking). A main-ball hit and a special-ball hit
// are different prize dimensions, so the UI must never pool them.
const SPECIAL_COLUMN_COUNTS = { euromillions: 2, eurodreams: 1, vikinglotto: 1 };

// Database folder names equal game names today, but the routes historically
// matched with includes() (e.g. a "keno_backup" folder still behaves as keno),
// so keep that tolerance. vikinglotto must be tested before lotto because
// "vikinglotto".includes("lotto") is true.
function gameFromFolder(folder) {
  const games = ["euromillions", "eurodreams", "vikinglotto", "lotto", "keno", "pick3"];
  for (const g of games) if (folder.includes(g)) return g;
  return folder;
}

// Split a real-result row (a full CSV row) into main and special numbers.
// Lotto's 7th value (the bonus) deliberately STAYS in the main pool: it is
// drawn from the same 1-45 drum and a predicted main matching it is a real
// hit (5+bonus is a prize tier) - only the separately-drawn special columns
// (stars/dream/viking) are split off, mirroring Helpers.main_special_split.
function splitRealResult(realResult, game) {
  if (!Array.isArray(realResult) || realResult.length === 0) return { mains: [], specials: [] };
  const s = SPECIAL_COLUMN_COUNTS[game] || 0;
  if (s > 0 && realResult.length > s) return { mains: realResult.slice(0, -s), specials: realResult.slice(-s) };
  return { mains: realResult.slice(), specials: [] };
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
function generateHeader(title = "Sequence Predictor") {
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
      
      /* DROPDOWN SETTINGS */
      .settings-container { position: relative; display: inline-block; }
      .settings-btn {
        background-color: #34495e; color: white; padding: 10px 15px;
        border: 1px solid #455a64; cursor: pointer; border-radius: 6px;
        font-size: 1em; transition: background 0.2s;
      }
      .settings-btn:hover { background-color: #2c3e50; }
      
      .settings-content {
        display: none; position: absolute; right: 0; top: 100%;
        background-color: white; min-width: 300px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.2); padding: 20px;
        z-index: 2000; border-radius: 8px; color: #333; border: 1px solid #ddd;
      }
      .settings-container:hover .settings-content { display: block; }
      
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
      input, select { 
        padding: 10px; margin: 5px 0 15px 0; 
        border: 1px solid #ccc; border-radius: 4px; width: 100%; box-sizing: border-box;
      }
      button { cursor: pointer; }

      .status-bar { font-size: 0.9em; color: #bdc3c7; margin-right: 15px; text-align: right;}
      .status-bar b { color: white; }
    </style>
  </head>
  <body>
    <div class="navbar">
      <div class="nav-group">
        <a href="/" style="font-size: 1.3em;">📊 Predictor</a>
        <a href="/database">History</a>
        <a id="optuna-link" href="#" target="_blank">Optuna</a>
      </div>

      <div class="nav-group">
        <div class="status-bar">
          <div>Model: <b>${selectedModel.join(', ')}</b></div>
          <div>Numbers: <b>${selectedPlayedNumbers.join(',')}</b></div>
        </div>
        
        <div class="settings-container">
          <button class="settings-btn">⚙️ Settings</button>
          <div class="settings-content">
            <h3 style="margin-top: 0;">Global Settings</h3>
            <form id="globalModelForm">
              <label><strong>Select Model(s):</strong></label><br>
              <select id="globalSelectedModel" multiple style="width: 100%; height: 120px;">
                <option value="all" ${selectedModel.includes('all') ? 'selected' : ''}>All Models</option>
                <option value="HybridStatisticalModel" ${selectedModel.includes('HybridStatisticalModel') ? 'selected' : ''}>HybridStatisticalModel</option>
                <option value="LaplaceMonteCarlo Model" ${selectedModel.includes('LaplaceMonteCarlo Model') ? 'selected' : ''}>LaplaceMonteCarlo</option>
                <option value="PoissonMarkov Model" ${selectedModel.includes('PoissonMarkov Model') ? 'selected' : ''}>PoissonMarkov</option>
                <option value="PoissonMonteCarlo Model" ${selectedModel.includes('PoissonMonteCarlo Model') ? 'selected' : ''}>PoissonMonteCarlo</option>
                <option value="MarkovBayesian Model" ${selectedModel.includes('MarkovBayesian Model') ? 'selected' : ''}>MarkovBayesian</option>
                <option value="Markov Model" ${selectedModel.includes('Markov Model') ? 'selected' : ''}>Markov</option>
                <option value="MarkovBayesianEnhanched Model" ${selectedModel.includes('MarkovBayesianEnhanched Model') ? 'selected' : ''}>MarkovBayesianEnhanced</option>
              </select>
              <button type="submit" style="width: 100%; background: #27ae60; color: white; border: none; padding: 10px; margin-top: 5px; border-radius: 4px;">Apply Models</button>
            </form>
            <hr style="margin: 20px 0; border: 0; border-top: 1px solid #eee;">
            <form id="globalPlayedNumbersForm">
              <label><strong>Keno Played Numbers:</strong></label><br>
              <input type="text" id="globalPlayedNumbers" value="${selectedPlayedNumbers.join(',')}" placeholder="4,5,6...">
              <button type="submit" style="width: 100%; background: #2980b9; color: white; border: none; padding: 10px; border-radius: 4px;">Update Numbers</button>
            </form>
          </div>
        </div>
      </div>
    </div>

    <script>
      // Toggle Card Logic
      function toggleCard(header) {
        const card = header.parentElement;
        card.classList.toggle('expanded');
      }

      // Dynamic Optuna Link
      document.addEventListener("DOMContentLoaded", function() {
        const optunaLink = document.getElementById("optuna-link");
        if(optunaLink) {
            // Uses current window hostname (e.g., localhost, 192.168.x.x, etc.) and adds port 3002
            optunaLink.href = \`\${window.location.protocol}//\${window.location.hostname}:3002\`;
        }
      });

      // Settings Logic
      document.getElementById('globalModelForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const options = document.getElementById('globalSelectedModel').selectedOptions;
        const values = Array.from(options).map(o => o.value);
        await fetch('/playedModel', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ selectedModel: values }) });
        window.location.reload();
      });

      document.getElementById('globalPlayedNumbersForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const val = document.getElementById('globalPlayedNumbers').value;
        const arr = val.split(',').map(n => n.trim()).filter(n => n);
        await fetch('/playedNumbers', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ playedNumbers: arr }) });
        window.location.reload();
      });
    </script>
    <div class="container">
  `;
}

function generateFooter() {
  return `</div></body></html>`;
}

// --- LOGIC: Filter Data ---
function filterDataByModel(data) {
  if (!data) return [];
  if (selectedModel.includes("all")) return data;
  return data.filter(modelItem => selectedModel.some(sel => modelItem.name === sel || modelItem.name.includes(sel)));
}

// --- LOGIC: Table Generation ---
// realResult is the full drawn row (mains + specials, or mains + lotto bonus);
// cells are highlighted index-aware so a predicted star only lights up against
// the drawn stars and a predicted main only against the drawn mains.
function generateTable(data, title = '', realResult = [], calcProfit = false, game = "") {
  const filteredData = filterDataByModel(data);
  if (filteredData.length === 0) return `<p style="padding: 10px; color: #888;">No predictions for selected model(s).</p>`;

  const specialCount = SPECIAL_COLUMN_COUNTS[game] || 0;
  const { mains: realMains, specials: realSpecials } = splitRealResult(realResult, game);
  // No real result (next-draw / home tables) -> no highlighting and no Hits column.
  const hasReal = realMains.length > 0;

  let html = `<div class="table-wrapper">`;
  if (title) html += `<div style="padding: 10px; font-weight: bold; background: #f8f9fa; border-bottom: 1px solid #ddd;">${title}</div>`;
  html += '<table border="1">';

  html += '<tr><th style="min-width: 150px;">Model</th><th style="width: 50px;">#</th>';
  if (filteredData.length > 0 && filteredData[0].predictions.length > 0) {
    Array.from({ length: filteredData[0].predictions[0].length }).forEach((_, i) => html += `<th>Num ${i + 1}</th>`);
  }
  if(hasReal) html += '<th>Hits</th>';
  if(calcProfit) html += '<th>Profit</th>';
  html += '</tr>';

  filteredData.forEach((model) => {
    model.predictions.forEach((row, rowIndex) => {
      const modelType = model.name || "not known";
      const { mains: ticketMains, specials: ticketSpecials } = splitTicket(row, realMains, specialCount);
      html += `<tr>
        <td style="font-weight: bold; background: #f9f9f9;">${modelType}</td>
        <td style="font-weight: bold; background: #f9f9f9;">${rowIndex + 1}</td>`;
      row.forEach((cell, cellIndex) => {
        // Trailing cells past the ticket's main block are special columns and
        // only match against the drawn specials; everything else only against
        // the drawn mains (pick3 keeps its historical by-inclusion behavior,
        // keno subset rows and RL mains-only rows have no special cells).
        const isSpecialCell = ticketSpecials.length > 0 && cellIndex >= ticketMains.length;
        const isMatching = hasReal && (isSpecialCell ? realSpecials.includes(cell) : realMains.includes(cell));
        html += `<td style="text-align: center; ${isMatching ? 'background: #2ecc71; color: white;' : ''}">${cell}</td>`;
      });
      if(hasReal) {
        const mainHits = ticketMains.filter(n => realMains.includes(n)).length;
        const specialHits = ticketSpecials.filter(n => realSpecials.includes(n)).length;
        // "3 (1)" = 3 main hits, 1 special hit; games without a special
        // column just show the main count.
        const hitDisplay = specialCount > 0 ? `${mainHits} (${specialHits})` : `${mainHits}`;
        html += `<td style="font-weight: bold; background: #f9f9f9;">${hitDisplay}</td>`;
      }
      if(calcProfit) {
        const profit = calculateProfit(row, realResult, game, modelType);
        html += `<td style="background: #f9f9f9;">${profit} €</td>`;
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
  const payoutTablePick3 = {
    straight: 500, box_with_doubles: 160, box_no_doubles: 80,
    front_pair: 50, back_pair: 50, last_number: 1, lost: -4 
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
      if (played != 3 || realResult.length != 3) return 0;
      const pred = prediction; const actual = realResult;
      const isSame = pred[0] === actual[0] && pred[1] === actual[1] && pred[2] === actual[2];
      const isPermutation = [...pred].sort().join('') === [...actual].sort().join('');
      if (isSame) return payoutTablePick3.straight;
      else if (isPermutation) {
        const countMap = {}; for (let n of pred) countMap[n] = (countMap[n] || 0) + 1;
        const hasDouble = Object.values(countMap).includes(2);
        return hasDouble ? payoutTablePick3.box_with_doubles : payoutTablePick3.box_no_doubles;
      } 
      else if (pred[0] === actual[0] && pred[1] === actual[1]) return payoutTablePick3.front_pair;
      else if (pred[1] === actual[1] && pred[2] === actual[2]) return payoutTablePick3.back_pair;
      else if (pred[2] === actual[2]) return payoutTablePick3.last_number;
      else return payoutTablePick3.lost;
    }
    default: {
      // Unreachable today (calcProfit is only enabled for keno/pick3), but
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
          Keno/Pick3 rank by average profit per bet (real payout tables); other games by average hits of the main ticket.
          Click a game row for the full model ranking. Greyed models have fewer scored draws than the ranking minimum.
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
          means it is drifting. Pick3 is scored positionally (digit in the right place). One peak is recorded per run
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
          Computed over the last 60 scored draws (pick3 per digit position, averaged). Normalized entropy near 1 and
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
  let html = generateHeader("Database Folders");
  html += '<h1>Available Database Folders</h1><div style="display: flex; gap: 10px; flex-wrap: wrap;">';
  folders.forEach((folder) => {
    html += `<form action="/database/${folder}" method="get">
      <button type="submit" style="padding: 15px 30px; font-size: 1.1em; cursor: pointer; background: white; border: 1px solid #ccc; border-radius: 5px;">${folder}</button>
    </form>`;
  });
  html += '</div>';
  html += generatePerformanceSummary();
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
  const calcProfit = game === "keno" || game === "pick3";
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

  let html = generateHeader(`${folder} Predictions`);
  html += `<h1>${folder}</h1><div>`;

  sortedMonths.forEach((month, index) => {
    filesByMonth[month].sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));
    let monthProfit = 0; let monthBest = { mains: 0, specials: 0 };

    const fileListHtml = filesByMonth[month].map(file => {
        const filePath = path.join(folderPath, file);
        const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        let fileProfit = 0; let fileBest = { mains: 0, specials: 0 };
        const validPredictions = filterDataByModel(jsonData.currentPrediction);

        if (validPredictions && validPredictions.length > 0) {
            if(calcProfit) {
                fileProfit = validPredictions.reduce((acc, predObj) => {
                    let pProfit = 0;
                    predObj.predictions.forEach(p => pProfit += calculateProfit(p, jsonData.realResult, game, predObj.name));
                    return acc + pProfit;
                }, 0);
            } else {
                // Best row is recomputed from the predictions (main hits vs
                // real mains only, then special hits as tie-break) instead of
                // trusting jsonData.matchingNumbers: old day JSONs still carry
                // the pooled main+special shape in that field, newer ones the
                // split one, and recomputing renders both vintages the same
                // way - and respects the active model filter.
                const { mains: realMains, specials: realSpecials } = splitRealResult(jsonData.realResult, game);
                validPredictions.forEach(predObj => {
                    predObj.predictions.forEach(p => {
                        const { mains: ticketMains, specials: ticketSpecials } = splitTicket(p, realMains, specialCount);
                        const mainHits = ticketMains.filter(n => realMains.includes(n)).length;
                        const specialHits = ticketSpecials.filter(n => realSpecials.includes(n)).length;
                        if (mainHits > fileBest.mains || (mainHits === fileBest.mains && specialHits > fileBest.specials)) {
                            fileBest = { mains: mainHits, specials: specialHits };
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
        // special numbers; other games just show the main count.
        const displayStat = calcProfit ? `${fileProfit} €` : `Match: ${fileBest.mains}${specialCount > 0 ? ` (${fileBest.specials})` : ''}`;

        return `<li style="padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between;">
            <a href="/database/${folder}/${file}" style="text-decoration: none; color: #333;">📄 ${file}</a>
            <span style="font-weight: bold; color: ${color};">${displayStat}</span>
        </li>`;
    }).join('');

    const monthColor = monthProfit > 0 ? '#27ae60' : (monthProfit < 0 ? '#c0392b' : '#7f8c8d');
    const headerStat = calcProfit ? `Total: ${monthProfit} €` : `Best Match: ${monthBest.mains}${specialCount > 0 ? ` (${monthBest.specials})` : ''}`;
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
  const calculateProfitFlag = game === "keno" || game === "pick3";
  const specialCount = SPECIAL_COLUMN_COUNTS[game] || 0;
  // Mains of the drawn row, used for the frequency-chart bar coloring: the
  // charted frequencies are main-number frequencies, so a bar must not turn
  // green just because its number came out as a star/dream/viking/bonus. The
  // array is interpolated into the chart script server-side - the browser has
  // no jsonData object (the previous client-side jsonData.realResult lookup
  // threw a ReferenceError and the analysis chart never rendered).
  const { mains: realMains } = splitRealResult(jsonData.realResult, game);

  let html = generateHeader(`${file} Details`);
  html += `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0;">${file}</h1>
        <a href="/database/${folder}" class="settings-btn" style="text-decoration: none;">Back to History</a>
    </div>

    <div class="card expanded">
        <div class="card-header" onclick="toggleCard(this)">
            <span class="card-title">Real Result</span><div class="card-icon">▼</div>
        </div>
        <div class="card-body">${generateList(jsonData.realResult)}</div>
    </div>

    <div class="card expanded">
        <div class="card-header" onclick="toggleCard(this)">
             <span class="card-title">Analysis of Prediction</span><div class="card-icon">▼</div>
        </div>
        <div class="card-body">
            ${generateTable(jsonData.currentPrediction, '', jsonData.realResult, calculateProfitFlag, game)}
            ${specialCount > 0
              ? `<p style="color: #7f8c8d; font-size: 0.85em; margin: 10px 0 0;">Hits are shown as <b>N (M)</b>: N hits among the main numbers, M among the special numbers (euromillions stars / eurodreams dream / vikinglotto viking). Cells highlight green only within their own group.</p>`
              : (game === 'lotto'
                ? `<p style="color: #7f8c8d; font-size: 0.85em; margin: 10px 0 0;">The bonus ball is not predicted, but a predicted number matching it counts as a hit (same 1-45 drum, 5+bonus is a prize tier).</p>`
                : '')}

            ${jsonData.currentNumberFrequency && Object.keys(jsonData.currentNumberFrequency).length > 0 ? `
                <div style="margin-top: 20px; height: 200px; width: 100%;">
                    <canvas id="chart-analysis"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('chart-analysis').getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: ${JSON.stringify(Object.keys(jsonData.currentNumberFrequency))},
                        datasets: [{
                            label: 'Freq',
                            data: ${JSON.stringify(Object.values(jsonData.currentNumberFrequency))},
                            backgroundColor: ${JSON.stringify(Object.keys(jsonData.currentNumberFrequency))}.map(n => ${JSON.stringify(realMains)}.includes(Number(n)) ? 'rgba(46, 204, 113, 0.8)' : 'rgba(52, 152, 219, 0.6)')
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

            ${jsonData.numberFrequency ? `
                <div style="margin-top: 20px; height: 200px; width: 100%;">
                    <canvas id="chart-detail"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('chart-detail').getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: ${JSON.stringify(Object.keys(jsonData.numberFrequency))},
                        datasets: [{ label: 'Freq', data: ${JSON.stringify(Object.values(jsonData.numberFrequency))}, backgroundColor: 'rgba(52, 152, 219, 0.6)' }]
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
  let html = generateHeader("Home - Dashboard");
  html += `<h1 style="margin-bottom: 20px;">New Predictions</h1>`;

  folders.forEach((folder) => {
    const folderPath = path.join(dataPath, folder);
    const files = fs.readdirSync(folderPath).filter((file) => file.endsWith('.json')).sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));

    if (files.length > 0) {
      const latestFile = files[0];
      const jsonData = JSON.parse(fs.readFileSync(path.join(folderPath, latestFile), 'utf-8'));

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
            ${generateTable(jsonData.newPrediction, '', [], false, '')}

            ${jsonData.numberFrequency ? `
                <div style="margin-top: 20px; height: 200px; width: 100%;">
                    <canvas id="chart-${folder}"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('chart-${folder}').getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: ${JSON.stringify(Object.keys(jsonData.numberFrequency))},
                        datasets: [{ label: 'Freq', data: ${JSON.stringify(Object.values(jsonData.numberFrequency))}, backgroundColor: 'rgba(52, 152, 219, 0.6)' }]
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

app.post('/playedNumbers', (req, res) => {
  let playedNumbers = req.body.playedNumbers;
  if (!playedNumbers) return res.status(400).send('No numbers');
  if (!Array.isArray(playedNumbers)) playedNumbers = [playedNumbers];
  selectedPlayedNumbers = playedNumbers.map(n => Number(n)).filter(n => !isNaN(n));
  res.json({ success: true });
});
  
app.post('/playedModel', (req, res) => {
  let playedModel = req.body.selectedModel;
  if (!Array.isArray(playedModel)) playedModel = [playedModel];
  selectedModel = playedModel;
  res.json({ success: true });
});

app.listen(config.PORT, () => { console.log(`Server running at http://${config.INTERFACE}:${config.PORT}`); });
exec('optuna-dashboard sqlite:///db.sqlite3 --host 0.0.0.0 --port 8080', (error) => { if(error) console.log("Optuna dashboard not started."); });