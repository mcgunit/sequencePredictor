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
        <a href="/database">Database</a>
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
            // Uses current window hostname (e.g., localhost, 192.168.x.x, etc.) and adds port 8080
            optunaLink.href = \`\${window.location.protocol}//\${window.location.hostname}:8080\`;
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
function generateTable(data, title = '', matchingNumbers = [], calcProfit = false, game = "") {
  const filteredData = filterDataByModel(data);
  if (filteredData.length === 0) return `<p style="padding: 10px; color: #888;">No predictions for selected model(s).</p>`;

  let html = `<div class="table-wrapper">`;
  if (title) html += `<div style="padding: 10px; font-weight: bold; background: #f8f9fa; border-bottom: 1px solid #ddd;">${title}</div>`;
  html += '<table border="1">';
  
  html += '<tr><th style="min-width: 150px;">Model</th><th style="width: 50px;">#</th>';
  if (filteredData.length > 0 && filteredData[0].predictions.length > 0) {
    Array.from({ length: filteredData[0].predictions[0].length }).forEach((_, i) => html += `<th>Num ${i + 1}</th>`);
  }
  if(calcProfit) html += '<th>Profit</th>';
  html += '</tr>';

  filteredData.forEach((model) => {
    model.predictions.forEach((row, rowIndex) => {
      const modelType = model.name || "not known";
      html += `<tr>
        <td style="font-weight: bold; background: #f9f9f9;">${modelType}</td>
        <td style="font-weight: bold; background: #f9f9f9;">${rowIndex + 1}</td>`;
      row.forEach((cell) => {
        const isMatching = matchingNumbers.includes(cell);
        html += `<td style="text-align: center; ${isMatching ? 'background: #2ecc71; color: white;' : ''}">${cell}</td>`;
      });
      if(calcProfit) {
        const profit = calculateProfit(row, matchingNumbers, game, modelType);
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
    default:
      const correctNumbers = prediction.filter(n => realResult.includes(n)).length;
      return `${correctNumbers}/${played}`;
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
  html += generateFooter();
  res.send(html);
});

// 2. Folder View
app.get('/database/:folder', (req, res) => {
  const folder = req.params.folder;
  const folderPath = path.join(dataPath, folder);
  let calcProfit = false; let game = "";
  if (!fs.existsSync(folderPath)) return res.status(404).send('Folder not found');
  if (folder.includes("keno")) { calcProfit = true; game = "keno"; } 
  if (folder.includes("pick3")) { calcProfit = true; game = "pick3"; }

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
    let monthProfit = 0; let monthMaxCorrect = 0;
    
    const fileListHtml = filesByMonth[month].map(file => {
        const filePath = path.join(folderPath, file);
        const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        let fileProfit = 0; let fileMaxCorrect = 0;
        const validPredictions = filterDataByModel(jsonData.currentPrediction);

        if (validPredictions && validPredictions.length > 0) {
            if(calcProfit) {
                fileProfit = validPredictions.reduce((acc, predObj) => {
                    let pProfit = 0;
                    predObj.predictions.forEach(p => pProfit += calculateProfit(p, jsonData.realResult, game, predObj.name));
                    return acc + pProfit;
                }, 0);
            } else {
                validPredictions.forEach(predObj => {
                    predObj.predictions.forEach(p => {
                        const match = p.filter(n => jsonData.realResult.includes(n)).length;
                        if(match > fileMaxCorrect) fileMaxCorrect = match;
                    });
                });
            }
        }
        monthProfit += fileProfit;
        if(fileMaxCorrect > monthMaxCorrect) monthMaxCorrect = fileMaxCorrect;
        const color = fileProfit > 0 ? 'green' : (fileProfit < 0 ? 'red' : 'orange');
        const displayStat = calcProfit ? `${fileProfit} €` : `Match: ${fileMaxCorrect}`;

        return `<li style="padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between;">
            <a href="/database/${folder}/${file}" style="text-decoration: none; color: #333;">📄 ${file}</a>
            <span style="font-weight: bold; color: ${color};">${displayStat}</span>
        </li>`;
    }).join('');

    const monthColor = monthProfit > 0 ? '#27ae60' : (monthProfit < 0 ? '#c0392b' : '#7f8c8d');
    const headerStat = calcProfit ? `Total: ${monthProfit} €` : `Best Match: ${monthMaxCorrect}`;
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
  let calculateProfitFlag = false; let game = "";
  if (!fs.existsSync(filePath)) return res.status(404).send('File not found');
  const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  if(folder.includes("keno")) { calculateProfitFlag = true; game = "keno"; } 
  if(folder.includes("pick3")) { calculateProfitFlag = true; game = "pick3"; }

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
        </div>
    </div>

    <div class="card expanded">
        <div class="card-header" onclick="toggleCard(this)">
             <span class="card-title">Next Draw Prediction</span><div class="card-icon">▼</div>
        </div>
        <div class="card-body">
            ${generateTable(jsonData.newPrediction, '', [], false, game)}
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
  html += `<h1 style="margin-bottom: 20px;">Dashboard</h1>`;

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
                <span class="card-meta">(${latestFile})</span>
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