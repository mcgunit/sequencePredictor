const express = require('express');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

const config = require("./config");

const app = express();

// Middleware to parse form data
app.use(express.urlencoded({ extended: true }));
app.use(express.json()); 

// Paths
const dataPath = path.join(__dirname, 'data', 'database');
const modelsPath = path.join(__dirname, 'data', 'models');

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
      body { font-family: Arial, sans-serif; margin: 0; padding-top: 60px; background-color: #f4f4f9; }
      
      /* STICKY NAVBAR */
      .navbar {
        position: fixed;
        top: 0;
        width: 100%;
        background-color: #333;
        color: white;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        z-index: 1000;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
      }
      .navbar a {
        color: white;
        text-decoration: none;
        margin-right: 15px;
        font-weight: bold;
      }
      .navbar a:hover { text-decoration: underline; }
      
      .nav-group { display: flex; align-items: center; }
      
      /* DROPDOWN SETTINGS */
      .settings-container {
        position: relative;
        display: inline-block;
      }
      .settings-btn {
        background-color: #555;
        color: white;
        padding: 8px 12px;
        border: none;
        cursor: pointer;
        border-radius: 4px;
      }
      .settings-content {
        display: none;
        position: absolute;
        right: 0;
        background-color: #f9f9f9;
        min-width: 250px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        padding: 15px;
        z-index: 1;
        border-radius: 5px;
        color: black;
      }
      .settings-container:hover .settings-content { display: block; }
      
      /* GLOBAL STYLES */
      h1, h2 { color: #333; }
      .container { padding: 20px; max-width: 1200px; margin: auto; }
      table { width: 100%; border-collapse: collapse; margin-top: 10px; background: white; }
      th, td { padding: 10px; border: 1px solid #ddd; text-align: center; }
      th { background-color: #444; color: white; }
      tr:nth-child(even) { background-color: #f2f2f2; }
      
      .status-bar {
        font-size: 0.85em;
        color: #bbb;
        margin-left: 20px;
      }
      
      input, select, button { padding: 5px; margin: 5px 0; }
    </style>
  </head>
  <body>
    <div class="navbar">
      <div class="nav-group">
        <a href="/">Home</a>
        <a href="/database">Database</a>
        <a href="/models">Models</a>
        <a href="http://localhost:8080" target="_blank">Optuna Dashboard</a>
      </div>

      <div class="nav-group">
        <div class="status-bar">
          Model: <b>${selectedModel.join(', ')}</b> | Numbers: <b>${selectedPlayedNumbers.join(',')}</b>
        </div>
        
        <div class="settings-container" style="margin-left: 15px;">
          <button class="settings-btn">⚙️ Settings</button>
          <div class="settings-content">
            <h4>Global Settings</h4>
            
            <form id="globalModelForm">
              <label><strong>Select Model(s):</strong></label><br>
              <select id="globalSelectedModel" multiple style="width: 100%; height: 100px;">
                <option value="all" ${selectedModel.includes('all') ? 'selected' : ''}>All Models</option>
                <option value="HybridStatisticalModel" ${selectedModel.includes('HybridStatisticalModel') ? 'selected' : ''}>HybridStatisticalModel</option>
                <option value="LaplaceMonteCarlo Model" ${selectedModel.includes('LaplaceMonteCarlo Model') ? 'selected' : ''}>LaplaceMonteCarlo</option>
                <option value="PoissonMarkov Model" ${selectedModel.includes('PoissonMarkov Model') ? 'selected' : ''}>PoissonMarkov</option>
                <option value="PoissonMonteCarlo Model" ${selectedModel.includes('PoissonMonteCarlo Model') ? 'selected' : ''}>PoissonMonteCarlo</option>
                <option value="MarkovBayesian Model" ${selectedModel.includes('MarkovBayesian Model') ? 'selected' : ''}>MarkovBayesian</option>
                <option value="Markov Model" ${selectedModel.includes('Markov Model') ? 'selected' : ''}>Markov</option>
                <option value="MarkovBayesianEnhanched Model" ${selectedModel.includes('MarkovBayesianEnhanched Model') ? 'selected' : ''}>MarkovBayesianEnhanced</option>
              </select>
              <button type="submit" style="width: 100%; background: #4CAF50; color: white; border: none;">Apply Models</button>
            </form>
            <hr>
            
            <form id="globalPlayedNumbersForm">
              <label><strong>Keno Played Numbers:</strong></label><br>
              <input type="text" id="globalPlayedNumbers" value="${selectedPlayedNumbers.join(',')}" style="width: 90%;">
              <button type="submit" style="width: 100%; background: #2196F3; color: white; border: none;">Update Numbers</button>
            </form>
            
          </div>
        </div>
      </div>
    </div>

    <script>
      // Model Form Handler
      document.getElementById('globalModelForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const options = document.getElementById('globalSelectedModel').selectedOptions;
        const values = Array.from(options).map(o => o.value);
        
        await fetch('/playedModel', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ selectedModel: values })
        });
        window.location.reload();
      });

      // Numbers Form Handler
      document.getElementById('globalPlayedNumbersForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const val = document.getElementById('globalPlayedNumbers').value;
        const arr = val.split(',').map(n => n.trim()).filter(n => n);
        
        await fetch('/playedNumbers', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ playedNumbers: arr })
        });
        window.location.reload();
      });
    </script>
    <div class="container">
  `;
}

function generateFooter() {
  return `</div></body></html>`;
}

// --- LOGIC: Filter Data by Selected Model ---
function filterDataByModel(data) {
  if (!data) return [];
  if (selectedModel.includes("all")) return data;
  
  // Filter the predictions array to only include selected models
  return data.filter(modelItem => {
    // Check if model.name is in selectedModel list
    // (Ensure exact matching or partial matching depending on your naming convention)
    return selectedModel.some(sel => modelItem.name === sel || modelItem.name.includes(sel));
  });
}

// --- LOGIC: Table Generation (Updated with Filter) ---
function generateTable(data, title = '', matchingNumbers = [], calcProfit = false, game = "") {
  // 1. Filter Data First
  const filteredData = filterDataByModel(data);

  if (filteredData.length === 0) return `<p>No data for selected model(s): ${selectedModel.join(', ')}</p>`;

  let table = '<table border="1" style="border-collapse: collapse; width: 100%;">';

  if (title) table += `<caption><strong>${title}</strong></caption>`;
  
  // Headers
  table += '<tr>' +
    '<th style="min-width: 150px;">Model</th>' +
    '<th style="width: 50px;">#</th>';

  if (filteredData.length > 0 && filteredData[0].predictions.length > 0) {
    Array.from({ length: filteredData[0].predictions[0].length }).forEach((_, i) => {
      table += `<th>Number ${i + 1}</th>`;
    });
  }
  if(title.includes("Current Prediction") && calcProfit) {
    table += '<th>Profit</th>';
  }
  table += '</tr>';

  // Rows
  filteredData.forEach((model) => {
    model.predictions.forEach((row, rowIndex) => {
      const modelType = model.name || "not known";
      table += `<tr>
        <td style="font-weight: bold; background: #f9f9f9;">${modelType}</td>
        <td style="font-weight: bold; background: #f9f9f9;">${rowIndex + 1}</td>`;

      row.forEach((cell) => {
        const isMatching = matchingNumbers.includes(cell);
        table += `<td style="text-align: center; ${isMatching ? 'background: #2ecc71; color: white;' : ''}">${cell}</td>`;
      });

      if(title.includes("Current Prediction") && calcProfit) {
        const profit = calculateProfit(row, matchingNumbers, game, modelType);
        table += `<td style="background: #f9f9f9;">${profit} €</td>`;
      }
      table += '</tr>';
    });
  });

  table += '</table>';
  return table;
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
    straight: 500,
    box_with_doubles: 160,
    box_no_doubles: 80,
    front_pair: 50,
    back_pair: 50,
    last_number: 1,
    lost: -4 
  };

  const played = prediction.length;

  switch (game) {
    case "keno": {
      // Use selectedPlayedNumbers size to determine payout tier, assuming user plays subset
      // Or if prediction size matches logic. For now, use prediction length.
      const correctNumbers = prediction.filter(n => realResult.includes(n)).length;
      if (played >= 2 && played <= 10 && payoutTableKeno[played]) {
        return payoutTableKeno[played][correctNumbers] ?? payoutTableKeno["lost"];
      } 
      // Keno typically implies picking X numbers. If model predicts 20, we assume user picks 'selectedPlayedNumbers' amount from top.
      return 0; 
    }

    case "pick3": {
      if (played != 3 || realResult.length != 3) return 0;
      const pred = prediction;
      const actual = realResult;
      const isSame = pred[0] === actual[0] && pred[1] === actual[1] && pred[2] === actual[2];
      const isPermutation = [...pred].sort().join('') === [...actual].sort().join('');

      if (isSame) return payoutTablePick3.straight;
      else if (isPermutation) {
        const countMap = {};
        for (let n of pred) countMap[n] = (countMap[n] || 0) + 1;
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
    let table = '<table style="width: auto; margin-bottom: 20px;">';
    if (title) table += `<caption><strong>${title}</strong></caption>`;
    table += '<tr>';
    data.forEach((item) => {
      table += `<td style="padding: 10px; background: #eee; font-size: 1.1em; font-weight: bold;">${item}</td>`;
    });
    table += '</tr></table>';
    return table;
  }
  return '';
}


// --- ROUTES ---

// 1. Database Index
app.get('/database', (req, res) => {
  const folders = fs.readdirSync(dataPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((dir) => dir.name);

  let html = generateHeader("Database Folders");
  html += '<h1>Available Database Folders</h1><div style="display: flex; gap: 10px; flex-wrap: wrap;">';
  
  folders.forEach((folder) => {
    html += `<form action="/database/${folder}" method="get">
      <button type="submit" style="padding: 15px 30px; font-size: 1.1em; cursor: pointer;">${folder}</button>
    </form>`;
  });
  
  html += '</div>';
  html += generateFooter();
  res.send(html);
});

// 2. Folder View (Months)
app.get('/database/:folder', (req, res) => {
  const folder = req.params.folder;
  const folderPath = path.join(dataPath, folder);
  let calcProfit = false;
  let game = "";

  if (!fs.existsSync(folderPath)) return res.status(404).send('Folder not found');
  
  if (folder.includes("keno")) { calcProfit = true; game = "keno"; } 
  if (folder.includes("pick3")) { calcProfit = true; game = "pick3"; }

  const files = fs.readdirSync(folderPath).filter((file) => file.endsWith('.json'));

  // Group by month
  const filesByMonth = files.reduce((acc, file) => {
    const date = new Date(file.replace('.json', ''));
    if(!isNaN(date)) {
        const monthYear = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        if (!acc[monthYear]) acc[monthYear] = [];
        acc[monthYear].push(file);
    }
    return acc;
  }, {});

  const sortedMonths = Object.keys(filesByMonth).sort((a, b) => new Date(a) - new Date(b));

  let html = generateHeader(`${folder} Predictions`);
  html += `<h1>Predictions in ${folder}</h1>`;

  // Grid layout for months
  html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">';

  sortedMonths.forEach(month => {
    // Sort files descending
    filesByMonth[month].sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));
    
    // Calculate Monthly Stats based on SELECTED MODEL
    let monthProfit = 0;
    let monthMaxCorrect = 0;

    const fileListHtml = filesByMonth[month].map(file => {
        const filePath = path.join(folderPath, file);
        const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        
        let fileProfit = 0;
        let fileMaxCorrect = 0;
        
        // Filter predictions by global selectedModel
        const validPredictions = filterDataByModel(jsonData.currentPrediction);

        if (validPredictions && validPredictions.length > 0) {
            if(calcProfit) {
                fileProfit = validPredictions.reduce((acc, predObj) => {
                    let pProfit = 0;
                    predObj.predictions.forEach(p => {
                        pProfit += calculateProfit(p, jsonData.realResult, game, predObj.name);
                    });
                    return acc + pProfit;
                }, 0);
            } else {
                // Count matching
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

        return `<li style="margin-bottom: 5px; display: flex; justify-content: space-between;">
            <a href="/database/${folder}/${file}" style="text-decoration: none; color: #333;">📄 ${file}</a>
            <span style="font-weight: bold; color: ${color};">${displayStat}</span>
        </li>`;
    }).join('');

    const monthColor = monthProfit > 0 ? 'green' : (monthProfit < 0 ? 'red' : '#333');
    const headerStat = calcProfit ? `Total: ${monthProfit} €` : `Best Match: ${monthMaxCorrect}`;

    html += `
    <div style="border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="background: #444; color: white; padding: 10px; text-align: center;">
            <div style="font-size: 1.2em; font-weight: bold;">${month}</div>
            <div style="color: ${monthColor}; background: white; border-radius: 4px; padding: 2px 8px; margin-top: 5px; display: inline-block;">
                ${headerStat}
            </div>
        </div>
        <ul style="list-style: none; padding: 15px; margin: 0; max-height: 300px; overflow-y: auto;">
            ${fileListHtml}
        </ul>
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
  let calculateProfitFlag = false;
  let game = "";

  if (!fs.existsSync(filePath)) return res.status(404).send('File not found');

  const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  
  if(folder.includes("keno")) { calculateProfitFlag = true; game = "keno"; } 
  if(folder.includes("pick3")) { calculateProfitFlag = true; game = "pick3"; }

  let html = generateHeader(`${file} Details`);
  
  html += `
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1>${file}</h1>
        <a href="/database/${folder}" class="settings-btn" style="text-decoration: none;">Back to ${folder}</a>
    </div>

    <h2>Real Result</h2>
    ${generateList(jsonData.realResult)}

    <h2>Current Prediction (Filtered by: ${selectedModel.join(', ')})</h2>
    ${generateTable(
      jsonData.currentPrediction, 
      '', 
      jsonData.realResult,
      calculateProfitFlag,
      game
    )}

    <h2>New Prediction for Next Draw</h2>
    ${generateTable(
      jsonData.newPrediction,
      '', 
      [],
      calculateProfitFlag,
      game
    )}
  `;
  html += generateFooter();
  res.send(html);
});

// 4. Models View
app.get('/models', (req, res) => {
    const folders = fs.readdirSync(modelsPath, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((dir) => dir.name);
  
    let html = generateHeader("Models");
    html += '<h1>Available Models</h1><ul>';
    folders.forEach((folder) => {
      html += `<li><a href="/models/${folder}" style="font-size: 1.2em;">📁 ${folder}</a></li>`;
    });
    html += '</ul>';
    html += generateFooter();
    res.send(html);
});

app.get('/models/:folder', (req, res) => {
    const folder = req.params.folder;
    const folderPath = path.join(modelsPath, folder);
  
    if (!fs.existsSync(folderPath)) return res.status(404).send('Folder not found');
  
    const files = fs.readdirSync(folderPath).filter((file) => file.endsWith('.png'));
  
    let html = generateHeader(`${folder} Images`);
    html += `<h1>Images in ${folder}</h1><div style="display: flex; flex-wrap: wrap; gap: 20px;">`;
    files.forEach((file) => {
      html += `
      <div style="border: 1px solid #ccc; padding: 10px; background: white; border-radius: 5px;">
        <h3>${file}</h3>
        <img src="/models/${folder}/${file}" alt="${file}" style="max-width: 400px; height: auto;">
      </div>`;
    });
    html += '</div>';
    html += generateFooter();
    res.send(html);
});

// 5. Home Page
app.get('/', (req, res) => {
  const folders = fs.readdirSync(dataPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((dir) => dir.name);

  let html = generateHeader("Home - Dashboard");
  html += `
    <h1>Dashboard</h1>
    <p>Welcome! Use the settings gear ⚙️ in the top right to filter models globally.</p>
    <div class="charts-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
  `;

  folders.forEach((folder) => {
    const folderPath = path.join(dataPath, folder);
    const files = fs.readdirSync(folderPath)
      .filter((file) => file.endsWith('.json'))
      .sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));

    if (files.length > 0) {
      const latestFile = files[0];
      const jsonData = JSON.parse(fs.readFileSync(path.join(folderPath, latestFile), 'utf-8'));

      html += `
        <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
          <h2>${folder} <span style="font-size: 0.6em; color: #888;">(${latestFile})</span></h2>
          
          <h4>Next Prediction:</h4>
          ${generateTable(jsonData.newPrediction, '')}

          ${jsonData.numberFrequency ? `
            <div style="margin-top: 15px;">
                <canvas id="chart-${folder}" style="max-height: 200px;"></canvas>
            </div>
            <script>
                new Chart(document.getElementById('chart-${folder}').getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ${JSON.stringify(Object.keys(jsonData.numberFrequency))},
                    datasets: [{
                    label: 'Frequency',
                    data: ${JSON.stringify(Object.values(jsonData.numberFrequency))},
                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
                });
            </script>
          ` : ''}
          
          <div style="margin-top: 15px; text-align: right;">
            <a href="/database/${folder}" class="settings-btn" style="text-decoration: none; background: #333;">View History</a>
          </div>
        </div>
      `;
    }
  });

  html += `</div>`;
  html += generateFooter();
  res.send(html);
});

// API Routes
app.post('/playedNumbers', (req, res) => {
    let playedNumbers = req.body.playedNumbers;
    if (!playedNumbers) return res.status(400).send('No numbers provided');
    if (!Array.isArray(playedNumbers)) playedNumbers = [playedNumbers];
    playedNumbers = playedNumbers.map(n => Number(n)).filter(n => !isNaN(n));
    
    selectedPlayedNumbers = playedNumbers;
    console.log('Updated played numbers:', selectedPlayedNumbers);
    res.json({ success: true });
});
  
app.post('/playedModel', (req, res) => {
    let playedModel = req.body.selectedModel;
    if (!Array.isArray(playedModel)) playedModel = [playedModel];
    
    selectedModel = playedModel;
    console.log('Updated model filter:', selectedModel);
    res.json({ success: true });
});

// Start Server
app.listen(config.PORT, () => {
  console.log(`Server running at http://${config.INTERFACE}:${config.PORT}`);
});

// Start Optuna Dashboard (Optional)
exec('optuna-dashboard sqlite:///db.sqlite3 --host 0.0.0.0 --port 8080', (error) => {
    if(error) console.log("Optuna dashboard not started (optional).");
});