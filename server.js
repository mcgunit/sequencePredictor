const express = require('express');
const path = require('path');
const fs = require('fs');

const config = require("./config");

const app = express();

// Middleware to parse form data
app.use(express.urlencoded({ extended: true }));
app.use(express.json()); // To parse JSON body

// Paths
const dataPath = path.join(__dirname, 'data', 'database');
const modelsPath = path.join(__dirname, 'data', 'models');


var selectedPlayedNumbers = [4,5,6,7,8,9,10]; // To select played numbers  for Keno
var selectedModel = ["all"]; // To select with wich model's predictions is played

function generateTable(data, title = '', matchingNumbers = [], calcProfit = false, game = "") {
  let table = '<table border="1" style="border-collapse: collapse; width: 100%;">';

  // Add title if provided
  if (title) table += `<caption><strong>${title}</strong></caption>`;
  
  // Define column headers
  table += '<tr>' +
    '<th style="padding: 5px; text-align: center; background: #333; color: white; min-width: 150px;">Model</th>' +
    '<th style="padding: 5px; text-align: center; background: #333; color: white; width: 50px;">#</th>';

  // Add number headers
  if (data.length > 0 && data[0].predictions.length > 0) {
    Array.from({ length: data[0].predictions[0].length }).forEach((_, i) => {
      table += `<th style="padding: 5px; text-align: center; background: #333; color: white;">Number ${i + 1}</th>`;
    });
  }
  if(title.includes("Current Prediction") && calcProfit) {
    table += '<th style="padding: 5px; text-align: center; background: #333; color: white;">Profit</th>';
  }
  table += '</tr>';

  // Add data rows with model identification
  data.forEach((model, modelIndex) => {
    model.predictions.forEach((row, rowIndex) => {
      const modelType = model.name || "not known";
      table += `<tr>
        <td style="padding: 5px; background: #f8f9fa; font-weight: bold; border-right: 2px solid #ddd;">
          ${modelType}
        </td>
        <td style="padding: 5px; text-align: center; font-weight: bold; background: #f8f9fa;">${rowIndex + 1}</td>`;

      let correctNumbers = 0;
      row.forEach((cell) => {
        const isMatching = matchingNumbers.includes(cell);
        if (isMatching) correctNumbers++;
        table += `<td style="padding: 5px; text-align: center; 
          ${isMatching ? 'background: #2ecc71; color: white;' : ''}">
          ${cell}
        </td>`;
      });

      if(title.includes("Current Prediction") && calcProfit) {
        const numbersPlayed = row.length;
        const profit = calculateProfit(numbersPlayed, correctNumbers, game, modelType);
        table += `<td style="padding: 5px; text-align: center; background: #f8f9fa;">${profit} €</td>`;
      }
      table += '</tr>';
    });
  });

  table += '</table>';
  return table;
}

function calculateProfit(numbersPlayed, correctNumbers, game, name) {
  // Define the Keno payout table
  const payoutTableKeno = {
    10: { 0: 3, 5: 1, 6: 4, 7: 10, 8: 200, 9: 2000, 10: 250000 },
    9: { 0: 3, 5: 2, 6: 5, 7: 50, 8: 500, 9: 50000 },
    8: { 0: 3, 5: 4, 6: 10, 7: 100, 8: 10000 },
    7: { 0: 3, 5: 3, 6: 30, 7: 3000 },
    6: { 3: 1, 4: 4, 5: 20, 6: 200 },
    5: { 3: 2, 4: 5, 5: 150 },
    4: { 2: 1, 3: 2, 4: 30 },
    3: { 2: 1, 3: 16 },
    2: { 2: 6.50 },
    "lost": -1
  };

  const payoutTablePick3 = {
    3: {3: 80},
    "lost": -1
  }

  switch (game) {
    case "keno": {
      if (payoutTableKeno[numbersPlayed] && selectedPlayedNumbers.includes(numbersPlayed) && (selectedModel.includes("all") || selectedModel.includes(name))) {
        if(payoutTableKeno[numbersPlayed][correctNumbers]) {
          return payoutTableKeno[numbersPlayed][correctNumbers];
        } else {
          return payoutTableKeno["lost"]; // No profit 
        }   
      } else {
        return 0; // In case of keno, the payout table is only defined for 2-10 numbers
      }
    }
    case "pick3": {
      if (payoutTablePick3[numbersPlayed] && (selectedModel.includes("all") || selectedModel.includes(name))) {
        if(payoutTablePick3[numbersPlayed][correctNumbers]) {
          return payoutTablePick3[numbersPlayed][correctNumbers];
        } else {
          return payoutTablePick3["lost"]; // No profit 
        }
      } else {
        return 0;
      }
    }
    default:
      return 0; // Don't calculate profit for other games
  }

}


function generateList(data, title = '') {
  if(Array.isArray(data) && data.length > 0) {
    let table = '<table border="1" style="border-collapse: collapse; width: 100%;">';
    if (title) table += `<caption><strong>${title}</strong></caption>`;
    table += '<tr>';
    data.forEach((item) => {
      table += `<td style="padding: 5px; text-align: center; width: 100px; min-width: 100px;">${item}</td>`;
    });
    table += '</tr>';
    table += '</table>';
    return table;
  }
}


// Serve static files
app.use('/models', express.static(modelsPath)); // Serves PNGs from models directory

// Router to handle database data
app.get('/database', (req, res) => {
  const folders = fs.readdirSync(dataPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((dir) => dir.name);

  let html = '<h1>Available Database Folders</h1><ul>';
  folders.forEach((folder) => {
    html += `<li><form action="/database/${folder}" method="get"><button type="submit">${folder}</button></form></li>`;
  });
  html += '</ul><form action="/" method="get"><button type="submit">Back to Home</button></form>';

  res.send(html);
});

app.get('/database/:folder', (req, res) => {
  const folder = req.params.folder;
  const folderPath = path.join(dataPath, folder);
  let calcProfit = false;
  let game = "";

  if (!fs.existsSync(folderPath)) {
    return res.status(404).send('Folder not found');
  }

  if (folder.includes("keno")) {
    calcProfit = true;
    game = "keno";
  } if (folder.includes("pick3")) {
    calcProfit = true;
    game = "pick3";
  }

  const files = fs.readdirSync(folderPath)
    .filter((file) => file.endsWith('.json'));

  // Group files by month and year
  const filesByMonth = files.reduce((acc, file) => {
    const date = new Date(file.replace('.json', ''));
    const monthYear = `${date.getFullYear()}-${date.getMonth() + 1}`;
    if (!acc[monthYear]) acc[monthYear] = [];
    acc[monthYear].push(file);
    return acc;
  }, {});

  // Sort months
  const sortedMonths = Object.keys(filesByMonth).sort((a, b) => new Date(a) - new Date(b));

  let html = `<h1>Predictions in ${folder}</h1>`;
  if(game == "keno") {
    html += `
    <form id="playedNumberForm">
      <label for="playedNumbers">Enter Numbers (comma separated):</label>
      <input type="text" id="playedNumbers" name="playedNumbers" placeholder="e.g. 4,5,6,7,8,9,10" required>
      <button type="submit">Submit</button>
      <button type="button" id="resetPlayedNumbers">Reset</button>
    </form>

    <form id="selectedModelForm">
      <label for="selectedModel">Enter Selected Model:</label>
      <select id="selectedModel" name="selectedModel(s)" multiple required>
        <option value="all">all</option>
        <option value="HybridStatisticalModel">HybridStatisticalModel</option>
        <option value="LaplaceMonteCarlo Model">LaplaceMonteCarlo Model</option>
        <option value="PoissonMarkov Model">PoissonMarkov Model</option>
        <option value="PoissonMonteCarlo Model">PoissonMonteCarlo Model</option>
        <option value="MarkovBayesian Model">MarkovBayesian Model</option>
        <option value="Markov Model">Markov Model</option>
        <option value="MarkovBayesianEnhanched Model">MarkovBayesianEnhanched Model</option>
      </select>
      <button type="submit">Submit</button>
      <button type="button" id="resetSelectedModel">Reset</button>
    </form>

    <div id="result">Played numbers: ${selectedPlayedNumbers}</div>
    <div id="result">Played model: ${selectedModel}</div>

    <script>
      const form = document.getElementById('playedNumberForm');

      form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const rawInput = document.getElementById('playedNumbers').value;
        const playedNumbersArray = rawInput.split(',').map(n => n.trim()).filter(n => n !== '');

        await fetch('/playedNumbers', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ playedNumbers: playedNumbersArray }),
        });

        form.reset();
        window.location.reload();
      });

      document.getElementById('resetPlayedNumbers').addEventListener('click', async () => {
        await fetch('/resetPlayedNumbers', { method: 'POST' });
        window.location.reload();
      });
    </script>

    <script>
      const modelForm = document.getElementById('selectedModelForm');

      modelForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const selectedOptions = Array.from(document.getElementById('selectedModel').selectedOptions);
        const selectedModelsArray = selectedOptions.map(option => option.value);

        await fetch('/playedModel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ selectedModel: selectedModelsArray }),
        });

        modelForm.reset();
        window.location.reload();
      });

      document.getElementById('resetSelectedModel').addEventListener('click', async () => {
        await fetch('/resetSelectedModel', { method: 'POST' });
        window.location.reload();
      });
    </script>
  `;
  }

  html += '<table border="1" style="border-collapse: collapse; width: 100%;">';

  // Create rows with a maximum of 3 columns per row
  for (let i = 0; i < sortedMonths.length; i += 3) {
    html += '<tr>';
    for (let j = i; j < i + 3 && j < sortedMonths.length; j++) {
      const month = sortedMonths[j];
      let monthTotalProfit = 0;

      // Calculate total profit for the month
      filesByMonth[month].forEach((file) => {
        const filePath = path.join(folderPath, file);
        const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

        if (calcProfit && jsonData.currentPrediction) {
          monthTotalProfit += jsonData.currentPrediction.reduce((acc, prediction) => {
            let predictionProfit = 0;
            prediction.predictions.forEach((pred) => {
              const correctNumbers = pred.filter(num => jsonData.realResult.includes(num)).length;
              predictionProfit += calculateProfit(pred.length, correctNumbers, game, prediction.name);
            });
            return acc + predictionProfit;
          }, 0);
        }
      });

      let profitColor = 'white';
      if (monthTotalProfit > 0) {
        profitColor = 'green';
      } else if (monthTotalProfit < 0) {
        profitColor = 'red';
      }

      html += `<th style="padding: 5px; text-align: center; background: #333; color: white;">
        ${month}<br><span style="color: ${profitColor};">Total Profit: ${monthTotalProfit} €</span>
      </th>`;
    }
    html += '</tr><tr>';
    for (let j = i; j < i + 3 && j < sortedMonths.length; j++) {
      const month = sortedMonths[j];
      html += '<td style="vertical-align: top;">';
      html += '<ul>';
      // Sort files within the month in descending order
      filesByMonth[month].sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));
      filesByMonth[month].forEach((file) => {
        const filePath = path.join(folderPath, file);
        const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

        let totalProfit = 0;
        if (calcProfit && jsonData.currentPrediction) {
          totalProfit = jsonData.currentPrediction.reduce((acc, prediction) => {
            let predictionProfit = 0;
            prediction.predictions.forEach((pred) => {
              const correctNumbers = pred.filter(num => jsonData.realResult.includes(num)).length;
              predictionProfit += calculateProfit(pred.length, correctNumbers, game, prediction.name);
            });
            return acc + predictionProfit;
          }, 0);
        }

        let profitColor = 'orange';
        if (totalProfit > 0) {
          profitColor = 'green';
        } else if (totalProfit < 0) {
          profitColor = 'red';
        }

        html += `<li>
          <form action="/database/${folder}/${file}" method="get" style="display: inline;">
            <button type="submit">${file}</button>
          </form>
          <span style="color: ${profitColor};">Profit: ${totalProfit} €</span>
        </li>`;
      });
      html += '</ul>';
      html += '</td>';
    }
    html += '</tr>';
  }

  html += '</table>';
  html += '<form action="/database" method="get"><button type="submit">Back to Database</button></form>';
  html += '<form action="/" method="get" style="margin-top: 10px;"><button type="submit">Back to Home</button></form>';

  res.send(html);
});

app.get('/database/:folder/:file', (req, res) => {
  const folder = req.params.folder;
  const file = req.params.file;
  const filePath = path.join(dataPath, folder, file);
  let calculateProfit = false;
  let game = "";

  if (fs.existsSync(filePath)) {
    const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    
    if(folder.includes("keno")) {
      calculateProfit = true;
      game = "keno";
    } if (folder.includes("pick3")) {
      calcProfit = true;
      game = "pick3";
    }

    // Generate HTML content
    let html = `
      <h1>${file} Results</h1>
      <h2>Current Prediction</h2>
      ${generateTable(
        jsonData.currentPrediction, 
        'Current Prediction', 
        [].concat(...jsonData.currentPrediction.map(prediction => prediction.predictions.flat().filter(num => jsonData.realResult.includes(num)))),
        calculateProfit,
        game
      )}
      <h2>Real Result</h2>
      ${generateList(jsonData.realResult, 'Real Result')}
      <h2>Matching Numbers</h2>
      <p><strong>Best Match Index:</strong> ${jsonData.matchingNumbers.model}</p>
      <p><strong>Best Match Sequence:</strong> ${generateList(jsonData.matchingNumbers.prediction)}</p>
      <!--<p><strong>Matching Numbers:</strong> ${generateList([].concat(...jsonData.currentPrediction.map(prediction => prediction.predictions.flat().filter(num => jsonData.realResult.includes(num)))))}</p>--!>
      <h2>New Prediction</h2>
      ${generateTable(
        jsonData.newPrediction,
        'New Prediction', 
        [],
        calculateProfit,
        game
      )}
      <form action="/database/${folder}" method="get" style="margin-top: 20px;"><button type="submit">Back to ${folder}</button></form>
      <form action="/" method="get"><button type="submit">Back to Home</button></form>
    `;

    res.send(html);
  } else {
    res.status(404).send('File not found');
  }
});

// Router to display available PNGs
app.get('/models', (req, res) => {
  const folders = fs.readdirSync(modelsPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((dir) => dir.name);

  let html = '<h1>Available Models</h1><ul>';
  folders.forEach((folder) => {
    html += `<li><form action="/models/${folder}" method="get"><button type="submit">${folder}</button></form></li>`;
  });
  html += '</ul><form action="/" method="get"><button type="submit">Back to Home</button></form>';

  res.send(html);
});

// Router to list PNGs in subfolders
app.get('/models/:folder', (req, res) => {
  const folder = req.params.folder;
  const folderPath = path.join(modelsPath, folder);

  if (!fs.existsSync(folderPath)) {
    return res.status(404).send('Folder not found');
  }

  const files = fs.readdirSync(folderPath).filter((file) => file.endsWith('.png'));

  let html = `<h1>Images in ${folder}</h1><ul>`;
  files.forEach((file) => {
    html += `
    <li>
      <h2>${file}</h2>
      <img src="/models/${folder}/${file}" alt="${file}" style="max-width: 100%; margin: 10px;">
    </li>`;
  });
  html += '</ul><form action="/models" method="get"><button type="submit">Back to Models</button></form>';
  html += '<form action="/" method="get" style="margin-top: 10px;"><button type="submit">Back to Home</button></form>';

  res.send(html);
});

// Home Page
app.get('/', (req, res) => {
  const folders = fs.readdirSync(dataPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((dir) => dir.name);

  let html = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Sequence Predictor Results</title>
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
      <style>
        .folder {
          margin: 10px 0;
          border: 1px solid #ddd;
          border-radius: 5px;
          padding: 10px;
        }
        .folder-title {
          cursor: pointer;
          font-weight: bold;
          background-color: #f9f9f9;
          padding: 10px;
          border-radius: 5px;
        }
        .folder-content {
          display: none;
          margin-top: 10px;
        }
        .charts-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 10px;
        }
        canvas {
          width: 100%;
          height: auto;
          max-height: 150px;
        }
        body {
         background-color: white;
        }
        .save-btn {
          margin-top: 20px;
        }
      </style>
    </head>
    <body>
      <h1>Sequence Predictor Results</h1>
      <div class="button-container" style="margin-top: 20px;">
        <form action="/database" method="get" style="display: inline;">
          <button type="submit">Go to Database</button>
        </form>
        <button id="saveAsPng" class="save-btn" style="display: inline;">Save as PNG</button>
      </div>
      <div>
  `;

  folders.forEach((folder) => {
    const folderPath = path.join(dataPath, folder);
    const files = fs.readdirSync(folderPath)
      .filter((file) => file.endsWith('.json'))
      .sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));

    if (files.length > 0) {
      const latestFile = files[0];
      const latestFilePath = path.join(folderPath, latestFile);
      const jsonData = JSON.parse(fs.readFileSync(latestFilePath, 'utf-8'));

      html += `
        <div class="folder">
          <div class="folder-title">${folder}</div>
          <div class="folder-content">
            <h3>${latestFile}</h3>
            ${generateTable(jsonData.newPrediction, 'New Prediction')}
            <div class="charts-grid">
      `;

      if (jsonData.numberFrequency) {
        const labels = Object.keys(jsonData.numberFrequency);
        const dataValues = Object.values(jsonData.numberFrequency);
      
        html += `
          <div>
            <h4>Number Frequency</h4>
            <canvas id="chart-${folder}"></canvas>
          </div>
          <script>
            const ctx${folder} = document.getElementById('chart-${folder}').getContext('2d');
            new Chart(ctx${folder}, {
              type: 'bar',
              data: {
                labels: ${JSON.stringify(labels)},
                datasets: [{
                  label: 'Probability',
                  data: ${JSON.stringify(dataValues)},
                  borderColor: 'rgba(75, 192, 192, 1)',
                  backgroundColor: 'rgba(75, 192, 192, 0.2)',
                  borderWidth: 1,
                }]
              },
              options: {
                responsive: true,
                plugins: {
                  legend: {
                    display: false,
                  }
                },
                scales: {
                  x: { title: { display: true, text: "Number" } },
                  y: { title: { display: true, text: "Probability" } }
                }
              }
            });
          </script>
        `;
      }

      html += `
            </div>
            <form action="/database/${folder}/${latestFile}" method="get" style="margin-top: 10px;">
              <button type="submit">View Comparison</button>
            </form>
          </div>
        </div>
      `;
    }
  });

  html += `
      </div>
      <script>
        document.addEventListener('DOMContentLoaded', () => {
          const folderTitles = document.querySelectorAll('.folder-title');
          folderTitles.forEach(title => {
            title.addEventListener('click', () => {
              const content = title.nextElementSibling;
              content.style.display = content.style.display === 'none' || !content.style.display ? 'block' : 'none';
            });
          });
        });

        document.getElementById('saveAsPng').addEventListener('click', () => {
          html2canvas(document.body).then((canvas) => {
            const link = document.createElement('a');
            link.download = 'page_snapshot.png';
            link.href = canvas.toDataURL();
            link.click();
          });
        });
      </script>
    </body>
    </html>
  `;

  res.send(html);
});

app.use(express.json()); // To parse JSON bodies

app.post('/playedNumbers', (req, res) => {
  let playedNumbers = req.body.playedNumbers;

  if (!playedNumbers) {
    return res.status(400).send('No numbers provided');
  }

  // Ensure it's always an array
  if (!Array.isArray(playedNumbers)) {
    playedNumbers = [playedNumbers];
  }

  // Convert all values to Numbers
  playedNumbers = playedNumbers.map(n => Number(n)).filter(n => !isNaN(n));

  if (playedNumbers.length === 0) {
    return res.status(400).send('Provided values are not valid numbers');
  }

  console.log('Received numbers:', playedNumbers);

  selectedPlayedNumbers = playedNumbers;

  res.send(`Played numbers: ${playedNumbers.join(', ')}`);
});

app.post('/playedModel', (req, res) => {
  let playedModel = req.body.selectedModel;

  if (!Array.isArray(playedModel)) {
    playedModel = [playedModel];
  }

  console.log('Received selectedModel:', playedModel);

  selectedModel = playedModel; 

  res.send('Selected model saved');
});

app.post('/resetPlayedNumbers', (req, res) => {

  selectedPlayedNumbers = [4,5,6,7,8,9,10]; // To select played numbers  for Keno
  res.send('Played numbers reset');
});

app.post('/resetSelectedModel', (req, res) => {
  selectedModel = "all"; // To select with wich model's predictions is played
  res.send('Selected model reset');
});

// Start the server
app.listen(config.PORT, () => {
  console.log(`Server running at http://${config.INTERFACE}:${config.PORT}`);
});
