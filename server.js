const express = require('express');
const path = require('path');
const fs = require('fs');

const config = require("./config");

const app = express();

// Paths
const dataPath = path.join(__dirname, 'data', 'database');
const modelsPath = path.join(__dirname, 'data', 'models');

function generateTable(data, title = '', matchingNumbers = [], calcProfit = false) {
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
        const profit = calculateProfit(numbersPlayed, correctNumbers);
        table += `<td style="padding: 5px; text-align: center; background: #f8f9fa;">${profit} €</td>`;
      }
      table += '</tr>';
    });
  });

  table += '</table>';
  return table;
}

function calculateProfit(numbersPlayed, correctNumbers) {
  // Define the Keno payout table
  const payoutTable = {
    8: { 0: 3, 5: 4, 6: 10, 7: 100, 8: 10000 },
    7: { 0: 3, 5: 3, 6: 30, 7: 3000 },
    6: { 3: 1, 4: 4, 5: 20, 6: 200 },
    5: { 3: 2, 4: 5, 5: 150 },
    4: { 2: 1, 3: 2, 4: 30 },
    3: { 2: 1, 3: 16 },
    2: { 2: 6.50 }
  };

  // Calculate the profit based on the payout table
  if (payoutTable[numbersPlayed] && payoutTable[numbersPlayed][correctNumbers]) {
    return payoutTable[numbersPlayed][correctNumbers];
  } else {
    return 0;
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

// Router to list JSON files in database subfolders
app.get('/database/:folder', (req, res) => {
  const folder = req.params.folder;
  const folderPath = path.join(dataPath, folder);

  if (!fs.existsSync(folderPath)) {
    return res.status(404).send('Folder not found');
  }

  const files = fs.readdirSync(folderPath)
    .filter((file) => file.endsWith('.json'))
    .sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', ''))); // Sort files by date

  let html = `<h1>JSON Files in ${folder}</h1><ul>`;
  files.forEach((file) => {
    html += `<li><form action="/database/${folder}/${file}" method="get"><button type="submit">${file}</button></form></li>`;
  });
  html += '<form action="/database" method="get"><button type="submit">Back to Database</button></form>';
  html += '<form action="/" method="get" style="margin-top: 10px;"><button type="submit">Back to Home</button></form>';

  res.send(html);
});

app.get('/database/:folder/:file', (req, res) => {
  const folder = req.params.folder;
  const file = req.params.file;
  const filePath = path.join(dataPath, folder, file);
  let calculateProfit = false;

  if (fs.existsSync(filePath)) {
    const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    
    if(folder === "keno") {
      calculateProfit = true;
    }

    // Generate HTML content
    let html = `
      <h1>${file} Results</h1>
      <h2>Current Prediction</h2>
      ${generateTable(
        jsonData.currentPrediction, 
        'Current Prediction', 
        [].concat(...jsonData.currentPrediction.map(prediction => prediction.predictions.flat().filter(num => jsonData.realResult.includes(num)))),
        calculateProfit
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
        calculateProfit
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

// Start the server
app.listen(config.PORT, () => {
  console.log(`Server running at http://${config.INTERFACE}:${config.PORT}`);
});
