const express = require('express');
const path = require('path');
const fs = require('fs');

const config = require("./config");

const app = express();

// Paths
const dataPath = path.join(__dirname, 'data', 'database');
const modelsPath = path.join(__dirname, 'data', 'models');

function generateTable(data, title = '', matchingNumbers = []) {
  let table = '<table border="1" style="border-collapse: collapse; width: 100%;">';

  // Add title as caption if provided
  if (title) table += `<caption><strong>${title}</strong></caption>`;

  // Determine headers based on the type
  let headers = data.length > 0 
    ? Array.from({ length: data[0].length }, (_, i) => `Column ${i + 1}`) 
    : [];

  // Add header row
  table += '<tr>';
  table += `<th style="padding: 5px; text-align: center; font-weight: bold; width: 100px;">#</th>`; // Index column
  headers.forEach((header) => {
    table += `<th style="padding: 5px; text-align: center; font-weight: bold;">${header}</th>`;
  });
  table += '</tr>';

  // Add rows with data
  data.forEach((row, rowIndex) => {
    if (rowIndex < 10) {
      table += '<tr>';
      table += `<td style="padding: 5px; text-align: center; font-weight: bold;">${rowIndex + 1}</td>`; // Row index
      row.forEach((cell) => {
        const isMatching = matchingNumbers.includes(cell); // Highlight if matching
        table += `<td style="padding: 5px; text-align: center; ${isMatching ? 'background-color: green; color: white;' : ''}">${cell}</td>`;
      });
    }
  });

  table += '</table>';
  return table;
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

// Serve JSON files with a formatted HTML layout
app.get('/database/:folder/:file', (req, res) => {
  const folder = req.params.folder;
  const file = req.params.file;
  const filePath = path.join(dataPath, folder, file);

  if (fs.existsSync(filePath)) {
    const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));


    // Generate HTML content
    let html = `
      <h1>${file} Results</h1>
      <h2>Current Prediction</h2>
      ${generateTable(jsonData.currentPrediction, 'Current Prediction', jsonData.matchingNumbers.matchingNumbers)}
      <h2>Real Result</h2>
      ${generateList(jsonData.realResult, 'Real Result')}
      <h2>Matching Numbers</h2>
      <p><strong>Best Match Index:</strong> ${jsonData.matchingNumbers.bestMatchIndex+1}</p>
      <p><strong>Best Match Sequence:</strong> ${generateList(jsonData.matchingNumbers.bestMatchSequence)}</p>
      <p><strong>Matching Numbers:</strong> ${generateList(jsonData.matchingNumbers.matchingNumbers)}</p>
      <h2>New Prediction</h2>
      ${generateTable(jsonData.newPrediction, 'New Prediction', [])}
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
      </style>
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
      </script>
    </head>
    <body>
      <h1>Sequence Predictor Results</h1>
      <h2>Predictions</h2>
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
            ${generateTable(jsonData.newPrediction, 'New Prediction')}
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
      <button id="saveAsPng" style="margin-top: 20px;">Save as PNG</button>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
      <script>
        document.getElementById('saveAsPng').addEventListener('click', () => {
          html2canvas(document.body).then((canvas) => {
            const link = document.createElement('a');
            link.download = 'home_page.png';
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
