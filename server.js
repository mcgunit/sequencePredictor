const express = require('express');
const path = require('path');
const fs = require('fs');

const config = require("./config");

const app = express();

// Paths
const dataPath = path.join(__dirname, 'data', 'database');
const modelsPath = path.join(__dirname, 'data', 'models');

// Serve static files
app.use('/models', express.static(modelsPath)); // Serves PNGs from models directory

// Router to handle database data
app.get('/database', (req, res) => {
  const folders = fs.readdirSync(dataPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((dir) => dir.name);

  let html = '<h1>Available Database Folders</h1><ul>';
  folders.forEach((folder) => {
    html += `<li><a href="/database/${folder}">${folder}</a></li>`;
  });
  html += '</ul><a href="/">Back to Home</a>';

  res.send(html);
});

// Router to list JSON files in database subfolders
app.get('/database/:folder', (req, res) => {
  const folder = req.params.folder;
  const folderPath = path.join(dataPath, folder);

  if (!fs.existsSync(folderPath)) {
    return res.status(404).send('Folder not found');
  }

  const files = fs.readdirSync(folderPath).filter((file) => file.endsWith('.json'));

  let html = `<h1>JSON Files in ${folder}</h1><ul>`;
  files.forEach((file) => {
    html += `<li><a href="/database/${folder}/${file}" target="_blank">${file}</a></li>`;
  });
  html += '</ul><a href="/database">Back to Database</a>';
  html += '<a href="/" style="display: block; margin-top: 10px;">Back to Home</a>';

  res.send(html);
});

// Serve JSON files with a formatted HTML layout
app.get('/database/:folder/:file', (req, res) => {
  const folder = req.params.folder;
  const file = req.params.file;
  const filePath = path.join(dataPath, folder, file);

  if (fs.existsSync(filePath)) {
    const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

    // Determine the type based on the folder name
    const type = folder.includes('lotto') ? 'lotto' : folder.includes('euromillions') ? 'euromillions' : 'generic';

    // Generate HTML content
    let html = `
      <h1>${file} Results</h1>
      <h2>Current Prediction</h2>
      ${generateTable(jsonData.currentPrediction, 'Current Prediction', jsonData.matchingNumbers.matchingNumbers, type)}
      <h2>Real Result</h2>
      ${generateList(jsonData.realResult, 'Real Result')}
      <h2>Matching Numbers</h2>
      <p><strong>Best Match Index:</strong> ${jsonData.matchingNumbers.bestMatchIndex}</p>
      <p><strong>Best Match Sequence:</strong> ${generateList(jsonData.matchingNumbers.bestMatchSequence)}</p>
      <p><strong>Matching Numbers:</strong> ${generateList(jsonData.matchingNumbers.matchingNumbers)}</p>
      <h2>New Prediction</h2>
      ${generateTable(jsonData.newPrediction, 'New Prediction', [], type)}
      <a href="/database/${folder}" style="display: block; margin-top: 20px;">Back to ${folder}</a>
      <a href="/">Back to Home</a>
    `;

    res.send(html);
  } else {
    res.status(404).send('File not found');
  }
});


function generateTable(data, title = '', matchingNumbers = [], type = 'euromillions') {
  let table = '<table border="1" style="border-collapse: collapse; width: 100%;">';

  // Add title as caption if provided
  if (title) table += `<caption><strong>${title}</strong></caption>`;

  // Determine headers based on the type
  let headers = [];
  if (type === 'lotto') {
      headers = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Number 6', 'Bonus'];
  } else if (type === 'euromillions') {
      headers = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Star 1', 'Star 2'];
  } else {
      headers = data.length > 0 ? Array.from({ length: data[0].length }, (_, i) => `Column ${i + 1}`) : [];
  }

  // Add header row with column names
  table += '<tr>';
  table += `<th style="padding: 5px; text-align: center; font-weight: bold;">#</th>`; // Index column header
  headers.forEach((header) => {
      table += `<th style="padding: 5px; text-align: center; font-weight: bold;">${header}</th>`;
  });
  table += '</tr>';

  // Add rows with data and index
  data.forEach((row, rowIndex) => {
      table += '<tr>';
      table += `<td style="padding: 5px; text-align: center; font-weight: bold;">${rowIndex + 1}</td>`; // Row index
      row.forEach((cell) => {
          const isMatching = matchingNumbers.includes(cell); // Check if the cell value is in matchingNumbers
          table += `<td style="padding: 5px; text-align: center; ${isMatching ? 'background-color: green; color: white;' : ''}">${cell}</td>`;
      });
      table += '</tr>';
  });

  table += '</table>';
  return table;
}

function generateList(data, title = '') {
  let table = '<table border="1" style="border-collapse: collapse; width: 100%;">';
  if (title) table += `<caption><strong>${title}</strong></caption>`;
  table += '<tr>';
  data.forEach((item) => {
      table += `<td style="padding: 5px; text-align: center;">${item}</td>`;
  });
  table += '</tr>';
  table += '</table>';
  return table;
}

// Router to display available PNGs
app.get('/models', (req, res) => {
  const folders = fs.readdirSync(modelsPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((dir) => dir.name);

  let html = '<h1>Available Models</h1><ul>';
  folders.forEach((folder) => {
    html += `<li><a href="/models/${folder}">${folder}</a></li>`;
  });
  html += '</ul><a href="/">Back to Home</a>';

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
  html += '</ul><a href="/models">Back to Models</a>';
  html += '<a href="/" style="display: block; margin-top: 10px;">Back to Home</a>';

  res.send(html);
});

// Home Page
app.get('/', (req, res) => {
  const folders = fs.readdirSync(dataPath, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((dir) => dir.name);

  let html = `
    <h1>Sequence Predictor Results</h1>
    <h2>Models</h2>
    <ul>
      <li><a href="/models">AI Models</a></li>
      <li><a href="/database">View All Database Data</a></li>
    </ul>
    <h2>Latest Predictions</h2>
    <ul>
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

      // Determine the type based on the folder name
      const type = folder.includes('lotto') ? 'lotto' : folder.includes('euromillions') ? 'euromillions' : 'generic';

      html += `
        <li>
          <h2>${folder}</h2>
          <p><strong>Date:</strong> ${latestFile.replace('.json', '')}</p>
          ${generateTable(jsonData.newPrediction, 'New Prediction', [], type)}
          <a href="/database/${folder}/${latestFile}" target="_blank">View Full Details</a>
        </li>
      `;
    }
  });

  html += `</ul>`;

  res.send(html);
});

// Start the server
app.listen(config.PORT, () => {
  console.log(`Server running at http://${config.INTERFACE}:${config.PORT}`);
});
