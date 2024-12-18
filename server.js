const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3001;

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

  res.send(html);
});

// Serve JSON files directly
app.get('/database/:folder/:file', (req, res) => {
  const folder = req.params.folder;
  const file = req.params.file;
  const filePath = path.join(dataPath, folder, file);

  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
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
    html += `<li><img src="/models/${folder}/${file}" alt="${file}" style="max-width: 300px; margin: 10px;"></li>`;
  });
  html += '</ul><a href="/models">Back to Models</a>';

  res.send(html);
});

// Default route
app.get('/', (req, res) => {
  res.send(`
    <h1>Sequence Predictor Results</h1>
    <ul>
      <li><a href="/database">Database Data</a></li>
      <li><a href="/models">Model Images</a></li>
    </ul>
  `);
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
