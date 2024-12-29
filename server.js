const express = require('express');
const path = require('path');
const fs = require('fs');

const config = require("./config");

const app = express();

// Paths
const dataPath = path.join(__dirname, 'data', 'database');
const modelsPath = path.join(__dirname, 'data', 'models');

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
  table += `<th style="padding: 5px; text-align: center; font-weight: bold; width: 100px; max-width: 10px;">#</th>`; // Index column header
  headers.forEach((header) => {
      table += `<th style="padding: 5px; text-align: center; width: 100px; min-width: 100px; font-weight: bold;">${header}</th>`;
  });
  table += '</tr>';

  // Add rows with data and index
  data.forEach((row, rowIndex) => {
      table += '<tr>';
      table += `<td style="padding: 5px; text-align: center; font-weight: bold; width: 10px; max-width: 10px;">${rowIndex + 1}</td>`; // Row index
      row.forEach((cell) => {
          const isMatching = matchingNumbers.includes(cell); // Check if the cell value is in matchingNumbers
          table += `<td style="padding: 5px; text-align: center; width: 100px; min-width: 100px; ${isMatching ? 'background-color: green; color: white;' : ''}">${cell}</td>`;
      });
      table += '</tr>';
  });

  table += '</table>';
  return table;
}

function generateTableWithMostFrequentNumbers(data, title = '', mostFrequentNumbers = [], type = 'euromillions') {

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
  table += `<th style="padding: 5px; text-align: center; font-weight: bold; width: 10px; max-width: 10px;">#</th>`; // Index column header
  headers.forEach((header) => {
    table += `<th style="padding: 5px; text-align: center; width: 100px; min-width: 100px; font-weight: bold;">${header}</th>`;
  });
  table += '</tr>';

  // Add rows with data and index
  data.forEach((row, rowIndex) => {
    table += '<tr>';
    table += `<td style="padding: 5px; text-align: center; font-weight: bold; width: 10px; max-width: 10px;">${rowIndex + 1}</td>`; // Row index
    row.forEach((cell) => {
      /*
      const isMostFrequent = mostFrequentNumbers.includes(cell); // Check if the cell value is in mostFrequentNumbers
      table += `<td style="padding: 5px; text-align: center; ${
        isMostFrequent ? 'background-color: orange; color: black;' : ''
      }">${cell}</td>`;
      */
      const isMostFrequent = mostFrequentNumbers.includes(cell); // Check if the cell value is in mostFrequentNumbers
      table += `<td style="padding: 5px; text-align: center;">${cell}</td>`;
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
    table += `<td style="padding: 5px; text-align: center; width: 100px; min-width: 100px;">${item}</td>`;
  });
  table += '</tr>';
  table += '</table>';
  return table;
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

    // Determine the type based on the folder name
    const type = folder.includes('lotto') ? 'lotto' : folder.includes('euromillions') ? 'euromillions' : 'generic';

    // Calculate most frequent numbers in the folder
    const folderPath = path.join(dataPath, folder);
    const files = fs.readdirSync(folderPath).filter((f) => f.endsWith('.json'));

    const allNumbers = files.flatMap((f) => {
      const fileData = JSON.parse(fs.readFileSync(path.join(folderPath, f), 'utf-8'));
      return fileData.newPrediction.flat();
    });

    const frequencyMap = allNumbers.reduce((map, num) => {
      map[num] = (map[num] || 0) + 1;
      return map;
    }, {});

    const mostFrequentNumbers = Object.entries(frequencyMap)
      .sort(([, a], [, b]) => b - a)
      .slice(0, type === 'euromillions' ? 7 : 6)
      .map(([num]) => parseInt(num, 10));

    // Calculate most frequent numbers in the current prediction
    const currentPredictionNumbers = jsonData.currentPrediction.flat();
    const currentFrequencyMap = currentPredictionNumbers.reduce((map, num) => {
      map[num] = (map[num] || 0) + 1;
      return map;
    }, {});

    const mostFrequentCurrentNumbers = Object.entries(currentFrequencyMap)
      .sort(([, a], [, b]) => b - a)
      .slice(0, type === 'euromillions' ? 7 : 6)
      .map(([num]) => parseInt(num, 10));

    // Generate HTML content
    let html = `
      <h1>${file} Results</h1>
      <h2>Current Prediction</h2>
      ${generateTable(jsonData.currentPrediction, 'Current Prediction', jsonData.matchingNumbers.matchingNumbers, type)}
      <h3>Most Frequent Numbers in Current Prediction</h3>
      ${generateList(mostFrequentCurrentNumbers, 'Most Frequent Numbers (Current Prediction)')}
      <h2>Real Result</h2>
      ${generateList(jsonData.realResult, 'Real Result')}
      <h2>Matching Numbers</h2>
      <p><strong>Best Match Index:</strong> ${jsonData.matchingNumbers.bestMatchIndex+1}</p>
      <p><strong>Best Match Sequence:</strong> ${generateList(jsonData.matchingNumbers.bestMatchSequence)}</p>
      <p><strong>Matching Numbers:</strong> ${generateList(jsonData.matchingNumbers.matchingNumbers)}</p>
      <h2>New Prediction</h2>
      ${generateTable(jsonData.newPrediction, 'New Prediction', [], type)}
      <h3>Most Frequent Numbers in ${folder}</h3>
      ${generateList(mostFrequentNumbers, 'Most Frequent Numbers (New Prediction)')}
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
      <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    </head>
    <body>
      <h1>Sequence Predictor Results</h1>
      <h2>Models</h2>
      <ul>
        <li><form action="/models" method="get"><button type="submit">AI Models</button></form></li>
        <li><form action="/database" method="get"><button type="submit">View All Database Data</button></form></li>
      </ul>
      <h2>Predictions For Next Drawing</h2>
      <ul>
  `;

  const allFrequentNumbers = { lotto: [], euromillions: { main: [], stars: [] } };

  folders.forEach((folder) => {
    const folderPath = path.join(dataPath, folder);
    const files = fs.readdirSync(folderPath)
      .filter((file) => file.endsWith('.json'))
      .sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));

    if (files.length > 0) {
      const latestFile = files[0];
      const latestFilePath = path.join(folderPath, latestFile);
      const jsonData = JSON.parse(fs.readFileSync(latestFilePath, 'utf-8'));

      const type = folder.includes('lotto') ? 'lotto' : folder.includes('euromillions') ? 'euromillions' : 'generic';

      let mostFrequentMain = [];
      let mostFrequentStars = [];
      let mostFrequentNumbers = [];

      if (type === 'euromillions') {
        const allNumbersMain = [];
        const allNumbersStars = [];

        files.forEach((file) => {
          const data = JSON.parse(fs.readFileSync(path.join(folderPath, file), 'utf-8'));
          if (Array.isArray(data.newPrediction)) {
            data.newPrediction.forEach((row) => {
              allNumbersMain.push(...row.slice(0, 5)); // First 5 numbers
              allNumbersStars.push(...row.slice(5)); // Last 2 stars
            });
          }
        });

        const frequencyMain = allNumbersMain.reduce((map, num) => {
          map[num] = (map[num] || 0) + 1;
          return map;
        }, {});

        const frequencyStars = allNumbersStars.reduce((map, num) => {
          map[num] = (map[num] || 0) + 1;
          return map;
        }, {});

        mostFrequentMain = Object.entries(frequencyMain)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 5)
          .map(([num]) => parseInt(num, 10)); // Get unique numbers

        mostFrequentStars = Object.entries(frequencyStars)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 2)
          .map(([num]) => parseInt(num, 10)); // Get unique numbers

        //console.log("Most Frequent Main: ", mostFrequentMain);
        //console.log("Most Frequent Stars: ", mostFrequentStars);

        allFrequentNumbers.euromillions.main.push(...mostFrequentMain);
        allFrequentNumbers.euromillions.stars.push(...mostFrequentStars);
      } else if (type === 'lotto') {
        const allNumbers = files.flatMap(file => {
          const data = JSON.parse(fs.readFileSync(path.join(folderPath, file), 'utf-8'));
          return Array.isArray(data.newPrediction) ? data.newPrediction.flat() : [];
        });

        const frequencyMap = allNumbers.reduce((map, num) => {
          map[num] = (map[num] || 0) + 1;
          return map;
        }, {});

        mostFrequentNumbers = Object.entries(frequencyMap)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 6)
          .map(([num]) => parseInt(num, 10)); // Get unique numbers

        allFrequentNumbers.lotto.push(...mostFrequentNumbers);
      }

      html += `
        <li>
          <h2>${folder}</h2>
          ${generateTableWithMostFrequentNumbers(jsonData.newPrediction, 'New Prediction', type === 'euromillions' ? [...mostFrequentMain, ...mostFrequentStars] : mostFrequentNumbers, type)}
          <p><strong>Most Occurring Numbers:</strong> ${type === 'euromillions' ? `Main: ${mostFrequentMain.join(', ')} | Stars: ${mostFrequentStars.join(', ')}` : mostFrequentNumbers.join(', ')}</p>
          <form action="/database/${folder}/${latestFile}" method="get"><button type="submit">View Full Details</button></form>
        </li>
      `;
    }
  });

  
  // Print button
  html += `
      </ul>
      <button id="saveAsPng" style="margin-top: 20px;">Save as PNG</button>
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
