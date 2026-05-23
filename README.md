
# Sequence Predictor

## The main idea

MarkovMonteCarlo:
    "Given recent transition patterns and column frequencies, which numbers are repeatedly generated?"

PoissonMonteCarlo:
    "Given historical count rates per position, which numbers repeatedly occur in simulated count scenarios?"

LaplaceMonteCarlo:
    "Given the typical center/spread of each position, which values repeatedly appear in simulated positional draws?"

Backtester:
    "Which of these actually produces more 2+, 3+, and 4+ hits than baselines over many historical draws?"


## Installation

### For Predictor (Python)

#### Virtual env

Create a virtual env:
```
python3 -m venv ~/sequencePredictor
```

Activate env:

```
source ~/sequencePredictor/bin/activate
```

To install, you will need to have Python 3.x and the following libraries installed:
- numpy
- tensorflow
- keras
- art
- keras-tcn

You can install these libraries using pip by running the following command:

Using the requirements file:

```
    python3 -m pip install -r requirements.txt
```

For CPU only: 
```
    python3 -m pip install numpy tensorflow==2.18 keras art pandas scikit-learn matplotlib keras-tcn==3.1.2
```

For GPU enabled:

```
    python3 -m pip install numpy tensorflow[and-cuda]==2.18 keras art pandas scikit-learn matplotlib keras-tcn==3.1.2
```

#### Docker

Check the dockerfile.

To build:

```
    docker build -t sequence_predictor .
```

Run:

```
    docker run --rm -it -u $(id -u) -v {absolute path to sequencePredictor repo}:/opt/sequencePredictor sequence_predictor /bin/bash
```

From this point you are inside the docker container with bash active. Now you can run or test code.

### For server (NodeJs)

Run in root of folder (where the package.json is located):

```
    npm i
```

If pm2 is needed also run:

```
    npm i pm2 -g
```
## How to run prediction

To run the complete flow run:

```
    python3 Predictor.py
```

To test model specific for example LSTM run:

```
    python3 LSTM.py
```

Check the __main__ section of the LSTM.py or GRU.py for pointing to data and set parameters for testing.

## Run server

The server is a NodeJS server with a plain simple html server side rendered front-end. No dependencies or heavy webpacks needed.
The server will listen on 0.0.0.0 and port 30001. This can be changed in the config.js file.

To run the server use the command:

```
    npm start
```

Pm2 can also be used. For this run:

```
    pm2 start server.js --name predictor --time --watch 
```

Then for saving this in the pm2 run list (needed for auto start):

```
    pm2 save
```

For having it with auto start at boot:

```
    pm2 startup
```

## Testing

To test a model when modifying or tuning you can run the LSTM.py or GRU.py directly and check the __main__ section. Use the test folder for trainingData and models if you don't want to touch the actual data (highly recommended). 
For testing You, in the `test` folder, can manually remove the last result from the .csv files and put it in the `sequenceToPredict_xxx.json` file. Then when tuning or changing the model, the results are compared. **It is of importance to take the latest result out of the test data**.

## Fetching data

It is possible to download the csv data containing the real draws on the website or via the url. But it is also possible to use the "API" with the following link: https://apim.prd.natlot.be/api/v4/draw-games/draws?status=PAYABLE&previous-draws=5 or for specific: https://apim.prd.natlot.be/api/v4/draw-games/draws?status=PAYABLE&date-from=1746057600000&size=62&date-to=1751414400000&game-names=Keno


## Disclaimer

The code within this repository comes with no guarantee, the use of this code is your responsibility. I take NO responsibility and/or liability for how you choose to use any of the source code available here. By using any of the files available in this repository, you understand that you are AGREEING TO USE AT YOUR OWN RISK. Once again, ALL files available here are for EDUCATION and/or RESEARCH purposes ONLY.
Please keep in mind that while LSTM.py uses advanced machine learning techniques to predict lottery numbers, there is no guarantee that its predictions will be accurate. Lottery results are inherently random and unpredictable, so it is important to use LSTM responsibly and not rely solely on its predictions.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this project as long as you give attribution to the original author.
