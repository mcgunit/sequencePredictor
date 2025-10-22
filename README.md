
# Sequence Predictor

## To check

- Small-to-Medium Data:
    Use TCNs or hybrid models (CNN + LSTM).
- Large Data:
    Transformers are ideal due to scalability and performance.
- If Sequence Dependencies are Weak:
    Pure CNN models like ResNet, EfficientNet, or MobileNet (better than GoogLeNet for general tasks).

---

### **1. AI Models for Sequence Prediction**
#### **

### **1. AI Models for Sequence Prediction**

#### (a) Recurrent Neural Networks (RNNs)
- **Key Models:** Simple RNNs, LSTMs (Long Short-Term Memory), GRUs (Gated Recurrent Units)
- **Why Use Them:** RNNs are designed for sequential data, where the order of inputs matters. LSTMs and GRUs are especially good at capturing long-term dependencies in time-series data.
- **Applications:** Predicting the next sequence of numbers by learning temporal dependencies.

#### (b) Transformer Models
- **Key Models:** Vanilla Transformer, GPT-like models, and Time Series Transformers.
- **Why Use Them:** Transformers handle sequence data using self-attention mechanisms, capturing both short-term and long-term dependencies. They are computationally efficient for long sequences.
- **Applications:** Advanced modeling tasks, especially when the sequence length is large.

#### (c) Convolutional Neural Networks (CNNs) for Sequences
- **Why Use Them:** 1D-CNNs can extract local patterns in time-series data. When combined with RNNs or used as standalone models, they are efficient for learning short-term features in the sequence.
- **Applications:** Feature extraction followed by sequence prediction.

#### (d) Encoder-Decoder Architectures
- **Why Use Them:** Useful for mapping input sequences to output sequences. They work well with RNNs or Transformers and are widely used in sequence-to-sequence problems.
- **Applications:** Multi-step predictions like forecasting the next 3 numbers in a sequence.

---

### **2. Statistical Models for Sequence Prediction**
#### (a) Autoregressive Integrated Moving Average (ARIMA)
- **Why Use It:** Captures temporal patterns, trends, and seasonality in data. Suitable for univariate time series.
- **Applications:** Predicting the next numbers if the sequence exhibits consistent patterns over time.

#### (b) Vector Autoregression (VAR)
- **Why Use It:** Handles multivariate time series with interdependencies between variables.
- **Applications:** Predicting sequences involving multiple correlated numbers.

#### (c) Exponential Smoothing (ETS Models)
- **Why Use It:** Useful for forecasting sequences with trends or seasonality.
- **Applications:** Time-series data with less complex dependencies.

#### (d) Hidden Markov Models (HMMs)
- **Why Use It:** Probabilistic models ideal for sequences where the next state depends on the current state.
- **Applications:** Predicting discrete sequences or sequences with hidden states.

---

### **3. Hybrid Models**
- **LSTM-ARIMA:** Combines LSTM’s ability to capture nonlinear patterns with ARIMA’s ability to handle seasonality.
- **DeepAR:** Probabilistic forecasting model using recurrent neural networks.

---

### **4. Choosing the Right Model**
- **Data Complexity:** If the relationships are complex and nonlinear, neural networks (LSTMs, Transformers) are better. For simpler data, ARIMA or VAR may suffice.
- **Sequence Length:** For long sequences, Transformers or 1D-CNNs are more efficient.
- **Multivariate or Univariate:** Use VAR for multivariate data; ARIMA for univariate.
- **Amount of Data:** Neural networks typically require more data, while statistical models work well with smaller datasets.



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
