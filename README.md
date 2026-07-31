
# Sequence Predictor

## The main idea

This project compares several probabilistic/statistical models, plus deep learning (LSTM/TCN) and gradient boosting (XGBoost), for lottery-style number prediction across multiple games (Euromillions, Lotto, EuroDreams, Keno, Pick3, VikingLotto).

### Statistical models (`src/`)

Each model generates predictions based on a different statistical interpretation of historical draw data:

- **Markov** builds a (configurable order) transition chain over historical draws, with recency weighting, pair-decay, smoothing, and softmax-based subset selection.
- **MarkovMonteCarlo** wraps a `Markov` instance and repeatedly samples/votes over many simulated tickets (instead of a single deterministic sample) to pick the numbers that win the most votes.
- **MarkovBayesian** / **MarkovBayesianEnhanced** apply Bayesian smoothing (alpha, softmax temperature, min-occurrence thresholds) on top of the Markov transition counts.
- **PoissonMonteCarlo** estimates which numbers repeatedly occur under historical position-based count rates, sampled via Monte Carlo simulation.
- **PoissonMarkov** blends Poisson and Markov probabilities using a tunable weight.
- **LaplaceMonteCarlo** estimates which values repeatedly appear when each sorted position is modeled by its historical center and spread.
- **HybridStatisticalModel** combines several of the above signals (softmax temperature, alpha, min occurrences, simulation count) into one blended model.

Every model shares the same interface (`setDataPath`, `run(...)`/`run_model_with_special_column`) so the Backtester and Predictor can drive them interchangeably. Euromillions' star numbers, EuroDreams' dream number, and VikingLotto's super viking number are modeled independently from the main numbers (see `Helpers.run_model_with_special_column`); Pick3 is positional, so predictions are kept in drawn order instead of sorted.

### Deep learning & boosting

- **LSTM/TCN** (`src/LSTM.py`, `src/TCN.py`) train a sequence model on the encoded draw history and predict the next draw directly.
- **XGBoost** (`src/XGBoost.py`) is an optional boosting pass over the statistical/DL predictions, enabled per game via the `useBoost` flag.

### How predictions are combined

`Predictor.py` runs every statistical model that is enabled in `bestParams_<game>.json` (in practice, hyperopt tunes each model's parameters individually but does not disable any of them — all enabled models run every time) and stores each model's raw output. The predictions are then combined into a single ranked frequency table by `Helpers.count_number_frequencies_from_new_prediction`, which does an **unweighted vote count**: every number suggested by any model/subset gets +1, normalized at the end. There is currently no weighting by a model's individual backtest performance.

### Hyperopt & backtesting

`HyperoptStatistics.py` uses Optuna to tune each statistical model's parameters per game, driven by `src/Backtester.py`, which evaluates each model using rolling historical validation: for every historical draw, the model is trained only on previous draws and compared against the next real result. Each model's best parameters and best backtest score are written to `bestParams_<game>.json` (used by `Predictor.py` at prediction time) and printed as a `profits` summary per model/game — that score is currently for reference only and does not feed back into model selection or weighting.

The goal is not to prove deterministic prediction, but to measure whether any method (or combination) produces more 2+, 3+, or 4+ hits than simple baselines over many historical draws.

## Ideas worth researching further

These are unimplemented directions worth exploring while pushing the statistical models further:

- **Ensemble / meta-learning layer** — replace the current unweighted vote count with a learned blend (e.g. logistic regression or gradient-boosted meta-model over each model's per-number probabilities), or at minimum weight votes by each model's own backtest score already computed by `HyperoptStatistics.py`/`Backtester.py`.
- **Hidden Markov Model with regime states** — model "hot/cold" number streaks as latent states (Baum-Welch/EM) instead of the current exponential recency-weight heuristics.
- **Dirichlet-multinomial priors** — replace the ad hoc `smoothingFactor`/`alpha` knobs in the Markov/Bayesian models with proper conjugate Bayesian updating, giving principled uncertainty estimates.
- **Negative-binomial / Poisson-Gamma mixture** — `PoissonMonteCarlo` currently assumes plain Poisson counts; lottery draw counts are typically overdispersed, so a Gamma-Poisson mixture may fit better than tuning `weightFactor` alone.
- **Copula-based co-occurrence modeling** — model the joint dependency structure between numbers directly (Gaussian/empirical copula) instead of the current pairwise decay-factor approximation.


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
