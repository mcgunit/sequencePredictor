
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

Every model above (except `HybridStatisticalModel`) also exposes `score_numbers()`, returning a `{number: score}` dict instead of a collapsed ticket — reused by `MetaLearner Model` (below) and by `Backtester.collect_scores` to build its training data.

Every model shares the same interface (`setDataPath`, `run(...)`/`run_model_with_special_column`) so the Backtester and Predictor can drive them interchangeably. Euromillions' star numbers, EuroDreams' dream number, and VikingLotto's super viking number are modeled independently from the main numbers (see `Helpers.run_model_with_special_column`); Pick3 is positional, so predictions are kept in drawn order instead of sorted.

### Deep learning & boosting

- **LSTM/TCN** (`src/LSTM.py`, `src/TCN.py`) train a sequence model on the encoded draw history and predict the next draw directly.
- **XGBoost** (`src/XGBoost.py`) is an optional boosting pass over the statistical/DL predictions, enabled per game via the `useBoost` flag.

### How predictions are combined

`Predictor.py` runs every statistical model that is enabled in `bestParams_<game>.json` (in practice, hyperopt tunes each model's parameters individually but does not disable any of them — all enabled models run every time) and stores each model's raw output as its own row, so each model's real-life performance can be tracked independently over time. Three additional, purely additive rows are then appended alongside them:

- **`WeightedEnsemble Model`** (Phase 0) — `Helpers.count_number_frequencies_from_new_prediction` counts every number suggested by any model/subset, weighted by that model's own Hyperopt/Backtester score (`bestParams_<game>.json["modelScores"]`, min-max scaled to a `[1, 2]` range so a poorly-scoring model is outweighted, never zeroed out; unscored models default to a neutral weight of 1). `addWeightedEnsemblePrediction` (`Predictor.py`) then turns that weighted vote into an actual ticket. For games with special columns (Euromillions star numbers, EuroDreams dream number, VikingLotto super viking), the main numbers and special column(s) are voted on and picked **separately** via `Helpers.count_number_frequencies_by_position`, then concatenated — the same positional split every individual model already keeps via `Helpers.run_model_with_special_column` — so the special slot(s) can't get crowded out by more numerous main-range numbers. For Keno, it also generates the same `use_5`..`use_10` sub-selections every individual model produces (see the shared subset generator below). The same weighted frequency also drives the `numberFrequency` chart shown in the web UI (home page and per-day history detail page), and is skipped entirely for Pick3 (positional, no notion of a frequency-ranked ticket).
- **`MetaLearner Model`** (Phase 1) — a real stacking meta-learner: instead of a hand-weighted vote, a small `LogisticRegression` (see `TrainMetaLearner.py` below) is trained to predict `P(number is drawn)` from each base model's own per-number score (`score_numbers()`, added to `Markov`, `MarkovMonteCarlo`, `MarkovBayesian`, `MarkovBayesianEnhanced`, `PoissonMonteCarlo`, `PoissonMarkov`, `LaplaceMonteCarlo` — `HybridStatisticalModel` is deliberately excluded since it's itself a vote-based ensemble of several of these, which would be circular). `Predictor.py` loads `data/models/<game>/meta_learner.joblib` if present, re-scores each base model, and ranks numbers by the trained model's predicted probability. For games with special columns, a **second, separately-trained model** (also in that same artifact) ranks the special column(s) independently — mirroring `WeightedEnsemble Model`'s split, and fixing an earlier bug where the meta-learner's `collect_scores` data accidentally only ever saw the special column, never the main numbers. Also generates Keno subsets the same way `WeightedEnsemble Model` does. If the artifact doesn't exist yet (no `TrainMetaLearner.py` run for that game), this row is skipped gracefully — no crash, no effect on anything else. Also skipped for Pick3.
- **`MetaLearnerV2 Model`** — "lens diversity" (see the old Ideas entry this replaced): a second, independently-trained model class per game, `GradientBoostingClassifier` (`data/models/<game>/meta_learner_v2.joblib`) instead of `MetaLearner Model`'s `LogisticRegression`. Added as its own tracked row rather than replacing `MetaLearner Model` — a tree-based model can pick up nonlinear interactions between base models' scores a linear one can't, and since its errors won't necessarily correlate with the logistic regression's, real-life tracking can show which (if either) is actually worth keeping. Reuses the exact same base-model scores `MetaLearner Model` computes (`Predictor.py` caches them by feature-name set so they're not scored twice), and follows the same main/special-column split and Keno-subset generation. `GradientBoostingClassifier` has no built-in `class_weight`, so class balance is applied via `sample_weight` instead, to match `LogisticRegression`'s `class_weight="balanced"`.
All three rows share one **subset generator** — `Helpers.generate_subset_from_scores(number_scores, ticket_numbers, subset_size, mode, temperature)` picks a Keno sub-selection out of an already-scored ticket. `mode="softmax"` probability-weights by score/temperature for some variation between runs; `mode="top"` is deterministic top-N. Configurable per game via `bestParams_<game>.json`'s `weightedEnsembleSubsetMode`/`Temperature`, `metaLearnerSubsetMode`/`Temperature`, and `metaLearnerV2SubsetMode`/`Temperature` — tuned automatically by `HyperoptStatistics.py`'s `KenoSubsetTuning` strategy (see below), not just manually set.

### Hyperopt & backtesting

`HyperoptStatistics.py` uses Optuna to tune each statistical model's parameters per game, driven by `src/Backtester.py`, which evaluates each model using rolling historical validation: for every historical draw, the model is trained only on previous draws and compared against the next real result. Each model's best parameters and best backtest score are written to `bestParams_<game>.json` (used by `Predictor.py` at prediction time), including a `modelScores` entry per model (its best backtest score, keyed by the same display name `Predictor.py` uses) that now feeds `WeightedEnsemble Model`'s weighting above. `Backtester.backtest()` also has an opt-in `collect_scores` param (off by default, so normal hyperopt runs are unaffected) that captures each model's `score_numbers()` output per backtested day — used only by `TrainMetaLearner.py`.

The goal is not to prove deterministic prediction, but to measure whether any method (or combination) produces more 2+, 3+, or 4+ hits than simple baselines over many historical draws.

### Training the meta-learner (`TrainMetaLearner.py`)

Trains both `MetaLearner Model` and `MetaLearnerV2 Model` per game:

```
python3 TrainMetaLearner.py --games lotto,keno --days 300
```

For each game, it instantiates the 7 base models using that game's already-tuned `bestParams_<game>.json`, backtests them once with `collect_scores=True` over the last `--days` draws, and builds a (day, number) → [each model's score, actual-drawn label] training table across the game's full number range — reused for both model variants, so the expensive backtest doesn't run twice. It trains `LogisticRegression(class_weight="balanced")` for `MetaLearner Model` and `GradientBoostingClassifier` (class balance via `sample_weight`, since it has no `class_weight`) for `MetaLearnerV2 Model` — each evaluated on a walk-forward holdout split (last 20% of days) for an honest accuracy/AUC sanity check, then refit on the full window before saving to `data/models/<game>/meta_learner.joblib` / `meta_learner_v2.joblib` respectively (same persistence convention as `XGBoost.py`).

For games with special columns (Euromillions/EuroDreams/VikingLotto), a **second model per variant** is trained the same way, purely on the special column's own `collect_scores` data (`_special_scores`) and its own number range — determined empirically from the data (`determine_special_range`), since no range is hardcoded anywhere (e.g. Euromillions stars turned out to be 1-12, EuroDreams dream number 1-5, VikingLotto super viking 1-8). Both models of a variant are persisted together in one artifact. `Backtester.backtest(collect_scores=True)` itself mirrors `Helpers.run_model_with_special_column`'s two-call convention (a main-only call, then a special-only call) so the two ranges are never mixed.

Skips Pick3 (positional, a per-number score ranking has no notion of digit order). `runHyperopt.sh` runs this automatically right after `HyperoptStatistics.py`, so the meta-learner is retrained on every fresh hyperopt pass.

## Ideas worth researching further

Unimplemented directions worth exploring while pushing the statistical models further:

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
