
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

Every model above (except `HybridStatisticalModel`) — and, since the alignment described below, `XGBoost` too — also exposes `score_numbers()`, returning a `{number: score}` dict instead of a collapsed ticket — reused by `MetaLearner Model` (below) and by `Backtester.collect_scores` to build its training data.

Every model shares the same interface (`setDataPath`, `run(...)`/`run_model_with_special_column`) so the Backtester and Predictor can drive them interchangeably. Euromillions' star numbers, EuroDreams' dream number, and VikingLotto's super viking number are modeled independently from the main numbers (see `Helpers.run_model_with_special_column`); Pick3 is positional, so predictions are kept in drawn order instead of sorted.

### Deep learning & boosting

Both `LSTM.py` and `TCN.py` frame prediction the same way: a sliding window of `windowSize` past draws (each draw's numbers embedded/encoded) feeds the network, whose output is one independent softmax per digit/number slot (`Reshape((digitsPerDraw, num_classes))` + softmax on the last axis) — effectively "which number is most likely at this position," not a joint draw-level distribution.

- **LSTM** (`src/LSTM.py`) — `Embedding` → single `Bidirectional(LSTM)` layer → `Dropout` → `Dense` → reshaped per-position softmax. Trains with Adam, categorical cross-entropy, `EarlyStopping`/`ReduceLROnPlateau`/`ModelCheckpoint`, and custom `digit_accuracy`/`any_digit_hit`/`full_draw_accuracy` metrics. This is the model type every game in `Predictor.py`/`HyperoptDeepLearning.py` currently uses. Two rough edges worth knowing about: a `SelfAttentionBlock` class is defined but never actually used in `create_model`, and the `num_lstm_layers`/`num_bidirectional_layers` setters have no effect on the built architecture (always one layer) despite being exposed as tunable — dead configuration surface from an earlier iteration.
- **TCN** (`src/TCN.py`, via the `keras-tcn` package) — stacked dilated-convolution `TCN` layers → **two actually-wired `SelfAttentionBlock`s** (custom `MultiHeadAttention` + FFN + LayerNorm, hand-rolled rather than a library dependency) → `GlobalAveragePooling1D` → dense softmax. More complete than `LSTM.py` today: it also blends its raw prediction with a live Markov chain over the same history (`lstmMarkovAlpha`-equivalent), and adds a `TopKCategoricalAccuracy(k=3)` metric. It's fully wired (same `run()` interface, same setters) but **not currently exercised** — every dataset in `HyperoptDeepLearning.py`'s sweep is hardcoded to `"lstm_model"`, so TCN's architecture is live code that never actually gets tuned or run in practice.
- **`UnifiedLstmTcn` / `UnifiedLstmGruTcn`** (`src/UnifiedLstmTcn.py`, `src/UnifiedLstmGruTcn.py`) — real architectural fusion, replacing the old `src/Unified.py` prototype (which trained LSTM/GRU/TCN separately and only averaged their output probabilities afterward — an ensemble, not a fusion, and never wired into anything). Both share one `Embedding` step (keeping `window_size` as the time dimension, unlike `LSTM.py`'s flattening, so branches stay shape-compatible), then run a `Bidirectional(LSTM)` branch and a stacked-`TCN` branch (plus a third `GRU` branch for the 3-branch variant) **concatenated together** before the shared attention/pooling/output head — the branches' learned representations mix before the network makes its prediction, not after. Live in `Predictor.py` as two additional rows (`UnifiedLstmTcn Model`, `UnifiedLstmGruTcn Model`) alongside `LSTM Base Model`, tuned by their own `HyperoptDeepLearning.py` studies (see `MODEL_REGISTRY`/`suggest_fused_params`, with `unifiedLstmTcn_*`/`unifiedLstmGruTcn_*`-prefixed `bestParams_<game>.json` keys so they don't clobber `LSTM Base Model`'s own tuned values). No Markov blending in either (unlike `TCN.py`) — out of scope for now.
### Gradient boosting (`src/BoostingBase.py`)

Three libraries × two formulations, **six independently tracked rows**, all sharing one implementation in `src/BoostingBase.py` so a difference between rows is attributable to the library or the formulation and not to incidental plumbing. Each subclass supplies only `_make_classifier()`; the window features, label encoding, ticket construction, subset generation, fit cache and persistence are common code.

| Row | Library | Formulation |
|---|---|---|
| `XGBoost Model` | XGBoost | per-position multiclass |
| `XGBoostMultiLabel Model` | XGBoost | multi-label |
| `LightGBM Model` | LightGBM | per-position multiclass |
| `LightGBMMultiLabel Model` | LightGBM | multi-label |
| `CatBoost Model` | CatBoost | per-position multiclass |
| `CatBoostMultiLabel Model` | CatBoost | multi-label |

**The two formulations** are the substantive comparison:

- **Per-position multiclass** (`PerPositionBoostingPredictor`) — one classifier per draw slot, over a flattened window of the `<prefix>PreviousDraws` preceding draws as raw values. This is the original formulation. It genuinely fits Pick3, where slot identity is real and digits repeat; for a non-positional game it imposes slot structure the game doesn't have, which is why it needs collision-refilling to produce a full ticket at all (several positions frequently pick the same number).
- **Multi-label** (`MultiLabelBoostingPredictor`) — one binary "is this number in the next draw" classifier per number in the range, over a **multi-hot** encoding of the window. Raw sorted values encode "the 3rd smallest number was 17", which is order-statistic structure; multi-hot encodes "17 was drawn 2 draws ago", which is what a membership question actually needs. It matches the structure of a non-positional game directly, its `score_numbers()` is a calibrated `P(drawn)` rather than an average over positional softmaxes, and it is far cheaper — Keno goes from 20 multiclass fits over 80 classes to 80 plain binary fits. **Skipped for Pick3**, where set membership can represent neither digit order nor repeated digits (the same reason `WeightedEnsemble`/`MetaLearner` are skipped there).

Per-library notes, since a shared search space only means something if each library interprets it comparably: LightGBM grows leaf-wise, so `num_leaves` is capped at `2**max_depth` (left at the default 31, a tuned depth of 2 would silently do nothing) and `subsample_freq=1` is set (`subsample` is otherwise ignored outright); CatBoost's symmetric trees make depth much more expensive, its `min_child_weight` maps to `min_data_in_leaf`, and `subsample` requires switching off the default Bayesian bootstrap.

Each row keeps its own `bestParams_<game>.json` key prefix (`xgBoost`, `xgBoostMl`, `lightGbm`, `lightGbmMl`, `catBoost`, `catBoostMl`) and `use<X>` flag, so the six never clobber each other's tuned values. `XGBoost Model` deliberately keeps the pre-existing `xgBoost` prefix and `useBoost` flag so its tracked history stays continuous. `BOOSTING_PARAM_SUFFIXES` / `apply_boosting_params` in `BoostingBase` are shared by `Predictor.py` (reading tuned values) and `HyperoptBoost.py` (writing them), so the two cannot drift on a key name.

- **XGBoost** (`src/XGBoost.py`) — the first of these, and where the shared interface came from. Before that work it was a loosely-wired "optional boosting pass"; aligning it to the same interface as the statistical models (`setDataPath` / `setSortedPrediction` / `clear` / `run(...)` / `score_numbers(...)`) fixed the following, all of which the shared base now gets right for every library:
  - `run()` ignored `skipLastColumns`/`specialColumnCount` entirely — it trained on every raw CSV column, so Lotto's unplayed bonus number was modeled as a real slot and the Euromillions stars / EuroDreams dream number / VikingLotto super viking were mixed into the main-number range instead of being modeled independently.
  - Training features were raw numbers but prediction features were `number - 1`, so every live prediction was made on inputs shifted one step away from anything the model had been trained on. Labels are now encoded through the game's actual observed label set, which also retires the `offsetByOne` flag (the setter is kept for compatibility, but the encoding handles any range, Pick3's `0-9` included).
  - The ticket was the per-position argmax, which for non-positional games routinely picked the same number in several positions — a "20-number" Keno ticket could really contain far fewer. Collisions are now refilled from the next-best candidates by average confidence; Pick3 keeps drawn order and duplicates, as it must.
  - Subsets came back as a bare list (every other model returns a `{size: subset}` dict) and were built by a hand-rolled nesting loop instead of the shared `Helpers.generate_subset_from_scores`. `xgBoostForceNested` now maps to that generator's deterministic `mode="top"`, which is inherently nested.
  - Missing classes were padded at full sample weight (polluting the fit); they're now padded at a near-zero weight, purely to satisfy `multi:softprob`'s "every class must appear" requirement.
  - `load()` hardcoded 20 positions (Keno's draw size), and `save()` wrote main and special-column fits to the same paths. Persistence is now opt-in (`setSaveModels`, off during backtests where many days run in parallel) and namespaced per variant. Thread count is likewise configurable (`setNumThreads`, defaulting to 1) so a Pool of backtest workers doesn't each spawn a full XGBoost thread pool.

`Predictor.py` and `HyperoptDeepLearning.py` both used to select the active model via `modelToUse = tcn if "lstm_model" not in model_type else lstm`, then called a hardcoded, LSTM-only setter block on whatever got selected — `TCNModel` has none of those setters, so this would have crashed immediately had `model_type` ever been anything but `"lstm_model"` (it never was, until now). `HyperoptDeepLearning.py` now dispatches through `MODEL_REGISTRY`, mapping each `model_type` to its own instance and its own `configure(model, modelParams)` function, so adding the two fused models didn't require perpetuating that landmine. `tcn_model` itself is still not exercised by either script's dataset sweep — a separate, pre-existing gap, not addressed here.

For reference, [sminerport/SequencePredictionANN](https://github.com/sminerport/SequencePredictionANN) — suggested as a comparison point — uses a much simpler single-hidden-layer sigmoid feedforward network trained with MSE, no recurrence/convolution/attention at all. It's a smaller step *below* what `LSTM.py`/`TCN.py` already do here, not an upgrade; its own README lists LSTM/GRU as a suggested future improvement. Not something to port in, but a useful baseline data point.

### How predictions are combined

`Predictor.py` runs every method that is enabled in `bestParams_<game>.json` — statistical, deep learning, and boosting alike (in practice, hyperopt tunes each model's parameters individually but does not disable any of them — all enabled models run every time) — and stores each one's raw output as its own row, so each method's real-life performance can be tracked independently over time and a history builds up per method. That's the point of the whole setup: `Markov Model`, `MarkovMonteCarlo Model`, `MarkovBayesian Model`, `MarkovBayesianEnhanched Model`, `PoissonMonteCarlo Model`, `PoissonMarkov Model`, `LaplaceMonteCarlo Model`, `HybridStatisticalModel`, `LSTM Base Model`, `TCN Base Model`, `UnifiedLstmTcn Model`, `UnifiedLstmGruTcn Model`, and the six boosting rows (`XGBoost`/`LightGBM`/`CatBoost` × per-position/multi-label) each get their own tracked prediction row.

All of them now also agree on **which Keno subset sizes to play**: every model asks `getKenoSubsetSizes` for the hyperopt-tuned `use_5`..`use_10` flags. The DL rows previously hardcoded `range(5, 11)`, so they placed a bet at all six sizes while every statistical model played only the tuned ones (currently just `use_10`) — making those rows' tracked results non-comparable with the rest. `HyperoptDeepLearning.py` reads the same tuned choice via `tuned_keno_subset_sizes`, falling back to all six only when a game has no tuned flags at all, so its profit signal never silently becomes zero. Three additional, purely additive rows are then appended alongside them:

- **`WeightedEnsemble Model`** (Phase 0) — `Helpers.count_number_frequencies_from_new_prediction` counts every number suggested by any model/subset, weighted by that model's own Hyperopt/Backtester score (`bestParams_<game>.json["modelScores"]`, min-max scaled to a `[1, 2]` range so a poorly-scoring model is outweighted, never zeroed out; unscored models default to a neutral weight of 1). `addWeightedEnsemblePrediction` (`Predictor.py`) then turns that weighted vote into an actual ticket. For games with special columns (Euromillions star numbers, EuroDreams dream number, VikingLotto super viking), the main numbers and special column(s) are voted on and picked **separately** via `Helpers.count_number_frequencies_by_position`, then concatenated — the same positional split every individual model already keeps via `Helpers.run_model_with_special_column` — so the special slot(s) can't get crowded out by more numerous main-range numbers. For Keno, it also generates the same `use_5`..`use_10` sub-selections every individual model produces (see the shared subset generator below). The same weighted frequency also drives the `numberFrequency` chart shown in the web UI (home page and per-day history detail page), and is skipped entirely for Pick3 (positional, no notion of a frequency-ranked ticket).
- **`MetaLearner Model`** (Phase 1) — a real stacking meta-learner: instead of a hand-weighted vote, a small `LogisticRegression` (see `TrainMetaLearner.py` below) is trained to predict `P(number is drawn)` from each base model's own per-number score (`score_numbers()`, on `Markov`, `MarkovMonteCarlo`, `MarkovBayesian`, `MarkovBayesianEnhanced`, `PoissonMonteCarlo`, `PoissonMarkov`, `LaplaceMonteCarlo` and `XGBoost` — `HybridStatisticalModel` is deliberately excluded since it's itself a vote-based ensemble of several of these, which would be circular). `Predictor.py` loads `data/models/<game>/meta_learner.joblib` if present, re-scores each base model, and ranks numbers by the trained model's predicted probability. For games with special columns, a **second, separately-trained model** (also in that same artifact) ranks the special column(s) independently — mirroring `WeightedEnsemble Model`'s split, and fixing an earlier bug where the meta-learner's `collect_scores` data accidentally only ever saw the special column, never the main numbers. Also generates Keno subsets the same way `WeightedEnsemble Model` does. If the artifact doesn't exist yet (no `TrainMetaLearner.py` run for that game), this row is skipped gracefully — no crash, no effect on anything else. Also skipped for Pick3.
- **`MetaLearnerV2 Model`** — "lens diversity" (see the old Ideas entry this replaced): a second, independently-trained model class per game, `GradientBoostingClassifier` (`data/models/<game>/meta_learner_v2.joblib`) instead of `MetaLearner Model`'s `LogisticRegression`. Added as its own tracked row rather than replacing `MetaLearner Model` — a tree-based model can pick up nonlinear interactions between base models' scores a linear one can't, and since its errors won't necessarily correlate with the logistic regression's, real-life tracking can show which (if either) is actually worth keeping. Reuses the exact same base-model scores `MetaLearner Model` computes (`Predictor.py` caches them by feature-name set so they're not scored twice), and follows the same main/special-column split and Keno-subset generation. `GradientBoostingClassifier` has no built-in `class_weight`, so class balance is applied via `sample_weight` instead, to match `LogisticRegression`'s `class_weight="balanced"`.
All three rows share one **subset generator** — `Helpers.generate_subset_from_scores(number_scores, ticket_numbers, subset_size, mode, temperature)` picks a Keno sub-selection out of an already-scored ticket. `mode="softmax"` probability-weights by score/temperature for some variation between runs; `mode="top"` is deterministic top-N. Configurable per game via `bestParams_<game>.json`'s `weightedEnsembleSubsetMode`/`Temperature`, `metaLearnerSubsetMode`/`Temperature`, and `metaLearnerV2SubsetMode`/`Temperature` — tuned automatically by `HyperoptStatistics.py`'s `KenoSubsetTuning` strategy (see below), not just manually set. That strategy reconstructs each day's ensemble tickets from a one-time backtest of the base models, so it now includes `XGBoost Model` too — necessary for the reconstruction to match what `Predictor.py` actually produces, but it does make that one cached precompute noticeably slower.

### History rebuild & model-weight reuse (operational behavior)

Two behaviors worth knowing when running `Predictor.py` by hand:

- **Gap recovery is automatic and non-destructive.** When database files are missing, `Predictor.py` anchors on the *newest* existing prediction json (scanning the game's actual draw dates, so twice-a-week games are handled correctly) and builds every missing draw after it — however large the outage gap — plus any interior holes within the last `-d/--days` draws (e.g. corrupted files you deleted). Existing files are never overwritten by this path; `update_matching_numbers` re-links the prediction-vs-result chain across old and new files afterwards. Previously it anchored on the *oldest* existing file inside a fixed `-d`-draw window and regenerated (overwrote) every valid day after it, and an outage longer than the window was silently truncated.
- **`-r/--rebuild_history` force-regenerates (overwrites) the last `-d` draws**, existing files included — use it after history corruption. This flag previously existed but did nothing.
- **NaN training runs are contained.** Exploding gradients on some hyperparameter combos (most often Keno's 20×80 output space) can drive the loss to `nan`. Training now stops at the first NaN batch (`TerminateOnNaN`) instead of burning the whole early-stop patience on dead epochs; if any earlier epoch was healthy, its best-`val_loss` checkpoint is restored (NaN epochs never rank as "best", so the checkpoint is always clean); if the very first batch was already NaN, the model raises and that day's row is skipped — a prediction from NaN weights is `argmax` over garbage, and recording it would pollute the tracked history and the `WeightedEnsemble` vote. Corrupted weights were already never persisted.
- **Saved DL model weights carry an architecture fingerprint** (`model_<game>.fingerprint.json` next to `.weights.h5`). Warm-starting from saved weights saves training time, but it silently ignored newly hyperopted parameters (or crashed the model's row on a shape mismatch) until someone manually deleted the model files. Now a fingerprint mismatch — any change to units/layers/window/heads/class counts — triggers one fresh retrain and rewrites the fingerprint; non-shape parameters (dropout, learning rate, l2, ...) still warm-start. Weights saved before fingerprinting existed are treated as mismatched once, so freshly hyperopted parameters take effect on the next run with no manual deletion.

### Hyperopt & backtesting

`HyperoptStatistics.py` uses Optuna to tune each statistical model's parameters per game, driven by `src/Backtester.py`, which evaluates each model using rolling historical validation: for every historical draw, the model is trained only on previous draws and compared against the next real result. Each model's best parameters and best backtest score are written to `bestParams_<game>.json` (used by `Predictor.py` at prediction time), including a `modelScores` entry per model (its best backtest score, keyed by the same display name `Predictor.py` uses) that now feeds `WeightedEnsemble Model`'s weighting above. `Backtester.backtest()` also has an opt-in `collect_scores` param (off by default, so normal hyperopt runs are unaffected) that captures each model's `score_numbers()` output per backtested day — used only by `TrainMetaLearner.py`.

`HyperoptDeepLearning.py` tunes the deep learning models per game and per model type (`lstm_model`, `tcn_model`, `unified_lstm_tcn_model`, `unified_lstm_gru_tcn_model`), each in its own Optuna study writing prefixed keys, scored primarily on held-out `val_loss` with real-draw profit as a small tie-breaker.

`HyperoptBoost.py` does the same job for the boosting model, and is now a direct mirror of `HyperoptStatistics.py` rather than the separate legacy pipeline it used to be. Concretely: it evaluates through `src/Backtester.py` (one Optuna study per game+strategy, `load_if_exists=True`, walk-forward over the last `--days` draws) instead of the old rebuild-a-JSON-cache-and-total-the-profit loop; scores trials with the shared `score_from_summary` (profit per bet where a payout table exists, avg hits otherwise) rather than raw total profit; searches per-model-prefixed Keno subset sizes via the same `suggest_keno_subset`; and merges its results into `bestParams_<game>.json` including a `modelScores["XGBoost Model"]` entry, so the boosting model's vote is weighted in `WeightedEnsemble Model` like every other model's. Its `STRATEGIES` / `STRATEGY_DISPLAY_NAMES` tables have the same shape as the statistical ones, so adding a second boosting method (LightGBM, CatBoost) is a one-entry change. The old version could not run at all: it still imported `src/LSTM_ARIMA_Model.py`, `src/RefinemePrediction.py` and `src/TopPrediction.py`, none of which exist in the repo any more.

```
python3 HyperoptBoost.py --games keno,lotto --days 31 --trials 15
python3 HyperoptBoost.py --strategies LightGBMMultiLabel,CatBoostMultiLabel --games keno
```

It tunes all six boosting rows, one Optuna study per game+strategy, from one shared search space (`suggest_boosting_params`) — shared deliberately, since comparing three libraries across two formulations only means something if each was given the same space rather than one being handed a luckier range. The multi-label strategies declare `games` excluding Pick3 and are skipped there.

> **⚠️ Cost warning — read before running this on Keno.** Boosting hyperopt on Keno is *hours per trial*: a single observed trial (`xgBoostEstimators: 300`, `xgBoostPreviousDraws: 36`) took **3h44m**, because each trial backtests `--days` days and each day trains 20 multiclass models over 80 classes. With six strategies that is not a run you can leave to a nightly cron. Practical options, in order of preference: tune Keno with the **multi-label strategies only** (dramatically cheaper — 80 binary fits instead of 20×80-class fits), pass a much smaller `--days` for Keno than for the other games, or cap `Estimators`/`PreviousDraws` in `suggest_boosting_params`. The six-number games are not affected to anything like the same degree.

Beyond the base parameters (`xgBoostEstimators`, `xgBoostLearningRate`, `xgBoostMaxdepth`, `xgBoostPreviousDraws`, `xgBoostTopK`, `xgBoostForceNested`) it also searches the regularisation knobs that were previously left at library defaults — `xgBoostSubsample`, `xgBoostColsampleByTree`, `xgBoostMinChildWeight`, `xgBoostRegLambda` — plus `xgBoostSubsetMode`/`xgBoostSubsetTemperature` for the shared Keno subset generator. A boosted tree ensemble fitted on a few hundred draws overfits trivially, so that was the most consequential untouched part of the search space. `runHyperopt.sh` runs it right after `HyperoptStatistics.py` (they share `process.lock`, so they cannot run concurrently) and before `TrainMetaLearner.py`.

The goal is not to prove deterministic prediction, but to measure whether any method (or combination) produces more 2+, 3+, or 4+ hits than simple baselines over many historical draws.

### Training the meta-learner (`TrainMetaLearner.py`)

Trains both `MetaLearner Model` and `MetaLearnerV2 Model` per game:

```
python3 TrainMetaLearner.py --games lotto,keno --days 300
```

For each game, it instantiates the 8 base models using that game's already-tuned `bestParams_<game>.json`, backtests them once with `collect_scores=True` over the last `--days` draws, and builds a (day, number) → [each model's score, actual-drawn label] training table across the game's full number range — reused for both model variants, so the expensive backtest doesn't run twice. It trains `LogisticRegression(class_weight="balanced")` for `MetaLearner Model` and `GradientBoostingClassifier` (class balance via `sample_weight`, since it has no `class_weight`) for `MetaLearnerV2 Model` — each evaluated on a walk-forward holdout split (last 20% of days) for an honest accuracy/AUC sanity check, then refit on the full window before saving to `data/models/<game>/meta_learner.joblib` / `meta_learner_v2.joblib` respectively (same persistence convention as `XGBoost.py`).

For games with special columns (Euromillions/EuroDreams/VikingLotto), a **second model per variant** is trained the same way, purely on the special column's own `collect_scores` data (`_special_scores`) and its own number range — determined empirically from the data (`determine_special_range`), since no range is hardcoded anywhere (e.g. Euromillions stars turned out to be 1-12, EuroDreams dream number 1-5, VikingLotto super viking 1-8). Both models of a variant are persisted together in one artifact. `Backtester.backtest(collect_scores=True)` itself mirrors `Helpers.run_model_with_special_column`'s two-call convention (a main-only call, then a special-only call) so the two ranges are never mixed.

Skips Pick3 (positional, a per-number score ranking has no notion of digit order). `runHyperopt.sh` runs this automatically after `HyperoptStatistics.py` and `HyperoptBoost.py`, so the meta-learner is retrained on every fresh hyperopt pass.

**`XGBoost Model` is one of those 8 base features** (`ModelFactory.BASE_MODEL_NAMES`), on the grounds that a boosted-tree score is a genuinely different signal from the Markov/Poisson family — which is the entire point of stacking. Two consequences worth knowing:

- **Old artifacts keep working, but don't gain the feature.** `XGBoost Model` is *appended* to `BASE_MODEL_NAMES`, and each artifact stores its own `feature_names` that `Predictor.py` builds its vectors from (skipping any name it can't score). A `meta_learner.joblib` trained before this change therefore still loads and predicts unchanged — it simply never asks for the boosting feature, and doesn't pay its cost either. Appending rather than inserting also keeps every existing column in the same position, so an old and a new artifact stay directly comparable. **Re-run `TrainMetaLearner.py` to actually pick the new feature up.**
- **It costs meaningfully more.** Every other base model's "fit" is a frequency or transition count; every XGBoost score is a real training run, so a `--days 300` training backtest trains it 300 times (600 for a special-column game, which scores main and special ranges separately). To keep that in hand, `src/XGBoost.py` caches its fit keyed on the data slice *and* every training hyperparameter, so `run()` and `score_numbers()` on the same day train once rather than twice, and it runs single-threaded under the `Backtester`, which already parallelises across days. `Predictor.py` benefits from the same cache: the fit made for the meta-learner's main-number pass is reused by `boostingMethod`'s own row. Consider a smaller `--days` for Keno (20 positions × 80 classes per fit) than for the 6-number games.

## Ideas worth researching further

Unimplemented directions worth exploring further:

**Statistical models:**
- **Hidden Markov Model with regime states** — model "hot/cold" number streaks as latent states (Baum-Welch/EM) instead of the current exponential recency-weight heuristics.
- **Dirichlet-multinomial priors** — replace the ad hoc `smoothingFactor`/`alpha` knobs in the Markov/Bayesian models with proper conjugate Bayesian updating, giving principled uncertainty estimates.
- **Negative-binomial / Poisson-Gamma mixture** — `PoissonMonteCarlo` currently assumes plain Poisson counts; lottery draw counts are typically overdispersed, so a Gamma-Poisson mixture may fit better than tuning `weightFactor` alone.
- **Copula-based co-occurrence modeling** — model the joint dependency structure between numbers directly (Gaussian/empirical copula) instead of the current pairwise decay-factor approximation.

**Boosting / stacking:**
- **Feed the newer boosting rows into the meta-learners** — only `XGBoost Model` (per-position) is a meta-learner feature. The other five expose `score_numbers()` and could be added, but six highly-correlated boosting features would mostly add cost and multicollinearity rather than signal; the multi-label rows are the interesting candidates, since their score is a direct calibrated `P(number drawn)` rather than an average over positional softmaxes. Worth revisiting once the tracked rows show which formulation actually performs.
- **Per-game search-space caps for Keno** — see the cost warning under Hyperopt below.

**Deep learning:**
- **Transformer-based sequence model** — a real self-attention-over-time architecture (multiple transformer encoder blocks over the windowed draw sequence, learned/positional embeddings) rather than the single post-hoc `SelfAttentionBlock` `TCN.py` already bolts on after its convolutions. Worth prototyping as a new `src/Transformer.py` following the same `run()`/setter interface as `LSTM.py`/`TCN.py` so it drops into `Predictor.py`/`HyperoptDeepLearning.py` the same way.
- **Actually exercise `TCN.py`** — it's fully wired and arguably more complete than `LSTM.py` (live Markov blend, top-3 accuracy metric) but every game in `HyperoptDeepLearning.py`'s sweep is hardcoded to `"lstm_model"`, so it never gets tuned or run today. Lowest-effort item on this list: just add TCN entries to that sweep and compare.
- **Clean up dead LSTM configuration** — `LSTM.py`'s `num_lstm_layers`/`num_bidirectional_layers` setters don't affect the built architecture (always one layer) despite being tunable in `HyperoptDeepLearning.py`; either wire them up for real multi-layer stacking or drop them so the tunable surface matches what's actually built.



## Quantum-assisted research

Quantum computing is being investigated as an additional research layer alongside the existing statistical, deep-learning, boosting, and stacking models. The purpose is not to assume that a quantum model can predict an inherently random draw. The purpose is to test whether quantum feature maps or hybrid quantum-classical models can detect reproducible structure that the existing classical models do not detect.

The main research hypothesis is:

> A correctly operated lottery-style game should not contain stable, exploitable temporal information. If a model appears to outperform suitable random and classical baselines, the result must remain reproducible under walk-forward validation, synthetic-random controls, shuffled-history controls, and an untouched holdout period.

A quantum model is therefore treated as another adversarial test of the sequence-generating process, comparable to evaluating the resilience of a system against another class of analysis. Failure to find predictive structure is evidence consistent with the modeled randomness assumptions, but it is not proof of perfect randomness. Apparent predictive structure is a signal for further investigation, not immediate proof that a game is predictable or biased.

### Recommended first integration: `QuantumMetaLearner Model`

The first quantum experiment should reuse the same per-number training data already collected for `MetaLearner Model` and `MetaLearnerV2 Model`.

For every backtested day and candidate number, the existing pipeline already produces a feature vector containing the scores from the seven supported base models:

- `Markov`
- `MarkovMonteCarlo`
- `MarkovBayesian`
- `MarkovBayesianEnhanced`
- `PoissonMonteCarlo`
- `PoissonMarkov`
- `LaplaceMonteCarlo`

The label remains unchanged:

```text
1 = the candidate number occurred in the next real draw
0 = the candidate number did not occur in the next real draw
```

The three meta-models can therefore be compared using the same input matrix, labels, chronological split, and ticket-construction logic:

```text
Base-model score vectors
          |
          +--> LogisticRegression
          |       `MetaLearner Model`
          |
          +--> GradientBoostingClassifier
          |       `MetaLearnerV2 Model`
          |
          +--> Quantum feature map / quantum classifier
                  `QuantumMetaLearner Model`
```

This answers a focused research question:

> Can a quantum feature map discover useful nonlinear relationships among the existing model scores that the logistic-regression and gradient-boosting meta-learners miss?

The quantum model should be added as its own independently tracked prediction row. It must not replace either existing meta-learner until repeated backtesting and real-life tracking demonstrate a reliable improvement.

For games with special columns, the current separation must remain intact:

```text
Main-number quantum model
Special-column quantum model
```

Euromillions stars, the EuroDreams dream number, and the VikingLotto super viking number must not be mixed with the main-number range. `Pick3` should initially remain excluded because the current meta-learning representation ranks numbers but does not represent positional digit order.

A possible artifact layout is:

```text
data/models/<game>/quantum_meta_learner.joblib
```

The persisted artifact should include everything required to reproduce inference:

- Feature-name order
- Feature scaler
- Optional dimensionality-reduction transform
- Quantum model parameters
- Classification threshold, if one is used
- Main-number model
- Optional special-column model
- Training metadata and package versions

### Quantum feature encoding

The seven base-model scores are classical values. Before they can be processed by a quantum circuit, they must be normalized and encoded into gate parameters.

A practical first implementation should reduce the seven scores to four features:

```text
Seven base-model scores
          |
          v
StandardScaler fitted on training data only
          |
          v
PCA or training-only feature selection
          |
          v
Four normalized features
          |
          v
Four-qubit parameterized circuit
```

Possible encodings include angle rotations such as `RY` or `RZ`, followed by entangling gates and trainable rotations. Circuit measurements produce expectation values or class probabilities that the normal Python pipeline can convert into per-number scores.

The scaler, PCA transform, and feature selector must be fitted only on the training partition. Fitting preprocessing on the full dataset would leak information from the holdout period.

Starting with four qubits keeps simulation and hyperparameter optimization manageable. A seven-qubit version can be researched later, but it will be substantially more expensive to train and simulate.

### Initial quantum model candidates

The first comparison should include two different quantum approaches where practical:

1. **Quantum-kernel classifier**
   - Encodes each feature vector into a quantum state.
   - Estimates similarity through a quantum kernel.
   - Uses the resulting kernel with a classical support-vector classifier.

2. **Variational quantum classifier**
   - Encodes the input scores as circuit rotations.
   - Applies a parameterized ansatz with entangling gates.
   - Uses a classical optimizer to train the circuit parameters.

The quantum-kernel model is the preferred first prototype because it provides a relatively clean comparison with a classical RBF support-vector machine. The variational classifier can be added after the data flow, persistence, and evaluation logic are stable.

### Training integration

`TrainMetaLearner.py` should collect the seven base-model scores only once and reuse the resulting dataset for all meta-learners:

```python
training_data = collect_meta_training_data(...)

train_logistic_meta_learner(training_data)
train_gradient_boosting_meta_learner(training_data)
train_quantum_meta_learner(training_data)
```

The expensive backtest must not be repeated separately for each meta-model. All variants must receive exactly the same:

- Historical days
- Candidate-number rows
- Base-model feature values
- Labels
- Main/special-column separation
- Walk-forward holdout boundary

This is necessary for a fair benchmark.

Quantum training should initially be opt-in because circuit simulation and quantum-kernel computation can be much slower than the existing classical meta-learners. A per-game configuration flag can control the feature:

```json
{
  "useQuantumMetaLearner": false
}
```

### Evaluation metrics

Ticket-level hit counts remain important, but the quantum model must also be evaluated at the per-number probability and ranking levels.

#### Per-number probability metrics

- ROC AUC
- Precision-recall AUC
- Brier score
- Log loss
- Expected calibration error
- Calibration curve

Accuracy alone is not sufficient. In a game that draws only a small subset of the available number range, a model can achieve high accuracy by predicting that every number will not be drawn.

#### Ranking metrics

- Precision at ticket size
- Recall at ticket size
- Mean rank of the numbers that were actually drawn
- Normalized discounted cumulative gain

#### Ticket-level metrics

- Average number of hits
- 2+ hit rate
- 3+ hit rate
- 4+ hit rate
- Maximum hit count
- Full hit-count distribution
- Relative improvement over uniform and frequency baselines

All metrics must be calculated over chronological out-of-sample predictions.

### Randomness-discrimination experiment

A second quantum research track should test whether real draw windows can be distinguished from synthetic fair-draw windows.

The classification problem is:

```text
Class 0 = synthetic draws generated according to the game rules
Class 1 = real historical draws
```

The main question is:

> Can a classical or quantum model identify a reproducible difference between the real history and a correctly simulated fair process?

The comparison should include:

- Logistic regression
- RBF support-vector machine
- Random forest
- Gradient boosting
- Small neural network
- Quantum-kernel classifier
- Variational quantum classifier

Balanced accuracy and ROC AUC near 50% on a genuinely untouched holdout period indicate that a model cannot reliably distinguish the two sources.

Performance above chance does not immediately imply manipulation or next-draw predictability. A classifier may instead detect:

- Historical game-rule changes
- Changes in number or special-number ranges
- Sorted versus drawn-order differences
- Missing or duplicated records
- API or preprocessing artifacts
- Equipment or draw-schedule changes
- Incorrect synthetic-data generation

Synthetic data must therefore reproduce the exact rules that applied during each historical period.

### Required negative controls

The complete research pipeline tests many models, games, features, windows, and hyperparameters. Apparent improvements can occur by chance, especially when only the best result is reported. The quantum experiments must therefore include the same optimization effort on negative controls.

#### Shuffled-history control

Randomly reorder the historical draws and run the same feature engineering, optimization, and evaluation process.

If performance remains similar after shuffling, the model is probably using marginal frequencies, range characteristics, sorted-position distributions, or another non-temporal property rather than learning next-draw dependence.

#### Synthetic-fair-history control

Generate many synthetic histories with the same length and rules as the real history. Run the complete hyperparameter optimization and backtest process on each synthetic history.

The relevant null comparison is not one random model. It is:

> The best score found after applying the same complete model-selection and optimization process to data known to be random.

This estimates how often the research process itself discovers apparently strong models in random data.

#### Irrelevant-feature control

Add at least one independently generated random feature. The model should not assign stable predictive importance to that feature across repeated training runs.

#### Lockbox period

Reserve the newest historical period as a final untouched test set. Model design, hyperparameters, preprocessing, feature selection, and metric selection must be frozen before this period is evaluated.

Repeatedly inspecting the same holdout period and modifying the model afterward converts the holdout into development data.

### Search versus prediction

The Grover experiments in the separate quantum-learning repository demonstrate how a known condition can be used to amplify matching candidates. Grover search is not itself a prediction model and does not create information about the next draw.

A later Sequence Predictor experiment could define an oracle such as:

```python
def is_promising(candidate_ticket, scoring_model, threshold):
    return scoring_model.score(candidate_ticket) >= threshold
```

Grover search could then amplify candidate tickets whose model score exceeds the threshold.

The workflow would be:

```text
Historical data
      |
      v
Classical or quantum predictive model
      |
      v
Score candidate tickets
      |
      v
Oracle marks tickets above a threshold
      |
      v
Grover amplification
      |
      v
Measured candidate tickets
```

This can research quantum candidate search, but the predictive information still comes from the scoring model. If the previous draws contain no useful signal about the next independent draw, Grover search cannot create such a signal.

Grover-based ticket search is therefore a later experimental phase, after the quantum meta-learner and randomness discriminator have been benchmarked.

### Within-draw structure versus temporal predictability

Sorted lottery numbers contain predictable positional structure even when the underlying draw is fair. The first sorted number tends to be lower than the final sorted number. A position-based model can learn this order-statistic structure without learning any dependence between successive draws.

The project should therefore report these concepts separately:

```text
Within-draw distributional structure
    versus
Between-draw temporal predictive value
```

A model that generates plausible sorted tickets or achieves low positional error is not necessarily producing more next-draw hits than a fair baseline.

For non-positional games, a multi-hot representation is useful for randomness and sequence experiments:

```text
One element per possible number
1 = number was drawn
0 = number was not drawn
```

`Pick3` should retain its positional representation because digit order is part of the result.

### Proposed implementation phases

#### Phase Q0: classical controls first

Before interpreting any quantum comparison:

- Actually tune and run `TCN.py` as a classical baseline.
- Remove or correctly wire the dead LSTM layer-count configuration.
- Add synthetic-fair-history benchmarks.
- Add shuffled-history benchmarks.
- Add per-number probability and ranking metrics.
- Establish an untouched chronological lockbox period.

#### Phase Q1: quantum meta-learning

- Reuse `TrainMetaLearner.py` score matrices.
- Add `QuantumMetaLearner Model` as a separate tracked row.
- Start with four training-only reduced features and four qubits.
- Compare with logistic regression, gradient boosting, and classical-kernel SVM.
- Persist preprocessing and model metadata with the quantum artifact.
- Keep the model opt-in during development.

#### Phase Q2: randomness discrimination

- Generate synthetic histories using exact period-specific game rules.
- Build real-versus-synthetic window features.
- Compare classical and quantum classifiers.
- Repeat the experiment across multiple synthetic seeds and chronological splits.
- Investigate any stable distinguishability before treating it as predictive evidence.

#### Phase Q3: direct quantum scoring

- Produce a quantum-assisted score for every candidate number.
- Reuse the existing ticket construction, special-column handling, Keno subset generation, Backtester, and UI tracking.
- Compare calibration and ranking before comparing final ticket hits.

#### Phase Q4: candidate search

- Define a ticket-level score and threshold.
- Build a toy Grover oracle for a deliberately small candidate space.
- Compare quantum search cost with direct classical ranking.
- Include circuit depth, native two-qubit gate count, noise sensitivity, and data-loading cost.

### Suggested project structure

```text
src/
├── QuantumMetaLearner.py
├── QuantumKernelModel.py
├── QuantumRandomnessDiscriminator.py
└── quantum/
    ├── __init__.py
    ├── feature_encoding.py
    ├── circuits.py
    ├── simulator.py
    ├── evaluation.py
    └── persistence.py

experiments/
└── quantum/
    ├── compare_meta_learners.py
    ├── real_vs_synthetic.py
    ├── synthetic_null_benchmark.py
    ├── shuffled_history_control.py
    ├── noise_sensitivity.py
    └── transpilation_cost.py
```

### Reporting requirements

Every quantum result should include:

- Game and historical date range
- Game rules and any rule-change boundaries
- Training, validation, and lockbox periods
- Feature list and preprocessing steps
- Number of qubits
- Feature map and circuit ansatz
- Circuit depth and operation counts
- Simulator or backend
- Shot count
- Noise model, if used
- Optimizer and optimization budget
- Hyperparameter search budget
- Classical models given an equivalent tuning budget
- Random seeds
- Per-number metrics
- Ranking metrics
- Ticket-level hit distribution
- Synthetic and shuffled-control results
- Runtime and computational cost

A quantum model should not be described as better based on one favorable backtest, one game, or one metric. Improvements should be stable across repeated seeds, chronological periods, and appropriate null controls.

### Research interpretation

The quantum extension is intended as an adversarial randomness and model-capability benchmark.

A defensible conclusion can state:

```text
Under the tested data representation, model family, optimization budget,
and chronological evaluation period, the quantum-assisted model did or
 did not detect reproducible structure beyond the selected classical baselines.
```

It should not state that failure to detect structure proves perfect randomness, or that a small retrospective uplift proves future lottery predictability.

All quantum experiments remain subject to the repository's education and research disclaimer. Simulated quantum circuits run on classical hardware and do not demonstrate quantum computational advantage.



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
