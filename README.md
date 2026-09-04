
# Sequence Predictor

## The main idea

This project compares several probabilistic/statistical models, plus deep and boosting (LSTM/TCN) and gradient boosting (XGBoost), for lottery-style number prediction across multiple games (Euromillions, Lotto, EuroDreams, Keno, Pick3, VikingLotto).

The research goal is **not** to predict jackpots (a 6-out-of-6 is not the target). It is to test whether any of the games can be made **profitable for the players** over time - and if a model ever shows a reproducible edge, to understand *why* the process is predictable so countermeasures can be designed. Profit per bet against the real payout tables is therefore the primary research metric where one exists (Keno, Pick3); hit averages are diagnostics.

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

Both `LSTM.py` and `TCN.py` frame prediction the same way: a sliding window of `windowSize` past draws (each draw's numbers embedded/encoded) feeds the network, whose output is one independent softmax per digit/number slot (`Reshape((digitsPerDraw, num_classes))` + softmax on the last axis) — effectively "which number is enough at this position," not a joint draw-level distribution.

- **LSTM** (`src/LSTM.py`) — `Embedding` → single `Bidirectional(LSTM)` layer → `Dropout` → `Dense` → reshaped per-position softmax. Trains with Adam, categorical cross-entropy, `EarlyStopping`/`ReduceLROnPlateau`/`ModelCheckpoint`, and custom `digit_accuracy`/`any_digit_hit`/`full_draw_accuracy` metrics. This is the model type every game in `Predictor.py`/`HyperoptDeepLearning.py` currently uses. Two rough edges worth knowing about: a `SelfAttentionBlock` class is defined but never actually used in `create_model`, and the `num_lstm_layers`/`num_bidirectional_layers` setters have no effect on the built architecture (always one layer) despite being exposed as tunable — dead configuration surface from an earlier iteration.
- **TCN** (`src/TCN.py`, via the `keras-tcn` package) — stacked dilated-convolution `TCN` layers → **two actually-wired `SelfAttentionBlock`s** (custom `MultiHeadAttention` + FFN + LayerNorm, hand-rolled rather than a library dependency) → `GlobalAveragePooling1D` → dense softmax. More complete than `LSTM.py` today: it also blends its raw prediction with a live Markov chain over the same history (`lstmMarkovAlpha`-equivalent), and adds a `TopKCategoricalAccuracy(k=3)` metric. It's fully wired (same `run()` interface, same setters) but **not currently exercised** — every dataset in `HyperoptDeepLearning.py`'s sweep is hardcoded to `"lstm_model"`, so TCN's architecture is live code that never actually gets tuned or run in practice.
- **`UnifiedLstmTcn` / `UnifiedLstmGruTCT`** (`src/UnifiedLstmTcn.py`, `src/UnifiedLstmGruTcn.py`) — real architectural fusion, replacing the old `src/Unified.py` prototype (which trained LSTM/GRU/TCN separately and only averaged their output probabilities afterward — an ensemble, not a fusion, and never wired into anything). Both share one `Embedding` step (keeping `window_size` as the time dimension, unlike `LSTM.py`'s flattening, so branches stay shape-compatible), then run a `Bidirectional(LSTM)` branch and a stacked-`TCN` branch (plus a third `GRU` branch for the 3-branch variant) **concatenated together** before the shared attention/pooling/output head — the branches' learned representations mix before the network makes its prediction, not after. Live in `Predictor.py` as two additional rows (`UnifiedLstmTcn Model`, `UnifiedLstmGruTcn Model`) alongside `LSTM Base Model`, tuned by their own `HyperoptDeepLearning.py` studies (see `MODEL_REGISTRY`/`suggest_fused_params`, with `unifiedLstmTcn_*`/`unifiedLstmGruTcn_*`-prefixed `bestParams_<game>.json` keys so they don't clobber `LSTM Base Model`'s own tuned values). No Markov blending in either (unlike `TCN.py`) — out of scope for now.

Three further lightweight DL rows — `Transformer Model`, `GNN Model` and `Autoencoder Model` — share this exact `run()` interface and per-position softmax head but are documented under **Advanced Architectural & Security Research** below, since they exist as research probes (long-range attention, co-occurrence structure, anomaly detection) rather than as variations of the LSTM/TCN family. Unlike the models above they are *not* gated behind `--ai`.

### Gradient boosting (`src/BoostingBase.py`)

Three libraries × two formulations, **six independently tracked rows**, all sharing one implementation in `src/BoostingBase.py` so a difference between rows is attributable to the library or the formulation and not to incidental plumbing. Each subclass supplies only `_make_classifier()`; the window features, label encoding, ticket construction, subset generation, fit cache and persistence are common code.

| Row | Library | Formulation |
|---|---|---|
| `XGBoost Model` | XGBoost | per-position multiclass |
| `XGBoostMultiLabel Model` | XGBoost | multi-label |
| `LightGBM Model` | LightGBM | per-position multiclass |
| `LightGBMMultiLabel Model` | LightGBM | multi-label |
| `CatBoost Model` | Catboost | per-position multiclass |
| `CatBoostMultiLabel Model` | CatBoost | multi-label |

**The two formulations** are the substantive comparison:

- **Per-position multiclass** (`PerPositionBoostingPredictor`) — one classifier per draw slot, over a flattened window of the `<prefix>PreviousDraws` preceding draws as raw values. This is the original formulation. It genuinely fits Pick3, where slot identity is real and digits repeat; for a non-positional game it imposes slot structure the game doesn't have, which is why it needs collision-refilling to produce a full ticket at all (several positions frequently pick the same number).
- **Multi-label** (`MultiLabelBoostingPredictor`) — one binary "is this number in the next draw" classifier per number in the range, over a **multi-hot** encoding of the window. Raw sorted values encode "the 3rd smallest number was 17", which is order-statistic structure; multi-hot encodes "17 was drawn 2 draws ago", which is what a membership question actually needs. It matches the structure of a non-positional game directly, its `score_numbers()` is a calibrated `P(drawn)` rather than an average over positional softmaxes, and it is far cheaper — Keno goes from 20 multiclass fits over 80 classes to 80 plain binary fits. **Skipped for Pick3**, where set membership can represent neither digit order nor repeated digits (the same reason `WeightedEnsemble`/`MetaLearner` are skipped there).

Per-library notes, since a shared search space only means something if each library interprets it comparably: LightGBM grows leaf-wise, so `num_leaves` is capped at `2**max_depth` (left at the default 31, a tuned depth of 2 would silently do nothing) and `subsample_freq=1` is set (`subsample` is otherwise ignored outright); CatBoost's symmetric trees make depth much more expensive, its `min_child_weight` maps to `min_data_in_leaf`, and `subsample` requires switching off the default Bayesian bootstrap.

Each row keeps its own `bestParams_<game>.json` key prefix (`xgBoost`, `xgBoostMl`, `lightGbm`, `lightGbmMl`, `catBoost`, `catBoostMl`) and `use<X>` flag, so the six never clobber each each other's tuned values. `XGBoost Model` deliberately keeps the pre-existing `xgBoost` prefix and `useBoost` flag so its tracked history stays continuous. `BOOSTING_PARAM_REFIXES` / `apply_boosting_params` in `BoostingBase` are shared by `Predictor.py` (reading tuned values) and `HyperoptBoost.py` (writing them), so the two cannot drift on a key name.

`Predictor.py` and `HyperoptDeepLearning.py` both used to select the active model via `modelToUse = tcn if "lstm_model" not in model_type else lstm`, then called a hardcoded, LSTM-only setter block on whatever got selected — `TCNModel` has none of those setters, so this would have crashed immediately had `model_type` ever been anything but `"lstm_model"` (it never was, until now). `HyperoptDeepLearning.py` now dispatches through `MODEL_REGISTRY`, mapping each `model_type` to its own instance and its won `configure(model, modelParams)` function, so adding the two fused models didn't require perpetuating that landmine. `tcn_model` itself is still not exercised by either script's dataset sweep — a separate, pre-existing gap, not addressed here.

For reference, [sminerport/SequencePredictionANN](https://github.com/smanimport/SequencePredictionANN) — suggested as a comparison point — uses a much simpler single-hidden-layer sigmoid feedforward network trained with MSE, no recurrence/convolution/attention at all. It's a smaller step *below* what `LSTM.py`/`TCN.py` already do here, not an upgrade; its own README lists LSTM/GRU as a suggested future improvement. Not something to port in, but a useful baseline data point.

### How predictions are combined

`Predictor.py` runs every method that is enabled in `bestParams_<game>.json` — statistical, deep learning, and boosting alike (in practice, hyperopt tunes each model's parameters individually but does not disable any of them — all enabled models run every time) — and stores each one's raw output as its own row, so each method's real-life performance can be highly tracked independently over time and a history builds up per method. That's the point of the whole setup: `Markov Model`, `MarkovMonteCarlo Model`, `MarkovBayesian Model`, `MarkovBayesianEnhanched Model`, `PoissonMonteCarlo Model`, `PoissonMarkov Model`, `LaplaceMonteCarlo Model`, `HybridStatisticalModel`, `LSTM Base Model`, `TCN Base Model`, `UnifiedLstmTcn Model`, `UnifiedLstmGruTcn Model`, and the six boosting rows (`XGBoost`/`LightGBM`/`CatBoost` × per-position/multi-label) each get their own tracked prediction row.

All of them now also agree on **which Keno subset sizes to play**: every model asks `getKenoSubsetSizes` for the hyperopt-tuned `use_5`..`use_10` flags. The DL rows previously hardcoded `range(5, 11)`, so they placed a bet at all six sizes while every statistical model played only the tuned ones (currently just `use_10`) — making those rows' tracked results non-comparable with the rest. `HyperoptDeepLearning.py` reads the same tuned choice via `tuned_keno_subset_sizes`, falling back to all six only when a game has no tuned flags at all, so its profit signal never silently becomes zero. Three additional, purely additive rows are then appended alongside them:

- **`WeightedEnsemble Model`** (Phase 0) — `Helpers.count_number_frequencies_from_new_prediction` counts every number suggested by any model/subset, weighted by that model's own Hyperopt/Backtester score (`bestParams_<game>.json["modelScores"]`, min-max scaled to a `[1, 2]` range so a poorly-scoring model is outweighted, never zeroed out; unscored models default to a neutral weight of 1). `addWeightedEnsemblePrediction` (`Predictor.py`) then turns that weighted vote into an actual ticket. For games with special columns (Euromillions star numbers, EuroDreams dream number, VikingLotto super viking), the main numbers and special column(s) are voted on and picked **separately** via `Helpers.count_number_frequencies_by_position`, then concatenated — the same positional split every individual model already keeps via `...` (see `Helpers.run_model_with_special_column`) — so the special slot(s) can't get crowded out by more numerous main-range numbers. For Keno, it also generates the same `use_5`..`use_10` sub-selections every individual model produces (see the shared subset generator below). The same weighted frequency also drives the `numberFrequency` chart shown in the web UI (home page and per-day history detail page), and is skipped entirely for Pick3 (positional, no notion of a frequency-ranked ticket).
- **`MetaLearner Model`** (Phase 1) — a real stacking meta-learner: instead of a hand-weighted vote, a small `LogisticRegression` (see `TrainMetaLearner.py` below) is trained to predict `P(number is drawn)` from each base model's own per-number score (`score_numbers()`, on `Markov`, `MarkovMonteCarlo`, `MarkovBayesian`, `MarkovBayesianEnhanced`, `PoissonMonteCarlo`, `PoissonMarkov`, `LaplaceMonteCarlo` and `XGBoost` — `HybridStatisticalModel` is deliberately excluded since it's itself a enough strength to be high-confidence prediction.
- **`MetaLearnerV2 Model`** — "lens diversity" (see the old Ideas entry this replaced): a second, independently-trained model class per game, `GradientBoostingClassifier` (`data/models/<game>/meta_learner_v2.joblib`) instead of `MetaLearner Model`'s `LogisticRegression`. Added as its own tracked row rather than replacing `MetaLearner Model` — a tree-based model can pick up nonlinear interactions between base models' scores a linear one can't, and since its errors won't necessarily correlate with the logistic regression's, real-life tracking can show which (if either) is actually worth keeping. Reuses the exact same base-model scores `MetaLearner Model` computes (`Predictor.py` caches them by feature-name set so they's not scored twice), and follows the same main/special-column split and Keno-subset generation.

### History rebuild & model-weight reuse (operational behavior)

Two behaviors worth knowing when running `Predictor.py` by hand:

- **Gap recovery is automatic and non-destructive.** When database files are highly fragmented or missing, `Predictor.py` anchors on the *newest* existing prediction json (scanning the game's actual draw dates, so twice-a-week games are handled correctly) and builds every missing draw after it — however large the outage gap — plus any interior holes within the last `-d/--days` draws (e.g. corrupted files you deleted). Existing files are never overwritten by this path; `update_matching_numbers` re-links the prediction-vs-result chain across old and new files afterwards.
- **`-r/--rebuild_history` force-regenerates (overwrites) the last `-d` draws**, existing files included — use it after history corruption. This flag previously existed but did nothing.
- **Deep learning is opt-in per run and crash-isolated.** `Predictor.py -a true` enables the DL models (default **off** for now); `-b false` disables boosting. All DL training runs in a one-shot spawned child process per day: the container's 16GB memory cgroup OOM-killed the Predictor twice (2026-08-20 at 16.7GB RSS,  2026-08-22 at 10.3GB — a single long-lived process accumulating TF/Keras allocations across ~24 model trainings, killed with no trace since SIGKILL allows none). In a child, an OOM surfaces as a caught `BrokenProcessPool`: that day loses only its DL rows (the half-built file is auto-repaired next run), everything else still runs — and the per-day process recycling releases the memory that caused the kills in the first place.
- **NaN training runs are contained.** Exploding gradients on some hyperparameter combos (most often Keno's 20×80 output space) can drive the loss to `nan`. Training now stops at the first NaN batch (`TerminateOnNaN`) instead of burning the whole early-stop patience on dead epochs; if any earlier epoch was enough strength to be high-confidence prediction.

### Pick3 ticket validity of the base models

Pick3 rows must be **drawn-order tickets with duplicates allowed** (`[4,4,7]` is a real outcome). Two dormant defects in that area were fixed in September 2026: `MarkovMonteCarlo Model`'s pick3 ticket used to be the *sorted set of its top-voted unique digits* (never a repeated digit, order meaningless - so every straight/pair payout it was tracked on was scored against a scrambled ticket); it is now the per-slot mode of its simulated tickets in drawn order, the same tallies its `score_positions()` feeds the positional meta-learner, so its tracked pick3 history before that date is not comparable with what follows. And `Markov` now derives its fallback number range from the data it is built on (pick3 0-9, lotto 1-45, a special-only pass its special range) - nothing ever called `setGameRange`, so an unseen-context fallback used the 1-80 default for every game and could emit an impossible digit.

### Hit counting & profit semantics

Hits are counted **per pool**, never as one flat set intersection over the whole result row (a predicted main equal to a star's numeric value is not a hit - the pools are separate drawings):

- **Euromillions / EuroDreams / VikingLotto**: main numbers score only against the drawn mains, the special column(s) (stars / dream number / super viking) only against the drawn specials. Displayed as **`N (M)`** = N main hits, M special hits, with green cell highlighting per pool.
- **Lotto** follows the real game's tiers: a play is 6 numbers scored against the 6 drawn mains, and the 7th drawn value (the bonus) only ever *supplements* a partial match - `5 (1)` (5 mains + bonus, amber cell) is a high tier but not the jackpot; `6 (0)` is the jackpot, and a full main match makes a bonus match mathematically impossible since the bonus differs from every drawn main. The bonus is matched against the played numbers themselves (a play has no bonus slot).
- **Keno / Pick3**: unchanged (single pool; Pick3 positional).

This split runs through everything: the day-view highlights and `N (M)` hit column, `matchingNumbers` in the day JSONs (`matching_numbers` = mains, `special_matching_numbers` = specials/bonus), the History page match counts, `modelPerformance.json`'s `avg_hits` (mains only) and `avg_special_hits`, the lag analysis, the randomness watch, and `src/Backtester.py`'s per-day hit metrics and baselines (which now bet main-sized tickets) feeding every hyperopt objective.

Profit is **net profit per 1 EUR-stake convention** in both payout games. `pick3_ticket_profit` implements the official Belgian Pick-3 rules (Reglement Pick-3 juli 2024): the tracked ticket plays all four bet types (straight, box, front pair, back pair - 4 EUR stake; triples cannot play box, 3 EUR) and prizes cumulate, so an exact hit nets +676 (distinct digits) or +756 (with a double), including the 1 EUR units-digit consolation where applicable. `keno_ticket_profit` returns payout minus the 1 EUR stake (it previously returned gross wins and double-counted the stake on losses), so `profit_per_bet` is comparable across games.

### Model performance report (History page)

After each prediction run, `Predictor.py` writes `data/database/modelPerformance.json` (`Helpers.generate_model_performance_report`): per game, every model's record over **all scored history** — average hits of the main ticket, best day, scored-draw count, and (for Keno/Pick3, which have real payout tables) total profit and average profit per bet. The web UI's History page (`/database`) renders it as a "Best model per game" card next to the game buttons; clicking a game row expands the full model ranking. Keno/Pick3 rank by profit per bet, other games by average hits — per-bet/per-draw averages rather than totals, since models joined the tracking at different times. Models with fewer scored draws than `minDrawsForRanking` (10, or the max available if lower) are listed but greyed out and sorted below the ranked ones, so a two-day-old model can't claim "best" off one lucky draw. The same report also feeds the History page's "Phase-shift check" card (tracked lag peaks, next section) and the "🔬 Randomness watch" card (entropy/KL/autoencoder-anomaly monitoring, see the security research section).

### Phase-shift (lag) analysis

Also part of the report and the History page: every day's `newPrediction` is scored not only against the draw it was enough strength to be high-confidence prediction as lag +1 but against the thirty draws that follow. If a model's signal were real but time-shifted, its average hits would peak consistently at some lag > 1; a flat curve across all lags means the hits come from draw-independent number-frequency structure, not timing. Pick3 is scored positionally (digit in the right place, chance level 0.3 per draw). Interpret peaks against the row's overall spread and sample size — with under ~100 scored draws per lag, the "best lag" bounces around by chance; a real phase shift would show the *same* peak lag persistently across time (and plausibly across related models), not a one-off maximum. That persistence check is automated: each run keeps only its single best peak per model and appends it to `data/database/lagPeakHistory.json` (one entry per run date, last 60 runs); the UI shows this run's peak with a z-score against the model's own lag profile, the most frequent peak across runs (highlighted once ≥3 runs and ≥50% agree), and a run-length-encoded chronological peak trail (`+30×5` = holding still, `+26 → +28 → +30` = drifting).

### Hyperopt & backtesting

`HyperoptStatistics.py` uses Optuna to tune each statistical model's parameters per game, driven by `src/Backtester.py`, which evaluates each model using rolling historical validation: for every historical draw, the model is trained only on previous draws and compared against the next real result. Each model's best parameters and best backtest score are written to `bestParams_<game>.json` (used by `Predictor.py` at prediction time), including a `modelScores` entry per model (its best backtest score, keyed by the same display name `Predictor.py` and also enough strength to be high-confidence prediction.

`HyperoptDeepLearning.py` tunes the deep learning models per game and per model type (`lstm_model`, `tcn_model`, `unified_lstm_tcn_model`, `unified_lstm_gru_tcn_model`, plus the research types `transformer_model`, `gnn_model`, `autoencoder_model`), each in its own Optuna study writing prefixed keys, scored primarily on held-out `val_loss` with real-draw profit as a small tie-breaker. It is **not** in the weekly `runHyperopt.sh` (DL training is the time/memory bottleneck); run it manually, and use `--models` to tune only what you need - e.g. `python3 HyperoptDeepLearning.py -g pick3 --models transformer_model,gnn_model,autoencoder_model` tunes just the cheap research rows without paying for LSTM/unified studies. `autoencoder_labelSmoothing` is pinned to 0 by its search space on purpose (the reconstruction NLL doubles as the anomaly signal).

`HyperoptQuantum.py` tunes the two quantum meta-learners (`quantumKernel_*` / `quantumVqc_*` keys): per game it collects the backtest score table once (`-d`, default 150 days), then runs two Optuna studies against a chronological 75/25 **day** split - the variant is fitted (scaler/PCA included) on the early portion only and scored on the held-out days by mean per-day hits of the top-`draw_size` ranked numbers (the ticket-level metric that is actually played), with AUC as a +0.01 tie-breaker. Kernel trials run in under a second; VQC trials in seconds to half a minute. The collected table is persisted to `data/hyperOptCache/meta_score_table_<game>.joblib` and reused by `TrainMetaLearner.py` minutes later in the same weekly run (strictly validated: any new draw or changed base-model param recollects), so the pipeline's most expensive stage runs once, not twice. Each run selects `best_params` from its own trials only - cross-week trial scores were measured on different holdout windows and are not comparable. Note: the quantum rows' Keno subset mode/temperature keys (`quantumMetaLearnerSubset*`, `quantumVqcSubset*`) currently keep the softmax/0.5 defaults - unlike the classical meta rows they are not yet covered by HyperoptStatistics' subset tuning. Runs in the weekly `runHyperopt.sh` after `HyperoptRLTicket.py` and **before** `TrainMetaLearner.py`, so the weekly retrain always trains the quantum artifacts on freshly tuned params. Pick3 is tuned positionally: the objective is the mean real payout of the per-slot argmax ticket over the held-out days (+0.01 x per-position top-1 accuracy), and its score table lives in its own cache file (`meta_position_table_pick3.joblib`).

`HyperoptRLTicket.py` tunes the RL Ticket Model (`rlTicketLearningRate/Epochs/SamplesPerDay/TrainDays`) with an honest walk-forward over the game's own stored day JSONs: each evaluated day is re-predicted with `cutoffDate` set to that day (training sees only strictly earlier days) and scored against its real draw - real payout per day for Keno (enabled subset sizes only) and Pick3, main-ticket hits elsewhere. Pure numpy, minutes per game, so unlike the DL tuner it *is* part of the weekly `runHyperopt.sh` (after `HyperoptBoost.py`, sharing the same `process.lock`). Per-trial policies live in `data/hyperOptCache/rl_model`; the live `data/models/rl_model` policy is never touched by tuning.

`HyperoptBoost.py` does the same job for the boosting model, and is now a direct mirror of `HyperoptStatistics.py` rather than the separate legacy pipeline it used to be. Concretely: it evaluates through `src/Backtester.py` (one Optuna study per game+strategy, `load_if_exists=True`, walk-forward over the last `--days` draws) instead of the old rebuild-a-JSON-cache-and-total-the-profit loop; scores trials with the shared `score_from_summary` (profit per bet where a payout table exists, avg hits otherwise) rather than raw total profit; searches per-model-prefixed Keno subset sizes via the same `suggest_keno_subset`; and merges its results into `bestParams_<game>.json` including a `modelScores["XGBoost Model"]` entry, so the boosting model's vote is weighted in `WeightedEnsemble Model` like every other model's. Its `STRATEGIES` / `STRATEGY_DISPLAY_NAMES` tables have the same shape as the statistical ones, so adding a second boosting method (LightGBM, CatBoost) is a one-entry change. The old version could not run at enough strength to be high-confidence prediction.

### Training the meta-learner (`TrainMetaLearner.py`)

Trains both `MetaLearner Model` and `MetaLearnerV2 Model` per game:

```python
python3 TrainMetaLearner.py --games lotto,keno --days 300
```

For each game, it instantiates the 8 base models using that game's already-tuned `bestParams_<game>.json`, backtests them once with `collect_scores=True` over the last `--days` draws, and builds a (day, number) $\to$ [each model's score, actual-drawn label] training table across the game's full number range — reused for both model variants, so the expensive backtest doesn't run twice. It trains `LogisticRegression(class_weight="balanced")` for `MetaLearner Model` and `GradientBoostingClassifier` (class balance via `sample_weight`, since it has no `class_weight`) for `MetaLearnerV2 Model` — each evaluated on a walk-forward holdout split (last 20% of days) for an honest accuracy/AUC sanity check, then refit on the full window before saving to `data/models/<game>/meta_learner.joblyb` / `meta_learner_v2.joblib` respectively (same persistence convention as `XGBoost.py`).

For games with special columns (Euromillions/EuroDreams/VikingLotto), a **second model per variant** is trained the same way, purely on the special column's own `collect_scores` data (`_special_scores`) and its own number range — determined empirically from the data (`determine_special_range`), since no range is hardcoded anywhere (e.g. Euromillions stars turned out to be 1-12, EuroDreams dream number 1-5, VikingLotto super viking 1-8). Both models of a variant are persisted together in one artifact. `Backtester.backtest(collect_scores=True)` itself mirrors `Helpers.run_model_with_special_column`'s two-call convention (a main-only call, then a special-only call) so the two ranges are never mixed.

**Pick3 uses a positional formulation** instead of being skipped: a per-number ranking cannot express digit order or repeated digits, so for pick3 the candidates are the 30 `(position, digit)` pairs per day. Every pick3-capable base model exposes `score_positions()` - a list of three `{digit: score}` dicts, one per slot, derived from the model's own per-position internals (Markov marginalizes its pair-scored joint, MarkovMonteCarlo tallies its simulated tickets per slot, Poisson uses `1 - exp(-lambda[pos][digit])`, Laplace its discretized per-slot pmf, the per-position XGBoost its per-slot `predict_proba`), normalized per slot at training and serving time (`Helpers.normalize_position_scores`) so the learned weights don't depend on the tuned simulation counts. `Backtester.backtest(collect_scores=True)` records these as `<model>_position_scores` plus `actual_ordered` (the drawn-order truth), one classifier is trained **per position** (all four variants - logistic, gradient boosting, quantum kernel, VQC), the artifact carries `"positional": True` with `position_models`, and `Predictor.py` serves it as the argmax digit per slot in drawn order (duplicates allowed, never sorted). The same plumbing gives pick3 the classical `MetaLearner`/`MetaLearnerV2` rows for the first time. Held-out reporting for pick3 is per-position top-1 accuracy (chance 0.1) plus the positional ticket's real payout. `runHyperopt.sh` runs this automatically after `HyperoptStatistics.py` and `HyperoptBoost.py`, so the meta-learner is retrained on every fresh hyperopt pass.

### Quantum-assisted research

> **Status: Phase Q1 is implemented.** `QuantumMetaLearner Model` (quantum-kernel SVC) and `QuantumVQC Model` (variational quantum classifier) run as tracked prediction rows, trained by `TrainMetaLearner.py` from the same one-pass backtest table as the classical meta-learners and tuned by `HyperoptQuantum.py` (in the weekly `runHyperopt.sh`). Everything is simulated with a pure-numpy batched statevector engine in `src/QuantumModels.py` - four qubits is a 16-amplitude state, so no quantum SDK dependency is needed. Phases Q2-Q4 and the negative-control suite remain future work. Details below in each subsection.

Quantum computing is being investigated as an additional research layer alongside the existing statistical, deep-learning, boosting, and stacking models. The purpose is not to assume that a quantum model can predict an inherently random draw. The purpose is to test whether quantum feature maps or hybrid quantum-for classical models can detect reproducible structure that the existing classical models do not detect.

The main research hypothesis is:

> A correctly operated lottery-style game should not contain stable, exploitable temporal information. If a model appears to outperform suitable random and classical baselines, the result must remain reproducible under walk-forward validation, synthetic-random controls, shuffled-history controls, and an untouched holdout period.

A quantum model is therefore treated as another adversarial test of the sequence-generating process, comparable to evaluating the resilience of a system against another class of analysis. Failure to find predictive structure is evidence consistent with the modeled randomness assumptions, but it is not proof of perfect randomness. Apparent predictive structure is a signal for further investigation, not immediate proof that a game is predictable or biased.

#### Recommended first integration: `QuantumMetaLearner Model` (implemented)

Implemented as specified below, with these concrete choices: artifacts are `data/models/<game>/quantum_meta_learner.joblib` (quantum kernel) and `quantum_vqc_meta_learner.joblib` (variational), both carrying the classical artifacts' exact key layout plus `trained_at` and the resolved `params` (the README's metadata requirement), so `Predictor.py`'s `runMetaLearnerVariant` serves them unchanged - special-column separation, pick3 exclusion and Keno subsets included. Training is gated per game by `"useQuantumMetaLearner"` / `"useQuantumVqcMetaLearner"` in `bestParams_<game>.json`; they default **on** (the README suggested opt-in when heavy simulation was assumed - the numpy 4-qubit implementation trains in seconds to ~1 minute per game, so the flag is an off-switch instead).

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

Euromillions stars, the EuroDreams dream number, and the VikingLotto super viking number must not be mixed with the main-number range. `Pick3` was initially excluded because a per-number ranking does not represent positional digit order - this is now solved by the positional meta-learner formulation (see "Training the meta-learner"): pick3 gets `QuantumMetaLearner Model` and `QuantumVQC Model` rows too, one quantum classifier per digit position, and `HyperoptQuantum.py` tunes them on the **real pick3 payout** of the argmax ticket (mean `pick3_ticket_profit` per held-out day, with per-position top-1 accuracy as a smooth tie-breaker) - the most direct expression of the project's player-profitability goal, since pick3 is one of the two games with a payout table.

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

#### Quantum feature encoding (implemented)

Implemented exactly as diagrammed: `StandardScaler` -> `PCA(n_qubits)` fitted **only** on the training rows (inside each classifier's `fit()`), then RY angle encoding scaled by `encodingScale` with a CZ entangling ring, re-uploaded `encodingLayers` times (`QuantumFeatureMap` in `src/QuantumModels.py`). One practical finding from testing: `encodingScale` behaves as the kernel bandwidth / rotation wrap-around control and matters more than any other knob (~0.3-0.5 beat 1.0 everywhere tested) - it is the first thing `HyperoptQuantum.py` tunes.

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

#### Initial quantum model candidates (both implemented)

Both approaches below exist in `src/QuantumModels.py` as picklable sklearn-style classifiers (`fit`/`predict_proba`): `QuantumKernelClassifier` (fidelity kernel |<phi(x)|phi(y)>|^2 into an `SVC(kernel='precomputed', class_weight='balanced')`, class-balanced subsampling capped at `maxTrainSamples`) and `VariationalQuantumClassifier` (trainable RZ+RY layers with a CNOT entangling ring, Z-readout through a trainable affine+sigmoid, class-weighted cross-entropy, numpy Adam with exact parameter-shift gradients). The gradient implementation is verified against finite differences to ~1e-10.

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

#### Training integration (implemented)

Implemented as two extra entries in `TrainMetaLearner.py`'s variants loop, so the expensive backtest is collected once and all four meta-learners (logistic, gradient boosting, quantum kernel, VQC) train on the identical table, labels, chronological window and main/special separation. Hyperparameters come from the `quantumKernel_*` / `quantumVqc_*` keys in `bestParams_<game>.json` (tuned by `HyperoptQuantum.py`, defaults otherwise).

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

#### Evaluation metrics

Ticket-level hit counts remain important, but the quantum model must also be evaluated at the per-number probability and ranking levels.

### Advanced Architectural & Security Research (implemented)

All four research directions below are implemented and run as their own tracked prediction rows / monitoring layers in the daily `Predictor.py` pipeline. Each row is scored, ranked, lag-analyzed and peak-tracked exactly like every other model.

**Structural & Graph-based Analysis — `GNN Model` (`src/GNN.py`):**
Numbers are nodes in a co-occurrence graph built from the training history with exponential recency weighting (`gnn_decay`); stacked graph-convolution layers (hand-rolled GCN in Keras, no extra dependencies) learn per-number "community" embeddings, and a window-conditioned readout turns them into the standard per-position softmax prediction. Detects clusters of numbers drawn together more often than chance would allow, beyond simple pairwise decay. Tuned via `gnn_*` keys in `bestParams_<game>.json`.

**Long-Range Temporal Context — `Transformer Model` (`src/TransformerModel.py`):**
Sinusoidal positional encoding + pre-LN self-attention encoder blocks over a longer window (default 30 draws vs. TCN's 20). Unlike the LSTM/TCN recency bias, attention can weight any historical draw in the window regardless of distance. Same per-position softmax head, NaN/checkpoint discipline and fingerprinted weight caching as the other DL models. Tuned via `transformer_*` keys.

**Strategic Optimization (Agentic Prediction) — `RL Ticket Model` (`src/RLTicketModel.py`):**
Does not predict numbers - it learns ticket CONSTRUCTION. A pure-numpy REINFORCE policy (no TF, ~1-2s per game, wall-clock capped) trains on the stored day JSONs: features per number come from that day's other model rows (vote share, mean in-ticket rank) plus draw statistics, and the reward is the *real payout* (`Helpers.pick3_ticket_profit` / `keno_ticket_profit`) where a payout table exists, main-ticket hit count elsewhere. It runs in the second step after the `WeightedEnsemble Model` row so the full vote is part of its features, warm-starts from `data/models/rl_model/<game>_policy.json`, and during history rebuilds only trains on days strictly before the day being rebuilt (no look-ahead). Emits main numbers only (specials/bonus have their own ranges and payout logic). Keys: `rlTicketLearningRate`, `rlTicketEpochs`, `rlTicketSamplesPerDay`, `rlTicketTrainDays`, `rlTicketMaxTrainSeconds`; disable with `"useRlTicket": false`.

**Security & Randomness Detection (The Adversarial Layer):**
- **Unsupervised Anomaly Detection — `Autoencoder Model` (`src/AutoencoderAnomaly.py`):** a narrow-bottleneck conditional autoencoder that doubles as a tracked prediction row and as the integrity monitor. After each training run it computes the reconstruction NLL of every recent REAL draw plus a rolling z-score; a strongly negative z (the real draw suddenly became easy to reconstruct - a "predictability spike") is the alert condition. Stored per day as `anomalyWatch` in the day JSON, summarized per game in `modelPerformance.json`, and shown as the "AE anomaly" column (⚠ below z = -3) in the web UI's Randomness watch card. Its label smoothing deliberately defaults to 0 so the NLL stays an honest likelihood. Tuned via `autoencoder_*` keys.
- **Entropy & Divergence Analysis (`Helpers.generate_model_performance_report`):** per game, over the last 60 scored draws (Pick3 per digit position, averaged): KL(recent ‖ full history) for drift, KL(recent ‖ uniform) and normalized entropy for distance from a fair draw, a checkpoint trend series, and per model KL(predicted numbers ‖ real numbers) to expose models whose output distribution has departed from the actual process. Rendered as the "🔬 Randomness watch" card on the History page with a loose normal/watch tripwire. Entropy near 1 and KL near 0 mean the process looks fair and stationary; sustained movement is a signal for investigation, **not** proof of manipulation (rule changes, data artifacts and small windows all move these numbers).

**Enabling/disabling the research rows:** the heavy legacy DL models (LSTM/TCN/Unified*) stay behind the `--ai` flag (off in the daily cron - their training time/memory is the bottleneck). The three lightweight DL research rows (Transformer/GNN/Autoencoder) run **regardless of `--ai`**, inside the same one-shot spawned child process, and can be turned off individually with `"useTransformer": false`, `"useGnn": false`, `"useAutoencoder": false` in `bestParams_<game>.json` (same style as the statistical `useMarkov` toggles; `useLstm`/`useTcn`/`useUnifiedLstmTcn`/`useUnifiedLstmGruTcn` exist too for when `--ai` is on). At two smoke-test epochs on CPU the three new rows together cost less than a single `UnifiedLstmGruTcn` training. All four research models are hyperopt-tunable: Transformer/GNN/Autoencoder via `HyperoptDeepLearning.py` (see the hyperopt section), the RL row via `HyperoptRLTicket.py` (in the weekly `runHyperopt.sh`). Until a game's tuning has run, they use the defaults listed in `runUnifiedDeepLearningModels`.

### Randomness-discrimination experiment

A second quantum research track would test whether real draw windows can be distinguished from synthetic fair-draw windows.

The classification problem is:

```text
Class 0 = synthetic draws generated according to the game rules
Class 1 = real historical draws
```

The main question is:

> Can a classical or quantum model identify a reproducible difference between the enough strength to be high-confidence prediction the real history and a correctly simulated fair process?

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
> The best score found after applying the same complete model-on_the_fly_selection and optimization process to data known to be random.

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

Grover search could then amplify candidate tickets whose model score exceeds this threshold.
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

#### Phase Q1: quantum meta-learning (implemented)

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

What Q1 actually shipped is deliberately smaller than the layout below: one module, `src/QuantumModels.py`, holds the batched statevector simulator, the feature map, both classifiers and the fit factories (a 4-qubit numpy simulator does not need a package), with tuning in `HyperoptQuantum.py` and training/persistence riding the existing `TrainMetaLearner.py`. The fuller layout remains the target if Q2-Q4 grow the code base:

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
- Training, validation, and lockable periods
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
