
# Sequence Predictor

## The main idea

This project compares several probabilistic/statistical models, plus deep and boosting (LSTM/TCN) and gradient boosting (XGBoost), for lottery-style number prediction across multiple games (Euromillions, Lotto, EuroDreams, Keno, Pick3, VikingLotto, Joker+).

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

All of them now also agree on **which Keno subset sizes to play**: every model asks `getKenoSubsetSizes` for the hyperopt-tuned `use_5`..`use_10` flags. The DL rows previously hardcoded `range(5, 11)`, so they placed a bet at all six sizes while every statistical model played only the tuned ones (currently just `use_10`) — making those rows' tracked results non-comparable with the rest. `HyperoptDeepLearning.py` reads the same tuned choice via `tuned_keno_subset_sizes`, falling back to all six only when a game has no tuned flags at all, so its profit signal never silently becomes zero. Two further rows come from the foundation-model track (roadmap item 5): **`Chronos Model`**, a pretrained time-series model asked zero-shot for each drawn position's next value, and **`OrderStatistics Baseline`**, the historical mode of each position - the yardstick the first one has to beat, since a sorted game's position 1 is its minimum and therefore predictable in distribution without the draw being predictable at all.

Four additional, purely additive rows are then appended alongside them:

- **`WeightedEnsemble Model`** (Phase 0) — `Helpers.count_number_frequencies_from_new_prediction` counts every number suggested by any model/subset, weighted by that model's own Hyperopt/Backtester score (`bestParams_<game>.json["modelScores"]`, min-max scaled to a `[1, 2]` range so a poorly-scoring model is outweighted, never zeroed out; unscored models default to a neutral weight of 1). `addWeightedEnsemblePrediction` (`Predictor.py`) then turns that weighted vote into an actual ticket. For games with special columns (Euromillions star numbers, EuroDreams dream number, VikingLotto super viking), the main numbers and special column(s) are voted on and picked **separately** via `Helpers.count_number_frequencies_by_position`, then concatenated — the same positional split every individual model already keeps via `...` (see `Helpers.run_model_with_special_column`) — so the special slot(s) can't get crowded out by more numerous main-range numbers. For Keno, it also generates the same `use_5`..`use_10` sub-selections every individual model produces (see the shared subset generator below). The same weighted frequency also drives the `numberFrequency` chart shown in the web UI (home page and per-day history detail page), and for the positional games (pick3, Joker+) the row is a **vote per slot** instead (roadmap item 7, `Helpers.build_positional_vote_predictions`): each of the rows that own a model (`POSITIONAL_VOTE_DEFAULT_ROWS` in `Predictor.py` - the statistical pick3 models and the per-position boosting rows; the meta rows are aggregates of these and the DL research rows emit identical tickets, so an all-rows vote would double-count) votes for the digit it put in each slot, weighted like the pooled vote; the slot's digit is the argmax, ties go to the lowest digit, duplicates across slots are kept and so is drawn order - a set-style top-3 would sort the digits and could never express the repeated digit that 28% of pick3 draws contain. The row carries `positionConfidence` (winning vote share per slot) and the day JSON a `positionFrequency` (one digit histogram per slot); the pooled `numberFrequency` and its counters are untouched, Keno's subset machinery depends on them. Joker+ gets the row only with `"useJokerplusEnsemble": true` in its bestParams (see the Joker+ section).
- **`SubsetEnsemble Model`** (roadmap item 2) - the same vote as `WeightedEnsemble Model`, restricted to a *searched subset* of rows. `HyperoptEnsemble.py` selects which rows take part (one include flag per candidate row as Optuna parameters, plus a weighted-versus-flat vote choice) and writes the selection to `bestParams_<game>.json` (`subsetEnsembleModels`, `subsetEnsembleWeighted`); `addSubsetEnsemblePrediction` serves it with the shared recipe in `Helpers.build_vote_ensemble_predictions` (which `WeightedEnsemble Model` now uses too), so the served ticket is exactly what the tuner scored. Not served until a selection exists, and skipped on a day one of the selected rows did not run - a vote among the remaining rows would be a different ensemble under this row's name. Its own result is never written into `modelScores`, so it cannot feed back into its members' weights. For pick3 the same per-slot vote is taken over the tuned subset (any row the tuner selected - meta and DL rows included when they win, unlike the untuned row's fixed set); Joker+ only behind `useJokerplusEnsemble`.
- **`MetaLearner Model`** (Phase 1) — a real stacking meta-learner: instead of a hand-weighted vote, a small `LogisticRegression` (see `TrainMetaLearner.py` below) is trained to predict `P(number is drawn)` from each base model's own per-number score (`score_numbers()`, on `Markov`, `MarkovMonteCarlo`, `MarkovBayesian`, `MarkovBayesianEnhanced`, `PoissonMonteCarlo`, `PoissonMarkov`, `LaplaceMonteCarlo` and `XGBoost` — `HybridStatisticalModel` is deliberately excluded since it's itself a enough strength to be high-confidence prediction.
- **`MetaLearnerV2 Model`** — "lens diversity" (see the old Ideas entry this replaced): a second, independently-trained model class per game, `GradientBoostingClassifier` (`data/models/<game>/meta_learner_v2.joblib`) instead of `MetaLearner Model`'s `LogisticRegression`. Added as its own tracked row rather than replacing `MetaLearner Model` — a tree-based model can pick up nonlinear interactions between base models' scores a linear one can't, and since its errors won't necessarily correlate with the logistic regression's, real-life tracking can show which (if either) is actually worth keeping. Reuses the exact same base-model scores `MetaLearner Model` computes (`Predictor.py` caches them by feature-name set so they's not scored twice), and follows the same main/special-column split and Keno-subset generation.

### History rebuild & model-weight reuse (operational behavior)

Two behaviors worth knowing when running `Predictor.py` by hand:

- **Gap recovery is automatic and non-destructive.** When database files are highly fragmented or missing, `Predictor.py` anchors on the *newest* existing prediction json (scanning the game's actual draw dates, so twice-a-week games are handled correctly) and builds every missing draw after it — however large the outage gap — plus any interior holes within the last `-d/--days` draws (e.g. corrupted files you deleted). Existing files are never overwritten by this path; `update_matching_numbers` re-links the prediction-vs-result chain across old and new files afterwards.
- **`-r/--rebuild_history` force-regenerates (overwrites) the last `-d` draws**, existing files included — use it after history corruption. This flag previously existed but did nothing.
- **Deep learning is opt-in per run, time-boxed and crash-isolated.** `Predictor.py -a true` enables the heavy DL models (default off; `runPredictor.sh` passes it from the first cron run after this change is deployed - production must pull the new script); `-b false` disables boosting. Every model's training run has a wall-clock budget (`--dl-model-seconds`, default 240 s; `src/TimeBudget.py`): at the budget the run stops the way an early stop does - Keras still runs the epoch's validation, `EarlyStopping(restore_best_weights=True)` puts the best epoch's weights back, `ModelCheckpoint` has saved the best epoch - so the model still predicts and only training quality degrades, never the row. The check runs after every batch, so a model can overrun its budget by one batch; the first batch of a freshly compiled graph is the expensive one (up to 50 s on the TCN in testing). The whole per-day DL child additionally has a hard deadline (`--dl-timeout`, default auto = enabled models x (per-model budget + 60 s for each model's data load, graph compilation, prediction and save, which run outside the training budget) + 120 s child start-up: 37 min for seven models at the default; `--dl-model-seconds 0` disables the automatic deadline too): the cheap research rows run first, then the LSTM, then TCN/unified, the child checkpoints its rows after every model (a marker ties the checkpoint to its game and day, so a leftover from a crashed run is never served), and at the deadline it is killed and the rows it had finished are kept (`Recovered N deep learning row(s)` in the log). This is what let the heavy rows back into the daily cron: in August their unbounded training blocked the prediction flow for hours, so they were switched off and had no tracked history for a month. Runtime consequence at the defaults: up to ~37 min of DL per game-day, so up to ~4.3 h on a day every game has a draw (typically 3-5 games do, 2-3 h), multiplied by the days rebuilt after a gap - which is why `runHyperopt.sh` now waits for the predictor's lock instead of skipping the week (see *Hyperopt & backtesting*). Each model's total time for the day is logged (`LSTM Base Model: 187s` - load, training, prediction and save together), which is the number to tune the budget on. All DL training runs in a one-shot spawned child process per day: the container's 16GB memory cgroup OOM-killed the Predictor twice (2026-08-20 at 16.7GB RSS,  2026-08-22 at 10.3GB — a single long-lived process accumulating TF/Keras allocations across ~24 model trainings, killed with no trace since SIGKILL allows none). In a child, an OOM surfaces as a caught `BrokenProcessPool`: that day loses only its DL rows (the half-built file is auto-repaired next run), everything else still runs — and the per-day process recycling releases the memory that caused the kills in the first place.
- **NaN training runs are contained.** Exploding gradients on some hyperparameter combos (most often Keno's 20×80 output space) can drive the loss to `nan`. Training now stops at the first NaN batch (`TerminateOnNaN`) instead of burning the whole early-stop patience on dead epochs; if any earlier epoch was enough strength to be high-confidence prediction.

### Pick3 ticket validity of the base models

Pick3 rows must be **drawn-order tickets with duplicates allowed** (`[4,4,7]` is a real outcome). Two dormant defects in that area were fixed in September 2026: `MarkovMonteCarlo Model`'s pick3 ticket used to be the *sorted set of its top-voted unique digits* (never a repeated digit, order meaningless - so every straight/pair payout it was tracked on was scored against a scrambled ticket); it is now the per-slot mode of its simulated tickets in drawn order, the same tallies its `score_positions()` feeds the positional meta-learner, so its tracked pick3 history before that date is not comparable with what follows. And `Markov` now derives its fallback number range from the data it is built on (pick3 0-9, lotto 1-45, a special-only pass its special range) - nothing ever called `setGameRange`, so an unseen-context fallback used the 1-80 default for every game and could emit an impossible digit.

### Joker+ (`jokerplus`)

Joker+ is a 6-digit positional game (digits 0-9 drawn *with replacement* - 86% of draws repeat a digit, leading zeros count) plus a zodiac sign from 12. It is modeled as a positional game with one **categorical special column**: the sign is encoded to a code 0-11 at every CSV parse site (`Helpers.encode_zodiac`/`decode_zodiac`, canonical regulation order, accepting the CSV spelling `Tweeling`, the regulation's `Tweelingen` and the English names the live API returns - `DataFetcher` parses the API's `'430109Scorpio'` form and writes the Dutch CSV spelling), so every model, backtest and report works on integers; the UI decodes it back to the name. All positional machinery generalizes to 6 slots (statistical pick3-capable set via `run_model_with_special_column`, per-position boosting, DL per-position softmax with a 12-class sign head, the positional meta-learner with a sign `special_model` - so the classical and quantum meta rows exist for Joker+ too). The per-slot vote rows (`WeightedEnsemble Model` / `SubsetEnsemble Model`, roadmap item 7) exist for Joker+ but stay switched off unless `"useJokerplusEnsemble": true` is set in `bestParams_jokerplus.json`, and `RL Ticket Model` is deliberately absent - **the player cannot choose the digits** (combinations are system-generated; only the sign is selectable), so digit "ticket construction" would be fiction for this game. That also frames the research value: Joker+ cannot be made player-profitable through digit prediction - any digit predictability found here is an integrity/countermeasure signal - while the sign is the one real lever (its own tier only refunds the stake). Hyperopt covers it in the statistical, boosting, deep-learning and quantum tuners with profit per bet as the objective (`HyperoptQuantum` scores the per-slot argmax digits only, sign unknown).

### Hit counting & profit semantics

Hits are counted **per pool**, never as one flat set intersection over the whole result row (a predicted main equal to a star's numeric value is not a hit - the pools are separate drawings):

- **Euromillions / EuroDreams / VikingLotto**: main numbers score only against the drawn mains, the special column(s) (stars / dream number / super viking) only against the drawn specials. Displayed as **`N (M)`** = N main hits, M special hits, with green cell highlighting per pool.
- **Lotto** follows the real game's tiers: a play is 6 numbers scored against the 6 drawn mains, and the 7th drawn value (the bonus) only ever *supplements* a partial match - `5 (1)` (5 mains + bonus, amber cell) is a high tier but not the jackpot; `6 (0)` is the jackpot, and a full main match makes a bonus match mathematically impossible since the bonus differs from every drawn main. The bonus is matched against the played numbers themselves (a play has no bonus slot).
- **Keno / Pick3**: unchanged (single pool; Pick3 positional).
- **Joker+** follows its regulation (Reglement Joker+, Sept 2023): prizes go to groups of *consecutive* positional matches counted from the **left** end or the **right** end of the 6-digit number, plus the zodiac sign. Hits are therefore tracked as **`L/R (Z)`** - leading run / trailing run / sign (e.g. `3/1 (1)`); a full match is `6/0`. The UI colors leading-run cells green, trailing-run cells blue and a matching sign amber. "hits" in the report and lag analysis is L+R.

Games are classified once as **positional** (`Helpers.is_positional_game`: Pick3 and Joker+ - drawn order kept, digits may repeat, per-position models, no set-style rows such as MarkovBayesian/multi-label boosting - the vote ensembles take a per-slot form there) or set-based; every positional branch keys on that predicate, not on a game name.

This split runs through everything: the day-view highlights and `N (M)` hit column, `matchingNumbers` in the day JSONs (`matching_numbers` = mains, `special_matching_numbers` = specials/bonus), the History page match counts, `modelPerformance.json`'s `avg_hits` (mains only) and `avg_special_hits`, the lag analysis, the randomness watch, and `src/Backtester.py`'s per-day hit metrics and baselines (which now bet main-sized tickets) feeding every hyperopt objective.

Profit is **net profit per 1 EUR-stake convention** in both payout games. `pick3_ticket_profit` implements the official Belgian Pick-3 rules (Reglement Pick-3 juli 2024): the tracked ticket plays all four bet types (straight, box, front pair, back pair - 4 EUR stake; triples cannot play box, 3 EUR) and prizes cumulate, so an exact hit nets +676 (distinct digits) or +756 (with a double), including the 1 EUR units-digit consolation where applicable. `keno_ticket_profit` returns payout minus the 1 EUR stake (it previously returned gross wins and double-counted the stake on losses), so `profit_per_bet` is comparable across games. `jokerplus_ticket_profit` implements Art. 27 of the Joker+ regulation: stake 1.50 EUR; all 6 digits -> 20,000 EUR (200,000 EUR fixed-minimum jackpot with the sign); otherwise `T[L] + T[R]` with `T = {1: 2, 2: 5, 3: 20, 4: 200, 5: 2000}` (left and right groups cumulate) plus 1.50 EUR for a matching sign.

### Which draw a prediction is for

A day JSON is named after the draw it scored, and its `newPrediction` is the prediction for the **next** draw of that game - the rows that reappear as the following file's `currentPrediction`. The home page and the day page's "Next Draw Prediction" card therefore label that table with the draw it applies to ("for the draw of Mon 21 Sep 2026"). The date is derived from the game's own stored history rather than a hardcoded calendar (`nextDrawDate` in `server.js`): the weekdays of the last 40 draws up to that file are the game's schedule, and the prediction applies to the first later date falling on one of them - so a schedule change is picked up by itself, and an older page is read with the schedule of its own time. Checked against the full stored history, this reproduces the draw that actually followed in 765 of 767 cases; the two misses are a missing 31 December file for the two daily games, i.e. a gap in the stored data rather than a wrong schedule. On the home page a date in the past is shown in red with "already drawn, no newer run yet": the newest stored prediction is then for a draw that has already taken place, which happens when the predictor has not run since (for example while the weekly hyperopt still holds `process.lock`).

### Model performance report (History page)

After each prediction run, `Predictor.py` writes `data/database/modelPerformance.json` (`Helpers.generate_model_performance_report`): per game, every model's record over **all scored history** — average hits of the main ticket, best day, scored-draw count, and (for Keno/Pick3, which have real payout tables) total profit and average profit per bet. The web UI's History page (`/database`) renders it as a "Best model per game" card next to the game buttons; clicking a game row expands the full model ranking. Keno/Pick3 rank by profit per bet, other games by average hits — per-bet/per-draw averages rather than totals, since models joined the tracking at different times. Models with fewer scored draws than `minDrawsForRanking` (10, or the max available if lower) are listed but greyed out and sorted below the ranked ones, so a two-day-old model can't claim "best" off one lucky draw. The same report also feeds the History page's "Phase-shift check" card (tracked lag peaks, next section) and the "🔬 Randomness watch" card (entropy/KL/autoencoder-anomaly monitoring, see the security research section).

Next to the single-row ranking, each game carries a `combinations` section (`Helpers._build_combination_report`, roadmap item 2), rendered as the "🧩 Best combination per game" card: every pair and triple of the ranked rows is scored on the draws all of its members were scored on (the same `minDrawsForRanking` floor). Keno/Pick3/Joker+ rank sets by **net profit per draw** of playing every member's tickets - profit is additive, so a set beats its best row only when at least two rows are positive on those draws, and the card states whether it does; a greedy build-up beyond the best triple keeps adding rows while profit per draw improves. (Profit per *bet*, the single-row metric, is a stake-weighted mean of the members' and can never beat the best member; it is shown, not ranked on.) The hit-scored games rank by the **best line held per draw**: the mean over draws of the best member's hits, the quantity a player holding several lines cares about, which rewards rows whose good days do not coincide - it grows with every extra line, so pairs and triples are ranked separately and no greedy build-up is run. Because hundreds of combinations are scored, the card shows how many were evaluated and a **shuffled-history control**: the identical search re-run 100 times with each draw's real result moved to another day (every ticket keeps its day), which gives the distribution of the best combination *by luck alone* and a p-value - the share of shuffles whose best combination did at least as well. Ticket-versus-result scores are computed once (one-hot matrix products for hits and Keno, whose payout depends only on played and matched counts; the positional payout functions for Pick3/Joker+), so the whole section adds under two seconds to the daily run. At the time of writing (2026-09-14) no game's p-value is anywhere near small: the card shows the winner of a lottery among combinations, which is exactly what it is meant to make visible.

### Phase-shift (lag) analysis

Also part of the report and the History page: every day's `newPrediction` is scored not only against the draw it was enough strength to be high-confidence prediction as lag +1 but against the thirty draws that follow. If a model's signal were real but time-shifted, its average hits would peak consistently at some lag > 1; a flat curve across all lags means the hits come from draw-independent number-frequency structure, not timing. Pick3 is scored positionally (digit in the right place, chance level 0.3 per draw). Interpret peaks against the row's overall spread and sample size — with under ~100 scored draws per lag, the "best lag" bounces around by chance; a real phase shift would show the *same* peak lag persistently across time (and plausibly across related models), not a one-off maximum. That persistence check is automated: each run keeps only its single best peak per model and appends it to `data/database/lagPeakHistory.json` (one entry per run date, last 60 runs); the UI shows this run's peak with a z-score against the model's own lag profile, the most frequent peak across runs (highlighted once ≥3 runs and ≥50% agree), and a run-length-encoded chronological peak trail (`+30×5` = holding still, `+26 → +28 → +30` = drifting).

### Hyperopt & backtesting

`HyperoptStatistics.py` uses Optuna to tune each statistical model's parameters per game, driven by `src/Backtester.py`, which evaluates each model using rolling historical validation: for every historical draw, the model is trained only on previous draws and compared against the next real result. Each model's best parameters and best backtest score are written to `bestParams_<game>.json` (used by `Predictor.py` at prediction time), including a `modelScores` entry per model (its best backtest score, keyed by the same display name `Predictor.py` and also enough strength to be high-confidence prediction.

`HyperoptDeepLearning.py` tunes the deep learning models per game and per model type (`lstm_model`, `tcn_model`, `unified_lstm_tcn_model`, `unified_lstm_gru_tcn_model`, plus the research types `transformer_model`, `gnn_model`, `autoencoder_model`), each in its own Optuna study writing prefixed keys, scored primarily on held-out `val_loss` with real-draw profit as a small tie-breaker. It is **not** in the weekly `runHyperopt.sh` (DL training is the time/memory bottleneck); run it manually, and use `--models` to tune only what you need - e.g. `python3 HyperoptDeepLearning.py -g pick3 --models transformer_model,gnn_model,autoencoder_model` tunes just the cheap research rows without paying for LSTM/unified studies. `autoencoder_labelSmoothing` is pinned to 0 by its search space on purpose (the reconstruction NLL doubles as the anomaly signal).

`HyperoptQuantum.py` tunes the two quantum meta-learners (`quantumKernel_*` / `quantumVqc_*` keys): per game it collects the backtest score table once (`-d`, default 300 days, the same window `TrainMetaLearner.py` uses), then runs two Optuna studies against a chronological 75/25 **day** split - the variant is fitted (scaler/PCA included) on the early portion only and scored on the held-out days by mean per-day hits of the top-`draw_size` ranked numbers (the ticket-level metric that is actually played), with AUC as a +0.01 tie-breaker. Kernel trials run in under a second; VQC trials in seconds to half a minute. The collected table is persisted to `data/hyperOptCache/meta_score_table_<game>.joblib` and reused by `TrainMetaLearner.py` minutes later in the same weekly run (strictly validated: any new draw or changed base-model param recollects), so the pipeline's most expensive stage runs once, not twice. Each run selects `best_params` from its own trials only - cross-week trial scores were measured on different holdout windows and are not comparable. Note: the quantum rows' Keno subset mode/temperature keys (`quantumMetaLearnerSubset*`, `quantumVqcSubset*`) currently keep the softmax/0.5 defaults - unlike the classical meta rows they are not yet covered by HyperoptStatistics' subset tuning. Runs in the weekly `runHyperopt.sh` after `HyperoptRLTicket.py` and **before** `TrainMetaLearner.py`, so the weekly retrain always trains the quantum artifacts on freshly tuned params. Pick3 is tuned positionally: the objective is the mean real payout of the per-slot argmax ticket over the held-out days (+0.01 x per-position top-1 accuracy), and its score table lives in its own cache file (`meta_position_table_pick3.joblib`).

`HyperoptRLTicket.py` tunes the RL Ticket Model (`rlTicketLearningRate/Epochs/SamplesPerDay/TrainDays`) with an honest walk-forward over the game's own stored day JSONs: each evaluated day is re-predicted with `cutoffDate` set to that day (training sees only strictly earlier days) and scored against its real draw - real payout per day for Keno (enabled subset sizes only) and Pick3, main-ticket hits elsewhere. Pure numpy, minutes per game, so unlike the DL tuner it *is* part of the weekly `runHyperopt.sh` (after `HyperoptBoost.py`, sharing the same `process.lock`). Per-trial policies live in `data/hyperOptCache/rl_model`; the live `data/models/rl_model` policy is never touched by tuning.

`HyperoptEnsemble.py` selects the rows of the `SubsetEnsemble Model` (roadmap item 2). Like the RL tuner it works on the stored day JSONs rather than the Backtester - the only place where every tracked row (statistical, boosting, deep learning, meta-learner, quantum) exists side by side for the same draws, tuned as it was at the time. Per game: the candidate rows are those present on at least 80% of the last `--days` (default 120) scoreable days *and* on at least three of the last five (a row the pipeline no longer emits cannot be a member of a served row - the `LSTM Base Model` row stopped appearing in mid-August 2026, still covered 79 of 87 lotto days, got selected, and the served row was then skipped daily for a missing member), minus the two vote rows and the RL row (appended after the vote, so never available to it); a combination is one include flag per candidate plus `subsetEnsembleWeighted` (vote with the `modelScores` weights or count every member once); each combination votes the selected rows with `Helpers.build_vote_ensemble_predictions` on every day all of them exist and is scored like the report scores served rows - profit per bet over the enabled Keno subsets, main-ticket hits elsewhere. One evaluation takes 2-20 ms in-process, so **the space is enumerated exhaustively** whenever 2^rows x 2 fits the evaluation budget (`-t`, default 16384 - twelve candidate rows); beyond that a deterministic local search runs (hill-climbing over single flag toggles from the all-in vote, weighted and flat, and from the best pair), every combination evaluated once. The evaluated combinations are written into the game's Optuna study as completed trials, best first (`--record`, default the best 256; writing a trial to SQLite is ~70 ms, the search itself under a second), so `db.sqlite3` (and the Optuna dashboard, when started by hand) hold this run's ranking without duplicates. This replaced a first version that drove the search with Optuna's TPE through the process-per-trial runner: sampling boolean flags independently, TPE collapses onto its mode and re-proposes the same subset trial after trial (a production run spent 162 trials on 37 distinct subsets, one of them 62 times), and the runner spent ~2 s launching each 5 ms evaluation. A subset is scored only on the days all of its members exist, and it counts only when that is at least 60% of the window - every scored subset then sits on nearly the same draws, which is what makes their values comparable. This rule came out of testing: with a looser floor the search found a six-row subset that happened to break even on the ten days its members coexisted and ranked it above every subset that lost keno's house edge over 300 days - with thousands of subsets in the space such a fluke is always there to be found. The consequence is that the weeks-old boosting/meta-learner/quantum rows are not candidates yet at the default window; they join automatically as they age into it, or right away with a shorter window (`--days 60`), at the price of fewer draws for everyone. The value of a subset is a lower confidence bound on the per-day score, mean − std/√days (`--confidence-penalty`, default 1; 0 gives the plain mean), which mildly favours the subsets covering more of the window; a subset of fewer than two rows is not a subset. The all-in weighted vote (the `WeightedEnsemble Model` over the same rows) is always among the evaluated combinations and is printed next to the winner, so the log shows what the selection gained. Only the derived selection is written to `bestParams_<game>.json` (`subsetEnsembleModels`, `subsetEnsembleWeighted`, `subsetEnsembleObjective`, `subsetEnsembleMean`, `subsetEnsembleTunedOn` - the latter with the window, the number of candidates and combinations and the search kind) - never the include flags, and never the row's own score into `modelScores`. The selection is this run's best; older runs' trials stay in the study as history (user attribute `run`), but were scored on a different window and are not compared. Runs in `runHyperopt.sh` after `HyperoptRLTicket.py`. The positional games take the per-slot vote branch (`Helpers.build_positional_vote_predictions`, the recipe the served rows use): pick3 is scored with `pick3_ticket_profit` (profit per bet of the cumulative four-bet ticket), Joker+ with `jokerplus_ticket_profit` and only when `useJokerplusEnsemble` is set, and because their payouts are sparse (most days every subset loses the stake) the value adds 0.01 x the mean share of slots the ticket got right as a tie-breaker. Honest caveat: the selection is made on the rows' stored real-life history, so on the day it is written the subset is in-sample by construction - what makes it a fair row is that its tracked results accrue only from then on, out of sample, next to every other row.

`HyperoptBoost.py` does the same job for the boosting model, and is now a direct mirror of `HyperoptStatistics.py` rather than the separate legacy pipeline it used to be. Two run-time guards keep it inside a weekend: **CatBoost's tree depth is searched over 1-7** (its symmetric trees cost 2^depth - measured median minutes per trial by depth: 1-6 → 1-4, 8 → 24, 10 → 64, with single trials up to 12 hours, while XGBoost/LightGBM keep the full 1-10 range), and **every trial has a wall-clock budget** (`--trial-timeout`, default 1200 s) after which the Backtester terminates its worker pool and Optuna records the trial as pruned. Every trial is also stamped with a monotone **fit-cost proxy** (estimators x window, x 2^depth for CatBoost), and a later configuration at least as expensive as one that already hit the budget is skipped immediately as a predicted timeout (recorded as pruned, not counted toward `--trials`). This matters because Optuna's TPE sampler proposes its first ten trials at random and on keno roughly half the search space cannot finish one refit block inside the budget: the per-position formulation trains 20 multiclass models over 80 numbers, and XGBoost/LightGBM grow one tree per class per round, so keno LightGBM burned six 20-minute timeouts in its first seven trials, each more expensive than the one that finished. The gate only trusts timeouts recorded under the same budget tag (machine, window, cadence, budget), so stronger hardware or a larger budget starts from a clean slate. Two further levers attack the real cost driver - refitting a boosted ensemble for each of the 31 evaluated days: a **refit cadence** (`--refit-every`, default 7) trains one fit per block of consecutive days (block-aligned on the data *before* the oldest day of the block, so it is conservative, never leaky; every day still builds its own prediction features, and the fit cache keeps the main and special-column fits side by side) - identical for every trial, so the ranking is untouched while production keeps refitting daily. Measured on eurodreams CatBoostMultiLabel (2 trials x 21 days): 7.5 min at cadence 1 -> 66 s at cadence 7, with objective values in the same range. Threads are per library, from measurement: CatBoost scales (one fit: 4 threads = 3.1x) and gets the cores its fewer workers leave free; XGBoost/LightGBM stay single-threaded (giving XGBoost 5 threads made a study 5x *slower*), and the tuner pins OpenBLAS/OpenMP to one thread per process because every worker otherwise inherited a 16-thread BLAS pool whose spin-waiting cost ~10 cores of pure overhead per fit; and **conservative pruning** (`--prune-percentile`, default 25, 0 disables): the partial score after each completed day is reported to an Optuna `PercentilePruner` that stops only trials in the bottom quartile after half the window and only once five trials have completed - pruned trials are recorded as pruned, never as bad scores. **Trials of one study run in parallel processes** (`--parallel-trials`, default auto = the cores the per-trial workers leave idle: with 5 refit blocks on 15 cores, 3 trials at a time) for the XGBoost/LightGBM strategies, whose single-threaded fits used 5 of 15 cores; CatBoost keeps one trial at a time and the idle cores as fit threads. A further trial only launches while the machine keeps `--memory-reserve-gb` (default 2) available after it, with the expected footprint measured on the trials already running, and under 1 GB available the youngest trial is stopped and recorded as pruned rather than inviting the OOM killer. Each trial runs in its own process (the Backtester passes worker state through a module global, so two trials cannot share one), which also means a crashing trial no longer aborts the remaining strategies of a game, and trials left RUNNING by a killed run are marked failed when the study is reopened; the sampler runs with `constant_liar` so concurrent trials do not sample the same neighbourhood. Honest note: with a refit cadence above 1 all blocks fit concurrently, so pruning saves little compute (it mainly pays off at cadence 1); it is kept because it is free and protects the cadence-1 path. CatBoost's `border_count` (split candidates per feature) is exposed as a knob - `--catboost-border-count` (default 254, CatBoost's own), written to `bestParams_<game>.json` as `catBoostBorderCount`/`catBoostMlBorderCount` so `Predictor.py` serves the same value - for experimentation only: measured on this data (integer features with at most 50 distinct values) 32 vs 254 made no difference in either fit time (1014 s vs 1016 s for a 50-classifier multi-label fit) or ranking (Spearman 1.0), so the default is left untouched. Concretely: it evaluates through `src/Backtester.py` (one Optuna study per game+strategy, `load_if_exists=True`, walk-forward over the last `--days` draws) instead of the old rebuild-a-JSON-cache-and-total-the-profit loop; scores trials with the shared `score_from_summary` (profit per bet where a payout table exists, avg hits otherwise) rather than raw total profit; searches per-model-prefixed Keno subset sizes via the same `suggest_keno_subset`; and merges its results into `bestParams_<game>.json` including a `modelScores["XGBoost Model"]` entry, so the boosting model's vote is weighted in `WeightedEnsemble Model` like every other model's. Its `STRATEGIES` / `STRATEGY_DISPLAY_NAMES` tables have the same shape as the statistical ones, so adding a second boosting method (LightGBM, CatBoost) is a one-entry change. The old version could not run at enough strength to be high-confidence prediction.

**Shared run machinery - `src/HyperoptRunner.py`.** The process-per-trial coordinator described above is one module used by every tuner whose trials are expensive (`HyperoptEnsemble.py` evaluates in-process: at 2-20 ms per subset the coordinator's launch overhead would dominate, and its study is written as a ranking instead), so the same guarantees hold across `runHyperopt.sh`: trials of one study run in their own forked processes, in parallel where the objective is single-threaded and the memory gate allows; a crashing trial is marked failed instead of aborting the game's remaining strategies; Ctrl+C or `kill` of the coordinator tears down every trial process with its workers (and the trial processes die with the parent); trials left `RUNNING` by a killed run are marked failed on the next start; the sampler runs with `constant_liar` whenever trials run concurrently. Per tuner: `HyperoptStatistics.py` keeps one trial at a time (the Backtester already spreads each trial's days over every core) but gains the wall-clock budget (`--trial-timeout`, default 1200 s) and prewarms the Keno subset-tuning precompute in the parent so the trial processes inherit it instead of rebuilding it (43 min in production); `HyperoptRLTicket.py` runs one trial per core (`--parallel-trials`, auto) with a scratch policy directory per trial and BLAS pinned to one thread; `HyperoptQuantum.py` does the same on top of the once-collected score table, with a 1800 s deadline per trial (`--trial-timeout`); `HyperoptDeepLearning.py` stays GPU-serial in its spawned child but that child now has a budget too (`--trial-timeout`, default 2 h - a hung trial once had to be found and killed by hand) and stale trials are cleaned at study open. Operational note: `log/hyperopt*.log` is written only when a tuner is started through `runHyperopt.sh`; a run started by hand from a terminal logs to that terminal, and the Optuna `db.sqlite3` trial timestamps are the reliable progress record either way. `runHyperopt.sh` itself first waits (up to 6 h, noted in `hyperoptStatistics.log`) for the daily predictor to release `process.lock` instead of letting each tuner exit on a live lock - with the deep learning rows back in the daily run, a long Saturday run would otherwise silently skip the week's tuning.

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

Implemented as specified below, with these concrete choices: artifacts are `data/models/<game>/quantum_meta_learner.joblib` (quantum kernel) and `quantum_vqc_meta_learner.joblib` (variational), both carrying the classical artifacts' exact key layout plus `trained_at` and the resolved `params` (the README's metadata requirement), so `Predictor.py`'s `runMetaLearnerVariant` serves them unchanged - special-column separation, the positional pick3/Joker+ branch and Keno subsets included. Training is gated per game by `"useQuantumMetaLearner"` / `"useQuantumVqcMetaLearner"` in `bestParams_<game>.json`; they default **on** (the README suggested opt-in when heavy simulation was assumed - the numpy 4-qubit implementation trains in seconds to ~1 minute per game, so the flag is an off-switch instead).

The first quantum experiment should reuse the same per-number training data already collected for `MetaLearner Model` and `MetaLearnerV2 Model`.

For every backtested day and candidate number, the existing pipeline already produces a feature vector containing the scores from the eight base models (`BASE_MODEL_NAMES` in `src/ModelFactory.py`):

- `Markov`
- `MarkovMonteCarlo`
- `MarkovBayesian`
- `MarkovBayesianEnhanced`
- `PoissonMonteCarlo`
- `PoissonMarkov`
- `LaplaceMonteCarlo`
- `XGBoost` (the per-position variant for pick3/Joker+)

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

Implemented exactly as diagrammed: `StandardScaler` -> `PCA(n_qubits)` fitted **only** on the training rows (inside each classifier's `fit()`), then RY angle encoding scaled by `encodingScale` with a CZ entangling ring, re-uploaded `encodingLayers` times (`QuantumFeatureMap` in `src/QuantumModels.py`). One practical finding from testing: `encodingScale` behaves as the kernel bandwidth / rotation wrap-around control and matters more than any other knob. Early hand tests favoured ~0.3-0.5 over 1.0; `HyperoptQuantum.py` currently searches 0.5-2.0 in steps of 0.25 (the tuned values per game sit at 0.5-1.0, i.e. at the bottom of that range), so extending the lower bound is on the roadmap (item 6).

The eight base-model scores are classical values. Before they can be processed by a quantum circuit, they must be normalized and encoded into gate parameters.

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

Both approaches below exist in `src/QuantumModels.py` as picklable sklearn-style classifiers (`fit`/`predict_proba`): `QuantumKernelClassifier` (fidelity kernel |<phi(x)|phi(y)>|^2 into an `SVC(kernel='precomputed', class_weight='balanced')`, class-balanced subsampling capped at `maxTrainSamples`) and `VariationalQuantumClassifier` (trainable RZ+RY layers with a CNOT entangling ring, Z-readout through a trainable affine+sigmoid, class-weighted cross-entropy, numpy Adam with exact parameter-shift gradients). The parameter-shift gradient still needs a finite-difference unit test in the repository (none exists yet - roadmap item 6).

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

Quantum training was going to be opt-in on the assumption that circuit simulation would be slow; the numpy 4-qubit simulator trains in seconds to a minute per game, so both variants are **on by default** and the per-game flags are off-switches (no `bestParams_<game>.json` sets them today):

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

**Enabling/disabling the research rows:** the heavy legacy DL models (LSTM/TCN/Unified*) stay behind the `--ai` flag - off from mid-August 2026 because their unbounded training blocked the prediction flow, on again in the daily cron from the first run after the time-boxed training change is deployed, since every training run is then capped (`--dl-model-seconds`, see *History rebuild & model-weight reuse*). The three lightweight DL research rows (Transformer/GNN/Autoencoder) run **regardless of `--ai`**, inside the same one-shot spawned child process, and can be turned off individually with `"useTransformer": false`, `"useGnn": false`, `"useAutoencoder": false` in `bestParams_<game>.json` (same style as the statistical `useMarkov` toggles; `useLstm`/`useTcn`/`useUnifiedLstmTcn`/`useUnifiedLstmGruTcn` exist too for when `--ai` is on). At two smoke-test epochs on CPU the three new rows together cost less than a single `UnifiedLstmGruTcn` training. All four research models are hyperopt-tunable: Transformer/GNN/Autoencoder via `HyperoptDeepLearning.py` (see the hyperopt section), the RL row via `HyperoptRLTicket.py` (in the weekly `runHyperopt.sh`). Until a game's tuning has run, they use the defaults listed in `runUnifiedDeepLearningModels`; the `LSTM Base Model` likewise falls back to `LSTMModel`'s own defaults for any tuning key a game's `bestParams_<game>.json` lacks (jokerplus has never been DL-tuned - the first `--ai` run on 2026-09-17 lost its whole DL step to a `KeyError` on `batchSize` before this fallback existed) and logs which keys are untuned.

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



## Roadmap - upcoming work & research goals

Planning notes for the next tracks, in the order I'd tackle them (small platform items first, then the research items that build on them). Decisions already taken are marked; open points are listed so they can be settled before the work starts.

### 1. GUI: remove the Settings dropdown (small) - done

Done 2026-09-13. The top-right "⚙️ Settings" dropdown held a *global model filter* whose option list was hardcoded to seven statistical models (the pipeline tracks ~25 rows per game, so the filter silently hid most of them when used) and a "Keno played numbers" form whose value was only echoed in the navbar and never used in any calculation. Removed: the dropdown and the navbar status text, the `/playedModel` and `/playedNumbers` endpoints together with the body-parsing middleware only they used, the `selectedModel`/`filterDataByModel` plumbing (every page shows every row), and the CSS for the dropdown, the status text and the form fields. The button style survives as `.nav-btn` because the day page's "Back to History" link reuses it. Verified by rendering every page (home, History, all seven game pages, oldest and newest day page per game) with the previous and the new `server.js`: outside the removed navbar block the HTML is byte-identical, and the two removed routes answer 404.

### 2. Best *combination* of models, next to best model (medium) - done

Done 2026-09-14, both readings:

- **Portfolio of rows** - `Helpers._build_combination_report`, rendered as the "🧩 Best combination per game" card on the History page (details under *Model performance report*). One decision changed against the original plan: it ranked portfolios by summed payout over summed stakes, i.e. profit per **bet** - but a portfolio's profit per bet is the stake-weighted mean of its members', so it can never beat the best member and the "best combination" would always have been the top rows restated. Payout games therefore rank by **profit per draw** (additive: a set wins only if at least two rows are positive on the shared draws, and the card states whether it beats the best single row), hit-scored games by the **best line held per draw** (pairs and triples ranked within their size; no greedy build-up, since every extra line raises it). Pairs, triples and the greedy build-up, the `minDrawsForRanking` floor on the shared draws, the number of combinations evaluated and the shuffled-history control (100 shuffles, p-value) are in as planned. First reading of the real data: no game's p-value is anywhere near small.
- **Searched subset-ensemble** - `SubsetEnsemble Model`, selected by the new `HyperoptEnsemble.py` (see *Hyperopt & backtesting*) and served by `Predictor.addSubsetEnsemblePrediction`; the vote recipe moved into `Helpers.build_vote_ensemble_predictions` so the served row and the tuner cannot drift apart. pick3 joined on 2026-09-17 with item 7's per-slot vote (Joker+ behind `useJokerplusEnsemble`).

Status 2026-09-16: selections exist for all five games (first production run 2026-09-15, exhaustive over 256-512 combinations per game). That run's lotto and eurodreams selections include the `LSTM Base Model` row, which the cron stopped emitting in mid-August (`--ai` off), so those two served rows are skipped daily until the cron runs with `--ai` again (the time-boxed deep learning change) - from then on they are served without a re-run. The candidacy rule now additionally requires presence on three of the last five scoreable days, so a selection can no longer include a row the pipeline has stopped emitting.

### 3. Login screen with basic auth (small) - done

Done 2026-09-17, as a login form with basic user management rather than HTTP Basic Auth (decisions taken with the owner: a login page with roles needs a session and a logout, which Basic cannot express). `auth.js`: the administrator is `WEB_USER` / `WEB_PASSWORD` from the environment (never on disk; unset = open access for local development), further accounts live in the gitignored `config/users.json` as salted `scrypt` hashes and are added, deleted and given new passwords by the administrator on `/admin/users` - no self-service reset. Two roles: the administrator sees everything, a user the predictions and the History pages. Sessions are a signed `HttpOnly` `SameSite=Strict` cookie (secret from `WEB_SESSION_SECRET`, minimum 32 bytes, or generated once into `config/session.secret`), `Secure` over HTTPS, 24 h with renewal and a 7-day absolute limit, revoked by a password change or deletion; five failed sign-ins per user name and client address (proxy-aware through `trust proxy`) lock for 15 minutes; forms carry a CSRF token; user names cannot equal the administrator's. An independent security review of the first version found and I fixed: a backslash open redirect in the post-login target, the lockout keying on the proxy's address, sessions surviving password changes, the CSRF token rotating on renewal, user-name enumeration through sign-in timing, no minimum secret length, an unbounded failure map, write access to the user page in open mode, and a bare 401 on logout with an expired session. No npm dependency was added (Node's `crypto` does the hashing and signing). The Optuna dashboard was **removed from the UI entirely** (navbar link and the `optuna-dashboard` process the server used to start) instead of being proxied - start it by hand when needed. HTTPS stays the deployment's job (reverse proxy), documented under *Run server*.

### 4. Same models for crypto and shares - a predictor, not a trading bot (large, own phase plan)

Research question: *do the principles that fail to find structure in a fair lottery find any in market prices?* - the same statistical, deep-learning, boosting and quantum model families, the same walk-forward evaluation, the same negative controls. Not a trading bot: the output is tracked predictions and honest scoring.

**Decisions taken:** storage moves from day JSONs to **SQLite inside this repo** (the repo already carries `db.sqlite3` for Optuna; one more file, no new service), and the cadence is **daily bars** - one "draw" per trading day keeps parity with the lottery framing and the daily cron rhythm.

Design sketch:
- **Data sources (answer to "is there an open API?")**: yes, for daily bars there are free, keyless options. Crypto: Binance's public REST endpoint `GET /api/v3/klines` (1d candles, no key, generous limits) with CoinGecko's free API as the fallback and as the source of the market-cap ranking (keyless, roughly 10-30 requests per minute, `/coins/{id}/market_chart`). Shares: Stooq's daily CSV download (`https://stooq.com/q/d/l/?s=nvda.us&i=d`, no key, end-of-day, Euronext tickers with a suffix such as `asml.nl`) with `yfinance` (unofficial Yahoo Finance) as the fallback; keyed free tiers such as Alpha Vantage (25 requests/day) or Twelve Data are alternatives if either breaks. Free equity data is end-of-day and for personal/research use, which is exactly this use; nothing here needs a paid feed. One fetch per instrument per day is far below every limit. Stored in `data/markets.sqlite3` (tables: instruments, bars, predictions, results, model_performance).
- **Universe (decision): start small and fixed.** Crypto: the top 5 coins by market cap at kickoff (BTC, ETH and the next three at that date), quoted in USDT; shares: a handful of liquid names for testing, NVIDIA first, plus e.g. AAPL, MSFT, ASML. The list is frozen in the `instruments` table for an evaluation window and only changed between windows - re-selecting "the top 5" every day would smuggle survivorship and look-ahead into the results.
- **Daily tracking, exactly like the lottery games**: every day each model writes its prediction per instrument (bin, direction, confidence) to `predictions`; the next trading day's bar settles it - hit (exact bin / adjacent bin / direction) and the paper P&L of the fixed rule with fees go to `results`, and `model_performance` ranks the rows over the scored history with the same `minDrawsForRanking` guard and the same "compared with the result the day after" semantics the History page has for draws. Crypto trades seven days a week, shares five: settlement is per instrument at its next available bar, so a Friday share prediction settles on Monday and the sequence a model sees has no artificial holiday rows.
- **Models to check, market-specific: GARCH.** A GARCH(1,1) (the `arch` package) with a zero or AR(1) mean forecasts the next day's conditional variance; the resulting predictive distribution of the return gives the probability of every return bin directly, so it is a proper probabilistic baseline row ("GARCH Model") next to the lottery models, and its variance forecast is a volatility-regime feature the other rows can consume. The expected outcome is instructive for the research question: GARCH predicts *volatility* well and *direction* not at all, so any bin accuracy above 1/K it shows comes from volatility clustering (the extreme bins), which is the known predictable component of returns - a reference for judging what the lottery-born models find.
- **Models to check, market-specific: Regime HMM** (from arXiv [2603.04441](https://arxiv.org/abs/2603.04441), "Explainable Regime-Aware Investing", Boukardagha, Feb 2026). The paper's machinery is a strictly causal Gaussian hidden Markov model refit daily on an expanding window of per-asset features (daily log return, 60-day rolling volatility, 20-day mean return, all built from bars up to *t-1*), with the number of regimes reselected periodically by a complexity-penalised one-step-ahead predictive log-likelihood, and each fitted regime matched to a persistent "template" by the closed-form 2-Wasserstein distance between Gaussians so regime identities stay stable across refits (no label switching). In the paper those regime probabilities feed a mean-variance allocator; that layer is the trading bot this track does not build, so it is left out. **How it becomes a row (`Regime HMM Model`)**: per instrument, slice the instrument's return coordinate out of each regime's Gaussian, weight the regimes by the one-step-ahead state distribution (last filtered posterior propagated through the transition matrix) and integrate that Gaussian mixture over the K quantile bins - bin by bin, not through a single moment-matched Gaussian, because the mixture shape is exactly where the regime information lives - giving the per-position bin probabilities every markets row emits; the dominant template is recorded with the prediction so the History page can show which regime the model believed it was in. One HMM per market group (crypto trades seven days, shares five, so one joint model would face misaligned calendars), features z-scored on the training window only, quantile edges and template initialisation likewise strictly before the prediction date. **What decides whether it is interesting**: regimes add information beyond GARCH only if the regime-conditional *means* differ - the variance part is volatility clustering, which GARCH already captures. So the row ships with two ablation controls tracked as rows of their own: the same HMM with means forced to zero (variance-only) and a single Gaussian (K = 1, the homoskedastic random walk), and it is judged by a proper scoring rule (mean log-score of the bin probabilities, with a bootstrap interval against GARCH) before hits or paper P&L are read; tuning on P&L of the fixed rule would quietly turn the row back into the allocator. **Hyperopt knobs**: regime range (2-6) and complexity penalty, validation slice length and reselection cadence, expanding vs rolling window, template smoothing rate, feature toggles (returns only / plus volatility / plus momentum) and lookbacks, covariance type and floor, optional Ledoit-Wolf shrinkage of each regime covariance; the bin count K stays fixed because it is the game. **Dependencies**: `hmmlearn` (0.3.3, not installed yet, limited-maintenance mode; set `covariance_type="full"`, iterations, `min_covar` and restarts explicitly - its defaults, diagonal covariance and 10 EM iterations, would silently change the model) and `scipy.linalg.sqrtm` for the closed-form Wasserstein distance (no optimal-transport library needed). Compute is seconds per refit at this size; template edge cases the paper leaves open (several regimes mapping to one template, a template receiving none) need a rule, e.g. one-to-one assignment when counts agree and a distance threshold that spawns a template otherwise.
- **Honest reading of that paper, so its numbers stay out of this README as expectations**: it is a single-author preprint (v1, no venue, no code, no data statement) that gives no numeric value for any hyperparameter, no tickers, no test-window dates and no significance test. Its headline Sharpe 2.18 vs 1.59 (equal weight) and drawdown -5.43% vs -14.62% (S&P 500) are portfolio-allocation metrics, gross of transaction costs and with a zero risk-free rate, over an undated out-of-sample window that its own tables and figures pin to about 680 trading days (roughly June 2023 to February 2026) - a window without 2008, 2020 or 2022 and coinciding with a strong gold rally; the strategy's average book (29% dollar proxy, 22% gold, 22% bonds, 0.4% oil, 26% equities) means a static low-volatility tilt is never ruled out as the explanation, and its own cumulative-return figure shows the S&P 500 falling about 19% where its table prints -14.62%. With 2.7 years of data the standard error of an annualised Sharpe is about 0.6, so the reported gap is about one standard error. None of that touches the regime-detection idea, which is standard and reproducible; it is why the row above is imported as a design and validated here against GARCH and the controls, not taken as a result.
- **Target - the lottery analogy made concrete**: discretize each instrument's next-day return into K quantile bins (e.g. 10, fitted on the training window only); a day's "draw" is then the vector of bin indices across instruments, i.e. a *positional* game (instrument = position, bin = digit) - exactly the pick3/Joker+ machinery (`is_positional_game`, per-position boosting, positional meta-learner, per-position DL heads). Direction (up/down) is the 2-bin special case.
- **Scoring**: per-position hits (bin exact / adjacent) plus a *paper P&L* of a fixed rule (long when the predicted bin is above the median, otherwise flat, minus a realistic fee) - the "profit" role the payout tables play for keno/pick3. Statistically the bar is the efficient-market prior: a synthetic geometric-random-walk control with matched volatility (the market equivalent of the synthetic fair-draw control), shuffled-history control, and an untouched holdout period, before any accuracy above 1/K is read as predictability.
- **Reuse**: `DataLoader`/`Backtester`/hyperopt scripts gain a market-backed loader; models stay untouched (they only see integer sequences). The RL row becomes meaningful again here (position sizing is a real player lever, unlike Joker+ digits).
- **Open points**: whether the UI gets a separate "Markets" section or reuses the History page (proposal: a Markets tab with the same three cards - predictions, results, model performance - fed from SQLite instead of JSON).

### 5. Chronos-2 as a per-position predictor (medium) - done (zero-shot)

Done 2026-09-20 as decided: zero-shot, no fine-tuning. Every drawn position is one univariate series, `amazon/chronos-2` is asked for the next value's distribution, and that distribution over the game's labels is the per-position score shape the positional consumers already understand - so it serves as the `Chronos Model` row (`src/ChronosModel.py`) with `run()`, `score_positions()` and `score_numbers()` like any other model, including the main/special split through `run_model_with_special_column`.

**It runs outside the pipeline's interpreter, and that was not optional.** The foundation stack pulls numpy 2.x; the pipeline runs TensorFlow 2.16 on numpy 1.26. Installing it into the system interpreter would have upgraded numpy underneath the daily predictor. The libraries therefore live in their own directory (`/root/.foundation-libs`, `FOUNDATION_LIBS` to move it) that only a small worker process puts on its path (`src/foundation/chronos_forecast.py`, JSON lines over stdin/stdout). The worker is kept warm for the life of the model object, because a backtest asks for one forecast per day; it is launched with `PR_SET_PDEATHSIG` and an `atexit` close, since the history rebuild runs the statistical step in a pool of spawned processes and eight orphaned 0.4 GB workers is how a 16 GB box dies quietly. Missing libraries, a crash or a timeout disable the row with one log line and produce no ticket, which every caller already treats as "this model produced nothing".

**Install** (once per machine, ~630 MB plus a 457 MB model cache; `python3-venv` is not present on this box, hence a plain target directory rather than a virtualenv):

```
    python3 -m pip install --target /root/.foundation-libs --index-url https://download.pytorch.org/whl/cpu torch
    python3 -m pip install --target /root/.foundation-libs chronos-forecasting
```

Installing the second one pulls the CUDA build's dependencies even though torch is the CPU wheel; `rm -rf /root/.foundation-libs/nvidia /root/.foundation-libs/triton` reclaims 2.8 GB and was verified not to affect forecasting.

**Measured** on this box, CPU only: model load 1.4 s, all six lotto positions 0.16 s, a further point in time 0.28 s, peak memory 0.84 GB. Cheap enough to run for every game every day without touching the GPU or the deep-learning budget.

**How to read the row, and the baseline that makes it readable.** For the sorted set games the per-position series *are* order statistics: position 1 of lotto is the minimum of six draws from 1-45, so a forecast of "about 5" is arithmetic, not prediction. Chronos reproduces exactly that - its six lotto positions peak at 2, 11, 18, 25, 35, 44 - which is why the row is meaningless against chance and is instead served next to a new `OrderStatistics Baseline` row (`Baselines.ColumnFrequencyBaseline`): the historical mode of each position, computed from the same data through the same interface, so the History page ranks them side by side. A per-position model is interesting only once it beats that row. For the positional games an honest process gives near-uniform forecasts, which is what happens: pick3's digit distributions have a most-to-least-likely ratio of about 1.6, and the resulting ticket is degenerate (`[9, 9, 9]` on 2026-09-20) precisely because no digit stands out - there the *scores* are the output worth reading, not the ticket, and a large stable deviation would be the finding.

Both rows are on by default and switch off with `"useChronos": false` / `"useOrderStatisticsBaseline": false` in `bestParams_<game>.json`. Open in this item: feeding the Chronos scores to the positional meta-learner as a base-model feature (`ModelFactory.BASE_MODEL_NAMES`, which needs a meta-learner retrain), and TimesFM-3 as the second foundation model through the identical protocol - the worker already isolates the stack, so that is a second entry rather than a second risk.

### 6. Quantum research - what Phases Q0-Q4 still need (large, research)

Status checked against the code on 2026-09-11, phase by phase (the section "Quantum-assisted research" above keeps the full design; this is only what is still open):

- **Q0 classical controls - partial.** Done: `TCN.py` is tunable and runs as `TCN Base Model`; the daily cron runs with `--ai` again from the first run after the time-boxed training change is deployed (see *History rebuild & model-weight reuse*), so TCN accrues a production track record from then on - before it, it never was a production baseline. Open: remove or wire the dead `num_lstm_layers` setting (stored and written to metadata, never used by `create_model`); add the **synthetic-fair-history** benchmark (generate histories with the game's exact rules, run the full hyperopt + backtest + meta-learner process on them, record how good the "best" model looks on known-random data) and the **shuffled-history** benchmark (reorder the real draws, rerun everything); persist per-number **calibration and ranking metrics** (Brier / log-loss, precision at draw size, reliability bins) instead of the printed accuracy/AUC line in `fit_meta_model`; and reserve an untouched chronological **lockbox** that `TrainMetaLearner.py`'s 80/20 and `HyperoptQuantum.py`'s 75/25 splits never see until designs are frozen. These controls are the precondition for reading any quantum-versus-classical comparison, which is why this item comes before Q2-Q4.
- **Q1 quantum meta-learning - implemented, with gaps.** Missing against its own bullet list: the classical-kernel (RBF) SVM comparison variant, ideally as its own tracked row; a durable record of the four-way comparison (logistic, gradient boosting, quantum kernel, VQC) rather than a printed line; package versions in the artifact next to `trained_at` and `params`; a finite-difference unit test for the parameter-shift gradient; the `encodingScale` search space extended below 0.5 (the tuned values sit at its lower edge). Operational: the pick3 and Joker+ quantum artifacts in production were trained on `src/QuantumModels.py` defaults because `HyperoptQuantum.py` never completed a cron run for them (its September attempt died in a reboot); the shared runner work makes the next `runHyperopt.sh` pass deliver those keys.
- **Q2 randomness discrimination - missing entirely.** Nothing exists: no synthetic-history generator (Q0 builds it), no real-versus-synthetic window features (multi-hot for set games, positional for pick3/Joker+), no classifier suite (logistic, RBF SVM, random forest, gradient boosting, small NN, quantum kernel, VQC) on a chronological holdout, no repetition across seeds and splits, no interpretation step that checks stable distinguishability against rule changes and data artifacts before calling it evidence. This is the phase that speaks directly to the research goal (is the process distinguishable from fair?), so it is the first quantum phase to build once Q0 is in.
- **Q3 direct quantum scoring - partial in the weak sense.** The Q1 rows already emit a per-number quantum-assisted probability and reuse ticket construction, special columns, Keno subsets, Backtester and UI. What Q3 adds is a quantum model scoring candidates **from draw-history features** rather than from the eight base-model scores, routed through `Backtester(collect_scores=True)` like a base model, and compared on calibration and ranking before ticket hits.
- **Q4 candidate search - missing entirely.** No ticket-level score/threshold oracle, no toy Grover amplification in the numpy simulator, no cost comparison against direct ranking, and none of the reporting the section promises (circuit depth, two-qubit gate count, noise sensitivity, data-loading cost). Lowest value for the research question (it is a cost study, not a predictability study); last in order.
- **Negative-control suite - missing.** Shuffled history, synthetic fair history, irrelevant-feature control (a random column appended to the meta-learner table must not gain stable importance) and the lockbox: none exist; the only random comparator today is the per-day random-ticket baseline. Shared with Q0 above - build once, use for every row, quantum or not.
- **Fuller layout** (`src/quantum/` package, `experiments/quantum/` scripts) only when Q2-Q4 code arrives; one module is right for what exists.

Suggested order: Q0 + controls, then the Q1 gaps, then Q2, then Q3, with Q4 only if the earlier phases leave a reason.

### 7. `WeightedEnsemble Model` for pick3 - a vote per position (medium) - done

Done 2026-09-17 as phase 1 of the design below: the per-slot hard vote on the models' final tickets (`Helpers.count_position_votes` / `build_positional_vote_predictions`, new code beside the untouched pooled counters), served as `WeightedEnsemble Model` over the rows that own a model (`POSITIONAL_VOTE_DEFAULT_ROWS`) and as `SubsetEnsemble Model` over the subset `HyperoptEnsemble.py`'s new positional branch selects (exhaustive include flags like the set games, scored with `pick3_ticket_profit`, lower confidence bound plus a 0.01 x per-slot accuracy tie-breaker; its own score is never written into `modelScores`), with `positionConfidence` on the row and `positionFrequency` in the day JSON. Joker+ is mechanically covered (six digit slots plus the sign as slot 7) but gated off behind `"useJokerplusEnsemble"` for the reason given in the Joker+ section. The side finding is fixed: `server.js` now scores pick3 with the same cumulative four-bet model as `Helpers.pick3_ticket_profit`, so the History page's pick3 profits match the backtest and report. Deferred: phase 2 (the soft vote blending each model's normalized `score_positions()` distribution, which first needs a `positionScores` sidecar in the day JSON) and excluding the ensemble row from the RL row's pick3 vote features (left as for every other game, where the RL row already sees the `WeightedEnsemble Model`). The original design note follows.

Today the row does not exist for the positional games at all: `addWeightedEnsemblePrediction` returns early for pick3 and Joker+, and the only aggregate that is computed, `numberFrequency`, pools digits across the three slots. Simply removing that guard would be wrong: the set-game path would take the three most frequently voted *distinct* digits and sort them ascending, which destroys exactly what pick3 pays on (straight, front pair and back pair are exact-order, the consolation depends on the last slot, and 28% of draws repeat a digit, which a distinct-top-3 can never express).

**Decision: a weighted vote per slot.** For each of the three positions, every model's digit in that slot casts a vote weighted like today's ensemble (`modelScores`, min-max mapped to 1-2); the slot's digit is the argmax, ties go to the lowest digit (the same rule the positional meta-learner and the RL row use), duplicates across slots are allowed, drawn order is kept, and the row carries a per-slot confidence. Phase 1 votes with the models' final tickets, which are available on both serve paths (fresh prediction and the history rebuild, whose second step runs in a separate process and only sees the day JSON). Phase 2 blends in each model's normalized `score_positions()` distribution (Markov, MarkovMonteCarlo, PoissonMonteCarlo, LaplaceMonteCarlo and the per-position boosting rows all have one) with a tuned mixing weight; that needs `statisticalMethod` to persist a `positionScores` sidecar in the day JSON first. The day JSON also gets a `positionFrequency` (three digit histograms) for a per-slot chart; the pooled `numberFrequency` and the existing counting functions stay untouched because Keno subset generation and the Keno subset tuning depend on them - the per-slot vote is new code beside them.

**Scoring and tuning.** The row is scored with `pick3_ticket_profit` (cumulative four-bet stake) on the positional Backtester table, exactly like the positional meta-learner rows, and tuned by a positional ensemble study: per-model weights with 0 meaning excluded (for pick3 this *is* the searched subset-ensemble of item 2), hard versus soft vote, tie-break rule; objective = mean profit per day plus a small per-slot accuracy tie-breaker, on a chronological day split. It must run after `HyperoptBoost.py` (which may retune the boosting params the cached table depends on), so it lives in `HyperoptEnsemble.py` (created by item 2, sequenced after `HyperoptRLTicket.py` in `runHyperopt.sh`; today it skips the positional games), and its own score is never written into `modelScores` (it would feed back into its own weights).

**Pitfalls the design has to handle.** The `modelScores` weights are a min-max over a 31-day profit per bet - lucky-strike dominated and sign-blind (for Joker+ the three negative boosting scores still map to weights 2.0 / 1.75 / 1.0, so XGBoost counts double); the three DL research rows emit identical tickets day after day and the four meta rows are aggregates of the base rows, so an all-rows vote double-counts - the default include set is the rows that own a model (statistical and per-position boosting rows), with meta and DL rows opt-in through the tuned weights; the RL Ticket row should exclude the ensemble from its pick3 vote features for the same reason. Joker+ is mechanically free (six slots plus a sign vote) but the Joker+ section above deliberately leaves the ensemble out because the player cannot choose the digits - ship it gated off by a flag and decide later. Side finding to fix in the same change: `server.js` scores a pick3 ticket first-match-wins without deducting the stake, while the backtest uses the cumulative model (an exact `[1,2,3]` is 500 in the UI and 676 in the backtest), so every pick3 row's History profit disagrees with its tuning profit today.

### 8. Pipeline scheduling inside the server, instead of cron (large, platform)

The daily predictor and the weekly tuning chain are two cron entries today, and cron can only start things at a fixed hour. That costs real time and hides failures: on 2026-09-19 the predictor finished at 10:30 and the chain sat idle until its 14:00 slot, then ran 16.5 h to 06:29 - while a `process.lock` already guarantees that only one Python job runs at a time, so the waiting was pure loss. A tuner that finds the lock held simply exits, which is how a whole week's tuning can be skipped in silence.

**Decision: the Node server owns the schedule** (a `jobs.js`-style module required by `server.js`, so the web process stays the one thing pm2 supervises). A single-worker queue mirrors the lock; the weekly chain is triggered by *Saturday's predictor finishing*, not by a clock; a run missed while the box was down is caught up; and every job is launched through `setsid --fork`. That last point is load-bearing and measured: pm2 here runs with `treekill`, so a job spawned as an ordinary child of the server is killed 1.6 s into any `pm2 restart` - after `setsid --fork` the job's parent is PID 1, invisible to that tree walk, so a deploy, a crash or a restart of the web server cannot touch a running 12-hour tuning job. State (running, last run, exit code, duration, git SHA) lives in the gitignored `config/`.

Decisions taken with the owner: a Saturday chain that overruns into Sunday 09:00 **queues** that day's predictor behind it rather than pre-empting the tuner; the three crontab entries are **removed** at cutover (this README is then the only description of the schedule, and a `config/scheduler.disabled` flag file falls back without a restart); the UI keeps its public Tailscale funnel, which makes onboarding a new user a link rather than a VPN setup, so every trigger route is admin-gated, POST-only and CSRF-checked; the **Jobs page is admin-only**, but a small "prediction running" indicator is visible to every user, since a visitor should be able to tell that today's numbers are still being computed.

Done ahead of the rest (2026-09-20): the service half. `services.js` supervises the Council API and the admin **Jobs** page shows it (see *Jobs page and supervised services*), which is the same status, logging and page shape the scheduled jobs will use. Open work in the same item: the predictor and the tuning chain move onto that page as scheduled jobs; `log/*.log` gets rotation (`predictor.log` is already 20 MB); and the run record carries the git SHA so "safe to deploy now" is visible.

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

To run the complete flow the way the daily cron does (`runPredictor.sh`):

```
    python3 Predictor.py -a true
```

`-a true` enables the heavy deep learning rows (LSTM/TCN/unified); without it only the lightweight research rows run, so a day built by hand without the flag lacks the LSTM row and any `SubsetEnsemble Model` selection that contains it is skipped for that day. `--dl-model-seconds` (default 240) caps each model's training, `--dl-timeout` (default auto) the whole per-day deep learning child - see *History rebuild & model-weight reuse*. `-s false` skips the commit and push at the end.

To test model specific for example LSTM run:

```
    python3 LSTM.py
```

Check the __main__ section of the LSTM.py or GRU.py for pointing to data and set parameters for testing.

## Run server

The server is a NodeJS server with a plain simple html server side rendered front-end. No dependencies or heavy webpacks needed.
The server listens on the interface and port set in `config.js` (`127.0.0.1:3001` by default).

To run the server use the command:

```
    npm start
```

### Login and users

Set two variables to require a login (`auth.js`, roadmap item 3) - either in the environment or in a `.env` file in the repo root (gitignored; copy `.env.example`), which `config.js` loads at startup with real environment variables taking precedence:

```
    cp .env.example .env      # then edit WEB_USER and WEB_PASSWORD
    npm start
```

or, without a file:

```
    WEB_USER=admin WEB_PASSWORD='a long passphrase' npm start
```

A change to `.env` needs a server restart (nodemon does not watch it).

- `WEB_USER` / `WEB_PASSWORD` are the **administrator**: they live only in the environment, never on disk. With both unset the pages are open (local development) and the Users page is read-only with a notice; with only one of them set the server refuses to start rather than silently running open.
- The administrator sees everything plus the **Users** page (`/admin/users`), where accounts are added, deleted and given a new password. A user signs in with their own password and sees the predictions and the History pages.
- **Every signed-in user has an account page** (`/account`, reached from their name in the navbar): a user changes their own password there (current password required, and the change signs their other browsers out while keeping the one that made it), and sees the sign-ins recorded for their name - so a session they did not start is visible to them, not only to the administrator. The administrator's own credential lives in `.env` and is deliberately not editable from the web, since it is the one that can change everyone else's; that page says so and how to change it.
- **Sign-ins are recorded** (`audit.js`): successful and failed sign-ins, lockouts, sign-outs, password changes and account creation or deletion, each with the time, the user name, the caller's address and a truncated browser string. The administrator reads them on `/admin/activity`, and the Users page shows last seen and the number of sign-ins per account. The trail is `config/audit.jsonl`, gitignored like the accounts, one JSON object per line, rotated at 2 MB with one older file kept - so it is bounded by construction and a write can never corrupt an earlier line. A failure to write it is logged and swallowed: an audit trail that can break a sign-in is worse than one that misses a line.
- Accounts are stored in `config/users.json` (user name, salted `scrypt` hash - never the password; every file account is a plain user, the administrator is only ever the environment one), the session cookie is signed with a secret from `WEB_SESSION_SECRET` (or `SESSION_SECRET`; at least 32 bytes - 64 hex characters or a long passphrase; shorter values are ignored with a warning) or, when unset, one generated once into `config/session.secret`. The `config/` folder is gitignored on purpose: the daily predictor commits everything under `data/`, which is why the accounts do not live there. Nodemon ignores the folder (`package.json`), and pm2 should run without `--watch` at all (see *Run server*).
- Sessions last 24 hours, renewed while in use but never beyond 7 days from the sign-in; changing a user's password ends that user's sessions, rotating `WEB_PASSWORD` ends the administrator's, and a deleted user is out on the next request. The cookie is `HttpOnly` + `SameSite=Strict` and gets the `Secure` flag when the request arrived over HTTPS (directly or via `X-Forwarded-Proto`). Five failed sign-ins for a user name from one client address lock that combination for 15 minutes; a sign-in with an unknown name costs the same time as with a known one, so names cannot be enumerated. All forms carry a CSRF token that survives a session renewal. `WEB_TRUST_PROXY` (default `loopback`) tells Express which reverse proxy's `X-Forwarded-For` / `X-Forwarded-Proto` to believe, so the lockout keys on the real client behind a proxy on this machine instead of on the proxy.
- The login form sends the password in clear text, so put the server behind HTTPS (a reverse proxy or TLS terminator) when it is reachable beyond the local machine.

The Optuna dashboard is no longer started or linked by the server; start it by hand when you want it: `optuna-dashboard sqlite:///db.sqlite3`.

### First login, and what's new

A new account meets a short introduction the first time it opens any page: what this project is, and what each page is for - with the Users, Jobs and Activity step shown only to an administrator. After that, a feature that ships with a note pops that note once, and only for the accounts that may see it. `announcements.js` is the single file to edit: one entry per feature, newest first, `level: 'major'` to open the dialog or `'minor'` to light a dot on the **?** in the navbar, `audience: 'all'` or `'admin'`. Every string is escaped when rendered, so the file holds text rather than markup, and `npm test` checks the rules that keep it safe to edit - permanent unique ids matching their date, known level and audience, no HTML, newest first - because an edited id would pop an old note at everyone again.

What an account has already seen lives in `config/user-state.json`, gitignored like the accounts, deliberately not in `users.json`: the administrator has no record there, and interface state should not sit next to password hashes. The dialog is injected by `generateHeader`, so it finds the reader on whichever page they land on; dismissing it is a plain form post that works without JavaScript, "Later" hides it for the session, and the **?** link opens `/whats-new`, which lists everything and can replay the introduction.

### Is a run in progress?

Every Python entry point takes the same `process.lock` in the repo root, so the home page can answer "is the pipeline busy" without a scheduler, and keeps answering it whether the job was started by cron, by hand or (later) by the server: a banner names the job (`Today's predictions are being computed`, `Weekly tuning: boosting models`, ...) and how long it has been running, and the page reloads once a minute while it lasts. The command line behind the locked PID is what names the job, and a lock whose process is gone shows nothing - the scripts clean stale locks up themselves. Every visitor sees this, not just the administrator: a reader who wonders why today's draw is missing gets the answer instead of an empty card.

### Jobs page and supervised services

`services.js` keeps the processes the UI depends on running, and the admin-only **Jobs** page (`/admin/jobs`) shows them: state, pid, uptime, restart count, last exit, the tail of the service log and start/stop/restart buttons (POST, CSRF-checked, admin only). Today it supervises one service, the **Council API** (`src/llmCouncil/api.py`), which until now had to be started by hand - so the Council page was a dead link whenever nobody had. It starts with the server, restarts with a growing backoff if it dies, is marked failed instead of hammered after five exits within ten minutes, and stops with the server. Something already answering on its port (the other checkout, or an instance started by hand) is adopted and shown as such rather than duplicated. `COUNCIL_API_AUTOSTART=off` disables it; the service log is `log/councilApi.log`, rotated at 10 MB.

A service is an ordinary child on purpose: it is cheap to restart and must simply always be up, so it should come and go with the server. The scheduled pipeline jobs of roadmap item 8 are the opposite case - a twelve-hour tuning run must survive a deploy - and will be launched through `setsid --fork` from a module that reuses the status, logging and page shape established here.

Pm2 can also be used (this is how production runs it). Leave `--watch` off: the Python pipeline writes into the repo on every run - predictions, tuned parameters, model artifacts - and a watching pm2 would restart the server underneath its own visitors several times a day. Restart deliberately after a deploy instead (`pm2 restart sequencePredictor`). The login variables come from `.env`:

```
    pm2 start server.js --name predictor --time
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
