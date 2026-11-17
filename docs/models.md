# Models: how the predictions are made

This page describes every prediction row `Predictor.py` writes: the statistical models, the deep-learning and gradient-boosting rows, the foundation models, the rows that combine other rows (vote rows, meta-learner rows, reinforcement-learning rows), the variant rows, and how a day of predictions is built, time-boxed and rebuilt after a gap. How rows are scored and ranked is on [scoring](scoring.md), how their parameters are tuned on [tuning](tuning.md), the market games on [markets](markets.md), and how runs are started and scheduled on [operations](operations.md).

## The main idea

The project compares many independent prediction methods on lottery-style draws and on market games built in the same shape.

- Lottery games: `euromillions`, `lotto`, `eurodreams`, `jokerplus`, `keno`, `pick3`, `vikinglotto`.
- Market games: `crypto`, `shares`, `cryptoweek`, `sharesweek` (see [markets](markets.md)).

The research goal is **not** to predict jackpots. It is to test whether any game can be made profitable for the players over time and, if a model ever shows a reproducible edge, to understand why the process is predictable so countermeasures can be designed. Profit against the real payout or prize tables is the primary metric where one exists; hit averages are diagnostics (see [scoring](scoring.md)). Nothing on this page is a recommendation to play or to trade.

Every enabled method runs every time and stores its raw output as its own row in the day file (`data/database/<game>/<Y-M-D>.json`), so each method builds its own tracked history. Parameters and on/off flags are read from `bestParams_<game>.json`; a tuner changes a model's parameters, it does not switch models off.

### Two kinds of game

- **Set games** (lotto, euromillions, eurodreams, vikinglotto, keno): a draw is a set of distinct numbers, stored sorted.
- **Positional games** (`Helpers.POSITIONAL_GAMES`: pick3, jokerplus, crypto, shares and their week games): a draw is an ordered sequence. Tickets keep drawn order and may repeat a value (`[4,4,7]` is a real pick3 outcome).

Special columns are modelled apart from the main numbers through `Helpers.run_model_with_special_column` (`SPECIAL_COLUMN_COUNTS`): two stars for euromillions, one dream number for eurodreams, one viking number for vikinglotto, the sign for jokerplus. Lotto's bonus number is dropped before modelling.

On the five games in `MultiPick.PROBABILITY_ORDER` (vikinglotto, euromillions, eurodreams, keno, lotto) tickets are stored most-probable-first; see [scoring](scoring.md).

## Row overview

| Family | Rows |
|---|---|
| Statistical | `Markov Model`, `MarkovMonteCarlo Model`, `MarkovBayesian Model`, `MarkovBayesianEnhanched Model`, `PoissonMonteCarlo Model`, `PoissonMarkov Model`, `LaplaceMonteCarlo Model`, `HybridStatisticalModel` |
| Deep learning, heavy | `LSTM Base Model`, `LSTM Smoothed Model`, `TCN Base Model`, `UnifiedLstmTcn Model`, `UnifiedLstmGruTcn Model` |
| Deep learning, light | `Transformer Model`, `GNN Model`, `Autoencoder Model` |
| Gradient boosting | `XGBoost Model`, `XGBoostMultiLabel Model`, `LightGBM Model`, `LightGBMMultiLabel Model`, `CatBoost Model`, `CatBoostMultiLabel Model` |
| Foundation | `Chronos Model`, `TimesFM Model`, and their yardstick `OrderStatistics Baseline` |
| Market rows | `GARCH Model`, `Regime HMM Model`, `Regime HMM ZeroMean Model`, `Regime HMM Single Model` (see [markets](markets.md)) |
| Meta-learner | `MetaLearner Model`, `MetaLearnerV2 Model`, `QuantumMetaLearner Model`, `QuantumVQC Model`, `ClassicalSVM Model` |
| Vote | `WeightedEnsemble Model`, `SubsetEnsemble Model` |
| Reinforcement learning | `RL Ticket Model` (lottery), `RL Position Model` (markets) |
| Variants | `<row> · Memorise`, `<row> · Sampled` |
| Council | `Council (record): <member>`, `Council (draws): <member>`, and `Council (record): combined`, `Council (draws): combined` (lottery games) |

## Statistical models (`src/`)

Each model reads the historical draws under a different statistical interpretation.

- **Markov** builds a transition chain of configurable order over the draws, with recency weighting, pair decay and smoothing. `markovTransitionMode` selects the chain: `column` (default; the same position from one draw to the next) or `within` (number to next number inside the sorted draw, set games only). The fallback number range is derived from the data the model is built on.
- **MarkovMonteCarlo** wraps a `Markov` instance and votes over many simulated tickets. On positional games its ticket is the per-slot mode of the simulated tickets, in drawn order.
- **MarkovBayesian** / **MarkovBayesianEnhanced** apply Bayesian smoothing (alpha, softmax temperature, minimum-occurrence threshold) on top of the Markov transition counts.
- **PoissonMonteCarlo** estimates which numbers recur under historical position-based count rates, sampled by Monte Carlo simulation.
- **PoissonMarkov** blends Poisson and Markov probabilities with a tunable weight.
- **LaplaceMonteCarlo** models each position by its historical centre and spread and samples from it.
- **HybridStatisticalModel** votes across Markov, MarkovBayesian, PoissonMonteCarlo, LaplaceMonteCarlo and PoissonMarkov: a frequency vote on set games, a per-position vote when the game is unsorted.

All models share one interface (`setDataPath`, `run(...)`), so `Predictor.py` and the Backtester drive them interchangeably.

- `score_numbers()` returns a `{number: score}` dict instead of a ticket. Every statistical model except `HybridStatisticalModel` has it, and so do the boosting rows, the foundation models and the baseline. The meta-learner rows and `Backtester.collect_scores` use it.
- `score_positions()` returns one score dict per slot. Markov, MarkovMonteCarlo, PoissonMonteCarlo, LaplaceMonteCarlo, the per-position boosting rows, the foundation models and the baseline have it; the positional meta-learner uses it.

### Flags

| Flag | Default when the key is absent |
|---|---|
| `useMarkov` | off |
| `useMarkovMonteCarlo` | off |
| `useMarkovBayesian` | on |
| `usevMarkovBayesianEnhanced` | on (the key is spelled this way) |
| `usePoissonMonteCarlo` | off |
| `usePoissonMarkov` | on |
| `useLaplaceMonteCarlo` | off |
| `useHybridStatisticalModel` | on |

On positional games MarkovBayesian, MarkovBayesianEnhanced, PoissonMarkov and HybridStatisticalModel are skipped (they have no per-position model), and `useMarkov`, `useMarkovMonteCarlo`, `usePoissonMonteCarlo` and `useLaplaceMonteCarlo` are on when their key is absent.

## Deep learning & boosting

All deep-learning rows frame the prediction the same way: a sliding window of `windowSize` past draws feeds the network, and the output is one independent softmax per slot (`Dense` then `Reshape((slots, classes))` then softmax on the last axis). That is "which value is most likely at this position", not a joint distribution over the draw. Special columns get their own output head with their own class range.

Common behaviour of every deep-learning row:

- Training uses `EarlyStopping(restore_best_weights=True)`, `ReduceLROnPlateau`, `ModelCheckpoint`, `TerminateOnNaN` and a wall-clock budget (see *The daily deep-learning time budget*).
- Weights are saved with a fingerprint of the architecture-shaping parameters (`<model>.weights.h5` and `<model>.fingerprint.json`). The next run loads them only when the fingerprint matches, so a tuned change of units, layers or window trains fresh instead of loading stale weights.
- A row's history depth is `<prefix>_yearsOfHistory`, falling back to the bare `yearsOfHistory` (default 20); the LSTM rows use the bare key.
- On keno every row plays the subset sizes `getKenoSubsetSizes` returns: the sizes whose `use_5` .. `use_10` flag is true in `bestParams_keno.json`.

### LSTM (`src/LSTM.py`)

`Reshape` (the window flattened to one symbol sequence) → `Embedding` → one `Bidirectional(LSTM)` layer → `Dropout` → `Dense` → per-position softmax heads. Metrics: accuracy, `digit_accuracy`, `any_digit_hit`.

- The network always has one recurrent layer: the `num_lstm_layers` / `num_bidirectional_layers` setters do not change the built architecture, and the `SelfAttentionBlock` class in the file is not used by `create_model`.
- Two rows run on it (`Predictor.py` `LSTM_ROWS`):

| Row | Flag | Loss | Weights |
|---|---|---|---|
| `LSTM Base Model` | `useLstm` | plain categorical cross-entropy; the tuned `labelSmoothing` is stored but not applied | `data/models/lstm_model` |
| `LSTM Smoothed Model` | `useLstmSmoothed` | the same network on the same tuned keys, with `labelSmoothing` in the loss of both heads | `data/models/lstm_smoothed_model` |

- The LSTM keys in `bestParams_<game>.json` are unprefixed: `batchSize`, `epochs`, `num_lstm_layers`, `lstm_units`, `num_bidirectional_layers`, `bidirectional_lstm_units`, `optimizer_type`, `learningRate`, `dropout`, `l2Regularization`, `useFinalLSTMLayer`, `earlyStopPatience`, `reduceLearningRatePatience`, `reduceLearningRateFactor`, `windowSize`, `labelSmoothing`.
- A missing key falls back to `LSTMModel`'s own default and the run logs how many keys are untuned.
- The day file's `newPredictionRaw` and `labels` are those of `LSTM Base Model`.

### TCN (`src/TCN.py`)

Stacked dilated-convolution `TCN` layers (the `keras-tcn` package) → two `SelfAttentionBlock`s (`MultiHeadAttention` + feed-forward + `LayerNorm`) → `GlobalAveragePooling1D` → per-position softmax heads. Loss: categorical cross-entropy with label smoothing. Metrics: accuracy, top-3 accuracy, `digit_accuracy`, `any_digit_hit`, `full_draw_accuracy`. Row `TCN Base Model`, keys `tcn_*`.

### UnifiedLstmTcn / UnifiedLstmGruTcn (`src/UnifiedLstmTcn.py`, `src/UnifiedLstmGruTcn.py`)

Architectural fusion rather than an ensemble. One `Embedding` step keeps the window as the time dimension, then a `Bidirectional(LSTM)` branch and a stacked `TCN` branch (plus a `GRU` branch in the three-branch model) are **concatenated** before the shared attention, pooling and output head, so the branches' representations mix before the prediction is made. Rows `UnifiedLstmTcn Model` and `UnifiedLstmGruTcn Model`, keys `unifiedLstmTcn_*` and `unifiedLstmGruTcn_*`. Loss and metrics as the TCN.

### Light rows

`Transformer Model`, `GNN Model` and `Autoencoder Model` share the same `run()` interface and per-position softmax head. They are research probes (long-range attention, co-occurrence structure, anomaly detection) and run without `--ai`. They are described under *Advanced Architectural & Security Research*.

### Parameters of the prefixed rows

`Predictor.py` `UNIFIED_DL_MODELS` lists the six prefixed rows; `configureUnifiedModel` reads `<prefix>_<key>` with these defaults:

| Key | Default | Key | Default |
|---|---|---|---|
| `batchSize` | 16 | `epochs` | 1000 |
| `lstmUnits` | 64 | `gruUnits` | 64 |
| `tcnUnits` | 64 | `numTcnLayers` | 2 |
| `dModel` | 64 | `numLayers` | 2 |
| `ffnFactor` | 4 | `numHeads` | 4 |
| `keyDim` | 32 | `gcnUnits` | 32 |
| `numGcnLayers` | 2 | `embeddingDim` | 16 |
| `decay` | 0.999 | `latentDim` | 16 |
| `encoderUnits` | 64 | `numEncoderLayers` | 2 |
| `dropout` | 0.3 | `l2Regularization` | 0.0005 |
| `learningRate` | 0.001 | `labelSmoothing` | 0.05 |
| `earlyStopPatience` | 20 | `reduceLearningRatePatience` | 5 |
| `reduceLearningRateFactor` | 0.5 | `windowSize` | 20 |

Row-specific defaults: the Transformer's `windowSize` is 30; the Autoencoder's `labelSmoothing` is 0.0 and its `dropout` 0.2.

| Row | Prefix | Flag | Needs `--ai` | Weights folder |
|---|---|---|---|---|
| `TCN Base Model` | `tcn` | `useTcn` | yes | `data/models/tcn_model` |
| `UnifiedLstmTcn Model` | `unifiedLstmTcn` | `useUnifiedLstmTcn` | yes | `data/models/unified_lstm_tcn_model` |
| `UnifiedLstmGruTcn Model` | `unifiedLstmGruTcn` | `useUnifiedLstmGruTcn` | yes | `data/models/unified_lstm_gru_tcn_model` |
| `Transformer Model` | `transformer` | `useTransformer` | no | `data/models/transformer_model` |
| `GNN Model` | `gnn` | `useGnn` | no | `data/models/gnn_model` |
| `Autoencoder Model` | `autoencoder` | `useAutoencoder` | no | `data/models/autoencoder_model` |

All `use…` flags of the deep-learning rows default to on. The seven network types are tuned by `HyperoptDeepLearning.py`, which dispatches through `MODEL_REGISTRY` (`lstm_model`, `tcn_model`, `unified_lstm_tcn_model`, `unified_lstm_gru_tcn_model`, `transformer_model`, `gnn_model`, `autoencoder_model`); see [tuning](tuning.md).

### Warm-start guard

Each row trains on from its loaded weights. `Helpers.fit_with_warm_start_guard`, which every deep-learning row class trains through, scores the loaded weights on the validation split before `fit` and puts them back when no epoch's `val_loss` beats that score, so a daily run cannot drift away from the held-out draws.

- Both numbers include the L2 term, so they compare.
- A run that ends on NaN gets the loaded weights back; a freshly initialised model is not guarded.
- One log line per row says which way it went: `Warm-start guard (<row> <game>): ...`.
- `last_val_loss`, which the tuner reads, is the `val_loss` of the weights actually served.
- The fingerprint carries `weightsVersion` (`Helpers.WEIGHTS_VERSION`, 2). Weights saved under another version train fresh once.

### NaN containment

- Training stops at the first NaN batch (`TerminateOnNaN`).
- If an earlier epoch was healthy, its best-`val_loss` checkpoint is restored; a NaN epoch never ranks as best.
- If the very first batch is NaN the model raises and that day's row is skipped, so a ticket from NaN weights is never recorded or voted on. Corrupted weights are never saved.

### Learning check (`src/LearningCheck.py`)

Each deep-learning row carries a label that says whether it does better than counting. It is a label only: nothing ranks, filters, hides or reweights on it.

After training, the row is read on its validation windows:

- `valLoss`: its cross-entropy without the L2 term.
- `baselineLoss`: the cross-entropy of the training split's per-position class frequencies, smoothed the way the row's own loss smooths.
- `margin`: how far below the baseline the validation windows' own frequencies get, the lowest loss any input-blind output can reach there.
- `verdict`: `learns` when `valLoss < baselineLoss - margin`, otherwise `counts-only` (a row worse than the frequencies included).
- `sameTicket` / `inputBlind`: the share of validation windows that give the most common argmax ticket; `inputBlind` is true at 95% or more.

The check is stored on the row as `learningCheck` and carried into `modelPerformance.json` as `learning_check`, dated by the day file it was checked on. Past days cannot be labelled, because the day files do not keep the probabilities.

## Gradient boosting (`src/BoostingBase.py`)

Three libraries in two formulations give six independently tracked rows. All share one implementation in `src/BoostingBase.py`, so a difference between rows is attributable to the library or the formulation. Each backend supplies only its classifier; window features, label encoding, ticket construction, subset generation, fit cache and persistence are common code.

| Row | Library | Formulation | Key prefix | Flag |
|---|---|---|---|---|
| `XGBoost Model` | XGBoost | per-position multiclass | `xgBoost` | `useBoost` |
| `XGBoostMultiLabel Model` | XGBoost | multi-label | `xgBoostMl` | `useXgBoostMultiLabel` |
| `LightGBM Model` | LightGBM | per-position multiclass | `lightGbm` | `useLightGbm` |
| `LightGBMMultiLabel Model` | LightGBM | multi-label | `lightGbmMl` | `useLightGbmMultiLabel` |
| `CatBoost Model` | CatBoost | per-position multiclass | `catBoost` | `useCatBoost` |
| `CatBoostMultiLabel Model` | CatBoost | multi-label | `catBoostMl` | `useCatBoostMultiLabel` |

All six flags default to on. `-b false` on the command line disables boosting for a run. Each row trains on one thread and keeps its models in `data/models/<prefix in lower case>_<game>_models`.

### The two formulations

- **Per-position multiclass** (`PerPositionBoostingPredictor`): one classifier per draw slot, over a flattened window of the `<prefix>PreviousDraws` preceding draws as raw values. It fits positional games, where slot identity is real and values repeat. On a set game it imposes slot structure the game does not have, so several positions can pick the same number and the ticket is refilled to full size.
- **Multi-label** (`MultiLabelBoostingPredictor`): one binary "is this number in the next draw" classifier per number in the range, over a multi-hot encoding of the window. It matches a set game directly, its `score_numbers()` is a `P(drawn)` per number, and it is cheaper (keno: 80 binary fits instead of 20 multiclass fits over 80 classes). It is **skipped on positional games**, where set membership can represent neither order nor repeated values.

### Parameters

`BOOSTING_PARAM_SUFFIXES` and `apply_boosting_params` in `BoostingBase` are shared by `Predictor.py` (reading tuned values) and `HyperoptBoost.py` (writing them), so the two cannot drift on a key name. A key is `<prefix><Suffix>`:

| Suffix | Default | Suffix | Default |
|---|---|---|---|
| `Estimators` | 200 | `LearningRate` | 0.1 |
| `Maxdepth` | 3 | `PreviousDraws` | 11 |
| `TopK` | 16 | `ForceNested` | true |
| `Subsample` | 1.0 | `ColsampleByTree` | 1.0 |
| `MinChildWeight` | 1.0 | `RegLambda` | 1.0 |
| `SubsetMode` | `softmax` | `SubsetTemperature` | 0.5 |

One further suffix applies to CatBoost only: `BorderCount` (the number of split candidates per feature, default 254).

### Per-library notes

- LightGBM grows leaf-wise, so `num_leaves` is capped at `2**max_depth` and `subsample_freq=1` is set (without it `subsample` is ignored).
- CatBoost's symmetric trees make depth more expensive; its `min_child_weight` maps to `min_data_in_leaf`, and `subsample` switches the bootstrap to Bernoulli.

## Foundation models (`src/FoundationModel.py`)

Two pretrained time-series models are asked, zero-shot and without fine-tuning, for the next value of every drawn position.

| Row | Class | Checkpoint | Flag (default) |
|---|---|---|---|
| `Chronos Model` | `src/ChronosModel.py` | `amazon/chronos-2` | `useChronos` (on) |
| `TimesFM Model` | `src/TimesFmModel.py` | `google/timesfm-3.0-pytorch` | `useTimesFm` (on) |
| `OrderStatistics Baseline` | `Baselines.ColumnFrequencyBaseline` | none | `useOrderStatisticsBaseline` (on) |

### How a forecast is made

- Every drawn position is one univariate series (position 1 of lotto over time, position 2, ...). The model returns a quantile curve for the next value; `src/foundation/quantile_labels.py` turns it into a probability per label of the game.
- The curve is anchored at the bounds of the game's own support (first label − 0.5, last label + 0.5) before interpolating, so tail mass is spread over the stretch it occupies and not folded onto the first and last label.
- `foundationContext` (default 512) is the number of past draws either model is shown.
- Both are thin subclasses of `FoundationModel`, which owns the worker, the cache and the ticket construction. They expose `run()`, `score_positions()` and `score_numbers()` like any other model, including the main/special split.
- Positional games: the most likely value per slot, in drawn order, duplicates allowed.
- Set games: one distinct number per position, each slot's best value that is not used yet and lies above the previous pick.
- Exact ties within a slot go to the value closest to the slot's own mean, then the lower value.

### Where the model runs

The foundation libraries need a newer numpy than the pipeline's TensorFlow, so they live in their own directory and never in the pipeline's interpreter.

- Library directory: `/root/.foundation-libs`, or the path in the `FOUNDATION_LIBS` environment variable.
- A small worker process (`src/foundation/chronos_forecast.py`, `src/foundation/timesfm_forecast.py`) puts that directory on its path and exchanges JSON lines over stdin/stdout.
- The worker stays warm for the life of the model object. It is launched with `PR_SET_PDEATHSIG` and closed at interpreter exit, so no worker outlives its parent.
- The first call waits up to 300 s (it may download the weights); a later request up to 120 s.
- Missing libraries, a worker crash or a timeout produce no ticket and one log line; the row is absent that day and nothing else is affected. `close()` re-arms the model, so a transient failure on one game does not drop the row for the next.
- `Predictor.py` closes the worker after each game's row.

Install, once per machine:

```
python3 -m pip install --target /root/.foundation-libs --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install --target /root/.foundation-libs chronos-forecasting
python3 -m pip install --target /root/.foundation-libs --no-deps timesfm==3.0.2
```

The second command pulls the CUDA build's dependencies although torch is the CPU wheel; `rm -rf /root/.foundation-libs/nvidia /root/.foundation-libs/triton` reclaims that space without affecting forecasting.

### Live day only

The two foundation rows are built for the live day only (`skipRows == 0`), never inside a history rebuild: the rebuild runs its statistical step in a pool of spawned processes and a worker in each would multiply the memory use. A rebuilt day therefore has no `Chronos Model` or `TimesFM Model` row. The baseline row is built on rebuilt days too.

### The baseline

On a sorted set game the per-position series are order statistics: position 1 is the minimum of the draw, so its distribution is predictable without the draw being predictable. `OrderStatistics Baseline` is therefore the row both foundation models have to beat.

- Set games: the historical mode of each position.
- Positional games: each column's own mode with repeats kept in their slots, ties to the lowest value (`Baselines.positional_column_frequency_ticket`).

### As a meta-learner feature

- `ModelFactory.BASE_MODEL_NAMES` includes `Chronos Model` and `TimesFM Model`. The Chronos column is on by default (`useChronosFeature`), the TimesFM column off (`useTimesFmFeature`); either exists only when the libraries are installed.
- The Backtester forks its worker pool, and a forked child must neither use the parent's worker nor start its own. `ModelFactory.prepare_foundation_scores` forecasts every day a backtest will ask for up front in the parent, closes the worker and switches the model to cache-only; a day nobody precomputed returns no signal.
- A foundation model is recognised by its `IS_FOUNDATION_MODEL` attribute, and `_start_worker` refuses when the model is used in a process other than the one that built it.
- `foundationContext` is part of the score-table cache fingerprint.
- Self-check: `python3 src/FoundationModel.py <game>` precomputes, forks children and asserts they served from the cache and started no worker.

## Council rows (`CouncilRows.py`, `src/CouncilTicket.py`)

Rows whose ticket comes from the local LLMs of the Council (`src/llmCouncil/`), on the seven lottery games. They are scored and ranked like every other row.

Two kinds, each as one row per Council member and one row named `combined`:

- **`Council (record): …`** The LLM reads the last 8 draws (5 for Keno), and for the ten best-ranked rows that play today: how many numbers each had right on those draws and what it plays for the next draw. It is a meta-learner done by an LLM; its comparison is `WeightedEnsemble Model` and `RL Ticket Model`. The risk to read for is that it follows whichever row did well lately.
- **`Council (draws): …`** The LLM reads only the last 30 draws (10 for Keno). On draws that test fair this is a biased random generator: it serves as a control, and its favourite numbers show as repetition.

The `combined` row is the Council's head model choosing one ticket from a tally of what the members chose (which numbers, by how many members). It is made when at least two members gave a ticket.

### A valid ticket or none

The answer is one line, `TICKET:` followed by the numbers, the special numbers after a plus sign. `CouncilTicket.read_answer` accepts it only as a ticket of the game: the count, the range, no number twice in a set game. An answer that is not a ticket is asked again, at most three tries in all; the next try names what was wrong with the last and is asked at a temperature of at least 0.5. After three tries the row is absent that day, as a foundation row is.

One leniency: in a set game an answer that lists more main numbers than asked is cut to the first ones, since the question asks for the most likely first. Special numbers and the digits of a positional game are never cut.

Lotto is asked for nine numbers: the first six are the ticket, the other three its multi-pick. Keno is asked for twenty numbers in order; their first 5 to 10 are the tickets, in the sizes the day's other rows play.

### What is recorded

Each row carries `council`: the kind, the member, the model the endpoint reports, the temperature, the seed, the number of tries and a hash of the prompt. A member without a seed in the Council's config gets one fixed by game, day, row and try. The prompts and every answer, refused ones included, are kept in `data/council/<game>/<day>.json`.

### Where they do not count

The Council rows are no input of another row: `SubsetEnsemble Model`'s member selection (`HyperoptEnsemble.py`), `RL Ticket Model` and its tuner, and the combination search leave them out. They are made after the day's other rows, so `WeightedEnsemble Model` never sees them.

### Forward only

They are made for the live day by a job of its own (see [operations](operations.md)) and never inside a history rebuild. A rebuild of a day that holds Council rows, `-r` included, puts them back unchanged.

## How predictions are combined

After the individual rows, several rows are built from other rows. They are additive: none of them changes the rows it reads.

### Order of a day

1. The deep-learning child process (light rows, the two LSTM rows, the heavy rows, the deep-learning memorise rows).
2. The statistical step: statistical models and their memorise rows, the foundation rows, the baseline, the market rows, the meta-learner rows.
3. The boosting rows, each followed by its memorise row.
4. The variant rows are taken aside (`holdVariantRows`).
5. `numberFrequency`, then `WeightedEnsemble Model`, `SubsetEnsemble Model`, and `RL Ticket Model` (or `RL Position Model`).
6. The variant rows are put back and the sampled rows are added.
7. Rows that ran on a manual override are labelled, and the day file is written with a `madeAt` UTC stamp.

### WeightedEnsemble Model

A weighted vote over the day's rows.

- **Weights**: each row's score in `bestParams_<game>.json["modelScores"]`, min-max scaled to `[1, 2]`, so a poorly scoring row is outweighed and never zeroed. A row without a score gets weight 1.
- **Set games**: `Helpers.count_number_frequencies_from_new_prediction` counts every number suggested by any row or subset, weighted; `addWeightedEnsemblePrediction` turns the vote into a ticket. Main numbers and special columns are voted on separately and concatenated, so a special slot cannot be crowded out by the main range.
- **Keno**: the row also produces the `use_5` .. `use_10` sub-selections, drawn from the vote's scores (`weightedEnsembleSubsetMode`, default `softmax`; `weightedEnsembleSubsetTemperature`, default 0.5).
- The weighted frequency is stored as the day's `numberFrequency`.

On **positional games** the row is a vote per slot (`Helpers.build_positional_vote_predictions`, `VoteRows.positional_vote`):

- The voters are the rows that own a per-position model, `POSITIONAL_VOTE_DEFAULT_ROWS` in `src/VoteRows.py`: `Markov Model`, `MarkovMonteCarlo Model`, `PoissonMonteCarlo Model`, `LaplaceMonteCarlo Model`, `XGBoost Model`, `LightGBM Model`, `CatBoost Model`. Aggregate rows and the deep-learning rows do not vote, so no signal is counted twice.
- Each voter votes for the value it put in each slot, weighted as above. The slot's value is the argmax; duplicates across slots and drawn order are kept.
- Ties on a lottery game go to the lowest digit.
- The row carries `positionConfidence` (the winning vote share per slot) and the day file a `positionFrequency` (one histogram per slot).
- Joker+ gets the vote rows only with `"useJokerplusEnsemble": true` in its parameters (see [scoring](scoring.md)).

The **market games** use the same per-slot vote over bins, with two rules of their own wherever a market vote is taken or scored:

- A tied slot goes to the tied bin nearest the middle of the bins, then the lower one, so a tie does not lean towards the most bearish call.
- `XGBoost Model` and `LightGBM Model` count as one voter, half the weight each when both vote (`MARKET_SHARED_VOTE_ROWS`).

Self-check: `python3 -m src.VoteRows`.

### SubsetEnsemble Model

The same vote, restricted to a searched subset of rows. `HyperoptEnsemble.py` selects the rows (see [tuning](tuning.md)) and writes `subsetEnsembleModels` and `subsetEnsembleWeighted` (default true) to `bestParams_<game>.json`; `addSubsetEnsemblePrediction` serves it through the same recipe (`Helpers.build_vote_ensemble_predictions`), so the served ticket is what the tuner scored.

- Not served until a selection of at least two rows exists.
- Skipped on a day one of the selected rows did not run: a vote among the rest would be a different ensemble under this row's name.
- A weighted selection is skipped on a day one member outweighs the others together (`VoteRows.effective_members`: its weight strictly above the sum of theirs), because the vote would then be that member's ticket. Exact equality is not a copy, and a flat vote cannot collapse. Keno is exempt: the sub-selections that are played are drawn from the vote's own scores.
- On positional games the per-slot vote runs over the selected rows, whichever rows the tuner selected.
- On the market games the vote is weighted by `subsetEnsembleWeights` once the tuner has written them (`modelScores` until then), and a selection of only XGBoost and LightGBM is one voter and skipped.
- Keno sub-selections use `subsetEnsembleSubsetMode` / `subsetEnsembleSubsetTemperature`, falling back to the weighted-ensemble keys.

### Meta-learner rows

A stacking meta-learner predicts `P(number is drawn)` from each base model's own per-number score instead of a hand-weighted vote. Training is described on [tuning](tuning.md); the quantum rows on [research](research.md). `Predictor.py` serves a row when its artifact exists under `data/models/<game>/`, and skips the row otherwise.

| Row | Artifact |
|---|---|
| `MetaLearner Model` | `meta_learner.joblib` |
| `MetaLearnerV2 Model` | `meta_learner_v2.joblib` |
| `QuantumMetaLearner Model` | `quantum_meta_learner.joblib` |
| `QuantumVQC Model` | `quantum_vqc_meta_learner.joblib` |
| `ClassicalSVM Model` | `classical_svm_meta_learner.joblib` |

- **Base models** (`ModelFactory.BASE_MODEL_NAMES`): Markov, MarkovMonteCarlo, MarkovBayesian, MarkovBayesianEnhanced, PoissonMonteCarlo, PoissonMarkov, LaplaceMonteCarlo, XGBoost, Chronos, TimesFM (opt-in) and, on the market games, the four market rows. `HybridStatisticalModel` is excluded because it is itself a vote over several of these. On positional games MarkovBayesian, MarkovBayesianEnhanced and PoissonMarkov are left out as well.
- **Features** come from the artifact's own `feature_names`, so an artifact never asks for a column it was not trained with. A feature with no model instance is served as 0.0.
- Base-model scores are computed once per distinct feature set and shared by the rows that use it.
- **Set games**: the top numbers by predicted probability. A second, separately trained model in the same artifact ranks the special column(s).
- **Keno**: sub-selections from the predicted scores, with `<row>SubsetMode` / `<row>SubsetTemperature` keys (`metaLearner…`, `metaLearnerV2…`, `quantumMetaLearner…`, `quantumVqc…`, `classicalSvm…`; defaults `softmax` and 0.5).
- **Positional games**: a positional artifact holds one model per slot and serves the per-slot argmax in drawn order, ties to the lowest value, duplicates kept; on Joker+ the sign comes from the artifact's special model. A non-positional artifact on a positional game is refused and the row skipped.

### Manual overrides

`Predictor.py` reads its parameters through `readServedParams`, which applies the manual overrides of `src/ParamOverrides.py` on top of `bestParams_<game>.json` and logs each one. A row that ran on an overridden key is marked `"manual": [keys]` in the day file. The Parameters page that sets overrides is on [tuning](tuning.md).

## Variant rows: settle, memorise, sampled

Every model is tracked in up to three forms (`src/RowVariants.py`). The name separator is ` · `.

- **settle**: the row under its own name, the ticket of its highest scores.
- **`<row> · Memorise`**: the model made to fit the recent draws as closely as it can.
- **`<row> · Sampled`**: a ticket drawn from the row's own scores instead of their top.

Variant rows carry `variant` and `parent` fields. A variant never gets a sampled twin.

### Memorise rows

`useMemoriseRows` (default true) switches them off for a game. They are never built on the market games.

- **Deep learning**: `LSTM Base Model` and the six prefixed rows (not `LSTM Smoothed Model`). A fresh shallow network (64 units, one layer; the LSTM with one LSTM and one bidirectional layer) with no dropout, L2 or label smoothing, new weights every day in `data/models/<folder>_memorise`, trained without early stopping or the learning-rate schedule until `DL_MEMORISE_SECONDS` (120 s) run out. They train after every settle row, so a deadline kill costs a memorise row first.
- **Statistical**: Markov, MarkovMonteCarlo, MarkovBayesian, MarkovBayesianEnhanced, PoissonMonteCarlo, PoissonMarkov and LaplaceMonteCarlo, when the settle row exists. The same model on the last `memoriseWindowDraws` draws only (default 30, `Helpers.recent_window`), alpha 1, minimum occurrences 1, no smoothing.
- **Boosting**: a fresh model with deep unregularised trees (depth 8, 300 rounds, learning rate 0.3, no minimum child weight, no L2, no subsampling); nothing is saved. CatBoost's memorise rows are built only with `memoriseCatBoost` true (default false).

### Sampled rows

Built on the five probability-order games only (vikinglotto, euromillions, eurodreams, keno, lotto), for every row that stored scores; a row without scores, such as `HybridStatisticalModel`, gets none.

- The scores are shifted to be non-negative and a ticket is drawn without replacement with `p_i ∝ s_i^(1/T)`.
- `T` is the row's temperature in `sampleTemperatures` (a map of row name to temperature), default 1. `T` at or below 0.01 gives the settle ticket; a high `T` approaches uniform.
- The draw is seeded by game, draw date and row name, so a saved ticket is reproducible.
- Keno sub-selections are the first numbers of the sampled ticket; lotto's extra numbers are drawn from the remaining numbers.
- The row stores `sampleTemperature` and the `scores` it was drawn from.

Self-check: `python3 -m src.RowVariants`.

### Where variants do not count

`Predictor.holdVariantRows` takes the variant rows aside while the vote chart (`numberFrequency`), the vote rows, `SubsetEnsemble Model` and the RL row read the day; the RL row's training history skips them too. In `modelPerformance.json` they are ranked after every other row and marked `variant`, and they stay out of the per-row chance test's family and the combination search (see [scoring](scoring.md)).

## History rebuild & model-weight reuse (operational behavior)

### Command-line flags of `Predictor.py`

| Flag | Default | Meaning |
|---|---|---|
| `-g/--games` | all eleven games | comma-separated games to run |
| `-d/--days` | 31 | how many recent draws the gap check and a rebuild cover |
| `-r/--rebuild_history` | false | overwrite the last `-d` draws |
| `-a/--ai` | false | run the heavy deep-learning rows |
| `-b/--boost` | true | run the boosting rows |
| `--dl-model-seconds` | 240 | training budget per deep-learning model; 0 = no budget |
| `--dl-timeout` | 0 | hard deadline of the deep-learning child in seconds; 0 = automatic |
| `-s/--save` | true | push the results to git |

How runs are started and scheduled, and the process lock, are on [operations](operations.md).

### Gap recovery

Gap recovery is automatic and does not overwrite complete files.

- The run anchors on the newest complete day file (complete = it has `labels`) and builds every draw after it, however large the gap. It scans the game's actual draw dates, so games with a few draws a week are handled.
- It also fills holes within the last `-d` draws. A half-built file counts as a hole and is repaired.
- A game without any day file gets the last `-d` draws; a new market game gets deep-learning rows on its newest day only.
- `update_matching_numbers` re-links the prediction-versus-result chain across old and new files afterwards.
- `-r/--rebuild_history` regenerates and overwrites the last `-d` draws, existing files included.

A rebuild runs in two steps: the statistical step for all days in a pool of up to eight spawned processes, then deep learning, boosting and the combining rows day by day in one process. Rows that learn from stored days (the RL rows) only read days strictly before the day being rebuilt.

### The daily deep-learning time budget

Deep learning is opt-in per run, time-boxed and crash-isolated.

- `-a true` enables the heavy rows (the two LSTM rows, TCN and the two unified rows). The three light rows and their memorise rows run without it.
- **Per-model budget** (`--dl-model-seconds`, `src/TimeBudget.py`): at the budget, training stops the way an early stop does. Keras still runs the epoch's validation, `EarlyStopping` puts the best epoch's weights back and `ModelCheckpoint` has saved the best epoch, so the model still predicts; only training quality degrades. The check runs after every batch, so a model can overrun by one batch, and the first batch of a freshly compiled graph is the expensive one.
- **One child process per day**: all deep-learning training of a day runs in a one-shot spawned child. Its memory is released when it exits, and a crash of the child (an out-of-memory kill, for instance) costs that day only the deep-learning rows it had not finished.
- **Order in the child**: light rows, then the two LSTM rows, then TCN and the unified rows, then the memorise rows.
- **Checkpoints**: the child writes its finished rows after every model to `data/hyperOptCache/dl_partial_<game>_<pid>_<ns>.json`. A marker ties the file to its game and day, so a leftover from a crashed run is never served; stale files are removed at the start.
- **Hard deadline** (`--dl-timeout`): at the deadline the child is killed and the rows it had finished are kept (`Recovered N deep learning row(s)` in the log).

The automatic deadline is

```
settle models x (--dl-model-seconds + 60) + memorise models x (120 + 60) + 120
```

The 60 s per model cover data load, graph compilation, prediction and save, which run outside the training budget; the last 120 s are the child's start-up. At the defaults a lottery game with `-a true` has eight settle and seven memorise models: 8 x 300 + 7 x 180 + 120 = 3780 s (63 minutes). `--dl-model-seconds 0` disables the automatic deadline as well.

Each model's total time for the day is logged (`LSTM Base Model: 187s`: load, training, prediction and save together); that is the number to tune the budget on.

## Advanced Architectural & Security Research

These rows run in the daily pipeline as tracked prediction rows or monitoring layers, and are scored and ranked like every other row.

### Structural & graph-based analysis: `GNN Model` (`src/GNN.py`)

Numbers are nodes in a co-occurrence graph built from the training history with exponential recency weighting (`gnn_decay`). Stacked graph-convolution layers (a GCN written in Keras, no extra dependency) learn per-number embeddings, and a window-conditioned readout turns them into the per-position softmax prediction. Keys `gnn_*`.

### Long-range temporal context: `Transformer Model` (`src/TransformerModel.py`)

Sinusoidal positional encoding and pre-LN self-attention encoder blocks over a longer window (default 30 draws). Attention can weight any draw in the window regardless of distance. Same output head, NaN and checkpoint handling and fingerprinted weights as the other deep-learning rows. Keys `transformer_*`.

### Ticket construction: `RL Ticket Model` (`src/RLTicketModel.py`)

The row does not predict numbers; it learns how to assemble a ticket from the day's other rows.

- **Policy**: a linear score per candidate number, tickets sampled without replacement (Plackett-Luce), trained with REINFORCE and a mean-reward baseline in pure numpy. Pick3 gets one softmax per position, so digits may repeat.
- **Features** per number: `voteShare` and `meanRank` over the day's other rows, `drawFrequency`, `drawsSinceSeen`, and a bias; z-scored within the day.
- **Reward**: the real payout on pick3 and keno (`Helpers.pick3_ticket_profit`, `keno_ticket_profit`), the main-ticket hit count elsewhere.
- **Training data**: the stored day files (`currentPrediction` rows and the real draw). During a rebuild only days strictly before the rebuilt day are used.
- It runs after the vote rows, warm-starts from the policy in `data/models/rl_model`, and its training is wall-clock capped.
- The policy builds the main numbers only. The special column(s) are the values the day's other rows put there most often (`RLTicketModel.voted_specials`, one vote per row, ties to the smaller value). A ticket without its full special column is scored on hits only (see [scoring](scoring.md)).
- Not built for Joker+. On the market games `RL Position Model` takes its place.

| Key | Default |
|---|---|
| `useRlTicket` | true |
| `rlTicketLearningRate` | 0.05 |
| `rlTicketEpochs` | 30 |
| `rlTicketSamplesPerDay` | 32 |
| `rlTicketTrainDays` | 120 |
| `rlTicketMaxTrainSeconds` | 60 |

The keys are tuned by `HyperoptRLTicket.py`; see [tuning](tuning.md).

### Position sizing: `RL Position Model` (`src/RLPositionModel.py`)

The market games' RL row learns how much to put on a call, not which bin to call.

- Its ticket is the `WeightedEnsemble Model`'s ticket (the same voters and vote rules). Its own contribution is `positions`: a size per instrument, from 0 (sit out) through 1 (the paper stake) up to 2.
- **Features**, in bin space: `upShare`, `consensus`, `garch`, `regime`, `vote`, `yesterday`, `recentAbs`, bias.
- **Policy**: a linear softmax over the sizes, trained with REINFORCE on the stored day files. The reward is a concave utility of the paper money, `money − riskAversion × money²`, so an intermediate size can be the right answer for an uncertain call.
- Without a trainable history and without a stored policy it falls back to size 1 where the vote is up and 0 where it is not.
- Weights in `data/models/rl_position`. Self-check: `python3 -m src.RLPositionModel`.

| Key | Default |
|---|---|
| `useRlPosition` | true |
| `rlPositionLearningRate` | 0.05 |
| `rlPositionEpochs` | 30 |
| `rlPositionSamplesPerDay` | 16 |
| `rlPositionTrainDays` | 120 |
| `rlPositionRiskAversion` | 0.1 |
| `rlPositionMaxTrainSeconds` | 60 |
| `rlPositionSitOut` | false (true serves the row at size 0) |

How positions are settled and booked is on [markets](markets.md).

### Anomaly detection: `Autoencoder Model` (`src/AutoencoderAnomaly.py`)

A narrow-bottleneck conditional autoencoder that is both a tracked prediction row and an integrity monitor.

- After each training run it computes the reconstruction negative log-likelihood of every recent real draw and a rolling z-score.
- A strongly negative z means the real draw became easy to reconstruct (a predictability spike). That is the alert condition: the lowest z of the last 30 draws below −3.
- The day file stores it as `anomalyWatch`: `model`, `latest_z`, `min_z_recent`, `alert` and the last 120 `scores`.
- Its label smoothing defaults to 0, so the likelihood is not biased. Keys `autoencoder_*`.

### Entropy & divergence analysis

`Helpers.generate_model_performance_report` computes, per game over the last 60 scored draws: KL(recent ‖ full history) for drift, KL(recent ‖ uniform) and normalised entropy for the distance from a fair draw, and per model the KL between predicted and real numbers. Entropy near 1 and KL near 0 mean the process looks fair and stationary. Sustained movement is a signal for investigation, **not** proof of manipulation: rule changes, data artefacts and small windows all move these numbers. How the figures are shown and read is on [research](research.md).
