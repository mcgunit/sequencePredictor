# Tuning: hyperopt, backtesting and the meta-learner

Six Optuna tuners search each model's parameters per game and write them to `bestParams_<game>.json` only through a champion/challenger gate. This page describes the tuners and their scores, the gate and its margins, the variety rule, the weekly tuning chain, the admin pages that start tuning runs and backtests by hand (Hyperopts, Parameters, Backtests), manual overrides, and the meta-learner trainer. The models themselves are in [models](models.md), the scores users see in [scoring](scoring.md), the lockbox and the controls in [research](research.md), the machines the runs use in [operations](operations.md).

## Overview

| Tuner | Tunes | Evaluates on | Default window | Default trials |
|---|---|---|---|---|
| `HyperoptStatistics.py` | the statistical rows, GARCH and Regime HMM, the keno sub-selection keys | `src/Backtester.py` walk-forward | 90 draws | 15 |
| `HyperoptBoost.py` | the six boosting rows | `src/Backtester.py` walk-forward | 90 draws | 15 |
| `HyperoptRLTicket.py` | RL Ticket (lotteries), RL Position (markets) | the stored day files | 40 days | 20 |
| `HyperoptEnsemble.py` | the members of the `SubsetEnsemble Model` | the stored day files | 120 days | 16384 evaluations |
| `HyperoptQuantum.py` | the two quantum meta-learners and their classical control | a holdout of the backtest score table | 300 days | 15 |
| `HyperoptDeepLearning.py` | the seven deep-learning model types | retrains per rebuilt draw | 3 draws | 15 |

Common to all six:

- One Optuna study per game and row, stored in `db.sqlite3` in the working directory and opened with `load_if_exists` (see *What a rerun continues*).
- A run's best trial is the challenger of the gate (`src/TuningGate.py`); nothing writes a study's all-time best.
- Every decision is recorded under `bestParams_<game>.json["tuningGate"][<row>]`.
- The lockbox days are never scored (see [research](research.md)); every gate record carries `lockbox_days_withheld` next to `window_days`.
- Every tuner holds `process.lock` while it runs.
- `--result-file <path>` also writes each study's progress and verdict as JSON (`src/TuningResult.py`); the Hyperopts page runs the tuners this way.
- `--variety-line <share>` sets the variety rule's line for the run (0.5-1.0 or 50-100; default the setting in `config/tuning.json`, else 0.95).

## The objective (`src/TuningScore.py`)

A trial's value is a lower confidence bound over the scored days: mean − 1 × std/√days. A single lucky payout inside the window therefore cannot decide a study. What is bounded depends on the game:

- **Keno:** the per-day profit per bet, every bet's profit capped at the lucky-strike threshold (20 EUR net) and measured against the fair capped value of a ticket of its size, so a fair ticket scores 0 in the mean. Trials and references bet the sub-selection sizes the game serves (the global `use_<n>` flags).
- **Pick3 and Joker+:** the per-day slot hits (Pick3's digits in the right slot, Joker+'s leading and trailing runs) plus 0.001 × the capped-profit bound as a tie-breaker.
- **Games without a payout table:** the hits bound.
- **GARCH and Regime HMM:** the bound of the per-day mean log-score of the probability the row gave the bin that happened (`score_rows(..., proper=True)`), floored at `1e-4`; a uniform forecast scores log(1/10) = −2.303. A hit rate is not a proper score, a near-one-hot forecast would game it.

The score's standard error over the days (`score_se`) is the unit of the gate's margin. The raw profit per bet, the number of lucky strikes and the days travel with every trial as the user attribute `tuning`. The pruning callbacks score the finished days with the same function, so partial and final values agree.

A robust objective ranks candidates; it does not make them good. A negative bound on keno or Pick3 is what a fair ticket scores.

## The gate (`src/TuningGate.py`)

`challenge()` compares three parameter sets on this run's window with this run's objective:

- **the challenger:** this run's best completed trial with a finite score that is not input-blind;
- **the incumbent:** the parameters in `bestParams_<game>.json`, re-scored;
- **the default:** what `Predictor.py` serves when the keys are missing, re-scored.

The references are scored by the tuner's own objective driven by an `optuna.trial.FixedTrial`, each in a forked child (`HyperoptRunner.run_isolated`), so they go through exactly the code that scored the trials.

Decisions:

- **replaced:** the challenger beats the incumbent and the default by the margin. The margin is the larger of `--gate-margin` (default 0) and k standard errors of the challenger's own score (`--gate-se-margin`).
- **reverted:** the default is written back. This happens when the challenger does not beat the default by the margin and either the default beats the incumbent by the revert margin, the challenger beats the incumbent by the margin, or the incumbent cannot be scored or is input-blind. An exact tie of the challenger with the default reverts.
- **kept:** everything else.

Rules that hold in every tuner:

- A row without tuned keys has the default as its incumbent.
- A reference that cannot be scored (timeout, error) cannot block, and the record says so.
- Fail closed: with k > 0 a challenger without a standard error is never written (`no standard error - kept`). The default is still served back when it beats the incumbent by the revert margin; that comparison does not involve the challenger.
- `--no-gate` writes the run's best trial without the comparison.
- `modelScores[<row>]` holds the score of what is served after the decision (the challenger on a replace, the re-scored incumbent on a keep). A row the gate could not score this run loses its entry. `Predictor.py` maps these scores to vote weights.

One standard error is a bar on a best-of-n selection, not a calibrated test.

### Margins per tuner

| Tuner | Challenger margin k (`--gate-se-margin`) | Revert margin |
|---|---|---|
| Statistical (all strategies, GARCH, Regime HMM, `KenoSubsetTuning`) | 1 | 0 (`--gate-revert-margin`) |
| Boosting | 1 | the challenger's margin |
| RL Ticket, RL Position | 1 | 0 (`--gate-revert-margin`) |
| Ensemble | 1 | 0 (`--gate-revert-margin`) |
| Quantum | 2.5 | the challenger's margin |
| Deep learning | 1 | 0 |

A margin-free revert is used where the default is a fixed, unselected reference. The quantum tuner keeps the margin on the revert because both sides are single fits whose score moves with the seed.

### The gate record

`tuningGate[<row>]` holds: `date`, `tuner`, `window_days`, `lockbox_days_withheld`, `margin_se` (the k asked for), `margin` and `revert_margin` (the margins used), `decision`, `reason`, `trial`, `study_best` (the study's all-time best, for context), the `challenger`, `incumbent` and `default` scores each with its diagnostics (`*_tuning`), and `variety` (the rule, the number of input-blind trials and references). Tuners add their own keys (below).

On the admin pages the Tuning gates drawer shows each row's margin as "margin X = k SE", gives a margin-free revert its own line and shows the RL Position row's *sits out* chip with its reason (`gates.js` `describe()`: `marginK`, `revertMargin`, `sitOut`, `sitOutReason`).

## The variety rule

A usable parameter set predicts new sequences when its input changes; recurring numbers are fine. A candidate that gives the same ticket at or above the line is *input-blind*:

- it is never the challenger;
- it is never served back as incumbent or default;
- it is not a score the challenger must beat.

When every trial of a run is input-blind, the served values stay.

Two measures:

- **Draws** (statistical, boosting, ensemble, quantum, RL Ticket): the share of the scored draws that got the row's most common ticket (`TuningScore.score_rows(variety_line=)`, `TuningResult.variety_of_tickets`). Fewer than two draws are not judged.
- **Windows** (deep learning): per rebuilt draw, the share of validation windows that got the most common ticket; the candidate is input-blind when that share is at the line on at least half of its draws, or when it gives the same ticket on the line's share of its rebuilt draws.

The market rows are not judged: GARCH, Regime HMM, RL Position and every strategy on the market games. A market call may rightly repeat, and RL Position sizes positions.

Every candidate's record carries the share, the number of different tickets and the flag; the gate record states the rule with its line. Variety is not skill.

### The setting

The line is a setting, `varietyLine` in `config/tuning.json` (0.95 when missing, allowed 0.5-1.0), read by every tuner through `TuningResult.load_variety_line`. The *Variety rule* card on the Hyperopts page writes it (`POST /api/admin/hyperopts/settings`, audit `hyperopt.settings`). The weekly chain uses the setting; a run from the Hyperopts page passes its own value (`--variety-line`, the setting by default).

## The statistical tuner (`HyperoptStatistics.py`)

Tunes each statistical row per game through `src/Backtester.py`: for every draw of the window the model is fitted only on the draws before it and compared with the real result.

```bash
python3 HyperoptStatistics.py -g lotto,keno -s Markov,PoissonMonteCarlo -d 90 -t 15
```

- Flags: `-g` games, `-s` strategies, `-d` window (90), `-t` trials (15), `--trial-timeout` (1200 s), `--gate-margin`, `--gate-se-margin` (1), `--gate-revert-margin` (0), `--no-gate`, `--save` (commit and push the result), `--result-file`, `--variety-line`.
- Strategies: `Markov`, `MarkovMonteCarlo`, `MarkovBayesian`, `MarkovBayesianEnhanced`, `PoissonMonteCarlo`, `PoissonMarkov`, `LaPlaceMonteCarlo`, `HybridStatistical`, `Garch`, `RegimeHmm`, `KenoSubsetTuning`.
- Studies are named `<game>_<strategy>`.
- The positional games (Pick3, Joker+, the market games) skip `MarkovBayesian`, `MarkovBayesianEnhanced`, `PoissonMarkov` and `HybridStatistical`: these pool number frequencies over all positions.
- One trial runs at a time; the Backtester spreads a trial's days over the cores. A trial over `--trial-timeout` is recorded as pruned.
- The defaults the gate compares with are `HyperoptStatistics.SERVED_DEFAULTS`.
- A game whose whole window falls inside the lockbox is skipped.

### The Markov transition mode

`markovTransitionMode` is a tuned knob of the `Markov Model`: `column` (the default: one table per column, from the number in a column to the number in the same column of the next draw) or `within` (one table over the game, from each number to the next inside the same sorted draw, read as a chain from a start state, `Markov.WITHIN_START`). The tuner offers both for the set games and scores them on the same days; the positional games have only `column`. How each mode builds a ticket is in [models](models.md).

After a Markov study the tuner also writes `markovSortedPrediction`, `markovUsePairScoring` and, for the set games, `markovPairScoringWeight` 0.

### The market rows

`Garch` and `RegimeHmm` tune `src/MarketModels.py`'s rows (`garch*` / `regimeHmm*` keys) on the four market games (`crypto`, `shares`, `cryptoweek`, `sharesweek`) with the log-score objective above.

- The three Regime HMM rows share the `regimeHmm*` keys; the defaults are `MarketModels.MARKET_DEFAULTS`.
- The Regime HMM gate record adds `features`, the feature set the rows run on after the decision.
- The served score stays out of `modelScores`: a log-score is not a vote weight.
- The models and the search space are in [markets](markets.md).

### The keno sub-selection step

`KenoSubsetTuning` tunes how the vote rows and the meta-learner rows (the quantum ones included) slice their keno sub-selections (mode and temperature keys). It is keno-only and not part of a plain run (`DEFAULT_STRATEGIES`); `-s KenoSubsetTuning` names it:

```bash
python3 HyperoptStatistics.py -g keno -s KenoSubsetTuning
```

- It is the last step of the weekly chain, after the meta-learner retrain: its day table scores the keno meta-learner artifacts and the weighted vote as they are on disk.
- The day table is built once in the parent process and inherited by the trial processes.
- Its boosting row refits once per block of 7 days (`KENO_DAY_TABLE_REFIT_EVERY`).
- The build is bounded at three trial budgets; when it does not finish, the step is skipped for the run and nothing is written.
- The tuners prune stale per-model keno flags (`lightgbm_use_5`, ...) from the file; nothing reads them.

## The boosting tuner (`HyperoptBoost.py`)

The same structure as the statistical tuner, for six strategies: `XGBoost`, `XGBoostMultiLabel`, `LightGBM`, `LightGBMMultiLabel`, `CatBoost`, `CatBoostMultiLabel`, on the seven lottery games. The MultiLabel strategies skip Pick3 and Joker+ (set membership cannot express digit order or repeats).

- Flags: `-g`, `-s`, `-d` (90), `-t` (15), `--trial-timeout` (1200 s), `--refit-every`, `--prune-percentile` (75), `--catboost-border-count` (254), `--parallel-trials` (0 = auto), `--memory-reserve-gb` (2), `--gate-margin`, `--gate-se-margin` (1), `--no-gate`, `--save`, `--result-file`, `--variety-line`.
- Studies are named `<game>_<strategy>`; the gate's defaults come from `src/BoostingBase.py`.
- All six share one search space with prefixed keys (`xgBoost*`, `xgBoostMl*`, `lightGbm*`, `lightGbmMl*`, `catBoost*`, `catBoostMl*`), so the libraries are compared on equal terms.
- CatBoost's tree depth is searched over 1-7 (its symmetric trees cost 2^depth); XGBoost and LightGBM over 1-10.

### Time and memory guards

- **Trial budget:** a trial over `--trial-timeout` is stopped by the Backtester and recorded as pruned with `timeout`.
- **Cost gate:** every trial is stamped with a fit-cost proxy (estimators × window, × 2^depth for CatBoost, × (1 + 0.1 × depth) otherwise). A configuration at least as expensive as one that already hit the budget under the same budget tag (machine, window, cadence, budget) is skipped as a predicted timeout and does not count toward `--trials`. The gate's references are never skipped.
- **Refit cadence** (`--refit-every`): one fit per block of N consecutive days, fitted on the data before the block's oldest day, so never leaky. Default: the window over five blocks, never below 7 (18 at 90 days). The cadence is the same for every trial and reference, so the ranking is unaffected; production refits daily. `1` refits every day.
- **Pruning** (`--prune-percentile`, 0 disables): the tuner's `CheckpointPruner` judges a trial only at half and at three quarters of the scored days, on its partial score at that day, against the completed trials of this run, once five of them exist. It stops the bottom quartile. Pruned trials are recorded as pruned, never as bad scores.
- **Threads:** XGBoost and LightGBM fit single-threaded; CatBoost gets the cores its workers leave free, at most 8. BLAS and OpenMP pools are pinned to one thread per process.
- **Parallel trials:** XGBoost and LightGBM studies run several trials at a time (auto: cores divided by the workers per trial); CatBoost runs one trial at a time. A further trial launches only while `--memory-reserve-gb` stays available after it; under 1 GB available the youngest trial is stopped and recorded as pruned.

`--catboost-border-count` is a run-level setting, written as `catBoostBorderCount` / `catBoostMlBorderCount` so `Predictor.py` serves the same value.

## The RL tuner (`HyperoptRLTicket.py`)

The game decides the row: RL Ticket on the lottery games (euromillions, lotto, eurodreams, keno, pick3, vikinglotto), RL Position on the four market games. Both walk forward over the game's stored day files (`data/database/<game>`): each evaluated day is re-predicted with training limited to strictly earlier days. A game needs at least 10 scoreable days.

- Flags: `-g`, `-d` (40), `-t` (20), `--parallel-trials` (0 = auto, one per core within the memory gate), `-s` (save), `--gate-margin`, `--gate-se-margin` (1), `--gate-revert-margin` (0), `--no-gate`, `--result-file`, `--variety-line`.
- Per-trial policies live in `data/hyperOptCache/rl_model`; the live policy in `data/models/rl_model` is never touched.
- The lockbox days stay in the walk-forward (the policy trains through them as in production) but are never scored.

### RL Ticket

- Knobs: `rlTicketLearningRate`, `rlTicketEpochs`, `rlTicketSamplesPerDay`, `rlTicketTrainDays`; defaults `RL_TICKET_DEFAULTS`.
- Score: the shared lower bound. Keno's served sub-selections as capped bets against a fair ticket of their size, Pick3 by slot hits with the capped profit as tie-breaker, main-ticket hits elsewhere.
- Studies `<game>-rl_ticket_gated`; record `tuningGate["RL Ticket Model"]`.
- The variety rule applies (draws measure).

### RL Position

- Knobs: `rlPositionLearningRate`, `rlPositionEpochs`, `rlPositionSamplesPerDay`, `rlPositionTrainDays`, `rlPositionRiskAversion`; defaults `RL_POSITION_DEFAULTS`.
- Studies `<game>-rl_position`; record `tuningGate["RL Position Model"]`, with `plain_rule` (size 1 wherever the vote is up) for comparison.
- **Sit-out:** a policy that never bets scores exactly 0, a fourth reference the gate does not know. The sizing the gate serves is used only when its score (mean − 1 SE) is above max(`--gate-margin`, one standard error of its own), i.e. its mean money per position is more than two standard errors above 0 (`position_sit_out`, `SIT_OUT_SE_MARGIN`). A sizing without a standard error sits out. Otherwise `rlPositionSitOut` is written true; the tuned knobs are still written.
- The record adds `sit_out`, `sit_out_reason`, `sit_out_floor`, `sit_out_se_margin`, `served_mean_size` and `served_mean_nonzero_size`. There is no floor on the exposure.
- The score never enters `modelScores`.
- The model and its books are in [markets](markets.md).

## The ensemble tuner (`HyperoptEnsemble.py`)

Selects the member rows of the `SubsetEnsemble Model` on the stored day files, the only place where every tracked row exists side by side for the same draws. Games: the seven lottery games and the four market games. Joker+ is tuned only when `bestParams_jokerplus.json` sets `useJokerplusEnsemble`.

- Flags: `-g`, `-d` (120), `-t` / `--max-evaluations` (16384), `--record` (256), `--confidence-penalty` (1), `-s`, `--gate-margin`, `--gate-se-margin` (1), `--gate-revert-margin` (0), `--no-gate`, `--result-file`, `--variety-line`.
- **Candidates:** rows present on at least 80% of the window's scoreable days and on at least three of the last five. Never the two vote rows, the RL rows or a variant row.
- **A combination:** one include flag per candidate plus `subsetEnsembleWeighted` (vote with the weights, or count every member once).
- **Feasible:** at least two rows, members coexisting on at least 60% of the window (at least 10 days), and not a weighted vote that one member outweighs. Every scored combination then sits on nearly the same draws.
- **Score:** the lower bound of the per-day score (`--confidence-penalty` 0 gives the plain mean): profit per bet over the served keno sub-selections, main-ticket hits on the other set games, the real payout for Pick3 and Joker+, the fixed rule in bin space on the markets (`Helpers.market_ticket_profit`). The positional games vote per slot and add 0.01 × the share of slots the ticket got right as a tie-breaker.
- **Search:** exhaustive whenever 2^candidates × 2 fits the evaluation budget (twelve candidates at the default); beyond that a deterministic hill-climb over single flag toggles from the all-in vote (weighted and flat) and from the best pair. Every combination is evaluated once, in-process.
- **Study:** the evaluated combinations are written to `<game>-subset_ensemble` as completed trials, best first (`--record`), tagged with the run.

### Gate

The default reference is the plain vote: every candidate row, weighted. A served selection that cannot be served (a member no longer emitted, nothing written yet, a weighted vote one member outweighs) is scored as nothing, so the run writes the challenger or the plain vote.

Written to `bestParams_<game>.json`: `subsetEnsembleModels`, `subsetEnsembleWeighted`, `subsetEnsembleObjective`, `subsetEnsembleMean`, `subsetEnsembleTunedOn` (window, candidates, combinations, search kind, decision). Never the include flags, never the row's own score into `modelScores`.

The record `tuningGate["SubsetEnsemble Model"]` adds the run's candidates, `effectiveMembers` (the members that can still change the served ticket under the run's weights) and `weightedEqualsFlat`.

### Keno and the markets

- **Keno slicing:** the row's sub-selections are scored with its own `subsetEnsembleSubsetMode` / `subsetEnsembleSubsetTemperature`, else the weighted vote's keys. The tuner writes what it scored with as the row's own keys and records it as `kenoSlicing`, so the row is served with the slicing it was gated with. That pin is read past on the next run; keys set by hand are kept.
- **Markets:** XGBoost and LightGBM count as one voter. The weighted vote uses each candidate's own score on this window (`member_scores`, mapped to [1, 2] by `subset_weights`), written as `subsetEnsembleWeights` whatever the gate decides, with the raw values in the record's `memberScores`. The lotteries use `modelScores`.

The selection is made on the rows' stored history, so on the day it is written it is in-sample; its tracked results accrue out of sample from then on.

## The quantum tuner (`HyperoptQuantum.py`)

Tunes the two quantum meta-learners and their classical control (`quantumKernel_*`, `quantumVqc_*`, `classicalSvm_*` keys). The rows themselves are described in [research](research.md).

- Flags: `-g` (default: every game except the week games), `-v` / `--variants` (`quantum_kernel`, `quantum_vqc`, `classical_svm`; default all three), `-d` (300), `-t` (15), `--parallel-trials`, `--trial-timeout` (1800 s), `-s`, `--gate-margin`, `--gate-se-margin` (2.5), `--no-gate`, `--result-file`, `--variety-line`.
- Per game the backtest score table is collected once and cached in `data/hyperOptCache/meta_score_table_<game>.joblib` (positional games: `meta_position_table_<game>.joblib`). `TrainMetaLearner.py` reuses it when no draw and no base-model parameter changed.
- The table's days are split chronologically 75/25. Each trial fits the variant (scaler and PCA included) on the early days and is scored on the held-out days. At least 10 collected days are needed.
- **Set games:** mean per-day hits of the top-`draw_size` ranked numbers, the ticket that is played, plus 0.01 × AUC as a tie-breaker; the margin unit is the standard error of the per-day hits.
- **Pick3 and Joker+:** one classifier per slot; the argmax ticket is scored by the positional objective above (slot-hit bound plus 0.001 × capped-profit bound). Joker+ is scored on its digits only.
- **Market games:** the mean per day of the fixed rule's P&L in bin space plus 0.01 × top-1 accuracy.
- Studies: `<game>-quantum_kernel`, `<game>-quantum_vqc`, `<game>-classical_svm`; with the suffix `_slot_bound` for Pick3 and Joker+.
- Defaults: `QUANTUM_DEFAULTS`, with gamma 0.25 standing in for scikit-learn's `"scale"` in the control's reference (noted in the record as `default_note`).
- Records: `tuningGate["QuantumMetaLearner Model" | "QuantumVQC Model" | "ClassicalSVM Model"]`.
- The rows' keno sub-selection keys are tuned by the keno sub-selection step, not here.
- A malformed `lockbox.json` stops this tuner before any backtest; the lockbox days leave the table before the split.

## The deep-learning tuner (`HyperoptDeepLearning.py`)

Tunes seven model types per lottery game, each in its own study writing prefixed keys: `lstm_model`, `tcn_model`, `unified_lstm_tcn_model`, `unified_lstm_gru_tcn_model`, `transformer_model`, `gnn_model`, `autoencoder_model`. It is not part of the weekly chain; it is started by hand or from the Hyperopts page.

```bash
python3 HyperoptDeepLearning.py -g pick3 --models transformer_model,gnn_model,autoencoder_model
```

- Flags: `-g` (default the seven lottery games), `-m` / `--models` (default all), `-d` rebuilt draws per trial (3), `-t` trials per run (15), `--trial-timeout` (7200 s), `--train-seconds` (240; 0 = unlimited), `--prune-after-days` (2; 0 = off), `--gate` (true), `--gate-se-margin` (1), `--aim` (`variety` or `skill`), `--loss-tolerance` (0.025), `--fresh`, `-s` / `--save`, `--result-file`, `--variety-line`.
- Every rebuilt draw is a retrain capped at `--train-seconds`, the daily predictor's budget per model (see [models](models.md)), so parameters are chosen for the way production trains them.
- Trials run one at a time on the GPU, each in a spawned child with the `--trial-timeout` budget.
- A `MedianPruner` stops a trial whose running score after `--prune-after-days` draws is below the median of the study's earlier trials at that draw; the first 3 trials are never pruned.
- Each study's decision is written to `bestParams_<game>.json` (and committed with `--save`) when the study ends, not at the end of the run.
- `autoencoder_labelSmoothing` is pinned to 0 by its search space: the reconstruction loss doubles as the anomaly signal.

### Resuming a run

A run keeps a marker in `data/hyperOptCache/deeplearning-run.json`. The same command after an interruption skips the studies it finished and runs only the trials a study still lacks. A run is defined by its games, models, trials, days, training seconds, gate, aim and loss tolerance; other settings start a new run. `--fresh` resets the marker.

### The aim

The tuner has two scores (`--aim`, the *Aim* field on the Hyperopts page). Both use the learning check's excess loss (`src/LearningCheck.py`): the row's plain cross-entropy on its validation windows minus that of the training frequencies, without label smoothing or L2 on either side.

- **`variety`** (the default): per rebuilt draw, the share of validation windows whose ticket differs from the most common one, minus a penalty once the excess loss passes the tolerance: `(1 − same-ticket share) − max(0, excess − tolerance) / tolerance`. Each tolerance's worth above the tolerance costs one full point. 1 is a different ticket on every window within the tolerance; a fixed ticket scores about 0.
- **`skill`**: minus the mean excess loss. 0 is as good as serving the counts, above 0 better.

Notes:

- `--loss-tolerance` is in absolute nats (0.001-1.0); a game with more numbers may need another value.
- Each aim has its own studies: `<game>-<model_type>-variety` and `<game>-<model_type>-skill`.
- The gate scores the served values and the defaults with the same aim, and the variety rule applies under both.
- The variety aim rewards tickets that follow the input; it does not measure hits.
- Under `skill`, an output that only shrinks the counts toward uniform can score above 0 without reading its input where frequencies drift; `learns_days` on the record is the test for that.

### Gate

The default reference is `Predictor.py`'s own defaults (`DeepLearningTuning.predictor_defaults`), trained and scored on the same draws with the same budget. The references cost one or two extra full trials per study.

The record `tuningGate["<row> Model"]` adds `aim`, `loss_tolerance`, `train_seconds`, a `lockbox` note, and per candidate `variety_mean`, `loss_penalty_mean`, `within_tolerance`, `learns_days`, `hits`, `val_loss_mean` and `excess_mean`.

Two limits of the score:

- Its standard error is taken over the rebuilt draws, which are warm-started retrains on almost the same history, so one standard error is a floor rather than a test.
- The validation split of each retrain contains the lockbox period; this tuner does not withhold it, and the record says so.

## Shared run machinery (`src/HyperoptRunner.py`)

Every tuner except the ensemble tuner (in-process evaluation) and the deep-learning tuner (its own GPU-serial child) runs its trials through `optimize_study()`:

- Each trial runs in its own forked process; the objective's loaded data is inherited through the fork.
- Trials run in parallel where the tuner allows, each launch gated on measured memory.
- A trial process that dies marks its trial failed; three failures in a row give up on the study, not the run.
- Ctrl+C or `kill` of the coordinator tears down every trial process with its workers.
- Trials left `RUNNING` by a killed run are marked failed at the next start (`fail_stale_running_trials`).
- The sampler runs with `constant_liar` when trials run concurrently.
- A trial the tuner's own cost gate skipped does not count toward the requested trials; attempts are capped at four times the request.
- An optional deadline per trial (`trial_timeout_seconds`) kills the trial tree and records it as pruned with `timeout`.

## What a rerun continues

Every tuner opens its study with `load_if_exists`: a second run of the same game and row (and, for deep learning, the same aim) adds its trials to the same study, and the sampler sees every earlier trial. The gate only lets a trial of the current run be the challenger; earlier runs' trials were scored on another window and are not compared.

*Fresh* on the deep-learning form (`--fresh`) only restarts the count of the run's own trials, the resume marker; it does not start an empty study.

The studies live in `db.sqlite3` of the folder a run works in. The GPU worker, the DL360 and this server's fallback folder each have their own study history (see [operations](operations.md)), separate from the checkout the weekly chain uses.

## The weekly tuning chain

The server's scheduler (`jobs.js`) starts the chain when Saturday's lottery run finishes, as seven jobs in order:

| Job | Command | Log |
|---|---|---|
| `hyperoptStatistics` | `HyperoptStatistics.py` | `log/hyperoptStatistics.log` |
| `hyperoptBoost` | `HyperoptBoost.py` | `log/hyperoptBoost.log` |
| `hyperoptRLTicket` | `HyperoptRLTicket.py` | `log/hyperoptRLTicket.log` |
| `hyperoptEnsemble` | `HyperoptEnsemble.py` | `log/hyperoptEnsemble.log` |
| `hyperoptQuantum` | `HyperoptQuantum.py` | `log/hyperoptQuantum.log` |
| `trainMetaLearner` | `TrainMetaLearner.py` | `log/TrainMetaLearner.log` |
| `kenoSubsetTuning` | `HyperoptStatistics.py -g keno -s KenoSubsetTuning` | `log/kenoSubsetTuning.log` |

- The quantum tuner runs before the retrain, so the quantum artifacts are trained on freshly tuned parameters and the score table is collected once.
- The keno step runs after the retrain, so it scores the artifacts that are served.
- Every job uses the same gate and the variety line from `config/tuning.json`.
- The deep-learning tuner is not in the chain.
- `runHyperopt.sh` runs the same order by hand, as a fallback; it first waits up to 6 hours for `process.lock`.
- A tuner started by hand from a terminal logs to that terminal.

The schedule, the queue and the Jobs page are in [operations](operations.md).

## The Hyperopts page

**Admin › Models › Hyperopts** (`/app/admin/hyperopts`, `hyperopts.js`, administrators only) starts any of the six tuners for one game by hand.

| Tuner | Choice | Games | Form defaults |
|---|---|---|---|
| Deep learning | one model type | lottery | 15 trials, 3 draws, 240 s training, prune after 2, trial limit 7200 s, gate margin 1 SE, aim, loss tolerance, fresh |
| Statistics | one strategy | lottery and markets | 15 trials, 90 draws, trial limit 1200 s, margin 1 SE |
| Boosting | one strategy | lottery | 15 trials, 90 draws, trial limit 1200 s, margin 1 SE |
| Ensemble | the row | lottery | 16384 evaluations, 120 days, margin 1 SE |
| Quantum | one row or all three | lottery, crypto, shares | 15 trials, 300 days, trial limit 1800 s, margin 2.5 SE |
| RL | the game decides the row | lottery and markets | 20 trials, 40 days, margin 1 SE |

Every form also has the variety line (50-100%, the setting by default). The page offers a strategy only on the games it applies to.

### Where a run goes

- Deep learning runs on the GPU worker; without one the start is refused (503).
- The other tuners run on the DL360 when it is configured, and move to this server when it cannot be reached; without a DL360 entry they run on this server directly.
- The budget is every trial plus the gate's two references at the trial limit, plus an hour (half an hour more on the DL360 lane); six hours for the ensemble and RL tuners; two weeks at most.
- The tuner runs with `-s false` (no commit, no push) and `--result-file`.

The machines, lanes, copies and power handling are in [operations](operations.md).

### Files and statuses

- `config/hyperopts/<id>.json`: the run's record.
- `config/hyperopts/<id>.result.json`: the tuner's result file, refreshed while it runs.
- `config/hyperopts/<id>.served.json`: the served `bestParams_<game>.json` as it was at the start.
- `log/hyperopts/<id>.log`: the tuner's output.

Statuses: queued, running, done, failed, out of time, stopped, lost, cancelled, ended. The list shows the newest 100 runs.

The result file (`src/TuningResult.py`) holds per study: `tuner`, `game`, `row`, `status`, `wanted`, `days`, `progress` (trials done), `current` (what is scored now), `decision`, `reason`, `summary`, `params` (the values the decision would write, with the keys the tuner derives from them), `remove` (keys it would remove), `modelScores`, `rows`, `record` (the gate record) and `challengerParams`. A game a tuner cannot tune is recorded as *skipped* with its reason. A run with several studies shows and applies each on its own.

### Record only, and Apply

Nothing a run finds is served by itself. The page shows each study's progress, the gate's verdict and the values it would write next to the served ones. *Apply* writes them into `bestParams_<game>.json` on this server; the next pipeline run serves them.

- **Apply the gate's result:** the values, removed keys, `modelScores` and gate records of the decision.
- **Apply this run's best:** the challenger trial against the gate; its record then reads *applied by hand against the gate's verdict* with decision `replaced`. Not offered when the gate already replaced.
- Every applied record is stamped `applied` (by, at, what, hyperopt).
- Apply is refused when the study is not done, has nothing to write, was already applied, or a job holds `process.lock`.
- When a value it would replace changed since the run started, Apply asks first (`anyway`).

### Routes

| Route | Does | Audit |
|---|---|---|
| `GET /api/admin/hyperopts` | the list, forms and setting | |
| `GET /api/admin/hyperopts/:id` | one run with its studies | |
| `GET /api/admin/hyperopts/:id/log` | the last 64 KB of the log | |
| `POST /api/admin/hyperopts/start` | start a run | `hyperopt.started` |
| `POST /api/admin/hyperopts/:id/cancel` | cancel a queued run or stop a running one | `hyperopt.cancelled` |
| `POST /api/admin/hyperopts/:id/apply` | apply (`what`: `gate` or `challenger`, `study`, `anyway`) | `hyperopt.applied` |
| `POST /api/admin/hyperopts/:id/delete` | delete a run | `hyperopt.deleted` |
| `POST /api/admin/hyperopts/settings` | the variety line | `hyperopt.settings` |

A stopped run's finished trials stay in the study.

### Deleting a run

*Delete* removes a run's record, result, served snapshot and log. It is refused while the run is queued or running (stop or cancel it first). It does not remove the run's trials from the study on the worker, and values applied from it stay in `bestParams_<game>.json`.

Tests: `node test/hyperopts.test.js`, `python3 -m src.TuningResult`.

## Parameters page and manual overrides

**Admin › Models › Parameters** (`/app/admin/parameters?game=`, `params.js`, administrators only) shows every setting in a game's `bestParams_<game>.json`, one folding card per model.

- Each key goes to the card of the model whose code reads it (`FAMILIES`; an exact key first, then the longest matching prefix).
- A key no family claims is shown under *Not assigned*; keys nothing reads under *Stored but not read*.
- Each card shows its switch, the last tuning-gate decision, its variant rows and its settings, each setting with a "?" help text (`web/src/pages/admin/paramHelp.ts`).
- A key missing from the file is *not set*: the model uses the default in its code, which the page does not repeat.
- Every deep-learning row reads its own `<prefix>_yearsOfHistory` (`Predictor.rowYearsOfHistory`); the LSTM rows read the bare `yearsOfHistory`, which is also the fallback of a row without its own key.

### Manual overrides

From a card, an administrator can set a manual value for a setting in the file, a model's switch, a shared setting (`yearsOfHistory`, `memoriseWindowDraws`, `useMemoriseRows`) or a row's sample temperature.

- A manual value wins over the tuners from the game's next predictor run until it is released.
- The values are kept in `config/paramOverrides.json` (gitignored, written only by the server), never in `bestParams_<game>.json`. The tuners keep writing their file underneath; a release puts the tuned value back.
- A value must be of the tuned value's kind (whole number, number, true/false or word), not below 0; a temperature above 0. Lists and objects are not editable.
- Routes: `GET /api/admin/parameters/:game`, `POST /api/admin/parameters/:game/override` (audit `params.override`), `POST /api/admin/parameters/:game/release` (audit `params.released`).

Where overrides apply:

- `Predictor.py` applies them wherever it reads the file (`src/ParamOverrides.py`) and labels each row that runs on a manual value, and its sampled and memorise twins, with `"manual": [keys]` in the day file.
- `TrainMetaLearner.py` applies them too: the meta-learners are trained on the base rows the daily run serves. Each artifact and `meta_learner_metrics.json` record the manual values under `manualOverrides`, and a manual value of a base row changes the cached score table's key, so the table is recollected.
- The tuners and the controls read the tuned file. A manual value is an experiment on the served rows, not a new tuned value.
- `HyperoptQuantum.py` shares the cached table but scores the tuned values, so while a base row has a manual value the trainer and the quantum tuner each recollect the table.

### Fresh retrain

Setting or releasing a deep-learning row's setting (or the shared `yearsOfHistory`, for the rows that read it) asks that row to train once from new weights (`fresh` in the overrides file). The predictor honours the request on the row's next successful training and records it in `data/models/fresh-<game>.json`; that run takes longer than usual. Statistical and boosting rows refit every run anyway.

Tests: `node test/params.test.js`, `python3 -m src.ParamOverrides`.

## Backtests from the admin pages

**Admin › Models › Backtests** (`/app/admin/backtests`, `backtests.js`), and the *Backtest* link on a model's card on Parameters, start a backtest of one model on one game over the newest N draws, with the served values (the tuned file plus the manual values) or the tuned values alone.

The backtest is the tuner's own evaluation: `RunBacktest.py` drives the row's tuner objective with a fixed trial that answers every setting with the chosen values, and the gate's default where the file has none. Its score is therefore the gate's score, comparable with the gate record when the window has the same size. The lockbox draws are backtested but never scored or shown.

| Rows | Games | Draws (default / max) | Runs on |
|---|---|---|---|
| Statistical | lottery; the four that run there also on the markets | 90 / 1000 | DL360 when configured, else this server |
| GARCH, Regime HMM | the four market games | 90 / 1000 | DL360 when configured, else this server |
| Six boosting rows | lottery | 90 / 1000 | DL360 when configured, else this server |
| Seven deep-learning rows | lottery | 3 / 30 | GPU worker |

Not covered: the quantum rows, the ensembles, the meta-learners, the RL rows and the deep-learning rows on the markets. The rows excluded on the positional games are the tuners' own.

```bash
python3 RunBacktest.py --rows        # the rows and games covered, as JSON
python3 RunBacktest.py --gpu-rows    # the rows that run on the GPU worker, as JSON
```

`RunBacktest.py --id <id> [--budget SECONDS]` is what the server runs; it is not started by hand.

### How it runs

- The server writes the request to `config/backtests/<id>.json` (gitignored) with status `queued`; the script rewrites it with its progress and result: `running`, then `done`, `failed`, `out of time` or `stopped`.
- **On this server:** the request queues the scheduler's `backtest` job. That job is never scheduled and cannot be *Run now*; several can wait, and every scheduled job goes before a queued backtest (`pickNext`). A backtest starts only with at least 15 minutes before the next scheduled run and gets a `--budget` that ends it 10 minutes before that run, at most 3 hours (`backtestBudget`). It holds `process.lock` and shows under Jobs & services.
- **On a worker:** the request goes to the worker's own queue, beside the schedule and without this server's `process.lock`. Without a GPU worker a deep-learning backtest is refused (503).
- A backtest that runs out of time keeps no score: try fewer draws.
- The Backtests page reads the record every 5 seconds; *Stop* ends the run and the record says *stopped*.
- The finished record holds the score (mean over the scored draws minus one standard error), every scored draw's hits (and profit where the game pays), the values it ran with, the manual ones among them, and the gate's own scores for the row.

### The deep-learning rows

LSTM, TCN, the two Unified rows, Transformer, GNN and Autoencoder are backtested with the deep-learning tuner's evaluation: every backtested draw is a retrain on the history before it with the 240 s training budget, in a spawned child. The score is the tuner's *skill* score: minus the mean excess loss over the counts (`DeepLearningTuning.day_score`), with its standard error over the draws and the learning check's verdict per draw. The settings come from the tuner's search space (`HyperoptDeepLearning.suggest_model_params`), answered with the served or tuned values and `Predictor.py`'s default where the file has none. Hits are on the record, not in the score. Each retrain's own validation split still contains the lockbox period, and the result says so.

### Routes and deleting

| Route | Does | Audit |
|---|---|---|
| `GET /api/admin/backtests` | the list, the rows table, the lanes | |
| `GET /api/admin/backtests/:id` | one backtest | |
| `POST /api/admin/backtests/start` | start (`game`, `row`, `days`, `values`: `served` or `tuned`) | `backtest.started` |
| `POST /api/admin/backtests/:id/cancel` | cancel a queued backtest or stop the running one | `backtest.cancelled` |
| `POST /api/admin/backtests/:id/delete` | delete the record | `backtest.deleted` |

*Delete* removes the backtest's record; it is refused while the backtest is queued or running. The lane's log keeps its lines.

Tests: `node test/backtests.test.js` (the row table against `RunBacktest.py --rows`), the budget and queue rules in `node test/jobs.test.js`.

## Training the meta-learner (`TrainMetaLearner.py`)

Trains five stacking variants per game on the base models' backtest scores: `MetaLearner Model` (logistic regression), `MetaLearnerV2 Model` (gradient boosting), the two quantum rows and the quantum kernel's classical control (see [research](research.md)).

```bash
python3 TrainMetaLearner.py --games lotto,keno --days 300
```

Flags: `-g` / `--games` (default every game), `-d` / `--days` (300), `--lockbox-report`.

### The training table

- The base models are built from the game's served parameters (`src/ModelFactory.py`, the tuned file with the manual overrides): the seven statistical rows `Markov`, `MarkovMonteCarlo`, `MarkovBayesian`, `MarkovBayesianEnhanched`, `PoissonMonteCarlo`, `PoissonMarkov`, `LaplaceMonteCarlo`, plus `XGBoost Model`.
- `Chronos Model` and `TimesFM Model` join only when switched on and installed.
- The positional games leave out the three rows without per-position modelling; the market games add the market rows.
- The models are backtested once with `collect_scores=True` over the last `--days` draws, giving a (day, number) → [each model's score, drawn label] table over the game's number range. The table is reused for every variant and cached (see *The quantum tuner*).
- Every artifact stores its own `feature_names`; `Predictor.py` builds its vectors from that list.

### Fitting and metrics

- `MetaLearner Model`: `LogisticRegression(class_weight="balanced")`. `MetaLearnerV2 Model`: `GradientBoostingClassifier` with class balance through `sample_weight`.
- Each variant is evaluated on a walk-forward holdout (the last 20% of days), then refitted on the full window before saving.
- Reported per variant (`src/Calibration.py`): Brier and log-loss against their base-rate values, AUC, ten reliability bins with the expected calibration error, and the top-`draw_size` hits per day against chance. For the positional games: per-slot top-1 accuracy and the argmax ticket's profit.
- Artifacts in `data/models/<game>/`: `meta_learner.joblib`, `meta_learner_v2.joblib`, `quantum_meta_learner.joblib`, `quantum_vqc_meta_learner.joblib`, `classical_svm_meta_learner.joblib`.
- Each artifact holds `metrics`, `trained_at`, the resolved `params`, `versions` (Python, numpy, scikit-learn, joblib) and `manualOverrides`.
- `data/models/<game>/meta_learner_metrics.json` holds the same metrics and versions for every variant, so one week reads against the last.

### Special columns

For the games with special columns (Euromillions, EuroDreams, VikingLotto, Joker+) a second model per variant is trained the same way on the special column's own scores (`_special_scores`) and its own range, determined from the data (`determine_special_range`). Both models of a variant are stored in one artifact; the special column's report is nested as `metrics.special` and duplicated as `special_metrics`. The Backtester collects main and special scores in two separate calls, so the ranges are never mixed.

### The positional formulation

A per-number ranking cannot express digit order or repeated digits. For the positional games (Pick3, Joker+ and the market games) the candidates are the (position, digit) pairs of a day:

- Every capable base model exposes `score_positions()`, one `{digit: score}` dict per slot, normalised per slot at training and serving time (`Helpers.normalize_position_scores`).
- The Backtester records them as `<model>_position_scores` next to `actual_ordered` (the drawn order).
- One classifier is trained per position, for all five variants; the artifact carries `"positional": True` with `position_models`.
- `Predictor.py` serves the argmax digit per slot in drawn order, duplicates allowed, never sorted.

### The lockbox

The lockbox (`lockbox.json`, `src/Lockbox.py`, see [research](research.md)) is enforced here: the declared period's days leave the table before any split or fit. `--lockbox-report` scores the freshly fitted artifacts on those days once and writes `data/models/<game>/lockbox_report.json`; it is not meant to run every week. A malformed `lockbox.json` stops the trainer before any backtest; the row tuners go on after one warning that the run is not lockbox-safe.
