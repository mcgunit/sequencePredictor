# Research: quantum, randomness and the controls

This page covers the research layer around the prediction rows: the quantum-assisted meta-learners and their classical control, the experiment that tries to tell real draws from fair simulated ones, the negative controls (synthetic, shuffled, irrelevant-feature and tuning controls, the per-row chance test), the lockbox period, and the rules for what a result may and may not be read as. The models themselves are on [models](models.md), the tuners and their gates on [tuning](tuning.md), the ranking on [scoring](scoring.md), the schedule and the worker machines on [operations](operations.md).

## Research hypothesis

Quantum models are one more research layer next to the statistical, deep-learning, boosting and stacking models. The purpose is not to assume that a quantum model can predict an inherently random draw, but to test whether quantum feature maps or hybrid quantum-classical models detect reproducible structure that the classical models do not.

> A correctly operated lottery-style game should not contain stable, exploitable temporal information. If a model appears to outperform suitable random and classical baselines, the result must remain reproducible under walk-forward validation, synthetic-random controls, shuffled-history controls, and an untouched holdout period.

A quantum model is treated as another adversarial test of the sequence-generating process. Failure to find predictive structure is evidence consistent with the modelled randomness assumptions, not proof of perfect randomness. Apparent predictive structure is a signal for further investigation, not proof that a game is predictable or biased.

The research phases are named Q0 to Q4:

| Phase | Subject | State |
|---|---|---|
| Q0 | Classical controls: null histories, per-row chance test, tuning control, feature control, lockbox | built |
| Q1 | Quantum meta-learning | built |
| Q2 | Randomness discrimination | built |
| Q3 | Direct quantum scoring of candidate numbers | not built |
| Q4 | Candidate search (Grover) | not built |

## Quantum meta-learners (Q1)

Three rows belong to the quantum track. Each is an independently tracked prediction row; none replaces a classical meta-learner.

| Row | Classifier | Artifact in `data/models/<game>/` | Switch in `bestParams_<game>.json` |
|---|---|---|---|
| `QuantumMetaLearner Model` | quantum-kernel SVC | `quantum_meta_learner.joblib` | `useQuantumMetaLearner` |
| `QuantumVQC Model` | variational quantum classifier | `quantum_vqc_meta_learner.joblib` | `useQuantumVqcMetaLearner` |
| `ClassicalSVM Model` | RBF-kernel SVC, the classical control | `classical_svm_meta_learner.joblib` | `useClassicalSvmMetaLearner` |

The three switches default to on; a game's file sets one to `false` to stop training that row.

Everything is simulated: `src/QuantumModels.py` holds a pure-numpy batched statevector engine, the feature map, the three classifiers and their fit functions. Four qubits is a 16-amplitude state, so there is no quantum SDK dependency. Simulated circuits run on classical hardware and demonstrate no quantum computational advantage.

### Training data

The quantum rows train on the same per-number table as `MetaLearner Model` and `MetaLearnerV2 Model`. For every backtested day and candidate number the table holds the scores of the game's base models (`BASE_MODEL_NAMES` in `src/ModelFactory.py`):

- `Markov Model`, `MarkovMonteCarlo Model`, `MarkovBayesian Model`, `MarkovBayesianEnhanched Model`
- `PoissonMonteCarlo Model`, `PoissonMarkov Model`, `LaplaceMonteCarlo Model`
- `XGBoost Model`
- `Chronos Model` when `useChronosFeature` is on (the default) and the library is installed
- `TimesFM Model` when `useTimesFmFeature` is on (off by default)
- the market rows, for the market games only

The positional games (pick3, Joker+) leave out the base models that have no per-position modelling. The label is:

```text
1 = the candidate number occurred in the next real draw
0 = the candidate number did not occur in the next real draw
```

`TrainMetaLearner.py` collects the backtest once and fits all five meta-learner variants from it (`meta_learner_variants`: logistic, gradient boosting, quantum kernel, VQC, classical SVM), so every variant sees the same days, candidate rows, feature values, labels, chronological window and main/special separation:

```text
Base-model score vectors
          |
          +--> LogisticRegression            MetaLearner Model
          +--> GradientBoostingClassifier    MetaLearnerV2 Model
          +--> Quantum kernel + SVC          QuantumMetaLearner Model
          +--> Variational quantum circuit   QuantumVQC Model
          +--> RBF kernel + SVC              ClassicalSVM Model
```

The research question this answers: can a quantum feature map find useful nonlinear relationships among the base-model scores that the logistic-regression and gradient-boosting meta-learners miss?

Special columns stay separate, as for the classical meta-learners: a main-number model and a special-column model per variant, so Euromillions stars, the EuroDreams dream number and the VikingLotto super viking number are never mixed with the main range. Pick3 and Joker+ use the positional formulation: one classifier per digit position, served as a ticket in drawn order.

### Feature encoding

The encoding is fitted inside each classifier's `fit()`, on the training rows only, so nothing leaks from the held-out period:

```text
Base-model scores
          |
          v
StandardScaler (training rows only)
          |
          v
PCA to n_qubits components (training rows only)
          |
          v
RY angle encoding, scaled by encodingScale, with a CZ entangling ring,
re-uploaded encodingLayers times (QuantumFeatureMap)
```

`encodingScale` acts as the kernel bandwidth and rotation wrap-around control and matters more than any other knob. The tuner searches it from 0.1 to 2.0 in steps of 0.05. The width is four qubits.

### The three classifiers

All three are picklable sklearn-style classifiers (`fit` / `predict_proba`).

- **`QuantumKernelClassifier`**: the fidelity kernel |<phi(x)|phi(y)>|^2 into an `SVC(kernel='precomputed', class_weight='balanced')`, on a class-balanced subsample capped at `maxTrainSamples`.
- **`VariationalQuantumClassifier`**: trainable RZ+RY layers with a CNOT entangling ring, a Z readout on the last qubit (the ring's final control, whose Z depends on all four qubits) through a trainable affine and sigmoid, class-weighted cross-entropy, numpy Adam with exact parameter-shift gradients.
- **`RbfKernelClassifier`**: the quantum-kernel classifier with the state-overlap kernel replaced by exp(-gamma |x - y|^2), on the same pipeline: same scaler, same PCA width, same class-balanced subsampling, same probability-calibrated, class-balanced SVC. `ClassicalSVM Model` and `QuantumMetaLearner Model` therefore differ in the kernel and in the settings each is tuned to, not in features, reduction or evaluation.

`python3 -m src.QuantumModels` (part of `npm test`) checks the parameter-shift gradient of every RZ and RY angle and both readout parameters against central finite differences of the training loss, and asserts that every angle has a live gradient.

### Parameters and artifacts

Hyperparameters are read from `bestParams_<game>.json`, with the module's defaults when a key is missing:

| Row | Keys (default) |
|---|---|
| Quantum kernel | `quantumKernel_nQubits` (4), `quantumKernel_encodingLayers` (2), `quantumKernel_encodingScale` (1.0), `quantumKernel_C` (1.0), `quantumKernel_maxTrainSamples` (2000) |
| VQC | `quantumVqc_nQubits` (4), `quantumVqc_numLayers` (2), `quantumVqc_encodingScale` (1.0), `quantumVqc_learningRate` (0.05), `quantumVqc_epochs` (80), `quantumVqc_batchSize` (128) |
| Classical SVM | `classicalSvm_nFeatures` (4), `classicalSvm_gamma` (`scale`), `classicalSvm_C` (1.0), `classicalSvm_maxTrainSamples` (2000) |

A manual override set on the Parameters page replaces a tuned value; the artifact and the metrics file then record it under `manualOverrides` (see [tuning](tuning.md)).

An artifact carries the classical artifacts' key layout (feature-name order, the fitted model with its scaler and PCA, the main and the optional special-column model, range and draw size, the `positional` flag and per-position models for pick3 and Joker+) plus `trained_at`, the resolved `params` and the package `versions`. `Predictor.py`'s `runMetaLearnerVariant` serves all five variants through the same code, including special columns, the positional branch and the Keno subsets. An artifact stores its own feature names and readout, so one trained under an earlier table keeps serving as trained until the next retrain.

### Tuning

`HyperoptQuantum.py` tunes the three rows, one study per row and game, each through the champion/challenger gate and the variety rule. The control row is tuned by the same objective on the same day split as the quantum rows, so it is not handicapped against them. It runs as a step of the weekly tuning chain before the meta-learner retrain, and can be started from the Hyperopts page. Scores, gate margins, flags and the chain are documented on [tuning](tuning.md).

## Randomness discrimination (Q2)

`RandomnessDiscrimination.py` asks whether any classifier can tell real draw windows from correctly simulated fair ones.

```text
Class 0 = synthetic draws generated according to the game rules
Class 1 = real historical draws
```

```
python3 RandomnessDiscrimination.py -g lotto,pick3,euromillions -w 10 -n 20 --draws 1000
```

### Features, suite and split

- Windows of `-w` draws (default 10) become per-number relative counts; for pick3 and Joker+ the counts are kept per slot, so digit order survives.
- The suite: logistic regression, RBF support-vector machine, random forest, gradient boosting, a small neural network, the quantum-kernel classifier and the variational quantum classifier. The quantum variants run on their defaults; `--skip-quantum` leaves them out.
- The split is chronological: the newest 25% of the windows are the test set, for both classes.
- `--draws` (default 1000) keeps only the most recent draws, where the rules have not changed. The window is cut before a history is shuffled, so the shuffled control keeps exactly the marginals of the draws it is compared on.

### Three comparisons

One AUC cannot be read on its own, so every run measures three comparisons through the identical pipeline:

| Comparison | Role |
|---|---|
| real vs synthetic | the question |
| synthetic vs synthetic | the null band: two independent fair histories, where the true answer is 0.5 |
| shuffled vs synthetic | the marginals control: the real draws reordered, so frequencies survive and time order does not |

The number compared is the best AUC of the suite per repetition. The best of seven classifiers is biased upwards even when all seven are worthless, so the real comparison is held against the null comparison's own best-of-suite, never against the mean of its individual classifiers.

### The verdict

`compute_verdict` (verdict version 2) gives one of five statuses:

| Status | Meaning |
|---|---|
| `too few draws` | fewer held-out test windows (both classes together) than `q2.few_windows` |
| `failed` | a comparison scored fewer than two seeds; no number is shown |
| `above` | the real mean clears the null bound and the shuffled control's bound |
| `above band, not shuffled` | the real mean clears the null bound but not the shuffled control's |
| `within` | the real mean is inside the null band |

- **The power rule** lives in `controlRules.json` at the repository root (`{"q2": {"few_windows": 250}}`), read by both `RandomnessDiscrimination.py` and `controls.js`. It is a property of the history's length, `--draws` and `--window`, fixed before any classifier runs. Games under the line keep running, so their numbers stay on the admin pages. Every record stores the `few_windows` it was judged by; a version 1 record, which stores none, is read by the current value.
- **The bounds come from the seed count.** *Above* needs the real mean (best AUC of the suite, mean over seeds) to clear the null mean + k sd, the one-sided 97.725% prediction bound for one more fair run, with k = t(n-1) x sqrt(1 + 1/n): 5.23 at 3 seeds, 2.43 at 10, 2.19 at 20. The shuffled control's bound is worked out the same way from its own seeds. The real mean is held against a single-run bound because averaging seeds removes only the synthetic side's noise, not the history's own.
- **Repetitions.** `-n` defaults to 20 (allowed 2 to 999); `--market-repetitions` overrides it for the market games when set.

The record stores `version`, `status`, `few_windows`, `windows_test`, `band_coverage`, `band_k`, `shuffled_k`, `null_threshold`, `null_threshold_se`, `shuffled_threshold`, `seeds_scored`, the per-seed values, `rule` and `power_rule`, and never a NaN. `--reanalyze` recomputes the verdicts of saved records under the current rule without rerunning anything. Self-check: `RandomnessDiscrimination._self_check()` (in `npm test`).

### What a Q2 result may be read as

Balanced accuracy and ROC AUC near 50% on held-out windows indicate that a model cannot reliably distinguish the two sources. Performance above chance does not imply manipulation or next-draw predictability. A classifier may instead detect:

- historical game-rule changes
- changes in number or special-number ranges
- sorted versus drawn-order differences
- missing or duplicated records
- API or preprocessing artefacts
- equipment or draw-schedule changes
- an incorrect synthetic generator

A synthetic history reproduces the current rules over the whole calendar, so any discrimination result has to be checked against the rule-change dates before it is called evidence of anything.

## Control histories

`src/ControlHistories.py` builds the histories behind every control, so the experiments cannot disagree about what a fair draw is. All are on the real calendar.

- **Synthetic**: draws generated uniformly at random under the game's own rules. The main range and draw size come from `HyperoptStatistics.GAME_CONFIG`; the trailing columns (stars, dream number, super viking, zodiac, bonus) take their ranges from the real history. No structure of any kind exists to be found.
- **Shuffled**: the real draws reordered. Every marginal survives (number frequencies, co-occurrence) and only time order dies. If performance stays similar after shuffling, a model is using marginal frequencies, range characteristics or sorted-position distributions, not next-draw dependence.
- **Market games**: there is no drawing rule to simulate, so the fair history is a geometric random walk with each instrument's real mean and volatility, cut into bins by the game's own causal rule. The shuffled history reorders the real game days' returns and re-cuts the bins by the same rule, so the returns' marginal survives and the bin frequencies move a little. Details are on [markets](markets.md).

Draws of an earlier version of a game are dropped from every control history (`current_era`): a draw dated before the game's era start (`Helpers.GAME_ERA_START`: keno 9 March 2008, lotto 1 October 2011) or with main numbers outside the configured range. The same table limits what the models load.

### Seeds

The control scripts take `--seed`. A number gives a reproducible run (default 1). `weekly` gives a fresh block of seeds every week: (ISO year x 100 + ISO week of the weekend's Saturday) x 10000 + 1 onwards (`ControlHistories.control_seed`). The week is the weekend's, so a catch-up run on Monday keeps its block, and a block is 10,000 seeds wide, so Q2's +1000 null side and the feature control's +1000 noise offsets stay inside it. The scheduled runs use `weekly`, so each week is a new test and not a rerun of the last one. The seeds used are in every record.

## Null histories: the Q0 band (`NullControls.py`)

The best row of a ranking is a maximum over rows, and the maximum of N noisy estimates sits above their mean even when every row is worthless. `NullControls.py` measures how good the best row looks on a history with nothing in it.

```
python3 NullControls.py -g lotto -m both -n 3 -d 120
```

- It builds the control histories (`-m synthetic`, `shuffled` or `both`; `-n` histories per mode, default 3; `--market-repetitions` for the market games) and scores them through the production path: the same models from `src/ModelFactory.py`, the same `src/Backtester.py`, the same random, global-frequency and column-frequency baselines, over `-d` days (default 120).
- The models run on their untuned defaults. The Chronos row is included only with `--foundation`, because each control history needs its own forecast pass.
- `--recent N` builds the control from the newest N draws only (default: the whole current-era history).
- Per game and mode it writes `data/controls/results/<game>-<mode>.json`: the average row, the best row, the selection gain (best minus average), the best-to-worst spread and the random baseline, with the per-seed scores. Special-number rows and baselines are not counted as leaderboard rows. A game's control folders of earlier seeds are removed.

The **band** shown on the admin pages is the mean over the control histories of the best row's score plus two standard deviations across them, taken over the higher of the two controls (`controls.js`). The shuffled control is usually the higher one: it keeps the real history's number frequencies, which the frequency-shaped models exploit.

The band is context only. It knows neither a row's record length nor a row's own fair level, so no row is labelled, greyed or counted against it; rows are judged by the per-row chance test below. The figures are a lower bound on the selection effect, because the rows are untuned: the tuning control measures the layer on top.

## Q0: the per-row chance test (`src/RowChance.py`)

The daily performance report tests every ranked row on its **live** draws (the live record, see [scoring](scoring.md)), on the hit scale the ranking uses: would the row's hits be this good by chance?

### The test

- **Lottery games**: the exact fair-draw null. Given each draw's ticket, a fair draw's hit count has a known distribution: hypergeometric for the set games and keno (the ticket's distinct in-range numbers against the numbers drawn), a distinct-digit law for pick3, the L + R run rule for Joker+. The record's total is their convolution, and p is the chance of at least the row's hits. Ranges are held to `HyperoptStatistics.GAME_CONFIG`.
- **Market games**: a block permutation null. The row's own tickets are shuffled among its own live days within blocks of 7 consecutive live days (live weeks for the week games), the outcomes fixed, 20,000 permutations seeded by the record. This removes a stretch's bin mix and the drift of its volatility regime. It does not remove a row that times the regime within a block, which carries no direction; the Market rows card against GARCH stays the reference for that. With tickets and outcomes both autocorrelated within a block the test is slightly anti-conservative.

### The family

Holm's correction at 5% runs over the rows with at least `minDrawsForRanking` live draws (10 live weeks for the week games). The page shows two figures, both of the current week and never a colour:

- **under 5%**: the rows with raw p < 0.05, always next to what chance puts there (the sum of the rows' own discrete levels for the exact test, rows x 0.05 for the markets)
- **holds**: the rows still under 5% after the correction

The payout games (keno, pick3, Joker+) are tested on hits too, as information only.

### Fresh evidence

Only new draws count as confirmation:

- From the first Sunday on which a row's live record reads p < 0.05, the row is **followed**.
- It is judged once, on as many new live draws as flagged it (at least `minDrawsForRanking`, at most 52): **confirmed** at p < 0.01, otherwise **not confirmed**.
- Until then it reads **running**, with the figures so far and no p. After 28 days without a new live draw it reads **stopped**.

Flags and verdicts are kept in `data/database/rowChance.json` (tracked, written only when it changes). The first run for a game replays every Sunday once, flagging only on or after the `since.json` freeze; later runs test only new Sundays, and a verdict is never recomputed. The one reset is a later `since.json` freeze, which drops the entries of rows flagged before it. A lottery game judged without git is tested but stores nothing; one whose liveness could not be judged is not tested and says why.

Each p stands with its row's live mean, live draw count and null on the same line. Self-check: `python3 -m src.RowChance` (in `npm test`).

## The selection effect of tuning (`TuningControls.py`)

`TuningControls.py` measures the layer above the rows: what the hyperopt itself gains on a history with nothing in it.

```
python3 TuningControls.py -g lotto -m both -n 3
```

- It runs the statistical tuner's own objectives (`HyperoptStatistics.STRATEGIES`, the robust score, the served Keno sizes) for `-t` trials on a control history and compares the best trial with the untuned defaults scored on the same history. This is the comparison the champion/challenger gate makes on the real history.
- On a fair synthetic history nothing can be found, so whatever the best of `-t` trials gains there is what choosing alone gives. The gate's real-history gain (`tuningGate[row].challenger - default`) is put next to it. A real gain inside the control band (mean plus two standard deviations of the control gains) is the effect of choosing, not of finding.
- Like with like: the maximum of n trials grows with n, so `-t` and `-d` default to the weekly tuner's own values (`HyperoptStatistics.DEFAULT_TRIALS`, `DEFAULT_DAYS`). The control withholds the lockbox days the tuner withholds. A gate record made with other trials, another window or another number of withheld lockbox days gets no verdict and says why.
- The band is a floor on the selection effect: the weekly study is warm (its sampler is guided by the completed trials of earlier weeks) while a control study starts cold.
- Defaults are modest: `-g lotto -m both -n 1`; `-n 3` gives the band a width. `--recent N` builds the control from the newest N draws. Records go to `data/controls/tuning/<game>-<mode>.json`.
- Pick3's control counts hits as the ranking's average hits do (a box counts three), which deliberately differs from the slot-hit objective the tuners score pick3 on; Joker+ is runs on both sides.

It is run by hand only and is not part of the schedule.

## Irrelevant-feature control (`IrrelevantFeatureControl.py`)

A random column appended to the meta-learner table must not gain stable importance. If a model gives weight to a column that is provably unrelated to the draw, its weights on the real columns are not evidence of anything either.

```
python3 IrrelevantFeatureControl.py -g lotto,pick3 -r 3 -k 3
```

- Every meta-learner variant `TrainMetaLearner.py` serves (`meta_learner_variants`, the same list with the same parameters) is refitted on the table the tuning chain cached, with the lockbox days removed first, exactly as the trainer removes them.
- `-k` noise columns (default 3) are appended: shuffled copies of real base-model columns, so each has exactly a base model's marginal distribution and no relation to the label or the day.
- `-r` repeats (default 3) with fresh noise, on the trainer's own chronological 80/20 day split.
- Importance is **permutation importance** on both sides of the split: the ranking power lost when one column is scrambled, measured as the drop in |AUC - 0.5|. The unsigned form is used because a Platt-scaled SVM fitted on noise can come out with inverted probabilities, where a signed drop would read a column the model leans on as ignored.
- The table is never recollected. A cache up to `--tolerate-rows` draws (default 7) behind the data file is accepted and the shortfall recorded; a game without a cached table is skipped and says so.

### Verdicts

On the **training rows** the question is whether the model leans on noise:

| Verdict | Rule |
|---|---|
| `fits noise` | a noise column matters at least a tenth as much as an average real column, and the pooled noise importance is more than two standard errors above zero |
| `inconclusive` | the ratio is exceeded but the pooled importance is not more than two standard errors above zero |
| `noise ignored` | otherwise |

The rule uses a ratio and not a share of the total, so one strongly informative base model cannot dilute it; the share is reported too.

On the **held-out days** the noise columns' importance is the feature-level null band (mean + 2 sd). The card lists the base models whose held-out importance clears it: the columns that carry more than noise does. With several real columns tested against a two-sigma band, one can clear it by chance in any week, so a column counts when it stays above the band week after week, not when it appears once. The held-out AUC of the fit without noise, next to the mean with noise, shows what the extra columns cost.

Records go to `data/controls/features/<game>.json`. The rules live in `src/FeatureControl.py`; its self-check (`python3 -m src.FeatureControl`, in `npm test`) shows a logistic regression on plenty of rows ignoring the noise and an unpruned tree on 300 rows fitting it.

## Lockbox period

The lockbox is a chronological period that model selection never sees. Repeatedly inspecting the same holdout and changing the design afterwards turns the holdout into development data; the lockbox is the period that does not get that treatment.

It is declared by hand in `lockbox.json` at the repository root. No file means no lockbox. Dates are inclusive:

```json
{
  "from": "2025-12-01",
  "to": "2026-02-28",
  "declared": "2026-09-27",
  "note": "..."
}
```

The daily predictor is never subject to it. Who honours it, and how:

| Script | Treatment |
|---|---|
| `TrainMetaLearner.py` | drops the period from its table before splitting and fitting; `--lockbox-report` scores the freshly fitted artifacts once on the lockbox days and writes `data/models/<game>/lockbox_report.json` |
| `HyperoptQuantum.py` | drops the period from its score table before the 75/25 day split, so its holdout holds none of it |
| `IrrelevantFeatureControl.py` | drops the period from its table, as the trainer does |
| `MarketRows.py` | withholds the period's days from every score |
| `HyperoptStatistics.py`, `HyperoptBoost.py`, `HyperoptEnsemble.py`, `HyperoptRLTicket.py` (ticket and position studies) | withhold the days from selection (below) |
| `TuningControls.py` | withholds the same days as the statistical tuner |
| `RunBacktest.py` (front-end backtests) | a statistical or boosting backtest scores through the tuner's objective with the lockbox loaded, so the same days are withheld; a deep-learning backtest does not withhold, as its tuner |
| `HyperoptDeepLearning.py` | does **not** withhold it: its score is the validation loss on the last 20% of each retrain's history, which contains the period; its gate records say so |

### Withheld from selection, not from fitting

In the four row tuners the lockbox days are withheld from everything that decides what is served: the trials, the gate's re-scores of the served parameters and the defaults, the standard error behind any margin, `modelScores`, the Keno subset day table and the boosting pruner's steps.

Fitting on a history that contains the days stays allowed, as in the daily predictor. The statistical and boosting tuners still backtest those days and leave them out of the score. The RL studies walk and warm-start through them without scoring them. The ensemble tuner, which fits nothing, drops their day files on loading.

The window is not lengthened to make up for withheld days, so while the window overlaps the lockbox fewer days are scored and the standard error is wider.

### Messages and records

- Each game logs `<game>: lockbox ... - N day(s) withheld from tuning, M scored`; a game with nothing left to score is skipped.
- Every gate record carries `lockbox_days_withheld` next to `window_days` (the ensemble's `subsetEnsembleTunedOn` carries `lockboxDaysWithheld`). The quantum tuner's records count the score-table days taken out before its split and say so with `lockbox_withheld_from: "table"`.
- A malformed `lockbox.json` is one `LOCKBOX WARNING` in the four row tuners, `TuningControls.py` and `RunBacktest.py`: the run says it is not lockbox-safe and goes on withholding nothing. `TrainMetaLearner.py`, `HyperoptQuantum.py`, `IrrelevantFeatureControl.py` and `MarketRows.py` stop with `LOCKBOX ERROR`.

Self-check: `python3 -m src.Lockbox` (in `npm test`).

## How the controls run

The scheduler's Sunday plan (*Weekly control experiments*) starts when Sunday's lottery predictor finishes and runs, in order:

| Job | Script and arguments | Cadence |
|---|---|---|
| Null histories (Q0 band, context) | `NullControls.py`, all eleven games, `-m both -n 3 --market-repetitions 5 --seed weekly -d 120` | first Sunday of the month |
| Randomness discrimination (Q2) | `RandomnessDiscrimination.py`, all eleven games, `-w 10 -n 20 --seed weekly` | weekly |
| Feature control | `IrrelevantFeatureControl.py`, all eleven games, `-r 3 -k 3 --seed weekly` | weekly |
| Market rows | `MarketRows.py -g crypto,shares,cryptoweek,sharesweek -d 250` (see [markets](markets.md)) | weekly |

Between two first Sundays the pages show the latest band with its date.

When a CPU worker is configured (`config/workers.json`), the plan's jobs run on that machine in their own queue: the code and the data a control reads are copied there once per occurrence, the control runs over ssh, and its records are copied back into `data/controls/`. These runs take no `process.lock` on the server. When the worker does not come up, the control and the ones behind it run in the server's own queue instead. A control run by hand, or in the server's queue, takes `process.lock` like every other entry point, so a hand-run predictor waits for it. The schedule, the catch-up window and the worker are documented on [operations](operations.md).

### Records

| Control | Record |
|---|---|
| Q0 per-row test | `modelPerformance.json` (per row, `chance`), `data/database/rowChance.json` |
| Q0 band | `data/controls/results/<game>-<mode>.json` |
| Q2 | `data/controls/discrimination/<game>-w<N>.json` |
| Feature control | `data/controls/features/<game>.json` |
| Tuning control | `data/controls/tuning/<game>-<mode>.json` |
| Market rows | `data/controls/markets/<market>-rows.json` |
| The week's reading | `data/readings/<sunday>.json` |

`data/controls/` is overwritten by every control run and not tracked. `controls.js` reads the records for the admin pages; a missing or broken record never fails a page.

### The week's reading

When the Sunday plan's last job finishes, `readings.js` writes `data/readings/<sunday>.json`: one summary per weekend of the controls for every game (the Q0 row tests as the report ran them, the band as context, Q2's real, shuffled and threshold AUC with seeds and test windows, each feature-control variant's verdict, every market row against uniform and against GARCH), each part with its own record's time. Readings are tracked, so the admin pages can count weeks ("2nd week above"). A reading stores the Q2 rule it was read under; a run of weeks restarts at a change of rule.

A week can be rebuilt from a copy of that week's records:

```
node readings.js --controls <dir> --sunday YYYY-MM-DD [--database <dir>] [--out <dir>] [--note <text>]
```

For a backfill, `--database` must point at the ranking as it stood that Sunday, never the current one.

### Admin pages

- **Admin > Controls**: four cards (*Q0 chance per row*, *Q2 randomness*, *Feature control*, *Market rows*) and a drawer per game, which also shows the Q0 band as context.
- **Admin > Randomness**: per game the daily tripwires (normalised entropy under 0.95, KL divergence against the history above 0.1, the autoencoder alert), their trend, the Q2 verdict and the per-model KL. A game's status reads *watch* when any tripwire is up or Q2 reads `above`, and the page says which.

## Within-draw structure versus temporal predictability

Sorted lottery numbers contain predictable positional structure even when the draw is fair: the first sorted number tends to be lower than the last. A position-based model can learn this order-statistic structure without learning any dependence between successive draws. Two concepts are therefore kept apart:

```text
Within-draw distributional structure
    versus
Between-draw temporal predictive value
```

A model that generates plausible sorted tickets or achieves low positional error is not necessarily producing more next-draw hits than a fair baseline. This is why a per-position model's score for a number is its inclusion probability summed over the positions, and why the shuffled control exists. How each row turns its scores into tickets and Keno subsets is on [models](models.md); the fair value of a Keno ticket per size is on [scoring](scoring.md) and [tuning](tuning.md).

For non-positional games the randomness experiments use a multi-hot representation (one element per possible number). Pick3 and Joker+ keep their positional representation because digit order is part of the result.

## Search versus prediction

Grover search amplifies candidates that match a known condition. It is not a prediction model and creates no information about the next draw: in a ticket search the predictive information would still come from the scoring model behind the oracle. If previous draws contain no useful signal about the next independent draw, Grover search cannot create one.

## Reporting requirements

Every quantum result should include:

- game and historical date range, game rules and any rule-change boundaries
- training, validation and lockbox periods
- feature list and preprocessing steps
- number of qubits, feature map and circuit ansatz, circuit depth and operation counts
- simulator or backend, shot count, noise model if used
- optimizer and optimization budget, hyperparameter search budget
- the classical models given an equivalent tuning budget
- random seeds
- per-number metrics, ranking metrics and the ticket-level hit distribution
- synthetic and shuffled-control results
- runtime and computational cost

A quantum model should not be described as better on the strength of one favourable backtest, one game or one metric. Improvements should be stable across repeated seeds, chronological periods and the appropriate null controls.

## Research interpretation

The quantum extension is an adversarial randomness and model-capability benchmark. A defensible conclusion can state:

```text
Under the tested data representation, model family, optimization budget,
and chronological evaluation period, the quantum-assisted model did or
did not detect reproducible structure beyond the selected classical baselines.
```

It should not state that failure to detect structure proves perfect randomness, or that a small retrospective uplift proves future lottery predictability.

Rules that apply to every number on this page:

- A best-of-N figure (best row, best trial, best classifier) is only read against the same best-of-N on a history with nothing in it.
- A real-history result that the shuffled control reproduces is about marginals or rules, not about time order.
- A band from few seeds is itself luck; bounds widen with fewer seeds, and a record under the power rule reads *too few draws*, not *within*.
- A single week above a band is not a finding. A row, column or game counts when it holds on fresh draws or week after week.
- A lottery row is confirmed only on draws that came after it was flagged, at a fixed length decided in advance.
- The untuned controls are lower bounds on the selection effect.

All quantum experiments remain subject to the repository's education and research disclaimer.

## Not built

- **Q3, direct quantum scoring**: a quantum-assisted score for every candidate number, reusing ticket construction, special-column handling, Keno subsets, the Backtester and the tracking.
- **Q4, candidate search**: a ticket-level score and threshold with a toy Grover oracle on a deliberately small candidate space, compared with direct classical ranking on cost (circuit depth, two-qubit gate count, noise sensitivity, data loading).
- **Period-specific synthetic rules**: a synthetic history uses the current rules over the whole calendar; earlier eras are dropped or windowed out instead of simulated.
- **A full hyperopt on every control history** for all tuners: the tuning control covers the statistical tuner only and is run by hand.
- **A wider quantum circuit** than four qubits, and a split of `src/QuantumModels.py` into a package with separate experiment scripts.

See [roadmap](roadmap.md) for the open items.
