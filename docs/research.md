# Research: quantum, randomness and the controls

> Moved out of the README on 5 Nov 2026, word for word; see "Out of date" at the end of this page for what the 5 Nov check found.

## Quantum-assisted research

> **Status: Phase Q1 is implemented, its listed gaps closed on 28 Sept 2026.** `QuantumMetaLearner Model` (quantum-kernel SVC), `QuantumVQC Model` (variational quantum classifier) and their classical control `ClassicalSVM Model` (the quantum-kernel classifier with an RBF kernel in place of the state overlap, on the same pipeline, with its own tuned C and sample cap) run as tracked prediction rows, trained by `TrainMetaLearner.py` from the same one-pass backtest table as the classical meta-learners and tuned by `HyperoptQuantum.py` (in the weekly `runHyperopt.sh`). Everything is simulated with a pure-numpy batched statevector engine in `src/QuantumModels.py` - four qubits is a 16-amplitude state, so no quantum SDK dependency is needed; `python3 -m src.QuantumModels` (part of `npm test`) checks its parameter-shift gradient against finite differences. The Q0 controls and the Q2 discrimination test run weekly (see *Null controls*); Q3 and Q4 remain future work. Details below in each subsection.

Quantum computing is being investigated as an additional research layer alongside the existing statistical, deep-learning, boosting, and stacking models. The purpose is not to assume that a quantum model can predict an inherently random draw. The purpose is to test whether quantum feature maps or hybrid quantum-for classical models can detect reproducible structure that the existing classical models do not detect.

The main research hypothesis is:

> A correctly operated lottery-style game should not contain stable, exploitable temporal information. If a model appears to outperform suitable random and classical baselines, the result must remain reproducible under walk-forward validation, synthetic-random controls, shuffled-history controls, and an untouched holdout period.

A quantum model is therefore treated as another adversarial test of the sequence-generating process, comparable to evaluating the resilience of a system against another class of analysis. Failure to find predictive structure is evidence consistent with the modeled randomness assumptions, but it is not proof of perfect randomness. Apparent predictive structure is a signal for further investigation, not immediate proof that a game is predictable or biased.

### Recommended first integration: `QuantumMetaLearner Model` (implemented)

Implemented as specified below, with these concrete choices: artifacts are `data/models/<game>/quantum_meta_learner.joblib` (quantum kernel), `quantum_vqc_meta_learner.joblib` (variational) and `classical_svm_meta_learner.joblib` (the RBF control, since 28 Sept 2026), all carrying the classical artifacts' exact key layout plus `trained_at`, the resolved `params` and the package `versions` (the README's metadata requirement), so `Predictor.py`'s `runMetaLearnerVariant` serves them unchanged - special-column separation, the positional pick3/Joker+ branch and Keno subsets included. Training is gated per game by `"useQuantumMetaLearner"` / `"useQuantumVqcMetaLearner"` / `"useClassicalSvmMetaLearner"` in `bestParams_<game>.json`; they default **on** (the README suggested opt-in when heavy simulation was assumed - the numpy 4-qubit implementation trains in seconds to ~1 minute per game, so the flag is an off-switch instead).

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

### Quantum feature encoding (implemented)

Implemented exactly as diagrammed: `StandardScaler` -> `PCA(n_qubits)` fitted **only** on the training rows (inside each classifier's `fit()`), then RY angle encoding scaled by `encodingScale` with a CZ entangling ring, re-uploaded `encodingLayers` times (`QuantumFeatureMap` in `src/QuantumModels.py`). One practical finding from testing: `encodingScale` behaves as the kernel bandwidth / rotation wrap-around control and matters more than any other knob. Early hand tests favoured ~0.3-0.5 over 1.0, and the first tuned values per game sat at 0.5-1.0, the bottom of the 0.5-2.0 grid `HyperoptQuantum.py` searched until 28 Sept 2026. It now searches 0.1-2.0 in steps of 0.05: every old grid value lies on the new grid, and Optuna continues the persisted studies, sampling the widened parameter independently of the earlier trials.

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

### Initial quantum model candidates (both implemented)

Both approaches below exist in `src/QuantumModels.py` as picklable sklearn-style classifiers (`fit`/`predict_proba`): `QuantumKernelClassifier` (fidelity kernel |<phi(x)|phi(y)>|^2 into an `SVC(kernel='precomputed', class_weight='balanced')`, class-balanced subsampling capped at `maxTrainSamples`) and `VariationalQuantumClassifier` (trainable RZ+RY layers with a CNOT entangling ring, Z-readout through a trainable affine+sigmoid, class-weighted cross-entropy, numpy Adam with exact parameter-shift gradients). The parameter-shift gradient is checked against central finite differences of the training loss - every RZ and RY angle in every layer and both readout parameters - by the module's self-check, `python3 -m src.QuantumModels`, part of `npm test` (agreement to 1e-10 over 18 parameters, and the check is shown to catch a 1% error in one of them). The check earned its place at once: the readout used to be qubit 0, the entangling ring's final *target*, and Z on a CNOT's target pulls back through the ring to an observable without that qubit - so the last layer's two rotations on qubit 0 could never change the output, and a finite-difference check passes them trivially as 0 = 0 (found in review on 28 Sept 2026). The readout is now the last qubit, the ring's final *control*, whose Z pulls back to all four qubits; the self-check asserts that every angle has a live gradient and shows the two the qubit-0 readout left dead. Served artifacts carry their own readout sign vector, so the VQC rows trained before then keep serving exactly as trained and pick up the corrected circuit at the next weekly retrain. The same module holds the classical control, `RbfKernelClassifier`: the quantum-kernel classifier with the state-overlap kernel replaced by exp(-gamma |x - y|^2) on the same pipeline - same scaler and PCA to the same width, same class-balanced subsampling, same probability-calibrated, class-balanced SVC - so `ClassicalSVM Model` and `QuantumMetaLearner Model` differ in the kernel and in the SVC settings each is tuned to (`C`, the sample cap, the RBF width: each row has its own `HyperoptQuantum.py` study, because a `C` chosen for the fidelity kernel is not the right `C` for an RBF), not in features, reduction or evaluation - which is the comparison the phase asked for.

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

### Training integration (implemented)

Implemented as three extra entries in `TrainMetaLearner.py`'s variants loop, so the expensive backtest is collected once and all five meta-learners (logistic, gradient boosting, quantum kernel, VQC, classical RBF control) train on the identical table, labels, chronological window and main/special separation. Hyperparameters come from the `quantumKernel_*` / `quantumVqc_*` / `classicalSvm_*` keys in `bestParams_<game>.json` (tuned by `HyperoptQuantum.py` - the control by the same objective on the same day split, so its row is not handicapped against tuned quantum rows - defaults otherwise).

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

### Evaluation metrics

Ticket-level hit counts remain important, but the quantum model must also be evaluated at the per-number probability and ranking levels.


## Randomness-discrimination experiment

A second quantum research track would test whether real draw windows can be distinguished from synthetic fair-draw windows.

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

## Required negative controls

The complete research pipeline tests many models, games, features, windows, and hyperparameters. Apparent improvements can occur by chance, especially when only the best result is reported. The quantum experiments must therefore include the same optimization effort on negative controls.

### Shuffled-history control

Randomly reorder the historical draws and run the same feature engineering, optimization, and evaluation process.
If performance remains similar after shuffling, the model is probably using marginal frequencies, range characteristics, sorted-position distributions, or another non-temporal property rather than learning next-draw dependence.

### Synthetic-fair-history control

Generate many synthetic histories with the same length and rules as the real history. Run the complete hyperparameter optimization and backtest process on each synthetic history.
The relevant null comparison is not one random model. It is:
> The best score found after applying the same complete model-on_the_fly_selection and optimization process to data known to be random.

This estimates how often the research process itself discovers apparently strong models in random data.

### Irrelevant-feature control

Add at least one independently generated random feature. The model should not assign stable predictive importance to that feature across repeated training runs.

### Lockbox period

Reserve the newest historical period as a final untouched test set. Model design, hyperparameters, preprocessing, feature selection, and metric selection must be frozen before this period is evaluated. Implemented (28 Sept 2026) as a hand-written `lockbox.json` at the repository root - `{"from": "2026-07-01", "to": "2026-09-26", "declared": "2026-09-27", "note": "..."}`, dates inclusive, no file meaning no lockbox - honoured by `TrainMetaLearner.py` and `HyperoptQuantum.py` (their tables drop the period before splitting and fitting) and evaluated once with `TrainMetaLearner.py --lockbox-report`; see *Training the meta-learner*.

**Since 26 Oct 2026 the four row tuners honour it too** - `HyperoptStatistics.py`, `HyperoptBoost.py`, `HyperoptEnsemble.py` and `HyperoptRLTicket.py` (ticket and position studies), through `src/Lockbox.py`. Their windows reach back into the period: 22 of the 90 tuning draws of euromillions, lotto and eurodreams, 12-13 of vikinglotto and the week games, 25-26 of the ensemble's ~100 days, 6 of vikinglotto's 40 RL ticket days. Until then two of them only printed a warning and two did not look at all, so every score that decided what was served counted those days, and the lockbox evaluation of the meta-learners rested on rows tuned inside it. Now the lockbox days are withheld from **selection**: the trials, the gate's re-scores of the served parameters and the defaults, the standard error behind any margin, `modelScores`, the Keno subset day table and the boosting pruner's steps. **Fitting** on a history that contains them stays allowed, as in the daily predictor: the statistical and boosting tuners still backtest those days and leave them out of the score, the RL studies still walk and warm-start through them without scoring them (the plain-rule reference skips them too), and the ensemble tuner, which fits nothing, drops their day files on loading. The window is not lengthened, so fewer days are scored and the standard error is wider until the overlap ages out: on 26 Oct 68 of 90 days for euromillions, lotto and eurodreams (lotto's served Markov: SE 0.0765 → 0.0861), 78 for vikinglotto, 78 and 77 for cryptoweek and sharesweek, 74 of lotto's 99 ensemble days, 34 of vikinglotto's 40 RL ticket days - and under the one-standard-error margins fewer replacements are likely. Each game logs `<game>: lockbox ... - N day(s) withheld from tuning, M scored` (a game with nothing left to score is skipped), every gate record carries `lockbox_days_withheld` next to `window_days` (the ensemble's `subsetEnsembleTunedOn` `lockboxDaysWithheld`; the quantum tuner's records carry it too, counting the score-table days taken out before its 75/25 split - its holdout holds none of them - and say so with `lockbox_withheld_from: "table"`), and `TuningControls.py` withholds the same days (*Null controls*). A malformed `lockbox.json` is one `LOCKBOX WARNING` in these tuners and `TuningControls.py`, saying the run is not lockbox-safe, and the run goes on withholding nothing; the trainer and the quantum tuner still stop with `LOCKBOX ERROR`.

Repeatedly inspecting the same holdout period and modifying the model afterward converts the holdout into development data.

## Null controls, and what the best row scores on nothing (implemented)

The two controls above stopped being a plan on 2026-09-21. `src/ControlHistories.py` builds them - **synthetic** (draws generated under the game's own rules, no structure of any kind) and **shuffled** (the real draws reordered, so every marginal survives and only time order dies), both on the real calendar - and two scripts use them. The two market games joined every control on 30 Sept 2026 (their week games on 8 Oct 2026) with their own definition of a fair history - a geometric random walk with the real volatility - and of a shuffled one, whose bins are re-cut from the reordered returns since 26 Oct 2026, so its bin frequencies move a little; both are described under roadmap item 4, M3. Since 27 Sept 2026 both run weekly for every game, as the scheduler's Sunday plan (*The schedule* below) - since 13 Oct 2026 on a fresh block of seeds every week (`--seed weekly`: (ISO year x 100 + ISO week of the weekend's Saturday) x 10000 + 1 on, `ControlHistories.control_seed` - the weekend's week, so the Sunday plan's Monday catch-up keeps its block, and 10,000 wide, so Q2's +1000 null side and the feature control's +1000 noise offsets stay inside it; the same for Q2 and the feature control's noise columns, and five histories instead of three for the four market games, `--market-repetitions 5` - Q2 since 27 Oct 2026 twenty for every game, *The verdict, version 2*; a hand run keeps the reproducible seed 1, the seeds are in every record, and `NullControls.py` removes a game's control folders of earlier seeds). Until then seeds 1-3 rebuilt the same synthetic histories every week, appending only the newest draws, so a second week re-ran the first week's test instead of drawing a new one - and the admin pages read their records (`controls.js`, `data/controls/`): the **Best model per game** card got a *Q0 band* column - what the best of the rows scores on a history with nothing in it, the mean over the control histories plus two standard deviations, taken over the higher of the two controls - and greyed every hits-ranked row at or under it; for the payout games the band is on average hits, not on the profit they rank by, so it is shown as information only. Since 27 Oct 2026 the band labels and greys no row: it is context in the Controls drawer, and every row is tested on its own live record instead (*Q0 - the per-row chance test*, below). The **Randomness watch** carries Q2's verdict next to its tripwires - the best AUC of the suite against the null band and the shuffled control, and a date - and its status turns to *watch* only when the real draws clear both. A third script, `TuningControls.py` (28 Sept 2026), measures the layer above the rows: **the selection effect of tuning**. It runs the statistical tuner's own objectives (`HyperoptStatistics.STRATEGIES`, the robust score, the served Keno sizes) for `-t` trials on a control history and compares the best trial with the untuned defaults scored on the same history - the comparison the champion/challenger gate makes on the real history every week. On a fair synthetic history nothing can be found, so whatever the best of `-t` trials gains there is what choosing alone gives; the gate's real-history gain (`tuningGate[row].challenger - default`) is put next to it, and a real gain inside the control band (mean plus two standard deviations of the control gains) is the effect of choosing, not of finding. Like with like, or not at all: the maximum of n trials grows with n, so `-t` and `-d` default to the weekly tuner's own values (`HyperoptStatistics.DEFAULT_TRIALS` / `DEFAULT_DAYS`), the gate record now carries `run_trials`, `study_trials` and `window_days`, and a record made with other values gets no verdict and says why. Since 26 Oct 2026 the control also withholds the lockbox days the tuner withholds (`HyperoptStatistics.LOCKBOX`; a control history keeps the real dates) and compares `lockbox_days_withheld` as well. A record that withheld a different number gets no verdict: every record made before that date (it withheld none), and, while the window's oldest edge is still inside the lockbox, a record made before the newest draw. Even then the band is a *floor* on the selection effect: the weekly study is warm - its sampler is guided by every completed trial of earlier weeks - while a control study starts cold, and a guided search finds a lucky window more surely than a random one. Hand-run and modest by default (`-g lotto -m both -n 1`; a statistical trial is seconds to a minute; `-n 3` gives the band a width); records in `data/controls/tuning/<game>-<mode>.json`. It takes `process.lock` like the other scripts. Pick3's control is in the same set count the card's average hits use (a box counts three; `Backtester._positional_hits` keeps that historical count for pick3 on purpose), which deliberately differs from the slot-hit objective the tuners score pick3 on; Joker+ is runs on both sides. The two control scripts take `process.lock` like every other entry point, so the home-page banner names them while they run and a hand-run predictor waits for them.

**5 Oct 2026 - two control defects found by the first reading, fixed the same day.** Keno's range in `HyperoptStatistics.GAME_CONFIG` (and in the boosting tuner's mirror of that table, `HyperoptBoost.py`) stood at 1-80, Markov's default, while the real game has drawn 20 of 1-70 since 9 March 2008 - the 58 draws of 2 January to 8 March 2008 are the earlier 1-80 game, 56 of them with numbers above 70. The synthetic keno control therefore drew the wrong game; Q2's keno window features carried ten slots only synthetic draws could fill, and its 4 Oct verdict for keno - *not temporal* in the job's log, shown on the Randomness watch card as *above the band, but no better than the shuffled control*, AUC 1.000 on the real and on the shuffled draws alike - was that artefact, not the data; the random-ticket baseline, drawn from 1-80, scored 4.96 hits on the synthetic control and 5.03 on the shuffled one, where a 1-70 ticket against 1-70 draws scores 5.71 (20 x 20/70); and the meta-learner tables for keno carried ten never-drawn numbers as free negatives, which flatters an AUC. Fixed: the range is 1-70 in both tables; `src/ControlHistories.read_real_rows` drops the draws of an earlier version of a game from every control history and says so (`current_era`: a date before the game's `ERA_START`, or main numbers outside the configured range - keno's 58 draws, nothing in any other game); and the Sunday plan of 11 Oct re-reads keno under the real game. Keno tuning and meta-learner numbers recorded before this date were measured over 80 numbers and are not comparable to later ones in the last decimals; the served meta-learner artifacts carry their own range and are unaffected until the next retrain; the cached keno score table stores no range and is reused as it is. The live `Markov Model` and `LaplaceMonteCarlo Model` keno rows had picked fourteen numbers above 70 in the 120 days before; the one literal 1-80 fallback (`Markov.generate_best_subset`) took the model's range that day, and the rest of the live side - the label space and the data every row loads - followed on 7 Oct 2026 (next paragraph). The second defect: the Sunday market report (`MarketRows.py`) never precomputed the foundation rows before the Backtester forked, so the Chronos row was *unavailable* in every worker and silently absent from `data/controls/markets/<market>-rows.json` (thirty log lines, an empty `errors`); it now calls `ModelFactory.prepare_foundation_scores` like every other collector, records the forecast counts (`foundation_days`), and writes a row that scored no day to `errors` with the reason (a row that also raised keeps its day count), which the market pages list apart from the rows that failed on some days.

**7 Oct 2026 - keno is a 1-70 game everywhere, its subsets follow each row's own probabilities, and the tuner measures every keno bet against its ticket size's fair value.** Three changes from the assessment of `inspiration/Keno_Predictions_Analysis.md`, which proposed one model and one hyperopt per subset size. Its premise - a training loss averaged over the sizes - does not exist in this pipeline: no base row's objective sees a subset. The fair expected loss is the same 46-48 cents per euro at every size, and for per-number probabilities around and above chance the top-k by probability is the expected-value-optimal k-subset for every k (checked by enumeration over every k-subset of a 20-ticket outside the repository; the 3 EUR prize for zero matches at sizes 7-10 would reverse that only for numbers far below chance), so the sizes are slices of one ranking, and what was worth doing was making that ranking honest. First, the range: `Helpers.get_unique_labels` says 1-70 for keno (80 until now - ten dead softmax classes in the deep-learning and boosting rows; the meta-learner tables' ten free negatives went with the 5 Oct `GAME_CONFIG` fix), and `Helpers.load_data` drops the draws of an earlier version of a game (`GAME_ERA_START`, the one table the controls' `current_era` reads too: keno before 9 March 2008), whatever `yearsOfHistory` says. The deep-learning rows' weight fingerprints carry the class count, so the keno rows retrain once in the next daily run; the boosting rows refit daily anyway; the RL ticket policy's five feature weights do not depend on the range and warm-start as on any other day, its candidate set simply shrinking to 1-70; the served meta-learner artifacts carry their own range until the Saturday retrain. Nothing is deleted by hand. The six keno control folders written on 4 Oct still hold 1-80 synthetic draws and fail the 1-70 encoder in a hand run until the Sunday plan rebuilds them, which it does every week before scoring. Second, the subsets: `MarkovBayesian Model` and `MarkovBayesianEnhanched Model` cut their subset with an ascending sort and a constant Markov term, so they served the ticket's *coldest* numbers; `PoissonMonteCarlo Model`, `PoissonMarkov Model` and `LaplaceMonteCarlo Model` drew a weighted random sample tilted towards the ticket's lowest numbers (linspace weights over `set(ticket)`); `Markov Model` and `MarkovMonteCarlo Model` ranked by global frequency, and the tuned `markovSubsetSelectionMode` had been searched for a year and a half without ever being read; the deep-learning rows took a softmax sample at temperature 0.5 and, their per-slot argmax ticket not being de-duplicated, weighed a repeated number twice and could return it twice. Now every row ranks its ticket by its own per-number mass and takes the top k: the Markov chain's untempered column masses (over sorted positions their sum is the chain's probability that a number is drawn at all), the MarkovMonteCarlo row's votes, the MarkovBayesian rows' transition mass from the last draw blended with their frequencies, the two Monte Carlo rows' simulated counts summed over the positions (counted once per placed position), the blend's weights for PoissonMarkov with the sub-models' scores breaking the ties inside a weight class, and the deep-learning rows' per-number probability, de-duplicated (`Helpers.generate_subset_from_scores`, mode `top`; the maximum over the positions until 8 Oct 2026 - see the next paragraph). The `markovSubsetSelectionMode` knob is retired rather than revived: the stored `softmax` at the stored temperature would have sampled the keno subset almost uniformly (the chain's masses span 0.27-0.30 on the real history), the tuner no longer searches it, and the setter stays for the files that carry the key. The Hybrid, foundation and baseline rows already ranked by their own scores, and so do five of the six boosting rows (the `CatBoostMultiLabel Model` samples at its tuned temperature until its tuner says otherwise). Third, the objective: `src/TuningScore.fair_capped_ev` is what a fair k-ticket is worth per euro under the objective's 20 EUR cap (-0.64 at k=5, -0.52 at 6, -0.67 at 7, -0.64 at 8, -0.68 at 9, -0.70 at 10; the raw expectation is -0.46 to -0.48 for every size, but a 6-ticket's prizes mostly sit under the cap and keep their value while a 10-ticket's mostly sit above it and are cut), and every keno bet's capped profit now has that value subtracted, so a fair ticket scores 0 in mean at every size and stored keno scores read alike across sizes and across changes of the served sizes. Within one run every candidate bets the same sizes, so the baseline is a constant there and changes no gate decision; the bound's penalty term still differs between sizes under the null (about -0.17 for a fair 6-ticket over 90 days, -0.10 for a 10-ticket). Keno gate scores and Optuna trial values recorded before this date are on the old scale (the gate compares only this run's candidates, so no decision depends on them); until both the statistical and the boosting tuner have re-scored keno once, a run in which one of them fails leaves `modelScores` on two scales for a week, which would tilt the `WeightedEnsemble` weights by tuner rather than by skill. The raw profit per bet in every record is unchanged, and keno's ranking (Admin › Models) shows profit per bet and bets per ticket size (`profit_by_size`). Not done, by choice: six models per row - nothing to gain, about nine more hours of Saturday tuning (five more copies of keno's 1.9-hour statistical pass of 3 Oct) and six gates per strategy to lose. The served sizes are still `use_10` alone, a value the last statistical tuning save of 23 July 2026 wrote and nothing has written since (the flags changed with every save before that; no tuner suggests them any more): with the fair baseline in place, serving all six sizes costs nothing (the same ticket, sliced) and shows every size on the page - setting `use_5` to `use_9` to true in `bestParams_keno.json` is the owner's call. A per-size hyperopt, if ever, belongs on the slicing read-out (`KenoSubsetTuning`: 15 trials in 28 s on 3 Oct after a 22-minute day-table precompute that six sizes would share), not on the base rows.

**8 Oct 2026 - the per-position rows' keno subsets picked the edges of the range.** The first morning with all six sizes served showed four of the seven deep-learning rows (`Transformer`, `GNN`, `Autoencoder`, `TCN Base`) with the same 5-subset, [1, 4, 63, 66, 70], the other three one or two numbers away from it ([1, 5, 10, 66, 70], [1, 4, 62, 66, 70], [1, 5, 62, 66, 70] - all seven beginning with 1 and ending with 66, 70), and the per-position boosting rows close by ([1, 3, 4, 67, 70] for `XGBoost Model`, [2, 3, 68, 69, 70] for `CatBoost Model`). Within a row that takes its sizes as the top of one ranking the six tickets nest by design - the 6-ticket is the 5-ticket plus the next number, so the per-size profits in the ranking compare the payout table on the same numbers (the `MetaLearner`, `QuantumMetaLearner`, `QuantumVQC`, `ClassicalSVM` and `CatBoostMultiLabel` rows still sample their subsets at their tuned temperature and do not nest) - but the edge numbers were a defect. These rows model the twenty sorted positions, and `Helpers.score_numbers_from_prediction` scored a number by its highest probability at any one position. The sorted positions of a 20-of-70 draw are order statistics: the first slot gives number 1 a probability of 2/7 and the last gives 70 the same, the second slot's best value (4) has 0.13, a middle slot's best about 0.06, so a row that has learned the marginals - all a fair draw lets it learn - always ranks 1, 70 and their neighbours first (within the served ticket 1, 70, 4, 66, 63 - that subset exactly), and the `top` mode of 7 Oct made that deterministic where the softmax sample before it only blurred it. The right per-number score of a per-position model is its inclusion probability, the probability summed over the positions (for one number the positions of a sorted draw are disjoint events): flat 20/70 on a fair table, and whatever the row believes beyond that is what the ranking shows (this morning's stored LSTM softmax spans 0.25-0.33 and its 5-subset under the sum is [1, 16, 32, 42, 60] - 1 stays because that row gives it 0.33). Changed: the deep-learning rows' score sums over the positions (which also reorders the lotto multi-pick of those rows and its three extras: on 3 and 7 Oct the edge numbers 44 and 1 led four of the seven rows, 4 or 45 took a lead place in the other three, and 45 was the first or second extra in six of them), and the boosting rows slice their keno subsets from the same summed inclusion probability (`BoostingBase._inclusion_probabilities` - the ranking `score_numbers()` already gave the meta-learners, on the 0-1 scale the tuned subset temperature was searched against; the averaged score would have made the `softmax` arm of the boosting tuner a near-uniform sample at every temperature in its range). The tuned `top_k` average still refills the ticket's argmax collisions: for the games scored on hits that is what `TopK` was tuned for, for keno it was not - the boosting tuner scores keno by the capped profit of the subsets alone, and until this change those were sliced from the `top_k` ranking, so the keno values of 3 Oct (xgBoostTopK 16, lightGbmTopK 3, catBoostTopK 29) were selected through a read-out that no longer exists, keep ranking the refill by the edge-peaked average, and are re-scored like for like by Saturday's gate. The statistical rows score numbers directly and are unchanged. Two rows keep edge numbers for other reasons and are not fixed by this change: the `WeightedEnsemble Model` votes over every row's twenty-number ticket, and the thirteen per-position tickets all carry 70 and nearly all carry 1 (they are the order-statistic modes), so the vote's ticket and its subsets kept 1 and 70 on every day of 1-8 Oct ([1, 4, 25, 32, 70] on 8 Oct) and will until the vote's membership or weighting changes; and the foundation rows (`Chronos Model` [1, 3, 25, 68, 70], `TimesFM Model` [1, 5, 44, 65, 70] - 1 and 70 in their first subset on nearly every day since 20 Sep) pool the positions already, but their per-slot distributions come from the quantile read-out (`src/foundation/quantile_labels.label_probabilities`), which clips a forecast's out-of-range mass onto the first and last label, so slot 1's tail lands on 1 and slot 20's on 70 - a boundary artefact of that read-out, to be corrected with the meta-learner retrain in view, since `score_numbers()` is also their meta-learner feature. `python3 -m src.Helpers` (in `npm test`) checks the inclusion property on the exact order-statistic table. The deep-learning rows' twenty-number tickets themselves were and are the order-statistic modes to within the near-ties of the middle slots: on 8 Oct five of the seven rows sat within one number of [1, 4, 10, 11, 15, 19, 24, 26, 30, 32, 39, 42, 45, 50, 54, 56, 60, 63, 66, 70] (within two of the exact modes at every position, within three of the `OrderStatistics Baseline`), `LSTM Base` and `UnifiedLstmGruTcn` had the same shape with seven to nine numbers shifted - a per-position model of a fair draw has nothing else to learn, which is why those rows and the vote they dominate (17 of the vote's 20 numbers) look alike, and why a keno row's number-level signal, if there ever is one, will show first in the rows that score numbers directly (the statistical, multi-label boosting and meta-learner rows). The keno subsets of the deep-learning and boosting rows recorded before this date - the History page's per-size profits of 8 Oct and the boosting tuner's keno trial values - were edge tickets; the deep-learning tuner reads its keno subsets through the same call but has no keno study in production (its only study is `jokerplus-lstm_model` of 17 Sep 2026) and clears its scratch cache at every trial. Checked the same day on the 6202 draws of the 1-70 era: number frequencies, pair co-occurrence and the overlap of consecutive draws all sit inside what twenty simulated fair histories produce (number chi-square 42 against 48 ± 9, pair chi-square 2103 against 2185 ± 164, consecutive-draw overlap 5.714 against 5.714 in date order), so there is no joint structure a subset-size-specific model could use that a per-number ranking does not already carry - the nested read-out stays, and the per-size question is answered by the fair capped values, not by six models.

```
    python3 NullControls.py -g lotto -m both -n 3 -d 120
    python3 RandomnessDiscrimination.py -g lotto,pick3,euromillions -w 10 -n 20 --draws 1000
```

**`NullControls.py` (Q0): how good does the BEST row look on data with nothing in it?** The best row of every ranking is a maximum over rows, and the maximum of N noisy estimates sits above their mean even when every row is worthless. Measured on lotto, 120 days x 3 seeds, eight model rows on their untuned defaults:

| control | average row | best row | selection gain | best-to-worst spread | random baseline |
|---|---|---|---|---|---|
| synthetic | 0.781 | 0.883 | **+0.102** | 0.219 | 0.794 |
| shuffled | 0.821 | 0.964 | **+0.143** | 0.289 | 0.794 |

(hits per draw; chance is 6²/45 = 0.8, and the measured random baseline of 0.794 is the apparatus confirming it measures what it claims.)

Three things to read off this. **A different row won almost every run** - PoissonMonteCarlo, XGBoost, Markov on the synthetic histories, PoissonMonteCarlo and PoissonMarkov on the shuffled ones - which is what a leaderboard looks like when there is nothing to lead. **The shuffled control is the harder bar**, not the softer one: it preserves whatever the real history's number frequencies happen to be, and the frequency-shaped models exploit that, so rows score higher there (0.821 average against 0.781) and selection buys more (+0.143 against +0.102). And therefore, on lotto, **a tracked row is not interesting until it beats about 0.96 hits per draw** - not the 0.8 of chance - because that is what the best of eight rows reaches on draws that are known to carry no signal at all.

One thing that looked like a bug and was not: the same winning row reached exactly the same score in two independent control histories (PoissonMonteCarlo at 0.9417 on synthetic seed 1 and shuffled seed 1; PoissonMarkov at 0.975 on two shuffled seeds). Every other row differed substantially in the same pairs - hits over 120 days are multiples of 1/120, so a collision among eleven rows is unremarkable. Worth recording because "identical number across two runs" is otherwise exactly what a silently reused dataset looks like.

**Q0 - the per-row chance test (27 Oct 2026, `src/RowChance.py`; replaces the band as a label).** The band says how good the best of eight untuned rows looks on 120 fair draws; it says nothing about one row on its own record, and greying rows against it mixed the two. The daily report now tests every row on its **live** draws (*The live record*, under *Model performance report*), on the Q0 card's own hit scale. Lottery: the exact chance that fair draws give tickets of the same shapes (6 or 7 numbers, distinct digits, Joker+ runs, numbers that cannot be drawn) at least as many hits - hypergeometric per draw (a distinct-digit law for pick3, the L + R runs for Joker+), convolved over the draws, the ranges held to `HyperoptStatistics.GAME_CONFIG`. Markets: the row's own tickets shuffled among its own live days within blocks of 7 consecutive live days (live weeks for the week games), the outcomes fixed, 20,000 permutations seeded by the record; that removes the stretch's bin mix and its volatility-regime drift, but not a row that times the regime within a block (no direction; the Market rows card against GARCH stays the reference), and with tickets and outcomes both autocorrelated within a block it is slightly anti-conservative. Holm runs over the rows with at least `minDrawsForRanking` live draws (the week games: 10 live weeks, so no family before about mid December 2026). *Under 5%* is always shown next to what chance puts there (the sum of the rows' own discrete levels; rows x 0.05 for the markets), and *holds* is the count still under 5% after the correction - both this week's figures, never a colour. **Fresh evidence is the only thing that counts**: from the first Sunday a row's live record reads p < 0.05 the row is *followed*, and it is judged once, on as many new live draws as flagged it (at least 10, at most 52): *confirmed* at p < 0.01, otherwise *not confirmed*; *running* until then (no p shown), *stopped* after four weeks without a new live draw. Flags and verdicts are kept in `data/database/rowChance.json` (tracked, one line per row, written only when they change): the first run replays every Sunday once (every row flagged on or after the `since.json` freeze: the replay applies the freeze as the reset rule would have, owner 30 Oct 2026), later runs test only new Sundays, a verdict is never recomputed, and the one reset is a later `since.json` freeze, for the rows flagged before it. A lottery game judged without git is tested but stores nothing; one whose liveness reads *unchecked* is not tested and says why. The Controls Q0 tile counts the rows confirmed in the hit-ranked games (keno, pick3 and Joker+ are tested on hits, information only); nothing is greyed on the Models pages any more, and each p stands with its row's live mean, live draw count and null on the same line. `NullControls.py` and its band are unchanged and are context in the Controls drawer only; the weekly job runs on as *Weekly controls: null histories (Q0 band, context)*. The Q0 streak counts the rows confirmed and followed; readings written before 1 Nov 2026 read *band rule*. Cost: about 2 s on the first run (the replay), under 1 s a day after. Replayed on a copy of production on 27 Oct: no row holds and none is confirmed in any game; 5 rows read under 5% across the eleven games against about 10 by chance; the replay without the freeze followed 8 rows and decided 6 *not confirmed* (keno 3, pick3 2, EuroDreams 1); production's first replay (28 Oct) ran so, before the owner decided on 30 Oct that the replay applies the freeze; an entry kept under the freeze without `testFrom` is therefore reset once by the next run (replayed on production's 29 Oct record: the verdicts of 28 Oct on rows flagged before 27 Sep are withdrawn, keno MarkovBayesian is followed from 27 Sep on 215 draws, pick3 LightGBM from 11 Oct, EuroDreams LaplaceMonteCarlo from 25 Oct; pick3 UnifiedLstmGruTcn, flagged on 27 Sep itself, stays *not confirmed*). Self-check: `python3 -m src.RowChance` (in `npm test`). Since 1 Nov 2026 the band itself is refreshed monthly, on the first Sunday of the month (the owner's decision of 30 Oct 2026, *The schedule (`jobs.js`)*): between two first Sundays the Controls drawer and the weekly reading show the latest band with its date.

**`RandomnessDiscrimination.py` (Q2): can anything tell real draw windows from fair simulated ones?** Windows of ten draws become multi-hot counts (per slot for pick3 and Joker+, so digit order survives), and the suite the design asks for - logistic, RBF SVM, random forest, gradient boosting, small NN, quantum kernel, VQC - is trained on a chronological split. Three comparisons run through the identical pipeline, because one number cannot be read on its own: *real vs synthetic* is the question, *synthetic vs synthetic* is the null band (two fair histories, true answer provably 0.5), and *shuffled vs synthetic* is the marginals control (same frequencies, no time order; for the markets since 26 Oct 2026 the same returns, reordered, with re-cut bins - roadmap item 4, M3). Result over 1000 draws and 5 repetitions, as the best AUC of the suite per repetition:

| game | real vs synthetic | null band | shuffled vs synthetic | verdict |
|---|---|---|---|---|
| lotto | 0.588 ± 0.049 | 0.565 ± 0.076 | 0.582 | within the null band |
| euromillions | 0.551 ± 0.050 | 0.544 ± 0.030 | 0.536 | within the null band |
| pick3 | 0.495 ± 0.043 | 0.540 ± 0.045 | 0.534 | within the null band |

**No classifier, classical or quantum, separates real draws from correctly simulated fair ones beyond what the same pipeline scores on two histories that are identical in law.** That is the honest negative result this phase existed to produce, and it is the one the rest of the project should be read against.

Two methodological lessons came out of building it, both worth keeping:

* **The shuffled control was wrong first, and its own null band caught it.** Permuting the whole 1995-2026 history and then taking its newest 600 draws samples every rule era the game ever had: per-number counts came out with a standard deviation of 13.5 against the real 9.1 (chi-square p < 0.0001), so the control no longer shared the marginals it exists to preserve - and it "separated" better than the real history. The window is cut *before* the shuffle now, and the two match exactly.
* **The verdict itself fell into the trap it was built to expose.** Comparing the best AUC over seven classifiers and five repetitions against the *mean* of the null comparison's individual runs made euromillions read as evidence (0.637 against a 0.598 threshold). Best-of-seven is biased upwards even when all seven are worthless, so it has to be held against the null's own best-of-seven: 0.551 ± 0.050 against 0.544 ± 0.030, which is nothing. `--reanalyze` recomputes the verdicts of saved records under the current rule, so a corrected rule can be applied without paying for the run twice.

**The verdict, version 2 (27 Oct 2026; first weekly run Sunday 1 Nov 2026).** On unchanged histories 6 of 11 Q2 labels changed between 18 and 25 Oct, for two reasons: the "null mean + 2 sd" of 3 seeds (5 for the markets) is itself luck - lotto's 3-seed thresholds came out 0.570, 0.572 and 0.751, against 0.686 ± 0.021 on 20 seeds - and *too few draws* came from the run itself (a band reaching 0.8), so it flipped with the seeds. Now: **one power rule**, in `controlRules.json` at the repository root, read by both `RandomnessDiscrimination.py` and `controls.js` (`{"q2": {"few_windows": 250}}`): a record with fewer held-out test windows reads *too few draws*, a property of the history's length, `--draws` and `--window` fixed before any classifier runs. Today that is EuroDreams and Viking Lotto (150 windows), the shares week (208) and the crypto week (108); they keep running, so their numbers stay on the admin pages, and every record stores the `few_windows` it was judged by. **The bounds come from the seed count**: *above* needs the real mean (best AUC of the suite, mean over seeds) to clear the null mean + k sd, the one-sided 97.725% prediction bound (the coverage of "+ 2 sd") for one more fair run, k = t(n-1) x √(1+1/n) - 5.23 at 3 seeds, 2.43 at 10, 2.19 at 20 - and the shuffled control's own bound worked out the same way from its seeds. The real mean is held against a single-run bound because averaging seeds removes only the synthetic side's noise, not the history's own; a paired test by seed was measured and left out (anti-conservative as a test of time order, and for the markets it would have been the only gate). A comparison that scored fewer than 2 seeds reads *failed*: no number, never *within*. **Twenty seeds a week for every game** (`-n 20`, the default now, no `--market-repetitions`; about 80 minutes). The record stores `version` 2, `status`, `few_windows`, `windows_test`, `band_coverage`, `band_k`, `shuffled_k`, `null_threshold` ± `null_threshold_se`, `shuffled_threshold`, `seeds_scored`, `rule` and `power_rule`, never a NaN. Measured on scratch (designer pools, 4,000 resampled weeks of 20 seeds): no false *above* on the seven lottery games, 0.06 label changes a week over the eleven games (all of them shares), crypto *above* in 99.7% of weeks; end to end on the 26 Oct histories, lotto 0.573 against a threshold of 0.686 ± 0.021 reads *within*, crypto 0.938 against 0.622, with its shuffled bound at 0.899, *above*. Older (version 1) records keep what their flags said, but the windows rule applies to them too, so the 25 Oct EuroDreams *above* (150 windows) reads *too few draws*. Weekly readings store `q2.rule`; a Q2 run of weeks restarts at a change of rule, and the first week says the week before was read under the old one. Self-check: `RandomnessDiscrimination._self_check()` (in `npm test`).

Limits, stated because they bound what the numbers mean: the models run on untuned defaults, so this measures selection over rows and not the selection over hyperparameters that a full hyperopt on a control history would add (that is the remaining Q0 item, and it makes these figures a lower bound); a synthetic history reproduces today's rules over the whole calendar, so the default window is restricted to the most recent draws where the rules are stable; and the foundation rows are excluded unless `--foundation` is passed, because each control history would need its own forecast pass.

**`IrrelevantFeatureControl.py` (negative control, since 28 Sept 2026): does any meta-learner give a random column stable importance?** The roadmap's last missing control. Every meta-learner variant `TrainMetaLearner.py` serves (logistic, gradient boosting, quantum kernel, VQC and the classical RBF control - `meta_learner_variants`, the same list with the same tuned parameters) is refitted on the table the Saturday chain cached - with the lockbox days removed first, exactly as the trainer removes them, so it is the trainer's table - with three **noise columns** appended: shuffled copies of real base-model columns, so each has exactly a base model's marginal distribution and no relation to the label or the day. Three repeats with fresh noise, the trainer's own chronological 80/20 day split, and **permutation importance** on both sides of the split - the ranking power lost when one column is scrambled, the drop in |AUC − 0.5| rather than the signed AUC drop, because a Platt-scaled SVM fitted on noise often comes out with probabilities inverted against the labels, and there a signed drop would read a column the model leans on as ignored (found in review; the self-check shows an inverted model with the upright one's importances). On the training rows it answers "does the model lean on noise?": a variant **fits noise** when a noise column matters in training at least a tenth as much as an average real column and the pooled noise importance lies more than two standard errors above zero (a ratio rather than a share of the total, so one strongly informative base model cannot dilute it; the share is reported too - with three noise columns among eleven, a model that cannot tell noise from signal lands near 27%, one that ignores it at 0) - and then its weights on the real columns are not evidence of anything either. On the held-out days the noise columns' importance is the **feature-level null band** (mean + 2 sd, the rule the other controls use), and the card lists the base models whose held-out importance clears it: the columns that carry more than noise does. With eight real columns tested against a two-sigma band, one clears it by chance about one week in five, so a column counts when it stays above the band week after week, not when it appears once. The held-out AUC of the fit without noise, next to the mean with noise, is what the extra columns cost. It runs in the Sunday controls plan after Q2, refits only - the table is never recollected here, a cache up to seven draws behind the data file is accepted and the shortfall recorded; about an hour for the seven games, almost all of it the four VQC refits per game (a keno-sized VQC fit at its tuned 120 epochs takes about 160 s, every other variant seconds) - writes `data/controls/features/<game>.json`, and `controls.js` puts the verdicts on the *Feature control* card of Admin › Controls. The rules live in `src/FeatureControl.py`; its self-check (`python3 -m src.FeatureControl`, in `npm test`) shows a logistic regression on plenty of rows ignoring the noise and an unpruned tree on 300 rows fitting it.

## Search versus prediction

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

## Within-draw structure versus temporal predictability

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

## Proposed implementation phases

### Phase Q0: classical controls first

Before interpreting any quantum comparison:

- Actually tune and run `TCN.py` as a classical baseline.
- Remove or correctly wire the dead LSTM layer-count configuration.
- Add synthetic-fair-history benchmarks.
- Add shuffled-history benchmarks.
- Add per-number probability and ranking metrics.
- Establish an untouched chronological lockbox period.

### Phase Q1: quantum meta-learning (implemented)

- Reuse `TrainMetaLearner.py` score matrices.
- Add `QuantumMetaLearner Model` as a separate tracked row.
- Start with four training-only reduced features and four qubits.
- Compare with logistic regression, gradient boosting, and classical-kernel SVM.
- Persist preprocessing and model metadata with the quantum artifact.
- Keep the model opt-in during development.

### Phase Q2: randomness discrimination

- Generate synthetic histories using exact period-specific game rules.
- Build real-versus-synthetic window features.
- Compare classical and quantum classifiers.
- Repeat the experiment across multiple synthetic seeds and chronological splits.
- Investigate any stable distinguishability before treating it as predictive evidence.

### Phase Q3: direct quantum scoring

- Produce a quantum-assisted score for every candidate number.
- Reuse the existing ticket construction, special-column handling, Keno subset generation, Backtester, and UI tracking.
- Compare calibration and ranking before comparing final ticket hits.

### Phase Q4: candidate search

- Define a ticket-level score and threshold.
- Build a toy Grover oracle for a deliberately small candidate space.
- Compare quantum search cost with direct classical ranking.
- Include circuit depth, native two-qubit gate count, noise sensitivity, and data-loading cost.

## Suggested project structure

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

## Reporting requirements

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

## Research interpretation

The quantum extension is intended as an adversarial randomness and model-capability benchmark.

A defensible conclusion can state:

```text
Under the tested data representation, model family, optimization budget,
and chronological evaluation period, the quantum-assisted model did or
did not detect reproducible structure beyond the selected classical baselines.
```

It should not state that failure to detect structure proves perfect randomness, or that a small retrospective uplift proves future lottery predictability.

All quantum experiments remain subject to the repository's education and research disclaimer. Simulated quantum circuits run on classical hardware and do not demonstrate quantum computational advantage.

## Out of date (check of 5 Nov 2026)

The text above was moved as it stood. Comparing it with the code on 5 Nov 2026 found these points no longer true; they are listed rather than rewritten so nothing changes meaning unseen:

- 'eight base models' / 'seven scores' are outdated (Chronos is a base model too); there are three kernel/variational variants (QuantumMetaLearner, QuantumVQC and the ClassicalSVM control).
- HyperoptQuantum tunes pick3 on the slot-hit lower bound since 14 Oct 2026, not on the pick3 payout.
- The randomness-discrimination experiment and the negative controls read as proposals but are built and run every Sunday (`jobs.js`; NullControls monthly since 1 Nov 2026).
- The lockbox example dates are illustrative: the real `lockbox.json` holds 2025-12-01 to 2026-02-28.
- `current_era` also cuts lotto before 2011-10-01 (6 of 42 until then), not only keno; keno serves all six sizes since 8 Oct 2026.
- *Proposed implementation phases*: Q0 is done except wiring or removing the dead LSTM layer-count setting; the Q1 flags are on by default; Q2 is built (answer so far: no difference found); Q3 and Q4 are not built. The *Suggested project structure* (`src/quantum/`, `experiments/`) was not built.
