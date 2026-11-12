import os, argparse, json, sys, time, functools, math, copy, inspect
# Pin the BLAS pools before numpy is imported: with trials running in parallel
# processes (HyperoptRunner) each would otherwise spin up a 16-thread OpenBLAS
# pool; the statevector batches are small, one thread per trial is the fast
# configuration, and the Backtester workers of collect_score_table need it too.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import cpu_count
import numpy as np
from art import text2art

from sklearn.metrics import roc_auc_score

from src.Backtester import Backtester
from src.DataLoader import DataLoader
from src.Helpers import Helpers
from src.HyperoptRunner import open_study, fail_stale_running_trials, optimize_study, install_sigterm_handler
from src.ModelFactory import BASE_MODEL_NAMES, build_models, expected_model_names, prepare_foundation_scores
from src.QuantumModels import fit_quantum_kernel, fit_quantum_vqc, fit_classical_svm
from src.Lockbox import load as load_lockbox, split_rows as lockbox_split, describe as describe_lockbox
from src.TuningGate import challenge, make_defaults, evaluate_reference
from src import TuningResult as tres
from src.TuningScore import score_bets_by_day, attrs_for_trial

# Reuse the per-game min/max/draw_size/skip_last_columns/special_column_count
# table and the (day, number) -> [scores, label] table builder instead of
# redefining them - TrainMetaLearner.py trains the quantum artifacts on
# exactly these, so tuning against a private copy could silently drift. The
# same goes for the positional games' (Pick3, Joker+) positional table,
# per-position fit and argmax-ticket scoring: the trial classifier must be
# fitted and played exactly the way the persisted artifact will be.
from HyperoptStatistics import GAME_CONFIG
from src.MarketData import WEEK_MARKETS
from TrainMetaLearner import (
    build_training_table, load_meta_score_table, save_meta_score_table, meta_table_kind, meta_table_start_index,
    build_positional_training_table, fit_position_models, evaluate_positional_holdout, positional_profit,
    positional_positions, POSITIONAL_CLASSES,
)

helpers = Helpers()

LOCK_FILE = os.path.join(os.getcwd(), "process.lock")

# A game needs at least this many collected backtest days before the 75/25
# day split leaves a holdout worth scoring - below that the mean-per-day
# objective is mostly draw luck.
MIN_EVALUATION_DAYS = 10

# Fraction of the collected days used to fit each trial's classifier; the
# remaining most-recent days are the scored holdout. More holdout share than
# fit_meta_model's 80/20 sanity check on purpose: a tuner RANKS trials by this
# number, so it needs a steadier holdout signal, not just a printed check.
TRAIN_DAY_FRACTION = 0.75


def is_running():
    """
    Checks if another instance is running based on the lock file.

    The PID written into the lock is verified to still be alive: a lock whose
    owner is gone is stale - left behind by a crashed or killed run (a hung
    2026-08-17 run held the lock for two days and silently blocked every cron
    start after it). A stale lock is removed and treated as not running.
    """
    if not os.path.exists(LOCK_FILE):
        return False
    try:
        with open(LOCK_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # signal 0 = existence check only, nothing is sent
        return True
    except (ValueError, ProcessLookupError):
        print("Removing stale lock file (owner process no longer exists)")
        remove_lock()
        return False
    except PermissionError:
        # Process exists but belongs to another user - definitely running.
        return True

def create_lock():
    """Creates the lock file."""
    try:
        with open(LOCK_FILE, "x") as f:  # "x" mode creates the file, failing if it exists
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False

def remove_lock():
    """Removes the lock file."""
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass

def print_intro():
    ascii_art = text2art("Quantum Hyperopt")
    print("============================================================")
    print("Quantum Hyperopt")
    print("Licence : MIT License")
    print(ascii_art)
    print("Find best parameters for the quantum meta-learner models and their classical SVM control")


# The encoding-scale grid shared by both quantum variants (see
# suggest_quantum_kernel). 0.1 to 2.0 by 0.05: 39 points, the old 0.5-2.0
# by 0.25 grid a subset of it.
ENCODING_SCALE_LOW = 0.1
ENCODING_SCALE_HIGH = 2.0
ENCODING_SCALE_STEP = 0.05


def suggest_quantum_kernel(trial):
    """
    Search space for QuantumKernelClassifier. The suggest names ARE the
    bestParams keys fit_quantum_kernel reads (see src/QuantumModels.py), so
    study.best_params can be merged into bestParams_<game>.json verbatim.
    nQubits stays a single-value categorical: 4 qubits (16 amplitudes) is the
    README quantum track's deliberate starting width, but routing it through
    Optuna anyway makes the tuned value land in the json for TrainMetaLearner
    to read and turns a future widening into a one-list change here.

    encodingScale (the kernel bandwidth, see README) searches 0.1-2.0 in
    steps of 0.05 since 28 Sept 2026 - it was 0.5-2.0 by 0.25 and the tuned
    values sat on its lower edge. Every old grid value lies on the new grid,
    and Optuna continues the persisted studies: a parameter whose
    distribution changed is sampled independently of the earlier trials.
    """
    return {
        "quantumKernel_nQubits": trial.suggest_categorical("quantumKernel_nQubits", [4]),
        "quantumKernel_encodingLayers": trial.suggest_int("quantumKernel_encodingLayers", 1, 3),
        "quantumKernel_encodingScale": trial.suggest_float("quantumKernel_encodingScale", ENCODING_SCALE_LOW,
                                                           ENCODING_SCALE_HIGH, step=ENCODING_SCALE_STEP),
        "quantumKernel_C": trial.suggest_float("quantumKernel_C", 0.1, 100, log=True),
        "quantumKernel_maxTrainSamples": trial.suggest_categorical("quantumKernel_maxTrainSamples", [1000, 2000, 4000]),
    }


def suggest_quantum_vqc(trial):
    """
    Search space for VariationalQuantumClassifier - same key-name contract as
    suggest_quantum_kernel. epochs steps by 30 rather than searching finely:
    the training loss surface is noisy enough that neighboring epoch counts
    are indistinguishable, and coarse steps keep the trial budget on the
    parameters that actually move the score (layers, scale, learning rate).
    """
    return {
        "quantumVqc_nQubits": trial.suggest_categorical("quantumVqc_nQubits", [4]),
        "quantumVqc_numLayers": trial.suggest_int("quantumVqc_numLayers", 1, 4),
        "quantumVqc_encodingScale": trial.suggest_float("quantumVqc_encodingScale", ENCODING_SCALE_LOW,
                                                        ENCODING_SCALE_HIGH, step=ENCODING_SCALE_STEP),
        "quantumVqc_learningRate": trial.suggest_float("quantumVqc_learningRate", 0.005, 0.1, log=True),
        "quantumVqc_epochs": trial.suggest_int("quantumVqc_epochs", 30, 120, step=30),
        "quantumVqc_batchSize": trial.suggest_categorical("quantumVqc_batchSize", [64, 128, 256]),
    }


def suggest_classical_svm(trial):
    """
    Search space for the quantum kernel's classical control
    (RbfKernelClassifier), same key-name contract. nFeatures mirrors the
    quantum rows' nQubits and stays 4 so the two rows keep reducing to the
    same width; gamma is the RBF width (sklearn's "scale" default is about
    1 / n_features on standardised features, so 0.25 sits inside the range),
    C and the sample cap the same ranges as the quantum kernel. RBF fits on
    at most 4000 rows take seconds, so this study costs a fraction of the
    quantum ones.
    """
    return {
        "classicalSvm_nFeatures": trial.suggest_categorical("classicalSvm_nFeatures", [4]),
        "classicalSvm_gamma": trial.suggest_float("classicalSvm_gamma", 0.01, 10.0, log=True),
        "classicalSvm_C": trial.suggest_float("classicalSvm_C", 0.1, 100, log=True),
        "classicalSvm_maxTrainSamples": trial.suggest_categorical("classicalSvm_maxTrainSamples", [1000, 2000, 4000]),
    }


# The champion/challenger gate (src/TuningGate.py, since 14 Oct 2026 for these
# studies too - the last tuner that wrote its run's best trial ungated): a
# run's best trial is written only if it beats the served parameters AND the
# untuned defaults, both re-scored on this run's holdout with this objective;
# the defaults are served back when they beat both. QUANTUM_DEFAULTS is what
# fit_quantum_kernel / fit_quantum_vqc / fit_classical_svm serve when
# bestParams_<game>.json has none of their keys (src/QuantumModels.py), the
# "default" reference. classicalSvm_gamma's code default is sklearn's "scale"
# (1 / (n_features x the features' variance)), which a FixedTrial cannot hold
# for a float parameter, so 0.25 - "scale" on four standardised components of
# unit variance - stands in for it; the two differ by the PCA components'
# variance, a reference, not the served value. quantumVqc_epochs 80 is off
# the study's 30-step grid; FixedTrial uses it as it is.
QUANTUM_DEFAULTS = {
    "quantum_kernel": {"quantumKernel_nQubits": 4, "quantumKernel_encodingLayers": 2, "quantumKernel_encodingScale": 1.0,
                       "quantumKernel_C": 1.0, "quantumKernel_maxTrainSamples": 2000},
    "quantum_vqc": {"quantumVqc_nQubits": 4, "quantumVqc_numLayers": 2, "quantumVqc_encodingScale": 1.0,
                    "quantumVqc_learningRate": 0.05, "quantumVqc_epochs": 80, "quantumVqc_batchSize": 128},
    "classical_svm": {"classicalSvm_nFeatures": 4, "classicalSvm_gamma": 0.25, "classicalSvm_C": 1.0,
                      "classicalSvm_maxTrainSamples": 2000},
}
# the served rows the gate's records are written under (bestParams["tuningGate"][row])
VARIANT_ROWS = {"quantum_kernel": "QuantumMetaLearner Model", "quantum_vqc": "QuantumVQC Model",
                "classical_svm": "ClassicalSVM Model"}
GATE_ENABLED = True
GATE_MARGIN = 0.0
# The challenger must also win by GATE_SE_MARGIN standard errors of its own
# score (TuningGate.challenge se_margin, as the boosting tuner since 13 Oct
# 2026): a run's best of n trials sits above a single reference by selection
# alone. Both objectives carry a standard error since 14 Oct 2026 - the
# positional score through TuningScore, the hits score as the standard
# error of the per-day hits.
# 2.5 since 27 Oct 2026 (1 until then): a quantum score is one fit, and a
# fit is mostly its seed - the training subsample (always active: 5,310-
# 12,390 table rows against maxTrainSamples of 1,000-4,000) and, for the
# VQC, its initial angles and batch order. One configuration refitted with
# other seeds on production's 25 Oct tables moved by a median 0.88 SE
# (kernel), 0.97 (SVM) and 0.84 (VQC), while seed-averaged configurations
# differed by 0-0.59 SE; the defaults moved 1.28 SE from one week to the
# next. So at one SE the gate replaced on seed luck: 34-46% of the rows a
# week in a simulation on fair-game data, and the 15 replacements of 18 Oct
# lost a median 1.41 SE when re-scored on 25 Oct (the three that had won by
# 2 SE or more, 3.44 - lotto QuantumMetaLearner from +2.36 SE to -5.21).
# At 2.5 SE the same simulation replaces 2-5% per row-week - still seed luck
# when it happens (no configuration predicts a fair draw), but rarely. The
# revert keeps the challenger's margin (revert_margin None): a margin-free
# revert here would compare two single fits, a coin flip on the seed.
GATE_SE_MARGIN = 2.5

# The variety rule (owner, 12 Nov 2026: the same gate everywhere, the weekly chain included). A trial's holdout
# prediction IS a ticket per holdout day - the top draw_size numbers of the day's ranking (the ticket Predictor plays
# from a meta-learner, sorted: a set of numbers) on the per-number games, the argmax ticket (in slot order) on Pick3
# and Joker+ - so the draws measure applies: a parameter set whose classifier gave the same ticket on VARIETY_LINE or
# more of the holdout days is input-blind (src/TuningResult.variety_of_tickets), neither written nor served back, and
# it blocks nothing (TuningGate.challenge(variety=True)). A classifier that ignores the base models' scores ranks the
# numbers the same way every day, which is exactly what this catches. Not on the market games (crypto, shares): their
# argmax "ticket" is a bin per instrument, a position that may rightly stay the same through a trend - their records
# carry no input_blind and the rule changes nothing there. main() sets the line (--variety-line, else
# config/tuning.json, else 0.95); tune_game(variety_line=None) leaves it off.
VARIETY_LINE = tres.DEFAULT_VARIETY_LINE
# What the trial processes and the gate's reference children report as "current" in the result file (12 Nov 2026):
# set in the parent before a reference is forked (the child inherits it), "trial" otherwise. A quantum fit takes
# seconds to minutes on a fixed table, so "current" is per trial / per reference, without a draw (day/days None);
# the score table's backtest, before the studies, reports its days as role "table".
CURRENT_ROLE = "trial"
RESULT_THROTTLE_SECONDS = 5.0

# (study suffix, factory bound to that variant's bestParams keys, search space)
VARIANTS = [
    ("quantum_kernel", fit_quantum_kernel, suggest_quantum_kernel),
    ("quantum_vqc", fit_quantum_vqc, suggest_quantum_vqc),
    # the classical control is tuned by the same objective on the same split,
    # so its row is not handicapped against tuned quantum rows
    ("classical_svm", fit_classical_svm, suggest_classical_svm),
]


def collect_score_table(dataset_name, game_cfg, path, days_back, lockbox=None, progress=None):
    """
    Same data-collection pass as TrainMetaLearner.train_meta_learner: backtest
    the 8 base models with collect_scores=True over the last days_back draws.
    Run ONCE per game and cached across the three studies - the backtest is
    the expensive part (XGBoost genuinely trains per day) while the tunables
    only affect the quantum classifier fitted on top, so re-collecting per
    trial or per study would multiply minutes of work for identical tables.
    Returns (results, model_names, main_actual_key, lockbox_withheld) or None
    when the game has no usable data; lockbox_withheld is how many table days
    the lockbox took out (0 without one), for the gate's records.

    The positional games (Pick3, Joker+ - Helpers.is_positional_game) collect
    the POSITIONAL table instead (positional base models, game=<name> so the
    Backtester stores each model's per-slot "_position_scores" next to
    "actual_ordered"), cached under their own meta_position_table_<game>.joblib
    so it can never be confused with a flat per-number table; the label key
    is "actual_ordered".

    progress (12 Nov 2026, the Hyperopts page): a callable (day, days) called as the backtest completes its days -
    the Backtester's progress_callback; nothing is called for a cached table.
    """
    is_positional = helpers.is_positional_game(dataset_name)
    table_kind = meta_table_kind(dataset_name)

    dataPath = os.path.join(path, "data", "trainingData", dataset_name)
    bestParamsPath = os.path.join(path, f"bestParams_{dataset_name}.json")
    bestParams = {}
    if os.path.exists(bestParamsPath):
        with open(bestParamsPath, "r") as infile:
            bestParams = json.load(infile)

    loader = DataLoader()
    loader.setDataPath(dataPath)
    loader.setGameRange(game_cfg["min"], game_cfg["max"])
    loader.setDrawSize(game_cfg["draw_size"])

    numbers, _, _ = loader.load_numbers(skipLastColumns=game_cfg["skip_last_columns"])
    total_rows = len(numbers)
    if total_rows == 0:
        print(f"No data found for {dataset_name}, skipping.")
        return None

    # the trainer's history floor (13 Oct 2026): without it a 300-day table of a 306-draw game started at row 6
    start_index = meta_table_start_index(total_rows, days_back)

    # Shared with TrainMetaLearner.py (same cache helpers): whichever script
    # collects first persists the table, the other reuses it - see the cache
    # validity rules next to load_meta_score_table.
    cached = load_meta_score_table(path, dataset_name, days_back, total_rows, bestParams, table_kind,
                                   model_names=expected_model_names(dataPath, bestParams, is_positional=is_positional, game=dataset_name))
    if cached is not None:
        results, model_names = cached
    else:
        models = build_models(dataPath, bestParams, is_positional=is_positional, game=dataset_name)
        model_names = [name for name in BASE_MODEL_NAMES if name in models]

        # Foundation models cannot run inside the Backtester's forked pool
        # (see ModelFactory.prepare_foundation_scores): every day is
        # forecast here, once, before the fork.
        prepare_foundation_scores(
            models, start_index, total_rows,
            skipLastColumns=game_cfg["skip_last_columns"],
            specialColumnCount=game_cfg["special_column_count"],
            label=f"{dataset_name}: ")

        backtester = Backtester(loader)
        for name, model in models.items():
            backtester.add_model(name, model)

        print(f"\n{dataset_name}: backtesting {total_rows - start_index} days with {len(models)} base models to collect training data...")
        results = backtester.backtest(
            start_index=start_index,
            end_index=total_rows,
            skipLastColumns=game_cfg["skip_last_columns"],
            special_column_count=game_cfg["special_column_count"],
            include_baselines=False,
            collect_scores=True,
            verbose=True,
            game=dataset_name if is_positional else None,
            progress_callback=(None if progress is None else
                               (lambda day, _rows, days=total_rows - start_index: progress(day, days)))
        )
        if results:
            save_meta_score_table(path, dataset_name, results, model_names, days_back, total_rows, bestParams, table_kind)

    if not results:
        print(f"No backtest rows produced for {dataset_name}, skipping.")
        return None

    # The lockbox (src/Lockbox.py): its days are never tuned on. They leave
    # the table here, before either 75/25 day split - the flat one below and
    # the positional one of Pick3 and Joker+, which returns early.
    locked = []
    if lockbox:
        results, locked = lockbox_split(results, list(getattr(loader, "dates", [])), lockbox)
        print(f"{dataset_name}: {describe_lockbox(lockbox)} - {len(locked)} table day(s) withheld from tuning, "
              f"{len(results)} remain")
        if not results:
            print(f"No backtest rows remain for {dataset_name} outside the lockbox, skipping.")
            return None

    if is_positional:
        return results, model_names, "actual_ordered", len(locked)

    # Tune on the MAIN-number table only, labeled with the same actual_main
    # key subtlety TrainMetaLearner relies on: for special-column games the
    # sorted "actual" merges main and special values whose ranges overlap
    # (e.g. Euromillions main 1-50 and star 1-12 both contain 9), which would
    # mislabel special-only numbers as drawn mains. The special-column table
    # is deliberately NOT scored here: bestParams holds one quantum parameter
    # set per game (TrainMetaLearner fits both the main and special artifacts
    # from it), the main table carries far more days x numbers of signal, and
    # the ticket-level hits metric below only exists for the main draw.
    main_actual_key = "actual_main" if game_cfg["special_column_count"] > 0 else "actual"
    return results, model_names, main_actual_key, len(locked)


def build_day_split(results, model_names, game_cfg, main_actual_key):
    """
    Honest chronological split for tuning, per the README's walk-forward rule
    (results must hold up when training only ever precedes scoring). The split
    is over DAYS, not flattened table rows: every Backtester row is one
    backtest day (order preserved - Backtester uses pool.imap, not
    imap_unordered, the same guarantee fit_meta_model's own holdout leans on),
    and each day expands to (max - min + 1) table rows, so splitting the
    flattened table instead could cut a day in half and leak part of the
    holdout's draw into training. The trial classifier is fitted on the early
    portion only - scaler/PCA included, since QuantumModels refits them inside
    fit() on exactly the data it is given - and never refitted on the full
    window here; the full-window refit is TrainMetaLearner's job once the
    tuned params are persisted.
    """
    split_index = int(len(results) * TRAIN_DAY_FRACTION)
    train_days, test_days = results[:split_index], results[split_index:]

    X_train, y_train = build_training_table(
        train_days, model_names, game_cfg["min"], game_cfg["max"],
        scores_suffix="_scores", actual_key=main_actual_key)
    X_test, y_test = build_training_table(
        test_days, model_names, game_cfg["min"], game_cfg["max"],
        scores_suffix="_scores", actual_key=main_actual_key)

    return X_train, y_train, X_test, y_test, len(test_days)


def build_positional_day_split(results, model_names, positions):
    """
    The positional games' version of build_day_split: the same chronological
    TRAIN_DAY_FRACTION split over whole DAYS, on the positional table. Each
    day is a fixed block of positions x 10 rows there (Pick3 3 x 10 = 30,
    Joker+ 6 x 10 = 60, see build_positional_training_table), so splitting
    the Backtester rows first and building the two tables from them keeps
    every holdout day's slots together - a flat-row split could put one slot
    of a draw in training and its others in the holdout. The holdout days'
    drawn-order results come back alongside, because the objective scores the
    argmax ticket with the real payout table against the actual draw, not
    against 0/1 labels.
    """
    split_index = int(len(results) * TRAIN_DAY_FRACTION)
    train_days, test_days = results[:split_index], results[split_index:]

    X_train, y_train = build_positional_training_table(train_days, model_names, positions, POSITIONAL_CLASSES)
    X_test, y_test = build_positional_training_table(test_days, model_names, positions, POSITIONAL_CLASSES)
    test_actual_ordered = [row["actual_ordered"] for row in test_days]

    return X_train, y_train, X_test, y_test, test_actual_ordered


def score_holdout(model, X_test, y_test, n_test_days, draw_size):
    """
    Mean per-day hits of the top-draw_size ranked numbers is the primary
    score because that IS the ticket Predictor actually plays from a
    meta-learner: rankByModel sorts the number range by predict_proba[:, 1]
    and the ticket is rankedNumbers[:draw_size], so a hyperparameter set only
    deserves a better score if it puts more actually-drawn numbers into that
    played ticket. AUC alone would happily reward a model that orders the
    mid-field well while misordering the handful of top slots the ticket is
    built from. AUC still enters as a +0.01*auc tie-breaker: per-day hits are
    integers, so the mean is quantized to 1/n_test_days steps and small
    holdouts produce many exact ties - the tiny AUC term breaks those ties
    toward the better-calibrated model without ever outweighing a single real
    hit (0.01*auc <= 0.01 < 1/n_test_days for any holdout under 100 days).

    Returns (value, mean_hits, auc, per-day hits, per-day tickets) - the tickets (12 Nov 2026, the variety rule) as
    sorted indices into the number range, one per holdout day.
    """
    proba = model.predict_proba(X_test)[:, 1]

    # build_training_table emits exactly (max - min + 1) rows per day, in day
    # order, so the flat holdout reshapes cleanly into per-day blocks
    numbers_per_day = len(y_test) // n_test_days
    day_proba = proba.reshape(n_test_days, numbers_per_day)
    day_labels = y_test.reshape(n_test_days, numbers_per_day)

    hits, tickets = [], []
    for day in range(n_test_days):
        top = np.argsort(day_proba[day])[::-1][:draw_size]
        hits.append(day_labels[day, top].sum())
        # the day's ticket as positions in the number range (min_number + index is the number), sorted: a set
        tickets.append(sorted(int(i) for i in top))
    mean_hits = float(np.mean(hits))

    # a single-class holdout has no defined AUC; 0.5 keeps the tie-breaker
    # neutral instead of erroring the whole trial
    auc = roc_auc_score(y_test, proba) if len(set(y_test.tolist())) > 1 else 0.5

    return mean_hits + 0.01 * auc, mean_hits, auc, [float(h) for h in hits], tickets


def hits_standard_error(hits):
    """The standard error of the per-day hits (sample std / sqrt(days)), as TuningScore reports score_se; None under two days."""
    if len(hits) < 2:
        return None
    return float(np.std(np.asarray(hits, dtype=float), ddof=1) / math.sqrt(len(hits)))


def report_trial_start(trial, result_file, study_key):
    """The result file's "current" (12 Nov 2026): this trial, or the gate reference CURRENT_ROLE names - no draw."""
    if result_file:
        tres.safe_write_result(result_file, study_key, current=tres.current(CURRENT_ROLE, None, None, trial=trial.number))


def objective_quantum(trial, suggest_func, fit_func, study_label,
                      X_train, y_train, X_test, y_test, n_test_days, draw_size, variety_line=None, result_file=None):
    """
    One trial: fit the variant on the early-day table with the suggested
    params, score the late-day holdout. Both quantum classes seed their own
    RNG from a fixed random_state default, so identical params always produce
    identical scores - trial differences come from the hyperparameters, not
    sampling luck (same reasoning as the Backtester's per-day reseed).
    Per-trial elapsed is printed because these fits are the budget: a study
    must stay at seconds-to-minutes per trial to fit runHyperopt.sh's weekly
    window, and the timing makes a runaway configuration visible in the log.

    variety_line (12 Nov 2026): the record also carries the draws variety fields of the holdout tickets (VARIETY_LINE's
    comment); result_file: "current" is written at the trial's start (study_label is the result file's key).
    """
    trialStart = time.time()
    report_trial_start(trial, result_file, study_label)

    params = suggest_func(trial)
    model = fit_func(X_train, y_train, params=params)
    value, mean_hits, auc, hits, tickets = score_holdout(model, X_test, y_test, n_test_days, draw_size)
    # what the score was made of, for the gate's record (TuningGate.challenge reads "tuning"), and the standard
    # error of the per-day hits - the unit of the gate's margin (GATE_SE_MARGIN)
    tuning = {"score": float(value), "kind": "holdout_hits", "days": int(n_test_days),
              "hits_mean": float(mean_hits), "auc": float(auc), "score_se": hits_standard_error(hits)}
    if variety_line is not None:
        tuning.update(tres.variety_of_tickets(tickets, variety_line))
    trial.set_user_attr("tuning", tuning)

    print(f"{study_label} trial {trial.number}: hits/day={mean_hits:.3f} "
          f"auc={auc:.4f} value={value:.4f} ({time.time() - trialStart:.1f}s)")
    return value


def objective_quantum_positional(trial, suggest_func, fit_func, study_label,
                                 X_train, y_train, X_test, test_actual_ordered, positions, game, slot_bound=True,
                                 variety_line=None, result_file=None):
    """
    Positional-game trial: fit one classifier per position on the early days
    (the suggested params bound with functools.partial, the same way
    TrainMetaLearner binds them for the persisted artifact) and play the
    argmax ticket on every holdout day.

    Pick3 and Joker+ (slot_bound=True, since 14 Oct 2026) are scored with
    the statistical tuner's positional objective (src/TuningScore.
    score_bets_by_day, positional=True): the lower confidence bound over the
    holdout days of the ticket's hits - Pick3's digits in the right slot,
    Joker+'s paid runs (Helpers.jokerplus_runs, L + R, the Backtester's own
    count for the game) - plus POSITIONAL_PROFIT_WEIGHT x the bound of the
    ticket's capped real payout (TrainMetaLearner.positional_profit:
    Helpers.pick3_ticket_profit, Helpers.jokerplus_ticket_profit) as the
    tie-breaker. Until then the objective was the mean raw payout per day
    (+0.01 x top-1 accuracy): the payout sits at -stake on almost every day
    with rare spikes (+46 for a Pick3 pair, +676 or +756 for a straight -
    every bet of the ticket pays), so one lucky ticket in a 15-75 day
    holdout decided a study - and, once gated, the gate. The raw payout per
    day and the top-1 accuracy stay on the record; the score's standard
    error is the gate's margin unit.

    The market games (crypto, shares; slot_bound=False) keep their
    objective: the mean per day of the fixed rule's P&L in bin space
    (Helpers.market_ticket_profit, bounded by a day's move - no lucky-strike
    problem) + 0.01 x top-1 accuracy, now with the standard error of the
    per-day P&L on the record for the gate's margin. Their studies keep
    their names.

    Joker+ is scored DIGITS-ONLY here: the positional table has the 6 digit
    slots and no sign, so the argmax ticket is a 6-int ticket and
    jokerplus_ticket_profit treats the sign as unknown (no 1.50 EUR sign
    refund, no 200,000 EUR jackpot tier). The zodiac special_model is not
    tuned by this study - like the other special-column games, bestParams
    holds one quantum parameter set per game and TrainMetaLearner fits both
    the positional and the special artifact from it.

    variety_line (12 Nov 2026): Pick3 and Joker+ records also carry the draws variety fields of the holdout's argmax
    tickets (never a market's: tune_game passes none there); result_file: "current" at the trial's start.
    """
    trialStart = time.time()
    report_trial_start(trial, result_file, study_label)

    params = suggest_func(trial)
    position_models = fit_position_models(
        X_train, y_train, functools.partial(fit_func, params=params), positions, POSITIONAL_CLASSES)
    mean_profit, accuracies, tickets = evaluate_positional_holdout(
        position_models, X_test, test_actual_ordered, positions, POSITIONAL_CLASSES, game)
    mean_accuracy = float(np.mean(accuracies))

    actual = [[int(digit) for digit in day[:positions]] for day in test_actual_ordered]
    played = [[int(digit) for digit in ticket] for ticket in tickets]
    profits = [positional_profit(game, ticket, day) for ticket, day in zip(played, actual)]
    accuracy_text = "/".join(f"{accuracy:.3f}" for accuracy in accuracies)
    if not slot_bound:
        value = float(mean_profit) + 0.01 * mean_accuracy
        trial.set_user_attr("tuning", {"score": value, "kind": "holdout_profit", "days": len(profits),
                                       "profit_per_day": float(mean_profit), "top1_accuracy": mean_accuracy,
                                       "score_se": hits_standard_error(profits)})
        print(f"{study_label} trial {trial.number}: profit/day={mean_profit:.3f} "
              f"top1 acc per pos={accuracy_text} (mean {mean_accuracy:.4f}) value={value:.4f} "
              f"({time.time() - trialStart:.1f}s)")
        return value
    if helpers.is_jokerplus(game):
        hits_by_day = [sum(helpers.jokerplus_runs(ticket, day)) for ticket, day in zip(played, actual)]
    else:
        hits_by_day = [sum(1 for t, a in zip(ticket, day) if t == a) for ticket, day in zip(played, actual)]
    tuning = score_bets_by_day([[profit] for profit in profits], hits_by_day, payout=False, positional=True)
    tuning.update({"profit_per_day": float(mean_profit), "top1_accuracy": mean_accuracy})
    attrs = attrs_for_trial(tuning)
    if variety_line is not None:
        attrs.update(tres.variety_of_tickets(played, variety_line))
    trial.set_user_attr("tuning", attrs)
    value = tuning["score"]

    hits_text = f"{tuning['hits_mean']:.3f}" if tuning.get("hits_mean") is not None else "-"
    print(f"{study_label} trial {trial.number}: {'paid runs' if helpers.is_jokerplus(game) else 'slot hits'}/day={hits_text} "
          f"(bound {value:.4f}) profit/day={mean_profit:.3f} top1 acc per pos={accuracy_text} (mean {mean_accuracy:.4f}) "
          f"({time.time() - trialStart:.1f}s)")
    return value


def study_name(dataset_name, variant_name):
    """A variant's Optuna study, which is also its key in the result file (the Pick3 and Joker+ studies started afresh
    on 14 Oct 2026: their earlier trials are raw payouts, which would guide the sampler on another scale than the hit
    bound)."""
    slot_bound = helpers.is_positional_game(dataset_name) and not helpers.is_market_game(dataset_name)
    return f"{dataset_name}-{variant_name}{'_slot_bound' if slot_bound else ''}"


def _optimize(studyName, optunaDatabase, objective, n_trials, parallel, trialTimeout, on_trial_done):
    """optimize_study, with the result file's per-trial hook when this checkout's HyperoptRunner offers one."""
    kwargs = {}
    if on_trial_done is not None and "on_trial_done" in inspect.signature(optimize_study).parameters:
        kwargs["on_trial_done"] = on_trial_done
    optimize_study(studyName, optunaDatabase, objective, n_trials, parallel=parallel,
                   expected_trial_gb=0.5, trial_timeout_seconds=trialTimeout, **kwargs)


def tune_game(dataset_name, game_cfg, path, days_back, n_trials, parallel, trialTimeout, optunaDatabase, lockbox=None,
              variants=None, variety_line=None, result_file=None):
    """
    One game: collect (or reuse) its score table, split it by day, run the
    three studies and put each through the gate. Extracted from __main__ on
    14 Oct 2026 so a scratch root can exercise it; __main__ only parses the
    arguments and loops over the games.

    12 Nov 2026 (the Hyperopts page): `variants` the variant names to tune (None: all three - quantum_kernel,
    quantum_vqc, classical_svm); `variety_line` the variety rule's line (None: off; main() always passes one; never
    applied on a market game); `result_file` the page's JSON file (None: nothing written). Each selected variant's
    study is "running" from the start of the game (the score table's backtest is part of its work: "current" role
    "table" with its days), "skipped" with the reason when the game cannot be tuned, then per study the trials'
    progress, "gate", "done" with what the decision wrote, or "failed".
    """
    print(f"\n{dataset_name.capitalize()}")
    # Positional games (Pick3, Joker+) are tuned on the positional
    # table (one classifier per slot = draw_size slots,
    # argmax-ticket profit objective) instead of the per-number
    # ranking every other game uses.
    is_positional = helpers.is_positional_game(dataset_name)
    positions = positional_positions(game_cfg) if is_positional else None
    # Pick3 and Joker+ are scored by the hit bound (objective_quantum_positional); the market games, positional
    # too (an instrument per slot), keep their P&L objective - the switch of 14 Oct 2026 was about lucky payouts
    slot_bound = is_positional and not helpers.is_market_game(dataset_name)
    # variety: never on a market game (VARIETY_LINE's comment)
    varietyLine = None if helpers.is_market_game(dataset_name) else variety_line
    selected = [v for v in VARIANTS if variants is None or v[0] in variants]
    studyKeys = {v[0]: study_name(dataset_name, v[0]) for v in selected}

    def write_all(**fields):
        for key in studyKeys.values():
            tres.safe_write_result(result_file, key, **fields)

    def skip(reason):
        print(f"Skipping {dataset_name}: {reason}")
        write_all(status="skipped", current=None, decision=None, reason=reason, finishedAt=tres.now_iso())

    if result_file:
        for variant_name, _, _ in selected:
            tres.safe_write_result(result_file, studyKeys[variant_name], tuner="quantum", game=dataset_name,
                                   row=VARIANT_ROWS[variant_name], variant=variant_name, status="running",
                                   wanted=int(n_trials), days=int(days_back), gate=bool(GATE_ENABLED),
                                   varietyLine=varietyLine, startedAt=tres.now_iso(), current=None)
    tableWritten = [0.0]

    def table_progress(day, days):
        now = time.time()
        if now - tableWritten[0] >= RESULT_THROTTLE_SECONDS or day >= days:
            tableWritten[0] = now
            write_all(current=tres.current("table", day, days))

    try:
        collected = collect_score_table(dataset_name, game_cfg, path, days_back, lockbox=lockbox,
                                        progress=table_progress if result_file else None)
    except Exception as e:
        write_all(status="failed", current=None, error=str(e)[:500])
        raise
    if collected is None:
        skip("no usable data or no backtest rows (outside the lockbox)")
        return
    results, model_names, main_actual_key, lockboxWithheld = collected

    if len(results) < MIN_EVALUATION_DAYS:
        skip(f"only {len(results)} backtest days collected "
             f"(need at least {MIN_EVALUATION_DAYS} for a meaningful 75/25 day split)")
        return

    test_actual_ordered = None
    if is_positional:
        X_train, y_train, X_test, y_test, test_actual_ordered = build_positional_day_split(
            results, model_names, positions)
        n_test_days = len(test_actual_ordered)
    else:
        X_train, y_train, X_test, y_test, n_test_days = build_day_split(
            results, model_names, game_cfg, main_actual_key)

    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        # can't fit (or rank) a binary classifier on one class -
        # only plausible with a degenerate/truncated data file
        skip("train or holdout labels contain a single class")
        return

    print(f"Tuning on {len(results) - n_test_days} train days / {n_test_days} holdout days "
          f"({len(X_train)} / {len(X_test)} table rows)")
    write_all(current=None, holdoutDays=int(n_test_days), trainDays=int(len(results) - n_test_days))

    jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
    existingData = {}
    if os.path.exists(jsonBestParamsFilePath):
        with open(jsonBestParamsFilePath, "r") as infile:
            existingData = json.load(infile)

    for variant_name, fit_func, suggest_func in selected:
        studyName = studyKeys[variant_name]
        try:
            tune_variant(dataset_name, variant_name, fit_func, suggest_func, studyName, existingData,
                         jsonBestParamsFilePath, is_positional, slot_bound, positions, X_train, y_train, X_test, y_test,
                         n_test_days, test_actual_ordered, game_cfg, lockboxWithheld, n_trials, parallel, trialTimeout,
                         optunaDatabase, varietyLine, result_file)
        except Exception as e:
            tres.safe_write_result(result_file, studyName, status="failed", current=None, error=str(e)[:500])
            raise


def tune_variant(dataset_name, variant_name, fit_func, suggest_func, studyName, existingData, jsonBestParamsFilePath,
                 is_positional, slot_bound, positions, X_train, y_train, X_test, y_test, n_test_days,
                 test_actual_ordered, game_cfg, lockboxWithheld, n_trials, parallel, trialTimeout, optunaDatabase,
                 varietyLine, result_file):
    """One variant's study, its gate and the write (tune_game's loop body until 12 Nov 2026)."""
    global CURRENT_ROLE
    study = open_study(studyName, optunaDatabase, parallel=parallel)
    fail_stale_running_trials(study)
    # the trials that existed before this run: the gate judges only this run's (TuningGate.run_trials)
    known_trials = {t.number for t in study.get_trials(deepcopy=False)}
    # what this study's decision changes, for the result file (TuningResult.writes_between)
    servedBefore = copy.deepcopy(existingData)

    if is_positional:
        objective = lambda trial, suggest_func=suggest_func, fit_func=fit_func, studyName=studyName: \
            objective_quantum_positional(
                trial, suggest_func, fit_func, studyName,
                X_train, y_train, X_test, test_actual_ordered, positions, dataset_name, slot_bound=slot_bound,
                variety_line=varietyLine, result_file=result_file
            )
    else:
        objective = lambda trial, suggest_func=suggest_func, fit_func=fit_func, studyName=studyName: \
            objective_quantum(
                trial, suggest_func, fit_func, studyName,
                X_train, y_train, X_test, y_test, n_test_days, game_cfg["draw_size"],
                variety_line=varietyLine, result_file=result_file
            )

    def report_progress():
        progressStudy = open_study(studyName, optunaDatabase, parallel=parallel, quiet=True)
        tres.safe_write_result(result_file, studyName, progress=tres.trial_progress(
            progressStudy.get_trials(deepcopy=False), known_trials))

    if result_file:
        report_progress()
    studyStart = time.time()
    # The score table above is inherited by every trial
    # process through the fork - collected once, fitted many
    # times in parallel.
    CURRENT_ROLE = "trial"
    _optimize(studyName, optunaDatabase, objective, n_trials, parallel, trialTimeout,
              report_progress if result_file else None)
    print(f"Study {studyName} finished in {time.time() - studyStart:.1f}s")
    study = open_study(studyName, optunaDatabase, parallel=parallel, quiet=True)

    gateEvaluate = None
    if result_file:
        tres.safe_write_result(result_file, studyName, status="gate", current=None,
                               progress=tres.trial_progress(study.get_trials(deepcopy=False), known_trials))

        def gateEvaluate(params, role):
            # TuningGate's own default (evaluate_reference in a forked child): the child inherits CURRENT_ROLE, and its
            # objective writes it as "current"
            global CURRENT_ROLE
            CURRENT_ROLE = role
            try:
                return evaluate_reference(objective, params, trialTimeout + 60, f"{studyName} {role}")
            finally:
                CURRENT_ROLE = "trial"

    # The gate (src/TuningGate.py): this run's best trial - never
    # study.best_params, whose trials were scored on other weeks'
    # holdout windows - against the served parameters and the untuned
    # defaults, both re-scored on this run's holdout by this objective.
    # Until 14 Oct 2026 this run's best trial was written as it was.
    # The margin is GATE_SE_MARGIN (2.5 SE since 27 Oct 2026), and the
    # revert takes the same one (revert_margin None, see GATE_SE_MARGIN).
    # The variety rule (12 Nov 2026, VARIETY_LINE) on the lottery games.
    outcome = challenge(study, known_trials, objective, existingData,
                        make_defaults(QUANTUM_DEFAULTS[variant_name], existingData),
                        timeout_seconds=trialTimeout + 60, label=studyName,
                        margin=GATE_MARGIN, enabled=GATE_ENABLED, window_days=n_test_days, se_margin=GATE_SE_MARGIN,
                        lockbox_withheld=lockboxWithheld, revert_margin=None, evaluate=gateEvaluate,
                        variety=varietyLine is not None,
                        variety_rule=tres.variety_rule(varietyLine, "draws") if varietyLine is not None else None)
    print(outcome["summary"])
    # The lockbox days left the score table before the 75/25 split (collect_score_table, 26 Oct 2026), so the
    # count is the table's: the holdout - window_days - is cut from the days left and holds none of them, unlike
    # a row tuner's window, which the withheld days are counted out of. The record says which.
    outcome["record"]["lockbox_withheld_from"] = "table"
    if variant_name == "classical_svm":
        # what the "default" reference was scored with, since it is not what an untuned row serves (see QUANTUM_DEFAULTS)
        outcome["record"]["default_note"] = "classicalSvm_gamma 0.25 stands in for sklearn's \"scale\" in the default reference, and is served on a revert"
    existingData.setdefault("tuningGate", {})[VARIANT_ROWS[variant_name]] = outcome["record"]
    if outcome["params"] is not None:
        existingData.update(outcome["params"])

    # Written after EACH study (not once per game) so a crash
    # in the second study can't throw away the first's result.
    with open(jsonBestParamsFilePath, "w+") as outfile:
        json.dump(existingData, outfile, indent=4)

    if result_file:
        # "Apply this run's best": the challenger trial's parameters - the suggest names ARE the bestParams keys and
        # nothing is derived from them. None when the gate wrote it itself, or no trial that is not input-blind scored.
        challengerParams = None
        if outcome.get("trial") is not None and outcome["decision"] != "replaced":
            try:
                challengerParams = dict(study.trials[outcome["trial"]].params)
            except Exception:
                challengerParams = None
        writes = tres.writes_between(servedBefore, existingData, rows=[VARIANT_ROWS[variant_name]])
        tres.safe_write_result(result_file, studyName, status="done", current=None, decision=outcome["decision"],
                               reason=outcome["reason"], summary=outcome["summary"], record=outcome["record"],
                               challengerParams=challengerParams,
                               progress=tres.trial_progress(study.get_trials(deepcopy=False), known_trials),
                               finishedAt=tres.now_iso(), **writes)


def _self_check():
    """
    The variety rule and the result file (12 Nov 2026), on a hand-made score table (no backtest, no quantum fit):
    python3 -c 'import HyperoptQuantum; HyperoptQuantum._self_check()'
    - score_holdout / objective_quantum: a classifier that ranks the numbers the same way every day is input-blind,
      one that follows the base models' scores is not, and without a line there are no variety fields;
    - tune_game on a scratch root with a result file (collect_score_table and VARIANTS replaced by hand-made ones, two
      trials, one at a time): running -> done with the record, the rows and what the decision wrote; with the stuck
      classifier every trial is blind and the decision is "kept"; a table too short is "skipped".
    """
    import contextlib
    import io
    import tempfile
    import optuna

    module = globals()
    cfg = GAME_CONFIG["lotto"]
    rng = np.random.RandomState(5)
    names = ["A Model", "B Model"]

    def table(days):
        rows = []
        for _ in range(days):
            drawn = sorted(int(n) for n in rng.choice(np.arange(cfg["min"], cfg["max"] + 1), cfg["draw_size"], replace=False))
            rows.append({**{f"{name}_scores": {n: float(rng.rand()) for n in range(cfg["min"], cfg["max"] + 1)} for name in names},
                         "actual": drawn})
        return rows

    class Fixed:
        """predict_proba from the first feature (seeing) or from the number's place in the range only (stuck)."""
        def __init__(self, stuck):
            self.stuck = stuck

        def predict_proba(self, X):
            width = cfg["max"] - cfg["min"] + 1
            p = np.tile(np.linspace(0.1, 0.9, width), len(X) // width) if self.stuck else 1 / (1 + np.exp(-X[:, 0]))
            return np.column_stack([1 - p, p])

    def fit_of(stuck):
        return lambda X, y, params=None: Fixed(stuck)

    X_train, y_train, X_test, y_test, n_test = build_day_split(table(40), names, cfg, "actual")
    for stuck in (True, False):
        trial = optuna.trial.FixedTrial(QUANTUM_DEFAULTS["classical_svm"])
        with contextlib.redirect_stdout(io.StringIO()):
            objective_quantum(trial, suggest_classical_svm, fit_of(stuck), "self-check", X_train, y_train, X_test, y_test,
                              n_test, cfg["draw_size"], variety_line=0.95)
        tuning = trial.user_attrs["tuning"]
        assert tuning["input_blind"] is stuck and tuning["tickets"] == n_test and tuning["kind"] == "holdout_hits", tuning
    trial = optuna.trial.FixedTrial(QUANTUM_DEFAULTS["classical_svm"])
    with contextlib.redirect_stdout(io.StringIO()):
        objective_quantum(trial, suggest_classical_svm, fit_of(True), "self-check", X_train, y_train, X_test, y_test,
                          n_test, cfg["draw_size"])
    assert "input_blind" not in trial.user_attrs["tuning"], "no line, no variety fields"

    saved = (module["collect_score_table"], module["VARIANTS"])
    try:
        for kind, days in (("stuck", 40), ("seeing", 40), ("short", 6)):
            rows = table(days)
            module["collect_score_table"] = lambda *a, rows=rows, **k: (rows, names, "actual", 0)
            module["VARIANTS"] = [("classical_svm", fit_of(kind == "stuck"), suggest_classical_svm)]
            with tempfile.TemporaryDirectory() as root:
                resultPath = os.path.join(root, "results", "run.json")
                with contextlib.redirect_stdout(io.StringIO()):
                    tune_game("lotto", cfg, root, 300, 2, 1, 60, f"sqlite:///{os.path.join(root, 'db.sqlite3')}",
                              variants=["classical_svm"], variety_line=0.95, result_file=resultPath)
                with open(resultPath) as handle:
                    entry = json.load(handle)["studies"][study_name("lotto", "classical_svm")]
                assert entry["tuner"] == "quantum" and entry["row"] == "ClassicalSVM Model" and entry["varietyLine"] == 0.95, entry
                if kind == "short":
                    assert entry["status"] == "skipped" and "backtest days" in entry["reason"], entry
                    continue
                with open(os.path.join(root, "bestParams_lotto.json")) as handle:
                    written = json.load(handle)
                assert entry["status"] == "done" and entry["current"] is None and entry["wanted"] == 2, entry
                assert entry["record"] == written["tuningGate"]["ClassicalSVM Model"] and "ClassicalSVM Model" in entry["rows"]
                assert entry["progress"]["done"] == 2 and entry["finishedAt"], entry["progress"]
                if kind == "stuck":
                    assert entry["decision"] == "kept" and "input-blind" in entry["reason"], entry["reason"]
                    assert entry["record"]["variety"]["blind_trials"] and entry["challengerParams"] is None, entry
                    assert entry["params"] == {}, entry["params"]
                else:
                    assert entry["record"]["variety"]["blind_trials"] == [], entry["record"]["variety"]
                    if entry["decision"] != "replaced":
                        assert set(entry["challengerParams"]) == set(QUANTUM_DEFAULTS["classical_svm"]), entry["challengerParams"]
    finally:
        module["collect_score_table"], module["VARIANTS"] = saved
    print("HyperoptQuantum self-check OK: a classifier stuck on one ranking is input-blind and never written; the result "
          "file goes running -> done with what the decision wrote, or skipped with the reason")


if __name__ == "__main__":
    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)

    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)

    install_sigterm_handler()

    try:
        try:
            helpers.git_pull()
        except Exception as e:
            print("Failed to get latest changes")

        parser = argparse.ArgumentParser(
            prog='Sequence Predictor',
            description='Tries to predict a sequence of numbers',
            epilog='Check it out'
        )

        # 150 days: enough holdout days (~37) that the hits-per-day mean is
        # not owned by a couple of lucky draws, while the one-off score
        # collection stays well under TrainMetaLearner's 300-day pass.
        # Default matches TrainMetaLearner's 300-day window on purpose: this
        # script runs first in runHyperopt.sh and persists its collected score
        # table, so collecting the full window here means the meta-learner
        # retrain minutes later reuses it instead of redoing the pipeline's
        # most expensive stage (the doubled collection cost was flagged in
        # review). Bonus: tuning holdout grows from ~37 to ~75 days.
        parser.add_argument('-d', '--days', type=int, default=300)
        parser.add_argument('-t', '--trials', type=int, default=15)
        parser.add_argument(
            '--parallel-trials', type=int, default=0,
            help='Upper bound on trials evaluated at the same time, each in its own process. 0 '
                 '(default) = auto: one per core (a quantum fit is single-threaded numpy), launched '
                 'only while the memory gate allows (see HyperoptRunner.optimize_study). 1 = off.')
        parser.add_argument(
            '--trial-timeout', type=int, default=1800,
            help='Wall-clock budget in seconds per trial; a trial over budget is killed and recorded '
                 'as pruned. Fits are seconds to minutes, this catches a runaway qubit/sample count.')
        parser.add_argument('-s', '--save', type=helpers.str2bool, default=True)
        parser.add_argument(
            '--gate-margin', type=float, default=0.0,
            help="How much a run's best trial must beat the served parameters and the untuned defaults by, "
                 "re-scored on this run's holdout, before it is written (src/TuningGate.py)")
        parser.add_argument('--no-gate', action='store_true', help="Write each study's best trial of this run without the comparison")
        parser.add_argument(
            '--gate-se-margin', type=float, default=GATE_SE_MARGIN,
            help="The margin in standard errors of the challenger's own score; the larger of this and --gate-margin "
                 "applies. 2.5 since 27 Oct 2026 (1 until then): a single fit moves about 0.9 SE with its seed alone")
        parser.add_argument(
            '-v', '--variants', type=str, default=",".join(v[0] for v in VARIANTS),
            help='Comma-separated variants (studies) to tune: ' + ", ".join(
                f"{v[0]} ({VARIANT_ROWS[v[0]]})" for v in VARIANTS) + '. Default: all three.')
        parser.add_argument(
            '--result-file', type=str, default=None,
            help="Also write each study's progress and the gate's verdict - with what it would write - to this JSON "
                 "file (src/TuningResult). The Hyperopts page runs the tuner this way with -s false (12 Nov 2026).")
        parser.add_argument(
            '--variety-line', type=float, default=None,
            help="The variety rule's line (owner, 12 Nov 2026): a parameter set whose classifier gives the same ticket "
                 "on this share of the holdout days or more is input-blind - neither written nor served back (lottery "
                 "games only). 0.5-1.0 (or 50-100); default the owner's setting in config/tuning.json, else 0.95.")
        parser.add_argument(
            '-g', '--games',
            type=str,
            # The week games (cryptoweek, sharesweek) are left out by default: three
            # studies per game on a chain step that already takes seven hours, for
            # rows with a few hundred weeks of history. Name them with -g to tune them.
            default=",".join(g for g in GAME_CONFIG if g not in WEEK_MARKETS),
            help='Comma-separated list of games, e.g. "keno,lotto"'
        )

        args = parser.parse_args()

        print_intro()

        days_back = int(args.days)
        n_trials = int(args.trials)
        pushToGit = bool(args.save)
        cores = max(1, cpu_count() - 1)
        parallel = cores if int(args.parallel_trials) <= 0 else max(1, min(int(args.parallel_trials), cores))
        trialTimeout = int(args.trial_timeout)
        GATE_ENABLED = not args.no_gate
        GATE_MARGIN = float(args.gate_margin)
        GATE_SE_MARGIN = float(args.gate_se_margin)
        resultFile = args.result_file or None
        VARIETY_LINE = (tres.clamp_line(args.variety_line) if args.variety_line is not None
                        else tres.load_variety_line(os.getcwd()))
        print(f"Variety line: {tres.line_text(VARIETY_LINE)} of the holdout days on one ticket = input-blind "
              f"(lottery games)")
        variants = [v.strip() for v in args.variants.split(',') if v.strip()]
        unknown_variants = [v for v in variants if v not in VARIANT_ROWS]
        if unknown_variants:
            print(f"Unknown variant(s), ignoring: {unknown_variants}")
        variants = [v for v in variants if v in VARIANT_ROWS]
        print("Selected variants:", variants)

        print("Push to git: ", pushToGit)
        print("Running ", n_trials, "trials")

        games = [g.strip() for g in args.games.split(',') if g.strip()]
        unknown_games = [g for g in games if g not in GAME_CONFIG]
        if unknown_games:
            print(f"Unknown game(s), ignoring: {unknown_games}")
        print("Selected games:", games)

        path = os.getcwd()
        # A malformed lockbox.json must stop the run here, before any backtest
        # is spent - not fail every game inside its own try and exit 0.
        try:
            lockbox = load_lockbox(path)
        except ValueError as e:
            print(f"LOCKBOX ERROR: {e}")
            sys.exit(1)
        if lockbox:
            print(f"Tuning under the {describe_lockbox(lockbox)}: its days are withheld from every table")
        optunaDatabase = "sqlite:///db.sqlite3"

        for dataset_name, game_cfg in GAME_CONFIG.items():
            if dataset_name not in games:
                continue
            try:
                tune_game(dataset_name, game_cfg, path, days_back, n_trials, parallel, trialTimeout, optunaDatabase, lockbox,
                          variants=variants, variety_line=VARIETY_LINE, result_file=resultFile)
            except Exception as e:
                print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")

        try:
            if pushToGit:
                helpers.git_push(commit_message="Saving latest quantum hyperopt")
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()
