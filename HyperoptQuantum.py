import os, argparse, json, sys, time, functools
from datetime import datetime
import optuna
import numpy as np
from art import text2art

from sklearn.metrics import roc_auc_score

from src.Backtester import Backtester
from src.DataLoader import DataLoader
from src.Helpers import Helpers
from src.ModelFactory import BASE_MODEL_NAMES, build_models
from src.QuantumModels import fit_quantum_kernel, fit_quantum_vqc

# Reuse the per-game min/max/draw_size/skip_last_columns/special_column_count
# table and the (day, number) -> [scores, label] table builder instead of
# redefining them - TrainMetaLearner.py trains the quantum artifacts on
# exactly these, so tuning against a private copy could silently drift. The
# same goes for Pick3's positional table, per-position fit and argmax-ticket
# scoring: the trial classifier must be fitted and played exactly the way the
# persisted artifact will be.
from HyperoptStatistics import GAME_CONFIG
from TrainMetaLearner import (
    build_training_table, load_meta_score_table, save_meta_score_table, meta_table_kind,
    build_positional_training_table, fit_position_models, evaluate_positional_holdout,
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
    print("Find best parameters for the quantum meta-learner models")


def suggest_quantum_kernel(trial):
    """
    Search space for QuantumKernelClassifier. The suggest names ARE the
    bestParams keys fit_quantum_kernel reads (see src/QuantumModels.py), so
    study.best_params can be merged into bestParams_<game>.json verbatim.
    nQubits stays a single-value categorical: 4 qubits (16 amplitudes) is the
    README quantum track's deliberate starting width, but routing it through
    Optuna anyway makes the tuned value land in the json for TrainMetaLearner
    to read and turns a future widening into a one-list change here.
    """
    return {
        "quantumKernel_nQubits": trial.suggest_categorical("quantumKernel_nQubits", [4]),
        "quantumKernel_encodingLayers": trial.suggest_int("quantumKernel_encodingLayers", 1, 3),
        "quantumKernel_encodingScale": trial.suggest_float("quantumKernel_encodingScale", 0.5, 2.0, step=0.25),
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
        "quantumVqc_encodingScale": trial.suggest_float("quantumVqc_encodingScale", 0.5, 2.0, step=0.25),
        "quantumVqc_learningRate": trial.suggest_float("quantumVqc_learningRate", 0.005, 0.1, log=True),
        "quantumVqc_epochs": trial.suggest_int("quantumVqc_epochs", 30, 120, step=30),
        "quantumVqc_batchSize": trial.suggest_categorical("quantumVqc_batchSize", [64, 128, 256]),
    }


# (study suffix, factory bound to that variant's bestParams keys, search space)
VARIANTS = [
    ("quantum_kernel", fit_quantum_kernel, suggest_quantum_kernel),
    ("quantum_vqc", fit_quantum_vqc, suggest_quantum_vqc),
]


def collect_score_table(dataset_name, game_cfg, path, days_back):
    """
    Same data-collection pass as TrainMetaLearner.train_meta_learner: backtest
    the 8 base models with collect_scores=True over the last days_back draws.
    Run ONCE per game and cached across both quantum studies - the backtest is
    the expensive part (XGBoost genuinely trains per day) while the tunables
    only affect the quantum classifier fitted on top, so re-collecting per
    trial or per study would multiply minutes of work for identical tables.
    Returns (results, model_names, main_actual_key) or None when the game has
    no usable data.

    Pick3 collects the POSITIONAL table instead (is_pick3=True base models,
    game="pick3" so the Backtester stores each model's per-slot
    "_position_scores" next to "actual_ordered"), cached under its own
    meta_position_table_pick3.joblib so it can never be confused with a flat
    per-number table; its label key is "actual_ordered".
    """
    is_pick3 = dataset_name == "pick3"
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

    start_index = max(0, total_rows - days_back)

    # Shared with TrainMetaLearner.py (same cache helpers): whichever script
    # collects first persists the table, the other reuses it - see the cache
    # validity rules next to load_meta_score_table.
    cached = load_meta_score_table(path, dataset_name, days_back, total_rows, bestParams, table_kind)
    if cached is not None:
        results, model_names = cached
    else:
        models = build_models(dataPath, bestParams, is_pick3=is_pick3)
        model_names = [name for name in BASE_MODEL_NAMES if name in models]

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
            game="pick3" if is_pick3 else None
        )
        if results:
            save_meta_score_table(path, dataset_name, results, model_names, days_back, total_rows, bestParams, table_kind)

    if not results:
        print(f"No backtest rows produced for {dataset_name}, skipping.")
        return None

    if is_pick3:
        return results, model_names, "actual_ordered"

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

    return results, model_names, main_actual_key


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


def build_positional_day_split(results, model_names):
    """
    Pick3's version of build_day_split: the same chronological
    TRAIN_DAY_FRACTION split over whole DAYS, on the positional table. Each
    day is a fixed block of 30 rows there (3 positions x 10 digits, see
    build_positional_training_table), so splitting the Backtester rows first
    and building the two tables from them keeps every holdout day's three
    slots together - a flat-row split could put one slot of a draw in training
    and its other two in the holdout. The holdout days' drawn-order results
    come back alongside, because the objective scores the argmax ticket with
    the real payout table against the actual draw, not against 0/1 labels.
    """
    split_index = int(len(results) * TRAIN_DAY_FRACTION)
    train_days, test_days = results[:split_index], results[split_index:]

    X_train, y_train = build_positional_training_table(train_days, model_names)
    X_test, y_test = build_positional_training_table(test_days, model_names)
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
    """
    proba = model.predict_proba(X_test)[:, 1]

    # build_training_table emits exactly (max - min + 1) rows per day, in day
    # order, so the flat holdout reshapes cleanly into per-day blocks
    numbers_per_day = len(y_test) // n_test_days
    day_proba = proba.reshape(n_test_days, numbers_per_day)
    day_labels = y_test.reshape(n_test_days, numbers_per_day)

    hits = []
    for day in range(n_test_days):
        top = np.argsort(day_proba[day])[::-1][:draw_size]
        hits.append(day_labels[day, top].sum())
    mean_hits = float(np.mean(hits))

    # a single-class holdout has no defined AUC; 0.5 keeps the tie-breaker
    # neutral instead of erroring the whole trial
    auc = roc_auc_score(y_test, proba) if len(set(y_test.tolist())) > 1 else 0.5

    return mean_hits + 0.01 * auc, mean_hits, auc


def objective_quantum(trial, suggest_func, fit_func, study_label,
                      X_train, y_train, X_test, y_test, n_test_days, draw_size):
    """
    One trial: fit the variant on the early-day table with the suggested
    params, score the late-day holdout. Both quantum classes seed their own
    RNG from a fixed random_state default, so identical params always produce
    identical scores - trial differences come from the hyperparameters, not
    sampling luck (same reasoning as the Backtester's per-day reseed).
    Per-trial elapsed is printed because these fits are the budget: a study
    must stay at seconds-to-minutes per trial to fit runHyperopt.sh's weekly
    window, and the timing makes a runaway configuration visible in the log.
    """
    trialStart = time.time()

    params = suggest_func(trial)
    model = fit_func(X_train, y_train, params=params)
    value, mean_hits, auc = score_holdout(model, X_test, y_test, n_test_days, draw_size)

    print(f"{study_label} trial {trial.number}: hits/day={mean_hits:.3f} "
          f"auc={auc:.4f} value={value:.4f} ({time.time() - trialStart:.1f}s)")
    return value


def objective_quantum_positional(trial, suggest_func, fit_func, study_label,
                                 X_train, y_train, X_test, test_actual_ordered):
    """
    Pick3 trial: fit one classifier per position on the early days (the
    suggested params bound with functools.partial, the same way
    TrainMetaLearner binds them for the persisted artifact), play the argmax
    ticket on every holdout day and score it with the real Pick3 payout table
    (Helpers.pick3_ticket_profit) against the drawn-order result - profit is
    the research metric for the payout games, and unlike hits it rewards the
    slot-exact straight/pair structure the positional model exists for. Mean
    per-position top-1 accuracy enters as a +0.01 tie-breaker: the daily
    profit is -4 on almost every day with rare spikes (+46 for a pair, several
    hundred for a straight), so the profit mean alone ties most trials, while
    0.01 * accuracy <= 0.01 can never outweigh a single real payout.
    """
    trialStart = time.time()

    params = suggest_func(trial)
    position_models = fit_position_models(X_train, y_train, functools.partial(fit_func, params=params))
    mean_profit, accuracies, _ = evaluate_positional_holdout(position_models, X_test, test_actual_ordered)
    mean_accuracy = float(np.mean(accuracies))
    value = mean_profit + 0.01 * mean_accuracy

    accuracy_text = "/".join(f"{accuracy:.3f}" for accuracy in accuracies)
    print(f"{study_label} trial {trial.number}: profit/day={mean_profit:.3f} "
          f"top1 acc per pos={accuracy_text} (mean {mean_accuracy:.4f}) value={value:.4f} "
          f"({time.time() - trialStart:.1f}s)")
    return value


if __name__ == "__main__":
    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)

    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)

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
        parser.add_argument('-s', '--save', type=helpers.str2bool, default=True)
        parser.add_argument(
            '-g', '--games',
            type=str,
            default=",".join(GAME_CONFIG.keys()),
            help='Comma-separated list of games, e.g. "keno,lotto"'
        )

        args = parser.parse_args()

        print_intro()

        days_back = int(args.days)
        n_trials = int(args.trials)
        pushToGit = bool(args.save)

        print("Push to git: ", pushToGit)
        print("Running ", n_trials, "trials")

        games = [g.strip() for g in args.games.split(',') if g.strip()]
        unknown_games = [g for g in games if g not in GAME_CONFIG]
        if unknown_games:
            print(f"Unknown game(s), ignoring: {unknown_games}")
        print("Selected games:", games)

        path = os.getcwd()
        optunaDatabase = "sqlite:///db.sqlite3"

        for dataset_name, game_cfg in GAME_CONFIG.items():
            if dataset_name not in games:
                continue
            try:
                print(f"\n{dataset_name.capitalize()}")
                # Pick3 is positional, so it is tuned on the positional table
                # (one classifier per slot, argmax-ticket profit objective)
                # instead of the per-number ranking every other game uses.
                is_pick3 = dataset_name == "pick3"

                collected = collect_score_table(dataset_name, game_cfg, path, days_back)
                if collected is None:
                    continue
                results, model_names, main_actual_key = collected

                if len(results) < MIN_EVALUATION_DAYS:
                    print(f"Skipping {dataset_name}: only {len(results)} backtest days collected "
                          f"(need at least {MIN_EVALUATION_DAYS} for a meaningful 75/25 day split)")
                    continue

                test_actual_ordered = None
                if is_pick3:
                    X_train, y_train, X_test, y_test, test_actual_ordered = build_positional_day_split(
                        results, model_names)
                    n_test_days = len(test_actual_ordered)
                else:
                    X_train, y_train, X_test, y_test, n_test_days = build_day_split(
                        results, model_names, game_cfg, main_actual_key)

                if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
                    # can't fit (or rank) a binary classifier on one class -
                    # only plausible with a degenerate/truncated data file
                    print(f"Skipping {dataset_name}: train or holdout labels contain a single class")
                    continue

                print(f"Tuning on {len(results) - n_test_days} train days / {n_test_days} holdout days "
                      f"({len(X_train)} / {len(X_test)} table rows)")

                jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
                existingData = {}
                if os.path.exists(jsonBestParamsFilePath):
                    with open(jsonBestParamsFilePath, "r") as infile:
                        existingData = json.load(infile)

                for variant_name, fit_func, suggest_func in VARIANTS:
                    studyName = f"{dataset_name}-{variant_name}"
                    study = optuna.create_study(
                        direction='maximize',
                        storage=optunaDatabase,
                        study_name=studyName,
                        load_if_exists=True
                    )

                    if is_pick3:
                        objective = lambda trial, suggest_func=suggest_func, fit_func=fit_func, studyName=studyName: \
                            objective_quantum_positional(
                                trial, suggest_func, fit_func, studyName,
                                X_train, y_train, X_test, test_actual_ordered
                            )
                    else:
                        objective = lambda trial, suggest_func=suggest_func, fit_func=fit_func, studyName=studyName: \
                            objective_quantum(
                                trial, suggest_func, fit_func, studyName,
                                X_train, y_train, X_test, y_test, n_test_days, game_cfg["draw_size"]
                            )

                    runStart = datetime.now()
                    studyStart = time.time()
                    study.optimize(objective, n_trials=n_trials)
                    print(f"Study {studyName} finished in {time.time() - studyStart:.1f}s")

                    # Best of THIS RUN's trials only - study.best_params would
                    # compare across weeks whose trials were scored on
                    # different data windows and holdout draws (the objective
                    # is quantized at 1/n_test_days hits), so one lucky old
                    # trial could pin best_params indefinitely and silently
                    # defeat the run-Quantum-before-TrainMetaLearner ordering.
                    # The study itself stays cumulative for the dashboard.
                    runTrials = [t for t in study.trials
                                 if t.state == optuna.trial.TrialState.COMPLETE
                                 and t.datetime_start is not None and t.datetime_start >= runStart]
                    if not runTrials:
                        print(f"No completed trials this run for {studyName} - keeping existing params")
                        continue
                    bestTrial = max(runTrials, key=lambda t: t.value)
                    print(f"Best Parameters for {studyName} (this run): ", bestTrial.params)
                    print(f"Best Score for {studyName} (this run): ", bestTrial.value,
                          f"(all-time study best: {study.best_value})")

                    existingData.update(bestTrial.params)

                    # Written after EACH study (not once per game) so a crash
                    # in the second study can't throw away the first's result.
                    with open(jsonBestParamsFilePath, "w+") as outfile:
                        json.dump(existingData, outfile, indent=4)

            except Exception as e:
                print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")

        try:
            if pushToGit:
                helpers.git_push(commit_message="Saving latest quantum hyperopt")
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()
