import os, argparse, json, sys
# Pin the BLAS/OpenMP pools before numpy is imported: every Backtester worker
# otherwise inherits a 16-thread OpenBLAS pool whose spin-waiting was measured
# (HyperoptBoost) at ~10 cores of pure overhead per single-threaded fit.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import optuna
import joblib
from art import text2art
from datetime import datetime

from src.Backtester import Backtester, BacktestTimeout
from src.DataLoader import DataLoader
from src.HyperoptRunner import (open_study, fail_stale_running_trials, has_completed_trials, optimize_study,
                                install_sigterm_handler)
from src.Markov import Markov
from src.MarkovMonteCarlo import MarkovMonteCarlo
from src.MarkovBayesian import MarkovBayesian
from src.MarkovBayesianEnhanched import MarkovBayesianEnhanced
from src.PoissonMonteCarlo import PoissonMonteCarlo
from src.PoissonMarkov import PoissonMarkov
from src.LaplaceMonteCarlo import LaplaceMonteCarlo
from src.HybridStatisticalModel import HybridStatisticalModel
from src.ModelFactory import BASE_MODEL_NAMES, build_models, prepare_foundation_scores
from src.TuningScore import score_rows, score_bets_by_day, attrs_for_trial, describe
from src.TuningGate import challenge, make_defaults
from src.Command import Command
from src.Helpers import Helpers
from src.DataFetcher import DataFetcher

command = Command()
helpers = Helpers()
dataFetcher = DataFetcher()

LOCK_FILE = os.path.join(os.getcwd(), "process.lock")

# Real per-game number ranges (see test data inspection) - needed so the
# Backtester's data-loader Markov instance and baselines use the actual game
# range instead of Markov's default (1-80, which only happens to match Keno).
GAME_CONFIG = {
    # Euromillions has 2 trailing star columns; EuroDreams/VikingLotto have 1
    # (dream number / super viking) - see Helpers.run_model_with_special_column.
    "euromillions": {"min": 1, "max": 50, "draw_size": 5, "skip_last_columns": 0, "special_column_count": 2},
    "lotto":        {"min": 1, "max": 45, "draw_size": 6, "skip_last_columns": 1, "special_column_count": 0},
    "eurodreams":   {"min": 1, "max": 40, "draw_size": 6, "skip_last_columns": 0, "special_column_count": 1},
    "keno":         {"min": 1, "max": 80, "draw_size": 20, "skip_last_columns": 0, "special_column_count": 0},
    "pick3":        {"min": 0, "max": 9, "draw_size": 3, "skip_last_columns": 0, "special_column_count": 0},
    "vikinglotto":  {"min": 1, "max": 48, "draw_size": 6, "skip_last_columns": 0, "special_column_count": 1},
    # Joker+: six digits 0-9 drawn WITH replacement in a fixed order (a
    # positional game like pick3, see Helpers.is_positional_game) plus one
    # trailing zodiac column - a 12-sign special column stored as its 0..11
    # code (Helpers.encode_zodiac), modeled independently like the
    # Euromillions stars. skip_last_columns stays 0 so the sign is present
    # for the special-column split; draw_size counts the digits only.
    "jokerplus":    {"min": 0, "max": 9, "draw_size": 6, "skip_last_columns": 0, "special_column_count": 1},
}

KENO_SUBSET_VALUES = [5, 6, 7, 8, 9, 10]

# Games with a real payout table (Helpers.keno_ticket_profit /
# pick3_ticket_profit / jokerplus_ticket_profit): the Backtester computes
# profit rows for them, so their tuning objective is profit_per_bet instead
# of avg hits (see score_from_summary).
PAYOUT_GAMES = ("keno", "pick3", "jokerplus")

# Models with no per-position modeling of their own (they pool number
# frequencies globally across all digit positions) - excluded entirely for
# the positional games (Pick3, Joker+ - Helpers.is_positional_game), matching
# the same disable list Predictor.py uses, since no amount of hyperparameter
# tuning fixes a structurally non-positional model there. The old name stays
# as an alias for anything still importing it.
DISABLED_FOR_POSITIONAL = {"MarkovBayesian", "MarkovBayesianEnhanced", "PoissonMarkov", "HybridStatistical"}
DISABLED_FOR_PICK3 = DISABLED_FOR_POSITIONAL


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
    ascii_art = text2art("Predictor Hyperopt")
    print("============================================================")
    print("Predictor Hyperopt")
    print("Licence : MIT License")
    print(ascii_art)
    print("Find best parameters for Predictor")


# The game's served Keno subset sizes, set per game in __main__ from
# bestParams_<game>.json (None = every playable size, what Predictor's own
# template serves). Module global so the objectives - which run in forked
# trial processes and in the gate's reference children - inherit it.
SERVED_KENO_SUBSETS = None


def keno_subsets_served(bestParams):
    """
    The Keno subset sizes the game serves: the global use_<n> flags, a
    missing flag read as served (Predictor's statisticalMethod and boosting
    templates default them to True; its vote-ensemble rows read a missing
    flag as not played - the file has carried all six keys for years).
    """
    return [size for size in KENO_SUBSET_VALUES if bestParams.get(f"use_{size}", True)]


def served_keno_subsets(model_name):
    """
    The Keno subset sizes a trial (and a gate reference) bets on: the sizes
    the game serves. Until September 2026 this was suggest_keno_subset, a
    searched per-model inclusion mask written as "<model>_use_<n>" keys -
    which Predictor never read (getKenoSubsetSizes uses the global flags), so
    a trial's value depended on ticket sizes that never reached production
    and the gate would have compared unlike tickets. Returns None when the
    game serves no playable size (the caller treats the trial as invalid).
    The stale per-model keys in older files are simply ignored.
    """
    sizes = list(KENO_SUBSET_VALUES) if SERVED_KENO_SUBSETS is None else list(SERVED_KENO_SUBSETS)
    return sizes or None


# Wall-clock budget per tuning trial (CLI --trial-timeout): a statistical trial
# is seconds, so this only catches a runaway configuration - the Backtester
# terminates its pool and the trial is recorded as PRUNED (see run_backtest).
TRIAL_TIMEOUT_SECONDS = 1200
# Champion/challenger gate (src/TuningGate.py), set from the CLI in __main__:
# this run's best trial replaces the served parameters only if it beats them
# and the untuned defaults, re-scored on the same window, by GATE_MARGIN.
GATE_ENABLED = True
GATE_MARGIN = 0.0


def run_backtest(model_name, model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back):
    """
    Builds a dedicated DataLoader configured with this game's real number
    range (so Backtester's baselines/bookkeeping aren't stuck on Markov's
    defaults), runs `model` through Backtester over the last
    `days_to_rebuild` days, and returns that model's compact summary dict
    (see Backtester.summarize): {"hits_avg", "profit_total", "main": {...},
    "subsets": {...}, "errors": {...}} plus "tuning", the objective of
    src/TuningScore.py (a lower confidence bound over the days, jackpots
    capped) that score_from_summary returns.

    Backtester reseeds numpy/random per backtested day (see Backtester.py),
    so results are deterministic for a given set of hyperparameters; the
    repeat-and-average the old Process-based pipeline did would return the
    same number, which is why the objective works on the spread over the
    days instead.
    """
    loader = DataLoader()
    loader.setDataPath(dataPath)
    loader.setGameRange(game_cfg["min"], game_cfg["max"])
    loader.setDrawSize(game_cfg["draw_size"])

    numbers, _, _ = loader.load_numbers(skipLastColumns=game_cfg["skip_last_columns"], years_back=years_back)
    total_rows = len(numbers)

    if total_rows == 0:
        return {}

    start_index = max(0, total_rows - days_to_rebuild)

    backtester = Backtester(loader)
    backtester.add_model(model_name, model)

    # Only Keno/Pick3/Joker+ have a real payout model to score profit with
    # (see Helpers.keno_ticket_profit/pick3_ticket_profit/
    # jokerplus_ticket_profit) - other games fall back to avg hits as the
    # tuning objective.
    game_param = dataset_name if dataset_name in PAYOUT_GAMES else None

    try:
        results = backtester.backtest(
            max_seconds=TRIAL_TIMEOUT_SECONDS,
            start_index=start_index,
            end_index=total_rows,
            generate_subsets=subsets,
            skipLastColumns=game_cfg["skip_last_columns"],
            years_back=years_back,
            include_baselines=False,
            verbose=False,
            game=game_param,
            special_column_count=game_cfg["special_column_count"]
        )
    except BacktestTimeout as e:
        print(f"Trial pruned: {e}")
        raise optuna.TrialPruned()

    summary = backtester.summarize(results)
    model_summary = dict(summary.get("models", {}).get(model_name, {}))
    model_summary["tuning"] = score_rows(results, model_name, payout=game_param is not None,
                                         positional=helpers.is_positional_game(dataset_name), game=dataset_name)
    return model_summary


def score_from_summary(model_summary):
    """
    Optuna objective value: the "tuning" score run_backtest attached
    (src/TuningScore.py - a lower confidence bound over the window's days,
    capped profit per bet where this game has a payout model, slot hits for
    the positional games, hits otherwise). A summary without it (an older
    caller) falls back to the pre-September-2026 value, raw profit_per_bet
    or avg hits: per bet so a trial betting fewer subset sizes is not
    penalised for placing fewer bets, but the number one jackpot-tier payout
    in the window decides (see Backtester.summarize()'s "lucky_strikes").
    """
    if not model_summary:
        return float("-inf")

    tuning = model_summary.get("tuning")
    if tuning is not None:
        return tuning["score"]

    profit_per_bet = model_summary.get("profit_per_bet")
    if profit_per_bet is not None:
        return profit_per_bet

    hits = model_summary.get("hits_avg")
    return hits if hits is not None else float("-inf")


def finish_trial(trial, model_summary):
    """
    score_from_summary plus the diagnostics on the trial (the raw profit per
    bet, the lucky strikes - src/TuningScore.py), so a look at the study
    shows what a value was made of.
    """
    tuning = model_summary.get("tuning") if model_summary else None
    trial.set_user_attr("tuning", attrs_for_trial(tuning))
    print(f"Trial {trial.number}: {describe(tuning)}")
    return score_from_summary(model_summary)


def objective_markov(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    # Positional games (Pick3, Joker+): digits stay in drawn order and Markov
    # scores whole tickets by pair affinity - see Helpers.is_positional_game.
    is_positional = helpers.is_positional_game(dataset_name)

    model = Markov()
    model.setDataPath(dataPath)
    model.setSoftMAxTemperature(trial.suggest_float('markovSoftMaxTemperature', 0.1, 1.0))
    model.setMinOccurrences(trial.suggest_int('markovMinOccurences', 1, 20))
    model.setAlpha(trial.suggest_float('markovAlpha', 0.1, 1.0))
    model.setRecencyWeight(trial.suggest_float('markovRecencyWeight', 0.1, 2.0))
    model.setRecencyMode(trial.suggest_categorical('markovRecencyMode', ["linear", "log", "constant"]))
    model.setPairDecayFactor(trial.suggest_float('markovPairDecayFactor', 0.1, 1.0))
    model.setSmoothingFactor(trial.suggest_float('markovSmoothingFactor', 0.01, 1.0))
    model.setSubsetSelectionMode(trial.suggest_categorical('markovSubsetSelectionMode', ["top", "softmax"]))
    model.setBlendMode(trial.suggest_categorical('markovBlendMode', ["linear", "harmonic", "log"]))
    model.setMarkovOrder(trial.suggest_int('markovOrder', 1, 3))

    model.setSortedPrediction(not is_positional)
    model.setUsePairScoring(is_positional)
    model.setPairScoringWeight(trial.suggest_float('markovPairScoringWeight', 0.1, 2.0) if is_positional else 0.0)

    subsets = []
    if "keno" in dataset_name:
        subsets = served_keno_subsets("markov")
        if subsets is None:
            return float("-inf")

    return finish_trial(trial, run_backtest("markov", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_markov_mc(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    is_positional = helpers.is_positional_game(dataset_name)

    base = Markov()
    base.setDataPath(dataPath)
    base.setSoftMAxTemperature(trial.suggest_float('markovMcSoftMaxTemperature', 0.1, 1.0))
    base.setMinOccurrences(trial.suggest_int('markovMcMinOccurences', 1, 20))
    base.setAlpha(trial.suggest_float('markovMcAlpha', 0.1, 1.0))
    base.setRecencyWeight(trial.suggest_float('markovMcRecencyWeight', 0.1, 2.0))
    base.setRecencyMode(trial.suggest_categorical('markovMcRecencyMode', ["linear", "log", "constant"]))
    base.setPairDecayFactor(trial.suggest_float('markovMcPairDecayFactor', 0.1, 1.0))
    base.setSmoothingFactor(trial.suggest_float('markovMcSmoothingFactor', 0.01, 1.0))
    base.setMarkovOrder(trial.suggest_int('markovMcOrder', 1, 3))
    base.setSortedPrediction(not is_positional)

    model = MarkovMonteCarlo(base)
    model.setNumOfSimulations(trial.suggest_int('markovMcNumSimulations', 100, 2000, step=100))

    subsets = []
    if "keno" in dataset_name:
        subsets = served_keno_subsets("markov_mc")
        if subsets is None:
            return float("-inf")

    return finish_trial(trial, run_backtest("markov_mc", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_markov_bayesian(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = MarkovBayesian()
    model.setDataPath(dataPath)
    model.setSoftMAxTemperature(trial.suggest_float('markovBayesianSoftMaxTemperature', 0.05, 1.0))
    model.setMinOccurrences(trial.suggest_int('markovBayesianMinOccurences', 3, 15))
    model.setAlpha(trial.suggest_float('markovBayesianAlpha', 0.2, 0.9))
    model.setSortedPrediction(not helpers.is_positional_game(dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = served_keno_subsets("markov_bayesian")
        if subsets is None:
            return float("-inf")

    return finish_trial(trial, run_backtest("markov_bayesian", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_markov_bayesian_enhanced(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = MarkovBayesianEnhanced()
    model.setDataPath(dataPath)
    model.setSoftMAxTemperature(trial.suggest_float('markovBayesianEnhancedSoftMaxTemperature', 0.1, 1.0))
    model.setAlpha(trial.suggest_float('markovBayesianEnhancedAlpha', 0.1, 1.0))
    model.setMinOccurrences(trial.suggest_int('markovBayesianEnhancedMinOccurences', 1, 20))
    model.setSortedPrediction(not helpers.is_positional_game(dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = served_keno_subsets("markov_bayesian_enhanced")
        if subsets is None:
            return float("-inf")

    return finish_trial(trial, run_backtest("markov_bayesian_enhanced", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_poisson_mc(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = PoissonMonteCarlo()
    model.setDataPath(dataPath)
    model.setNumOfSimulations(trial.suggest_int('poissonMonteCarloNumberOfSimulations', 100, 1000, step=100))
    model.setWeightFactor(trial.suggest_float('poissonMonteCarloWeightFactor', 0.1, 1.0))
    model.setSortedPrediction(not helpers.is_positional_game(dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = served_keno_subsets("poisson_mc")
        if subsets is None:
            return float("-inf")

    return finish_trial(trial, run_backtest("poisson_mc", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_poisson_markov(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = PoissonMarkov()
    model.setDataPath(dataPath)
    weight = trial.suggest_float('poissonMarkovWeight', 0.1, 1.0)
    model.setWeights(poisson_weight=weight, markov_weight=1 - weight)
    model.setNumberOfSimulations(trial.suggest_int('poissonMarkovNumberOfSimulations', 100, 1000, step=100))
    model.setSortedPrediction(not helpers.is_positional_game(dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = served_keno_subsets("poisson_markov")
        if subsets is None:
            return float("-inf")

    return finish_trial(trial, run_backtest("poisson_markov", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_laplace_mc(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = LaplaceMonteCarlo()
    model.setDataPath(dataPath)
    model.setNumOfSimulations(trial.suggest_int('laplaceMonteCarloNumberOfSimulations', 100, 1000, step=100))
    model.setSortedPrediction(not helpers.is_positional_game(dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = served_keno_subsets("laplace_mc")
        if subsets is None:
            return float("-inf")

    return finish_trial(trial, run_backtest("laplace_mc", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_hybrid(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = HybridStatisticalModel()
    model.setDataPath(dataPath)
    model.setSoftMaxTemperature(trial.suggest_float('hybridStatisticalModelSoftMaxTemperature', 0.1, 1.0))
    model.setAlpha(trial.suggest_float('hybridStatisticalModelAlpha', 0.1, 1.0))
    model.setMinOccurrences(trial.suggest_int('hybridStatisticalModelMinOcurrences', 1, 20))
    model.setNumberOfSimulations(trial.suggest_int('hybridStatisticalModelNumberOfSimulations', 100, 1000, step=100))
    model.setSortedPrediction(not helpers.is_positional_game(dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = served_keno_subsets("hybrid_statistical")
        if subsets is None:
            return float("-inf")

    return finish_trial(trial, run_backtest("hybrid_statistical", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


# Caches the (expensive, one-time) precompute build_keno_ensemble_day_data does
# for objective_keno_subset_tuning, keyed by dataset_name - study.optimize()
# calls the objective once per trial in the same process, and none of that
# precompute depends on the subset mode/temperature being searched, so it only
# needs to run once per hyperopt invocation instead of once per trial.
_KENO_SUBSET_TUNING_CACHE = {}


def build_keno_ensemble_day_data(dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    """
    Backtests the 7 base models once, using their OWN already-tuned
    bestParams_<dataset_name>.json params (not re-tuned here), then
    reconstructs - for each backtested day - the exact WeightedEnsemble Model
    and MetaLearner Model main tickets and per-number score dicts Predictor.py
    would have produced that day. This is Keno-only (the only game with
    sub-selections), and everything here is independent of the subset
    mode/temperature objective_keno_subset_tuning searches over.

    Returns {"subset_sizes": [...], "days": [{"actual", "weighted_ticket",
    "weighted_scores", "meta_ticket"|None, "meta_scores"|None}, ...]}.
    """
    bestParamsPath = os.path.join(os.getcwd(), f"bestParams_{dataset_name}.json")
    bestParams = {}
    if os.path.exists(bestParamsPath):
        with open(bestParamsPath, "r") as infile:
            bestParams = json.load(infile)

    # The served sizes - the global use_<n> flags, the same set
    # served_keno_subsets gives every other strategy (SERVED_KENO_SUBSETS is
    # set in __main__ before this precompute runs; the fallback covers a
    # direct call). Predictor's statisticalMethod, where the meta-learner
    # rows are served, defaults a missing flag to True the same way.
    subset_sizes = list(SERVED_KENO_SUBSETS) if SERVED_KENO_SUBSETS is not None else keno_subsets_served(bestParams)
    if not subset_sizes:
        return {"subset_sizes": [], "days": []}

    loader = DataLoader()
    loader.setDataPath(dataPath)
    loader.setGameRange(game_cfg["min"], game_cfg["max"])
    loader.setDrawSize(game_cfg["draw_size"])

    numbers, _, _ = loader.load_numbers(skipLastColumns=game_cfg["skip_last_columns"], years_back=years_back)
    total_rows = len(numbers)
    if total_rows == 0:
        return {"subset_sizes": subset_sizes, "days": []}

    start_index = max(0, total_rows - days_to_rebuild)

    models = build_models(dataPath, bestParams, is_pick3=False)
    model_names = [name for name in BASE_MODEL_NAMES if name in models]

    # Foundation models cannot run inside the Backtester's forked pool (see
    # ModelFactory.prepare_foundation_scores). This precompute happens in the
    # parent, which is also where this whole table is built so the trial
    # processes inherit it - so it is paid once per run, not once per trial.
    prepare_foundation_scores(
        models, start_index, total_rows,
        skipLastColumns=game_cfg["skip_last_columns"],
        label=f"{dataset_name} keno subsets: ")

    backtester = Backtester(loader)
    for name, model in models.items():
        backtester.add_model(name, model)

    # The one step of this tuner no trial budget reaches: it runs in the
    # coordinator, once per game. Bounded by its own wall clock (three trial
    # budgets: measured 43 min at 31 days on 2026-09-05 with a 90-tree
    # depth-8 XGBoost served, ~6 min a week later with the 10-tree one) so a
    # slow served configuration cannot hold the weekly chain for hours - the
    # row is then skipped this run instead.
    try:
        results = backtester.backtest(
            max_seconds=3 * TRIAL_TIMEOUT_SECONDS,
            start_index=start_index,
            end_index=total_rows,
            skipLastColumns=game_cfg["skip_last_columns"],
            years_back=years_back,
            include_baselines=False,
            collect_scores=True,
            verbose=False
        )
    except BacktestTimeout as e:
        print(f"Keno ensemble day table not built ({e}) - KenoSubsetTuning is skipped this run")
        return {"subset_sizes": subset_sizes, "days": []}

    model_scores = bestParams.get("modelScores", {})

    def load_meta_artifact(filename):
        artifact_path = os.path.join(os.getcwd(), "data", "models", dataset_name, filename)
        if not os.path.exists(artifact_path):
            return None
        try:
            return joblib.load(artifact_path)
        except Exception as e:
            print(f"Failed to load {filename} for {dataset_name}, skipping its subset tuning: {e}")
            return None

    meta_artifact = load_meta_artifact("meta_learner.joblib")
    meta_v2_artifact = load_meta_artifact("meta_learner_v2.joblib")

    def rank_by_meta_artifact(artifact, row):
        feature_names = artifact["feature_names"]
        number_range = list(range(artifact["min_number"], artifact["max_number"] + 1))
        feature_matrix = [
            [row.get(f"{name}_scores", {}).get(number, 0.0) for name in feature_names]
            for number in number_range
        ]
        probabilities = artifact["model"].predict_proba(feature_matrix)[:, 1]
        ranked_numbers = [n for _, n in sorted(zip(probabilities, number_range), reverse=True)]
        ticket = sorted(ranked_numbers[:artifact["draw_size"]])
        return ticket, dict(zip(number_range, probabilities))

    days = []
    for row in results:
        actual = row.get("actual", [])

        newPrediction = [
            {"name": name, "predictions": [row.get(f"{name}_prediction", [])]}
            for name in model_names
        ]
        weighted_scores = helpers.count_number_frequencies_from_new_prediction(
            {"newPrediction": newPrediction}, model_scores=model_scores)
        weighted_ticket_entry = helpers.build_weighted_ensemble_prediction(weighted_scores, game_cfg["draw_size"])
        if not weighted_ticket_entry:
            continue

        day = {
            "actual": actual,
            "weighted_ticket": weighted_ticket_entry["predictions"][0],
            "weighted_scores": weighted_scores,
            "meta_ticket": None,
            "meta_scores": None,
            "meta_v2_ticket": None,
            "meta_v2_scores": None,
        }

        if meta_artifact is not None:
            day["meta_ticket"], day["meta_scores"] = rank_by_meta_artifact(meta_artifact, row)

        if meta_v2_artifact is not None:
            day["meta_v2_ticket"], day["meta_v2_scores"] = rank_by_meta_artifact(meta_v2_artifact, row)

        days.append(day)

    return {"subset_sizes": subset_sizes, "days": days}


def objective_keno_subset_tuning(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    """
    Tunes Helpers.generate_subset_from_scores' mode/temperature for
    WeightedEnsemble Model, MetaLearner Model, and MetaLearnerV2 Model - Keno
    only (the only game with sub-selections). Unlike every other objective
    above, this doesn't re-tune any base model or subset size choice; it
    reuses the (cached, one-time) backtest from build_keno_ensemble_day_data
    and only searches over how each ensemble's already-ranked ticket gets
    sliced into a playable 5-10-number subset.
    """
    if "keno" not in dataset_name:
        return 0.0

    cached = _KENO_SUBSET_TUNING_CACHE.get(dataset_name)
    if cached is None:
        cached = build_keno_ensemble_day_data(dataset_name, dataPath, game_cfg, days_to_rebuild, years_back)
        _KENO_SUBSET_TUNING_CACHE[dataset_name] = cached

    if not cached["days"]:
        return float("-inf")

    weighted_mode = trial.suggest_categorical("weightedEnsembleSubsetMode", ["top", "softmax"])
    weighted_temperature = trial.suggest_float("weightedEnsembleSubsetTemperature", 0.05, 2.0)
    meta_mode = trial.suggest_categorical("metaLearnerSubsetMode", ["top", "softmax"])
    meta_temperature = trial.suggest_float("metaLearnerSubsetTemperature", 0.05, 2.0)
    meta_v2_mode = trial.suggest_categorical("metaLearnerV2SubsetMode", ["top", "softmax"])
    meta_v2_temperature = trial.suggest_float("metaLearnerV2SubsetTemperature", 0.05, 2.0)

    bets_by_day = []

    for day in cached["days"]:
        day_bets = []
        for subset_size in cached["subset_sizes"]:
            subset = helpers.generate_subset_from_scores(
                day["weighted_scores"], day["weighted_ticket"], subset_size,
                mode=weighted_mode, temperature=weighted_temperature)
            profit = helpers.keno_ticket_profit(subset, day["actual"])
            if profit is not None:
                day_bets.append(profit)

            if day["meta_ticket"] is not None:
                subset = helpers.generate_subset_from_scores(
                    day["meta_scores"], day["meta_ticket"], subset_size,
                    mode=meta_mode, temperature=meta_temperature)
                profit = helpers.keno_ticket_profit(subset, day["actual"])
                if profit is not None:
                    day_bets.append(profit)

            if day["meta_v2_ticket"] is not None:
                subset = helpers.generate_subset_from_scores(
                    day["meta_v2_scores"], day["meta_v2_ticket"], subset_size,
                    mode=meta_v2_mode, temperature=meta_v2_temperature)
                profit = helpers.keno_ticket_profit(subset, day["actual"])
                if profit is not None:
                    day_bets.append(profit)

        bets_by_day.append(day_bets)

    # The same objective as the model strategies (src/TuningScore.py): a lower
    # confidence bound of the per-day capped profit per bet, not the raw mean
    # that one 6/6 in the window could decide.
    tuning = score_bets_by_day(bets_by_day, payout=True)
    trial.set_user_attr("tuning", attrs_for_trial(tuning))
    print(f"Trial {trial.number}: {describe(tuning)}")
    return tuning["score"]


# Maps a -s/--strategies CLI name to its objective + the "use<X>" flag Predictor.py
# reads from bestParams_<game>.json to decide whether to run that model live.
#
# "games" restricts a strategy to the games it actually applies to; omitted (or
# None) means every game. Without this, a game-specific strategy still got a
# study created and every one of --trials trials run against it, each returning
# a constant no-op score - wasted runtime plus a meaningless "Best Score: 0.0"
# in the log and an empty study row in db.sqlite3 for every game it never
# applied to.
# What Predictor.py serves when a tuned key is missing from bestParams_<game>.json
# - its built-in template (Predictor.py, the bestParams_json_object literal)
# and the .get() fallbacks of the ensemble subset keys. The "default" reference
# of the champion/challenger gate (src/TuningGate.py): a first tuning of a row
# has to beat this, not just exist. Kept in step with Predictor.py by hand.
SERVED_DEFAULTS = {
    "markovSoftMaxTemperature": 0.10002049510925136,
    "markovMinOccurences": 9,
    "markovAlpha": 0.20682688936213361,
    "markovRecencyWeight": 1.591825953176242,
    "markovRecencyMode": "constant",
    "markovPairDecayFactor": 0.34980042438509473,
    "markovSmoothingFactor": 0.6342058116675424,
    "markovSubsetSelectionMode": "softmax",
    "markovBlendMode": "log",
    "markovOrder": 1,
    "markovPairScoringWeight": 0.0,
    "markovMcSoftMaxTemperature": 0.1,
    "markovMcMinOccurences": 9,
    "markovMcAlpha": 0.2,
    "markovMcRecencyWeight": 1.0,
    "markovMcRecencyMode": "constant",
    "markovMcPairDecayFactor": 0.3,
    "markovMcSmoothingFactor": 0.6,
    "markovMcOrder": 1,
    "markovMcNumSimulations": 1000,
    "markovBayesianSoftMaxTemperature": 0.24235148017270242,
    "markovBayesianMinOccurences": 14,
    "markovBayesianAlpha": 0.1452615422969012,
    "markovBayesianEnhancedSoftMaxTemperature": 0.4244268734953605,
    "markovBayesianEnhancedAlpha": 0.4015984866176651,
    "markovBayesianEnhancedMinOccurences": 19,
    "poissonMonteCarloNumberOfSimulations": 600,
    "poissonMonteCarloWeightFactor": 0.836053158339262,
    "poissonMarkovWeight": 0.48068822894893704,
    "poissonMarkovNumberOfSimulations": 100,
    "laplaceMonteCarloNumberOfSimulations": 900,
    "hybridStatisticalModelSoftMaxTemperature": 0.918188590362822,
    "hybridStatisticalModelAlpha": 0.7874157368729954,
    "hybridStatisticalModelMinOcurrences": 19,
    "hybridStatisticalModelNumberOfSimulations": 900,
    "weightedEnsembleSubsetMode": "softmax",
    "weightedEnsembleSubsetTemperature": 0.5,
    "metaLearnerSubsetMode": "softmax",
    "metaLearnerSubsetTemperature": 0.5,
    "metaLearnerV2SubsetMode": "softmax",
    "metaLearnerV2SubsetTemperature": 0.5,
}

STRATEGIES = {
    "Markov": {"objective": objective_markov, "use_key": "useMarkov"},
    "MarkovMonteCarlo": {"objective": objective_markov_mc, "use_key": "useMarkovMonteCarlo"},
    "MarkovBayesian": {"objective": objective_markov_bayesian, "use_key": "useMarkovBayesian"},
    "MarkovBayesianEnhanced": {"objective": objective_markov_bayesian_enhanced, "use_key": "usevMarkovBayesianEnhanced"},
    "PoissonMonteCarlo": {"objective": objective_poisson_mc, "use_key": "usePoissonMonteCarlo"},
    "PoissonMarkov": {"objective": objective_poisson_markov, "use_key": "usePoissonMarkov"},
    "LaPlaceMonteCarlo": {"objective": objective_laplace_mc, "use_key": "useLaplaceMonteCarlo"},
    "HybridStatistical": {"objective": objective_hybrid, "use_key": "useHybridStatisticalModel"},
    # Not a base model - tunes WeightedEnsemble/MetaLearner's Keno subset
    # mode/temperature (see objective_keno_subset_tuning). No use_key: it
    # doesn't gate a run/skip flag, Predictor.py reads its tuned params
    # unconditionally whenever it builds a Keno subset. Keno-only: it's the
    # only game with sub-selections, so there is nothing to tune anywhere else.
    "KenoSubsetTuning": {"objective": objective_keno_subset_tuning, "use_key": None, "games": ("keno",)},
}

# Maps a STRATEGIES key to the exact "name" Predictor.py gives that model's
# prediction entry in listOfDecodedPredictions - so the per-model backtest
# score saved here can be looked up directly by Helpers.count_number_frequencies_from_new_prediction
# without a second translation step.
STRATEGY_DISPLAY_NAMES = {
    "Markov": "Markov Model",
    "MarkovMonteCarlo": "MarkovMonteCarlo Model",
    "MarkovBayesian": "MarkovBayesian Model",
    "MarkovBayesianEnhanced": "MarkovBayesianEnhanched Model",
    "PoissonMonteCarlo": "PoissonMonteCarlo Model",
    "PoissonMarkov": "PoissonMarkov Model",
    "LaPlaceMonteCarlo": "LaplaceMonteCarlo Model",
    "HybridStatistical": "HybridStatisticalModel",
}


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

        parser.add_argument(
            '-d', '--days', type=int, default=90,
            help='Backtest window in draws, for every trial and for the gate\'s references. Was 31 '
                 'until September 2026, when the objective turned out to be decided by whether one '
                 'jackpot fell inside the window (README "Hyperopt & backtesting").')
        parser.add_argument('-t', '--trials', type=int, default=15)
        parser.add_argument(
            '--gate-margin', type=float, default=0.0,
            help='How much this run\'s best trial must beat the served parameters AND the untuned '
                 'defaults by - both re-scored on the same window - before it replaces them '
                 '(src/TuningGate.py). 0 = any improvement.')
        parser.add_argument(
            '--no-gate', action='store_true',
            help='Write this run\'s best trial without comparing it to what is served. Never the '
                 'study\'s all-time best any more: that is how one lucky window locked itself in.')
        parser.add_argument(
            '--trial-timeout', type=int, default=1200,
            help='Wall-clock budget in seconds per tuning trial; a trial over budget is pruned '
                 '(recorded, not scored). Statistical trials take seconds, this catches runaways.')
        parser.add_argument(
            '-s', '--strategies',
            type=str,
            default=",".join(STRATEGIES.keys()),
            help='Comma-separated list of strategies, e.g. "PoissonMonteCarlo,Markov,..."'
        )
        parser.add_argument(
            '-g', '--games',
            type=str,
            default=",".join(GAME_CONFIG.keys()),
            help='Comma-separated list of games, e.g. "keno,pick3,jokerplus"'
        )

        args = parser.parse_args()

        print_intro()

        current_year = datetime.now().year
        print("Current Year:", current_year)

        daysToRebuild = int(args.days)
        n_trials = int(args.trials)
        TRIAL_TIMEOUT_SECONDS = int(args.trial_timeout)
        GATE_ENABLED = not args.no_gate
        GATE_MARGIN = float(args.gate_margin)
        years_back = None  # None = all available data

        strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]
        print("Selected strategies:", strategies)

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
                dataPath = os.path.join(path, "data", "trainingData", dataset_name)
                file = f"{dataset_name}-gamedata-NL-{current_year}.csv"

                try:
                    if os.path.exists(os.path.join(dataPath, file)):
                        print("Starting data fetcher")
                        filePath = os.path.join(dataPath, file)
                        dataFetcher.startDate = dataFetcher.calculate_start_date(filePath)
                        gameName = {
                            "euromillions": "Euro+Millions",
                            "lotto": "Lotto",
                            "eurodreams": "EuroDreams",
                            "keno": "Keno",
                            "pick3": "Pick3",
                            "vikinglotto": "Viking+Lotto",
                            "jokerplus": "Joker%2B",
                        }.get(dataset_name, "")
                        # A failed/stalled fetch must not abort this game's
                        # tuning - the CSV on disk is at worst one draw behind.
                        try:
                            dataFetcher.getLatestData(gameName, filePath)
                        except Exception as e:
                            print(f"Data fetch failed for {dataset_name} - continuing with the existing CSV: {e}")
                except Exception as e:
                    print("Failed to fetch data: ", e)

                jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
                existingData = {}
                if os.path.exists(jsonBestParamsFilePath):
                    with open(jsonBestParamsFilePath, "r") as infile:
                        existingData = json.load(infile)

                # Trials and gate references bet the sizes Predictor serves.
                SERVED_KENO_SUBSETS = keno_subsets_served(existingData) if "keno" in dataset_name else None
                if "keno" in dataset_name:
                    print(f"Keno subset sizes served (use_<n>): {SERVED_KENO_SUBSETS}")

                is_positional = helpers.is_positional_game(dataset_name)
                profits = {}

                for strategy_name in strategies:
                    if strategy_name not in STRATEGIES:
                        print(f"Unknown strategy '{strategy_name}', skipping")
                        continue

                    if is_positional and strategy_name in DISABLED_FOR_POSITIONAL:
                        print(f"Skipping {strategy_name} for {dataset_name} - not a positional/per-column model")
                        continue

                    strategy = STRATEGIES[strategy_name]

                    applicable_games = strategy.get("games")
                    if applicable_games and dataset_name not in applicable_games:
                        print(f"Skipping {strategy_name} for {dataset_name} - only applies to: {', '.join(applicable_games)}")
                        continue

                    studyName = f"{dataset_name}_{strategy_name}"

                    if strategy_name == "KenoSubsetTuning" and dataset_name not in _KENO_SUBSET_TUNING_CACHE:
                        # Trials run in their own processes (HyperoptRunner):
                        # the one-time ensemble backtest must be built here,
                        # in the parent, so every trial inherits it through
                        # the fork instead of rebuilding it (6-43 min in prod
                        # at 31 days, depending on the served XGBoost).
                        print(f"Precomputing the Keno ensemble day table for {strategy_name}")
                        _KENO_SUBSET_TUNING_CACHE[dataset_name] = build_keno_ensemble_day_data(
                            dataset_name, dataPath, game_cfg, daysToRebuild, years_back)
                    if strategy_name == "KenoSubsetTuning" and not _KENO_SUBSET_TUNING_CACHE[dataset_name]["days"]:
                        print(f"Skipping {strategy_name} for {dataset_name} - no day table this run")
                        continue

                    study = open_study(studyName, optunaDatabase)
                    fail_stale_running_trials(study)
                    # Trials that exist before this run - the gate compares
                    # only what this run adds (TuningGate.run_trials).
                    known_trials = {t.number for t in study.get_trials(deepcopy=False)}

                    objective = lambda trial, strategy=strategy, dataset_name=dataset_name, dataPath=dataPath, \
                                       game_cfg=game_cfg: strategy["objective"](
                        trial, dataset_name, dataPath, game_cfg, daysToRebuild, years_back)
                    # One trial at a time: the Backtester already spreads each
                    # trial's days over every core, so a second concurrent
                    # trial would only add memory.
                    optimize_study(studyName, optunaDatabase, objective, n_trials, parallel=1, expected_trial_gb=0.3)

                    study = open_study(studyName, optunaDatabase, quiet=True)
                    if not has_completed_trials(study):
                        print(f"No completed trials for {strategy_name} (all pruned/failed) - keeping existing params")
                        existingData.get("modelScores", {}).pop(STRATEGY_DISPLAY_NAMES.get(strategy_name, strategy_name), None)
                        continue

                    # Champion/challenger (src/TuningGate.py): this run's best
                    # trial is written only if it beats the served parameters
                    # and the untuned defaults, re-scored on this run's window.
                    # Never study.best_params: the study's all-time best is the
                    # trial that caught the biggest payout (pick3 Markov's 21
                    # per bet and PoissonMonteCarlo's 42 are one and two
                    # straights in a 31-day window, nothing else).
                    display_name = STRATEGY_DISPLAY_NAMES.get(strategy_name, strategy_name)
                    outcome = challenge(
                        study, known_trials, objective, existingData, make_defaults(SERVED_DEFAULTS, existingData),
                        timeout_seconds=TRIAL_TIMEOUT_SECONDS + 120, label=studyName,
                        margin=GATE_MARGIN, enabled=GATE_ENABLED, window_days=daysToRebuild)
                    print(outcome["summary"])
                    existingData.setdefault("tuningGate", {})[display_name] = outcome["record"]
                    if outcome["params"] is not None:
                        existingData.update(outcome["params"])
                    if outcome["served_score"] is not None:
                        profits[strategy_name] = outcome["served_score"]
                    else:
                        # Kept without a score on this run's objective: drop a
                        # raw-profit-era entry (pick3 Markov 21/bet) rather than
                        # let it take the whole [1, 2] vote-weight range against
                        # bounds of about -1..+0.2 (Helpers._build_model_weight_lookup).
                        existingData.get("modelScores", {}).pop(display_name, None)

                    # Predictor.py reads these directly for Markov (they're not
                    # tuned separately per non-positional game the way the
                    # other models' sortedPrediction is, since Predictor.py
                    # derives that one from the game name at runtime for
                    # everyone else).
                    if strategy_name == "Markov":
                        existingData['markovSortedPrediction'] = not is_positional
                        existingData['markovUsePairScoring'] = is_positional
                        if not is_positional:
                            existingData['markovPairScoringWeight'] = 0.0

                # Do not make a choice for best strategy - Predictor.py still
                # runs every enabled model so their real-life performance can
                # be compared over time. The score is only used to weight each
                # model's vote in Helpers.count_number_frequencies_from_new_prediction's
                # combined numberFrequency view, not to disable any model.
                if profits:
                    print("Strategy scores: ", profits)
                    modelScores = existingData.get("modelScores", {})
                    modelScores.update({
                        STRATEGY_DISPLAY_NAMES[strategy_name]: score
                        for strategy_name, score in profits.items()
                        if strategy_name in STRATEGY_DISPLAY_NAMES
                    })
                    existingData["modelScores"] = modelScores

                with open(jsonBestParamsFilePath, "w+") as outfile:
                    json.dump(existingData, outfile, indent=4)

            except Exception as e:
                print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")

        try:
            for filename in os.listdir(os.getcwd()):
                if 'wget' in filename:
                    file_path = os.path.join(os.getcwd(), filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
        except Exception as e:
            print("Failed to cleanup folder")

        try:
            helpers.git_push(commit_message="Saving latest statistical hyperopt")
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()
