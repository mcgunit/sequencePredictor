import os, argparse, json, sys
import optuna
import joblib
from art import text2art
from datetime import datetime

from src.Backtester import Backtester
from src.DataLoader import DataLoader
from src.Markov import Markov
from src.MarkovMonteCarlo import MarkovMonteCarlo
from src.MarkovBayesian import MarkovBayesian
from src.MarkovBayesianEnhanched import MarkovBayesianEnhanced
from src.PoissonMonteCarlo import PoissonMonteCarlo
from src.PoissonMarkov import PoissonMarkov
from src.LaplaceMonteCarlo import LaplaceMonteCarlo
from src.HybridStatisticalModel import HybridStatisticalModel
from src.ModelFactory import BASE_MODEL_NAMES, build_models
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
}

KENO_SUBSET_VALUES = [5, 6, 7, 8, 9, 10]

# Models with no per-position modeling of their own (they pool number
# frequencies globally across all digit positions) - excluded entirely for
# Pick3, matching the same disable list Predictor.py uses, since no amount of
# hyperparameter tuning fixes a structurally non-positional model there.
DISABLED_FOR_PICK3 = {"MarkovBayesian", "MarkovBayesianEnhanced", "PoissonMarkov", "HybridStatistical"}


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


def suggest_keno_subset(trial, model_name):
    """
    Binary inclusion mask over the 5-10 playable Keno subset sizes, for one
    specific model. Returns None (caller should treat the trial as invalid) if
    the resulting subset is empty - a meaningless trial.

    Param names are prefixed with `model_name` (e.g. "markov_use_5") rather
    than a shared "use_5" - each strategy runs its own independent Optuna
    study and searches its own subset choice, so sharing bare "use_5"/etc.
    keys across strategies means whichever strategy's study.optimize() call
    happens to run last silently overwrites every other strategy's tuned
    choice in bestParams_<game>.json when merged in. Prefixing keeps each
    strategy's own choice distinct so nothing gets clobbered.
    """
    inclusion_mask = [trial.suggest_categorical(f"{model_name}_use_{v}", [True, False]) for v in KENO_SUBSET_VALUES]
    subset = [v for v, include in zip(KENO_SUBSET_VALUES, inclusion_mask) if include]

    if not subset:
        return None

    return subset


def run_backtest(model_name, model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back):
    """
    Builds a dedicated DataLoader configured with this game's real number
    range (so Backtester's baselines/bookkeeping aren't stuck on Markov's
    defaults), runs `model` through Backtester over the last
    `days_to_rebuild` days, and returns that model's compact summary dict
    (see Backtester.summarize): {"hits_avg", "profit_total", "main": {...},
    "subsets": {...}, "errors": {...}}.

    Backtester reseeds numpy/random per backtested day (see Backtester.py),
    so results are deterministic for a given set of hyperparameters - no need
    to repeat/average multiple runs per trial like the old Process-based
    pipeline did.
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

    # Only Keno/Pick3 have a real payout model to score profit with (see
    # Helpers.keno_ticket_profit/pick3_ticket_profit) - other games fall back
    # to avg hits as the tuning objective.
    game_param = dataset_name if dataset_name in ("keno", "pick3") else None

    results = backtester.backtest(
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

    summary = backtester.summarize(results)
    return summary.get("models", {}).get(model_name, {})


def score_from_summary(model_summary):
    """
    Optuna objective value: profit_per_bet if this game has a payout model,
    else avg hits. profit_per_bet (not profit_total) so trials aren't scored
    unfairly by how many subset sizes they happen to bet on - e.g. a trial
    betting only subset size 5 shouldn't look "worse" than one betting all 6
    sizes purely because it places fewer bets. Note this does NOT protect
    against a single rare jackpot-tier payout dominating the score (see
    Backtester.summarize()'s "lucky_strikes" field to check for that
    separately - profit_per_bet is just as vulnerable to one big hit as
    profit_total is, only rescaled).
    """
    if not model_summary:
        return float("-inf")

    profit_per_bet = model_summary.get("profit_per_bet")
    if profit_per_bet is not None:
        return profit_per_bet

    hits = model_summary.get("hits_avg")
    return hits if hits is not None else float("-inf")


def objective_markov(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    is_pick3 = "pick3" in dataset_name

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

    model.setSortedPrediction(not is_pick3)
    model.setUsePairScoring(is_pick3)
    model.setPairScoringWeight(trial.suggest_float('markovPairScoringWeight', 0.1, 2.0) if is_pick3 else 0.0)

    subsets = []
    if "keno" in dataset_name:
        subsets = suggest_keno_subset(trial, "markov")
        if subsets is None:
            return float("-inf")

    return score_from_summary(run_backtest("markov", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_markov_mc(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    is_pick3 = "pick3" in dataset_name

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
    base.setSortedPrediction(not is_pick3)

    model = MarkovMonteCarlo(base)
    model.setNumOfSimulations(trial.suggest_int('markovMcNumSimulations', 100, 2000, step=100))

    subsets = []
    if "keno" in dataset_name:
        subsets = suggest_keno_subset(trial, "markov_mc")
        if subsets is None:
            return float("-inf")

    return score_from_summary(run_backtest("markov_mc", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_markov_bayesian(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = MarkovBayesian()
    model.setDataPath(dataPath)
    model.setSoftMAxTemperature(trial.suggest_float('markovBayesianSoftMaxTemperature', 0.05, 1.0))
    model.setMinOccurrences(trial.suggest_int('markovBayesianMinOccurences', 3, 15))
    model.setAlpha(trial.suggest_float('markovBayesianAlpha', 0.2, 0.9))
    model.setSortedPrediction(not ("pick3" in dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = suggest_keno_subset(trial, "markov_bayesian")
        if subsets is None:
            return float("-inf")

    return score_from_summary(run_backtest("markov_bayesian", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_markov_bayesian_enhanced(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = MarkovBayesianEnhanced()
    model.setDataPath(dataPath)
    model.setSoftMAxTemperature(trial.suggest_float('markovBayesianEnhancedSoftMaxTemperature', 0.1, 1.0))
    model.setAlpha(trial.suggest_float('markovBayesianEnhancedAlpha', 0.1, 1.0))
    model.setMinOccurrences(trial.suggest_int('markovBayesianEnhancedMinOccurences', 1, 20))
    model.setSortedPrediction(not ("pick3" in dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = suggest_keno_subset(trial, "markov_bayesian_enhanced")
        if subsets is None:
            return float("-inf")

    return score_from_summary(run_backtest("markov_bayesian_enhanced", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_poisson_mc(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = PoissonMonteCarlo()
    model.setDataPath(dataPath)
    model.setNumOfSimulations(trial.suggest_int('poissonMonteCarloNumberOfSimulations', 100, 1000, step=100))
    model.setWeightFactor(trial.suggest_float('poissonMonteCarloWeightFactor', 0.1, 1.0))
    model.setSortedPrediction(not ("pick3" in dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = suggest_keno_subset(trial, "poisson_mc")
        if subsets is None:
            return float("-inf")

    return score_from_summary(run_backtest("poisson_mc", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_poisson_markov(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = PoissonMarkov()
    model.setDataPath(dataPath)
    weight = trial.suggest_float('poissonMarkovWeight', 0.1, 1.0)
    model.setWeights(poisson_weight=weight, markov_weight=1 - weight)
    model.setNumberOfSimulations(trial.suggest_int('poissonMarkovNumberOfSimulations', 100, 1000, step=100))
    model.setSortedPrediction(not ("pick3" in dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = suggest_keno_subset(trial, "poisson_markov")
        if subsets is None:
            return float("-inf")

    return score_from_summary(run_backtest("poisson_markov", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_laplace_mc(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = LaplaceMonteCarlo()
    model.setDataPath(dataPath)
    model.setNumOfSimulations(trial.suggest_int('laplaceMonteCarloNumberOfSimulations', 100, 1000, step=100))
    model.setSortedPrediction(not ("pick3" in dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = suggest_keno_subset(trial, "laplace_mc")
        if subsets is None:
            return float("-inf")

    return score_from_summary(run_backtest("laplace_mc", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


def objective_hybrid(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
    model = HybridStatisticalModel()
    model.setDataPath(dataPath)
    model.setSoftMaxTemperature(trial.suggest_float('hybridStatisticalModelSoftMaxTemperature', 0.1, 1.0))
    model.setAlpha(trial.suggest_float('hybridStatisticalModelAlpha', 0.1, 1.0))
    model.setMinOccurrences(trial.suggest_int('hybridStatisticalModelMinOcurrences', 1, 20))
    model.setNumberOfSimulations(trial.suggest_int('hybridStatisticalModelNumberOfSimulations', 100, 1000, step=100))
    model.setSortedPrediction(not ("pick3" in dataset_name))

    subsets = []
    if "keno" in dataset_name:
        subsets = suggest_keno_subset(trial, "hybrid_statistical")
        if subsets is None:
            return float("-inf")

    return score_from_summary(run_backtest("hybrid_statistical", model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back))


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

    # Same lookup Predictor.py's getKenoSubsetSizes does - a single global
    # choice shared by every model/row, not the per-model-prefixed use_X
    # choices the individual objectives above each search independently.
    subset_sizes = [size for size in (5, 6, 7, 8, 9, 10) if bestParams.get(f"use_{size}")]
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

    backtester = Backtester(loader)
    for name, model in models.items():
        backtester.add_model(name, model)

    results = backtester.backtest(
        start_index=start_index,
        end_index=total_rows,
        skipLastColumns=game_cfg["skip_last_columns"],
        years_back=years_back,
        include_baselines=False,
        collect_scores=True,
        verbose=False
    )

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

    total_profit = 0.0
    bet_count = 0

    for day in cached["days"]:
        for subset_size in cached["subset_sizes"]:
            subset = helpers.generate_subset_from_scores(
                day["weighted_scores"], day["weighted_ticket"], subset_size,
                mode=weighted_mode, temperature=weighted_temperature)
            profit = helpers.keno_ticket_profit(subset, day["actual"])
            if profit is not None:
                total_profit += profit
                bet_count += 1

            if day["meta_ticket"] is not None:
                subset = helpers.generate_subset_from_scores(
                    day["meta_scores"], day["meta_ticket"], subset_size,
                    mode=meta_mode, temperature=meta_temperature)
                profit = helpers.keno_ticket_profit(subset, day["actual"])
                if profit is not None:
                    total_profit += profit
                    bet_count += 1

            if day["meta_v2_ticket"] is not None:
                subset = helpers.generate_subset_from_scores(
                    day["meta_v2_scores"], day["meta_v2_ticket"], subset_size,
                    mode=meta_v2_mode, temperature=meta_v2_temperature)
                profit = helpers.keno_ticket_profit(subset, day["actual"])
                if profit is not None:
                    total_profit += profit
                    bet_count += 1

    return total_profit / bet_count if bet_count else float("-inf")


# Maps a -s/--strategies CLI name to its objective + the "use<X>" flag Predictor.py
# reads from bestParams_<game>.json to decide whether to run that model live.
#
# "games" restricts a strategy to the games it actually applies to; omitted (or
# None) means every game. Without this, a game-specific strategy still got a
# study created and every one of --trials trials run against it, each returning
# a constant no-op score - wasted runtime plus a meaningless "Best Score: 0.0"
# in the log and an empty study row in db.sqlite3 for every game it never
# applied to.
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

        parser.add_argument('-d', '--days', type=int, default=31)
        parser.add_argument('-t', '--trials', type=int, default=15)
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
            help='Comma-separated list of games, e.g. "keno,pick3"'
        )

        args = parser.parse_args()

        print_intro()

        current_year = datetime.now().year
        print("Current Year:", current_year)

        daysToRebuild = int(args.days)
        n_trials = int(args.trials)
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
                        }.get(dataset_name, "")
                        dataFetcher.getLatestData(gameName, filePath)
                except Exception as e:
                    print("Failed to fetch data: ", e)

                jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
                existingData = {}
                if os.path.exists(jsonBestParamsFilePath):
                    with open(jsonBestParamsFilePath, "r") as infile:
                        existingData = json.load(infile)

                is_pick3 = "pick3" in dataset_name
                profits = {}

                for strategy_name in strategies:
                    if strategy_name not in STRATEGIES:
                        print(f"Unknown strategy '{strategy_name}', skipping")
                        continue

                    if is_pick3 and strategy_name in DISABLED_FOR_PICK3:
                        print(f"Skipping {strategy_name} for Pick3 - not a positional/per-column model")
                        continue

                    strategy = STRATEGIES[strategy_name]

                    applicable_games = strategy.get("games")
                    if applicable_games and dataset_name not in applicable_games:
                        print(f"Skipping {strategy_name} for {dataset_name} - only applies to: {', '.join(applicable_games)}")
                        continue

                    studyName = f"{dataset_name}_{strategy_name}"

                    study = optuna.create_study(
                        direction='maximize',
                        storage=optunaDatabase,
                        study_name=studyName,
                        load_if_exists=True
                    )

                    objective = lambda trial: strategy["objective"](
                        trial, dataset_name, dataPath, game_cfg, daysToRebuild, years_back
                    )

                    study.optimize(objective, n_trials=n_trials)

                    print(f"Best Parameters for {strategy_name}: ", study.best_params)
                    print(f"Best Score for {strategy_name}: ", study.best_value)

                    profits[strategy_name] = study.best_value
                    existingData.update(study.best_params)

                    # Predictor.py reads these directly for Markov (they're not
                    # tuned separately per non-pick3 game the way the other
                    # models' sortedPrediction is, since Predictor.py derives
                    # that one from the game name at runtime for everyone else).
                    if strategy_name == "Markov":
                        existingData['markovSortedPrediction'] = not is_pick3
                        existingData['markovUsePairScoring'] = is_pick3
                        if not is_pick3:
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
