import os, argparse, json, sys
# Pin the BLAS/OpenMP thread pools before numpy (via optuna) is imported: every
# Backtester worker otherwise inherits a 16-thread OpenBLAS pool whose
# spin-waiting was measured at ~10 cores of pure overhead per single-threaded
# CatBoost fit (load average 45 on 16 cores, 3x slower trials). The boosting
# libraries get their thread count explicitly (setNumThreads) and CatBoost uses
# its own pool, so this only removes waste.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import optuna
import multiprocessing
import time
import signal
import warnings
from multiprocessing import cpu_count

# TPESampler(constant_liar=True) (parallel trials, see open_study) is flagged
# experimental by Optuna; the warning would otherwise print once per trial.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

from art import text2art
from datetime import datetime

from src.Backtester import Backtester, BacktestTimeout
from src.DataLoader import DataLoader
from src.XGBoost import XGBoostPredictor, XGBoostMultiLabelPredictor
from src.LightGBM import LightGBMPredictor, LightGBMMultiLabelPredictor
from src.CatBoost import CatBoostPredictor, CatBoostMultiLabelPredictor
from src.BoostingBase import apply_boosting_params
from src.Command import Command
from src.Helpers import Helpers
from src.DataFetcher import DataFetcher

command = Command()
helpers = Helpers()
dataFetcher = DataFetcher()

LOCK_FILE = os.path.join(os.getcwd(), "process.lock")

# Same per-game configuration HyperoptStatistics.py uses - the Backtester's
# DataLoader needs the real number range/draw size instead of falling back to
# a default, and skip_last_columns/special_column_count decide how the trailing
# bonus/special column(s) are handled (Lotto's bonus number is dropped; the
# Euromillions stars / EuroDreams dream number / VikingLotto super viking are
# modeled independently - see Helpers.run_model_with_special_column).
GAME_CONFIG = {
    "euromillions": {"min": 1, "max": 50, "draw_size": 5, "skip_last_columns": 0, "special_column_count": 2},
    "lotto":        {"min": 1, "max": 45, "draw_size": 6, "skip_last_columns": 1, "special_column_count": 0},
    "eurodreams":   {"min": 1, "max": 40, "draw_size": 6, "skip_last_columns": 0, "special_column_count": 1},
    "keno":         {"min": 1, "max": 80, "draw_size": 20, "skip_last_columns": 0, "special_column_count": 0},
    "pick3":        {"min": 0, "max": 9, "draw_size": 3, "skip_last_columns": 0, "special_column_count": 0},
    "vikinglotto":  {"min": 1, "max": 48, "draw_size": 6, "skip_last_columns": 0, "special_column_count": 1},
    # Joker+: six digits 0-9 in drawn order (positional like pick3, see
    # Helpers.is_positional_game) plus the zodiac sign as a 12-class special
    # column (codes 0..11, Helpers.encode_zodiac) - same entry as
    # HyperoptStatistics.GAME_CONFIG.
    "jokerplus":    {"min": 0, "max": 9, "draw_size": 6, "skip_last_columns": 0, "special_column_count": 1},
}

KENO_SUBSET_VALUES = [5, 6, 7, 8, 9, 10]

# Games with a real payout table - the Backtester computes profit rows for
# them and score_from_summary tunes on profit_per_bet (mirrors
# HyperoptStatistics.PAYOUT_GAMES).
PAYOUT_GAMES = ("keno", "pick3", "jokerplus")


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
    print("Predictor Hyperopt - Boosting")
    print("Licence : MIT License")
    print(ascii_art)
    print("Find best boosting parameters for Predictor")


def suggest_keno_subset(trial, model_name):
    """
    Binary inclusion mask over the 5-10 playable Keno subset sizes for one
    specific model - identical to HyperoptStatistics.py's version, including
    the model-name prefix on every param name so this study's tuned choice
    can't be silently overwritten by another strategy's study when both get
    merged into the same bestParams_<game>.json. Returns None (caller should
    treat the trial as invalid) if the mask selects nothing.
    """
    inclusion_mask = [trial.suggest_categorical(f"{model_name}_use_{v}", [True, False]) for v in KENO_SUBSET_VALUES]
    subset = [v for v, include in zip(KENO_SUBSET_VALUES, inclusion_mask) if include]

    if not subset:
        return None

    return subset


def run_backtest(model_name, model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back,
                 max_seconds=None, refit_every=1, num_workers=None, trial=None):
    """
    Same Backtester-driven evaluation HyperoptStatistics.py uses (rolling
    walk-forward over the last `days_to_rebuild` draws, each day retrained on
    only the data before it), returning that model's compact summary dict.
    Backtester reseeds numpy/random per (day, model), so a given set of
    hyperparameters scores deterministically - no repeat/average needed.
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

    # Streaming pruning: after every completed day the partial summary is
    # scored with the SAME function the final value uses and reported to
    # Optuna; PercentilePruner (see __main__) then stops a trial that sits in
    # the bottom quartile after half the window. Pruned trials are recorded
    # as PRUNED - never as a bad score - so the ranking of finished trials is
    # untouched.
    progress_callback = None
    if trial is not None:
        def progress_callback(iteration, rows):
            partial = backtester.summarize(rows).get("models", {}).get(model_name, {})
            trial.report(score_from_summary(partial), step=iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()

    results = backtester.backtest(
        max_seconds=max_seconds,
        chunksize=max(1, int(refit_every)),
        num_workers=num_workers,
        progress_callback=progress_callback,
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
    Optuna objective value: profit_per_bet where this game has a payout model,
    else avg hits - same rationale as HyperoptStatistics.score_from_summary
    (per-bet so a trial betting fewer subset sizes isn't penalised for placing
    fewer bets; still vulnerable to a single jackpot-tier payout dominating,
    see Backtester.summarize()'s "lucky_strikes").
    """
    if not model_summary:
        return float("-inf")

    profit_per_bet = model_summary.get("profit_per_bet")
    if profit_per_bet is not None:
        return profit_per_bet

    hits = model_summary.get("hits_avg")
    return hits if hits is not None else float("-inf")


def suggest_boosting_params(trial, prefix):
    """
    One shared search space for every boosting model, with each key prefixed
    (see BOOSTING_PARAM_SUFFIXES in src/BoostingBase.py) so the six models'
    tuned values land under their own bestParams_<game>.json keys instead of
    clobbering each other - the same reasoning as
    HyperoptDeepLearning.MODEL_PARAM_PREFIX and suggest_keno_subset below.

    Shared deliberately: the point of running three libraries over two
    formulations is to compare them, which only means anything if each was
    given the same search space rather than one being handed a luckier range.

    Includes the regularisation knobs (subsample / colsample /
    min_child_weight / reg_lambda) that were previously left at library
    defaults - a boosted ensemble on a few hundred draws overfits trivially,
    so those are the most consequential part of the space.
    """
    return {
        f"{prefix}Estimators": trial.suggest_int(f'{prefix}Estimators', 10, 300, step=10),
        f"{prefix}LearningRate": trial.suggest_float(f'{prefix}LearningRate', 0.01, 1.0, log=True),
        # CatBoost grows symmetric (oblivious) trees - every tree is full, so
        # cost scales with 2^depth. Measured on this box across all CatBoost
        # trials: median minutes per trial by depth 1-6: 1-4, 7: 9, 8: 24,
        # 9: 61, 10: 64 (single trials up to 12 hours), while the other two
        # libraries stay at minutes for the whole 1-10 range. The deep trials
        # won studies no more often than chance on a 31-day hits objective.
        f"{prefix}Maxdepth": trial.suggest_int(f'{prefix}Maxdepth', 1, 7 if prefix.startswith("catBoost") else 10),
        f"{prefix}PreviousDraws": trial.suggest_int(f'{prefix}PreviousDraws', 1, 50, step=1),
        f"{prefix}TopK": trial.suggest_int(f'{prefix}TopK', 1, 30),
        f"{prefix}ForceNested": trial.suggest_categorical(f'{prefix}ForceNested', [True, False]),
        f"{prefix}Subsample": trial.suggest_float(f'{prefix}Subsample', 0.5, 1.0),
        f"{prefix}ColsampleByTree": trial.suggest_float(f'{prefix}ColsampleByTree', 0.5, 1.0),
        f"{prefix}MinChildWeight": trial.suggest_float(f'{prefix}MinChildWeight', 1.0, 10.0),
        f"{prefix}RegLambda": trial.suggest_float(f'{prefix}RegLambda', 0.1, 10.0, log=True),
        f"{prefix}SubsetMode": trial.suggest_categorical(f'{prefix}SubsetMode', ["top", "softmax"]),
        f"{prefix}SubsetTemperature": trial.suggest_float(f'{prefix}SubsetTemperature', 0.05, 2.0),
    }


# Wall-clock budget per tuning trial and the CatBoost border count, set from
# the CLI in __main__ (module globals so the shared objective can read them).
# A trial over budget is recorded as PRUNED - see make_boosting_objective.
TRIAL_TIMEOUT_SECONDS = 1200
CATBOOST_BORDER_COUNT = 254
# Refit cadence (days per fit) for the walk-forward evaluation and the pruning
# percentile (0 disables pruning) - set from the CLI in __main__.
REFIT_EVERY = 7
PRUNE_PERCENTILE = 25.0
# Pruning only after half the window has been scored and only once this many
# trials have COMPLETED, so a good configuration with an unlucky first week is
# never judged against too little evidence.
PRUNE_MIN_COMPLETED_TRIALS = 5
# Upper bound on threads handed to one CatBoost fit (scaling flattens beyond a
# handful of threads on ~2000-row problems); workers x threads never exceeds
# the machine.
CATBOOST_MAX_THREADS = 8
# Trial-level parallelism (optimize_study): how many trials of one study run
# at the same time, each in its own process with its own Backtester worker
# pool. 0 = auto: the cores the per-trial workers leave idle (cores // workers,
# i.e. 3 on a 16-core box with the default 5 refit blocks); an explicit N is an
# upper bound on that. CatBoost strategies always run one trial at a time -
# they use the idle cores as fit threads instead (measured to scale, see
# make_boosting_objective), and their fits are the memory-hungry ones.
PARALLEL_TRIALS = 0
# Memory gate for concurrent launches: a further trial starts only if the
# machine keeps at least MEMORY_RESERVE_GB available after it (its expected
# footprint is the largest RSS measured on the trials already running); below
# MEMORY_HARD_FLOOR_GB the youngest concurrent trial is stopped and recorded
# as pruned rather than letting the cgroup OOM killer pick a victim (this box
# was OOM-killed at 16 GB before). Launches are spaced PARALLEL_RAMP_SECONDS
# apart so a fresh trial has loaded its data before its footprint is read;
# kept short because it caps the parallelism of fast studies (a eurodreams
# trial takes ~10 s).
MEMORY_RESERVE_GB = 2.0
MEMORY_HARD_FLOOR_GB = 1.0
PARALLEL_RAMP_SECONDS = 3
# Stamped on every trial (user_attrs): the cost gate in make_boosting_objective
# only trusts timeouts recorded under the same tag - same machine, window,
# cadence and budget - so a faster box or a larger --trial-timeout starts from
# a clean slate. Set in __main__.
BUDGET_TAG = ""
# Set by the objective, inside a trial process, when the cost gate skipped the
# trial without evaluating it; optimize_study then doesn't count it.
_LAST_TRIAL_SKIPPED = False
EXIT_TRIAL_SKIPPED = 3


def fit_cost(prefix, params):
    """
    Monotone proxy for how long one fit of this configuration takes, only ever
    compared within ONE study (same game, formulation and library). Histogram
    boosting pays roughly rows x features per tree, so the drivers are
    estimators x window (features are window x draw size). Depth is a minor
    factor for XGBoost/LightGBM (keno XGBoost, measured: depth-1 trials at
    5200 timed out, depth-10 trials at 880 finished) but the dominant one for
    CatBoost's symmetric trees (2^depth leaves, see suggest_boosting_params).
    """
    estimators = params[f"{prefix}Estimators"]
    window = params[f"{prefix}PreviousDraws"]
    depth = params[f"{prefix}Maxdepth"]
    if prefix.startswith("catBoost"):
        return float(estimators * window * (2 ** depth))
    return float(estimators * window * (1 + 0.1 * depth))


def known_timeout_cost_floor(study, tag):
    """
    Cheapest configuration of this study that actually hit the trial budget
    under the same budget tag, or None. Only real timeouts count - trials the
    gate itself skipped are predictions, not evidence.
    """
    floor = None
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,)):
        attrs = t.user_attrs
        if attrs.get("timeout") and attrs.get("budget_tag") == tag and "fit_cost" in attrs:
            floor = attrs["fit_cost"] if floor is None else min(floor, attrs["fit_cost"])
    return floor


def plan_workers(days_to_rebuild):
    """(cores, refit blocks, workers per trial): one worker per block, capped by cores."""
    blocks = max(1, -(-days_to_rebuild // max(1, REFIT_EVERY)))
    cores = max(1, cpu_count() - 1)
    return cores, blocks, max(1, min(cores, blocks))


def parallel_trials_for(prefix, days_to_rebuild):
    """Concurrent trials for one strategy - see PARALLEL_TRIALS."""
    if prefix.startswith("catBoost"):
        return 1
    cores, _, workers = plan_workers(days_to_rebuild)
    auto = max(1, cores // workers)
    return auto if PARALLEL_TRIALS <= 0 else max(1, min(PARALLEL_TRIALS, auto))


def make_boosting_objective(model_class, prefix, backtest_name):
    """
    Builds the Optuna objective for one boosting model. All six differ only in
    which class gets instantiated and which key prefix its params are stored
    under, so they share one objective body - no per-model copies to keep in
    sync.
    """
    def objective(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
        global _LAST_TRIAL_SKIPPED
        _LAST_TRIAL_SKIPPED = False
        params = suggest_boosting_params(trial, prefix)
        model = model_class()
        apply_boosting_params(model, params, prefix)
        model.setDataPath(dataPath)

        # Cost gate. The first ~10 trials of a study are random (TPE start-up)
        # and on the big games about half the search space cannot finish one
        # refit block inside the budget (keno LightGBM: 6 of the first 7
        # trials timed out, 20 min each, every one with a larger estimators x
        # window than the one that finished). A configuration at least as
        # expensive as one that already hit the budget on this machine is
        # recorded as pruned immediately instead of burning the budget again;
        # optimize_study doesn't count it toward --trials, so the study still
        # evaluates the requested number of real configurations.
        cost = fit_cost(prefix, params)
        trial.set_user_attr("fit_cost", cost)
        trial.set_user_attr("budget_tag", BUDGET_TAG)
        trial.set_user_attr("worker_pid", os.getpid())
        floor = known_timeout_cost_floor(trial.study, BUDGET_TAG)
        if floor is not None and cost >= floor:
            reason = (f"fit cost {cost:.0f} >= {floor:.0f} of a trial that hit the "
                      f"{TRIAL_TIMEOUT_SECONDS}s budget")
            trial.set_user_attr("skipped", reason)
            _LAST_TRIAL_SKIPPED = True
            print(f"Trial {trial.number} skipped (predicted timeout): {reason}")
            raise optuna.TrialPruned()

        # Positional games (Pick3: digit order decides straight/box/pair
        # payouts; Joker+: leading/trailing runs of six digits) keep their
        # digits in drawn order instead of being sorted/deduplicated - see
        # Helpers.is_positional_game.
        model.setSortedPrediction(not helpers.is_positional_game(dataset_name))

        # Refit cadence (BoostingBase.setRefitEvery): one fit per block of
        # REFIT_EVERY consecutive days instead of one per day; a block's days
        # share a fit so they run on one worker (the Backtester's chunksize),
        # one worker per block. Library threads stay at ONE: giving XGBoost
        # the spare cores instead was measured to make a 3-trial study 5x
        # SLOWER (2.5 -> 12.5 min) - its OpenMP pool spins up to all cores
        # per worker and thrashes the box on these tiny fits, the same
        # pathology as the serve-path fix in Predictor.py. The cadence's
        # saving is the fewer fits, not per-fit parallelism.
        model.setRefitEvery(REFIT_EVERY)
        cores, blocks, num_workers = plan_workers(days_to_rebuild)
        # CatBoost genuinely scales with threads on this data (one fit: 4
        # threads = 3.1x faster) and is the slow library, so with fewer
        # workers than cores it gets the spare ones (CATBOOST_THREADS cap);
        # XGBoost/LightGBM get 1 - see the measured slowdown above.
        if prefix.startswith("catBoost"):
            model.setNumThreads(max(1, min(CATBOOST_MAX_THREADS, cores // num_workers)))
        else:
            model.setNumThreads(1)
        # Never persist during tuning: many workers would race on the same
        # path, and a tuning-trial fit isn't worth keeping anyway.
        model.setSaveModels(False)
        if prefix.startswith("catBoost"):
            # Not a tuned knob - a run-level speed setting (see
            # BOOSTING_PARAM_SUFFIXES "BorderCount"), also written into
            # bestParams below so Predictor serves the same value.
            model.setBorderCount(CATBOOST_BORDER_COUNT)

        subsets = []
        if "keno" in dataset_name:
            subsets = suggest_keno_subset(trial, backtest_name)
            if subsets is None:
                return float("-inf")

        started = time.time()
        try:
            summary = run_backtest(
                backtest_name, model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back,
                max_seconds=TRIAL_TIMEOUT_SECONDS, refit_every=REFIT_EVERY, num_workers=num_workers,
                trial=trial if PRUNE_PERCENTILE > 0 else None)
        except BacktestTimeout as e:
            # One pathological combination must not cost the study hours -
            # Optuna records the trial as PRUNED and moves on. The timeout is
            # stamped on the trial so the cost gate above skips anything at
            # least as expensive from now on.
            trial.set_user_attr("timeout", True)
            trial.set_user_attr("seconds", round(time.time() - started, 1))
            print(f"Trial {trial.number} pruned: {e}")
            raise optuna.TrialPruned()
        trial.set_user_attr("seconds", round(time.time() - started, 1))
        return score_from_summary(summary)

    objective.prefix = prefix  # optimize_study sizes the parallelism per library
    return objective


# Maps a -s/--strategies CLI name to its objective + the "use<X>" flag
# Predictor.py reads from bestParams_<game>.json to decide whether to run that
# model live - same structure as HyperoptStatistics.STRATEGIES.
#
# Three libraries x two formulations (per-position multiclass vs multi-label
# set membership), each tracked as its own row in Predictor.py. Prefixes match
# Predictor.BOOSTING_MODELS exactly; "xgBoost"/"useBoost" are kept as-is
# because they already exist in every bestParams_<game>.json.
#
# An optional "games" tuple restricts a strategy to the games it applies to
# (same convention as HyperoptStatistics.STRATEGIES); omitted means every game.
# The multi-label models exclude the positional games (Pick3, Joker+ -
# Helpers.is_positional_game) - they model set membership, which can't
# represent digit order or repeated digits. The old name stays as an alias.
NON_POSITIONAL_GAMES = tuple(g for g in GAME_CONFIG if not helpers.is_positional_game(g))
NON_PICK3_GAMES = NON_POSITIONAL_GAMES

STRATEGIES = {
    "XGBoost": {
        "objective": make_boosting_objective(XGBoostPredictor, "xgBoost", "xgboost"),
        "use_key": "useBoost"},
    "XGBoostMultiLabel": {
        "objective": make_boosting_objective(XGBoostMultiLabelPredictor, "xgBoostMl", "xgboost_ml"),
        "use_key": "useXgBoostMultiLabel", "games": NON_POSITIONAL_GAMES},
    "LightGBM": {
        "objective": make_boosting_objective(LightGBMPredictor, "lightGbm", "lightgbm"),
        "use_key": "useLightGbm"},
    "LightGBMMultiLabel": {
        "objective": make_boosting_objective(LightGBMMultiLabelPredictor, "lightGbmMl", "lightgbm_ml"),
        "use_key": "useLightGbmMultiLabel", "games": NON_POSITIONAL_GAMES},
    "CatBoost": {
        "objective": make_boosting_objective(CatBoostPredictor, "catBoost", "catboost"),
        "use_key": "useCatBoost"},
    "CatBoostMultiLabel": {
        "objective": make_boosting_objective(CatBoostMultiLabelPredictor, "catBoostMl", "catboost_ml"),
        "use_key": "useCatBoostMultiLabel", "games": NON_POSITIONAL_GAMES},
}

# Maps a STRATEGIES key to the exact "name" Predictor.py gives that model's
# prediction entry, so the backtest score saved here is looked up directly by
# Helpers.count_number_frequencies_from_new_prediction when weighting
# WeightedEnsemble Model's vote.
STRATEGY_DISPLAY_NAMES = {
    "XGBoost": "XGBoost Model",
    "XGBoostMultiLabel": "XGBoostMultiLabel Model",
    "LightGBM": "LightGBM Model",
    "LightGBMMultiLabel": "LightGBMMultiLabel Model",
    "CatBoost": "CatBoost Model",
    "CatBoostMultiLabel": "CatBoostMultiLabel Model",
}


def make_storage(url):
    """
    One RDBStorage per process - a SQLAlchemy engine must not be used across a
    fork. The sqlite busy timeout is raised from Python's 5 s default: with
    concurrent trial processes each reporting a partial score per completed
    day, a write can briefly find the file locked and should wait, not raise.
    """
    return optuna.storages.RDBStorage(url, engine_kwargs={"connect_args": {"timeout": 60}})


def open_study(study_name, storage, days_to_rebuild, parallel, quiet=False):
    """
    Study handle with this run's pruner and sampler. Both live in the process,
    not the db, so every trial process builds the same ones. constant_liar
    makes TPE treat trials other processes are still running as pessimistic
    observations, so concurrent trials don't sample the same neighbourhood
    (Optuna's documented setting for parallel optimization).
    """
    # Conservative by construction - bottom quartile only, only after half the
    # window, only once 5 trials completed.
    pruner = (optuna.pruners.PercentilePruner(
                  PRUNE_PERCENTILE,
                  n_startup_trials=PRUNE_MIN_COMPLETED_TRIALS,
                  n_warmup_steps=max(1, days_to_rebuild // 2))
              if PRUNE_PERCENTILE > 0 else optuna.pruners.NopPruner())
    sampler = optuna.samplers.TPESampler(constant_liar=parallel > 1)
    verbosity = optuna.logging.get_verbosity()
    if quiet:
        # Every trial process re-opens the study; one "Using an existing
        # study" line per trial is noise.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        return optuna.create_study(
            direction='maximize',
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler,
        )
    finally:
        optuna.logging.set_verbosity(verbosity)


def fail_stale_running_trials(study):
    """
    process.lock guarantees a single HyperoptBoost at a time, so a trial still
    RUNNING when its study is opened was left behind by a killed run (Ctrl+C,
    reboot, OOM). Marked failed: constant_liar would otherwise treat it as live
    forever.
    """
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        print(f"Marking stale trial {t.number} of {study.study_name} as failed (left by an earlier run)")
        study.tell(t.number, state=optuna.trial.TrialState.FAIL, skip_if_finished=True)


def _trial_process(study_name, strategy_name, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back,
                   storage_url, parallel):
    """
    Body of one trial process (forked by optimize_study): open the study, run
    exactly one trial, exit. The module globals set in __main__ are inherited
    through the fork. Exit code EXIT_TRIAL_SKIPPED tells the parent the cost
    gate skipped this trial without evaluating it.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    try:
        # Die with the coordinator (Linux prctl PR_SET_PDEATHSIG): a killed or
        # crashed parent must not leave trial processes fitting for a run
        # that is gone. SIGTERM lands in _exit_on_sigterm below, which unwinds
        # through the Backtester's `with Pool` and so terminates the workers.
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass
    study = open_study(study_name, make_storage(storage_url), days_to_rebuild, parallel, quiet=True)
    strategy = STRATEGIES[strategy_name]
    study.optimize(
        lambda trial: strategy["objective"](trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back),
        n_trials=1)
    if _LAST_TRIAL_SKIPPED:
        sys.exit(EXIT_TRIAL_SKIPPED)


def _exit_on_sigterm(signum, frame):
    """
    SIGTERM (kill <pid>) as an exception instead of an instant death, so the
    coordinator's cleanup runs (trial trees killed, process.lock removed) and
    a trial process unwinds through the Backtester's `with Pool`, terminating
    its workers. Inherited by the forked trial processes - and by their pool
    workers, where it must NOT fire: a worker is inside a library fit, and an
    exception raised from a C callback there corrupted the heap on the way
    out (glibc "corrupted size vs. prev_size"). Workers take the default
    instant death instead.
    """
    if multiprocessing.current_process().name.startswith("ForkPoolWorker"):
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return
    raise SystemExit(128 + signum)


def _read_meminfo():
    values = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) * 1024
    return values


def total_memory_gb():
    try:
        return _read_meminfo()["MemTotal"] / 2 ** 30
    except (OSError, KeyError, ValueError):
        return 0.0


def available_memory_gb():
    """MemAvailable, further capped by the cgroup v2 limit when one is set (this box is a container)."""
    try:
        available = _read_meminfo()["MemAvailable"]
    except (OSError, KeyError, ValueError):
        return float("inf")
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        with open("/sys/fs/cgroup/memory.current") as f:
            current = int(f.read().strip())
        if limit != "max":
            available = min(available, max(0, int(limit) - current))
    except (OSError, ValueError):
        pass
    return available / 2 ** 30


def _process_table():
    """{pid: (ppid, rss_bytes, state)} straight from /proc (ps is unreliable on this box)."""
    table = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        ppid, rss, state = None, 0, "?"
        try:
            with open(f"/proc/{name}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                    elif line.startswith("State:"):
                        state = line.split()[1]
        except (OSError, ValueError, IndexError):
            continue
        if ppid is not None:
            table[int(name)] = (ppid, rss, state)
    return table


def _descendants(pid, table):
    found, stack = [], [pid]
    while stack:
        parent = stack.pop()
        kids = [p for p, (pp, _, _) in table.items() if pp == parent]
        found.extend(kids)
        stack.extend(kids)
    return found


def tree_rss_gb(pid):
    """Resident memory of a trial process plus its Backtester workers."""
    table = _process_table()
    return sum(table[p][1] for p in [pid] + _descendants(pid, table) if p in table) / 2 ** 30


def kill_tree(pid, grace_seconds=5):
    """
    SIGTERM a trial process together with its Backtester workers - killed on
    their own, the workers would keep fitting as orphans - then SIGKILL what
    survives the grace period.
    """
    table = _process_table()
    pids = _descendants(pid, table) + [pid]
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        table = _process_table()
        if not any(p in table and not table[p][2].startswith("Z") for p in pids):
            return
        time.sleep(0.2)
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass


def optimize_study(study_name, strategy_name, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back,
                   storage_url, n_trials, parallel):
    """
    Runs n_trials evaluated trials of one study, up to `parallel` at a time,
    each in its own forked process (the Backtester hands its worker state to
    the pool through a module global, so two trials can't share a process).

    - Trials the cost gate skipped don't count toward n_trials; attempts are
      capped at 4x n_trials so a study can't spin forever.
    - A further trial launches only when the memory gate allows (see
      MEMORY_RESERVE_GB), taking the largest footprint measured on the running
      trials as the expected cost of one more; under MEMORY_HARD_FLOOR_GB the
      youngest trial is stopped and recorded as pruned.
    - A trial process that dies (exception, OOM kill) leaves its trial
      RUNNING; it is marked failed here so the study never stalls on it, and
      three failures in a row give up on this study instead of the whole run
      (previously one exception aborted every remaining strategy of the game).
    """
    _, _, workers = plan_workers(days_to_rebuild)
    footprint_seen = 0.3 + 0.5 * workers  # expected GB per trial until measured (0.5 GB/worker seen on keno)
    print(f"{study_name}: {n_trials} trials, up to {parallel} at a time x {workers} workers, "
          f"memory reserve {MEMORY_RESERVE_GB:g} GB, {available_memory_gb():.1f} GB available")

    def mark_trial_of(pid, state, note):
        study = open_study(study_name, make_storage(storage_url), days_to_rebuild, parallel, quiet=True)
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
            if t.user_attrs.get("worker_pid") == pid:
                print(f"Trial {t.number} {note}")
                study.tell(t.number, state=state, skip_if_finished=True)

    ctx = multiprocessing.get_context("fork")
    running = {}  # pid -> (Process, launched_at)
    evaluated = attempts = failures_in_a_row = 0
    max_attempts = n_trials * 4
    aborted = False

    def launch_allowed():
        if not running:
            return True
        newest = max(launched_at for _, launched_at in running.values())
        if time.time() - newest < PARALLEL_RAMP_SECONDS:
            return False
        return available_memory_gb() - footprint_seen >= MEMORY_RESERVE_GB

    try:
        while True:
            for pid, (proc, _) in list(running.items()):
                if proc.is_alive():
                    continue
                proc.join()
                del running[pid]
                if proc.exitcode == EXIT_TRIAL_SKIPPED:
                    continue
                evaluated += 1
                if proc.exitcode == 0:
                    failures_in_a_row = 0
                    continue
                failures_in_a_row += 1
                mark_trial_of(pid, optuna.trial.TrialState.FAIL,
                              f"marked failed - its process exited with code {proc.exitcode}")
                if failures_in_a_row >= 3:
                    print(f"{study_name}: three trial processes failed in a row - giving up on this study")
                    aborted = True

            want_more = not aborted and evaluated + len(running) < n_trials and attempts < max_attempts
            if not want_more and not running:
                break

            if want_more and len(running) < parallel and launch_allowed():
                sys.stdout.flush()
                sys.stderr.flush()
                proc = ctx.Process(
                    target=_trial_process, name=f"trial:{study_name}",
                    args=(study_name, strategy_name, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back,
                          storage_url, parallel))
                proc.start()
                running[proc.pid] = (proc, time.time())
                attempts += 1
                continue

            if running:
                settled = [pid for pid, (_, launched_at) in running.items()
                           if time.time() - launched_at >= PARALLEL_RAMP_SECONDS]
                footprint_seen = max([footprint_seen] + [tree_rss_gb(pid) for pid in settled])
                if len(running) > 1 and available_memory_gb() < MEMORY_HARD_FLOOR_GB:
                    youngest = max(running, key=lambda p: running[p][1])
                    footprint_seen = max(footprint_seen, tree_rss_gb(youngest))
                    print(f"{study_name}: {available_memory_gb():.1f} GB available - stopping the youngest "
                          f"concurrent trial (process {youngest}) before the OOM killer does")
                    kill_tree(youngest)
                    running.pop(youngest)[0].join()
                    mark_trial_of(youngest, optuna.trial.TrialState.PRUNED, "pruned - stopped under memory pressure")
            time.sleep(2)
    except BaseException:
        # Ctrl+C or a crash of the coordinator: never leave trial processes
        # (and their worker pools) computing for a run that is gone.
        for pid, (proc, _) in running.items():
            kill_tree(pid)
            proc.join(timeout=10)
        raise


if __name__ == "__main__":
    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)

    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _exit_on_sigterm)

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
        parser.add_argument(
            '-t', '--trials', type=int, default=15,
            help='Evaluated trials per study and run (studies persist in db.sqlite3, so runs add up). '
                 'Trials the cost gate skips as a predicted timeout are not counted.')
        parser.add_argument(
            '--trial-timeout', type=int, default=1200,
            help='Wall-clock budget in seconds per tuning trial; a trial over budget is '
                 'pruned (recorded, not scored), and every later configuration at least as '
                 'expensive (fit_cost: estimators x window, x 2^depth for CatBoost) is skipped '
                 'immediately as a predicted timeout - but only against timeouts recorded on this '
                 'machine with the same window, cadence and budget. Default 20 minutes - the '
                 'study medians are minutes, the outliers were hours.')
        parser.add_argument(
            '--refit-every', type=int, default=7,
            help='Refit cadence for the walk-forward evaluation: one boosted fit per block of N '
                 'consecutive days (block-aligned, never leaky - see BoostingBase.setRefitEvery). '
                 '1 = refit every day (production behavior, slowest); 7 (default) measured ~7x '
                 'faster on the CatBoost strategies (eurodreams CatBoostMultiLabel, 2 trials x 21 '
                 'days: 7.5 min -> 66 s); equal to --days = one fit per trial. Same cadence for '
                 'every trial, so the ranking is unaffected.')
        parser.add_argument(
            '--prune-percentile', type=float, default=25.0,
            help='Optuna PercentilePruner threshold: stop a trial whose partial score after half the '
                 'window is in the bottom N%% of completed trials (needs 5 completed trials first). '
                 '0 disables pruning.')
        parser.add_argument(
            '--catboost-border-count', type=int, default=254,
            help='CatBoost border_count (split candidates per feature), CatBoost\'s own default '
                 '254. Experimentation knob only - measured no speed or ranking difference on '
                 'this data (multi-hot features have a single split candidate anyway). Written '
                 'to bestParams_<game>.json as catBoost*BorderCount so Predictor serves the '
                 'same value.')
        parser.add_argument(
            '--parallel-trials', type=int, default=0,
            help='Upper bound on trials of one study evaluated at the same time, each in its own '
                 'process with its own worker pool. 0 (default) = auto: the cores the per-trial '
                 'workers leave idle (cores // workers; 3 on this box), so XGBoost/LightGBM studies '
                 'that ran on 5 of 15 cores fill the machine. CatBoost strategies always run one '
                 'trial at a time and use the idle cores as fit threads instead. 1 = off. A second '
                 'trial only starts while the memory gate allows (see --memory-reserve-gb).')
        parser.add_argument(
            '--memory-reserve-gb', type=float, default=2.0,
            help='Memory that must stay available after launching one more concurrent trial (its '
                 'footprint is measured on the running trials); below 1 GB available the youngest '
                 'concurrent trial is stopped and recorded as pruned instead of risking the OOM killer.')
        parser.add_argument(
            '-s', '--strategies',
            type=str,
            default=",".join(STRATEGIES.keys()),
            help='Comma-separated list of strategies, e.g. "XGBoost"'
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
        CATBOOST_BORDER_COUNT = int(args.catboost_border_count)
        REFIT_EVERY = max(1, int(args.refit_every))
        PRUNE_PERCENTILE = max(0.0, float(args.prune_percentile))
        PARALLEL_TRIALS = max(0, int(args.parallel_trials))
        MEMORY_RESERVE_GB = max(0.0, float(args.memory_reserve_gb))
        BUDGET_TAG = (f"{cpu_count()}c-{total_memory_gb():.0f}g-d{daysToRebuild}"
                      f"-r{REFIT_EVERY}-t{TRIAL_TIMEOUT_SECONDS}")
        try:
            # Logs are read with tail -f while a run is redirected to a file.
            sys.stdout.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
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

                profits = {}

                for strategy_name in strategies:
                    if strategy_name not in STRATEGIES:
                        print(f"Unknown strategy '{strategy_name}', skipping")
                        continue

                    strategy = STRATEGIES[strategy_name]

                    applicable_games = strategy.get("games")
                    if applicable_games and dataset_name not in applicable_games:
                        print(f"Skipping {strategy_name} for {dataset_name} - only applies to: {', '.join(applicable_games)}")
                        continue

                    studyName = f"{dataset_name}_{strategy_name}"
                    parallel = parallel_trials_for(strategy["objective"].prefix, daysToRebuild)

                    study = open_study(studyName, make_storage(optunaDatabase), daysToRebuild, parallel)
                    fail_stale_running_trials(study)

                    optimize_study(studyName, strategy_name, dataset_name, dataPath, game_cfg, daysToRebuild,
                                   years_back, optunaDatabase, n_trials, parallel)

                    # Fresh handle: the trials were written by the trial processes.
                    study = open_study(studyName, make_storage(optunaDatabase), daysToRebuild, parallel, quiet=True)

                    # Run-level setting, not a tuned param: recorded even when
                    # the study below yields nothing, so a non-default
                    # --catboost-border-count is never silently dropped.
                    if strategy_name.startswith("CatBoost"):
                        catPrefix = "catBoostMl" if "MultiLabel" in strategy_name else "catBoost"
                        existingData[f"{catPrefix}BorderCount"] = CATBOOST_BORDER_COUNT

                    # With per-trial budgets a study can end with every trial
                    # pruned - study.best_params would then raise and abort
                    # the remaining strategies of this game. Keep whatever was
                    # tuned before and move on.
                    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                        print(f"No completed trials for {strategy_name} (all pruned/failed) - keeping existing params")
                        continue

                    print(f"Best Parameters for {strategy_name}: ", study.best_params)
                    print(f"Best Score for {strategy_name}: ", study.best_value)

                    profits[strategy_name] = study.best_value
                    existingData.update(study.best_params)

                    # Predictor.py gates this model on its use_key; hyperopt
                    # never disables a model (the same policy the statistical
                    # hyperopt follows) - every method keeps producing its own
                    # tracked row so real-life results stay comparable.
                    if strategy["use_key"]:
                        existingData[strategy["use_key"]] = True

                # The score is only used to weight this model's vote in
                # Helpers.count_number_frequencies_from_new_prediction's
                # combined numberFrequency view - never to disable a model.
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
            helpers.git_push(commit_message="Saving latest boosting hyperopt")
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()
