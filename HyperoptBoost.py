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
from multiprocessing import cpu_count

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


def make_boosting_objective(model_class, prefix, backtest_name):
    """
    Builds the Optuna objective for one boosting model. All six differ only in
    which class gets instantiated and which key prefix its params are stored
    under, so they share one objective body - no per-model copies to keep in
    sync.
    """
    def objective(trial, dataset_name, dataPath, game_cfg, days_to_rebuild, years_back):
        model = model_class()
        apply_boosting_params(model, suggest_boosting_params(trial, prefix), prefix)
        model.setDataPath(dataPath)

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
        blocks = max(1, -(-days_to_rebuild // max(1, REFIT_EVERY)))
        cores = max(1, cpu_count() - 1)
        num_workers = max(1, min(cores, blocks))
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

        try:
            summary = run_backtest(
                backtest_name, model, dataset_name, dataPath, game_cfg, subsets, days_to_rebuild, years_back,
                max_seconds=TRIAL_TIMEOUT_SECONDS, refit_every=REFIT_EVERY, num_workers=num_workers,
                trial=trial if PRUNE_PERCENTILE > 0 else None)
        except BacktestTimeout as e:
            # One pathological combination must not cost the study hours -
            # Optuna records the trial as PRUNED and moves on.
            print(f"Trial {trial.number} pruned: {e}")
            raise optuna.TrialPruned()
        return score_from_summary(summary)

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
            '--trial-timeout', type=int, default=1200,
            help='Wall-clock budget in seconds per tuning trial; a trial over budget is '
                 'pruned (recorded, not scored). Default 20 minutes - the study medians are '
                 'minutes, the outliers were hours.')
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

                    # Pruner is per process (not persisted in the study db):
                    # conservative by construction - bottom quartile only,
                    # only after half the window, only once 5 trials completed.
                    pruner = (optuna.pruners.PercentilePruner(
                                  PRUNE_PERCENTILE,
                                  n_startup_trials=PRUNE_MIN_COMPLETED_TRIALS,
                                  n_warmup_steps=max(1, daysToRebuild // 2))
                              if PRUNE_PERCENTILE > 0 else optuna.pruners.NopPruner())
                    study = optuna.create_study(
                        direction='maximize',
                        storage=optunaDatabase,
                        study_name=studyName,
                        load_if_exists=True,
                        pruner=pruner
                    )

                    objective = lambda trial: strategy["objective"](
                        trial, dataset_name, dataPath, game_cfg, daysToRebuild, years_back
                    )

                    study.optimize(objective, n_trials=n_trials)

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
