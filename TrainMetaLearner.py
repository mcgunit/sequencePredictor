import os, argparse, json, functools
import numpy as np
import joblib

from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, roc_auc_score

from src.Backtester import Backtester
from src.DataLoader import DataLoader
from src.Helpers import Helpers
from src.ModelFactory import BASE_MODEL_NAMES, build_models
from src.QuantumModels import fit_quantum_kernel, fit_quantum_vqc

from HyperoptStatistics import GAME_CONFIG

helpers = Helpers()


def build_training_table(rows, model_names, min_number, max_number, scores_suffix="_scores", actual_key="actual"):
    """
    Turns Backtester rows (from a collect_scores=True backtest) into a flat
    supervised table: one row per (backtest day, number in [min_number,
    max_number]), with a feature per base model's score for that number (0 if
    unscored) and a binary label for whether that number was actually drawn
    that day.

    scores_suffix / actual_key let the same function build either the main
    table (model_scores, actual_main) or the special-column table
    (model_special_scores, actual_special) for games with a special column -
    see Backtester.py's collect_scores split and actual_main/actual_special.
    Using the plain "actual" (sorted, main+special merged) here would be
    wrong for special-column games, since main and special ranges often
    overlap (e.g. Euromillions main 1-50 and star 1-12 both include values
    like 9) and merging them would mislabel a special-only number as drawn
    for the main table too.
    """
    features = []
    labels = []

    for row in rows:
        actual = set(row.get(actual_key, []))
        for number in range(min_number, max_number + 1):
            feature = [row.get(f"{name}{scores_suffix}", {}).get(number, 0.0) for name in model_names]
            features.append(feature)
            labels.append(1 if number in actual else 0)

    return np.array(features, dtype=float), np.array(labels, dtype=int)


def determine_special_range(dataPath, special_column_count):
    """
    Euromillions/EuroDreams/VikingLotto special columns (star numbers, dream
    number, super viking) don't have a hardcoded range anywhere in the repo -
    every model just works with whatever values occur in the data. Derive it
    empirically here so the meta-learner's special-column feature grid
    (range(special_min, special_max + 1)) covers the real range instead of
    guessing.
    """
    _, _, _, _, _, numbers, _, _ = helpers.load_data(dataPath, specialColumnCount=special_column_count)
    return int(numbers.min()), int(numbers.max())


def fit_logistic_regression(X, y):
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    return model


def fit_gradient_boosting(X, y):
    """
    The "lens diversity" companion to fit_logistic_regression (see README's
    Ideas section) - a tree-based classifier sees nonlinear interactions
    between base models' scores a linear model can't, so its errors won't
    necessarily correlate with LogisticRegression's. GradientBoostingClassifier
    has no built-in class_weight, so class balance is applied via sample_weight
    instead, to match fit_logistic_regression's class_weight="balanced".
    """
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y, sample_weight=compute_sample_weight(class_weight="balanced", y=y))
    return model


def fit_meta_model(results, model_names, min_number, max_number, scores_suffix, actual_key, label, fit_func):
    """
    Splits results into a walk-forward holdout (last 20% of days - Backtester
    preserves day order via pool.imap, not imap_unordered, so this is a
    genuine train-before/test-after split), fits+reports on that split for an
    honest sanity check, then refits on the full window before returning -
    standard practice once you're ready to persist, not a leakage shortcut.
    """
    split_index = int(len(results) * 0.8)
    train_rows, test_rows = results[:split_index], results[split_index:]

    X_train, y_train = build_training_table(train_rows, model_names, min_number, max_number, scores_suffix, actual_key)
    X_test, y_test = build_training_table(test_rows, model_names, min_number, max_number, scores_suffix, actual_key)

    meta_model = fit_func(X_train, y_train)

    if len(test_rows) > 0 and len(set(y_test.tolist())) > 1:
        y_pred = meta_model.predict(X_test)
        y_proba = meta_model.predict_proba(X_test)[:, 1]
        print(f"{label}: held-out accuracy={accuracy_score(y_test, y_pred):.4f} "
              f"auc={roc_auc_score(y_test, y_proba):.4f} (train_days={len(train_rows)}, test_days={len(test_rows)})")
    else:
        print(f"{label}: not enough held-out days/classes to report accuracy/AUC "
              f"(train_days={len(train_rows)}, test_days={len(test_rows)})")

    X_full, y_full = build_training_table(results, model_names, min_number, max_number, scores_suffix, actual_key)
    meta_model = fit_func(X_full, y_full)

    return meta_model


# The backtest with collect_scores=True is the most expensive stage of the
# weekly pipeline (XGBoost genuinely retrains per day), and HyperoptQuantum.py
# runs the identical collection minutes before this script does in
# runHyperopt.sh. These helpers let whichever script collects first persist
# the table so the other reuses it instead of recomputing bit-identical rows
# (Backtester reseeds per (day index, model), so same data + same base params
# = same table).
#
# Validity is deliberately strict - all three must match, else recollect:
# - total_rows: any new draw shifts every per-day seed, so a single fresh
#   draw invalidates the cache (bounds staleness to same-data runs);
# - days: the cache must cover at least the requested window (the newest
#   days are a suffix slice);
# - base params: the subset of bestParams the base models are built from
#   (ModelFactory.build_models reads only markov*/poisson*/laplace*/xgBoost*
#   keys; quantum/meta/subset keys merged later in the pipeline don't touch
#   the base-model scores, so they must NOT invalidate the cache).
BASE_PARAM_PREFIXES = ("markov", "poisson", "laplace", "xgBoost")


def base_param_subset(bestParams):
    return {k: v for k, v in (bestParams or {}).items() if k.startswith(BASE_PARAM_PREFIXES)}


def meta_table_cache_path(path, dataset_name):
    cacheDir = os.path.join(path, "data", "hyperOptCache")
    os.makedirs(cacheDir, exist_ok=True)
    return os.path.join(cacheDir, f"meta_score_table_{dataset_name}.joblib")


def save_meta_score_table(path, dataset_name, results, model_names, days_back, total_rows, bestParams):
    try:
        joblib.dump({
            "results": results,
            "model_names": model_names,
            "days": days_back,
            "total_rows": total_rows,
            "base_params": base_param_subset(bestParams),
        }, meta_table_cache_path(path, dataset_name))
    except Exception as e:
        print(f"Could not persist the {dataset_name} score-table cache (continuing): {e}")


def load_meta_score_table(path, dataset_name, days_back, total_rows, bestParams):
    """Returns (results, model_names) sliced to the newest days_back days, or None."""
    cachePath = meta_table_cache_path(path, dataset_name)
    if not os.path.exists(cachePath):
        return None
    try:
        cache = joblib.load(cachePath)
    except Exception:
        return None
    if cache.get("total_rows") != total_rows:
        return None
    if cache.get("days", 0) < days_back or len(cache.get("results") or []) == 0:
        return None
    if cache.get("base_params") != base_param_subset(bestParams):
        return None
    results = cache["results"]
    sliced = results[-min(len(results), days_back):] if days_back else results
    print(f"{dataset_name}: reusing the cached base-model score table "
          f"({len(sliced)} of {len(results)} cached days) - skipping the backtest")
    return sliced, cache["model_names"]


def train_meta_learner(dataset_name, game_cfg, path, days_back):
    if dataset_name == "pick3":
        print("Skipping Pick3 - positional game, a per-number score ranking has no notion of digit order.")
        return

    dataPath = os.path.join(path, "data", "trainingData", dataset_name)
    bestParamsPath = os.path.join(path, f"bestParams_{dataset_name}.json")
    bestParams = {}
    if os.path.exists(bestParamsPath):
        with open(bestParamsPath, "r") as infile:
            bestParams = json.load(infile)

    specialColumnCount = game_cfg["special_column_count"]

    loader = DataLoader()
    loader.setDataPath(dataPath)
    loader.setGameRange(game_cfg["min"], game_cfg["max"])
    loader.setDrawSize(game_cfg["draw_size"])

    numbers, _, _ = loader.load_numbers(skipLastColumns=game_cfg["skip_last_columns"])
    total_rows = len(numbers)
    if total_rows == 0:
        print(f"No data found for {dataset_name}, skipping.")
        return

    start_index = max(0, total_rows - days_back)

    cached = load_meta_score_table(path, dataset_name, days_back, total_rows, bestParams)
    if cached is not None:
        results, model_names = cached
    else:
        models = build_models(dataPath, bestParams, is_pick3=False)
        model_names = [name for name in BASE_MODEL_NAMES if name in models]

        backtester = Backtester(loader)
        for name, model in models.items():
            backtester.add_model(name, model)

        print(f"\n{dataset_name}: backtesting {total_rows - start_index} days with {len(models)} base models to collect training data...")
        results = backtester.backtest(
            start_index=start_index,
            end_index=total_rows,
            skipLastColumns=game_cfg["skip_last_columns"],
            special_column_count=specialColumnCount,
            include_baselines=False,
            collect_scores=True,
            verbose=True
        )
        if results:
            save_meta_score_table(path, dataset_name, results, model_names, days_back, total_rows, bestParams)

    if not results:
        print(f"No backtest rows produced for {dataset_name}, skipping.")
        return

    # Special-column games (Euromillions/EuroDreams/VikingLotto) need their
    # main numbers and special column(s) modeled - and scored - completely
    # separately (own range, own labels), the same way every individual
    # model already keeps them separate via Helpers.run_model_with_special_column.
    # A single flat model over the combined range would only ever have been
    # trained on whichever range collect_scores happened to score (see
    # Backtester.py's collect_scores fix) and couldn't produce a
    # correctly-ranged special-column prediction either way.
    main_actual_key = "actual_main" if specialColumnCount > 0 else "actual"

    special_min = special_max = None
    if specialColumnCount > 0:
        special_min, special_max = determine_special_range(dataPath, specialColumnCount)

    modelDir = os.path.join(path, "data", "models", dataset_name)
    os.makedirs(modelDir, exist_ok=True)

    # "Lens diversity" (see README's Ideas section): a second, independently
    # trained model class per game, added as its own MetaLearnerV2 Model row
    # in Predictor.py rather than replacing MetaLearner Model - both get
    # tracked side by side.
    variants = [
        (fit_logistic_regression, "meta_learner.joblib", dataset_name, {}),
        (fit_gradient_boosting, "meta_learner_v2.joblib", f"{dataset_name} (v2)", {}),
    ]

    # Quantum meta-learners (README's quantum research track): two more
    # lenses over the exact same training table - a quantum-kernel SVC and a
    # variational quantum circuit (see src/QuantumModels.py) - each its own
    # independently tracked Predictor row, never a replacement for the
    # classical rows. Appended to the SAME variants list so the expensive
    # backtest above still runs exactly once and every variant (special-column
    # models included, via the shared fit_meta_model path) fits from the one
    # shared `results`. The README suggested opt-in (default false) back when
    # it assumed a heavy quantum-framework simulation; the numpy 4-qubit
    # statevector implementation trains in minutes, so the flags default ON
    # and exist as an off-switch instead. Hyperparameters are resolved from
    # bestParams_<game>.json here (falling back to src/QuantumModels.py's
    # documented defaults) and bound with functools.partial because
    # fit_meta_model only ever calls fit_func(X, y); the resolved dict also
    # goes into the artifact so a saved model documents what trained it.
    if bestParams.get("useQuantumMetaLearner", True):
        quantum_kernel_params = {
            "quantumKernel_nQubits": bestParams.get("quantumKernel_nQubits", 4),
            "quantumKernel_encodingLayers": bestParams.get("quantumKernel_encodingLayers", 2),
            "quantumKernel_encodingScale": bestParams.get("quantumKernel_encodingScale", 1.0),
            "quantumKernel_C": bestParams.get("quantumKernel_C", 1.0),
            "quantumKernel_maxTrainSamples": bestParams.get("quantumKernel_maxTrainSamples", 2000),
        }
        variants.append((
            functools.partial(fit_quantum_kernel, params=quantum_kernel_params),
            "quantum_meta_learner.joblib", f"{dataset_name} (quantum kernel)",
            quantum_kernel_params,
        ))

    if bestParams.get("useQuantumVqcMetaLearner", True):
        quantum_vqc_params = {
            "quantumVqc_nQubits": bestParams.get("quantumVqc_nQubits", 4),
            "quantumVqc_numLayers": bestParams.get("quantumVqc_numLayers", 2),
            "quantumVqc_encodingScale": bestParams.get("quantumVqc_encodingScale", 1.0),
            "quantumVqc_learningRate": bestParams.get("quantumVqc_learningRate", 0.05),
            "quantumVqc_epochs": bestParams.get("quantumVqc_epochs", 80),
            "quantumVqc_batchSize": bestParams.get("quantumVqc_batchSize", 128),
        }
        variants.append((
            functools.partial(fit_quantum_vqc, params=quantum_vqc_params),
            "quantum_vqc_meta_learner.joblib", f"{dataset_name} (quantum vqc)",
            quantum_vqc_params,
        ))

    for fit_func, artifact_filename, label, variant_params in variants:
        meta_model = fit_meta_model(
            results, model_names, game_cfg["min"], game_cfg["max"],
            scores_suffix="_scores", actual_key=main_actual_key, label=label, fit_func=fit_func)

        artifact = {
            "model": meta_model,
            "feature_names": model_names,
            "min_number": game_cfg["min"],
            "max_number": game_cfg["max"],
            "draw_size": game_cfg["draw_size"],
            # Training metadata (README's persistence requirement): when the
            # artifact was produced and the exact hyperparameters it was
            # fitted with ({} for the classical variants - theirs are fixed
            # in code above). Additive keys only, so Predictor.py's
            # runMetaLearnerVariant serves old and new artifacts unchanged.
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "params": variant_params,
        }

        if specialColumnCount > 0:
            special_meta_model = fit_meta_model(
                results, model_names, special_min, special_max,
                scores_suffix="_special_scores", actual_key="actual_special",
                label=f"{label} (special column)", fit_func=fit_func)

            artifact.update({
                "special_model": special_meta_model,
                "special_min_number": special_min,
                "special_max_number": special_max,
                "special_draw_size": specialColumnCount,
            })

        artifact_path = os.path.join(modelDir, artifact_filename)
        joblib.dump(artifact, artifact_path)
        print(f"{label}: saved meta-learner to {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train Meta-Learner",
        description="Trains a stacking meta-learner over each statistical model's per-number score, per game"
    )
    parser.add_argument(
        "-g", "--games",
        type=str,
        default=",".join(g for g in GAME_CONFIG.keys() if g != "pick3"),
        help='Comma-separated list of games, e.g. "lotto,keno"'
    )
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=300,
        help="How many most-recent draws to backtest for training data"
    )
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    unknown_games = [g for g in games if g not in GAME_CONFIG]
    if unknown_games:
        print(f"Unknown game(s), ignoring: {unknown_games}")

    path = os.getcwd()

    for dataset_name in games:
        if dataset_name not in GAME_CONFIG:
            continue
        try:
            train_meta_learner(dataset_name, GAME_CONFIG[dataset_name], path, args.days)
        except Exception as e:
            print(f"Failed to train meta-learner for {dataset_name}: {e}")
