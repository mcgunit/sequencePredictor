import os, argparse, json
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.Backtester import Backtester
from src.DataLoader import DataLoader
from src.Helpers import Helpers
from src.ModelFactory import BASE_MODEL_NAMES, build_models

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


def fit_meta_model(results, model_names, min_number, max_number, scores_suffix, actual_key, label):
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

    meta_model = LogisticRegression(class_weight="balanced", max_iter=1000)
    meta_model.fit(X_train, y_train)

    if len(test_rows) > 0 and len(set(y_test.tolist())) > 1:
        y_pred = meta_model.predict(X_test)
        y_proba = meta_model.predict_proba(X_test)[:, 1]
        print(f"{label}: held-out accuracy={accuracy_score(y_test, y_pred):.4f} "
              f"auc={roc_auc_score(y_test, y_proba):.4f} (train_days={len(train_rows)}, test_days={len(test_rows)})")
    else:
        print(f"{label}: not enough held-out days/classes to report accuracy/AUC "
              f"(train_days={len(train_rows)}, test_days={len(test_rows)})")

    X_full, y_full = build_training_table(results, model_names, min_number, max_number, scores_suffix, actual_key)
    meta_model.fit(X_full, y_full)

    return meta_model


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
    meta_model = fit_meta_model(
        results, model_names, game_cfg["min"], game_cfg["max"],
        scores_suffix="_scores", actual_key=main_actual_key, label=dataset_name)

    artifact = {
        "model": meta_model,
        "feature_names": model_names,
        "min_number": game_cfg["min"],
        "max_number": game_cfg["max"],
        "draw_size": game_cfg["draw_size"],
    }

    if specialColumnCount > 0:
        special_min, special_max = determine_special_range(dataPath, specialColumnCount)
        special_meta_model = fit_meta_model(
            results, model_names, special_min, special_max,
            scores_suffix="_special_scores", actual_key="actual_special", label=f"{dataset_name} (special column)")

        artifact.update({
            "special_model": special_meta_model,
            "special_min_number": special_min,
            "special_max_number": special_max,
            "special_draw_size": specialColumnCount,
        })

    modelDir = os.path.join(path, "data", "models", dataset_name)
    os.makedirs(modelDir, exist_ok=True)
    artifact_path = os.path.join(modelDir, "meta_learner.joblib")
    joblib.dump(artifact, artifact_path)
    print(f"{dataset_name}: saved meta-learner to {artifact_path}")


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
