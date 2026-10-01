#!/usr/bin/env python3
"""
Irrelevant-feature control (README "Null controls", roadmap item 6): a random
column appended to the meta-learner table must not gain stable importance.

WHY. The meta-learner rows (MetaLearner, MetaLearnerV2, QuantumMetaLearner,
QuantumVQC, ClassicalSVM) are fitted on the base models' per-number scores.
Their held-out metrics say how well they rank; they do not say whether the
weights behind that ranking mean anything. A model that gives weight to a
column provably unrelated to the draw is fitting noise, and then its weights
on the real columns are noise too - the selection effect the null band
measures at the row level, one level down. This control asks each served
variant that question directly.

HOW. For every game, the variants TrainMetaLearner.py serves (the same list,
the same tuned parameters: TrainMetaLearner.meta_learner_variants) are
refitted on the table the Saturday chain cached, with K noise columns
appended: shuffled copies of real base-model columns, so each has exactly a
base model's marginal distribution and no relation to the label or the day.
The lockbox days (lockbox.json, src/Lockbox.py) leave the table first, exactly
as the trainer removes them, so this IS the trainer's table. R repeats with
fresh noise, the trainer's own chronological 80/20 day split, and permutation
importance (the ranking power lost when one column is scrambled: the drop in
|AUC - 0.5|, see src/FeatureControl.py) on both sides of the split:

  training rows   a variant FITS NOISE when a noise column matters at least a
                  tenth as much as an average real column and the pooled noise
                  importance is more than two standard errors above zero
  held-out days   the noise columns' importance is the band a real base model
                  must clear to count as carrying more than noise does

The rules are src/FeatureControl.py (self-check: python3 -m src.FeatureControl).

WHAT IT DOES NOT DO. It never recollects the table: a backtest costs hours,
and the control is a reading of the table the trainer used. It runs on Sunday,
a day after the chain, so a cache up to --tolerate-rows draws behind the data
file is accepted and the shortfall recorded. A game without a cached table is
skipped and says so.

OUTPUT. data/controls/features/<game>.json, read by controls.js for the
History page's "Meta-learner feature control" card.

    python3 IrrelevantFeatureControl.py -g lotto,pick3 -r 3 -k 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

from src.DataLoader import DataLoader
from src.Helpers import Helpers
from src.ControlHistories import CONTROL_ROOT
from src.FeatureControl import (append_noise, noise_names, permutation_importance, flat_predict,
                                positional_predict, summarise, describe)
from src.HyperoptRunner import install_sigterm_handler
from src.Lockbox import load as load_lockbox, split_rows as lockbox_split, describe as describe_lockbox, as_json as lockbox_json
from src.ModelFactory import expected_model_names
from HyperoptStatistics import GAME_CONFIG, is_running, create_lock, remove_lock
from TrainMetaLearner import (VARIANT_KEYS, POSITIONAL_CLASSES, build_positional_training_table, build_training_table,
                              fit_position_models, load_meta_score_table, meta_learner_variants, meta_table_kind,
                              positional_positions)

helpers = Helpers()

RESULT_DIR = os.path.join(CONTROL_ROOT, "features")
TRAIN_FRACTION = 0.8          # fit_meta_model's own held-out rule
DEFAULT_DAYS = 300            # the trainer's window, so the cached table qualifies
DEFAULT_MAX_ROWS = 6000       # rows (whole days) the permutation importance is computed on, per side


def load_table(game, cfg, path, days, tolerate_rows):
    """The cached meta-learner table for the game, or None (with the reason printed)."""
    positional = helpers.is_positional_game(game)
    dataPath = os.path.join(path, "data", "trainingData", game)
    bestParamsPath = os.path.join(path, f"bestParams_{game}.json")
    bestParams = {}
    if os.path.exists(bestParamsPath):
        with open(bestParamsPath, "r") as infile:
            bestParams = json.load(infile)

    loader = DataLoader()
    loader.setDataPath(dataPath)
    loader.setGameRange(cfg["min"], cfg["max"])
    loader.setDrawSize(cfg["draw_size"])
    numbers, _, _ = loader.load_numbers(skipLastColumns=cfg["skip_last_columns"])
    total_rows = len(numbers)
    if total_rows == 0:
        print(f"{game}: no data, skipping")
        return None
    dates = list(getattr(loader, "dates", []))

    cached = load_meta_score_table(
        path, game, days, total_rows, bestParams, meta_table_kind(game),
        model_names=expected_model_names(dataPath, bestParams, is_positional=positional, game=game),
        tolerate_rows=tolerate_rows)
    if cached is None:
        print(f"{game}: no usable cached meta-learner table (data/hyperOptCache/meta_{meta_table_kind(game)}_table_{game}.joblib) - "
              f"the control reads the table the Saturday chain collects and never recollects it; skipping")
        return None
    results, model_names = cached
    return results, list(model_names), bestParams, total_rows, positional, dates


def split_by_days(X, y, block, fraction=TRAIN_FRACTION):
    """Chronological split on whole days (a day is `block` contiguous rows)."""
    n_days = len(y) // block
    cut = int(n_days * fraction) * block
    return X[:cut], y[:cut], X[cut:], y[cut:], cut // block, n_days - cut // block


def run_game(game, cfg, path, days, repeats, noise_count, seed, variant_keys, tolerate_rows, max_rows, lockbox=None):
    loaded = load_table(game, cfg, path, days, tolerate_rows)
    if loaded is None:
        return None
    results, model_names, bestParams, total_rows, positional, dates = loaded
    special = int(cfg["special_column_count"])

    # The lockbox days leave before any split or fit, as in the trainer - the
    # cache is written before that split, so it still holds them. The
    # control's fits and held-out days are then the trainer's own.
    results, locked_rows = lockbox_split(results, dates, lockbox)
    if lockbox:
        print(f"{game}: {describe_lockbox(lockbox)} - {len(locked_rows)} table day(s) withheld, {len(results)} remain")
        if not results:
            print(f"{game}: nothing left outside the lockbox, skipping")
            return None

    if positional:
        positions = positional_positions(cfg)
        X, y = build_positional_training_table(results, model_names, positions, POSITIONAL_CLASSES)
        block = positions * POSITIONAL_CLASSES
    else:
        positions = None
        # main numbers only: the special column has its own table and model
        X, y = build_training_table(results, model_names, cfg["min"], cfg["max"], "_scores",
                                    "actual_main" if special > 0 else "actual")
        block = cfg["max"] - cfg["min"] + 1

    X_train, y_train, X_test, y_test, train_days, test_days = split_by_days(X, y, block)
    if test_days == 0 or len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        print(f"{game}: not enough days or classes for a split ({train_days} train / {test_days} test days), skipping")
        return None
    print(f"\n{game}: {len(results)} table days ({train_days} train / {test_days} held-out), {len(model_names)} base models, "
          f"{repeats} repeat(s) x {noise_count} noise column(s)")

    def fit_predict(fit_func, X_fit, y_fit):
        if positional:
            models = fit_position_models(X_fit, y_fit, fit_func, positions, POSITIONAL_CLASSES)
            return positional_predict(models, positions, POSITIONAL_CLASSES)
        return flat_predict(fit_func(X_fit, y_fit))

    newest_index = None
    try:
        newest_index = int(results[-1].get("index"))
    except (AttributeError, TypeError, ValueError):
        pass

    variants_out = {}
    for fit_func, _artifact, _label, params, key in meta_learner_variants(game, bestParams):
        if variant_keys and key not in variant_keys:
            continue
        started = time.time()
        try:
            rng = np.random.default_rng(seed)
            predict = fit_predict(fit_func, X_train, y_train)
            train_auc, train_imp = permutation_importance(predict, X_train, y_train, rng, max_rows, block)
            held_auc, held_imp = permutation_importance(predict, X_test, y_test, rng, max_rows, block)
            without = {"train_auc": train_auc, "heldout_auc": held_auc,
                       "train": train_imp.tolist(), "heldout": held_imp.tolist()}

            runs = []
            for r in range(repeats):
                rng_r = np.random.default_rng(seed + 1000 * (r + 1))
                # noise over the WHOLE table, then the same day split - a noise
                # column is a permutation of a real one across all rows
                X_with, sources = append_noise(X, rng_r, noise_count)
                Xw_train, _, Xw_test, _, _, _ = split_by_days(X_with, y, block)
                predict_r = fit_predict(fit_func, Xw_train, y_train)
                tr_auc, tr_imp = permutation_importance(predict_r, Xw_train, y_train, rng_r, max_rows, block)
                he_auc, he_imp = permutation_importance(predict_r, Xw_test, y_test, rng_r, max_rows, block)
                runs.append({"seed": seed + 1000 * (r + 1), "noise_sources": [model_names[j] for j in sources],
                             "train_auc": tr_auc, "heldout_auc": he_auc,
                             "train": tr_imp.tolist(), "heldout": he_imp.tolist()})

            summary = summarise(model_names, noise_count, runs, without)
            summary.update({"params": params, "seconds": round(time.time() - started, 1),
                            "columns_order": model_names + noise_names(noise_count), "runs": runs})
            variants_out[key] = summary
            print(describe(key, summary) + f" ({summary['seconds']}s)")
        except Exception as e:
            # one variant failing (a library missing, a degenerate fit) must
            # not cost the game its other verdicts
            print(f"{game} {key}: control failed - {e}")
            variants_out[key] = {"error": str(e), "seconds": round(time.time() - started, 1)}

    record = {
        "game": game,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "AUC",
        "table": {"days": len(results), "train_days": train_days, "test_days": test_days, "block_rows": block,
                  "draws_in_file": total_rows, "newest_index": newest_index,
                  "behind_file_by": (total_rows - 1 - newest_index) if newest_index is not None else None,
                  "lockbox": lockbox_json(lockbox), "lockbox_days_withheld": len(locked_rows),
                  "model_names": model_names, "positional": positional,
                  "kind": "positional" if positional else "main numbers"},
        "repeats": repeats, "noise_columns": noise_count, "seed": seed, "max_importance_rows": max_rows,
        "variants": variants_out,
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = os.path.join(RESULT_DIR, f"{game}.json")
    with open(out, "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"{game}: written to {out}")
    return record


def main():
    parser = argparse.ArgumentParser(
        prog="Irrelevant-feature control",
        description="Refit every served meta-learner variant with noise columns appended and report whether any gives them stable importance")
    parser.add_argument("-g", "--games", default="lotto",
                        help=f"Comma-separated games, from {','.join(GAME_CONFIG)}")
    parser.add_argument("-r", "--repeats", type=int, default=3, help="Fits with fresh noise columns per variant")
    parser.add_argument("-k", "--noise-columns", type=int, default=3, help="Noise columns appended per fit")
    parser.add_argument("-d", "--days", type=int, default=DEFAULT_DAYS, help="Table days to read (the cache must hold at least this many)")
    parser.add_argument("--variants", default="", help=f"Comma-separated subset of {','.join(VARIANT_KEYS)}; default all")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tolerate-rows", type=int, default=7,
                        help="Accept a cached table at most this many draws behind the data file (default 7: Sunday after a Saturday chain)")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                        help="Rows (whole days) the permutation importance is computed on, per side")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    unknown = [g for g in games if g not in GAME_CONFIG]
    if unknown:
        print(f"Unknown game(s), ignoring: {unknown}")
    games = [g for g in games if g in GAME_CONFIG]
    variant_keys = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown_variants = [v for v in variant_keys if v not in VARIANT_KEYS]
    if unknown_variants:
        print(f"Unknown variant(s), ignoring: {unknown_variants}")
    variant_keys = [v for v in variant_keys if v in VARIANT_KEYS]

    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)
    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)
    install_sigterm_handler()

    path = os.getcwd()
    started = time.time()
    try:
        # A malformed lockbox.json stops the run here, as it stops the trainer
        # and the quantum tuner - a control that silently used the lockbox
        # would be worse than no control.
        try:
            lockbox = load_lockbox(path)
        except ValueError as e:
            print(f"LOCKBOX ERROR: {e}")
            sys.exit(1)
        if lockbox:
            print(f"Under the {describe_lockbox(lockbox)}: its days are withheld from every table")
        for game in games:
            try:
                run_game(game, GAME_CONFIG[game], path, args.days, args.repeats, args.noise_columns, args.seed,
                         variant_keys, args.tolerate_rows, args.max_rows, lockbox=lockbox)
            except Exception as e:
                print(f"{game}: control failed - {e}")
    finally:
        remove_lock()
    print(f"\nDone in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
