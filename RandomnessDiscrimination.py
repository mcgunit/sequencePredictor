#!/usr/bin/env python3
"""
Q2 randomness discrimination (README roadmap item 6 and "Randomness-
discrimination experiment"): can any classifier tell real draw windows from
correctly simulated fair ones?

  Class 1 = real historical draws
  Class 0 = synthetic draws generated under the game's rules

This is the phase that speaks directly to the research question. A classifier
that reliably separates the two has found a way in which the real process is
not the fair process - though NOT necessarily a predictable one, and not
necessarily anything about the draw at all (see the interpretation list
below). A classifier stuck at chance is the honest negative result.

WHY THERE ARE THREE COMPARISONS, not one. A single number - "AUC 0.55 on the
holdout" - cannot be read, because this pipeline has a noise floor of its own
and because "different from synthetic" has many boring causes. So every run
measures all three, through the identical feature builder, split and model
suite:

  real vs synthetic       the question.
  synthetic vs synthetic  the NULL BAND: two independent fair histories,
                          where the true answer is provably 0.5. Whatever
                          this scores is what the pipeline scores on nothing,
                          and the real comparison has to beat it to mean
                          anything at all.
  shuffled vs synthetic   the MARGINALS control: the real draws reordered, so
                          every frequency and co-occurrence survives and only
                          time order is destroyed. If this separates as well
                          as the real comparison, the classifier is reading
                          marginals or rules - not next-draw dependence.

Interpretation, from the README's own list: performance above chance can come
from historical rule changes, changed number ranges, sorted-versus-drawn
order, missing or duplicated records, preprocessing artifacts, schedule
changes, or simply an incorrect synthetic generator - all before any claim
about the draw itself. The default window therefore covers only the most
recent draws (--draws), where the rules have not changed; widen it and you
are measuring the rule history as much as the randomness.

  python3 RandomnessDiscrimination.py -g lotto -w 10 -n 3
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ControlHistories import control_draws, main_numbers, CONTROL_ROOT
from src.Helpers import Helpers
from src.QuantumModels import fit_quantum_kernel, fit_quantum_vqc
from HyperoptStatistics import GAME_CONFIG

helpers = Helpers()
RESULT_DIR = os.path.join(CONTROL_ROOT, "discrimination")


# --- features ----------------------------------------------------------------
def window_features(game, draws, window):
    """
    One row per window of `window` consecutive draws.

    Set games: how often each number came up in the window (multi-hot counts
    over the game's range, divided by the window length). Positional games:
    the same per slot, so digit order survives - a pick3 window is three
    ten-bin histograms, not one.

    These are deliberately the plainest features that can express what people
    look for in draw histories - hot and cold numbers, repeats, coverage. A
    fair process makes them fluctuate; only a biased one makes them
    fluctuate differently from a simulated fair process.
    """
    cfg = GAME_CONFIG[game]
    values = main_numbers(game, draws)
    positional = helpers.is_positional_game(game)
    low, high = cfg["min"], cfg["max"]
    span = high - low + 1

    rows = []
    for start in range(0, len(values) - window + 1):
        chunk = values[start:start + window]
        if positional:
            counts = np.zeros((cfg["draw_size"], span), dtype=float)
            for draw in chunk:
                for position, value in enumerate(draw):
                    counts[position][int(value) - low] += 1
            rows.append((counts / window).ravel())
        else:
            counts = np.zeros(span, dtype=float)
            for draw in chunk:
                for value in draw:
                    counts[int(value) - low] += 1
            rows.append(counts / window)
    return np.asarray(rows, dtype=float)


def dataset(game, window, positive, negative, draws_kept):
    """
    (X, y) for one comparison, with the windows of both classes in the same
    chronological order - so a chronological split cuts both at the same
    point in time, not at an arbitrary place in one of them.
    """
    def history(kind, seed):
        # draws_kept goes INTO control_draws, not after it: see the note in
        # src/ControlHistories.control_draws - shuffling the whole history
        # and then taking its tail samples every era the game ever had, which
        # silently breaks the one property the shuffled control must keep.
        _, all_draws = control_draws(game, kind, seed, recent=draws_kept)
        return all_draws

    x_positive = window_features(game, history(*positive), window)
    x_negative = window_features(game, history(*negative), window)
    size = min(len(x_positive), len(x_negative))
    x_positive, x_negative = x_positive[-size:], x_negative[-size:]
    X = np.concatenate([x_positive, x_negative])
    y = np.concatenate([np.ones(size), np.zeros(size)])
    order = np.concatenate([np.arange(size), np.arange(size)])   # window index in time
    return X, y, order


def chronological_split(X, y, order, holdout=0.25):
    """The newest `holdout` of the windows are the test set, for both classes."""
    cut = int(max(order) * (1 - holdout))
    train = order <= cut
    return X[train], y[train], X[~train], y[~train]


# --- the model suite ----------------------------------------------------------
def classifiers(skip_quantum=False, seed=0):
    suite = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "rbf_svm": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed)),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=seed),
        "small_nn": lambda: make_pipeline(StandardScaler(),
                                          MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=seed)),
    }
    if not skip_quantum:
        # The same two quantum variants the meta-learner rows use
        # (src/QuantumModels.py), on their defaults: this is a comparison of
        # families, not a tuning contest - and tuning the suite on the
        # question it is meant to answer is how a null result turns into a
        # false positive.
        suite["quantum_kernel"] = lambda: _Factory(fit_quantum_kernel)
        suite["quantum_vqc"] = lambda: _Factory(fit_quantum_vqc)
    return suite


class _Factory:
    """Adapts QuantumModels' fit_X(X, y) factories to the fit/predict_proba shape."""

    def __init__(self, factory):
        self._factory = factory
        self._model = None

    def fit(self, X, y):
        self._model = self._factory(X, y)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {"auc": float(roc_auc_score(y_test, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions))}


# --- one comparison -----------------------------------------------------------
COMPARISONS = {
    # name: (positive class, negative class) - the seed offsets keep the two
    # sides independent where both are generated.
    "real_vs_synthetic": (("real", 0), ("synthetic", 0)),
    "synthetic_vs_synthetic": (("synthetic", 1000), ("synthetic", 0)),
    "shuffled_vs_synthetic": (("shuffled", 0), ("synthetic", 0)),
}


def run_comparison(game, name, window, seed, draws_kept, skip_quantum):
    positive, negative = COMPARISONS[name]
    # The seed varies the GENERATED side only - the real history is the one
    # that exists, so repetitions measure how much the answer moves with the
    # synthetic draw, which is the only randomness this experiment has.
    positive = (positive[0], positive[1] + seed)
    negative = (negative[0], negative[1] + seed)
    X, y, order = dataset(game, window, positive, negative, draws_kept)
    X_train, y_train, X_test, y_test = chronological_split(X, y, order)
    results = {}
    for label, build in classifiers(skip_quantum, seed).items():
        started = time.time()
        try:
            results[label] = score(build(), X_train, y_train, X_test, y_test)
            results[label]["seconds"] = round(time.time() - started, 1)
        except Exception as exc:
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
    return results, {"windows_train": int(len(y_train)), "windows_test": int(len(y_test)),
                     "features": int(X.shape[1])}


def compute_verdict(record):
    """
    The verdict has to compare LIKE WITH LIKE, and getting that wrong is the
    error this whole experiment exists to expose.

    Each repetition runs a suite of seven classifiers and the interesting
    number is the best of them - but the best of seven is biased upwards even
    when all seven are worthless, so it cannot be held against the AVERAGE of
    the null comparison's individual runs. It has to be held against the
    distribution of the null's OWN best-of-seven. Measured on euromillions
    2026-09-21, the difference decides the answer: the best single real AUC
    (0.637) clears the mean+2sd of the null's individual AUCs (0.598) and
    reads as evidence, while the real best-of-suite averaged over seeds
    (0.551 +/- 0.050) sits inside the null's best-of-suite (0.544 +/- 0.030)
    and reads, correctly, as nothing.
    """
    def per_seed_max(name):
        values = []
        for run in record["runs"].get(name, []):
            scores = [entry["auc"] for entry in run.values() if "auc" in entry]
            if scores:
                values.append(max(scores))
        return values

    def stats(values):
        if not values:
            return float("nan"), 0.0
        return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)

    real_values, null_values, shuffled_values = (per_seed_max("real_vs_synthetic"),
                                                 per_seed_max("synthetic_vs_synthetic"),
                                                 per_seed_max("shuffled_vs_synthetic"))
    real_mean, real_sd = stats(real_values)
    null_mean, null_sd = stats(null_values)
    shuffled_mean, shuffled_sd = stats(shuffled_values)
    threshold = null_mean + 2 * null_sd
    return {
        "metric": "best AUC of the classifier suite, per repetition",
        "real_max_of_suite_mean": round(real_mean, 4), "real_max_of_suite_sd": round(real_sd, 4),
        "null_max_of_suite_mean": round(null_mean, 4), "null_max_of_suite_sd": round(null_sd, 4),
        "shuffled_max_of_suite_mean": round(shuffled_mean, 4), "shuffled_max_of_suite_sd": round(shuffled_sd, 4),
        "null_threshold_2sd": round(threshold, 4),
        "per_seed_real": [round(v, 4) for v in real_values],
        "per_seed_null": [round(v, 4) for v in null_values],
        "per_seed_shuffled": [round(v, 4) for v in shuffled_values],
        "above_null_band": bool(real_mean > threshold),
        "survives_marginals_control": bool(real_mean > shuffled_mean),
    }


def reanalyze():
    """Recompute the verdicts of saved records under the current rule - so a
    corrected rule can be applied to a run that took an hour, without paying
    for it twice."""
    import glob
    for path in sorted(glob.glob(os.path.join(RESULT_DIR, "*.json"))):
        with open(path) as handle:
            record = json.load(handle)
        record["verdict"] = compute_verdict(record)
        with open(path, "w") as handle:
            json.dump(record, handle, indent=2)
        v = record["verdict"]
        print(f"{record['game']:<14} real {v['real_max_of_suite_mean']:.3f} +/- {v['real_max_of_suite_sd']:.3f} | "
              f"null {v['null_max_of_suite_mean']:.3f} +/- {v['null_max_of_suite_sd']:.3f} | "
              f"shuffled {v['shuffled_max_of_suite_mean']:.3f} | "
              f"{'ABOVE the null band' if v['above_null_band'] else 'within the null band - no evidence'}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="Randomness discrimination",
        description="Can a classifier tell real draw windows from fair simulated ones? (README roadmap item 6, Q2)")
    parser.add_argument("-g", "--games", default="lotto", help='Comma-separated games (default lotto)')
    parser.add_argument("-w", "--window", type=int, default=10, help="Draws per window (default 10)")
    parser.add_argument("-n", "--repetitions", type=int, default=3, help="Independent synthetic histories")
    parser.add_argument("--draws", type=int, default=1000,
                        help="Use only the most recent N draws, where the rules have not changed (0 = all)")
    parser.add_argument("--skip-quantum", action="store_true", help="Classical suite only (much faster)")
    parser.add_argument("--reanalyze", action="store_true",
                        help="Recompute the verdicts of saved records under the current rule and exit")
    args = parser.parse_args()

    if args.reanalyze:
        return reanalyze()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    unknown = [g for g in games if g not in GAME_CONFIG]
    if unknown:
        print(f"Unknown games: {', '.join(unknown)}")
        return 1

    os.makedirs(RESULT_DIR, exist_ok=True)
    for game in games:
        print(f"\n{game}: window {args.window} draws, newest {args.draws or 'all'} draws, "
              f"{args.repetitions} repetitions")
        record = {"game": game, "window": args.window, "draws": args.draws,
                  "repetitions": args.repetitions, "quantum": not args.skip_quantum,
                  "generated_at": datetime.now().isoformat(timespec="seconds"), "runs": {}}
        for name in COMPARISONS:
            per_seed = []
            for seed in range(1, args.repetitions + 1):
                results, shape = run_comparison(game, name, args.window, seed, args.draws, args.skip_quantum)
                per_seed.append(results)
                record.setdefault("shape", shape)
            record["runs"][name] = per_seed
            print(f"  {name}:")
            for label in per_seed[0]:
                aucs = [run[label]["auc"] for run in per_seed if "auc" in run[label]]
                if not aucs:
                    print(f"    {label:<20} failed: {per_seed[0][label].get('error')}")
                    continue
                spread = f" +/- {statistics.stdev(aucs):.3f}" if len(aucs) > 1 else ""
                print(f"    {label:<20} AUC {statistics.fmean(aucs):.3f}{spread}")

        verdict = record["verdict"] = compute_verdict(record)
        real = verdict["real_max_of_suite_mean"]
        shuffled = verdict["shuffled_max_of_suite_mean"]
        null_mean, null_sd = verdict["null_max_of_suite_mean"], verdict["null_max_of_suite_sd"]
        threshold = verdict["null_threshold_2sd"]
        null = threshold
        with open(os.path.join(RESULT_DIR, f"{game}-w{args.window}.json"), "w") as handle:
            json.dump(record, handle, indent=2)

        print(f"  best-of-suite per repetition: real {real:.3f} +/- {verdict['real_max_of_suite_sd']:.3f} | "
              f"shuffled {shuffled:.3f} | null {null_mean:.3f} +/- {null_sd:.3f} (threshold {threshold:.3f})")
        if real <= null:
            print("  -> no evidence: the real comparison does not clear what this pipeline scores on "
                  "two histories known to be identical in law.")
        elif real <= shuffled:
            print("  -> not temporal: reordering the real draws keeps the same separation, so whatever is "
                  "being detected is a marginal or a rule, not next-draw dependence.")
        else:
            print("  -> above both controls. Before calling this evidence, check the README's list: rule "
                  "changes in the window, range changes, sorted-versus-drawn order, missing or duplicated "
                  "records, and the synthetic generator itself.")
    print(f"\nRecords written to {os.path.relpath(RESULT_DIR)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
