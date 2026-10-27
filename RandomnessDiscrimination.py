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
                          For a market game, since 26 Oct 2026, the returns
                          are reordered and their bins re-cut causally, so
                          the bin frequencies move a little (total variation
                          0.006-0.035 over this script's window): the series
                          breaks there (src/ControlHistories.py).

Interpretation, from the README's own list: performance above chance can come
from historical rule changes, changed number ranges, sorted-versus-drawn
order, missing or duplicated records, preprocessing artifacts, schedule
changes, or simply an incorrect synthetic generator - all before any claim
about the draw itself. The default window therefore covers only the most
recent draws (--draws), where the rules have not changed; widen it and you
are measuring the rule history as much as the randomness.

The verdict (compute_verdict, version 2 since 27 Oct 2026) says "too few
draws" under controlRules.json's few_windows held-out windows - decided by the
history before any classifier runs - and otherwise holds the real comparison
against a bound from the null's and the shuffled control's own seeds that
widens by itself when the seeds are few, so give a hand run its 20 seeds too:

  python3 RandomnessDiscrimination.py -g lotto -w 10 -n 20
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime

import numpy as np
from scipy.stats import norm, t as student_t
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ControlHistories import control_draws, main_numbers, CONTROL_ROOT, control_seed, repetitions_for
from src.Helpers import Helpers
from src.QuantumModels import fit_quantum_kernel, fit_quantum_vqc
from HyperoptStatistics import GAME_CONFIG, is_running, create_lock, remove_lock

helpers = Helpers()
RESULT_DIR = os.path.join(CONTROL_ROOT, "discrimination")
# Q2's power rule, one number shared with the server: controlRules.json at the repository root, which controls.js
# reads too (27 Oct 2026). Until then it lived twice - here as a 0.8 ceiling on the null band, in admin.js as 250
# test windows - and the two disagreed. The file is tracked, so a missing one fails here, loudly.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controlRules.json")) as _rules:
    Q2_RULES = json.load(_rules)["q2"]
FEW_WINDOWS = int(Q2_RULES["few_windows"])


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


# --- the verdict --------------------------------------------------------------
# Version 2 (27 Oct 2026; the Sunday run of 1 Nov 2026 is the first under it). Version 1 read "above" when the real
# mean cleared the null mean + 2 sd of three seeds (five for the markets) and the bare shuffled mean, and "too few
# draws" when that threshold reached 0.8. On the same histories 6 of 11 labels changed between 18 and 25 Oct 2026:
# the 2 sd of three seeds is itself a coin toss (lotto's 3-seed thresholds came out 0.570, 0.572 and 0.751, against
# 0.673 +/- 0.019 from 20 seeds), the seven fair lottery games between them read "above" about once every two weeks,
# and "too few draws" flipped with the seeds (eurodreams, the two week games). Version 2:
#   too few draws  fewer held-out windows than FEW_WINDOWS (controlRules.json). A property of the history's length,
#                  --draws and --window, fixed before any classifier runs, never by an AUC; the short games still run
#                  at the full seed count, so their numbers stay on the page.
#   null band      the real mean above the one-sided BAND_COVERAGE prediction bound for ONE further fair run, from the
#                  n null seeds: null mean + band_factor(n) x null sd. The bound widens by itself when the seeds are
#                  few (k 5.23 at 3, 2.43 at 10, 2.19 at 20), so a hand run on three seeds can no longer produce the
#                  18 Oct lotto label. The real MEAN is held against a single-run bound on purpose: the real history
#                  is the one that exists, and averaging seeds removes only the synthetic side's noise, never the part
#                  of a run's variance that belongs to the history itself (3-44% on the 26 Oct grid, wide intervals).
#   shuffled       the same bound on the shuffled control's own seeds: real mean > shuffled mean + band_factor(n) x
#                  shuffled sd. Symmetric with the null leg and as conservative. A paired t by seed was measured and
#                  left out: it leaves out the real ordering's own place among its reorderings, so as a test of time
#                  order it is anti-conservative, and for the markets (crypto clears the null band by 0.3 every week)
#                  it would have been the only gate.
#   failed         a suite that scored fewer than 2 seeds has no spread, so there is no bound: every number None, both
#                  flags False, and the status says "failed" - a broken run never reads as a clean "within".
# Measured on the 26 Oct 2026 scratch pools (40 seeds per lottery game, 20 per market game, the market shuffled
# control re-cut as on 26 Oct), 4,000 resampled weeks of 20 seeds: no false "above" on the lottery games, 0.06 label
# changes a week over the 11 games (all of them shares, a genuine borderline case), crypto "above" in 99.7%.
BAND_COVERAGE = float(norm.cdf(2.0))   # 0.97725: the one-sided coverage "+ 2 sd" stood for, now made exact
VERDICT_VERSION = 2
STATUSES = ("above", "above band, not shuffled", "within", "too few draws", "failed")


def band_factor(n):
    """
    k in "mean + k sd" over n runs: the one-sided BAND_COVERAGE prediction bound for one further run, a t quantile
    with n - 1 degrees of freedom widened by sqrt(1 + 1/n) for the uncertainty of the mean (5.227 at 3 runs, 2.433
    at 10, 2.193 at 20). None under 2 runs: one run has no spread.
    """
    if n < 2:
        return None
    return float(student_t.ppf(BAND_COVERAGE, n - 1) * math.sqrt(1 + 1 / n))


def compute_verdict(record, few_windows=FEW_WINDOWS):
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

    The rule itself (version 2, the block above): "too few draws" under
    few_windows held-out windows, else both bounds from the seeds' own spread.
    The record keeps the few_windows it was judged by, so a later change of
    controlRules.json never re-labels it silently.
    """
    def per_seed_max(name):
        # a seed whose whole suite failed has no best AUC and is left out
        values = []
        for run in record["runs"].get(name, []):
            scores = [entry["auc"] for entry in run.values() if "auc" in entry]
            if scores:
                values.append(max(scores))
        return values

    def mean_sd(values):
        return (statistics.fmean(values), statistics.stdev(values)) if len(values) >= 2 else (None, None)

    def number(value, digits=4):
        # None, never NaN: the admin pages' JSON reader rejects NaN and would drop the record
        return None if value is None or value != value else round(float(value), digits)

    real_values, null_values, shuffled_values = (per_seed_max("real_vs_synthetic"),
                                                 per_seed_max("synthetic_vs_synthetic"),
                                                 per_seed_max("shuffled_vs_synthetic"))
    # the first run's shape (run_game): every record main() writes has one; a hand-made one without it is judged on
    # its AUCs alone
    windows_test = (record.get("shape") or {}).get("windows_test")
    windows_test = int(windows_test) if isinstance(windows_test, int) else None
    too_few = windows_test is not None and windows_test < int(few_windows)
    failed = min(len(real_values), len(null_values), len(shuffled_values)) < 2

    real_mean = real_sd = null_mean = null_sd = shuffled_mean = shuffled_sd = None
    band_k = shuffled_k = threshold = threshold_se = shuffled_threshold = None
    above_band = survives = False
    if not failed:
        (real_mean, real_sd), (null_mean, null_sd), (shuffled_mean, shuffled_sd) = (
            mean_sd(real_values), mean_sd(null_values), mean_sd(shuffled_values))
        n_null = len(null_values)
        band_k = band_factor(n_null)
        threshold = null_mean + band_k * null_sd
        # the threshold's own standard error (mean and sd both estimated from the n null seeds): about 0.42 sd at 20
        threshold_se = null_sd * math.sqrt(1 / n_null + band_k ** 2 / (2 * (n_null - 1)))
        shuffled_k = band_factor(len(shuffled_values))
        shuffled_threshold = shuffled_mean + shuffled_k * shuffled_sd
        above_band = real_mean > threshold
        survives = real_mean > shuffled_threshold

    if too_few:
        status = "too few draws"
    elif failed:
        status = "failed"
    elif above_band and survives:
        status = "above"
    elif above_band:
        status = "above band, not shuffled"
    else:
        status = "within"

    return {
        "version": VERDICT_VERSION,
        "status": status,
        "metric": "best AUC of the classifier suite, per repetition",
        "real_max_of_suite_mean": number(real_mean), "real_max_of_suite_sd": number(real_sd),
        "null_max_of_suite_mean": number(null_mean), "null_max_of_suite_sd": number(null_sd),
        "shuffled_max_of_suite_mean": number(shuffled_mean), "shuffled_max_of_suite_sd": number(shuffled_sd),
        # the seeds whose suite scored at all, per comparison: under 2 in any of them is "failed"
        "seeds_scored": {"real": len(real_values), "null": len(null_values), "shuffled": len(shuffled_values)},
        "band_coverage": round(BAND_COVERAGE, 5),
        "band_k": number(band_k, 3), "shuffled_k": number(shuffled_k, 3),
        "null_threshold": number(threshold), "null_threshold_se": number(threshold_se),
        "shuffled_threshold": number(shuffled_threshold),
        "per_seed_real": [round(v, 4) for v in real_values],
        "per_seed_null": [round(v, 4) for v in null_values],
        "per_seed_shuffled": [round(v, 4) for v in shuffled_values],
        "above_null_band": bool(above_band),
        "survives_marginals_control": bool(survives),
        "too_few_draws": bool(too_few),
        "few_windows": int(few_windows),
        "windows_test": windows_test,
        "rule": (f"above = the real mean above the null band's {BAND_COVERAGE:.1%} bound for one fair run (null mean "
                 f"+ k sd, k from the seed count: 2.19 at 20) and above the shuffled control's bound (shuffled mean + "
                 f"k sd, the same way); too few draws under {int(few_windows)} held-out windows, fixed by the history "
                 f"before the run; failed when a comparison scored fewer than 2 seeds"),
        "power_rule": (f"too few draws under {int(few_windows)} held-out windows, both classes together: set by the "
                       f"history's length, --draws and --window before any classifier runs, never by an AUC - on "
                       f"fewer windows the fair-vs-fair band is too wide to show anything either way"),
    }


def _f3(value):
    return "-" if value is None else f"{value:.3f}"


def verdict_line(record):
    """One line per record: its status and the numbers it was read from (the log, --reanalyze)."""
    v = record["verdict"]
    n = v.get("seeds_scored") or {}
    return (f"{record['game']:<14} {v['status']:<25} real {_f3(v['real_max_of_suite_mean'])} +/- "
            f"{_f3(v['real_max_of_suite_sd'])} | null {_f3(v['null_max_of_suite_mean'])} +/- "
            f"{_f3(v['null_max_of_suite_sd'])} -> threshold {_f3(v['null_threshold'])} +/- "
            f"{_f3(v['null_threshold_se'])} (k {_f3(v['band_k'])}) | shuffled {_f3(v['shuffled_max_of_suite_mean'])} "
            f"-> {_f3(v['shuffled_threshold'])} | seeds {n.get('real', '-')}/{n.get('null', '-')}/"
            f"{n.get('shuffled', '-')} | {v['windows_test'] if v['windows_test'] is not None else '?'} test windows")


def reanalyze(folder=None):
    """Recompute the verdicts of saved records under the current rule - so a
    corrected rule can be applied to a run that took an hour, without paying
    for it twice. folder: the records' folder, RESULT_DIR by default (a copy
    elsewhere is how a new rule is tried without touching the live records)."""
    import glob
    for path in sorted(glob.glob(os.path.join(folder or RESULT_DIR, "*.json"))):
        with open(path) as handle:
            record = json.load(handle)
        record["verdict"] = compute_verdict(record)
        with open(path, "w") as handle:
            json.dump(record, handle, indent=2)
        print(verdict_line(record))
    return 0


def _self_check():
    """compute_verdict on hand-made records: python3 -c 'import RandomnessDiscrimination as R; R._self_check()' (npm test)."""
    # the constants: k from the seed count, the coverage "+ 2 sd" stood for, the power rule from controlRules.json
    assert [round(band_factor(n), 3) for n in (3, 10, 20)] == [5.227, 2.433, 2.193], [band_factor(n) for n in (3, 10, 20)]
    assert band_factor(1) is None and band_factor(0) is None and band_factor(2) > band_factor(3)
    assert round(BAND_COVERAGE, 5) == 0.97725 and round(float(student_t.ppf(BAND_COVERAGE, 19)), 2) == 2.14
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controlRules.json")) as handle:
        assert FEW_WINDOWS == int(json.load(handle)["q2"]["few_windows"]) > 0, FEW_WINDOWS

    def runs(values, spare=0.01):
        # one seed per value: the best of a two-model suite is the value; None = a seed whose whole suite failed
        return [{"logistic": {"auc": v - spare}, "random_forest": {"auc": v}} if v is not None
                else {"logistic": {"error": "ValueError: boom"}, "random_forest": {"error": "ValueError: boom"}}
                for v in values]

    def record(real, null, shuffled, windows=496):
        return {"game": "lotto", "shape": {"windows_train": 1486, "windows_test": windows, "features": 45},
                "runs": {"real_vs_synthetic": runs(real), "synthetic_vs_synthetic": runs(null),
                         "shuffled_vs_synthetic": runs(shuffled)}}

    def verdict(rec, few=250):
        v = compute_verdict(rec, few_windows=few)
        json.dumps(v, allow_nan=False)   # never NaN, whatever failed
        assert v["version"] == VERDICT_VERSION and v["status"] in STATUSES and v["few_windows"] == few, v
        if v["status"] not in ("too few draws", "failed"):
            # the status is the flags', and the flags are the numbers'
            assert v["status"] == ("above" if v["above_null_band"] and v["survives_marginals_control"] else
                                   "above band, not shuffled" if v["above_null_band"] else "within"), v
            assert v["above_null_band"] == (v["real_max_of_suite_mean"] > v["null_threshold"]), v
            assert v["survives_marginals_control"] == (v["real_max_of_suite_mean"] > v["shuffled_threshold"]), v
        return v

    z = list(np.random.RandomState(0).standard_normal(20))
    z = [(x - statistics.fmean(z)) / statistics.stdev(z) for x in z]   # mean 0, sd 1 exactly
    null20 = [0.57 + 0.05 * x for x in z]
    real20 = [0.75 + 0.03 * x for x in z[::-1]]
    shuffled20 = [0.60 + 0.05 * x for x in z[1:] + z[:1]]

    # 20 seeds, real far above both: above; the threshold is mean + 2.193 sd with its standard error
    v = verdict(record(real20, null20, shuffled20))
    assert v["status"] == "above" and v["band_k"] == 2.193 and v["shuffled_k"] == 2.193, v
    assert abs(v["null_threshold"] - (0.57 + 2.1934 * 0.05)) < 1e-3 and abs(v["shuffled_threshold"] - (0.60 + 2.1934 * 0.05)) < 1e-3, v
    assert abs(v["null_threshold_se"] - 0.05 * math.sqrt(1 / 20 + 2.1934 ** 2 / 38)) < 1e-4, v
    assert v["seeds_scored"] == {"real": 20, "null": 20, "shuffled": 20} and v["band_coverage"] == 0.97725 and len(v["per_seed_real"]) == 20
    # the same with the shuffled control separating as well as the real draws, seed by seed: a marginal, not time
    v = verdict(record(real20, null20, real20))
    assert v["status"] == "above band, not shuffled" and v["above_null_band"] and not v["survives_marginals_control"], v
    # the 18 Oct 2026 lotto block, 3 seeds: version 1 read it "above" (0.590 against 0.570); k = 5.23 reads it within
    v = verdict(record([0.58, 0.61, 0.58], [0.52, 0.48, 0.53], [0.55, 0.52, 0.56]))
    assert v["status"] == "within" and v["band_k"] == 5.227 and v["null_threshold"] > 0.64 and not v["above_null_band"], v
    # too few draws: the windows decide it, before and whatever the AUCs say - even 0.95 against a fair 0.57
    v = verdict(record([0.95] * 20, null20, shuffled20, windows=150))
    assert v["status"] == "too few draws" and v["too_few_draws"] and v["windows_test"] == 150, v
    assert v["null_threshold"] is not None and v["above_null_band"], "a too-few record keeps its numbers for the page"
    assert verdict(record(real20, null20, shuffled20, windows=249))["status"] == "too few draws"
    assert verdict(record(real20, null20, shuffled20, windows=250))["status"] == "above", "250 windows are enough"
    assert verdict(record(real20, null20, shuffled20, windows=208), few=200)["status"] == "above", "the rule is the record's few_windows"
    # a seed whose whole suite failed is left out of its comparison, not counted as a zero and not shifted
    v = verdict(record([None] + real20[1:], null20, shuffled20))
    assert v["seeds_scored"]["real"] == 19 and v["per_seed_real"] == [round(x, 4) for x in real20[1:]] and v["status"] == "above", v
    # each bound's k from its own control's seeds - the null's 20 above, not the real side's 19; a shuffled control
    # that scored 10 gets the wider k of 10 (review, 27 Oct 2026: a k from the wrong count passed every case before)
    assert v["band_k"] == 2.193 and v["shuffled_k"] == 2.193, v
    v = verdict(record(real20, null20, shuffled20[:10]))
    assert v["band_k"] == 2.193 and v["shuffled_k"] == 2.433 and v["seeds_scored"]["shuffled"] == 10, v
    # failed: fewer than 2 seeds in any comparison - no number, both flags False, never "within"
    for broken in (record(real20, [0.55], shuffled20), record([None] * 20, null20, shuffled20),
                   record(real20, null20, [0.6, None, None]), record([], [], [])):
        v = verdict(broken)
        assert v["status"] == "failed" and not v["above_null_band"] and not v["survives_marginals_control"], v
        assert all(v[k] is None for k in ("real_max_of_suite_mean", "real_max_of_suite_sd", "null_max_of_suite_mean",
                                          "null_max_of_suite_sd", "shuffled_max_of_suite_mean", "shuffled_max_of_suite_sd",
                                          "band_k", "shuffled_k", "null_threshold", "null_threshold_se", "shuffled_threshold")), v
    # ... unless the history already decided it: too few draws first
    assert verdict(record([None] * 3, [0.5], [], windows=150))["status"] == "too few draws"
    # controls whose seeds all scored the same: no spread, each bound is its mean (no division by zero); two seeds
    # are the least that give a bound, and a wide one (k 17.1)
    v = verdict(record([0.62, 0.64], [0.6, 0.6], [0.5, 0.5]))
    assert v["null_threshold"] == 0.6 and v["null_threshold_se"] == 0.0 and v["shuffled_threshold"] == 0.5 and v["status"] == "above", v
    assert verdict(record([0.62, 0.64], [0.6, 0.6], [0.5, 0.52]))["status"] == "above band, not shuffled"
    # a real mean that only touches the bound has not cleared it
    assert verdict(record([0.6, 0.6], [0.6, 0.6], [0.5, 0.5]))["status"] == "within"
    # the seed counts main() accepts: 2 to 999 per game, the market games' own count included
    assert repetitions_error(["lotto", "crypto"], 20, 0) is None and repetitions_error(["lotto", "shares"], 999, 2) is None
    assert repetitions_error(["lotto"], 1, 0) and repetitions_error(["lotto"], 1000, 0) and repetitions_error(["lotto"], 0, 5)
    assert repetitions_error(["lotto", "crypto"], 20, 1) and repetitions_error(["crypto"], 20, 1000) and not repetitions_error(["lotto"], 20, 1)
    # the record keeps the rule it was judged by, by default controlRules.json's
    assert compute_verdict(record(real20, null20, shuffled20))["few_windows"] == FEW_WINDOWS
    print("RandomnessDiscrimination self-check OK: k from the seed count (5.227 / 2.433 / 2.193), too few draws by the "
          "windows, both bounds, a failed suite never reads within")


def run_game(game, window, draws, seeds, skip_quantum):
    """
    One game's record: the three comparisons over the seeds, and the verdict. Writes nothing - main() does - so a
    check can run a game into a folder of its own. shape is the first run's; every run of a game is cut from the
    same history, --draws and --window, so a run whose shape differs is a bug: it is recorded (shape_mismatch) and
    the run goes on, rather than an assert costing the rest of the Sunday's games over metadata (27 Oct 2026).
    """
    record = {"game": game, "window": window, "draws": draws,
              "repetitions": len(seeds), "seeds": seeds, "quantum": not skip_quantum,
              "generated_at": datetime.now().isoformat(timespec="seconds"), "runs": {}}
    for name in COMPARISONS:
        per_seed = []
        for seed in seeds:
            results, shape = run_comparison(game, name, window, seed, draws, skip_quantum)
            per_seed.append(results)
            if "shape" not in record:
                record["shape"] = shape
            elif shape != record["shape"]:
                record.setdefault("shape_mismatch", []).append({"comparison": name, "seed": seed, **shape})
        record["runs"][name] = per_seed
        print(f"  {name}:")
        for label in per_seed[0]:
            aucs = [run[label]["auc"] for run in per_seed if "auc" in run[label]]
            if not aucs:
                print(f"    {label:<20} failed: {per_seed[0][label].get('error')}")
                continue
            spread = f" +/- {statistics.stdev(aucs):.3f}" if len(aucs) > 1 else ""
            print(f"    {label:<20} AUC {statistics.fmean(aucs):.3f}{spread}")
    record["verdict"] = compute_verdict(record)
    return record


def repetitions_error(games, repetitions, market_repetitions):
    """
    Each game's own count of seeds (repetitions_for: --market-repetitions included) must be 2 to 999: one run has
    no spread, so no band; and the null side's second fair history is seed + 1000 inside the week's 10,000-seed
    block (control_seed), so from 1000 seeds on a null history would be another seed's negative one. The message
    for the first game outside that, or None. main() checks it before the first game runs, not an hour in.
    """
    for game in games:
        count = repetitions_for(game, repetitions, market_repetitions)
        if not 2 <= count <= 999:
            return f"{game} would run {count} repetitions: -n and --market-repetitions must give every game 2 to 999"
    return None


# What each status means, for the log under its numbers
MEANING = {
    "too few draws": "the history gives fewer held-out windows than the power rule asks for, so this record cannot "
                     "show evidence either way - fixed by the history's length before the run, not by these AUCs.",
    "failed": "a comparison scored fewer than 2 seeds (see the errors above), so there is no band to read: "
              "not a negative result, no result.",
    "within": "no evidence: the real comparison does not clear the bound one run on two histories known to be "
              "identical in law stays under.",
    "above band, not shuffled": "not temporal: the real draws reordered keep the same separation, so whatever is "
                                "being detected is a marginal or a rule, not next-draw dependence.",
    "above": "above both controls. Before calling this evidence, check the README's list: rule changes in the "
             "window, range changes, sorted-versus-drawn order, missing or duplicated records, and the synthetic "
             "generator itself.",
}


def main():
    parser = argparse.ArgumentParser(
        prog="Randomness discrimination",
        description="Can a classifier tell real draw windows from fair simulated ones? (README roadmap item 6, Q2)")
    parser.add_argument("-g", "--games", default="lotto", help='Comma-separated games (default lotto)')
    parser.add_argument("-w", "--window", type=int, default=10, help="Draws per window (default 10)")
    parser.add_argument("-n", "--repetitions", type=int, default=20,
                        help="Independent synthetic histories, 2 to 999 (default 20, the Sunday plan's since 1 Nov 2026)")
    parser.add_argument("--seed", default="1",
                        help='First seed; repetitions count up from it. "weekly" = a fresh block of seeds every ISO week '
                             '(the Sunday plan, since 13 Oct 2026); a number reproduces a run')
    parser.add_argument("--market-repetitions", type=int, default=0,
                        help="Repetitions for the market games (crypto, shares and their week games); 0 = -n")
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
    wrong = repetitions_error(games, args.repetitions, args.market_repetitions)
    if wrong:
        parser.error(wrong)

    os.makedirs(RESULT_DIR, exist_ok=True)
    first_seed = control_seed(args.seed)
    for game in games:
        repetitions = repetitions_for(game, args.repetitions, args.market_repetitions)
        seeds = list(range(first_seed, first_seed + repetitions))
        print(f"\n{game}: window {args.window} draws, newest {args.draws or 'all'} draws, "
              f"{repetitions} repetitions (seeds {seeds[0]}..{seeds[-1]})")
        record = run_game(game, args.window, args.draws, seeds, args.skip_quantum)
        with open(os.path.join(RESULT_DIR, f"{game}-w{args.window}.json"), "w") as handle:
            json.dump(record, handle, indent=2)

        verdict = record["verdict"]
        print(f"  best-of-suite per repetition, mean over seeds: {verdict_line(record)}")
        if record.get("shape_mismatch"):
            print(f"  WARNING: {len(record['shape_mismatch'])} runs were cut to another shape than the first "
                  f"({record['shape']}): shape_mismatch in the record")
        print(f"  -> {verdict['status']}: {MEANING[verdict['status']]}")
    print(f"\nRecords written to {os.path.relpath(RESULT_DIR)}/")
    return 0


if __name__ == "__main__":
    # The same process.lock every other entry point takes (imported from
    # HyperoptStatistics so the rule lives in one place): the scheduler and a
    # hand-run runPredictor.sh both decide from it whether the box is busy, and
    # a control run is a full backtest per history. A live lock means exit, as
    # for the tuners - the scheduler queues the job behind the holder.
    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)
    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)
    try:
        sys.exit(main())
    finally:
        remove_lock()
