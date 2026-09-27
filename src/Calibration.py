"""
Per-number calibration and ranking metrics for the meta-learners (README
roadmap item 6, Q0: "persist per-number calibration and ranking metrics
instead of the printed accuracy/AUC line").

A meta-learner emits, for every number of a draw, a probability that it is
drawn. Accuracy at 0.5 says almost nothing about that: with 6 of 45 numbers
drawn a model that answers "no" everywhere scores 87%. What a probability is
FOR is being right on average - calibration - and putting the drawn numbers
at the top - ranking - and those are measured here:

  brier            mean squared error of the probability against the 0/1
                   outcome; the constant base rate scores base_rate x (1 -
                   base_rate), reported as brier_base_rate for reference
  log_loss         mean negative log-likelihood, probabilities clipped
  auc              ranking quality over the flat table (None with one class)
  reliability      ten equal-width probability bins: how often the numbers
                   given a probability in the bin were actually drawn, next to
                   the mean probability in the bin; ece is the count-weighted
                   mean gap between the two (0 = perfectly calibrated)
  precision_at_k   the fraction of the top-draw_size numbers of each day that
                   were drawn, averaged over the days: the ticket-level
                   number the History page ranks on, with hits_at_k = precision
                   x draw_size and chance = draw_size / numbers per day

`block` is the number of table rows per day (max - min + 1, the flat table's
day-major layout, see TrainMetaLearner.build_training_table), which is what
lets the flat table be cut back into days for the per-day ranking.

    python3 -m src.Calibration     # self-check on synthetic data
"""

import math

import numpy as np

RELIABILITY_BINS = 10
EPS = 1e-6


def _round(value, digits=4):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def reliability(y_true, p, bins=RELIABILITY_BINS):
    """Equal-width bins over [0, 1]; only bins with members are listed."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    ece = 0.0
    for b in range(bins):
        mask = which == b
        count = int(mask.sum())
        if not count:
            continue
        mean_predicted = float(p[mask].mean())
        observed = float(y_true[mask].mean())
        ece += count / len(p) * abs(mean_predicted - observed)
        rows.append({"lo": _round(edges[b]), "hi": _round(edges[b + 1]), "count": count,
                     "mean_predicted": _round(mean_predicted), "observed": _round(observed)})
    return rows, float(ece)


def precision_at_k(y_true, p, block, k):
    """
    Mean over days of the drawn fraction among the top-k probabilities of the
    day. The flat table is day-major with `block` rows per day; a trailing
    partial block is not a day and is dropped.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    days = len(p) // block
    if days == 0 or k <= 0 or k > block:
        return None, 0
    y_days = y_true[:days * block].reshape(days, block)
    p_days = p[:days * block].reshape(days, block)
    top = np.argsort(-p_days, axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(y_days, top, axis=1).sum(axis=1)
    return float(hits.mean() / k), days


def calibration_report(y_true, p, block=None, draw_size=None):
    """The metrics documented in the module docstring, JSON-safe. None when empty."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    n = len(y_true)
    if n == 0 or len(p) != n:
        return None
    base_rate = float(y_true.mean())
    q = np.clip(p, EPS, 1 - EPS)
    report = {
        "rows": int(n),
        "base_rate": _round(base_rate),
        "brier": _round(np.mean((p - y_true) ** 2)),
        "brier_base_rate": _round(base_rate * (1 - base_rate)),
        "log_loss": _round(-np.mean(y_true * np.log(q) + (1 - y_true) * np.log(1 - q))),
        "log_loss_base_rate": _round(-(base_rate * math.log(max(base_rate, EPS)) + (1 - base_rate) * math.log(max(1 - base_rate, EPS)))),
        "auc": None,
    }
    if 0 < base_rate < 1:
        # rank-based AUC without sklearn: P(score of a drawn number > score of an undrawn one)
        order = np.argsort(p, kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        # average ranks for ties
        sorted_p = p[order]
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_p[j + 1] == sorted_p[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
            i = j + 1
        positives = y_true == 1
        n_pos = int(positives.sum())
        n_neg = n - n_pos
        report["auc"] = _round((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    rows, ece = reliability(y_true, p)
    report["reliability"] = rows
    report["ece"] = _round(ece)
    if block and draw_size:
        precision, days = precision_at_k(y_true, p, int(block), int(draw_size))
        report["precision_at_k"] = _round(precision)
        report["hits_at_k"] = _round(precision * draw_size) if precision is not None else None
        report["chance_precision"] = _round(draw_size / block)
        report["chance_hits"] = _round(draw_size * draw_size / block)
        report["days"] = days
        report["k"] = int(draw_size)
    return report


def describe(report, label=""):
    """One log line for the trainer."""
    if not report:
        return f"{label}: no held-out rows to score"
    parts = [f"brier {report['brier']} (base rate {report['brier_base_rate']})",
             f"log-loss {report['log_loss']} ({report['log_loss_base_rate']})",
             f"auc {report['auc']}", f"ece {report['ece']}"]
    if report.get("precision_at_k") is not None:
        parts.append(f"top-{report['k']} hits/day {report['hits_at_k']} (chance {report['chance_hits']}, {report['days']} days)")
    return f"{label}: " + ", ".join(parts)


if __name__ == "__main__":
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    rng = np.random.default_rng(1)
    days, block, k = 200, 45, 6
    y = np.zeros((days, block))
    for d in range(days):
        y[d, rng.choice(block, size=k, replace=False)] = 1
    y = y.ravel()

    perfect = calibration_report(y, y, block, k)
    check(perfect["brier"] == 0 and perfect["precision_at_k"] == 1.0 and perfect["hits_at_k"] == 6.0 and perfect["auc"] == 1.0,
          f"a perfect predictor scores perfectly: {perfect}")
    check(perfect["ece"] == 0 and perfect["days"] == days, "perfect calibration, every day counted")

    noise = calibration_report(y, rng.uniform(size=len(y)), block, k)
    check(abs(noise["precision_at_k"] - k / block) < 0.03, f"random scores hit at chance: {noise['precision_at_k']} vs {k / block:.3f}")
    check(abs(noise["auc"] - 0.5) < 0.02, f"random AUC near 0.5: {noise['auc']}")
    check(noise["brier"] > noise["brier_base_rate"], "uniform noise is worse than the base rate on Brier")

    calibrated = np.where(y == 1, rng.beta(4, 2, size=len(y)), rng.beta(2, 6, size=len(y)))
    # rescale so the mean probability equals the base rate: then the bins should be near the diagonal
    cal = calibration_report(y, calibrated, block, k)
    check(cal["auc"] > 0.8 and cal["precision_at_k"] > 0.3, f"an informative score ranks well: {cal['auc']}, {cal['precision_at_k']}")
    check(len(cal["reliability"]) >= 5 and all(r["count"] > 0 for r in cal["reliability"]), "reliability bins are populated")
    check(sum(r["count"] for r in cal["reliability"]) == len(y), "every row lands in exactly one bin")

    constant = calibration_report(y, np.full(len(y), k / block), block, k)
    check(abs(constant["brier"] - constant["brier_base_rate"]) < 1e-9 and constant["ece"] < 0.01 and constant["auc"] == 0.5,
          f"the base-rate constant: Brier equals the base-rate Brier, calibrated, AUC 0.5 (all ties): {constant['brier']} {constant['ece']} {constant['auc']}")

    check(calibration_report([], []) is None and calibration_report([1, 0], [0.5]) is None, "empty or mismatched input -> None")
    one_class = calibration_report([1, 1, 1], [0.2, 0.9, 0.5])
    check(one_class["auc"] is None and one_class["brier"] is not None, "one class: no AUC, Brier still defined")
    partial = calibration_report(y[:block * 3 + 7], y[:block * 3 + 7], block, k)
    check(partial["days"] == 3, "a trailing partial day is dropped from the ranking")
    check(calibration_report(y, y, block, block + 1)["precision_at_k"] is None, "k larger than the block is not a ranking")

    import json
    json.dumps(cal)
    check("top-6 hits/day 6.0" in describe(perfect, "x"), describe(perfect, "x"))

    for message in failures:
        print("FAIL:", message)
    print(f"Calibration self-check: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    print("  ", describe(cal, "informative synthetic"))
    print("  ", describe(noise, "uniform noise"))
    raise SystemExit(1 if failures else 0)
