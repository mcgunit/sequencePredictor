"""
Irrelevant-feature control for the meta-learners (README "Null controls",
roadmap item 6): a random column appended to the meta-learner table must not
gain stable importance.

The meta-learners (logistic, gradient boosting, quantum kernel, VQC and the
classical RBF control) are fitted on the base models' scores. If one of them
gives weight to a column that is provably unrelated to the draw, then its
weights on the real columns are not evidence of anything either - the model
is fitting noise, and its held-out metrics are the selection effect over
noise the null band measures at the row level. The control therefore appends
noise columns to the table and asks two questions per variant:

  training side   how much of the model's attribution goes to the noise
                  columns? (permutation importance on the training rows - a
                  model that fits noise loses ranking power when its noise
                  column is scrambled; a model that ignores it loses nothing)
  held-out side   which real columns matter more than noise does? (the same
                  importance on the held-out days; the noise columns' spread
                  there is the band a real base model must clear)

Importance is the ranking power lost, |AUC - 0.5| before minus after the
scramble, not the plain AUC drop: a Platt-scaled SVM fitted on noise often
comes out with probabilities INVERTED against the labels (AUC below 0.5 on
its own rows), and there scrambling a column it leans on moves the AUC UP,
which a signed drop would read as "ignored". Ranking power in either
direction reads "leaned on" either way; a column the model ignores still
scores exactly 0.

The noise columns are shuffled copies of real columns: the same marginal
distribution as a base model's scores, no relation to the label or the day.
Pure functions over arrays and fitted models; IrrelevantFeatureControl.py
loads the tables, runs the weekly loop and writes data/controls/features/.

Self-check: python3 -m src.FeatureControl
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score

# A variant "fits noise" when, on its training rows, a noise column matters
# at least NOISE_RATIO_LIMIT times as much as an average real column AND the
# pooled noise importance is more than T_LIMIT standard errors above zero (so
# a handful of tiny positive numbers does not count). The ratio, not the
# share of the total, because one or two strongly informative base models
# would otherwise dilute the noise's share however hard the model leaned on
# it. A model that cannot tell noise from signal lands near 1.0; one that
# ignores noise at 0. The share of the total attribution is still reported
# (with three noise columns among eleven, "cannot tell" reads near 27%).
NOISE_RATIO_LIMIT = 0.10
T_LIMIT = 2.0
# The held-out band is the noise columns' mean + 2 sd - the rule the other
# controls use for their null comparisons.
BAND_SD = 2.0


def auc(y, p):
    """ROC AUC, NaN when the labels have one class (then nothing can be ranked)."""
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(p, dtype=float)))


def noise_names(count):
    return [f"noise{i + 1}" for i in range(count)]


def append_noise(X, rng, count=3):
    """
    Appends `count` shuffled copies of real columns to X. Each noise column is
    a row permutation of one real column, so it has exactly a base model's
    marginal distribution and no relation to the label or to the day. Source
    columns are drawn without replacement while there are enough of them.
    Returns (X_with_noise, source_column_indices).
    """
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    sources = rng.choice(m, size=count, replace=count > m)
    noise = np.column_stack([X[rng.permutation(n), j] for j in sources])
    return np.hstack([X, noise]), [int(j) for j in sources]


def subsample_blocks(n_rows, block, max_rows, rng):
    """
    Row indices of at most max_rows rows taken as whole blocks (a block is one
    day of the day-major tables, so a positional table keeps its
    day x position x digit layout). None when no subsampling is needed.
    """
    if max_rows is None or n_rows <= max_rows:
        return None
    block = max(1, int(block))
    n_blocks = n_rows // block
    keep_blocks = np.sort(rng.choice(n_blocks, size=max(1, max_rows // block), replace=False))
    return np.concatenate([np.arange(b * block, (b + 1) * block) for b in keep_blocks])


def permutation_importance(predict, X, y, rng, max_rows=None, block=1, score=auc, chance=0.5):
    """
    importance[j] = |score(X) - chance| - |score(X with column j permuted
    across rows) - chance|: the ranking power the model loses when column j
    is scrambled, in either direction (see the module header for why not the
    signed drop). A column the model ignores scores exactly 0; a column it
    leans on scores positive on the rows it was fitted on and, if the column
    carries real information, on held-out rows too. Rows may be subsampled
    (max_rows, whole blocks, seeded) to bound the cost; the same rows are used
    for every column. Returns (base_score - the plain score, so a caller can
    still see an inverted model - and the importances).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    keep = subsample_blocks(len(X), block, max_rows, rng)
    if keep is not None:
        X, y = X[keep], y[keep]
    base = score(y, predict(X))
    power = abs(base - chance)
    importances = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        permuted = X.copy()
        permuted[:, j] = permuted[rng.permutation(len(permuted)), j]
        importances[j] = power - abs(score(y, predict(permuted)) - chance)
    return float(base), importances


def flat_predict(model):
    """predict(X) -> P(class 1) for a flat per-number meta-learner."""
    return lambda X: model.predict_proba(np.asarray(X, dtype=float))[:, 1]


def positional_predict(position_models, positions, classes):
    """
    predict(X) over a day-major positional table (TrainMetaLearner.
    build_positional_training_table): each slot's classifier scores its own
    rows, and the result is the flat P(digit in slot) vector aligned with the
    table's labels, so one AUC covers every slot.
    """
    def predict(X):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        n_days = len(X) // (positions * classes)
        X_days = X.reshape(n_days, positions, classes, n_features)
        out = np.zeros((n_days, positions, classes))
        for pos, model in enumerate(position_models):
            out[:, pos] = model.predict_proba(X_days[:, pos].reshape(-1, n_features))[:, 1].reshape(n_days, classes)
        return out.reshape(-1)
    return predict


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def summarise(feature_names, noise_count, runs, without=None):
    """
    Folds the repeats of one variant into the record the page reads.

    runs:    one dict per repeat, fitted WITH the noise columns: "train" and
             "heldout" are the permutation importances over the real columns
             followed by the noise columns; "train_auc" / "heldout_auc" the
             base scores.
    without: the same for the single fit WITHOUT noise (importances over the
             real columns only), or None.
    """
    n_real = len(feature_names)
    train = np.array([r["train"] for r in runs], dtype=float)
    held = np.array([r["heldout"] for r in runs], dtype=float)
    noise_train = train[:, n_real:].ravel()
    noise_held = held[:, n_real:].ravel()
    real_train_mean = np.nanmean(train[:, :n_real], axis=0)
    real_held_mean = np.nanmean(held[:, :n_real], axis=0)

    # share of the positive training-side attribution that lands on noise,
    # per repeat, then averaged
    shares = []
    for row in train:
        positive = np.clip(np.nan_to_num(row), 0.0, None)
        total = positive.sum()
        shares.append(float(positive[n_real:].sum() / total) if total > 0 else 0.0)
    noise_share = float(np.mean(shares))

    pooled = noise_train[np.isfinite(noise_train)]
    n = len(pooled)
    mean = float(pooled.mean()) if n else 0.0
    sd = float(pooled.std(ddof=1)) if n > 1 else 0.0
    if sd > 0:
        t = mean / (sd / math.sqrt(n))
    else:
        t = math.inf if mean > 0 else 0.0
    # how much a noise column matters in training against an average real one
    real_positive = np.clip(np.nan_to_num(real_train_mean), 0.0, None)
    real_mean_importance = float(real_positive.mean()) if n_real else 0.0
    if real_mean_importance > 0:
        ratio = mean / real_mean_importance
    else:
        ratio = math.inf if mean > 0 else 0.0
    fits_noise = bool(ratio > NOISE_RATIO_LIMIT and t > T_LIMIT)

    pooled_held = noise_held[np.isfinite(noise_held)]
    held_mean = float(pooled_held.mean()) if len(pooled_held) else 0.0
    held_sd = float(pooled_held.std(ddof=1)) if len(pooled_held) > 1 else 0.0
    band = held_mean + BAND_SD * held_sd
    order = np.argsort(-real_held_mean)
    above = [feature_names[j] for j in order if np.isfinite(real_held_mean[j]) and real_held_mean[j] > band]

    with_auc = [r["heldout_auc"] for r in runs if r.get("heldout_auc") is not None and math.isfinite(r["heldout_auc"])]
    heldout_with = float(np.mean(with_auc)) if with_auc else None
    heldout_without = _finite(without["heldout_auc"]) if without and without.get("heldout_auc") is not None else None
    cost = (heldout_without - heldout_with) if heldout_without is not None and heldout_with is not None else None

    columns = {}
    for j, name in enumerate(feature_names):
        columns[name] = {
            "train_importance": _finite(real_train_mean[j]),
            "heldout_importance": _finite(real_held_mean[j]),
            "above_noise_band": name in above,
            "heldout_importance_without_noise": _finite(without["heldout"][j]) if without else None,
        }

    return {
        "repeats": len(runs),
        "noise_columns": noise_count,
        "noise": {
            "train_importance_mean": mean,
            "train_importance_sd": sd,
            "train_t": _finite(t) if math.isfinite(t) else None,
            "share_of_train_importance": noise_share,
            "noise_to_real_ratio": _finite(ratio) if math.isfinite(ratio) else None,
            "real_train_importance_mean": real_mean_importance,
            "heldout_importance_mean": held_mean,
            "heldout_importance_sd": held_sd,
            "heldout_band": band,
        },
        "columns": columns,
        "real_above_noise_band": above,
        "heldout_auc_without_noise": heldout_without,
        "heldout_auc_with_noise_mean": heldout_with,
        "heldout_cost_of_noise": cost,
        "fits_noise": fits_noise,
        "rule": (f"fits_noise when a noise column's training-side permutation importance is more than "
                 f"{NOISE_RATIO_LIMIT:g} times an average real column's and the pooled noise importance lies more "
                 f"than {T_LIMIT:g} standard errors above zero; a real column is above the noise band when its "
                 f"held-out importance exceeds the noise columns' mean + {BAND_SD:g} sd"),
    }


def describe(name, summary):
    """One line per variant for the log."""
    noise = summary["noise"]
    verdict = "FITS NOISE" if summary["fits_noise"] else "noise ignored"
    above = ", ".join(summary["real_above_noise_band"]) or "none"
    without = summary.get("heldout_auc_without_noise")
    with_noise = summary.get("heldout_auc_with_noise_mean")
    auc_text = ""
    if without is not None and with_noise is not None:
        auc_text = f" | held-out AUC {without:.4f} -> {with_noise:.4f} with noise"
    t_text = f"{noise['train_t']:.1f}" if noise.get("train_t") is not None else "inf"
    ratio = noise.get("noise_to_real_ratio")
    ratio_text = f"{ratio:.2f}x" if ratio is not None else "inf"
    return (f"{name}: {verdict} - noise {ratio_text} an average real column in training, "
            f"{noise['share_of_train_importance']:.1%} of the attribution (t {t_text}){auc_text} | "
            f"above the noise band: {above}")


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(5)

    # A table with six "base models", two of them informative, and a label
    # that depends on those two only.
    n = 4000
    X = rng.normal(size=(n, 6))
    logit = 2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    names = [f"m{i}" for i in range(6)]
    cut = int(n * 0.8)

    # 1. A column the model ignores has importance exactly 0.
    ignore_all = lambda Z: np.full(len(Z), 0.5) + 1e-9 * Z[:, 0]
    _, imp = permutation_importance(ignore_all, X[cut:], y[cut:], np.random.default_rng(1))
    assert np.max(np.abs(imp[1:])) == 0.0, "ignored columns must score exactly zero"
    print("a column the model ignores has permutation importance exactly 0")

    # 1b. An inverted model - probabilities anti-correlated with the labels,
    #     as a Platt-scaled SVM fitted on noise often is - leans on exactly
    #     the columns the upright one does, and the importance says so.
    upright = LogisticRegression(max_iter=1000).fit(X[:cut], y[:cut])
    good = flat_predict(upright)
    inverted = lambda Z: 1.0 - good(Z)
    base_up, imp_up = permutation_importance(good, X[cut:], y[cut:], np.random.default_rng(7))
    base_inv, imp_inv = permutation_importance(inverted, X[cut:], y[cut:], np.random.default_rng(7))
    assert base_up > 0.8 and base_inv < 0.2, (base_up, base_inv)
    assert np.allclose(imp_up, imp_inv), "an inverted model must show the same importances as the upright one"
    assert imp_up[0] > 0.05 and imp_up[1] > 0.05
    print(f"an inverted model (AUC {base_inv:.3f}) shows the same column importances as the upright one (AUC {base_up:.3f})")

    # 2. Noise columns are shuffled copies: same marginal, no relation.
    X_with, sources = append_noise(X, np.random.default_rng(2), 3)
    assert X_with.shape == (n, 9) and len(sources) == 3
    for k, j in enumerate(sources):
        assert np.allclose(np.sort(X_with[:, 6 + k]), np.sort(X[:, j])), "a noise column must be a permutation of its source"
        assert abs(np.corrcoef(X_with[:, 6 + k], y)[0, 1]) < 0.05
    print("noise columns are row permutations of real columns, uncorrelated with the label")

    # 3. Whole-block subsampling keeps blocks intact.
    keep = subsample_blocks(1000, block=10, max_rows=300, rng=np.random.default_rng(3))
    assert len(keep) == 300 and all(keep[i * 10] % 10 == 0 for i in range(30))
    print("row subsampling keeps whole day blocks")

    # 4. A well-posed model on plenty of rows ignores noise, and its two
    #    informative columns clear the noise band on the held-out side.
    def run(fit, repeats=3, noise_count=3, seed=10):
        runs = []
        base = fit(X[:cut], y[:cut])
        predict = flat_predict(base)
        r0 = np.random.default_rng(seed)
        tr_auc, tr_imp = permutation_importance(predict, X[:cut], y[:cut], r0, max_rows=2000)
        he_auc, he_imp = permutation_importance(predict, X[cut:], y[cut:], r0)
        without = {"train_auc": tr_auc, "heldout_auc": he_auc, "train": tr_imp.tolist(), "heldout": he_imp.tolist()}
        for r in range(repeats):
            rr = np.random.default_rng(seed + 100 * (r + 1))
            Xw, _ = append_noise(X, rr, noise_count)
            model = fit(Xw[:cut], y[:cut])
            p = flat_predict(model)
            tr_auc, tr_imp = permutation_importance(p, Xw[:cut], y[:cut], rr, max_rows=2000)
            he_auc, he_imp = permutation_importance(p, Xw[cut:], y[cut:], rr)
            runs.append({"train_auc": tr_auc, "heldout_auc": he_auc, "train": tr_imp.tolist(), "heldout": he_imp.tolist()})
        return summarise(names, noise_count, runs, without)

    logistic = run(lambda A, b: LogisticRegression(class_weight="balanced", max_iter=1000).fit(A, b))
    print(describe("logistic", logistic))
    assert not logistic["fits_noise"], "a logistic regression on 3200 rows must not fit noise"
    assert logistic["noise"]["noise_to_real_ratio"] < NOISE_RATIO_LIMIT
    assert set(logistic["real_above_noise_band"]) >= {"m0", "m1"}, logistic["real_above_noise_band"]
    assert not (set(logistic["real_above_noise_band"]) & {"m3", "m4", "m5"}) or len(logistic["real_above_noise_band"]) <= 3
    assert logistic["heldout_cost_of_noise"] is not None and abs(logistic["heldout_cost_of_noise"]) < 0.02

    # 5. A model that cannot tell noise from signal: a one-nearest-neighbour
    #    classifier memorises the training rows and weighs every column
    #    alike, so scrambling a noise column costs it as much training AUC
    #    as scrambling a real one - and the control says so.
    def memoriser(A, b):
        return KNeighborsClassifier(n_neighbors=1).fit(A[:300], b[:300])
    overfit = run(memoriser)
    print(describe("1-NN on 300 rows", overfit))
    assert overfit["fits_noise"], "a 1-NN on 300 rows must be flagged as fitting noise"
    assert overfit["noise"]["noise_to_real_ratio"] > NOISE_RATIO_LIMIT
    # an unpruned tree on the same rows sits in between: it leans on noise,
    # less than on the two informative columns, and is still flagged
    tree = run(lambda A, b: DecisionTreeClassifier(random_state=0).fit(A[:300], b[:300]))
    print(describe("unpruned tree on 300 rows", tree))
    assert tree["fits_noise"], "an unpruned tree on 300 rows must be flagged as fitting noise"

    # 6. The positional predict covers every slot and lines up with the table.
    class Slot:
        def __init__(self, k):
            self.k = k

        def predict_proba(self, Z):
            p = 1.0 / (1.0 + np.exp(-(Z[:, 0] + self.k)))
            return np.column_stack([1 - p, p])
    positions, classes = 3, 10
    Zt = rng.normal(size=(5 * positions * classes, 4))
    predict = positional_predict([Slot(0), Slot(1), Slot(2)], positions, classes)
    flat = predict(Zt)
    assert flat.shape == (len(Zt),)
    Zd = Zt.reshape(5, positions, classes, 4)
    manual = Slot(1).predict_proba(Zd[:, 1].reshape(-1, 4))[:, 1].reshape(5, classes)
    assert np.allclose(flat.reshape(5, positions, classes)[:, 1], manual)
    print("positional predict scores each slot with its own model in table order")

    # 7. Determinism: same seed, same record.
    again = run(lambda A, b: LogisticRegression(class_weight="balanced", max_iter=1000).fit(A, b))
    assert again["noise"] == logistic["noise"] and again["columns"] == logistic["columns"]
    print("the control is deterministic for a given seed")
    print("FeatureControl self-check OK")


if __name__ == "__main__":
    _self_check()
