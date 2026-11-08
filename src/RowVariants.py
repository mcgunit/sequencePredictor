"""
The variant rows of a model (owner, 8 Nov 2026): next to each model's own row ("settle": the ticket of its highest
scores, what every row has always served) two tracked rows for the admin pages -

  <row> · Sampled    the ticket DRAWN from the row's own scores instead of their top: each number's chance is its
                     score's share, sharpened or flattened by a tuned temperature, drawn without repeats. Same
                     scores, other ticket - so the row varies from draw to draw where the top barely moves.
  <row> · Memorise   the model fitted to chase the noise (built per family elsewhere; named here).

Why (the 8 Nov 2026 repetition check): on a draw with nothing to learn, a model either settles on the counts and
serves the same ticket day after day (the deep-learning rows now repeat 70-95% of their numbers), or memorises
noise and varies (the early shallow LSTM/GRU/TCN). Neither predicts: hits per ticket were at chance in both
periods. The sampled row is the honest way to vary: it claims nothing beyond the row's own scores.

Temperature T (per model and game: bestParams "sampleTemperatures": {"<row name>": T}, default 1):
  p_i  proportional to  s_i ** (1 / T)      s = the row's scores made non-negative (shifted by the minimum when one
                                            is negative; all equal or all zero = uniform)
  T -> 0 is the settle ticket (the top), T = 1 the scores' own shares, T large uniform.

The draw is seeded with game, draw date and row name (seed_for): a saved ticket is reproducible, and a rebuilt day
draws the same ticket again. The ticket is stored most-probable-first (the number games' rule, 2 Nov 2026).

Kept out of the ensembles, the meta-learner features and the best-combination search (owner, 8 Nov 2026): a
sampled row has its parent's scores, so as an input it adds nothing but noise; memorise rows are admitted later
only through the gate (with versus without them).

    python3 -m src.RowVariants     # self-check
"""

import math
import zlib

import numpy as np

SAMPLED = "Sampled"
MEMORISE = "Memorise"
SEPARATOR = " · "
VARIANTS = (SAMPLED, MEMORISE)
DEFAULT_TEMPERATURE = 1.0
# below this the draw is the top-k itself (p ** (1/T) overflows long before T reaches 0)
MIN_TEMPERATURE = 0.01


def variant_name(row_name, variant):
    return f"{row_name}{SEPARATOR}{variant}"


def split_name(row_name):
    """(parent row name, variant) for a variant row, (row_name, None) for any other row."""
    for variant in VARIANTS:
        suffix = SEPARATOR + variant
        if isinstance(row_name, str) and row_name.endswith(suffix):
            return row_name[: -len(suffix)], variant
    return row_name, None


def is_variant(row_name):
    return split_name(row_name)[1] is not None


def without_variants(rows):
    """The rows a consumer of the model rows reads by default: every row but the variant rows."""
    return [r for r in rows if not (isinstance(r, dict) and is_variant(r.get("name")))]


def seed_for(game, date, row_name):
    """A stable 32-bit seed per game, draw date (any str()-able form) and row."""
    return zlib.crc32(f"{game}|{date}|{row_name}".encode("utf-8"))


def probabilities(scores, temperature=DEFAULT_TEMPERATURE):
    """
    (labels, p): the tempered draw probabilities of a {label: score} dict. Non-finite scores count as the lowest.
    temperature at or below MIN_TEMPERATURE is not handled here (sample_ticket takes the top instead).
    """
    labels = list(scores)
    if not labels:
        return labels, np.array([])
    s = np.array([float(scores[k]) if isinstance(scores[k], (int, float)) and math.isfinite(scores[k]) else np.nan
                  for k in labels])
    if np.all(np.isnan(s)):
        return labels, np.full(len(labels), 1.0 / len(labels))
    s = np.where(np.isnan(s), np.nanmin(s), s)
    if s.min() < 0:
        s = s - s.min()
    if s.sum() <= 0:
        return labels, np.full(len(labels), 1.0 / len(labels))
    p = s / s.sum()
    t = max(float(temperature), MIN_TEMPERATURE)
    # in log space: p ** (1/T) underflows for small T; a zero share stays zero
    with np.errstate(divide="ignore"):
        logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)) / t, -np.inf)
    logp -= logp[np.isfinite(logp)].max()
    w = np.exp(logp)
    return labels, w / w.sum()


def _top(scores, k):
    return [label for label, _ in sorted(scores.items(), key=lambda kv: -float(kv[1]) if isinstance(kv[1], (int, float))
                                         and math.isfinite(kv[1]) else math.inf)[:k]]


def sample_ticket(scores, k, temperature, seed):
    """
    k distinct labels drawn from {label: score} with the tempered probabilities, most-probable-first (by score).
    Fewer than k labels with a chance: the rest by score. temperature <= MIN_TEMPERATURE: the top k.
    """
    if not scores or k <= 0:
        return []
    k = min(int(k), len(scores))
    if temperature is None or float(temperature) <= MIN_TEMPERATURE:
        return _top(scores, k)
    labels, p = probabilities(scores, temperature)
    rng = np.random.default_rng(seed)
    possible = int((p > 0).sum())
    drawn = list(rng.choice(len(labels), size=min(k, possible), replace=False, p=p)) if possible else []
    chosen = [labels[i] for i in drawn]
    if len(chosen) < k:
        chosen += [label for label in _top(scores, len(scores)) if label not in chosen][: k - len(chosen)]
    order = {label: i for i, label in enumerate(_top(scores, len(scores)))}
    return sorted(chosen, key=lambda label: order[label])


def sample_positional(slot_scores, temperature, seed):
    """One label per slot (Pick3's digits, Joker+'s positions, a market slot), each slot drawn on its own."""
    ticket = []
    for i, scores in enumerate(slot_scores):
        pick = sample_ticket(scores, 1, temperature, (seed + 7919 * (i + 1)) % (2 ** 32))
        ticket.append(pick[0] if pick else None)
    return ticket


def temperature_for(best_params, row_name):
    """The row's tuned temperature (bestParams "sampleTemperatures"[row name]), DEFAULT_TEMPERATURE without one."""
    table = (best_params or {}).get("sampleTemperatures")
    value = table.get(row_name, DEFAULT_TEMPERATURE) if isinstance(table, dict) else DEFAULT_TEMPERATURE
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE
    return value if math.isfinite(value) and value >= 0 else DEFAULT_TEMPERATURE


# The row's own number scores, left on the row by Predictor.applyMultiPick (the same scores its ticket is ordered
# by) until add_sampled_rows reads and removes them. String keys, so the deep-learning child's rows carry them
# through JSON unchanged.
SCORES_KEY = "_sampleScores"


def stash_scores(row, scores):
    try:
        row[SCORES_KEY] = {str(int(k)): float(v) for k, v in scores.items()
                           if isinstance(v, (int, float)) and math.isfinite(float(v))}
    except (TypeError, ValueError):
        row.pop(SCORES_KEY, None)


def add_sampled_rows(rows, game, day, best_params, mains_count, extra_count=0):
    """
    Appends "<row> · Sampled" to `rows` for every row that left its scores (stash_scores), and removes the
    stashed scores from every row - call it after the last consumer of the day's rows (the ensembles, the RL
    ticket), right before the day file is written. The sampled row: the mains drawn from the row's scores with
    its temperature (the parent's ticket length, its specials kept as they are - the scores cover the mains
    only), each further ticket of the row (keno's sub-selections) the first that many of the sampled ticket, a
    multi-pick game's extras drawn from the numbers left. Returns the names added.
    """
    added = []
    for row in list(rows):
        if not isinstance(row, dict):
            continue
        stashed = row.pop(SCORES_KEY, None)
        name = row.get("name")
        if not stashed or not name or is_variant(name) or not row.get("predictions") or not row["predictions"][0]:
            continue
        scores = {int(k): v for k, v in stashed.items()}
        ticket = list(row["predictions"][0])
        count = min(mains_count or len(ticket), len(ticket))
        mains, specials = ticket[:count], ticket[count:]
        temperature = temperature_for(best_params, name)
        seed = seed_for(game, day, name)
        drawn = sample_ticket(scores, len(mains), temperature, seed)
        if len(drawn) < len(mains):
            continue
        # the scores it was drawn from, rounded: what the temperature tuner scores (log_score) against the draws
        sampled = {"name": variant_name(name, SAMPLED), "predictions": [[int(n) for n in drawn] + list(specials)],
                   "variant": SAMPLED, "parent": name, "sampleTemperature": temperature,
                   "scores": {str(k): round(float(v), 6) for k, v in sorted(scores.items())}}
        for extra_ticket in row["predictions"][1:]:
            sampled["predictions"].append([int(n) for n in drawn[:len(extra_ticket)]] if extra_ticket else extra_ticket)
        if extra_count and row.get("multiPick"):
            rest = {k: v for k, v in scores.items() if k not in set(drawn)}
            extras = sample_ticket(rest, extra_count, temperature, (seed + 1) % (2 ** 32))
            if extras:
                sampled["multiPick"] = [int(n) for n in extras]
        rows.append(sampled)
        added.append(sampled["name"])
    for row in rows:
        if isinstance(row, dict):
            row.pop(SCORES_KEY, None)
    return added


def log_score(scores, drawn, temperature):
    """
    The mean log-probability the tempered single-draw distribution gives the drawn labels: the proper score the
    temperature is tuned on (one lucky ticket cannot carry it, unlike hits). Labels the scores lack count as the
    smallest probability present.
    """
    labels, p = probabilities(scores, temperature)
    if not labels:
        return None
    index = {label: i for i, label in enumerate(labels)}
    floor = max(float(p[p > 0].min()) if (p > 0).any() else 1e-12, 1e-12)
    values = [math.log(max(float(p[index[d]]), floor)) if d in index else math.log(floor) for d in drawn]
    return sum(values) / len(values) if values else None


def _self_check():
    scores = {n: float(n) for n in range(1, 46)}  # 45 favours 45 most
    # determinism and form
    a = sample_ticket(scores, 6, 1.0, seed_for("lotto", "2026-11-8", "Markov Model"))
    b = sample_ticket(scores, 6, 1.0, seed_for("lotto", "2026-11-8", "Markov Model"))
    c = sample_ticket(scores, 6, 1.0, seed_for("lotto", "2026-11-11", "Markov Model"))
    assert a == b and len(set(a)) == 6 and a != c, (a, b, c)
    assert a == sorted(a, reverse=True), a                                    # most-probable-first
    # T -> 0 is the settle ticket; T large is near uniform
    assert sample_ticket(scores, 6, 0.0, 1) == [45, 44, 43, 42, 41, 40]
    assert sample_ticket(scores, 6, 0.005, 1) == [45, 44, 43, 42, 41, 40]
    _, flat = probabilities(scores, 1000.0)
    assert flat.max() / flat.min() < 1.01
    _, sharp = probabilities(scores, 0.05)
    assert sharp.argmax() == 44 and sharp.max() > 0.25 and sharp.max() > 10 * flat.max()   # 45 vs 44: (45/44)**20 ~ 1.6
    # T = 1 keeps the shares
    _, own = probabilities({1: 1.0, 2: 3.0}, 1.0)
    assert abs(own[1] - 0.75) < 1e-12
    # frequencies of inclusion follow the scores (many seeds)
    hits = {n: 0 for n in scores}
    for s in range(3000):
        for n in sample_ticket(scores, 6, 1.0, s):
            hits[n] += 1
    assert hits[45] > 2.5 * hits[5], (hits[45], hits[5])
    # negative, non-finite, degenerate scores
    assert len(sample_ticket({1: -3.0, 2: -1.0, 3: float("nan"), 4: 0.5}, 2, 1.0, 3)) == 2
    assert sorted(sample_ticket({1: 0.0, 2: 0.0, 3: 0.0}, 3, 1.0, 3)) == [1, 2, 3]
    assert sample_ticket({1: 1.0, 2: 0.0, 3: 0.0}, 2, 1.0, 3)[0] == 1          # only 1 has a chance; the rest by score
    assert sample_ticket({}, 6, 1.0, 1) == []
    # positional: one per slot, deterministic, independent slots
    slots = [{d: 1.0 for d in range(10)} for _ in range(3)]
    p1 = sample_positional(slots, 1.0, 42)
    assert p1 == sample_positional(slots, 1.0, 42) and len(p1) == 3 and all(0 <= d <= 9 for d in p1)
    assert sample_positional([{7: 5.0, 1: 0.1}], 0.0, 1) == [7]
    # names
    name = variant_name("Markov Model", SAMPLED)
    assert name == "Markov Model · Sampled" and split_name(name) == ("Markov Model", SAMPLED)
    assert split_name("Markov Model") == ("Markov Model", None) and not is_variant("RL Ticket Model")
    assert without_variants([{"name": name}, {"name": "Markov Model"}]) == [{"name": "Markov Model"}]
    # temperature key
    assert temperature_for({"sampleTemperatures": {"Markov Model": 0.4}}, "Markov Model") == 0.4
    assert temperature_for({}, "Markov Model") == DEFAULT_TEMPERATURE
    assert temperature_for({"sampleTemperatures": {"Markov Model": "x"}}, "Markov Model") == DEFAULT_TEMPERATURE
    # the day's rows: a lotto row with multi-pick, a euromillions-like row with stars, keno with subsets, a row
    # without scores, and the stash removed from every row
    import json
    lotto = {"name": "Markov Model", "predictions": [[45, 44, 43, 42, 41, 40]], "multiPick": [39, 38, 37]}
    stash_scores(lotto, scores)
    hybrid = {"name": "HybridStatisticalModel", "predictions": [[1, 2, 3, 4, 5, 6]]}
    rows = json.loads(json.dumps([lotto, hybrid]))                          # through JSON, as the DL child's rows come
    assert add_sampled_rows(rows, "lotto", "2026-11-8", {}, 6, 3) == ["Markov Model · Sampled"]
    s_row = rows[-1]
    assert len(rows) == 3 and all(SCORES_KEY not in r for r in rows)
    assert s_row["parent"] == "Markov Model" and s_row["variant"] == SAMPLED and len(set(s_row["predictions"][0])) == 6
    assert s_row["scores"]["45"] == 45.0 and len(s_row["scores"]) == 45
    assert len(s_row["multiPick"]) == 3 and not set(s_row["multiPick"]) & set(s_row["predictions"][0])
    rows2 = json.loads(json.dumps([dict(lotto, **{SCORES_KEY: {str(k): v for k, v in scores.items()}}), hybrid]))
    add_sampled_rows(rows2, "lotto", "2026-11-8", {}, 6, 3)
    assert rows2[-1] == s_row                                                  # reproducible
    em = {"name": "GNN Model", "predictions": [[50, 49, 48, 47, 46, 2, 11]]}
    stash_scores(em, {n: float(n) for n in range(1, 51)})
    rows = [em]
    add_sampled_rows(rows, "euromillions", "2026-11-6", {"sampleTemperatures": {"GNN Model": 0.0}}, 5)
    assert rows[-1]["predictions"][0] == [50, 49, 48, 47, 46, 2, 11]          # T=0: the parent's mains, stars kept
    keno = {"name": "XGBoost Model", "predictions": [list(range(70, 50, -1)), list(range(70, 60, -1))]}
    stash_scores(keno, {n: float(n) for n in range(1, 71)})
    rows = [keno]
    add_sampled_rows(rows, "keno", "2026-11-7", {}, 20)
    k_row = rows[-1]
    assert len(k_row["predictions"][0]) == 20 and k_row["predictions"][1] == k_row["predictions"][0][:10]
    assert add_sampled_rows([dict(s_row, **{SCORES_KEY: {"1": 1.0}})], "lotto", "d", {}, 6) == []   # never of a variant
    # log score: a flatter distribution scores ln(1/45) on anything; the true favourite better when sharper
    assert abs(log_score(scores, [3], 1e6) - math.log(1 / 45)) < 1e-4
    assert log_score(scores, [45, 44], 0.5) > log_score(scores, [45, 44], 1.0)
    print("RowVariants self-check OK")


if __name__ == "__main__":
    _self_check()
