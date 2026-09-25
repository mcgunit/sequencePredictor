"""
The objective the Hyperopt* tuners rank their trials by, and its diagnostics.

Why not profit per bet. Until September 2026 a trial's value was its raw
profit per bet over the tuning window (average hits where a game has no
payout table). On a 31-day window that number is decided by whether one
jackpot-tier payout fell inside it. The keno LightGBM parameters served from
10 Sept 2026 scored 5.0 per bet, of which one draw - a 6/6 with its nested
5/5, +348 EUR - was everything; the other 60 bets lost 38 EUR. The pick3
XGBoost and CatBoost parameters of the same week were single straights (+756
and +676 EUR) on windows that otherwise lost, and the pick3 statistical
studies' best values (21 and 42 per bet) are one and two straights. The
repeat-and-average per trial that the tuners once had would not have helped:
the Backtester is seeded per day, so a repeat returns the same number - the
luck is in the draws of the window, not in the model's randomness.

What is scored instead - one number per trial, higher is better:

  payout games scored as sets (keno)
      the lower confidence bound of the per-day profit per bet, after every
      single bet's profit is capped at LUCKY_STRIKE_CAP (Metrics'
      LUCKY_STRIKE_THRESHOLD, 20 EUR net). A jackpot counts as one good bet,
      not as the whole window; the 1-10 EUR tiers keep their full value.
  positional games (pick3, Joker+)
      the lower confidence bound of the per-day slot hits, which every draw
      informs, plus POSITIONAL_PROFIT_WEIGHT x the capped profit bound as a
      tie-breaker. Almost every pick3 window is a flat loss once a straight
      is capped, so profit alone would rank noise. Pick3's slot hits are
      counted here from the drawn-order ticket and draw the Backtester
      stores ("<model>_prediction", "actual_ordered"): its "_hits" column is
      the historical SET count for that game, which scores a right digit in
      the wrong slot - a box - like a straight. Joker+'s "_hits" is already
      L + R, the runs the game pays on, and is used as is.
  games without a payout table (lotto, euromillions, ...)
      the lower confidence bound of the per-day main-ticket hits - the same
      metric as before, now with the bound.

The bound is mean - CONFIDENCE_PENALTY x std / sqrt(days), the estimator
HyperoptEnsemble.py already uses for its subsets: a trial is rewarded for
what the window supports, not for its best day. The raw profit per bet and
the number of lucky strikes travel along as diagnostics (Optuna trial user
attributes, the gate record in bestParams_<game>.json - see TuningGate.py),
so a hijack stays visible instead of being averaged away in silence.

    python3 -m src.TuningScore      # self-check on synthetic rows
"""

import math

import numpy as np

try:
    from src.Metrics import Metrics
except ImportError:  # imported from inside src/, the Backtester's sys.path style
    from Metrics import Metrics

# Net profit above which one bet is a jackpot-tier event (keno 6/7 pays 30,
# every pick3 prize but the 1 EUR consolation) - capped so it counts once.
LUCKY_STRIKE_CAP = float(Metrics.LUCKY_STRIKE_THRESHOLD)
# Standard errors subtracted from the mean; 1.0 is what HyperoptEnsemble uses.
CONFIDENCE_PENALTY = 1.0
# Positional games: capped-profit bound x this is added to the slot-hit bound.
# Per-day capped profit per bet spans about -4..+20, slot-hit bounds differ by
# hundredths, so at 0.001 the profit only ever separates equal hit rates.
POSITIONAL_PROFIT_WEIGHT = 0.001

NEG_INF = float("-inf")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def bet_profits(row, model_name):
    """
    Every bet this model placed on one backtest day: the main ticket's profit
    (pick3, Joker+ - "<model>_profit") and each Keno subset's
    ("<model>_subset_<size>_profit"), as the Backtester wrote them.
    """
    profits = []
    main = row.get(f"{model_name}_profit")
    if _number(main):
        profits.append(float(main))
    prefix = f"{model_name}_subset_"
    for key, value in row.items():
        if key.startswith(prefix) and key.endswith("_profit") and _number(value):
            profits.append(float(value))
    return profits


def slot_hits(row, model_name):
    """
    Digits in the right slot for one positional-game day, from the ticket in
    drawn order and the draw in drawn order the Backtester stores. None when
    the row does not carry both (an older row, a day the model errored).
    """
    prediction = row.get(f"{model_name}_prediction")
    actual = row.get("actual_ordered")
    if not isinstance(prediction, (list, tuple)) or not isinstance(actual, (list, tuple)) or not prediction or not actual:
        return None
    try:
        return float(sum(1 for p, a in zip(prediction, actual) if int(p) == int(a)))
    except (TypeError, ValueError):
        return None


def counts_slots_itself(game):
    """Pick3 (any positional game but Joker+): slot hits come from slot_hits(), not from "_hits"."""
    return "jokerplus" not in str(game or "").lower()


def lower_bound(values, penalty=CONFIDENCE_PENALTY):
    """
    mean - penalty x sample std / sqrt(n) of `values`; the plain mean for a
    single value or penalty 0; None when there is nothing to bound.
    """
    n = len(values)
    if n == 0:
        return None
    mean = float(np.mean(values))
    if n == 1 or not penalty:
        return mean
    return mean - float(penalty) * float(np.std(values, ddof=1)) / math.sqrt(n)


def score_bets_by_day(bets_by_day, hits_by_day=None, payout=True, positional=False,
                      penalty=CONFIDENCE_PENALTY, cap=LUCKY_STRIKE_CAP):
    """
    The objective from per-day material: `bets_by_day` is a list with, per
    day, the list of that day's bet profits (net EUR); `hits_by_day` the
    per-day hit count where one exists. Returns the dict documented on
    score_rows(). Days without bets (and without hits) are not days.
    """
    profit_days = []
    bets = 0
    total = 0.0
    strikes = 0
    for profits in bets_by_day or []:
        profits = [float(p) for p in profits if _number(p)]
        if not profits:
            continue
        bets += len(profits)
        total += sum(profits)
        strikes += sum(1 for p in profits if p >= cap)
        profit_days.append(sum(min(p, cap) for p in profits) / len(profits))
    hits_days = [float(h) for h in (hits_by_day or []) if _number(h)]

    capped_bound = lower_bound(profit_days, penalty)
    hits_bound = lower_bound(hits_days, penalty)

    if positional:
        kind = "positional"
        if hits_bound is None:
            score = NEG_INF
        else:
            score = hits_bound + (POSITIONAL_PROFIT_WEIGHT * capped_bound if capped_bound is not None else 0.0)
    elif payout:
        kind = "capped_profit"
        score = capped_bound if capped_bound is not None else NEG_INF
    else:
        kind = "hits"
        score = hits_bound if hits_bound is not None else NEG_INF

    return {
        "score": float(score),
        "kind": kind,
        "days": max(len(profit_days), len(hits_days)),
        "bets": bets,
        "profit_per_bet": (total / bets) if bets else None,
        "capped_profit_bound": capped_bound,
        "hits_mean": float(np.mean(hits_days)) if hits_days else None,
        "hits_bound": hits_bound,
        "lucky_strikes": strikes,
    }


def score_rows(rows, model_name, payout=False, positional=False, game=None,
               penalty=CONFIDENCE_PENALTY, cap=LUCKY_STRIKE_CAP):
    """
    Score one model over the Backtester's per-day rows - Backtester.backtest()'s
    return value, or the rows finished so far when a trial is being considered
    for pruning (same function, so the partial and the final score agree).

    `game` names the game for the positional branch: for every positional
    game but Joker+ the per-day hits are digits in the right slot, counted
    from the row's drawn-order ticket and draw (slot_hits) with "_hits" only
    as a fallback for rows without them; Joker+ reads "_hits" (L + R runs).

    Returns a dict:
      score                the objective (-inf when nothing was scored)
      kind                 "capped_profit" | "positional" | "hits"
      days                 backtest days with a scored result for this model
      bets                 bets placed (payout games), else 0
      profit_per_bet       raw, uncapped mean profit per bet - the old objective
      capped_profit_bound  lower bound of the per-day capped profit per bet
      hits_mean, hits_bound
      lucky_strikes        bets at or above the cap
    """
    bets_by_day = []
    hits_by_day = []
    by_slot = positional and counts_slots_itself(game)
    for row in rows or []:
        profits = bet_profits(row, model_name)
        if profits:
            bets_by_day.append(profits)
        hits = slot_hits(row, model_name) if by_slot else None
        if hits is None:
            hits = row.get(f"{model_name}_hits")
        if _number(hits):
            hits_by_day.append(float(hits))
    return score_bets_by_day(bets_by_day, hits_by_day, payout=payout, positional=positional,
                             penalty=penalty, cap=cap)


def json_safe(value):
    """A non-finite float becomes None: bestParams_<game>.json is read by Node as well, which rejects -Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def attrs_for_trial(tuning):
    """The diagnostics as JSON-safe values for trial.set_user_attr / bestParams."""
    if not tuning:
        return None
    return {key: json_safe(value) for key, value in tuning.items()}


def describe(tuning):
    """One log line: the score, what kind it is, and the raw numbers behind it."""
    if not tuning:
        return "no scored days"
    # Accepts the JSON-safe form attrs_for_trial stores on a trial as well,
    # where a non-finite score has become None.
    score = tuning.get("score")
    text = f"score {score:+.4f}" if _number(score) else "score -inf"
    kind = {"capped_profit": "capped profit/bet bound", "positional": "slot-hit bound",
            "hits": "hits bound"}.get(tuning.get("kind"), str(tuning.get("kind")))
    parts = [f"{text} ({kind}, {tuning.get('days', 0)} days)"]
    if tuning.get("bets"):
        raw = tuning.get("profit_per_bet")
        raw_text = f"{raw:+.2f}" if _number(raw) else "n/a"
        parts.append(f"raw {raw_text}/bet over {tuning['bets']} bets, {tuning.get('lucky_strikes', 0)} lucky strike(s)")
    if _number(tuning.get("hits_mean")):
        parts.append(f"hits {tuning['hits_mean']:.2f}/day")
    return " | ".join(parts)


if __name__ == "__main__":
    # Synthetic rows, no models: what the September 2026 hijacks looked like
    # and what the objective must do with them.
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    def keno_rows(day_values):
        # two subset bets per day (sizes 5 and 6), like the served keno trial
        return [{"index": i, "m_subset_5_profit": a, "m_subset_6_profit": b}
                for i, (a, b) in enumerate(day_values)]

    # Trial 12 of keno_LightGBM: 60 losing bets, one 6/6 (+199) with its
    # nested 5/5 (+149) on day 23, raw 5.0 per bet.
    jackpot = keno_rows([(-1, -1)] * 22 + [(149, 199)] + [(-1, -1)] * 8)
    # A modest but consistent trial: a 3-of-5 (+1) or 4-of-6 (+3) every few days.
    steady = keno_rows([(1, -1) if i % 4 == 0 else (-1, 3) if i % 7 == 0 else (-1, -1) for i in range(31)])
    s_jackpot = score_rows(jackpot, "m", payout=True)
    s_steady = score_rows(steady, "m", payout=True)
    check(abs(s_jackpot["profit_per_bet"] - (348 - 60) / 62) < 1e-9, f"raw keno profit/bet should be {(348 - 60) / 62:.3f}, got {s_jackpot['profit_per_bet']}")
    check(s_jackpot["lucky_strikes"] == 2, f"two strikes expected, got {s_jackpot['lucky_strikes']}")
    check(s_jackpot["score"] < 0, f"one jackpot day must not make the score positive: {s_jackpot['score']}")
    check(s_steady["score"] > s_jackpot["score"],
          f"steady small wins ({s_steady['score']:.3f}) must outrank the jackpot window ({s_jackpot['score']:.3f})")
    check(s_jackpot["kind"] == "capped_profit" and s_jackpot["days"] == 31 and s_jackpot["bets"] == 62, "keno bookkeeping")

    # pick3: one straight (+756) among 30 losses versus three pairs (+46 each).
    # Rows carry the drawn-order ticket and draw, as the Backtester writes them;
    # "m_hits" is the Backtester's SET count for pick3 and must not be what is scored.
    def pick3_rows(profits, tickets, draw=(1, 2, 3)):
        return [{"index": i, "m_profit": p, "m_prediction": list(t), "actual_ordered": list(draw),
                 "m_hits": len(set(t) & set(draw))} for i, (p, t) in enumerate(zip(profits, tickets))]
    miss, exact, back_pair, box = (7, 8, 9), (1, 2, 3), (7, 2, 3), (3, 2, 1)
    straight = pick3_rows([-4] * 19 + [756] + [-4] * 11, [miss] * 19 + [exact] + [miss] * 11)
    pairs = pick3_rows([46 if i in (5, 17, 27) else -4 for i in range(31)],
                       [back_pair if i in (5, 17, 27) else miss for i in range(31)])
    s_straight = score_rows(straight, "m", payout=True, positional=True, game="pick3")
    s_pairs = score_rows(pairs, "m", payout=True, positional=True, game="pick3")
    check(s_pairs["hits_mean"] == 6 / 31 and s_straight["hits_mean"] == 3 / 31, "pick3 hits are slot hits, not the set count")
    # right digits in the wrong slots (a box, unpaid on the straight/pair bets) must not score as a straight
    boxes = pick3_rows([76 if i % 2 == 0 else -4 for i in range(31)], [box if i % 2 == 0 else miss for i in range(31)])
    s_boxes = score_rows(boxes, "m", payout=True, positional=True, game="pick3")
    check(s_boxes["hits_mean"] == 16 / 31, f"a box counts its one right-slot digit, not three: {s_boxes['hits_mean']}")
    check(s_pairs["score"] > s_boxes["score"] or s_pairs["hits_mean"] < s_boxes["hits_mean"], "pairs vs boxes ranked by slot hits")
    # Joker+ keeps the Backtester's L + R "_hits"; a row without the ordered keys falls back to "_hits"
    check(score_rows([{"index": 0, "m_hits": 4, "m_prediction": [1, 2, 3, 4, 5, 6, 0], "actual_ordered": [1, 2, 9, 9, 5, 6, 0]}],
                     "m", payout=True, positional=True, game="jokerplus")["hits_mean"] == 4.0, "jokerplus reads _hits")
    check(score_rows([{"index": 0, "m_hits": 2}], "m", positional=True, game="pick3")["hits_mean"] == 2.0, "fallback to _hits")
    check(abs(s_straight["profit_per_bet"] - (756 - 30 * 4) / 31) < 1e-9, "raw pick3 profit/bet")
    check(s_pairs["score"] > s_straight["score"],
          f"three pairs ({s_pairs['score']:.4f}) must outrank one straight ({s_straight['score']:.4f})")
    check(s_straight["kind"] == "positional", "pick3 is scored positionally")
    # the profit term is a tie-breaker only: equal hits, different (capped) profit
    one_slot = (1, 9, 9)
    same_hits_a = pick3_rows([-4] * 31, [one_slot] * 31)
    same_hits_b = pick3_rows([-3] * 31, [one_slot] * 31)
    a, b = (score_rows(same_hits_a, "m", True, True, "pick3")["score"],
            score_rows(same_hits_b, "m", True, True, "pick3")["score"])
    check(b > a and (b - a) < 0.01, f"profit must only break ties between equal hit rates: {a} vs {b}")

    # hits-only games: the bound, not the mean, and the partial rows agree
    lotto = [{"index": i, "m_hits": h} for i, h in enumerate([1, 0, 2, 1, 0, 1, 3, 0, 1, 1])]
    s_lotto = score_rows(lotto, "m", payout=False)
    check(s_lotto["kind"] == "hits" and s_lotto["hits_mean"] == 1.0 and s_lotto["score"] < 1.0, "lotto hits bound below the mean")
    check(score_rows(lotto[:5], "m")["days"] == 5, "partial rows score the finished days only")

    # nothing scored, errors, wrong model name
    check(score_rows([], "m", payout=True)["score"] == NEG_INF, "no rows -> -inf")
    check(score_rows([{"index": 0, "m_error": "boom"}], "m", payout=True)["score"] == NEG_INF, "error rows are not results")
    check(score_rows(jackpot, "other", payout=True)["score"] == NEG_INF, "another model's rows do not count")
    check(score_rows(jackpot, "m", payout=True, penalty=0)["capped_profit_bound"] == np.mean(
        [(min(a, 20) + min(b, 20)) / 2 for a, b in [(-1, -1)] * 22 + [(149, 199)] + [(-1, -1)] * 8]), "penalty 0 is the capped mean")

    # bound arithmetic
    check(lower_bound([]) is None and lower_bound([2.0]) == 2.0, "empty/single bounds")
    check(abs(lower_bound([1, 2, 3, 4]) - (2.5 - np.std([1, 2, 3, 4], ddof=1) / 2)) < 1e-12, "bound formula")

    # diagnostics survive JSON and read as a line
    import json
    json.dumps(attrs_for_trial(score_rows([], "m", payout=True)))
    check("2 lucky strike" in describe(s_jackpot) and "raw +4.65/bet" in describe(s_jackpot), describe(s_jackpot))
    # the stored (JSON-safe) form of an all-error trial must describe, not raise
    stored = attrs_for_trial(score_rows([{"index": 0, "m_error": "boom"}], "m", payout=True))
    check(stored["score"] is None and describe(stored).startswith("score -inf"), f"describe on the stored form: {describe(stored)}")
    check(describe(json.loads(json.dumps(stored))) == describe(stored), "describe after a JSON round trip")

    for message in failures:
        print("FAIL:", message)
    print(f"TuningScore self-check: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    print("  keno   jackpot window:", describe(s_jackpot))
    print("  keno   steady window: ", describe(s_steady))
    print("  pick3  one straight:  ", describe(s_straight))
    print("  pick3  three pairs:   ", describe(s_pairs))
    raise SystemExit(1 if failures else 0)
