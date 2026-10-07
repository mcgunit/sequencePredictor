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
      Since 7 Oct 2026 each keno bet's capped profit has the fair capped
      value of its ticket size subtracted (fair_capped_ev): the cap leaves a
      fair 6-ticket at -0.52 and a fair 10-ticket at -0.70 of its stake,
      because a 6-ticket's prizes mostly sit under the 20 EUR cap and a
      10-ticket's mostly above it. A fair ticket now scores 0 in mean at
      every size, so stored keno scores read alike across sizes and across
      changes of the served sizes; within one run every candidate bets the
      same sizes, so the baseline is a constant there and changes no
      decision. The bound still differs between sizes under the null (its
      penalty term scales with the capped payoff's spread: about -0.17 for
      a fair 6-ticket over 90 days, -0.10 for a 10-ticket), and the recorded
      raw profit per bet stays what it was.
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
  the market rows (GARCH, Regime HMM on crypto and shares - proper=True)
      the lower confidence bound of the per-day mean log-score: the log of
      the probability the row gave the bin that then happened, averaged
      over the instruments, floored at LOG_SCORE_FLOOR. The proper score of
      README item 4 - a row that puts its mass on the right bins scores
      high, a row that bets one bin per slot is punished when it misses,
      and hit rates (which a near-one-hot forecast games) are diagnostics
      only. The uniform forecast scores log(1/K) = -2.303 at ten bins.

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
# A bin probability is scored no lower than this (a proper score needs a
# finite log); the same value as MarketModels.LOG_FLOOR, which the market
# rows apply when they report their probabilities - the callers pass that
# constant explicitly so the two cannot drift apart.
LOG_SCORE_FLOOR = 1e-4

NEG_INF = float("-inf")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


_FAIR_CAPPED = {}


def fair_capped_ev(size, cap=LUCKY_STRIKE_CAP, drawn=20, pool=70):
    """
    What a `size`-number keno ticket is worth per 1 EUR under a fair draw,
    with every bet capped the way the objective caps it - the baseline a
    keno bet's capped profit is measured against (7 Oct 2026), so that
    ticket sizes compare. The cap is not size-neutral: on a fair 20-of-70
    draw a 6-ticket keeps -0.52 of its stake under it and a 10-ticket -0.70,
    while the raw expectation is -0.46 to -0.48 for every size; without the
    baseline a tuner offered several sizes would favour the 6-ticket for the
    cap, not for skill. Hypergeometric matches against
    Helpers.PAYOUT_TABLE_KENO; None for a size the table does not pay.
    """
    if size is None or not 5 <= int(size) <= 10:       # the playable sizes, as keno_ticket_profit tracks them
        return None
    key = (int(size), float(cap), drawn, pool)
    if key in _FAIR_CAPPED:
        return _FAIR_CAPPED[key]
    try:
        from src.Helpers import Helpers      # lazy: this module stays light for the tuners' children
    except ImportError:                      # imported from inside src/ (the Backtester's style)
        from Helpers import Helpers
    table = Helpers.PAYOUT_TABLE_KENO.get(int(size))
    if not table:
        _FAIR_CAPPED[key] = None
        return None
    stake = -Helpers.PAYOUT_TABLE_KENO["lost"]
    total = math.comb(pool, int(size))
    value = 0.0
    for matches in range(0, int(size) + 1):
        probability = math.comb(drawn, matches) * math.comb(pool - drawn, int(size) - matches) / total
        payout = table.get(matches)
        net = (payout - stake) if payout is not None else Helpers.PAYOUT_TABLE_KENO["lost"]
        value += probability * min(net, cap)
    _FAIR_CAPPED[key] = value
    return value


def bet_profits(row, model_name):
    """
    Every bet this model placed on one backtest day: the main ticket's profit
    (pick3, Joker+ - "<model>_profit") and each Keno subset's
    ("<model>_subset_<size>_profit"), as the Backtester wrote them.
    Each bet is a (profit, size) pair: the size is the subset's number count
    (the fair baseline of fair_capped_ev applies to it), None for a main
    ticket.
    """
    profits = []
    main = row.get(f"{model_name}_profit")
    if _number(main):
        profits.append((float(main), None))
    prefix = f"{model_name}_subset_"
    for key, value in row.items():
        if key.startswith(prefix) and key.endswith("_profit") and _number(value):
            size = key[len(prefix):-len("_profit")]
            profits.append((float(value), int(size) if size.isdigit() else None))
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


def log_score(row, model_name, floor=LOG_SCORE_FLOOR):
    """
    The proper score of one day: the mean over the slots of the log of the
    probability the model gave the bin that then happened, from the per-slot
    scores the Backtester stores with collect_scores=True
    ("<model>_position_scores": one {bin: score} per slot, any scale - each
    slot is normalised to sum to one here, as Helpers.normalize_position_
    scores does for the meta-learners) and the draw in drawn order. A
    probability under `floor` scores as `floor`. Returns (score, k) with k
    the number of bins of the widest slot (so the caller can quote the
    uniform forecast's log(1/k)), or (None, None) when the row does not
    carry both.
    """
    slots = row.get(f"{model_name}_position_scores")
    actual = row.get("actual_ordered") or row.get("actual")
    if not isinstance(slots, (list, tuple)) or not isinstance(actual, (list, tuple)) or not slots or not actual:
        return None, None
    if len(slots) != len(actual):
        return None, None
    logs, k = [], 0
    for slot, a in zip(slots, actual):
        if not isinstance(slot, dict) or not slot:
            continue
        try:
            a = int(a)
            total = float(sum(float(v) for v in slot.values()))
            p = float(slot.get(a, slot.get(str(a), 0.0)))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(total) or total <= 0:
            continue
        logs.append(math.log(max(p / total, floor)))
        k = max(k, len(slot))
    if not logs:
        return None, None
    return float(np.mean(logs)), k


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
                      penalty=CONFIDENCE_PENALTY, cap=LUCKY_STRIKE_CAP, logs_by_day=None, proper=False, bins=None):
    """
    The objective from per-day material: `bets_by_day` is a list with, per
    day, the list of that day's bet profits (net EUR); `hits_by_day` the
    per-day hit count where one exists; `logs_by_day` the per-day mean
    log-score where one exists, which with proper=True IS the objective
    (its lower bound; `bins` names K for the uniform reference). Returns the
    dict documented on score_rows(). Days without bets (and without hits or
    log-scores) are not days.
    """
    profit_days = []
    bets = 0
    total = 0.0
    strikes = 0
    fair_total = 0.0
    fair_bets = 0
    for profits in bets_by_day or []:
        # a bet is a profit or a (profit, size) pair - the size names the keno
        # ticket whose fair capped value (fair_capped_ev) is subtracted, so a
        # day's number is "capped profit above a fair ticket" and sizes compare
        today = []
        for item in profits or []:
            profit, size = (item if isinstance(item, (tuple, list)) and len(item) == 2 else (item, None))
            if _number(profit):
                today.append((float(profit), size))
        if not today:
            continue
        bets += len(today)
        total += sum(p for p, _ in today)
        strikes += sum(1 for p, _ in today if p >= cap)
        adjusted = []
        for p, size in today:
            fair = fair_capped_ev(size, cap) if size is not None else None
            if fair is not None:
                fair_total += fair
                fair_bets += 1
            adjusted.append(min(p, cap) - (fair or 0.0))
        profit_days.append(sum(adjusted) / len(adjusted))
    hits_days = [float(h) for h in (hits_by_day or []) if _number(h)]
    log_days = [float(v) for v in (logs_by_day or []) if _number(v)]

    capped_bound = lower_bound(profit_days, penalty)
    hits_bound = lower_bound(hits_days, penalty)
    log_bound = lower_bound(log_days, penalty)

    if proper:
        kind = "log_score"
        score = log_bound if log_bound is not None else NEG_INF
    elif positional:
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

    out = {
        "score": float(score),
        "kind": kind,
        "days": max(len(profit_days), len(hits_days), len(log_days)),
        "bets": bets,
        "profit_per_bet": (total / bets) if bets else None,
        "capped_profit_bound": capped_bound,
        # the mean fair capped value of the keno bets that were measured against one (None elsewhere)
        "fair_capped_per_bet": (fair_total / fair_bets) if fair_bets else None,
        "hits_mean": float(np.mean(hits_days)) if hits_days else None,
        "hits_bound": hits_bound,
        "lucky_strikes": strikes,
    }
    if proper or log_days:
        out["log_score_mean"] = float(np.mean(log_days)) if log_days else None
        out["log_score_bound"] = log_bound
        out["uniform_log_score"] = math.log(1.0 / bins) if bins else None
    return out


def score_rows(rows, model_name, payout=False, positional=False, game=None,
               penalty=CONFIDENCE_PENALTY, cap=LUCKY_STRIKE_CAP, proper=False, floor=LOG_SCORE_FLOOR):
    """
    Score one model over the Backtester's per-day rows - Backtester.backtest()'s
    return value, or the rows finished so far when a trial is being considered
    for pruning (same function, so the partial and the final score agree).

    `game` names the game for the positional branch: for every positional
    game but Joker+ the per-day hits are digits in the right slot, counted
    from the row's drawn-order ticket and draw (slot_hits) with "_hits" only
    as a fallback for rows without them; Joker+ reads "_hits" (L + R runs).

    `proper` (the market rows) scores the per-day log-score of the stored
    slot probabilities instead (log_score(); the rows must have been
    backtested with collect_scores=True), slot hits staying as diagnostics.

    Returns a dict:
      score                the objective (-inf when nothing was scored)
      kind                 "capped_profit" | "positional" | "hits" | "log_score"
      days                 backtest days with a scored result for this model
      bets                 bets placed (payout games), else 0
      profit_per_bet       raw, uncapped mean profit per bet - the old objective
      capped_profit_bound  lower bound of the per-day capped profit per bet
      hits_mean, hits_bound
      lucky_strikes        bets at or above the cap
      log_score_mean, log_score_bound, uniform_log_score   (proper rows only)
    """
    bets_by_day = []
    hits_by_day = []
    logs_by_day = []
    bins = 0
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
        if proper:
            value, k = log_score(row, model_name, floor)
            if value is not None:
                logs_by_day.append(value)
                bins = max(bins, k or 0)
    return score_bets_by_day(bets_by_day, hits_by_day, payout=payout, positional=positional,
                             penalty=penalty, cap=cap, logs_by_day=logs_by_day, proper=proper, bins=bins or None)


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
            "hits": "hits bound", "log_score": "log-score bound"}.get(tuning.get("kind"), str(tuning.get("kind")))
    if _number(tuning.get("fair_capped_per_bet")):
        kind = f"{kind} above a fair ticket ({tuning['fair_capped_per_bet']:+.2f}/bet)"
    parts = [f"{text} ({kind}, {tuning.get('days', 0)} days)"]
    if _number(tuning.get("log_score_mean")):
        uniform = tuning.get("uniform_log_score")
        parts.append(f"log-score {tuning['log_score_mean']:+.3f}/day"
                     + (f" (uniform {uniform:+.3f})" if _number(uniform) else ""))
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
    # the fair capped baseline: a fair 6-ticket keeps more of its stake under the cap than a fair 10-ticket
    check(abs(fair_capped_ev(6) - (-0.524)) < 0.01 and abs(fair_capped_ev(10) - (-0.703)) < 0.01 and fair_capped_ev(4) is None,
          f"fair capped values: {fair_capped_ev(6)}, {fair_capped_ev(10)}, {fair_capped_ev(4)}")
    # a bet exactly at its size's fair capped value scores 0, whatever the size
    even = score_bets_by_day([[(fair_capped_ev(6), 6), (fair_capped_ev(10), 10)]] * 10)
    check(abs(even["score"]) < 1e-9 and abs(even["fair_capped_per_bet"] - (fair_capped_ev(6) + fair_capped_ev(10)) / 2) < 1e-9,
          f"fair bets must score 0 at every size: {even['score']}")
    check(s_jackpot["fair_capped_per_bet"] is not None and s_jackpot["profit_per_bet"] == (348 - 60) / 62, "raw profit per bet is not adjusted")
    # plain floats still work (pick3's main ticket carries no size)
    check(score_bets_by_day([[-1.0, 3.0]])["bets"] == 2, "float bets")

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

    # the market rows: the log-score of the stored slot probabilities, floored,
    # with slot hits as diagnostics only
    def market_rows(slot_probs, draws):
        return [{"index": i, "m_position_scores": [dict(s) for s in probs], "actual_ordered": list(draw),
                 "m_prediction": [max(s, key=s.get) for s in probs]} for i, (probs, draw) in enumerate(zip(slot_probs, draws))]
    uniform = [{b: 0.1 for b in range(10)}] * 2
    sharp = [{b: (0.91 if b == 3 else 0.01) for b in range(10)}] * 2
    hedged = [{b: (0.29 if b == 3 else 0.31 if b == 4 else 0.05) for b in range(10)}] * 2   # argmax 4: fewer hits than sharp
    s_uniform = score_rows(market_rows([uniform] * 10, [(3, 3)] * 10), "m", positional=True, game="crypto", proper=True)
    check(s_uniform["kind"] == "log_score" and s_uniform["days"] == 10, "market rows are scored by log-score")
    check(abs(s_uniform["log_score_mean"] - math.log(0.1)) < 1e-12 and abs(s_uniform["score"] - math.log(0.1)) < 1e-12,
          f"the uniform forecast scores log(1/10): {s_uniform['log_score_mean']}")
    check(abs(s_uniform["uniform_log_score"] - math.log(0.1)) < 1e-12, "K is read from the slots")
    # a sharp row that is right every day beats uniform; when it is wrong on four days, the hedged row beats it
    s_sharp = score_rows(market_rows([sharp] * 10, [(3, 3)] * 10), "m", positional=True, game="crypto", proper=True)
    check(s_sharp["score"] > s_uniform["score"], "a right sharp forecast beats uniform")
    check(s_sharp["hits_mean"] == 2.0, "slot hits travel as a diagnostic")
    s_sharp_wrong = score_rows(market_rows([sharp] * 10, [(3, 3)] * 6 + [(4, 4)] * 4), "m", positional=True, game="crypto", proper=True)
    s_hedged = score_rows(market_rows([hedged] * 10, [(3, 3)] * 6 + [(4, 4)] * 4), "m", positional=True, game="crypto", proper=True)
    check(s_hedged["score"] > s_sharp_wrong["score"] and s_sharp_wrong["hits_mean"] > s_hedged["hits_mean"],
          f"hit rate must not decide: hedged {s_hedged['score']:.3f} vs sharp-but-wrong {s_sharp_wrong['score']:.3f}")
    # the floor: a zero probability on the realised bin scores log(floor), not -inf
    one_hot = [{b: (1.0 if b == 3 else 0.0) for b in range(10)}] * 2
    s_one_hot = score_rows(market_rows([one_hot] * 2, [(3, 3), (5, 5)]), "m", positional=True, game="crypto", proper=True)
    check(abs(s_one_hot["log_score_mean"] - (0.0 + math.log(LOG_SCORE_FLOOR)) / 2) < 1e-12, f"floor applied: {s_one_hot['log_score_mean']}")
    # raw counts (any scale) are normalised per slot; string keys are accepted; a row without slots has no score
    counts = [{str(b): (300 if b == 3 else 100) for b in range(10)}] * 2
    s_counts = score_rows(market_rows([counts], [(3, 3)]), "m", positional=True, game="crypto", proper=True)
    check(abs(s_counts["log_score_mean"] - math.log(300 / 1200)) < 1e-12, f"counts normalised per slot: {s_counts['log_score_mean']}")
    check(score_rows([{"index": 0, "m_prediction": [3, 3], "actual_ordered": [3, 3]}], "m", positional=True, proper=True)["score"] == NEG_INF,
          "no slot probabilities -> nothing scored")
    check(log_score({"m_position_scores": [{0: 1.0}], "actual_ordered": [0, 1]}, "m") == (None, None), "slot count must match the draw")
    # the bound, not the mean; the diagnostics read as a line and survive JSON
    check(s_sharp_wrong["score"] < s_sharp_wrong["log_score_mean"], "log-score bound below the mean")
    check("log-score bound" in describe(s_uniform) and "(uniform -2.303)" in describe(s_uniform), describe(s_uniform))
    import json as _json
    _json.dumps(attrs_for_trial(s_one_hot))

    # nothing scored, errors, wrong model name
    check(score_rows([], "m", payout=True)["score"] == NEG_INF, "no rows -> -inf")
    check(score_rows([{"index": 0, "m_error": "boom"}], "m", payout=True)["score"] == NEG_INF, "error rows are not results")
    check(score_rows(jackpot, "other", payout=True)["score"] == NEG_INF, "another model's rows do not count")
    check(abs(score_rows(jackpot, "m", payout=True, penalty=0)["capped_profit_bound"] - np.mean(
        [(min(a, 20) - fair_capped_ev(5) + min(b, 20) - fair_capped_ev(6)) / 2
         for a, b in [(-1, -1)] * 22 + [(149, 199)] + [(-1, -1)] * 8])) < 1e-9, "penalty 0 is the capped mean above the fair baseline")

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
