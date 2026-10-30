"""
Q0, the per-row chance test (27 Oct 2026, B3 of the 26 Oct weekly reading; README "Controls"): would a row's hits on
its live draws be this good by chance, and has a row that looked good gone on to show it on new draws?

Until 27 Oct 2026 the Q0 card read each ranked row's hits per draw over ALL its stored draws against one band per game
- the best of 8 untuned rows on 120 fair draws plus 2 sd across 3 seeds (NullControls.py) - and called the rows over it
"above the band". With the rows' own tickets and fair draws that label showed at least one row above in 98% of lotto
weeks (26-57% in the other games): the band knows neither a row's record length nor its own fair level, which depends
on its ticket's shape (lotto's 7-number tickets of Nov 2025 - May 2026 are worth 0.933 hits against 0.800, pick3's by
its distinct digits, keno's DL rows hold numbers 71-80 that cannot be drawn) and, in a market, on the bins a stretch
favours. And week after week it re-read the same record plus a draw or two ("3rd week above"). The band stays as
context in the Controls drawer only; nothing is labelled, greyed or counted on it any more.

THE TEST, per row, on its LIVE draws - the live record of Helpers.generate_model_performance_report: a lottery ticket
saved before the draw's sales close (a ticket rewritten after the draw as it stood at the close; src/Liveness.py), a
market day whose ticket was out before the close - on the card's own hit scale (main hits per draw; Joker+ L + R; a
market's slot hits per day or week):
  lottery  the EXACT fair-draw null. Given each draw's ticket, a fair draw's hit count has a known distribution: the
           hypergeometric for the set games and keno's 20-number ticket (its distinct in-range numbers against the
           drawn count; the ranges are HyperoptStatistics.GAME_CONFIG's, asserted by the self-check), pick3's by the
           ticket's distinct digits, Joker+'s run rule over its positions. The record's total is their convolution:
           p = P(total >= the row's hits). Exact for any record length and ticket shape. A permutation null would
           condition on the stretch's own draws and could never see the one edge a lottery could have (numbers that
           come up more often), so it was not used (the design's measurements, 26 Oct 2026).
  markets  a BLOCK permutation null - there is no fair draw: the bins are expanding-window deciles and a stretch's bin
           mix is not uniform. The row's own tickets are shuffled among its own live days within blocks of BLOCK
           consecutive live days (live weeks for the week games), the outcomes fixed, PERMUTATIONS times, seeded by
           the record (crc32 of 'game|row|n|last date', so p does not move between two runs on the same record):
           p = (1 + #{shuffled >= observed}) / (B + 1). A row that calls the same bins every day scores the same
           shuffled (p = 1), and the bins a few days favour stay with those days: a stretch's bin mix and the drift
           of its volatility regime do not count as skill. What it does not remove: a row that TIMES the regime
           within a week can still pass - real, but it carries no direction; the market rows' proper score against
           GARCH (MarketRows.py) stays the reference for that. A free shuffle over the whole stretch counted regime
           drift (the critic, 26 Oct: crypto LSTM Base p 0.027 free, 0.053 in blocks of 7).
The family: the rows with at least minDrawsForRanking live draws (week games: 10 live weeks - no family before about
13 Dec 2026). Holm over the family at ALPHA ("holds"); "under" = raw p < ALPHA, always next to what chance puts there
(the sum of each row's real discrete level P(p < ALPHA) for the exact test, m x ALPHA for the permutation). Under fair
draws Holm gives a false "holds" in 3-4% of weeks per card, so with eight cards about one week in four shows one
somewhere: "holds" and "under" are this week's figures, never a colour, a chip or a streak (admin.js). The payout
games (keno, pick3, Joker+) are tested on hits too, information only.

FRESH EVIDENCE, the one thing that counts (a row "confirmed"). From the first Sunday F on which a row's live draws
dated before F (at least minDrawsForRanking of them) read p < ALPHA, the row is "followed": its live draws dated on or
after F are kept apart, and once there are verdictLength = clamp(the draws that flagged it, minDrawsForRanking,
MAX_VERDICT) of them, exactly those are tested alone - "confirmed" at p < VERDICT_ALPHA, "not confirmed" otherwise.
Before that it is "running" (the figures so far, no p: a p read week after week until it dips is no verdict), and
"stopped" once no new live draw has come for STOPPED_DAYS. Measured for a worthless lotto row (sim_fresh.py): reading
the fresh p every week at 0.05 would confirm 0.91 false rows per 27-row card-year; the fixed length at 0.01 about 0.04.
A +0.4 hits/draw row is confirmed within a year 33-65% of the time: the card will mostly read "0 confirmed", which for
a lottery is the honest answer.

PERSISTED, so F and a verdict never move: data/database/rowChance.json (FILE_NAME), written by the report with sorted
keys and only when it changes, so a rerun without a new draw leaves it - and git - as it was. Per game: checkedThrough
(the newest Sunday tested), freeze (the since.json date it was kept under), testFrom (after a reset, the first Sunday a
flag may take) and per followed row {since (F), flagDraws, flagP, flagMean, flagNull, verdictLength, state ('running'
or the verdict), verdictAt (the date of the draw the verdict was taken on), decidedOn (the report's date that took it),
p, mean, null (the verdict's figures)}. First write wins: the first run with no entry for a game replays its Sundays
ONCE (every row ever flagged on a Sunday on or after the since.json freeze declared then - the first replay applies the
freeze as the reset rule would have, owner 30 Oct 2026, so no flag from before it is kept - an entry the replay of
28 Oct kept under the freeze without it (no testFrom) is reset once by the next run; daily runs end with the
replay's flags and verdicts, decidedOn apart, as long as every draw dated before a Sunday is in by that Sunday's run:
27 Oct 2026 review, on production's record); every later run tests only the Sundays after checkedThrough, for the rows not followed yet, and
reads a decided verdict back - never recomputed, whatever later happens to the record (a liveness rescan, a change of
minDrawsForRanking, a rewritten day file). A renamed row is a new row. The one reset: a since.json freeze declared
after the game's entry was kept (freeze moves forward) drops the entries of the rows flagged before the new freeze
date - the design they were flagged under is gone - and those rows are tested again from the first Sunday on or after
it; no row is flagged on a Sunday before it from then on (a freeze dated ahead holds every flag until its day). A
lottery game judged without git ('stamp only') reads its entries back and stores nothing that day; one whose liveness
could not be judged gets no test at all.

Pure: functions of the day table, the models and the stored entry; the report (src/Helpers.py) reads and writes the
file. The prototype and the calibration scripts (sim_fair.py, sim_fresh.py, placebo.py) are in the B3 design's
scratch folder, not in CI.

    python3 -m src.RowChance        # self-check (no production file is read or written)
"""

import ast
import copy
import json
import math
import os
import zlib
from datetime import date, datetime, timedelta

import numpy as np

ALPHA = 0.05            # the per-row screen, Holm's level and the flag
VERDICT_ALPHA = 0.01    # the fresh verdict (owner's decision, 26 Oct 2026)
PERMUTATIONS = 20000
BLOCK = 7               # market shuffles stay within this many consecutive live days (weeks for the week games)
MAX_VERDICT = 52        # the longest verdict: EuroMillions' 92-draw flag would otherwise take until Sep 2027
STOPPED_DAYS = 28       # a followed row with no new live draw for four weeks reads "stopped"
FILE_NAME = "rowChance.json"
VERSION = 1
RULE = ("per row on its live draws: lottery against the exact fair-draw null (hypergeometric, pick3 distinct digits, "
        f"Joker+ runs), markets against its own tickets shuffled within blocks of {BLOCK} live days or weeks "
        f"({PERMUTATIONS} permutations); Holm at {ALPHA} over the rows with at least minDrawsForRanking live draws; "
        f"followed from the first Sunday its record read p < {ALPHA}, judged once on as many new live draws as flagged "
        f"it (at most {MAX_VERDICT}): confirmed at p < {VERDICT_ALPHA}")

# game: (kind, lowest number, highest number, numbers drawn) - the self-check holds these to HyperoptStatistics.GAME_CONFIG
GAMES = {
    "lotto": ("set", 1, 45, 6),
    "euromillions": ("set", 1, 50, 5),
    "eurodreams": ("set", 1, 40, 6),
    "vikinglotto": ("set", 1, 48, 6),
    "keno": ("set", 1, 70, 20),
    "pick3": ("pick3", 0, 9, 3),
    "jokerplus": ("joker", 0, 9, 6),
    "crypto": ("market", 0, 9, 5),
    "shares": ("market", 0, 9, 4),
    "cryptoweek": ("market", 0, 9, 5),
    "sharesweek": ("market", 0, 9, 4),
}


def kind_of(game):
    """'set', 'pick3', 'joker' or 'market'; None for a game this test does not know."""
    spec = GAMES.get(game)
    return spec[0] if spec else None


def method_of(game):
    return "permutation" if kind_of(game) == "market" else "exact"


# --- the fair-draw distributions --------------------------------------------------------------------------------------
def hypergeom_pmf(population, drawn, ticket):
    """P(k of a ticket's `ticket` distinct numbers are among `drawn` numbers drawn from `population`), k = 0..min."""
    total = math.comb(population, ticket)
    top = min(ticket, drawn)
    return np.array([math.comb(drawn, k) * math.comb(population - drawn, ticket - k) / total for k in range(top + 1)])


_PICK3 = {}
_JOKER = {}


def joker_hits(ticket, result):
    """Joker+'s hits as Helpers.jokerplus_runs scores them: the leading run plus the trailing run, all positions when
    every one matches (the runs never count a position twice)."""
    n = min(len(ticket), len(result))
    left = 0
    while left < n and int(ticket[left]) == int(result[left]):
        left += 1
    if left == n:
        return n
    right = 0
    while right < n - left and int(ticket[n - 1 - right]) == int(result[n - 1 - right]):
        right += 1
    return left + right


def draw_pmf(game, ticket, drawn):
    """The distribution of a fair draw's hits against this ticket on the card's scale (index = hits). drawn: how many
    main numbers that draw had (its result's length)."""
    kind = kind_of(game)
    if kind == "set":
        _, lo, hi, _ = GAMES[game]
        inside = len({int(v) for v in ticket if lo <= int(v) <= hi})
        return hypergeom_pmf(hi - lo + 1, drawn, inside)
    if kind == "pick3":
        # |set(ticket) & set(result)| with the result's digits drawn independently 0-9: only the ticket's count of
        # distinct digits matters
        distinct = len({int(v) for v in ticket})
        key = (distinct, drawn)
        if key not in _PICK3:
            counts = np.zeros(min(distinct, drawn) + 1)
            mine = set(range(distinct))
            for outcome in range(10 ** drawn):
                digits = {(outcome // 10 ** i) % 10 for i in range(drawn)}
                counts[len(mine & digits)] += 1
            _PICK3[key] = counts / 10 ** drawn
        return _PICK3[key]
    if kind == "joker":
        # each position matches with probability 1/10, independently; the runs as joker_hits counts them
        positions = min(len(ticket), drawn)
        if positions not in _JOKER:
            pmf = np.zeros(positions + 1)
            for mask in range(2 ** positions):
                match = [(mask >> i) & 1 for i in range(positions)]
                chance = math.prod(0.1 if m else 0.9 for m in match)
                pmf[joker_hits([0] * positions, [0 if m else 1 for m in match])] += chance
            _JOKER[positions] = pmf
        return _JOKER[positions]
    raise ValueError(f"{game}: no fair-draw distribution (a market has none)")


class Tail:
    """A running convolution: add a record's draws in date order, read P(total >= its hits) at any point."""

    def __init__(self):
        self.acc = np.array([1.0])
        self.n = 0
        self.total = 0
        self.mean = 0.0
        self.var = 0.0

    def add(self, pmf, hits):
        self.acc = np.convolve(self.acc, pmf)
        self.n += 1
        self.total += int(hits)
        x = np.arange(len(pmf))
        m = float((x * pmf).sum())
        self.mean += m
        self.var += float(((x - m) ** 2 * pmf).sum())

    def p(self):
        return float(min(1.0, max(0.0, self.acc[self.total:].sum()))) if self.total < len(self.acc) else 0.0

    def level(self, alpha=ALPHA):
        """P(p < alpha) under the null: the largest tail under alpha - a discrete test's real level."""
        tails = np.cumsum(self.acc[::-1])[::-1]
        under = tails[tails < alpha]
        return float(under.max()) if len(under) else 0.0


def exact_test(game, recs):
    tail = Tail()
    for r in recs:
        tail.add(draw_pmf(game, r["ticket"], len(r["result"])), r["hits"])
    return {"p": tail.p(), "null": tail.mean / tail.n, "nullSd": math.sqrt(tail.var) / tail.n, "level": tail.level()}


def _seed(game, name, recs):
    return zlib.crc32(f"{game}|{name}|{len(recs)}|{recs[-1]['date'].isoformat()}".encode())


def hits_matrix(recs):
    """M[j, i]: ticket j's slot hits against outcome i (a market's rule: the same bin in the same slot)."""
    width = max(max(len(r["ticket"]) for r in recs), max(len(r["result"]) for r in recs))
    # a missing slot never matches: tickets padded with -1, outcomes with -2
    tickets = np.full((len(recs), width), -1)
    results = np.full((len(recs), width), -2)
    for i, r in enumerate(recs):
        tickets[i, :len(r["ticket"])] = [int(v) for v in r["ticket"]]
        results[i, :len(r["result"])] = [int(v) for v in r["result"]]
    return (tickets[:, None, :] == results[None, :, :]).sum(-1)


def permutation_test(game, name, recs, permutations=PERMUTATIONS, block=BLOCK):
    """The row's own tickets shuffled among its live days within consecutive blocks of `block` (the record in date
    order, the last block shorter), the outcomes fixed; p = (1 + #{S >= observed}) / (B + 1)."""
    n = len(recs)
    M = hits_matrix(recs)
    observed = float(np.trace(M)) / n
    rng = np.random.default_rng(_seed(game, name, recs))
    shuffled = np.zeros(permutations)
    null = 0.0
    for start in range(0, n, block):
        idx = np.arange(start, min(start + block, n))
        sub = M[np.ix_(idx, idx)]
        null += float(sub.sum()) / len(idx)
        if len(idx) == 1:
            shuffled += sub[0, 0]
            continue
        order = np.argsort(rng.random((permutations, len(idx))), axis=1)   # order[k, i]: the ticket that lands on day i
        shuffled += sub[order, np.arange(len(idx))[None, :]].sum(axis=1)
    shuffled /= n
    p = (1 + int((shuffled >= observed - 1e-9).sum())) / (permutations + 1)
    return {"p": p, "null": null / n, "nullSd": float(shuffled.std()), "level": ALPHA}


def test(game, name, recs):
    return permutation_test(game, name, recs) if kind_of(game) == "market" else exact_test(game, recs)


def holm(ps):
    """Holm's step-down adjusted p values, in the order given."""
    m = len(ps)
    out = [None] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda k: ps[k])):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


# --- Sundays ----------------------------------------------------------------------------------------------------------
def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def sunday_on_or_before(day):
    day = _as_date(day)
    return day - timedelta(days=(day.weekday() + 1) % 7)


def sunday_on_or_after(day):
    day = _as_date(day)
    return day + timedelta(days=(6 - day.weekday()) % 7)


def sundays(first, last):
    """The Sundays from `first` (a Sunday) through `last`."""
    day = first
    while day <= last:
        yield day
        day += timedelta(days=7)


def _sig(x):
    return None if x is None else float(f"{x:.4g}")


def _r4(x):
    return None if x is None else round(float(x), 4)


def flag_of(game, name, recs, min_draws, tested):
    """The first Sunday of `tested` (in order) whose record - the live draws dated before it, at least min_draws of
    them - read p < ALPHA, with that record's figures; None when none did."""
    if not tested or len(recs) < min_draws:
        return None
    if kind_of(game) != "market":
        tail, i = Tail(), 0
        for sunday in tested:
            while i < len(recs) and recs[i]["date"] < sunday:
                tail.add(draw_pmf(game, recs[i]["ticket"], len(recs[i]["result"])), recs[i]["hits"])
                i += 1
            if tail.n >= min_draws and tail.p() < ALPHA:
                return sunday, recs[:i], {"p": tail.p(), "null": tail.mean / tail.n}
        return None
    for sunday in tested:
        prefix = [r for r in recs if r["date"] < sunday]
        if len(prefix) >= min_draws:
            result = permutation_test(game, name, prefix)
            if result["p"] < ALPHA:
                return sunday, prefix, result
    return None


# --- one game ---------------------------------------------------------------------------------------------------------
def records_of(days):
    """{name: [{date, ticket, result, hits}]} in date order, from the report's day table: each scored day's
    {"date", "result" (the drawn mains), "live": {name: {"ticket" (as scored, in order), "hits"}}}."""
    out = {}
    for day in sorted(days, key=lambda d: _as_date(d["date"])):
        when = _as_date(day["date"])
        for name, row in (day.get("live") or {}).items():
            out.setdefault(name, []).append({"date": when, "ticket": list(row["ticket"]), "result": list(day["result"]),
                                             "hits": int(row["hits"])})
    return out


def _new_entry(freeze):
    return {"checkedThrough": None, "freeze": freeze, "testFrom": None, "rows": {}}


def game_chance(game, days, models, min_draws, today, stored=None, freeze=None, persist=True):
    """
    One game's row tests. days: the report's scored days (records_of); models: the report's rows [{"name", "draws"}]
    (draws = every stored draw); stored: this game's entry of rowChance.json, or None (then its Sundays are replayed
    once); freeze: the since.json date or None; persist: False when the record must not be learnt from (a lottery game
    judged without git) - no Sunday is tested and no verdict kept, the entry comes back as it was.
    -> ({name: chance block}, the game's summary, the entry to keep).
    """
    today = _as_date(today)
    freeze = _as_date(freeze).isoformat() if freeze else None
    recs = records_of(days)
    method = method_of(game)

    # the cumulative test of every row with a live draw
    per = {}
    for m in models:
        name = m["name"]
        mine = recs.get(name, [])
        block = {"draws": len(mine), "notLive": max(0, int(m.get("draws") or 0) - len(mine))}
        if mine:
            result = test(game, name, mine)
            block.update(mean=_r4(sum(r["hits"] for r in mine) / len(mine)), null=_r4(result["null"]), nullSd=_r4(result["nullSd"]),
                         p=_sig(result["p"]), inFamily=len(mine) >= min_draws, pHolm=None, last=mine[-1]["date"].isoformat(),
                         _level=result["level"], _p=result["p"])
        else:
            block.update(inFamily=False)
        per[name] = block
    family = [n for n, b in per.items() if b.get("inFamily")]
    for name, adjusted in zip(family, holm([per[n]["_p"] for n in family])):
        per[name]["pHolm"], per[name]["_pHolm"] = _sig(adjusted), adjusted

    # the followed rows: read back, flagged on the Sundays not tested yet, decided once
    entry = copy.deepcopy(stored) if stored else _new_entry(freeze)
    entry.setdefault("rows", {})
    newest = sunday_on_or_before(today)
    reset = None
    if persist:
        # a later freeze - or an entry kept under the freeze by the first replay of 28 Oct 2026, before the owner's
        # decision of 30 Oct that the replay applies it (no testFrom yet): reset once, as the reset rule would have
        legacy = bool(stored and freeze and not entry.get("testFrom"))
        if stored and freeze and (legacy or entry.get("freeze") is None or freeze > entry["freeze"]):
            # a later freeze: the rows flagged under the design before it start again from it, and from now on no flag
            # takes a Sunday before it (testFrom; a freeze declared for a date still to come holds every flag till then)
            reset = sunday_on_or_after(freeze)
            entry["rows"] = {n: e for n, e in entry["rows"].items() if e.get("since") and e["since"] >= freeze}
            entry["testFrom"] = reset.isoformat()
        if not stored and freeze:
            # the first replay keeps the freeze as the reset rule would have (owner, 30 Oct 2026): no row is flagged on a
            # Sunday before it, so a design the freeze set aside decides no permanent verdict
            entry["testFrom"] = sunday_on_or_after(freeze).isoformat()
        if freeze and (entry.get("freeze") is None or freeze > entry["freeze"]):
            entry["freeze"] = freeze
        checked = _as_date(entry["checkedThrough"]) if entry.get("checkedThrough") else None
        floor = _as_date(entry["testFrom"]) if entry.get("testFrom") else None
        for m in models:
            name = m["name"]
            mine = recs.get(name, [])
            if name in entry["rows"] or len(mine) < min_draws:
                continue
            first = sunday_on_or_after(mine[0]["date"] + timedelta(days=1))   # the first Sunday after its first draw
            start = max(first, checked + timedelta(days=7)) if checked else first
            if reset is not None and stored and (name in (stored.get("rows") or {})):
                start = max(first, reset)
            if floor is not None:
                start = max(start, floor)
            flag = flag_of(game, name, mine, min_draws, list(sundays(start, newest)))
            if flag:
                sunday, prefix, figures = flag
                flagged = len(prefix)
                entry["rows"][name] = {
                    "since": sunday.isoformat(), "flagDraws": flagged, "flagP": _sig(figures["p"]),
                    "flagMean": _r4(sum(r["hits"] for r in prefix) / flagged), "flagNull": _r4(figures["null"]),
                    "verdictLength": min(max(flagged, min_draws), MAX_VERDICT),
                    "state": "running", "verdictAt": None, "decidedOn": None, "p": None, "mean": None, "null": None,
                }
        if newest >= (checked or newest):
            entry["checkedThrough"] = newest.isoformat()

    for name, kept in entry["rows"].items():
        if name not in per:
            continue
        since = _as_date(kept["since"])
        fresh = [r for r in recs.get(name, []) if r["date"] >= since]
        need = kept["verdictLength"]
        out = {k: kept[k] for k in ("since", "flagDraws", "flagP", "flagMean", "flagNull", "verdictLength")}
        out["draws"] = len(fresh)
        out["last"] = fresh[-1]["date"].isoformat() if fresh else None
        if kept["state"] in ("confirmed", "not confirmed"):
            # decided: read back, never recomputed
            out.update({k: kept[k] for k in ("state", "verdictAt", "decidedOn", "p", "mean", "null")})
        elif len(fresh) >= need:
            judged = fresh[:need]
            result = test(game, name, judged)
            verdict = {"state": "confirmed" if result["p"] < VERDICT_ALPHA else "not confirmed",
                       "verdictAt": judged[-1]["date"].isoformat(), "decidedOn": today.isoformat(), "p": _sig(result["p"]),
                       "mean": _r4(sum(r["hits"] for r in judged) / need), "null": _r4(result["null"])}
            out.update(verdict)
            if persist:
                kept.update(verdict)
        else:
            # the figures so far, no p: read week after week, a running p would be a verdict taken when it dips
            stale = (today - (fresh[-1]["date"] if fresh else since)).days >= STOPPED_DAYS
            out.update(state="stopped" if stale else "running", verdictAt=None, decidedOn=None, p=None)
            if fresh:
                so_far = test(game, name, fresh) if kind_of(game) != "market" else None
                out["mean"] = _r4(sum(r["hits"] for r in fresh) / len(fresh))
                out["null"] = _r4(so_far["null"] if so_far else _market_null(fresh))
            else:
                out["mean"] = out["null"] = None
        per[name]["fresh"] = out

    states = [b["fresh"]["state"] for b in per.values() if b.get("fresh")]
    summary = {
        "method": method, "alpha": ALPHA, "verdictAlpha": VERDICT_ALPHA, "minDraws": min_draws,
        "permutations": PERMUTATIONS if method == "permutation" else None, "block": BLOCK if method == "permutation" else None,
        "family": len(family), "liveMax": max((b["draws"] for b in per.values()), default=0),
        "under": sum(1 for n in family if per[n]["_p"] < ALPHA),
        "expected": round(sum(per[n]["_level"] for n in family), 2),
        "holds": sum(1 for n in family if per[n]["_pHolm"] < ALPHA),
        "followed": states.count("running") + states.count("stopped"), "running": states.count("running"),
        "stopped": states.count("stopped"), "confirmed": states.count("confirmed"), "notConfirmed": states.count("not confirmed"),
        "checkedThrough": entry.get("checkedThrough") if persist else (stored or {}).get("checkedThrough"),
        "stored": "written" if persist else "read only",
    }
    for block in per.values():
        for key in ("_p", "_pHolm", "_level"):
            block.pop(key, None)
    return per, summary, (entry if persist else stored)


def _market_null(recs):
    """A market record's shuffled level without the shuffles: within each block, the mean over every ticket-day pair."""
    M = hits_matrix(recs)
    n = len(recs)
    total = 0.0
    for start in range(0, n, BLOCK):
        idx = np.arange(start, min(start + BLOCK, n))
        total += float(M[np.ix_(idx, idx)].sum()) / len(idx)
    return total / n


def unchecked(game, reason):
    """The summary of a lottery game whose liveness could not be judged: no test at all (stored entries untouched)."""
    return {"method": method_of(game), "alpha": ALPHA, "verdictAlpha": VERDICT_ALPHA, "error": reason, "family": 0}


# --- the file ---------------------------------------------------------------------------------------------------------
def read_store(database_dir):
    """rowChance.json as {"version", "games": {game: entry}}; an empty store when there is none; None when the file is
    there but cannot be read (then nothing is written over it this run)."""
    path = os.path.join(database_dir, FILE_NAME)
    if not os.path.exists(path):
        return {"version": VERSION, "games": {}}
    try:
        with open(path) as handle:
            store = json.load(handle)
        if store.get("version") != VERSION or not isinstance(store.get("games"), dict):
            return None
        return store
    except (OSError, ValueError, AttributeError):
        return None


def store_text(store):
    """One line per followed row, keys sorted: a day that flags a row or decides one changes one line in git."""
    lines = ["{", f'  "version": {VERSION},', f'  "rule": {json.dumps(RULE)},', '  "games": {']
    games = sorted(store.get("games", {}).items())
    for gi, (game, entry) in enumerate(games):
        rows = sorted((entry.get("rows") or {}).items())
        lines.append(f"    {json.dumps(game)}: {{")
        lines.append(f'      "checkedThrough": {json.dumps(entry.get("checkedThrough"))},')
        lines.append(f'      "freeze": {json.dumps(entry.get("freeze"))},')
        lines.append(f'      "testFrom": {json.dumps(entry.get("testFrom"))},')
        if rows:
            lines.append('      "rows": {')
            for ri, (name, row) in enumerate(rows):
                lines.append(f"        {json.dumps(name)}: {json.dumps(row, sort_keys=True)}{',' if ri < len(rows) - 1 else ''}")
            lines.append("      }")
        else:
            lines.append('      "rows": {}')
        lines.append("    }" + ("," if gi < len(games) - 1 else ""))
    lines += ["  }", "}", ""]
    return "\n".join(lines)


def write_store(database_dir, store):
    """Writes rowChance.json when its text changed (through a temporary file and a rename); True when written."""
    path = os.path.join(database_dir, FILE_NAME)
    text = store_text(store)
    try:
        with open(path) as handle:
            if handle.read() == text:
                return False
    except OSError:
        pass
    tmp = os.path.join(database_dir, f".{FILE_NAME}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


# --- self-check -------------------------------------------------------------------------------------------------------
def _self_check():
    import contextlib
    import io
    import random
    import shutil
    import sys
    import tempfile

    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # the ranges and draw sizes are HyperoptStatistics.GAME_CONFIG's, read from its source (importing it would load
    # optuna and the models)
    with open(os.path.join(root, "HyperoptStatistics.py")) as handle:
        tree = ast.parse(handle.read())
    config = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "GAME_CONFIG" for t in node.targets))
    for game, (_, lo, hi, drawn) in GAMES.items():
        c = config.get(game)
        check(c is not None and (c["min"], c["max"], c["draw_size"]) == (lo, hi, drawn),
              f"{game}: range {lo}-{hi}, {drawn} drawn, against GAME_CONFIG {c and (c['min'], c['max'], c['draw_size'])}")
    check(set(config) == set(GAMES), f"every game of GAME_CONFIG is known here: {sorted(set(config) ^ set(GAMES))}")

    mean = lambda pmf: float((np.arange(len(pmf)) * pmf).sum())
    check(abs(mean(draw_pmf("lotto", range(1, 7), 6)) - 0.8) < 1e-9 and abs(mean(draw_pmf("lotto", range(1, 8), 6)) - 0.93333) < 1e-4,
          "lotto: 6 numbers 0.800 hits, 7 numbers 0.933")
    check(len(draw_pmf("lotto", [1, 2, 3, 4, 46, 47], 6)) == 5 and abs(mean(draw_pmf("lotto", [1, 2, 3, 4, 46, 47], 6)) - 4 * 6 / 45) < 1e-9,
          "lotto: two numbers out of range leave four that can hit")
    check(abs(mean(draw_pmf("lotto", [5, 5, 6, 7, 8, 9], 6)) - 5 * 6 / 45) < 1e-9, "a repeated number counts once")
    check(abs(mean(draw_pmf("euromillions", range(1, 6), 5)) - 0.5) < 1e-9 and abs(mean(draw_pmf("vikinglotto", range(1, 7), 6)) - 0.75) < 1e-9
          and abs(mean(draw_pmf("eurodreams", range(1, 7), 6)) - 0.9) < 1e-9, "EuroMillions 0.5, Vikinglotto 0.75, EuroDreams 0.9")
    check(abs(mean(draw_pmf("keno", range(1, 21), 20)) - 5.7143) < 1e-4 and abs(mean(draw_pmf("keno", list(range(1, 19)) + [71, 80], 20)) - 5.1429) < 1e-4,
          "keno: 20 numbers in range 5.714, 18 in range 5.143")
    check([round(mean(draw_pmf("pick3", t, 3)), 3) for t in ([1, 1, 1], [1, 1, 2], [1, 2, 3])] == [0.271, 0.542, 0.813], "pick3 by distinct digits")
    joker = draw_pmf("jokerplus", [1, 2, 3, 4, 5, 6], 6)
    check(abs(joker.sum() - 1) < 1e-12 and abs(mean(joker) - 0.2222) < 1e-4, f"Joker+: sums to 1, mean 0.2222 ({mean(joker):.4f})")
    # the hypergeometric against brute force on a small population, and Joker+'s run pmf against every outcome scored
    # by Helpers' own rule
    from itertools import combinations
    brute = np.zeros(4)
    for drawn in combinations(range(10), 3):
        brute[len({0, 1, 2, 3} & set(drawn))] += 1
    check(np.allclose(hypergeom_pmf(10, 3, 4), brute / brute.sum()), "hypergeometric = enumeration (10 numbers, 3 drawn, ticket of 4)")
    sys.path.insert(0, root)
    from src.Helpers import Helpers
    helpers = Helpers()
    outcomes = np.array([[(o // 10 ** i) % 10 for i in range(5, -1, -1)] for o in range(10 ** 6)], dtype=np.int8)
    ticket = np.array([3, 1, 4, 1, 5, 9])
    match = outcomes == ticket
    left = np.where(match.all(axis=1), 6, np.argmin(match, axis=1))
    right = np.argmin(match[:, ::-1], axis=1)
    every = np.bincount(np.where(left == 6, 6, left + right), minlength=7) / 10 ** 6
    check(np.allclose(every, joker), f"Joker+ pmf = every outcome's L + R: {np.round(every, 5)}")
    sample = random.Random(7).sample(range(10 ** 6), 3000)
    check(all(joker_hits(list(ticket), list(outcomes[o])) == sum(helpers.jokerplus_runs(list(ticket), list(outcomes[o]))) for o in sample),
          "joker_hits is Helpers.jokerplus_runs' L + R")

    # the exact tail against brute force: two pick3 draws (every pair of outcomes) and three Joker+ draws
    d0, d1 = date(2026, 1, 5), date(2026, 1, 6)
    pick = [{"date": d0, "ticket": [1, 2, 3], "result": [1, 1, 9], "hits": 1}, {"date": d1, "ticket": [4, 4, 5], "result": [4, 5, 0], "hits": 2}]
    hits_a = np.array([len({1, 2, 3} & {o // 100, (o // 10) % 10, o % 10}) for o in range(1000)])
    hits_b = np.array([len({4, 5} & {o // 100, (o // 10) % 10, o % 10}) for o in range(1000)])
    brute_p = float(((hits_a[:, None] + hits_b[None, :]) >= 3).mean())
    check(abs(exact_test("pick3", pick)["p"] - brute_p) < 1e-12, f"pick3: exact p {exact_test('pick3', pick)['p']} = enumeration {brute_p}")
    tail = Tail()
    for _ in range(3):
        tail.add(joker, 0)
    tail.total = 2
    three = np.convolve(np.convolve(joker, joker), joker)
    check(abs(tail.p() - three[2:].sum()) < 1e-12 and tail.level() <= ALPHA, "Tail: P(total >= 2) over three Joker+ draws, its level under alpha")

    # the block permutation
    rng = random.Random(4)
    days = [date(2026, 10, 1) + timedelta(days=i) for i in range(28)]
    outcomes = [[rng.randrange(10) for _ in range(5)] for _ in days]
    constant = [{"date": d, "ticket": [4, 4, 5, 3, 5], "result": o, "hits": sum(a == b for a, b in zip([4, 4, 5, 3, 5], o))} for d, o in zip(days, outcomes)]
    check(permutation_test("crypto", "C", constant)["p"] == 1.0, "a constant ticket scores the same shuffled: p = 1")
    distinct = [[i % 10, (i * 3) % 10, (i * 7) % 10, (i + 5) % 10, (i * 9 + 1) % 10] for i in range(len(days))]
    seer = [{"date": d, "ticket": o, "result": o, "hits": 5} for d, o in zip(days, distinct)]
    check(permutation_test("crypto", "S", seer)["p"] == 1 / (PERMUTATIONS + 1), "a row that knows each day's outcome: p = 1 / (B + 1)")
    noisy = [{"date": d, "ticket": o2, "result": o, "hits": sum(a == b for a, b in zip(o2, o))}
             for d, o, o2 in zip(days, outcomes, outcomes[1:] + outcomes[:1])]
    again = permutation_test("crypto", "C", noisy)
    check(again == permutation_test("crypto", "C", noisy) and again["p"] != permutation_test("crypto", "D", noisy)["p"],
          "seeded by game, row, length and last date: the same record gives the same p, another row its own")
    # regime timing: four weeks alternating calm (bin 2) and wild (bin 8); a row that calls the week's regime passes a
    # free shuffle but not the blocks - the regime is not a call on the day
    regime = [[2] * 5 if (i // 7) % 2 == 0 else [8] * 5 for i in range(28)]
    timer = [{"date": d, "ticket": o, "result": o, "hits": 5} for d, o in zip(days, regime)]
    check(permutation_test("crypto", "R", timer)["p"] == 1.0 and permutation_test("crypto", "R", timer, block=28)["p"] < 0.001,
          "a row that only knows the week's regime: p 1 in blocks of 7, < 0.001 shuffled freely")
    check(abs(permutation_test("crypto", "C", constant)["null"] - _market_null(constant)) < 1e-12, "the block null without shuffles")
    check(all(abs(np.trace(hits_matrix(x)) / len(x) - sum(r["hits"] for r in x) / len(x)) < 1e-12 for x in (constant, seer, timer)),
          "the matrix diagonal is the hits as scored")

    check([round(x, 4) for x in holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06], f"Holm by hand: {holm([0.01, 0.04, 0.03])}")
    check(sunday_on_or_before(date(2026, 10, 27)) == date(2026, 10, 25) and sunday_on_or_before(date(2026, 10, 25)) == date(2026, 10, 25)
          and sunday_on_or_after(date(2026, 9, 27)) == date(2026, 9, 27) and sunday_on_or_after(date(2026, 9, 28)) == date(2026, 10, 4), "Sundays")

    # one lottery game, a row that runs hot and is then followed: lotto draws Wednesday and Saturday from 3 Jan 2026
    draws = []
    day = date(2026, 1, 3)
    while len(draws) < 60:
        draws.append(day)
        day += timedelta(days=4 if day.weekday() == 5 else 3)
    result = [1, 2, 3, 4, 5, 6]

    def table(hot_until, live=lambda i: True):
        """'Hot' hits 3 a draw up to draw `hot_until`, 0 after; 'Cold' 1 on every other draw (0.5 against 0.8)."""
        out = []
        for i, d in enumerate(draws):
            rows = {}
            if live(i):
                rows["Hot Model"] = {"ticket": [1, 2, 3, 40, 41, 42] if i < hot_until else [40, 41, 42, 43, 44, 45], "hits": 3 if i < hot_until else 0}
            rows["Cold Model"] = {"ticket": [1, 30, 31, 32, 33, 34], "hits": 1} if i % 2 else {"ticket": [30, 31, 32, 33, 34, 35], "hits": 0}
            out.append({"date": d, "result": result, "live": rows})
        return out

    models = [{"name": "Hot Model", "draws": 60}, {"name": "Cold Model", "draws": 60}]
    end = draws[-1] + timedelta(days=1)
    per, summary, entry = game_chance("lotto", table(12, live=lambda i: i != 5), models, 10, end)
    hot, cold = per["Hot Model"], per["Cold Model"]
    first_sunday = sunday_on_or_after(draws[10] + timedelta(days=1))   # the 10th live draw is draws[10] (draws[5] not live)
    check(hot["draws"] == 59 and hot["notLive"] == 1 and hot["inFamily"] and cold["draws"] == 60 and cold["notLive"] == 0,
          f"a draw not live is left out and counted apart: {hot['draws']} live, {hot['notLive']} not")
    check(entry["rows"]["Hot Model"]["since"] == first_sunday.isoformat() and entry["rows"]["Hot Model"]["flagDraws"] == 10,
          f"flagged on the first Sunday with 10 live draws before it: {entry['rows'].get('Hot Model')}")
    fresh = hot["fresh"]
    check(fresh["verdictLength"] == 10 and fresh["state"] == "not confirmed" and fresh["mean"] == round((1 * 3) / 10, 4)
          and fresh["verdictAt"] == [d for d in draws if d >= first_sunday][9].isoformat() and fresh["decidedOn"] == end.isoformat(),
          f"judged on exactly the first 10 new draws (one hot draw left, then nine cold): {fresh}")
    check("Cold Model" not in entry["rows"] and "fresh" not in cold and summary["family"] == 2 and summary["notConfirmed"] == 1
          and summary["followed"] == 0 and entry["checkedThrough"] == sunday_on_or_before(end).isoformat(),
          f"the cold row is never followed; the summary counts the verdict: {summary}")
    check(summary["holds"] == sum(1 for b in per.values() if b.get("pHolm") is not None and b["pHolm"] < ALPHA)
          and summary["under"] == sum(1 for b in per.values() if b.get("inFamily") and b["p"] < ALPHA) and 0 < summary["expected"] < 2 * ALPHA,
          f"holds, under and the chance count from the rows themselves: {summary}")

    # first write wins: a decided verdict is read back even when the record changes under it (a rescan, a rewrite)
    rewritten = table(30)
    per2, _, entry2 = game_chance("lotto", rewritten, models, 10, end, stored=entry)
    check(per2["Hot Model"]["fresh"]["state"] == "not confirmed" and entry2 == entry,
          f"a decided verdict is never recomputed: {per2['Hot Model']['fresh']['state']}")
    # a row that keeps hitting is confirmed; before its last fresh draw it is running, with no p
    per3, _, entry3 = game_chance("lotto", table(60), models, 10, end)
    check(per3["Hot Model"]["fresh"]["state"] == "confirmed" and per3["Hot Model"]["fresh"]["p"] < VERDICT_ALPHA, f"a lasting row is confirmed: {per3['Hot Model']['fresh']}")
    early = draws[18] + timedelta(days=1)
    short_table = [d for d in table(60) if d["date"] <= draws[18]]
    per4, s4, entry4 = game_chance("lotto", short_table, models, 10, early)
    f4 = per4["Hot Model"]["fresh"]
    check(f4["state"] == "running" and f4["p"] is None and f4["mean"] == 3.0 and f4["draws"] < f4["verdictLength"] and s4["running"] == 1,
          f"running: the figures so far, no p: {f4}")
    # later runs test only the new Sundays: the stored flag does not move when an earlier Sunday would now read hot
    per5, _, entry5 = game_chance("lotto", table(60), models, 10, end, stored=entry4)
    check(entry5["rows"]["Hot Model"]["since"] == entry4["rows"]["Hot Model"]["since"] and per5["Hot Model"]["fresh"]["state"] == "confirmed",
          "the next run reads the flag back and decides once its new draws are in")
    stopped, _, _ = game_chance("lotto", short_table, models, 10, early + timedelta(days=STOPPED_DAYS), stored=entry4)
    check(stopped["Hot Model"]["fresh"]["state"] == "stopped", "four weeks without a new live draw: stopped")
    # the verdict length: at least minDraws, at most MAX_VERDICT
    many = []
    day = date(2025, 1, 4)
    for i in range(130):
        many.append({"date": day, "result": result, "live": {"Long Model": {"ticket": [1, 2, 40, 41, 42, 43] if i % 4 else [1, 2, 3, 41, 42, 43], "hits": 2 if i % 4 else 3}}})
        day += timedelta(days=4 if day.weekday() == 5 else 3)
    _, _, long_entry = game_chance("lotto", many, [{"name": "Long Model", "draws": 130}], 10, day)
    check(long_entry["rows"]["Long Model"]["verdictLength"] == min(max(long_entry["rows"]["Long Model"]["flagDraws"], 10), MAX_VERDICT),
          f"the verdict length is the flag's draws, clamped: {long_entry['rows']['Long Model']}")
    # a run judged without git stores nothing and tests no Sunday; the stored entries are still read
    per6, s6, kept = game_chance("lotto", table(60), models, 10, end, stored=entry4, persist=False)
    check(kept is entry4 and s6["stored"] == "read only" and per6["Hot Model"]["fresh"]["since"] == entry4["rows"]["Hot Model"]["since"]
          and entry4["rows"]["Hot Model"]["state"] == "running", "read only: nothing tested, nothing kept")
    # a later since.json freeze resets the rows flagged before it, which start again from it
    per7, _, entry7 = game_chance("lotto", table(60), models, 10, end, stored=dict(entry4, freeze="2025-01-01"), freeze=draws[30])
    since7 = entry7["rows"].get("Hot Model", {}).get("since")
    check(entry7["freeze"] == draws[30].isoformat() and since7 is not None and since7 >= draws[30].isoformat(),
          f"a later freeze: the row flagged before it is flagged again from it: {entry7['rows'].get('Hot Model')}")
    # an entry kept under the freeze before 30 Oct (no testFrom): the next run resets it once, then never again
    legacy = dict(entry4, freeze=draws[30].isoformat(), testFrom=None)
    _, _, migrated = game_chance("lotto", table(60), models, 10, end, stored=legacy, freeze=draws[30])
    since_m = migrated["rows"].get("Hot Model", {}).get("since")
    check(migrated["testFrom"] == sunday_on_or_after(draws[30]).isoformat() and (since_m is None or since_m >= draws[30].isoformat()),
          f"a pre-30-Oct entry is reset once: {migrated}")
    _, _, again = game_chance("lotto", table(60), models, 10, end, stored=migrated, freeze=draws[30])
    check(again["rows"] == migrated["rows"] and again["testFrom"] == migrated["testFrom"], "and never again")
    # the first replay keeps a freeze already declared: no flag takes a Sunday before it
    _, _, first_frozen = game_chance("lotto", table(60), models, 10, end, freeze=draws[30])
    since_ff = first_frozen["rows"].get("Hot Model", {}).get("since")
    check(first_frozen["testFrom"] == sunday_on_or_after(draws[30]).isoformat() and (since_ff is None or since_ff >= draws[30].isoformat()),
          f"the first replay applies a declared freeze: {first_frozen}")
    _, _, entry8 = game_chance("lotto", table(60), models, 10, end, stored=entry5, freeze=date(2025, 1, 1))
    check(entry8["rows"] == entry5["rows"], "an earlier or the same freeze resets nothing")
    ahead = end + timedelta(days=10)
    _, _, entry9 = game_chance("lotto", table(60), models, 10, end, stored=dict(entry4, freeze="2025-01-01"), freeze=ahead)
    _, _, entry10 = game_chance("lotto", table(60), models, 10, end + timedelta(days=7), stored=entry9, freeze=ahead)
    check("Hot Model" not in entry9["rows"] and "Hot Model" not in entry10["rows"] and entry10["testFrom"] == sunday_on_or_after(ahead).isoformat(),
          f"a freeze dated ahead: no row is flagged before its day, not even by the next week's run: {entry10}")

    # A game that draws on Sundays too (27 Oct 2026 review: lotto's Wednesdays and Saturdays above left every rule below
    # unchecked). Keno from Thursday 1 Jan 2026, every day: 1-10 Jan are the 10 draws before Sunday 11 Jan.
    #   Edge   hot on 1-10 Jan, then a fresh p between VERDICT_ALPHA and ALPHA on 11-20 Jan: flagged on 11 Jan by the
    #          10 draws before it - the draw dated that Sunday is the first fresh one, never in the record that flagged
    #          it - and judged on 11-20 Jan: not confirmed, the verdict needs p < VERDICT_ALPHA
    #   Late   cold for 60 draws, then all 20 numbers every day: flagged on more than MAX_VERDICT draws, judged on MAX_VERDICT
    #   Gone   hot on 1-10 Jan, live on 11-13 Jan only: running for 27 days after its last draw, stopped from STOPPED_DAYS
    #   Young  3 live draws of 15 hits: its p is tiny, but it is outside the family and the "under" count
    every_day = [date(2026, 1, 1) + timedelta(days=i) for i in range(150)]
    twenty = list(range(1, 21))
    edge = Tail()
    for _ in range(10):
        edge.add(draw_pmf("keno", twenty, 20), 0)
    tails = np.cumsum(edge.acc[::-1])[::-1]
    edge_total = int(np.argmax(tails < ALPHA))            # the smallest 10-draw total under ALPHA
    check(VERDICT_ALPHA <= tails[edge_total] < ALPHA, f"keno: a 10-draw total between the two levels ({edge_total}: {tails[edge_total]:.4f})")
    edge_hits = [edge_total // 10 + (1 if i < edge_total % 10 else 0) for i in range(10)]

    def keno_table(last):
        out = []
        for i, d in enumerate(every_day):
            if d >= last:
                break
            rows = {"Edge Model": {"ticket": twenty, "hits": 12 if i < 10 else (edge_hits[i - 10] if i < 20 else 6)},
                    "Late Model": {"ticket": twenty, "hits": 2 if i < 60 else 20}}
            if i < 13:
                rows["Gone Model"] = {"ticket": twenty, "hits": 12 if i < 10 else 6}
            if 140 <= i < 143:
                rows["Young Model"] = {"ticket": twenty, "hits": 15}
            out.append({"date": d, "result": twenty, "live": rows})
        return out

    keno_models = [{"name": n, "draws": 150} for n in ("Edge Model", "Late Model", "Gone Model", "Young Model")]
    kend = every_day[-1] + timedelta(days=1)
    perk, sk, ek = game_chance("keno", keno_table(kend), keno_models, 10, kend)
    fe = perk["Edge Model"].get("fresh") or {}
    check(ek["rows"].get("Edge Model", {}).get("since") == "2026-01-11" and fe.get("flagDraws") == 10 and fe.get("verdictAt") == "2026-01-20"
          and fe.get("state") == "not confirmed" and fe.get("p") is not None and VERDICT_ALPHA <= fe["p"] < ALPHA,
          f"a Sunday's draw is fresh, never in the flag's record; a fresh p under {ALPHA} but not {VERDICT_ALPHA} is not confirmed: {fe}")
    late = ek["rows"].get("Late Model") or {}
    check(late.get("flagDraws", 0) > MAX_VERDICT and late.get("verdictLength") == MAX_VERDICT,
          f"a flag on more than {MAX_VERDICT} draws is judged on {MAX_VERDICT}: {late}")
    young = perk["Young Model"]
    check(young["draws"] == 3 and young["p"] < ALPHA and not young["inFamily"] and young["pHolm"] is None and "Young Model" not in ek["rows"]
          and sk["family"] == 3 and sk["under"] == sum(1 for n in ("Edge Model", "Late Model", "Gone Model") if perk[n]["p"] < ALPHA),
          f"a row with fewer than minDraws live draws: outside the family, Holm and the under count: {young} {sk}")
    gone_last = every_day[12]
    still, _, _ = game_chance("keno", keno_table(gone_last + timedelta(days=STOPPED_DAYS - 1)), keno_models, 10, gone_last + timedelta(days=STOPPED_DAYS - 1))
    gone, _, _ = game_chance("keno", keno_table(gone_last + timedelta(days=STOPPED_DAYS)), keno_models, 10, gone_last + timedelta(days=STOPPED_DAYS))
    check(still["Gone Model"]["fresh"]["state"] == "running" and gone["Gone Model"]["fresh"]["state"] == "stopped"
          and gone["Gone Model"]["fresh"]["draws"] == 3, f"stopped from exactly {STOPPED_DAYS} days after the last new live draw")
    # the same for a market: the day dated on the flag's Sunday is fresh, never in the shuffled record that flagged it
    seen = [[i % 10, (i * 3) % 10, (i * 7) % 10, (i + 5) % 10, (i * 9 + 1) % 10] for i in range(30)]
    market_days = [{"date": every_day[i], "result": seen[i], "live": {"Seer Model": {"ticket": seen[i], "hits": 5}}} for i in range(30)]
    _, _, em = game_chance("crypto", market_days, [{"name": "Seer Model", "draws": 30}], 10, every_day[30])
    seer_kept = em["rows"].get("Seer Model") or {}
    check(seer_kept.get("since") == "2026-01-11" and seer_kept.get("flagDraws") == 10 and seer_kept.get("verdictAt") == "2026-01-20"
          and seer_kept.get("state") == "confirmed", f"a market's flag and verdict around the Sunday's own day: {seer_kept}")

    # the file: sorted, one line per row, written only when it changes
    folder = tempfile.mkdtemp(prefix="rowchance-")
    try:
        store = {"version": VERSION, "games": {"lotto": entry5, "crypto": _new_entry(None)}}
        check(write_store(folder, store) and not write_store(folder, copy.deepcopy(store)), "written once, unchanged the second time")
        text = open(os.path.join(folder, FILE_NAME)).read()
        check(read_store(folder) == {**json.loads(text)} and json.loads(text)["games"]["lotto"] == entry5 and text.index('"crypto"') < text.index('"lotto"'),
              "the file reads back as written, games sorted")
        with open(os.path.join(folder, FILE_NAME), "w") as handle:
            handle.write('{"version": 1, "games": {"lo')
        check(read_store(folder) is None and read_store(os.path.join(folder, "none")) == {"version": VERSION, "games": {}},
              "a broken file reads None (nothing written over it), a missing one empty")
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    # the report: a crypto game (stamped tickets, live by the day) and a lotto game in a folder that is no git checkout
    # (src/Liveness.py: "stamp only" - the tests run, nothing is stored for it)
    repo = tempfile.mkdtemp(prefix="rowchance-report-")
    try:
        database = os.path.join(repo, "data", "database")
        os.makedirs(os.path.join(database, "crypto"))
        os.makedirs(os.path.join(database, "lotto"))
        start = date(2026, 8, 30)
        outcomes = [[rng.randrange(10) for _ in range(5)] for _ in range(40)]
        for i in range(40):
            d = start + timedelta(days=i)
            ticket = outcomes[i] if i % 3 else [0, 0, 0, 0, 0]   # Seer knows two days in three
            made = datetime(d.year, d.month, d.day, 8, 0).isoformat() + "+00:00"   # the next day's ticket, out before its close
            with open(os.path.join(database, "crypto", f"{d.year}-{d.month}-{d.day}.json"), "w") as handle:
                json.dump({"realResult": outcomes[i], "madeAt": made,
                           "currentPrediction": [{"name": "Seer Model", "predictions": [ticket]}, {"name": "Flat Model", "predictions": [[4, 4, 5, 3, 5]]}],
                           "newPrediction": [{"name": "Seer Model", "predictions": [[0] * 5]}, {"name": "Flat Model", "predictions": [[4, 4, 5, 3, 5]]}]}, handle)
        for i, d in enumerate(draws[:20]):
            made = (datetime(d.year, d.month, d.day, 21, 0) - timedelta(days=3)).isoformat() + "+00:00"
            with open(os.path.join(database, "lotto", f"{d.year}-{d.month}-{d.day}.json"), "w") as handle:
                json.dump({"realResult": [1, 2, 3, 4, 5, 6, 7], "madeAt": made,
                           "currentPrediction": [{"name": "A Model", "predictions": [[1, 2, 3, 40, 41, 42]]}],
                           "newPrediction": [{"name": "A Model", "predictions": [[1, 2, 3, 40, 41, 42]]}]}, handle)

        def report():
            with contextlib.redirect_stdout(io.StringIO()):
                return Helpers().generate_model_performance_report(database, combinationShuffles=0)

        first = report()
        crypto = first["games"]["crypto"]
        seer = next(m for m in crypto["models"] if m["name"] == "Seer Model")
        flat = next(m for m in crypto["models"] if m["name"] == "Flat Model")
        check(crypto["chance"]["method"] == "permutation" and crypto["chance"]["block"] == BLOCK and crypto["chance"]["stored"] == "written",
              f"crypto: the block permutation, stored: {crypto.get('chance')}")
        check(seer["chance"]["draws"] == seer["live_draws"] and seer["chance"]["notLive"] == seer["draws"] - seer["live_draws"]
              and seer["chance"]["p"] < 0.001 and flat["chance"]["p"] == 1.0 and seer["chance"]["fresh"]["since"],
              f"a row's live draws as the report counts them; the seer passes and is followed, the flat row cannot: {seer['chance']} {flat['chance']}")
        lotto = first["games"]["lotto"]
        check(lotto["liveness"]["checked"] == "stamp only" and lotto["chance"]["stored"] == "read only: liveness judged without git" and lotto["models"][0]["chance"]["draws"] >= 10,
              f"lotto judged by stamps alone: tested, nothing stored: {lotto.get('chance')}")
        kept_text = open(os.path.join(database, FILE_NAME)).read()
        check(json.loads(kept_text)["games"].get("crypto", {}).get("rows", {}).get("Seer Model") and "lotto" not in json.loads(kept_text)["games"],
              "the file keeps crypto's followed row and nothing of the stamp-only lotto")
        mtime = os.stat(os.path.join(database, FILE_NAME)).st_mtime_ns
        second = report()
        check(open(os.path.join(database, FILE_NAME)).read() == kept_text and os.stat(os.path.join(database, FILE_NAME)).st_mtime_ns == mtime
              and second["games"]["crypto"]["chance"] == crypto["chance"],
              "a rerun with no new draw: the file untouched, the same figures")
        # a failure in the tests costs the fields, never the report
        import importlib
        module = importlib.import_module("src.RowChance")   # the module the report imports (this file runs as __main__)
        kept_function = module.game_chance
        module.game_chance = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            failed = report()
        finally:
            module.game_chance = kept_function
        check("chance" not in failed["games"]["crypto"] and all("chance" not in m for m in failed["games"]["crypto"]["models"])
              and failed["games"]["crypto"]["bestModel"] and open(os.path.join(database, FILE_NAME)).read() == kept_text,
              "a test that raises: no chance fields, the report and the file as they were")
        # a file caught broken is never written over: the tests run, nothing is kept, the page says why
        with open(os.path.join(database, FILE_NAME), "w") as handle:
            handle.write('{"version": 1, "games": {"cry')
        broken_run = report()
        check(open(os.path.join(database, FILE_NAME)).read() == '{"version": 1, "games": {"cry'
              and broken_run["games"]["crypto"]["chance"]["stored"] == f"read only: {FILE_NAME} could not be read"
              and broken_run["games"]["crypto"]["chance"]["family"] == crypto["chance"]["family"],
              f"an unreadable {FILE_NAME}: tested, nothing written over it, said why")
    finally:
        shutil.rmtree(repo, ignore_errors=True)

    if failures:
        print("RowChance self-check FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("RowChance self-check passed")


if __name__ == "__main__":
    _self_check()
