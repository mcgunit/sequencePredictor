"""
The backtest record's month arithmetic (roadmap item 13, decided 21 and 22 Nov 2026).

A row's record is: in how many of ten past calendar months did playing its ticket on every draw end in profit.
This module holds what needs no model and no backtest - which months, a backtested draw's stake and profit, a
month's totals, and the same months played with random tickets - so it can be checked on its own. BacktestRecord.py
runs the backtests and writes the files; backtestRecord.js reads them for the pages.

  the months      the ten newest complete calendar months that do not touch the lockbox (lockbox.json): its days
                  stay shut, so a month with one of them is left out and an older month takes its place.
  a draw's money  Keno: every playable sub-selection the row serves is one 1 EUR bet. Pick3: the ticket on all bet
                  types (4 EUR, 3 for a triple). Joker+: one 1.50 EUR ticket. Lotto, EuroMillions, EuroDreams,
                  Vikinglotto: one grid at the draw's price against the draw's own prize table (src/PrizeTables.py);
                  a draw without a table is not played (counted as `unscored`).
  random tickets  SIMS random players per kind of play (a game's one ticket; for Keno one per ticket size), each
                  month's total per player kept. A row's reference is the average number of months in profit over
                  those players; a combination's is the same over the sums of as many players as it has members.
                  One random ticket's own count would be luck (decided 22 Nov 2026).

    python3 -m src.BacktestMonths      # self-check (in npm test)
"""

from __future__ import annotations

import zlib
from datetime import date

MONTHS = 10
SIMS = 400
KENO_SIZES = (5, 6, 7, 8, 9, 10)
PRIZE_GAMES = ("lotto", "euromillions", "eurodreams", "vikinglotto")
# the offsets that keep two plays of one member, and two members of one combination, on different random players
# (backtestRecord.js uses the same two numbers)
MEMBER_STEP, PLAY_STEP = 53, 17


def month_of(day):
    return str(day)[:7]


def previous_month(month):
    year, number = int(month[:4]), int(month[5:7])
    return f"{year - 1}-12" if number == 1 else f"{year}-{number - 1:02d}"


def month_span(month):
    """(first day, last day) of YYYY-MM, as ISO days."""
    year, number = int(month[:4]), int(month[5:7])
    following = date(year + 1, 1, 1) if number == 12 else date(year, number + 1, 1)
    return f"{month}-01", date.fromordinal(following.toordinal() - 1).isoformat()


def wanted_months(today, lockbox=None, count=MONTHS):
    """The `count` newest complete calendar months before `today` (ISO day) that hold no lockbox day, oldest first."""
    shut_from, shut_to = (lockbox or {}).get("from"), (lockbox or {}).get("to")
    months, month = [], previous_month(month_of(today))
    while len(months) < count:
        first, last = month_span(month)
        if not (shut_from and shut_to and first <= shut_to and last >= shut_from):
            months.append(month)
        month = previous_month(month)
    return months[::-1]


def segments(months):
    """Runs of months that follow each other: [(first day, last day, [months])] - one backtest each."""
    runs = []
    for month in sorted(months):
        if runs and previous_month(month) == runs[-1][-1]:
            runs[-1].append(month)
        else:
            runs.append([month])
    return [(month_span(run[0])[0], month_span(run[-1])[1], run) for run in runs]


def draw_money(game, helpers, name, row, prize=None, typical=None, bonus=None):
    """
    (bets, stake, profit, plays) of one backtested draw for the model `name` (a Backtester result row), or None when
    the draw cannot be played (no prize table, not a full grid, no playable keno sub-selection).
    `plays`: the kinds of play made - the keno ticket sizes as strings, else ["ticket"].
    """
    ticket = row.get(f"{name}_prediction")
    if game == "keno":
        sizes = [size for size in KENO_SIZES if isinstance(row.get(f"{name}_subset_{size}_profit"), (int, float))]
        if not sizes:
            return None
        return len(sizes), float(len(sizes)), float(sum(row[f"{name}_subset_{size}_profit"] for size in sizes)), [str(s) for s in sizes]
    if game == "pick3":
        profit = row.get(f"{name}_profit")
        if not isinstance(profit, (int, float)) or not ticket:
            return None
        cost = helpers.PAYOUT_TABLE_PICK3["bet_cost"]
        return 1, float((3 if len(set(ticket)) == 1 else 4) * cost), float(profit), ["ticket"]
    if game == "jokerplus":
        profit = row.get(f"{name}_profit")
        if not isinstance(profit, (int, float)):
            return None
        return 1, 1.5, float(profit), ["ticket"]
    if game in PRIZE_GAMES:
        from src import PrizeTables
        numbers, stars = PrizeTables.GRID[game]
        if not ticket or len(ticket) != numbers + stars:
            return None
        main_hits = row.get(f"{name}_hits")
        if game == "lotto":
            special_hits = int(bonus is not None and int(bonus) in set(map(int, ticket)))
        else:
            special_hits = row.get(f"{name}_special_hits")
        if main_hits is None or special_hits is None:
            return None
        profit = PrizeTables.hits_profit(game, int(main_hits), int(special_hits), prize, typical)
        if profit is None:
            return None
        return 1, PrizeTables.grid_price(game, prize), float(profit), ["ticket"]
    return None


def month_totals(lines, months):
    """{month: {draws, bets, stake, profit, unscored}} from [(day, money or None)], for the wanted months only."""
    out = {month: {"draws": 0, "bets": 0, "stake": 0.0, "profit": 0.0, "unscored": 0} for month in months}
    for day, money in lines:
        month = out.get(month_of(day))
        if month is None:
            continue
        if money is None:
            month["unscored"] += 1
            continue
        month["draws"] += 1
        month["bets"] += money[0]
        month["stake"] += money[1]
        month["profit"] += money[2]
    for month in out.values():
        month["stake"], month["profit"] = round(month["stake"], 2), round(month["profit"], 2)
    return out


def profit_months(months):
    """How many of the months ended in profit, among those with a played draw: (in profit, played)."""
    played = [m for m in months.values() if m.get("draws")]
    return sum(1 for m in played if m["profit"] > 0), len(played)


def special_range(draws, mains):
    """(lowest, highest) special value seen in the draws' columns after the first `mains` - the range a random special is drawn from."""
    values = [int(v) for draw in draws for v in draw[mains:]]
    return (min(values), max(values)) if values else None


def random_month(game, helpers, month, draws, config, prizes=None, typical=None, specials=None, sims=SIMS):
    """
    {play: [each random player's total for the month]} over the month's draws (each a full row in drawn order, with its
    ISO day first: (day, values)). Seeded by game and month, so a month is the same whenever it is simulated.
    `config`: the game's GAME_CONFIG entry; `specials`: special_range of the game's recent draws.
    """
    import numpy as np
    rng = np.random.default_rng(zlib.crc32(f"{game}|{month}".encode()))
    low, high, size = int(config["min"]), int(config["max"]), int(config["draw_size"])
    plays = [str(s) for s in KENO_SIZES] if game == "keno" else ["ticket"]
    totals = {play: [0.0] * sims for play in plays}
    for day, values in draws:
        values = [int(v) for v in values]
        entry = (prizes or {}).get(day) if game in PRIZE_GAMES else None
        if game in PRIZE_GAMES and (not entry or not entry.get("price") or not entry.get("tiers")):
            continue   # not played by the rows either
        for sim in range(sims):
            if game == "keno":
                for play in plays:
                    ticket = rng.choice(np.arange(low, high + 1), size=int(play), replace=False)
                    totals[play][sim] += helpers.keno_ticket_profit([int(n) for n in ticket], values)
            elif game == "pick3":
                totals["ticket"][sim] += helpers.pick3_ticket_profit([int(n) for n in rng.integers(low, high + 1, size=3)], values)
            elif game == "jokerplus":
                sign = int(rng.integers(specials[0], specials[1] + 1))
                ticket = [int(n) for n in rng.integers(0, 10, size=6)] + [sign]
                totals["ticket"][sim] += helpers.jokerplus_ticket_profit(ticket, values)
            else:
                from src import PrizeTables
                numbers, stars = PrizeTables.GRID[game]
                mains = set(int(n) for n in rng.choice(np.arange(low, high + 1), size=numbers, replace=False))
                real_mains, real_specials = set(values[:size]), set(values[size:])
                if game == "lotto":
                    special_hits = len(mains & real_specials)
                else:
                    picked = set(int(n) for n in rng.choice(np.arange(specials[0], specials[1] + 1), size=stars, replace=False))
                    special_hits = len(picked & real_specials)
                totals["ticket"][sim] += PrizeTables.hits_profit(game, len(mains & real_mains), special_hits, entry, typical)
    return {play: [round(v, 2) for v in series] for play, series in totals.items()}


def random_count(random_months, members, months=None):
    """
    The average number of months in profit over the random players, for `members` = [[plays of member 1], ...]:
    player s of a month is the sum of each member's plays, every play on a player of its own (the offsets above).
    None when a month or a play has no simulation. backtestRecord.js computes the same.
    """
    months = sorted(months or random_months)
    if not months or not members:
        return None
    total, sims = 0, None
    for month in months:
        plays = random_months.get(month)
        if not plays:
            return None
        sims = min(len(series) for series in plays.values())
        for sim in range(sims):
            value = 0.0
            for m, member in enumerate(members):
                for p, play in enumerate(member):
                    series = plays.get(play)
                    if series is None:
                        return None
                    value += series[(sim + m * MEMBER_STEP + p * PLAY_STEP) % sims]
            total += value > 0
    return round(total / sims, 2) if sims else None


def _self_check():
    from src.Helpers import Helpers
    helpers = Helpers()
    box = {"from": "2025-12-01", "to": "2026-02-28"}
    months = wanted_months("2026-11-22", box)
    assert months == ["2025-10", "2025-11", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"], months
    assert wanted_months("2026-11-01", None, 3) == ["2026-08", "2026-09", "2026-10"]
    assert wanted_months("2026-03-15", {"from": "2026-02-27", "to": "2026-02-28"}, 2) == ["2025-12", "2026-01"], "a month with one lockbox day is out"
    assert month_span("2026-02") == ("2026-02-01", "2026-02-28") and month_span("2025-12") == ("2025-12-01", "2025-12-31")
    runs = segments(months)
    assert [(a, b, len(c)) for a, b, c in runs] == [("2025-10-01", "2025-11-30", 2), ("2026-03-01", "2026-10-31", 8)], runs

    # a draw's money per game family
    keno = {"m_subset_5_profit": -1, "m_subset_7_profit": 2, "m_subset_4_profit": None}
    assert draw_money("keno", helpers, "m", keno) == (2, 2.0, 1.0, ["5", "7"])
    assert draw_money("keno", helpers, "m", {}) is None
    assert draw_money("pick3", helpers, "m", {"m_prediction": [1, 2, 3], "m_profit": -4.0}) == (1, 4.0, -4.0, ["ticket"])
    assert draw_money("pick3", helpers, "m", {"m_prediction": [7, 7, 7], "m_profit": -3.0})[1] == 3.0, "a triple cannot play box"
    assert draw_money("jokerplus", helpers, "m", {"m_prediction": [1] * 7, "m_profit": 0.5}) == (1, 1.5, 0.5, ["ticket"])
    table = {"price": 1.5, "jackpot": 1000000.0, "tiers": [
        {"name": "6", "mains": 6, "special": 0, "amount": 0.0}, {"name": "3+1", "mains": 3, "special": 1, "amount": 11.5},
        {"name": "3", "mains": 3, "special": 0, "amount": 6.25}]}
    lotto = {"m_prediction": [1, 2, 3, 4, 5, 6], "m_hits": 3}
    assert draw_money("lotto", helpers, "m", lotto, table, {}, bonus=40) == (1, 1.5, 4.75, ["ticket"])
    assert draw_money("lotto", helpers, "m", lotto, table, {}, bonus=6)[2] == 10.0, "one of the six is the bonus number"
    assert draw_money("lotto", helpers, "m", lotto, None, {}) is None, "no table: not played"
    assert draw_money("lotto", helpers, "m", {"m_prediction": [1, 2, 3], "m_hits": 3}, table, {}) is None, "not a full grid"
    assert draw_money("euromillions", helpers, "m", {"m_prediction": [1, 2, 3, 4, 5, 6, 7], "m_hits": 0, "m_special_hits": 0},
                      {"price": 2.5, "tiers": [{"name": "5+2", "mains": 5, "special": 2, "amount": 1.0}]}, {}) == (1, 2.5, -2.5, ["ticket"])

    # months
    lines = [("2026-10-01", (1, 4.0, -4.0)), ("2026-10-02", (1, 4.0, 76.0)), ("2026-09-30", (1, 4.0, -4.0)), ("2026-09-29", None), ("2025-12-05", (1, 4.0, 500.0))]
    totals = month_totals(lines, ["2026-08", "2026-09", "2026-10"])
    assert totals["2026-10"] == {"draws": 2, "bets": 2, "stake": 8.0, "profit": 72.0, "unscored": 0}
    assert totals["2026-09"] == {"draws": 1, "bets": 1, "stake": 4.0, "profit": -4.0, "unscored": 1}
    assert profit_months(totals) == (1, 2), "a month without a played draw is not a month played"

    # random players: the same whenever simulated, one series per play, the house wins on average
    config = {"min": 0, "max": 9, "draw_size": 3}
    draws = [(f"2026-10-{d:02d}", [d % 10, (d * 3) % 10, (d * 7) % 10]) for d in range(1, 31)]
    first = random_month("pick3", helpers, "2026-10", draws, config, sims=60)
    assert first == random_month("pick3", helpers, "2026-10", draws, config, sims=60) and list(first) == ["ticket"] and len(first["ticket"]) == 60
    assert first != random_month("pick3", helpers, "2026-09", draws, config, sims=60), "another month, other players"
    assert sum(first["ticket"]) / 60 < 0, "random Pick3 tickets lose on average"
    kcfg = {"min": 1, "max": 70, "draw_size": 20}
    kdraws = [("2026-10-01", list(range(1, 21))), ("2026-10-02", list(range(11, 31)))]
    krand = random_month("keno", helpers, "2026-10", kdraws, kcfg, sims=40)
    assert sorted(krand) == sorted(str(s) for s in KENO_SIZES) and all(len(v) == 40 for v in krand.values())
    prand = random_month("lotto", helpers, "2026-10", [("2026-10-03", [1, 2, 3, 4, 5, 6, 7]), ("2026-10-07", [1, 2, 3, 4, 5, 6, 7])],
                         {"min": 1, "max": 45, "draw_size": 6}, prizes={"2026-10-03": table}, typical={}, sims=40)
    assert min(prand["ticket"]) >= -1.5, "the draw without a table is not played"

    # the count over random players
    sims = {"2026-09": {"ticket": [1.0, -1.0, -1.0, -1.0]}, "2026-10": {"ticket": [1.0, 1.0, -1.0, -1.0]}}
    assert random_count(sims, [["ticket"]]) == 0.75, random_count(sims, [["ticket"]])
    assert random_count(sims, [["ticket"]], ["2026-10"]) == 0.5
    assert random_count(sims, [["other"]]) is None and random_count(sims, [["ticket"]], ["2026-08"]) is None and random_count({}, [["ticket"]]) is None
    two = random_count(sims, [["ticket"], ["ticket"]])
    assert two is not None and 0 <= two <= 2
    print("BacktestMonths self-check OK")


if __name__ == "__main__":
    _self_check()
