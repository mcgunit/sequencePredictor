"""
Lotto multi-pick (system play) - the 7th, 8th and 9th numbers.

The owner's proposal (3 Oct 2026): keep predicting six numbers, but give
every row three more - the model's next most probable numbers - so a
reader can play a system of 7, 8 or 9 numbers (Belgian Lotto: 7 numbers =
7 grids = 10.50 EUR, 8 = 28 grids = 42 EUR, 9 = 84 grids = 126 EUR, at
1.50 EUR a grid), and order the whole row by the model's probability,
highest first, instead of small to large.

Two things this module keeps honest:

  the extras come from the row's OWN ranking     ordered_and_extras() takes
  the ticket the model played and its {number: score} ranking and returns
  the six in score order plus the next `extra` numbers it did not play; a
  row without a ranking (a vote, the RL ticket) gets none.

  more numbers win more by arithmetic alone       chance() is the exact
  hypergeometric: with 6 of 45 drawn, 6 numbers hit 3 or more of the drawn
  mains (the smallest prize that needs no bonus number; the bonus ranks are
  not scored here) 2.4% of the time, 7 numbers 3.9%, 8 numbers 5.9%, 9
  numbers 8.4% - 3.5 times the chance at 84 times the price. Any extra number hits 6/45 = 13.3% of the time by luck.
  So the question the tracking answers is not "do 9 numbers hit more than
  6" (they must) but "do the model's 7th-9th numbers hit more often than
  13.3%", and "do 9 model numbers beat the 8.4% that 9 random numbers get".

    python3 -m src.MultiPick        # self-check (in npm test)
"""

from __future__ import annotations

from math import comb

# game -> the rule. Matched on the game name; "vikinglotto" is not "lotto".
MULTI_PICK = {
    "lotto": {"extra": 3, "pool": (1, 45), "draw": 6, "min_hits": 3, "grid_price": 1.5,
              "grids": {6: 1, 7: 7, 8: 28, 9: 84}},
}


def config_for(name):
    """The multi-pick rule for a game name / folder, or None."""
    text = str(name or "").lower()
    if "vikinglotto" in text:
        return None
    for game, cfg in MULTI_PICK.items():
        if game in text:
            return dict(cfg, game=game)
    return None


def ordered_and_extras(ticket, scores, extra=3, pool=(1, 45)):
    """
    (the ticket's numbers in score order, highest first; the `extra` best
    numbers of the pool the ticket does not hold, same order). `scores` is
    {number: score} (string keys tolerated - JSON round trips make them);
    a ticket number without a score sorts last among the six, by number.
    Returns (list(ticket), []) when there is no ranking at all.
    """
    numbers = [int(n) for n in ticket]
    if not scores:
        return numbers, []
    ranking = {}
    for key, value in scores.items():
        try:
            ranking[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not ranking:
        return numbers, []
    low, high = pool
    ordered = sorted(numbers, key=lambda n: (-ranking.get(n, float("-inf")), n))
    held = set(numbers)
    candidates = [n for n in range(low, high + 1) if n not in held and n in ranking]
    extras = sorted(candidates, key=lambda n: (-ranking[n], n))[:max(0, int(extra))]
    return ordered, extras


def chance(picked, pool=45, drawn=6, min_hits=3):
    """
    Exact chances for `picked` numbers against `drawn` of `pool`: the
    expected hits and the probability of at least `min_hits` hits.
    """
    total = comb(pool, picked)
    at_least = sum(comb(drawn, k) * comb(pool - drawn, picked - k) for k in range(min_hits, min(drawn, picked) + 1)) / total
    return {"picked": picked, "expected_hits": drawn * picked / pool, "win": at_least}


def hits(numbers, real_mains):
    """How many of `numbers` are among the drawn main numbers."""
    return len({int(n) for n in numbers} & {int(n) for n in real_mains})


def chance_table(cfg):
    """Chance per system size the game sells, plus the single extra number's chance."""
    low, high = cfg["pool"]
    pool = high - low + 1
    return {
        "per_size": {size: dict(chance(size, pool, cfg["draw"], cfg["min_hits"]), grids=grids, price=grids * cfg["grid_price"])
                     for size, grids in cfg["grids"].items()},
        "per_extra": cfg["draw"] / pool,
        "min_hits": cfg["min_hits"],
    }


def _self_check():
    cfg = config_for("lotto")
    assert cfg and cfg["extra"] == 3 and config_for("vikinglotto") is None and config_for("data/database/lotto") and config_for("keno") is None
    scores = {n: 1.0 / n for n in range(1, 46)}          # 1 is the most probable, 45 the least
    scores[12] = 0.9                                     # a late number the model rates second
    ordered, extras = ordered_and_extras([3, 7, 12, 21, 36, 44], scores)
    assert ordered == [12, 3, 7, 21, 36, 44] and extras == [1, 2, 4], (ordered, extras)
    ordered, extras = ordered_and_extras([3, 7, 12, 21, 36, 44], {str(k): v for k, v in scores.items()}, extra=2)
    assert ordered[0] == 12 and extras == [1, 2], "string keys from a JSON round trip are fine"
    assert ordered_and_extras([5, 1, 9], None) == ([5, 1, 9], []) and ordered_and_extras([5, 1, 9], {"x": "y"}) == ([5, 1, 9], [])
    partial = ordered_and_extras([3, 7, 12], {3: 0.5, 12: 0.9}, extra=1, pool=(1, 45))
    assert partial == ([12, 3, 7], [])  , partial   # 7 has no score and sorts last; nothing outside the ticket has a score
    six, nine = chance(6), chance(9)
    assert abs(six["expected_hits"] - 0.8) < 1e-12 and abs(nine["expected_hits"] - 1.2) < 1e-12
    assert abs(six["win"] - 0.02382) < 0.0001 and abs(nine["win"] - 0.08395) < 0.0001, (six, nine)
    assert abs(chance(6, min_hits=6)["win"] - 1 / 8145060) < 1e-15
    table = chance_table(cfg)
    assert table["per_size"][9]["price"] == 126.0 and table["per_size"][7]["price"] == 10.5 and table["per_size"][8]["grids"] == 28
    assert abs(table["per_extra"] - 6 / 45) < 1e-12
    assert hits([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 9, 20, 30, 40, 45]) == 2 and hits([], [1]) == 0
    print(f"MultiPick self-check OK: 6 numbers win (3+) {six['win']:.2%}, 9 numbers {nine['win']:.2%} for "
          f"{table['per_size'][9]['price']:.0f} EUR; an extra number hits {table['per_extra']:.1%} by luck")


if __name__ == "__main__":
    _self_check()
