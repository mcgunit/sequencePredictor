"""
Prize tables of the pari-mutuel games - what a ticket really won.

Keno, Pick3 and Joker+ pay from fixed tables (Helpers.PAYOUT_TABLE_*). Lotto,
EuroMillions and EuroDreams pay per draw: the National Lottery's draw API
(the same call DataFetcher reads the numbers from) carries every draw's
prize tiers - per tier its name, the amount one winner got and how many
Belgians won it - back to at least 2015 (checked 3 Nov 2026). This module
keeps those tables, one file per game next to its draws
(data/trainingData/<game>/<game>-prizes.json, {date: entry}), and scores a
ticket against the table of its own draw, so the performance report can
rank these games on profit too (owner, 3 Nov 2026; average hits stays the
fallback wherever a draw has no table).

The rules a ticket is scored by (owner's decisions, 3 Nov 2026):

  one grid, the row's own ticket   Lotto: the six numbers (the 7th-9th of
      the multi-pick stay a hit-rate study), EuroMillions five numbers and
      two stars, EuroDreams six numbers and the dream number - at the
      draw's base price (1.50, 2.50, 2.50 EUR).
  a tier nobody won pays 0 in the API (the money rolls over), so a hit on
      it is valued as: the top tier the draw's announced jackpot; any other
      tier its median over the stored draws where it did pay; EuroDreams'
      two annuities their rule values (20,000 a month for 30 years, 2,000 a
      month for 5 years), whatever the API shows.
  the raffles (Lucky Lotto, My Bonus) are drawn on a code, not on the
      numbers: not scored.

Vikinglotto is not here: its tiers ('6/', '5+Fav', ... '1') could not be
matched to the game's rules (the winner counts do not fit "1 = one number"),
so it keeps average hits until they are confirmed.

    python3 -m src.PrizeTables      # self-check (in npm test)
"""

from __future__ import annotations

import json
import os
import re
import statistics
import tempfile
from datetime import datetime, timezone

# game folder -> the API's game-names parameter
GAMES = {"lotto": "Lotto", "euromillions": "Euro+Millions", "eurodreams": "EuroDreams"}
# one grid: (numbers, specials) - a longer ticket plays its first ones. Lotto's
# statistical rows carried 7 values until 13 May 2026: the 7th a guess at the
# bonus column (one Markov row's was 46, in a 1-45 game), not a 7th number one
# can play - their grid is the first six.
GRID = {"lotto": (6, 0), "euromillions": (5, 2), "eurodreams": (6, 1)}
# EuroDreams' annuities, per their rule (the API shows the monthly amount, or 0)
ANNUITY = {"eurodreams": {"6+1": 20000.0 * 12 * 30, "6+0": 2000.0 * 12 * 5}}
API = "https://apim.prd.natlot.be/api/v4/draw-games/draws"
HEADERS = {"User-Agent": "wget/1.21.4", "Accept": "*/*", "Accept-Encoding": "identity", "Connection": "Keep-Alive"}
WINDOW_DAYS = 120       # one request per window: at most 2 draws a week, well under the API's page of 62


def game_of(name):
    """The game folder name contains (lotto, euromillions, eurodreams), else None - 'vikinglotto' is not 'lotto'."""
    name = str(name or "").lower()
    if "vikinglotto" in name:
        return None
    return next((g for g in GAMES if g in name), None)


def file_path(dataPath, game):
    """data/trainingData/<game>/<game>-prizes.json for a game's training folder."""
    return os.path.join(dataPath, f"{game}-prizes.json")


def day_key(value):
    """'YYYY-MM-DD' of a date, a datetime or a day file's name ('2026-11-2')."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value)
    text = text[:-5] if text.endswith(".json") else text[:10]
    return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")


def tier_key(game, name):
    """
    (main hits, special hits or None for any) a tier pays on, from the API's
    tier name: Lotto '  5' / '5+1' (the +1 is the bonus number), EuroMillions
    '4+2' (stars), EuroDreams '6+1' / '5/' (the / is any dream number).
    None for a name that does not read.
    """
    text = str(name).strip()
    match = re.fullmatch(r"(\d)\s*(?:\+\s*(\d)|(/))?", text)
    if not match:
        return None
    mains = int(match.group(1))
    if match.group(3):
        return (mains, None)
    special = int(match.group(2)) if match.group(2) is not None else 0
    return (mains, special)


def entry_of(game, draw):
    """
    (date, entry) of one API draw: entry = {price, jackpot, tiers: [{name,
    mains, special, amount, winners, type}]}, amounts in EUR, the regular
    tiers only (the raffles left out). None when the draw has no tiers.
    """
    tiers = []
    for tier in draw.get("prizeTiers") or []:
        if str(tier.get("drawType", "")).strip().lower() not in ("regular", "common"):   # EuroMillions: 'Regular ' with a space
            continue
        key = tier_key(game, tier.get("name"))
        if key is None:
            continue
        tiers.append({"name": str(tier.get("name")).strip(), "mains": key[0], "special": key[1],
                      "amount": round(float(tier.get("shareAmount") or 0) / 100, 2),
                      "winners": int(tier.get("shareCount") or 0), "type": tier.get("prizeType")})
    if not tiers:
        return None
    jackpot = None
    for item in draw.get("jackpots") or []:
        if isinstance(item, dict) and item.get("amount"):
            jackpot = round(float(item["amount"]) / 100, 2)
            break
    if jackpot is None and draw.get("estimatedJackpot"):
        jackpot = round(float(draw["estimatedJackpot"]) / 100, 2)
    price = (draw.get("gameRuleSet") or {}).get("basePrice")
    day = datetime.fromtimestamp(draw["drawTime"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return day, {"price": round(float(price) / 100, 2) if price else None, "jackpot": jackpot, "tiers": tiers}


def load(dataPath, game):
    """{date: entry} of a game's prize file; {} when there is none or it does not read."""
    try:
        with open(file_path(dataPath, game), "r") as handle:
            table = json.load(handle)
        return table if isinstance(table, dict) else {}
    except (OSError, ValueError):
        return {}


def record(dataPath, game, draws):
    """Adds the draws' tables to the game's prize file (written whole, atomically). Returns how many were new."""
    table = load(dataPath, game)
    added = 0
    for draw in draws or []:
        try:
            parsed = entry_of(game, draw)
        except (KeyError, TypeError, ValueError):
            continue
        if parsed and (parsed[0] not in table or not table[parsed[0]].get("tiers")):
            table[parsed[0]] = parsed[1]
            added += 1
    if added:
        write(dataPath, game, table)
    return added


def write(dataPath, game, table):
    """The game's prize file, written whole to a temporary file and renamed (a reader never sees half of it)."""
    os.makedirs(dataPath, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=dataPath, suffix=".tmp")
    with os.fdopen(handle, "w") as out:
        json.dump(dict(sorted(table.items())), out, indent=1)
        out.write("\n")
    os.replace(temp, file_path(dataPath, game))


def fetch(game, start, end, get=None):
    """The API's draws of a game between two datetimes (PAYABLE ones, with their tiers)."""
    if get is None:
        import requests
        get = requests.get
    url = (f"{API}?status=PAYABLE&date-from={int(start.timestamp()) * 1000}&size=62"
           f"&date-to={int(end.timestamp()) * 1000}&game-names={GAMES[game]}")
    response = get(url=url, headers=HEADERS, timeout=(10, 120))
    return (response.json() or {}).get("draws", [])


def backfill(dataPath, game, dates, get=None, log=print, now=None):
    """
    Fetches the tables of the given draw dates (the days the game has scored
    tickets for) that the prize file lacks, one request per WINDOW_DAYS.
    A failed request costs that window only (the next run asks again).
    Returns how many tables were added.
    """
    now = now or datetime.now()
    have = load(dataPath, game)
    missing = sorted({day_key(d) for d in dates} - set(have))
    added, window = 0, []
    def flush():
        nonlocal added
        if not window:
            return
        start = datetime.strptime(window[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(window[-1], "%Y-%m-%d").replace(hour=23, minute=59, tzinfo=timezone.utc)
        try:
            draws = fetch(game, start, end, get=get)
            added += record(dataPath, game, draws)
        except Exception as e:
            log(f"prize tables {game}: {window[0]} .. {window[-1]} could not be fetched ({e.__class__.__name__}: {e})")
            window.clear()
            return
        # A date the API answered without a draw, a week old or more, is kept
        # as "no table" so it is not asked for again every run; a newer one
        # may still be on its way to PAYABLE. Only when the answer brought
        # tables: an empty one (or one this module cannot read) says nothing
        # about its dates.
        table = load(dataPath, game)
        answered = any(table.get(d, {}).get("tiers") for d in window) and any(entry_of(game, x) for x in draws)
        settled = [d for d in window if answered and d not in table and (now - datetime.strptime(d, "%Y-%m-%d")).days >= 7]
        if settled:
            write(dataPath, game, {**table, **{d: {"price": None, "jackpot": None, "tiers": [], "none": "no draw in the API"} for d in settled}})
        window.clear()
    for day in missing:
        if window and (datetime.strptime(day, "%Y-%m-%d") - datetime.strptime(window[0], "%Y-%m-%d")).days > WINDOW_DAYS:
            flush()
        window.append(day)
    flush()
    if missing:
        log(f"prize tables {game}: {len(missing)} draw(s) without a table, {added} fetched")
    return added


def typical_amounts(table):
    """{tier name: the median amount over the draws where that tier paid} - the value of a tier nobody won."""
    paid = {}
    for entry in table.values():
        for tier in entry.get("tiers") or []:
            if tier.get("amount"):
                paid.setdefault(tier["name"], []).append(float(tier["amount"]))
    return {name: statistics.median(values) for name, values in paid.items()}


def ticket_profit(game, mains, specials, real_mains, real_specials, entry, typical=None):
    """
    Net EUR of one grid (GRID) on a draw: the tier its hits reach (special hits
    against the draw's specials; Lotto has none on the ticket - its bonus
    number counts when one of the six is it) minus the draw's price. None
    when the draw has no table or no price, or when the ticket is not a full
    grid: fewer mains, or fewer specials than the game plays (the RL Ticket
    rows stored before 3 Nov 2026 without their stars, dream or viking) - such
    a draw counts on hits only.
    """
    if not entry or not entry.get("price") or not entry.get("tiers"):
        return None
    numbers, stars = GRID[game]
    mains, specials = list(mains)[:numbers], list(specials)[:stars]
    if len(mains) < numbers or len(specials) < stars:
        return None
    main_hits = len(set(map(int, mains)) & set(map(int, real_mains)))
    special_hits = len(set(map(int, mains if game == "lotto" else specials)) & set(map(int, real_specials)))
    tiers = entry["tiers"]
    tier = (next((t for t in tiers if t["mains"] == main_hits and t["special"] == special_hits), None)
            or next((t for t in tiers if t["mains"] == main_hits and t["special"] is None), None))
    payout = 0.0
    if tier is not None:
        payout = ANNUITY.get(game, {}).get(tier["name"])
        if payout is None:
            payout = float(tier.get("amount") or 0)
            if not payout:
                payout = (float(entry.get("jackpot") or 0) if tier is tiers[0]
                          else float((typical or {}).get(tier["name"], 0.0)))
    return round(payout - float(entry["price"]), 2)


def _self_check():
    assert day_key("2026-11-2.json") == "2026-11-02" and day_key("2026-10-24") == "2026-10-24"
    assert game_of("lotto") == "lotto" and game_of("vikinglotto") is None and game_of("keno") is None
    assert tier_key("lotto", "  5") == (5, 0) and tier_key("lotto", "5+1") == (5, 1)
    assert tier_key("eurodreams", "5/") == (5, None) and tier_key("euromillions", "4+2") == (4, 2) and tier_key("x", "COD") is None
    draw = {"drawTime": 1792861200000, "gameRuleSet": {"basePrice": 150}, "jackpots": [{"amount": 150000000}],
            "prizeTiers": [{"drawType": "Regular", "name": "  6", "shareAmount": 0, "shareCount": 0, "prizeType": "FIXED"},
                           {"drawType": "Regular", "name": "5+1", "shareAmount": 0, "shareCount": 0},
                           {"drawType": "Regular", "name": "  3", "shareAmount": 750, "shareCount": 70000},
                           {"drawType": "Regular", "name": "1+1", "shareAmount": 150, "shareCount": 9},
                           {"drawType": "Lucky Lotto", "name": "COD", "shareAmount": 30000, "shareCount": 500}]}
    day, entry = entry_of("lotto", draw)
    assert day == "2026-10-24" and entry["price"] == 1.5 and entry["jackpot"] == 1500000.0 and len(entry["tiers"]) == 4
    real = [1, 2, 3, 4, 5, 6]
    assert ticket_profit("lotto", [1, 2, 3, 40, 41, 42], [], real, [7], entry) == 6.0          # 3 hits: 7.50 - 1.50
    assert ticket_profit("lotto", [1, 40, 41, 42, 43, 7], [], real, [7], entry) == 0.0        # 1+1: the stake back
    assert ticket_profit("lotto", [1, 40, 41, 42, 43, 44], [], real, [7], entry) == -1.5       # no tier
    assert ticket_profit("lotto", real, [], real, [7], entry) == 1500000.0 - 1.5               # nobody won: the jackpot
    assert ticket_profit("lotto", [1, 2, 3, 4, 5, 7], [], real, [7], entry, {"5+1": 60000.0}) == 59998.5   # the median
    assert ticket_profit("lotto", real, [], real, [7], None) is None
    assert ticket_profit("lotto", [1, 2, 3, 40, 41, 42, 4], [], real, [7], entry) == 6.0       # a 7th value is not played
    assert ticket_profit("lotto", [1, 2, 3], [], real, [7], entry) is None
    dreams = {"price": 2.5, "jackpot": None, "tiers": [{"name": "6+1", "mains": 6, "special": 1, "amount": 0.0},
                                                       {"name": "5/", "mains": 5, "special": None, "amount": 381.3}]}
    assert ticket_profit("eurodreams", real, [3], real, [3], dreams) == 7200000.0 - 2.5
    assert ticket_profit("eurodreams", real, [], real, [3], dreams) is None                 # no dream number: hits only
    assert ticket_profit("eurodreams", [1, 2, 3, 4, 5, 40], [2], real, [3], dreams) == 378.8   # 5 and any dream
    assert typical_amounts({"a": {"tiers": [{"name": "5+1", "amount": 10}]}, "b": {"tiers": [{"name": "5+1", "amount": 30}, {"name": "6", "amount": 0}]}}) == {"5+1": 20.0}

    folder = tempfile.mkdtemp()
    class Response:
        def __init__(self, draws): self.draws = draws
        def json(self): return {"draws": self.draws}
    asked = []
    def get(url, headers, timeout):
        asked.append(url)
        return Response([draw])
    assert backfill(folder, "lotto", ["2026-10-24", "2026-10-24", "2026-10-20"], get=get, log=lambda m: None,
                    now=datetime(2026, 11, 3)) == 1 and len(asked) == 1
    assert load(folder, "lotto")["2026-10-20"]["tiers"] == []                                            # no draw: asked once
    assert backfill(folder, "lotto", ["2026-10-24", "2026-10-20"], get=get, log=lambda m: None) == 0 and len(asked) == 1
    assert backfill(folder, "lotto", ["2026-11-01"], get=get, log=lambda m: None, now=datetime(2026, 11, 3)) == 0
    assert "2026-11-01" not in load(folder, "lotto")                                                     # too new to give up on
    assert load(folder, "lotto")["2026-10-24"]["price"] == 1.5
    def broken(url, headers, timeout):
        raise OSError("down")
    assert backfill(folder, "lotto", ["2025-01-04", "2026-10-24"], get=broken, log=lambda m: None) == 0
    assert record(folder, "lotto", [{"drawTime": 1, "prizeTiers": []}]) == 0
    def empty(url, headers, timeout):
        return Response([])
    assert backfill(folder, "lotto", ["2025-03-01"], get=empty, log=lambda m: None, now=datetime(2026, 11, 3)) == 0
    assert "2025-03-01" not in load(folder, "lotto")                                                     # an empty answer marks nothing
    spaced = dict(draw, prizeTiers=[dict(t, drawType="Regular ") for t in draw["prizeTiers"]])
    assert len(entry_of("euromillions", spaced)[1]["tiers"]) == 4
    print("PrizeTables self-check OK")


if __name__ == "__main__":
    _self_check()
