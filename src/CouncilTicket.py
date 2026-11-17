"""
The Council rows (docs/roadmap.md item 12, 17 Nov 2026): tracked rows whose ticket comes from the local LLMs of the
Council (src/llmCouncil). This module is the part without a network: the rows' names, what a valid ticket of each
game is, the prompts, reading an answer, and the summary an LLM is given. CouncilRows.py (the job) asks the models.

Two kinds, each one row per Council member and one for the head's combination of the members' tickets:

  record   the LLM reads what the other rows played on the recent draws, how many numbers each had right, and their
           tickets for the coming draw. A meta-learner done by an LLM; its comparison is the weighted ensemble and the
           RL ticket row. The risk to read for: it follows whichever row did well lately.
  draws    the LLM reads only the recent draws. On draws that test fair this is a biased random generator: a control,
           and its favourite numbers show as repetition.

The rows are scored like every other row and are no input of another row: the ensemble rows, the RL ticket row, the
tuners that pick ensemble members and the combination search leave them out (is_council). They exist from the day
they were first made: never built inside a history rebuild, and a rebuild keeps the ones a day file holds (kept_rows).

A valid ticket or none: an answer that is not a ticket of the game is asked again (ATTEMPTS in all) and then the row
is absent that day, as a foundation row is.

Run `python3 -m src.CouncilTicket` for the self-check.
"""

import hashlib
import json
import os
import re
from datetime import datetime

PREFIX = "Council"
KINDS = ("record", "draws")
COMBINED = "combined"
ATTEMPTS = 3

# What the games' tickets are. mains: how many numbers, lo..hi; distinct: a set game (False: positional digits that
# may repeat); special: (count, lo, hi, what it is called) for the trailing columns; ask: how many main numbers the
# LLM is asked for (lotto: its 6 and the 3 extra numbers of the multi-pick; keno: the ranking of 20, whose first
# 5-10 are the tickets). The ranges are those of HyperoptStatistics.GAME_CONFIG and Helpers.SPECIAL_UNIQUE_LABELS.
GAMES = {
    "euromillions": {"label": "EuroMillions", "mains": 5, "lo": 1, "hi": 50, "distinct": True, "special": (2, 1, 12, "stars")},
    "lotto":        {"label": "Lotto", "mains": 6, "lo": 1, "hi": 45, "distinct": True, "special": None, "ask": 9, "drawn_mains": 6},
    "eurodreams":   {"label": "EuroDreams", "mains": 6, "lo": 1, "hi": 40, "distinct": True, "special": (1, 1, 5, "dream number")},
    "jokerplus":    {"label": "Joker+", "mains": 6, "lo": 0, "hi": 9, "distinct": False, "special": (1, 0, 11, "zodiac sign as a number 0-11")},
    "keno":         {"label": "Keno", "mains": 20, "lo": 1, "hi": 70, "distinct": True, "special": None},
    "pick3":        {"label": "Pick3", "mains": 3, "lo": 0, "hi": 9, "distinct": False, "special": None},
    "vikinglotto":  {"label": "Vikinglotto", "mains": 6, "lo": 1, "hi": 48, "distinct": True, "special": (1, 1, 8, "Viking number")},
}
KENO_SIZES = (5, 6, 7, 8, 9, 10)

# How much an LLM is given: the context of a small local model is a few thousand tokens.
RECENT_DRAWS = {"record": 8, "draws": 30}
RECENT_DRAWS_KENO = {"record": 5, "draws": 10}
RECORD_ROWS = 10


def row_name(kind, member):
    """The row's name in the day files: "Council (record): Qwen2.5", "Council (draws): combined"."""
    return f"{PREFIX} ({kind}): {member}"


def is_council(name):
    return isinstance(name, str) and any(name.startswith(f"{PREFIX} ({kind}): ") for kind in KINDS)


def without_council(rows):
    """The rows that are not Council rows: what another row may read."""
    return [r for r in rows if not (isinstance(r, dict) and is_council(r.get("name")))]


def kept_rows(day_file):
    """The Council rows a day file holds: what a rebuild of that day puts back (they cannot be made again)."""
    try:
        with open(day_file, "r") as handle:
            rows = json.load(handle).get("newPrediction") or []
    except Exception:
        return []
    return [r for r in rows if isinstance(r, dict) and is_council(r.get("name"))]


def put_back(rows, kept):
    """Adds the kept Council rows to a rebuilt day's rows (none twice). Returns how many were put back."""
    have = {r.get("name") for r in rows if isinstance(r, dict)}
    fresh = [r for r in (kept or []) if r.get("name") not in have]
    rows.extend(fresh)
    return len(fresh)


# --- a ticket ---------------------------------------------------------------------------------------------------
def asked(game):
    spec = GAMES[game]
    return spec.get("ask", spec["mains"])


def validate(game, numbers):
    """
    (ticket, None) or (None, why not). `numbers` are the integers of the answer in order: the main numbers asked for,
    then the special ones. The ticket is what the day file stores as the row's first prediction, except for lotto,
    where the first 6 are the ticket and the other 3 the multi-pick (as_row splits them).
    """
    spec = GAMES[game]
    need = asked(game)
    special = spec["special"]
    total = need + (special[0] if special else 0)
    if len(numbers) != total:
        return None, f"{len(numbers)} numbers where {total} were asked"
    mains, extra = numbers[:need], numbers[need:]
    if any(not (spec["lo"] <= n <= spec["hi"]) for n in mains):
        return None, f"a number outside {spec['lo']}-{spec['hi']}"
    if spec["distinct"] and len(set(mains)) != len(mains):
        return None, "a number twice"
    if special:
        count, lo, hi, what = special
        if any(not (lo <= n <= hi) for n in extra):
            return None, f"a {what.split(' as ')[0]} outside {lo}-{hi}"
        if count > 1 and len(set(extra)) != len(extra):
            return None, f"the same {what.split(' as ')[0]} twice"
    return list(mains) + list(extra), None


def as_row(game, name, ticket, meta, keno_sizes=KENO_SIZES):
    """The row as the day file holds it. keno: the ranking and its first 5-10 as the tickets, like the other rows."""
    row = {"name": name, "predictions": [list(ticket)]}
    if game == "lotto":
        row["predictions"] = [list(ticket[:6])]
        row["multiPick"] = list(ticket[6:9])
    elif game == "keno":
        row["predictions"] += [list(ticket[:size]) for size in keno_sizes]
    row["council"] = meta
    return row


ANSWER_LINE = re.compile(r"TICKET\s*[:=]\s*(.*)", re.IGNORECASE)
_INTEGERS = re.compile(r"(?<![\d.])\d{1,2}(?![\d.])")


def _line(text):
    """The text after the last "TICKET:" of an answer, up to the end of that line; None without one."""
    found = ANSWER_LINE.findall(str(text or ""))
    if not found:
        return None
    # the line only: what follows it (an explanation the model adds anyway) is not the ticket
    return found[-1].splitlines()[0] if found[-1] else ""


def parse(text):
    """The integers after the last "TICKET:" of an answer, or None when it has no such line."""
    line = _line(text)
    return None if line is None else [int(n) for n in _INTEGERS.findall(line)]


def read_answer(game, text):
    """
    (ticket, None) or (None, why not) for one answer. One leniency, for the set games only: an answer that lists more
    main numbers than asked is cut to the first ones - the question asks for the most likely first, and the small
    models tend to go on listing (17 Nov 2026). The special numbers are never cut, a positional game's digits never
    (more digits there is several tickets, not one), and a number twice among those kept is still no ticket.
    """
    line = _line(text)
    if line is None:
        return None, "no TICKET line"
    spec = GAMES[game]
    need = asked(game)
    parts = [[int(n) for n in _INTEGERS.findall(part)] for part in line.split("+")]
    if spec["special"]:
        if len(parts) > 2:
            return None, "more than one ticket"
        if len(parts) == 2:
            mains, extra = parts
            if spec["distinct"] and len(mains) > need:
                mains = mains[:need]
            return validate(game, mains + extra)
        return validate(game, parts[0])
    numbers = [n for part in parts for n in part]
    if spec["distinct"] and len(numbers) > need:
        numbers = numbers[:need]
    return validate(game, numbers)


def tally(game, tickets):
    """The members' tickets as the head reads them: what was chosen and by how many, never the tickets as lines."""
    spec = GAMES[game]
    need = asked(game)

    def counted(values):
        count = {}
        for value in values:
            count[value] = count.get(value, 0) + 1
        return ", ".join(f"{n} ({k}x)" for n, k in sorted(count.items(), key=lambda item: (-item[1], item[0])))

    if spec["distinct"]:
        lines = ["Numbers the advisers chose (and how many of them chose it): " + counted([n for _, t in tickets for n in t[:need]])]
    else:
        lines = [f"Position {i + 1}, the advisers chose: " + counted([t[i] for _, t in tickets]) for i in range(need)]
    if spec["special"]:
        lines.append(f"For the {spec['special'][3].split(' as ')[0]} the advisers chose: " + counted([n for _, t in tickets for n in t[need:]]))
    return lines


# --- the summary ------------------------------------------------------------------------------------------------
def day_files(folder):
    """A game's day files, oldest first: [(date, path)]."""
    out = []
    try:
        names = os.listdir(folder)
    except OSError:
        return out
    for name in names:
        try:
            out.append((datetime.strptime(name[:-5], "%Y-%m-%d"), os.path.join(folder, name)))
        except ValueError:
            continue
    return sorted(out)


def _drawn(game, real):
    """(main numbers, special numbers) of a drawn result as the day file holds it."""
    spec = GAMES[game]
    real = [int(n) for n in (real or [])]
    if game == "lotto":
        return real[:spec["drawn_mains"]], []          # the 7th is the bonus: not part of a ticket
    count = spec["special"][0] if spec["special"] else 0
    return (real[:-count], real[-count:]) if count else (real, [])


def _played(game, row):
    """(main numbers, special numbers) a row played: its first prediction (keno: the first 10 of its ranking)."""
    try:
        ticket = [int(n) for n in row["predictions"][0]]
    except Exception:
        return None
    spec = GAMES[game]
    if game == "keno":
        return ticket[:10], []
    count = spec["special"][0] if spec["special"] else 0
    return (ticket[:-count], ticket[-count:]) if count else (ticket, [])


def hits(game, row, real):
    """How many main numbers a row had right on a draw: in the drawn set, or on the right position."""
    played = _played(game, row)
    if played is None:
        return None
    mains, _ = _drawn(game, real)
    if GAMES[game]["distinct"]:
        return len(set(played[0]) & set(mains))
    return sum(1 for a, b in zip(played[0], mains) if a == b)


def _numbers(values):
    return " ".join(str(v) for v in values)


def _ticket_text(game, mains, special):
    return _numbers(mains) + (f" + {_numbers(special)}" if special else "")


def summary(game, days, ranked_names=None, kind="record"):
    """
    What the LLM reads. `days`: the game's day files as dicts, oldest first, the last being the day whose
    newPrediction holds the tickets for the coming draw. Returns {"draws": [...lines], "rows": [...lines]}; rows is
    empty for the draws kind. ranked_names: the report's ranking, best first - the first RECORD_ROWS of them that
    play today are shown (without it, the first rows of the day file).
    """
    keep = (RECENT_DRAWS_KENO if game == "keno" else RECENT_DRAWS)[kind]
    recent = [d for d in days if d.get("realResult")][-keep:]
    draws = []
    for d in recent:
        mains, special = _drawn(game, d["realResult"])
        draws.append(_ticket_text(game, mains, special))
    out = {"draws": draws, "rows": []}
    if kind != "record" or not days:
        return out
    today = [r for r in (days[-1].get("newPrediction") or []) if isinstance(r, dict) and not is_council(r.get("name"))
             and " · " not in str(r.get("name")) and _played(game, r)]
    by_name = {r["name"]: r for r in today}
    order = [n for n in (ranked_names or []) if n in by_name] + [r["name"] for r in today if r["name"] not in (ranked_names or [])]
    for name in order[:RECORD_ROWS]:
        right = []
        for d in recent:
            was = next((r for r in (d.get("currentPrediction") or []) if isinstance(r, dict) and r.get("name") == name), None)
            h = hits(game, was, d["realResult"]) if was else None
            right.append("-" if h is None else str(h))
        mains, special = _played(game, by_name[name])
        out["rows"].append(f"{name} | right on those draws: {' '.join(right)} | plays now: {_ticket_text(game, mains, special)}")
    return out


# --- the prompts ------------------------------------------------------------------------------------------------
SYSTEM = ("You choose lottery numbers. You answer with one line and nothing else, in exactly the format asked. "
          "No explanation.")


def _format(game):
    spec = GAMES[game]
    need = asked(game)
    special = spec["special"]
    if spec["distinct"]:
        what = f"{need} different numbers from {spec['lo']} to {spec['hi']}, the one you find most likely first"
    else:
        what = f"{need} digits from {spec['lo']} to {spec['hi']}, one per position in the order drawn (a digit may repeat)"
    if special:
        count, lo, hi, name = special
        what += (f", then a plus sign and {count} different {name} from {lo} to {hi}" if count > 1
                 else f", then a plus sign and 1 {name}" + ("" if " as a number" in name else f" from {lo} to {hi}"))
    example = " ".join(["N"] * need) + (" + " + " ".join(["S"] * special[0]) if special else "")
    total = need + (special[0] if special else 0)
    what += f". That is exactly {total} numbers in all, no more and no fewer"
    return what, example


def prompt(game, kind, info):
    """The question one member is asked: the game, what it reads, the answer's format."""
    spec = GAMES[game]
    what, example = _format(game)
    lines = [f"Game: {spec['label']}. Choose a ticket for the next draw.", ""]
    lines.append(f"The last {len(info['draws'])} draws, oldest first" + (" (numbers + " + spec["special"][3].split(" as ")[0] + ")" if spec["special"] else "") + ":")
    lines += info["draws"]
    if kind == "record":
        lines += ["", "Prediction models, the best of the long record first. For each: how many numbers it had right on "
                      "each of those draws (oldest first, - when it did not play), and what it plays for the next draw:"]
        lines += info["rows"]
    lines += ["", f"Answer with one line: TICKET: followed by {what}.", f"Format: TICKET: {example}"]
    return "\n".join(lines)


def head_prompt(game, kind, info, tickets):
    """
    The question the head is asked: the same, with what the members chose to combine. tickets: [(member, ticket)].
    The members' tickets are given as a tally: given as lines, the head answered with all of them one after the
    other (17 Nov 2026).
    """
    base = prompt(game, kind, info).split("\n\nAnswer with one line")[0]
    what, example = _format(game)
    lines = [base, "", f"{len(tickets)} advisers who read the same each chose a ticket."] + tally(game, tickets)
    lines += ["", "Choose the final ticket from these.", f"Answer with one line: TICKET: followed by {what}.", f"Format: TICKET: {example}"]
    return "\n".join(lines)


def sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def seed_for(game, day, name, attempt):
    """The seed sent with a request when the Council's config sets none: fixed per game, day, row and attempt."""
    return int(hashlib.sha256(f"{game}|{day}|{name}|{attempt}".encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31 - 1)


def _self_check():
    assert row_name("record", "Qwen2.5") == "Council (record): Qwen2.5" and is_council("Council (draws): combined")
    assert not is_council("Markov Model") and not is_council("Council") and not is_council(None)
    assert without_council([{"name": "Council (draws): a"}, {"name": "Markov Model"}]) == [{"name": "Markov Model"}]

    # reading an answer
    assert parse("TICKET: 1 2 3") == [1, 2, 3] and parse("I think\nticket = 4, 5, 6\nBecause 77 came 3 times") == [4, 5, 6]
    assert parse("1 2 3") is None and parse("TICKET: 3 14 22 + 2 11") == [3, 14, 22, 2, 11]
    assert parse("TICKET: 1 2 3\nTICKET: 7 8 9") == [7, 8, 9] and parse("TICKET: 1.5 200 7") == [7]
    ok = lambda game, text: read_answer(game, text)
    assert ok("pick3", "TICKET: 5 0 5") == ([5, 0, 5], None)
    assert ok("pick3", "TICKET: 5 0")[1] == "2 numbers where 3 were asked" and ok("pick3", "TICKET: 5 0 12")[1] == "a number outside 0-9"
    assert ok("pick3", "the answer is 5 0 5")[1] == "no TICKET line"
    assert ok("euromillions", "TICKET: 1 2 3 4 50 + 1 12")[0] == [1, 2, 3, 4, 50, 1, 12]
    assert ok("euromillions", "TICKET: 1 2 3 4 4 + 1 12")[1] == "a number twice"
    assert ok("euromillions", "TICKET: 1 2 3 4 5 + 3 3")[1] == "the same stars twice"
    assert ok("euromillions", "TICKET: 1 2 3 4 5 + 3 13")[1] == "a stars outside 1-12"
    assert ok("eurodreams", "TICKET: 1 2 3 4 5 40 + 5")[0] and ok("eurodreams", "TICKET: 1 2 3 4 5 41 + 5")[1]
    assert ok("vikinglotto", "TICKET: 1 2 3 4 5 48 + 8")[0] and ok("vikinglotto", "TICKET: 1 2 3 4 5 48 + 9")[1]
    assert ok("jokerplus", "TICKET: 3 3 3 0 9 1 + 11")[0] == [3, 3, 3, 0, 9, 1, 11] and ok("jokerplus", "TICKET: 3 3 3 0 9 1 + 12")[1]
    assert ok("lotto", "TICKET: 1 2 3 4 5 6 7 8 9")[0] and ok("lotto", "TICKET: 1 2 3 4 5 6")[1] and ok("lotto", "TICKET: 1 2 3 4 5 6 7 8 46")[1]
    keno = list(range(1, 21))
    assert ok("keno", "TICKET: " + " ".join(map(str, keno)))[0] == keno and ok("keno", "TICKET: " + " ".join(map(str, keno[:19] + [1])))[1]

    # the rows as the day file holds them
    meta = {"kind": "draws"}
    assert as_row("pick3", "n", [1, 2, 3], meta) == {"name": "n", "predictions": [[1, 2, 3]], "council": meta}
    lotto = as_row("lotto", "n", [1, 2, 3, 4, 5, 6, 7, 8, 9], meta)
    assert lotto["predictions"] == [[1, 2, 3, 4, 5, 6]] and lotto["multiPick"] == [7, 8, 9]
    kenoRow = as_row("keno", "n", keno, meta, keno_sizes=(5, 10))
    assert kenoRow["predictions"] == [keno, keno[:5], keno[:10]]

    # hits: a set game, a positional game, lotto without its bonus, the specials apart
    assert hits("lotto", {"predictions": [[1, 2, 3, 40, 41, 42]]}, [1, 2, 3, 4, 5, 6, 40]) == 3
    assert hits("pick3", {"predictions": [[5, 0, 8]]}, [8, 0, 5]) == 1
    assert hits("euromillions", {"predictions": [[1, 2, 3, 4, 5, 1, 2]]}, [1, 2, 30, 31, 32, 3, 4]) == 2
    assert hits("keno", {"predictions": [list(range(1, 21))]}, list(range(8, 28))) == 3
    assert hits("pick3", {"predictions": []}, [1, 2, 3]) is None

    # the summary and the prompts
    days = [{"realResult": [1, 2, 3], "currentPrediction": [{"name": "A Model", "predictions": [[1, 2, 9]]}], "newPrediction": []},
            {"realResult": [4, 5, 6], "currentPrediction": [{"name": "A Model", "predictions": [[9, 9, 9]]}, {"name": "B Model", "predictions": [[4, 5, 6]]}],
             "newPrediction": [{"name": "A Model", "predictions": [[7, 7, 7]]}, {"name": "B Model", "predictions": [[1, 1, 1]]},
                               {"name": "A Model · Sampled", "predictions": [[2, 2, 2]]}, {"name": "Council (draws): x", "predictions": [[3, 3, 3]]}]}]
    info = summary("pick3", days, ranked_names=["B Model", "A Model"])
    assert info["draws"] == ["1 2 3", "4 5 6"], info
    assert info["rows"] == ["B Model | right on those draws: - 3 | plays now: 1 1 1", "A Model | right on those draws: 2 0 | plays now: 7 7 7"], info
    assert summary("pick3", days, kind="draws")["rows"] == []
    text = prompt("pick3", "record", info)
    assert "Game: Pick3" in text and "B Model | right" in text and text.endswith("Format: TICKET: N N N")
    assert "B Model" not in prompt("pick3", "draws", summary("pick3", days, kind="draws"))
    assert prompt("euromillions", "draws", {"draws": [], "rows": []}).endswith("Format: TICKET: N N N N N + S S")
    assert "2 different stars from 1 to 12" in prompt("euromillions", "draws", {"draws": [], "rows": []})
    assert "1 zodiac sign as a number 0-11" in prompt("jokerplus", "draws", {"draws": [], "rows": []})
    head = head_prompt("euromillions", "draws", {"draws": ["1 2 3 4 5 + 1 2"], "rows": []}, [("a", [1, 2, 3, 4, 5, 1, 2]), ("b", [6, 7, 8, 9, 10, 3, 4])])
    assert "Numbers the advisers chose (and how many of them chose it): 1 (1x), 2 (1x)" in head and "For the stars the advisers chose: 1 (1x)" in head
    assert head.count("Answer with one line") == 1 and "a:" not in head and "2 advisers" in head
    assert tally("pick3", [("a", [2, 9, 9]), ("b", [4, 9, 2])]) == ["Position 1, the advisers chose: 2 (1x), 4 (1x)", "Position 2, the advisers chose: 9 (2x)", "Position 3, the advisers chose: 2 (1x), 9 (1x)"]
    # more main numbers than asked: a set game's are cut to the first, a positional game's never, the specials never
    assert ok("lotto", "TICKET: 1 2 3 4 5 6 7 8 9 10 11")[0] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert ok("lotto", "TICKET: 1 2 3 4 5 6 7 8 8 10 11")[1] == "a number twice"
    assert ok("euromillions", "TICKET: 1 2 3 4 5 6 7 + 1 2")[0] == [1, 2, 3, 4, 5, 1, 2]
    assert ok("euromillions", "TICKET: 1 2 3 4 5 + 1 2 3")[1] and ok("euromillions", "TICKET: 1 2 3 4 5 + 1 11 + 9 16 24 33 48 + 7 11")[1] == "more than one ticket"
    assert ok("euromillions", "TICKET: 1 2 3 4 5 6 1 2")[1] and ok("pick3", "TICKET: 2 9 9 4 8 2")[1] == "6 numbers where 3 were asked"
    assert ok("jokerplus", "TICKET: 3 3 3 0 9 1 4 + 11")[1]
    # every game's own example is a ticket of that game once its letters are numbers
    for game, spec in GAMES.items():
        example = _format(game)[1]
        numbers = iter(range(spec["lo"] if spec["distinct"] else 1, 99))
        filled = re.sub(r"[NS]", lambda m: str(next(numbers)) if m.group(0) == "N" else "1", example)
        if spec["special"] and spec["special"][0] > 1:
            filled = filled.rsplit("+", 1)[0] + "+ 1 2"
        assert read_answer(game, "TICKET: " + filled)[0], (game, filled, read_answer(game, "TICKET: " + filled))

    # a rebuild keeps the Council rows
    rows = [{"name": "Markov Model"}]
    assert put_back(rows, [{"name": "Council (draws): x"}, {"name": "Markov Model"}]) == 1 and len(rows) == 2
    assert kept_rows("/nonexistent/file.json") == []
    assert seed_for("pick3", "2026-11-17", "n", 1) == seed_for("pick3", "2026-11-17", "n", 1) != seed_for("pick3", "2026-11-17", "n", 2)
    print("CouncilTicket: self-check passed")


if __name__ == "__main__":
    _self_check()
