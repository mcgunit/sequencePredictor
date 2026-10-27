"""
The lockbox period (README "Lockbox period", roadmap item 6 / Q0).

A chronological period of draws that model selection never sees: the
meta-learner trainer and the quantum tuner drop its days from their tables
before splitting and fitting, the row tuners withhold them from every score
that decides what is served (below), and the frozen artifacts are scored on
it once, on request (TrainMetaLearner.py --lockbox-report). Repeatedly
inspecting the same holdout and changing the design afterwards turns the
holdout into development data; the lockbox is the period that does not get
that treatment.

Declared by hand in lockbox.json at the repository root - no file, no lockbox:

    {"from": "2026-07-01", "to": "2026-09-26", "declared": "2026-09-27",
     "note": "designs frozen after the September 2026 tuning changes"}

Dates are inclusive. The period should end at or before the declaration: a
lockbox that extends into the future would lock out the newest draws from
every weekly retrain, and the daily predictor is never subject to it - it
predicts; the lockbox is about what the trainers learn from.

The row tuners (HyperoptStatistics.py, HyperoptBoost.py, HyperoptEnsemble.py,
HyperoptRLTicket.py - its ticket and position studies) score on the newest
draws, and their windows reach back into the lockbox (26 Oct 2026: 22 of the
90 tuning draws of euromillions, lotto and eurodreams, 12-13 of vikinglotto
and the week games; 25-26 of the ensemble's ~100 days, 12 of vikinglotto's
48; 6 of vikinglotto's 40 RL ticket days). Until 26 Oct 2026 they only
warned (Statistics, Boost) or did not look at all (Ensemble, RLTicket), so
the lockbox evaluation of the meta-learners rested on rows tuned inside it.
Since then they withhold the lockbox days from SELECTION:
trial scores, the gate's re-scores of the served parameters and the defaults
(they go through the same objective), the standard error any margin is taken
in, modelScores and the keno subset day table are all computed on the
window's days outside the lockbox. FITTING on history that contains lockbox
draws stays allowed - the daily predictor does it too, and the walk-forwards
still step through those days (a backtested row is fitted on everything
before it, the RL policies warm-start through them) without scoring them.
The window is not lengthened to make up the withheld days: fewer scored
days, a wider standard error, and the overlap ages out as the window moves.
The withheld count is recorded in each of these tuners' tuningGate records
(lockbox_days_withheld, out of the record's window_days). The quantum
tuner's records carry it too, but there it counts score-table days, taken
out before the 75/25 split, so its holdout holds none of them
(lockbox_withheld_from "table"). A malformed lockbox.json is a warning in
these tuners (load_or_warn) - the run then is NOT lockbox-safe and says so -
while the trainer and the quantum tuner stop with LOCKBOX ERROR.

    python3 -m src.Lockbox         # self-check
"""

import json
import os
from datetime import date, datetime

FILE_NAME = "lockbox.json"


def _to_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()[:10]
    return datetime.strptime(text, "%Y-%m-%d").date()


def load(root=None):
    """The declared lockbox as {"from": date, "to": date, ...} or None."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, FILE_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{FILE_NAME}: not valid JSON ({exc})") from exc
    return parse(raw)


def load_or_warn(root=None, out=print):
    """
    load() for the row tuners and TuningControls: a malformed declaration is
    said once, as a warning, and the run goes on without a lockbox - which
    makes its tuning NOT lockbox-safe, and the warning says that too (until 26
    Oct 2026 it only said the file was ignored). The trainer and the quantum
    tuner stop instead (LOCKBOX ERROR): their tables are the lockbox's own.
    """
    try:
        return load(root)
    except ValueError as exc:
        out(f"LOCKBOX WARNING: {exc} - ignoring lockbox.json this run: this tuning is NOT lockbox-safe "
            "(every day of the window is scored, lockbox days included)")
        return None


def parse(raw):
    """Validate a declaration; raises ValueError on a malformed one."""
    if not raw:
        return None
    try:
        start, end = _to_date(raw["from"]), _to_date(raw["to"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"{FILE_NAME}: 'from' and 'to' must be YYYY-MM-DD dates ({exc})") from exc
    if end < start:
        raise ValueError(f"{FILE_NAME}: 'to' ({end}) lies before 'from' ({start})")
    return {"from": start, "to": end, "declared": raw.get("declared"), "note": raw.get("note")}


def covers(lockbox, when):
    """Whether a draw date falls inside the lockbox (inclusive)."""
    if not lockbox or when is None:
        return False
    when = _to_date(when)
    return lockbox["from"] <= when <= lockbox["to"]


def split_rows(rows, dates, lockbox):
    """
    Backtester rows carry "index", their position in the loader's numbers;
    `dates` are the loader's dates in the same order. Returns (kept, locked):
    rows outside and inside the lockbox, both in their original order. Rows
    whose index has no date are kept - nothing is silently dropped.
    """
    if not lockbox:
        return list(rows), []
    kept, locked = [], []
    for row in rows:
        index = row.get("index")
        when = dates[index] if isinstance(index, int) and 0 <= index < len(dates) else None
        (locked if covers(lockbox, when) else kept).append(row)
    return kept, locked


def split_by_date(items, lockbox, date_of=lambda item: item[0]):
    """
    (kept, locked) like split_rows, for items that carry their own date - the
    day-JSON tuners' (date, rows, realResult) entries by default. An item
    without a date is kept.
    """
    if not lockbox:
        return list(items), []
    kept, locked = [], []
    for item in items:
        (locked if covers(lockbox, date_of(item)) else kept).append(item)
    return kept, locked


def window_overlap(dates, window_days, lockbox):
    """How many of the newest `window_days` dates fall inside the lockbox."""
    if not lockbox or not dates or window_days <= 0:
        return 0
    return sum(1 for when in list(dates)[-int(window_days):] if covers(lockbox, when))


def window_counts(dates, window_days, lockbox):
    """
    (withheld, scored) for a walk-forward over the newest `window_days` dates
    (all of them when the history is shorter): the days inside the lockbox
    and the days left to score.
    """
    days = min(max(0, int(window_days)), len(dates or []))
    withheld = window_overlap(dates, days, lockbox)
    return withheld, days - withheld


def describe_withheld(game, lockbox, withheld, scored):
    """The row tuners' log line, in the quantum tuner's words."""
    return f"{game}: {describe(lockbox)} - {withheld} day(s) withheld from tuning, {scored} scored"


def describe(lockbox):
    if not lockbox:
        return "no lockbox declared"
    text = f"lockbox {lockbox['from']} to {lockbox['to']}"
    if lockbox.get("declared"):
        text += f" (declared {lockbox['declared']})"
    return text


def as_json(lockbox):
    if not lockbox:
        return None
    return {"from": lockbox["from"].isoformat(), "to": lockbox["to"].isoformat(),
            "declared": lockbox.get("declared"), "note": lockbox.get("note")}


if __name__ == "__main__":
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    box = parse({"from": "2026-07-01", "to": "2026-09-26", "declared": "2026-09-27"})
    check(box["from"] == date(2026, 7, 1) and box["to"] == date(2026, 9, 26), "dates parse")
    check(covers(box, "2026-07-01") and covers(box, date(2026, 9, 26)) and covers(box, datetime(2026, 8, 15, 20, 0)),
          "inclusive on both ends, datetime accepted")
    check(not covers(box, "2026-06-30") and not covers(box, "2026-09-27") and not covers(None, "2026-08-01") and not covers(box, None),
          "outside, no lockbox, no date")
    for bad in ({"from": "2026-09-30", "to": "2026-09-01"}, {"from": "soon"}, {"to": "2026-09-01"}):
        try:
            parse(bad)
            failures.append(f"malformed declaration accepted: {bad}")
        except ValueError:
            pass
    check(parse(None) is None and parse({}) is None, "an empty declaration is no lockbox")

    dates = [datetime(2026, 6, 1) .__class__(2026, 6, 1 + i) if i < 29 else datetime(2026, 7, i - 28) for i in range(60)]
    rows = [{"index": i, "v": i} for i in range(60)] + [{"index": 999}, {"v": "no index"}]
    kept, locked = split_rows(rows, dates, box)
    check(len(locked) == 31 and all(covers(box, dates[r["index"]]) for r in locked), f"July rows locked: {len(locked)}")
    check(len(kept) == 31 and kept[-2:] == [{"index": 999}, {"v": "no index"}], "June rows and dateless rows kept, order preserved")
    check(split_rows(rows, dates, None) == (rows, []), "no lockbox: everything kept")
    check(window_overlap(dates, 10, box) == 10 and window_overlap(dates, 40, box) == 31 and window_overlap(dates, 10, None) == 0,
          "overlap counts the newest window's locked dates")
    check(describe(box).startswith("lockbox 2026-07-01 to 2026-09-26") and describe(None) == "no lockbox declared", describe(box))
    json.dumps(as_json(box))

    # the row tuners' helpers (26 Oct 2026): the window's withheld and scored days, and day-JSON entries by date
    check(window_counts(dates, 40, box) == (31, 9) and window_counts(dates, 10, box) == (10, 0)
          and window_counts(dates, 90, box) == (31, 29) and window_counts(dates, 40, None) == (0, 40),
          "window_counts: newest window, capped at the history, nothing withheld without a lockbox")
    entries = [(dates[i], f"rows {i}", [i]) for i in range(60)] + [(None, "undated", [])]
    kept, locked = split_by_date(entries, box)
    check(len(locked) == 31 and len(kept) == 30 and kept[-1][1] == "undated" and [e[1] for e in kept[:2]] == ["rows 0", "rows 1"],
          "split_by_date: July entries locked, the rest and an undated one kept, in order")
    check(split_by_date(entries, None) == (entries, []), "split_by_date without a lockbox keeps everything")
    check(split_by_date([{"day": "2026-07-04"}, {"day": "2026-06-04"}], box, date_of=lambda e: e["day"]) == ([{"day": "2026-06-04"}], [{"day": "2026-07-04"}]),
          "split_by_date with its own date accessor")
    check(describe_withheld("lotto", box, 22, 68).startswith("lotto: lockbox 2026-07-01 to 2026-09-26")
          and describe_withheld("lotto", box, 22, 68).endswith("- 22 day(s) withheld from tuning, 68 scored"), describe_withheld("lotto", box, 22, 68))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        said = []
        check(load_or_warn(tmp, out=said.append) is None and not said, "load_or_warn: no file, no lockbox, nothing said")
        with open(os.path.join(tmp, FILE_NAME), "w") as handle:
            handle.write('{"from": "2026-07-01", "to": "2026-09-26"}x')
        check(load_or_warn(tmp, out=said.append) is None and len(said) == 1 and "NOT lockbox-safe" in said[0]
              and said[0].startswith("LOCKBOX WARNING"), f"load_or_warn: a malformed file is one warning that says so: {said}")
        with open(os.path.join(tmp, FILE_NAME), "w") as handle:
            json.dump({"from": "2026-07-01", "to": "2026-09-26"}, handle)
        check(load_or_warn(tmp, out=said.append) == load(tmp) and len(said) == 1, "load_or_warn: a good file loads silently")

    # The repository's own declaration, if any: a malformed lockbox.json stops
    # the trainer and the quantum tuner with LOCKBOX ERROR before any work, so
    # it must fail here, in npm test, and not on a Saturday morning (a stray
    # byte after the closing brace did exactly that on 27 Sept 2026).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        declared = load(repo_root)
        check(declared is None or describe(declared).startswith("lockbox "), "the repository's lockbox.json loads")
    except ValueError as exc:
        failures.append(f"the repository's lockbox.json does not load: {exc}")
    check(load("/nonexistent/root") is None, "no file, no lockbox")

    for message in failures:
        print("FAIL:", message)
    print(f"Lockbox self-check: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    raise SystemExit(1 if failures else 0)
