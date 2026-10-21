"""
The forward record's start date (README "Model performance report").

The History page ranks every row over ALL its scored history. Once a design
is frozen - the September 2026 objective, gate and cadence changes, the
lockbox declared on 2026-09-27 - the honest track record of that design is
the draws predicted AFTER the freeze, and the "since" ranking is exactly
that: the same per-row aggregates, restricted to stored days on or after a
declared date, ranked with the same metric and the same minimum-draws guard.
Nothing is rebuilt or overwritten; the date is a filter over the days that
were predicted before their draw - for a market game literally so: a day
counts only if its ticket was out before the day's close
(Helpers._days_predicted_before_draw, on MarketSettle.ticket_times), which
leaves out the back-built history the market games started with on 1 and
5 Oct 2026 and any day a missed run rebuilt the morning after (14 Oct 2026).

Declared by hand in since.json at the repository root - no file, no since
ranking:

    {"since": "2026-09-27", "label": "September 2026 design freeze"}

    python3 -m src.Since        # self-check
"""

import json
import os
from datetime import date, datetime

FILE_NAME = "since.json"


def _to_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()


def parse(raw):
    """Validate a declaration; ValueError on a malformed one; None for none."""
    if not raw:
        return None
    try:
        since = _to_date(raw["since"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"{FILE_NAME}: 'since' must be a YYYY-MM-DD date ({exc})") from exc
    return {"since": since, "label": str(raw.get("label") or f"since {since.isoformat()}")}


def load(root=None):
    """The declared start as {"since": date, "label": str}, or None."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, FILE_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{FILE_NAME}: not valid JSON ({exc})") from exc
    except OSError as exc:  # a directory named since.json, an unreadable file
        raise ValueError(f"{FILE_NAME}: cannot be read ({exc})") from exc
    return parse(raw)


def on_or_after(since, when):
    """Whether a stored day belongs to the forward record (None dates never do)."""
    if not since or when is None:
        return False
    return _to_date(when) >= since["since"]


if __name__ == "__main__":
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    s = parse({"since": "2026-09-27", "label": "freeze"})
    check(s["since"] == date(2026, 9, 27) and s["label"] == "freeze", "parses")
    check(parse({"since": "2026-09-27"})["label"] == "since 2026-09-27", "default label")
    check(parse(None) is None and parse({}) is None, "no declaration")
    for bad in ({"since": "soon"}, {"label": "x"}, {"since": "2026-13-01"}):
        try:
            parse(bad)
            failures.append(f"accepted {bad}")
        except ValueError:
            pass
    check(on_or_after(s, datetime(2026, 9, 27)) and on_or_after(s, "2026-10-01") and not on_or_after(s, date(2026, 9, 26)),
          "inclusive start")
    check(not on_or_after(None, "2026-10-01") and not on_or_after(s, None), "no declaration or no date")
    check(load("/nonexistent") is None, "no file, no since")
    import tempfile as _tf
    bad_root = _tf.mkdtemp(prefix="since-bad-")
    os.mkdir(os.path.join(bad_root, FILE_NAME))
    try:
        load(bad_root)
        failures.append("a directory named since.json was accepted")
    except ValueError:
        pass
    os.rmdir(os.path.join(bad_root, FILE_NAME)); os.rmdir(bad_root)
    check(parse({"since": "2026-09-27", "label": ["a", "list"]})["label"] == "['a', 'list']", "a non-string label becomes text")

    # The report itself: fake day files, one row scored on both sides of the date.
    import shutil
    import sys
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.Helpers import Helpers
    root = tempfile.mkdtemp(prefix="since-")
    game_dir = os.path.join(root, "lotto")
    os.makedirs(game_dir)
    days = [("2026-9-20", [1, 2, 3, 4, 5, 6, 7]), ("2026-9-26", [1, 2, 3, 4, 5, 6, 7]),
            ("2026-9-27", [1, 2, 3, 4, 5, 6, 7]), ("2026-9-28", [1, 2, 3, 4, 5, 6, 7])]
    for name, real in days:
        # row A: 3 hits before the date, 6 after; row B: the reverse
        after = name >= "2026-9-27"
        a = [1, 2, 3, 40, 41, 42] if not after else [1, 2, 3, 4, 5, 6]
        b = [1, 2, 3, 4, 5, 6] if not after else [1, 2, 3, 40, 41, 42]
        with open(os.path.join(game_dir, f"{name}.json"), "w") as handle:
            json.dump({"realResult": real, "currentPrediction": [{"name": "A Model", "predictions": [a]},
                                                                {"name": "B Model", "predictions": [b]}]}, handle)
    Helpers().generate_model_performance_report(root, combinationShuffles=0, since=s)
    report = json.load(open(os.path.join(root, "modelPerformance.json")))
    lotto = report["games"]["lotto"]
    check(lotto["bestModel"] in ("A Model", "B Model") and lotto["models"][0]["draws"] == 4, "all-history ranking unchanged")
    since_block = lotto.get("since")
    check(since_block and since_block["date"] == "2026-09-27" and since_block["label"] == "freeze", f"since block present: {since_block and list(since_block)}")
    check(since_block["models"][0]["name"] == "A Model" and since_block["models"][0]["draws"] == 2
          and since_block["models"][0]["avg_hits"] == 6.0, f"since ranking counts only the days from the date: {since_block['models'][0]}")
    check(since_block["minDrawsForRanking"] == 2 and since_block["metric"] == "avg_hits", "same guard and metric as the full ranking")
    check(report["since"] == {"date": "2026-09-27", "label": "freeze"}, "the declaration is recorded at report level")
    Helpers().generate_model_performance_report(root, combinationShuffles=0, since=None)
    report = json.load(open(os.path.join(root, "modelPerformance.json")))
    check("since" not in report["games"]["lotto"] and "since" not in report, "no declaration, no block, no marker")
    Helpers().generate_model_performance_report(root, combinationShuffles=0, since=parse({"since": "2030-01-01"}))
    report = json.load(open(os.path.join(root, "modelPerformance.json")))
    check("since" not in report["games"]["lotto"] and report["since"]["date"] == "2030-01-01",
          "a future date has no days yet: no block, but the declaration is recorded")
    shutil.rmtree(root, ignore_errors=True)

    # A market game: only the days predicted before their draw count. Two back-built days (ticket stamped after
    # the close), two live ones; the row that scores on the back-built days must not lead the since block.
    repo = tempfile.mkdtemp(prefix="since-market-")
    market_root = os.path.join(repo, "data", "database")
    crypto_dir = os.path.join(market_root, "crypto")
    os.makedirs(crypto_dir)
    # a day's ticket is the newPrediction of the day file BEFORE it (MarketSettle.ticket_times), so the stamp of
    # file N decides day N+1: 9-28 stamped after 10-01 (back-built 9-29 and 9-30), 9-30 and 10-01 stamped before
    stamps = {"2026-9-28": "2026-10-01T09:00:00+00:00", "2026-9-29": "2026-10-01T09:00:00+00:00",
              "2026-9-30": "2026-09-30T08:40:00+00:00", "2026-10-1": "2026-10-01T08:40:00+00:00", "2026-10-2": "2026-10-02T08:40:00+00:00"}
    for name, made in stamps.items():
        # A scores on the three days outside the forward record (the first stored day, which has no ticket time,
        # and the two back-built ones) and leads all history 15 hits to 10; B scores on the two live days only
        a = [5, 5, 5, 5, 5] if name in ("2026-9-28", "2026-9-29", "2026-9-30") else [0, 0, 0, 0, 0]
        b = [0, 0, 0, 0, 0] if name in ("2026-9-28", "2026-9-29", "2026-9-30") else [5, 5, 5, 5, 5]
        with open(os.path.join(crypto_dir, f"{name}.json"), "w") as handle:
            json.dump({"realResult": [5, 5, 5, 5, 5], "madeAt": made,
                       "currentPrediction": [{"name": "A Model", "predictions": [a]}, {"name": "B Model", "predictions": [b]}]}, handle)
    live = Helpers._days_predicted_before_draw("crypto", market_root)
    check(live == {"2026-10-01", "2026-10-02"}, f"the days whose ticket was out before the close: {sorted(live)}")
    check(Helpers._days_predicted_before_draw("lotto", market_root) is None, "a lottery game has no such filter")
    Helpers().generate_model_performance_report(market_root, combinationShuffles=0, since=parse({"since": "2026-09-01", "label": "freeze"}))
    report = json.load(open(os.path.join(market_root, "modelPerformance.json")))
    crypto = report["games"]["crypto"]
    since_block = crypto.get("since")
    check(crypto["models"][0]["draws"] == 5 and crypto["bestModel"] == "A Model", f"all-history counts every stored day and A leads it: {crypto['models'][0]}")
    check(since_block and since_block["bestModel"] == "B Model" and all(m["draws"] == 2 for m in since_block["models"]),
          f"the since block counts only the two days predicted before their draw: {since_block and since_block['models']}")
    check(all(m.get("live_draws") == 2 for m in crypto["models"]), f"every row's live draws over its whole record: {crypto['models']}")
    # a later freeze date shortens the since block, never a row's live draws (admin.js reads those)
    Helpers().generate_model_performance_report(market_root, combinationShuffles=0, since=parse({"since": "2026-10-02", "label": "later freeze"}))
    later = json.load(open(os.path.join(market_root, "modelPerformance.json")))["games"]["crypto"]
    check(all(m["draws"] == 1 for m in later["since"]["models"]) and all(m.get("live_draws") == 2 for m in later["models"]),
          f"a later freeze: one since draw, still two live draws: {later['models']}")
    shutil.rmtree(repo, ignore_errors=True)

    for message in failures:
        print("FAIL:", message)
    print(f"Since self-check: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    raise SystemExit(1 if failures else 0)
