"""
The market games (README roadmap item 4, phase M2): cutting the positional
"draws" from the bars in data/markets.sqlite3.

Each market (crypto, shares) is one positional game: instrument = position,
the next day's return bin = digit. A day's return per instrument is put
into one of K quantile bins whose edges are fitted on that instrument's
returns STRICTLY BEFORE the day (an expanding window, at least MIN_HISTORY
returns), so the bins are equiprobable by construction, nothing about a day
leaks into its own label, and the past never changes when a day is added -
the yearly CSV files below are stable. Direction (up or down) is the
two-bin reading of the same bins.

The game history is written where every lottery game keeps its history,
data/trainingData/<market>/<market>-gamedata-NL-<year>.csv, in the same
shape (Datum;Nummer 1;...;Nummer N, newest first), so DataLoader,
Backtester, the models, the hyperopts and Predictor.py read a market like
they read pick3. Next to it, <market>-returns.tsv keeps the aligned log
returns the bins were cut from (phase M3): the rows that model returns
rather than bins read it, and a control history gets its own. The bin edges of every day are stored back in the SQLite
store (table game_days) because a predicted bin only means something with
the edges it was cut from: they turn a bin back into a return interval, a
predicted price and a direction for the Markets pages and the settlement.

Self-check: python3 -m src.MarketGame (part of npm test; no network).
"""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
from datetime import date, datetime, timedelta, timezone

try:
    from src.MarketData import aligned_returns, closes, instruments, base_market, is_week_game, WEEK_MARKETS
except ImportError:  # imported from within src/
    from MarketData import aligned_returns, closes, instruments, base_market, is_week_game, WEEK_MARKETS

K_BINS = 10          # digits 0..9, like pick3
MIN_HISTORY = 250    # returns an instrument needs before its first day can be binned (about a trading year)
# The week games (README roadmap item 4, M5): one draw per week - the week's
# log return (the sum of its aligned daily returns) cut into K bins of the
# instrument's own past WEEKS, the draw dated on the week's last day. A
# week counts only once it is over: for crypto when the UTC week has ended
# (Monday 00:00 UTC), for shares from Saturday on (the Friday session has
# closed) - so Monday's crypto run and Saturday's shares run make the ticket
# for the whole coming week. MIN_WEEK_HISTORY weeks (about two years) before
# the first week can be binned.
MIN_WEEK_HISTORY = 100
FILE_PATTERN = "{game}-gamedata-NL-{year}.csv"   # Predictor.py's convention for every game
# The returns the bins were cut from, next to the yearly files: every aligned
# day (the warm-up before the first game day included), oldest first, one
# column per instrument. Not a .csv on purpose - Helpers.load_data and the
# control builder read every .csv in a game folder as draws - and it is what
# the market-specific rows (src/MarketModels.py: GARCH, Regime HMM) model,
# so a control history (src/ControlHistories.py) carries its own.
RETURNS_FILE = "{game}-returns.tsv"

GAME_SCHEMA = """
CREATE TABLE IF NOT EXISTS game_days (
    market   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    position INTEGER NOT NULL,
    symbol   TEXT    NOT NULL,
    ret      REAL    NOT NULL,
    bin      INTEGER NOT NULL,
    edges    TEXT    NOT NULL,
    PRIMARY KEY (market, date, position)
);
"""


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

def quantile_edges(returns, k=K_BINS):
    """The k-1 inner edges that cut `returns` into k equiprobable bins."""
    values = np.asarray(returns, dtype=float)
    if len(values) < k:
        raise ValueError(f"need at least {k} returns for {k} bins, got {len(values)}")
    return [float(e) for e in np.quantile(values, [i / k for i in range(1, k)])]


def bin_of(ret, edges):
    """0..k-1: the number of edges at or below the return."""
    return int(np.searchsorted(np.asarray(edges, dtype=float), float(ret), side="right"))


def bin_interval(index, edges):
    """[low, high) of a bin; the outer bins are open (None)."""
    low = edges[index - 1] if index > 0 else None
    high = edges[index] if index < len(edges) else None
    return low, high


def representative_return(index, edges):
    """
    The return a predicted bin stands for: the midpoint of its interval; for
    the two open bins, the edge moved outward by half the neighbouring bin's
    width, so a predicted price can be drawn for them too.
    """
    low, high = bin_interval(index, edges)
    if low is not None and high is not None:
        return 0.5 * (low + high)
    if low is None:                       # lowest bin
        width = (edges[1] - edges[0]) if len(edges) > 1 else 0.0
        return edges[0] - 0.5 * width
    width = (edges[-1] - edges[-2]) if len(edges) > 1 else 0.0   # highest bin
    return edges[-1] + 0.5 * width


def direction_of(index, edges):
    """+1 when the bin's representative return is positive, -1 otherwise (0 only for an exactly zero return)."""
    value = representative_return(index, edges)
    return 1 if value > 0 else (-1 if value < 0 else 0)


def predicted_price(last_close, index, edges):
    return float(last_close) * math.exp(representative_return(index, edges))


def price_interval(last_close, index, edges):
    """The price range a predicted bin covers; None at an open end."""
    low, high = bin_interval(index, edges)
    return (None if low is None else float(last_close) * math.exp(low),
            None if high is None else float(last_close) * math.exp(high))


# ---------------------------------------------------------------------------
# The game history
# ---------------------------------------------------------------------------

def cut_game(days, matrix, k=K_BINS, min_history=MIN_HISTORY):
    """
    From aligned daily log returns (days x positions) to game days: for each
    day t from min_history on, per position the bin of matrix[t] under the
    edges of matrix[:t]. Returns a list of {date, bins, edges, returns} in
    date order. Causal by construction: day t's edges see returns before t.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or len(days) != len(matrix):
        raise ValueError("days and matrix disagree")
    n_days, n_pos = matrix.shape
    out = []
    for t in range(min_history, n_days):
        bins, edges_all = [], []
        for pos in range(n_pos):
            edges = quantile_edges(matrix[:t, pos], k)
            edges_all.append(edges)
            bins.append(bin_of(matrix[t, pos], edges))
        out.append({"date": days[t], "bins": bins, "edges": edges_all, "returns": [float(v) for v in matrix[t]]})
    return out


def week_of(day):
    """The ISO (year, week) a date string belongs to."""
    iso = date.fromisoformat(str(day)[:10]).isocalendar()
    return (iso[0], iso[1])


# NYSE Group's announced closures (the same table markets.js carries): a shares week whose last
# aligned day is the day before one of these is a full week, not one still waiting for Friday's bar
NYSE_CLOSED = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def week_closed_by(game, last_day):
    """
    Whether `last_day` (the newest aligned day of a week) is the day the
    game's week ends on, so the week's bars are all in: Sunday for crypto;
    for shares Friday, or an earlier weekday when every later weekday of
    that week is an announced closure. A week whose last bar has not arrived
    yet (a source lagging a day) is not cut until it has - otherwise a
    six-day "week" would be dated a day early, re-cut the next morning, and
    the ticket made on it orphaned.
    """
    d = date.fromisoformat(str(last_day)[:10])
    if game == "cryptoweek":
        return d.weekday() == 6
    if d.weekday() > 4:
        return False
    return all((d + timedelta(days=k)).isoformat() in NYSE_CLOSED for k in range(1, 5 - d.weekday()))


def week_complete(game, week, now):
    """
    Whether the ISO week `week` is over for the game at date `now` (UTC): a
    past ISO week always; the current one never for crypto (its week runs to
    Sunday 24:00 UTC) and from Saturday for shares (its last session, Friday,
    has closed).
    """
    current = (now.isocalendar()[0], now.isocalendar()[1])
    if current > week:
        return True
    if current < week:
        return False
    return game == "sharesweek" and now.weekday() >= 5


def weekly_returns(days, matrix, game, now=None, log=None):
    """
    Aligned daily log returns -> one row per COMPLETE week: (week dates, the
    date of the week's last aligned day; week matrix, the sum of the week's
    daily returns per position). A week with no aligned day has no row. The
    newest week is cut only when the calendar says it is over AND its last
    bar is in (week_closed_by); until then it waits, and `log` says so.
    """
    now = now or datetime.now(timezone.utc).date()
    matrix = np.asarray(matrix, dtype=float)
    groups, order = {}, []
    for day, row in zip(days, matrix):
        key = week_of(day)
        if key not in groups:
            groups[key] = [day, np.zeros(len(row))]
            order.append(key)
        groups[key][0] = day
        groups[key][1] = groups[key][1] + row
    wdays, wrows = [], []
    for n, key in enumerate(order):
        if not week_complete(game, key, now):
            continue
        if n == len(order) - 1 and not week_closed_by(game, groups[key][0]):
            if log:
                log(f"{game}: the week of {groups[key][0]} is over by the calendar but its last bar is not in the store yet - not cut until it is")
            continue
        wdays.append(groups[key][0])
        wrows.append(groups[key][1])
    return wdays, (np.asarray(wrows) if wrows else np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0)))


def game_returns(conn, market, now=None, log=None):
    """(days, matrix, symbols) a game is cut from: the base market's aligned daily returns, summed per complete week for a week game."""
    days, matrix, symbols = aligned_returns(conn, base_market(market))
    if is_week_game(market):
        days, matrix = weekly_returns(days, matrix, market, now, log=log)
    return days, matrix, symbols


def history_needed(market, min_history=None):
    if min_history is not None:
        return int(min_history)
    return MIN_WEEK_HISTORY if is_week_game(market) else MIN_HISTORY


def build_game(conn, market, k=K_BINS, min_history=None, now=None):
    """The game's days from the store (a week game: its complete weeks), with the instrument symbols in position order."""
    days, matrix, symbols = game_returns(conn, market, now)
    min_history = history_needed(market, min_history)
    if len(days) <= min_history:
        return [], symbols
    return cut_game(days, matrix, k, min_history), symbols


def store_game_days(conn, market, game_days, symbols):
    """
    Writes (or rewrites) the game days into game_days and removes the stored
    days the new cut no longer has, so the table is exactly the cut. Returns
    rows written.
    """
    conn.executescript(GAME_SCHEMA)
    new_dates = {day["date"] for day in game_days}
    stale = [r[0] for r in conn.execute("SELECT DISTINCT date FROM game_days WHERE market = ?", (market,)).fetchall()
             if r[0] not in new_dates]
    for stale_date in stale:
        conn.execute("DELETE FROM game_days WHERE market = ? AND date = ?", (market, stale_date))
    # a cut over fewer instruments (one retired) must not leave the old cut's
    # extra slots behind on the dates that stay - a day would read a symbol twice
    conn.execute("DELETE FROM game_days WHERE market = ? AND position >= ?", (market, len(symbols)))
    written = 0
    for day in game_days:
        for pos, symbol in enumerate(symbols):
            conn.execute(
                "INSERT INTO game_days (market, date, position, symbol, ret, bin, edges) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(market, date, position) DO UPDATE SET symbol = excluded.symbol, ret = excluded.ret, "
                "bin = excluded.bin, edges = excluded.edges",
                (market, day["date"], pos, symbol, day["returns"][pos], day["bins"][pos], json.dumps(day["edges"][pos])))
            written += 1
    conn.commit()
    return written


def load_game_day(conn, market, date):
    """The stored bins and edges of one day, by position, or None."""
    conn.executescript(GAME_SCHEMA)
    rows = conn.execute("SELECT position, symbol, ret, bin, edges FROM game_days WHERE market = ? AND date = ? ORDER BY position",
                        (market, date)).fetchall()
    if not rows:
        return None
    return {"date": date, "symbols": [r["symbol"] for r in rows], "returns": [r["ret"] for r in rows],
            "bins": [r["bin"] for r in rows], "edges": [json.loads(r["edges"]) for r in rows]}


def latest_edges(conn, market):
    """The newest day's edges per position: what a prediction for the NEXT day is cut with (the next day's own edges include one more return, a negligible shift)."""
    conn.executescript(GAME_SCHEMA)
    row = conn.execute("SELECT MAX(date) FROM game_days WHERE market = ?", (market,)).fetchone()
    return load_game_day(conn, market, row[0]) if row and row[0] else None


def write_game_csv(game_days, folder, game):
    """
    The yearly CSV files Predictor.py and DataLoader read: header
    Datum;Nummer 1;..;Nummer N, one row per game day, newest first within a
    file (as the lottery files are). Every year is rewritten from the game
    days - the content is fully derived, and an unchanged year produces an
    identical file, so git sees a change only where a day was added or a
    bar was revised. Returns the files written.
    """
    if not game_days:
        return []
    os.makedirs(folder, exist_ok=True)
    n_pos = len(game_days[0]["bins"])
    by_year = {}
    for day in game_days:
        by_year.setdefault(day["date"][:4], []).append(day)
    written = []
    for year, days in sorted(by_year.items()):
        path = os.path.join(folder, FILE_PATTERN.format(game=game, year=year))
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["Datum"] + [f"Nummer {i + 1}" for i in range(n_pos)])
            for day in sorted(days, key=lambda d: d["date"], reverse=True):
                writer.writerow([day["date"]] + [str(b) for b in day["bins"]])
        written.append(path)
    # a year the cut no longer has must not linger with rows from another
    # cut: the folder is exactly the game, nothing else
    prefix = FILE_PATTERN.split("{year}")[0].format(game=game)      # "<game>-gamedata-NL-"
    for name in os.listdir(folder):
        if name.startswith(prefix) and name.endswith(".csv") and name[len(prefix):-4] not in by_year:
            os.remove(os.path.join(folder, name))
    return written


def read_game_csv(folder):
    """The game rows back from the yearly files, oldest first: [(date, [bins])]."""
    rows = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(folder, name), newline="") as handle:
            reader = csv.reader(handle, delimiter=";")
            next(reader, None)
            for line in reader:
                if line:
                    rows.append((line[0], [int(v) for v in line[1:]]))
    rows.sort(key=lambda r: r[0])
    return rows


def write_returns_file(days, matrix, symbols, folder, game):
    """
    <game>-returns.tsv: Datum<TAB>SYMBOL.. header, one row per aligned day
    (oldest first), the log return per instrument. Fully derived like the
    yearly files: an unchanged history produces an identical file.
    """
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, RETURNS_FILE.format(game=game))
    matrix = np.asarray(matrix, dtype=float)
    lines = ["\t".join(["Datum"] + [str(s) for s in symbols])]
    for day, row in zip(days, matrix):
        lines.append("\t".join([str(day)] + [f"{float(v):.10g}" for v in row]))
    with open(target, "w", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
    return target


def read_returns_file(folder, game=None):
    """
    (days, matrix, symbols) from the folder's returns file - the one named
    for `game`, else the only *-returns.tsv there (a control folder is named
    for its seed, not its game). None when there is none.
    """
    if not os.path.isdir(folder):
        return None
    candidates = [RETURNS_FILE.format(game=game)] if game else []
    candidates += sorted(n for n in os.listdir(folder) if n.endswith("-returns.tsv"))
    for name in candidates:
        target = os.path.join(folder, name)
        if not os.path.exists(target):
            continue
        days, rows = [], []
        with open(target, newline="") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                days.append(parts[0])
                rows.append([float(v) for v in parts[1:]])
        matrix = np.asarray(rows, dtype=float) if rows else np.zeros((0, max(0, len(header) - 1)))
        return days, matrix, header[1:]
    return None


def daily_refresh(path, market, fetch=True, db_path=None, log=print):
    """
    What a market game needs before Predictor.py runs it, in one call: the
    store opened (and the universe installed the first time), the bars
    brought up to date from their sources unless fetch=False, and the game
    days re-cut and written to data/trainingData/<market>/. Returns the
    update summaries and the game days. Called by Predictor.py in the place
    where a lottery game downloads its CSV, and by MarketsDaily.py by hand.
    A source that does not answer leaves the game one day short, never
    without a history: the CSV on disk is what the run predicts from.
    """
    try:
        from src.MarketData import DEFAULT_DB, connect, install_universe, update_all
    except ImportError:
        from MarketData import DEFAULT_DB, connect, install_universe, update_all
    conn = connect(db_path or os.path.join(path, DEFAULT_DB))
    added, conflicts = install_universe(conn)
    if added:
        log(f"{market}: universe installed ({', '.join(added)})")
    for conflict in conflicts:
        log(f"{market}: universe CONFLICT: {conflict}")
    # a week game is cut from its base market's bars, fetched by the base game's own refresh moments before
    summaries = update_all(conn, market=market, log=log) if (fetch and not is_week_game(market)) else []
    folder = os.path.join(path, "data", "trainingData", market)
    game_days, _ = refresh(conn, market, folder, log=log)
    conn.close()
    return summaries, game_days


def refresh(conn, market, folder, k=K_BINS, min_history=None, log=print, now=None):
    """Store + CSV for one game (a market, or a market's week game). Returns (game days, files written)."""
    days, matrix, symbols = game_returns(conn, market, now, log=log)
    min_history = history_needed(market, min_history)
    game_days = cut_game(days, matrix, k, min_history) if len(days) > min_history else []
    if not game_days:
        log(f"{market}: not enough aligned history to cut a game (need more than {min_history} common {'weeks' if is_week_game(market) else 'days'})")
        return [], []
    conn.executescript(GAME_SCHEMA)
    previous_first = conn.execute("SELECT MIN(date) FROM game_days WHERE market = ?", (market,)).fetchone()[0]
    if previous_first and previous_first != game_days[0]["date"]:
        # The expanding window is anchored on the store's first common day:
        # when that moves, every later day's edges move with it and the
        # history is a different cut from the one the stored day files were
        # scored on. Said loudly, because the repair is a -r rebuild.
        log(f"{market}: WARNING the first game day moved from {previous_first} to {game_days[0]['date']} - the history was "
            f"re-cut; the stored day files under data/database/{market} were scored on the old cut (Predictor.py -r rebuilds them)")
    store_game_days(conn, market, game_days, symbols)
    files = write_game_csv(game_days, folder, market)
    # the returns next to the bins: what the GARCH and Regime HMM rows model
    files.append(write_returns_file(days, matrix, symbols, folder, market))
    log(f"{market}: {len(game_days)} game days {game_days[0]['date']} .. {game_days[-1]['date']} over {symbols}, "
        f"{len(files) - 1} yearly file(s) and the returns file written")
    return game_days, files


# ---------------------------------------------------------------------------
# Self-check (no network)
# ---------------------------------------------------------------------------

def _self_check():
    import tempfile
    try:
        from src.MarketData import connect, install_universe, upsert_bars
    except ImportError:
        from MarketData import connect, install_universe, upsert_bars

    rng = np.random.default_rng(11)

    # 1. Bins: equiprobable on the fitting sample, edges monotone, intervals tile the line.
    sample = rng.standard_t(4, size=2000) * 0.02
    edges = quantile_edges(sample, 10)
    assert len(edges) == 9 and all(edges[i] < edges[i + 1] for i in range(8))
    counts = np.bincount([bin_of(r, edges) for r in sample], minlength=10)
    assert counts.min() >= 190 and counts.max() <= 210, counts
    assert bin_of(edges[0], edges) == 1 and bin_of(edges[0] - 1e-12, edges) == 0 and bin_of(10.0, edges) == 9
    assert bin_interval(0, edges) == (None, edges[0]) and bin_interval(9, edges) == (edges[8], None)
    assert bin_interval(4, edges) == (edges[3], edges[4])
    mids = [representative_return(b, edges) for b in range(10)]
    assert all(mids[i] < mids[i + 1] for i in range(9)) and abs(mids[4] - 0.5 * (edges[3] + edges[4])) < 1e-15
    assert direction_of(9, edges) == 1 and direction_of(0, edges) == -1
    p = predicted_price(100.0, 7, edges)
    lo, hi = price_interval(100.0, 7, edges)
    assert lo < p < hi and price_interval(100.0, 0, edges)[0] is None and price_interval(100.0, 9, edges)[1] is None
    print("bins: ten equiprobable quantile bins, monotone edges, intervals tile the line, prices follow")

    # 1b. Weeks: the aligned days of 2026-09-21 (Mon) .. 2026-10-04 (Sun) plus Monday 2026-10-05
    wk_days = [(date(2026, 9, 21) + timedelta(days=i)).isoformat() for i in range(15)]
    wk_matrix = np.arange(15 * 2, dtype=float).reshape(15, 2) / 100.0
    # crypto: at Sunday 2026-10-04 the second week is not over; on Monday 2026-10-05 it is; the lone Monday is never a week yet
    d1, m1 = weekly_returns(wk_days, wk_matrix, "cryptoweek", now=date(2026, 10, 4))
    assert d1 == ["2026-09-27"] and np.allclose(m1[0], wk_matrix[:7].sum(axis=0)), (d1, m1)
    d2, m2 = weekly_returns(wk_days, wk_matrix, "cryptoweek", now=date(2026, 10, 5))
    assert d2 == ["2026-09-27", "2026-10-04"] and np.allclose(m2[1], wk_matrix[7:14].sum(axis=0)), d2
    # shares: weekdays only; the week is over from Saturday, and a holiday Friday still closes the week on Saturday
    sh_days = [d for d in wk_days if date.fromisoformat(d).weekday() < 5 and d != "2026-10-02"]   # the second Friday is a holiday
    sh_matrix = np.ones((len(sh_days), 2)) * 0.01
    d3, _ = weekly_returns(sh_days, sh_matrix, "sharesweek", now=date(2026, 10, 2))
    assert d3 == ["2026-09-25"], d3
    d4, m4 = weekly_returns(sh_days, sh_matrix, "sharesweek", now=date(2026, 10, 3))
    # with Monday 2026-10-05 already in the data the Thursday-ended week is a real four-day week (the Friday was a gap for all)
    assert d4 == ["2026-09-25", "2026-10-01"] and abs(m4[1][0] - 0.04) < 1e-12, (d4, m4)
    # ... but when the data ENDS on that Thursday and its Friday is no announced closure, the Friday bar may still arrive: not cut yet, said in the log
    lag_notes = []
    d4l, _ = weekly_returns([d for d in sh_days if d <= "2026-10-01"], np.ones((len([d for d in sh_days if d <= "2026-10-01"]), 2)) * 0.01, "sharesweek", now=date(2026, 10, 3), log=lag_notes.append)
    assert d4l == ["2026-09-25"] and lag_notes and "2026-10-01" in lag_notes[0], (d4l, lag_notes)
    sh_days2 = [d for d in [(date(2026, 12, 21) + timedelta(days=i)).isoformat() for i in range(4)]]   # Mon 21 .. Thu 24 Dec, the data ends there
    d4b, _ = weekly_returns(sh_days2, np.ones((len(sh_days2), 2)) * 0.01, "sharesweek", now=date(2026, 12, 26))
    assert d4b == ["2026-12-24"], d4b   # Christmas Friday is an announced closure: the Thursday closes the week
    assert week_complete("cryptoweek", week_of("2026-10-04"), date(2026, 10, 4)) is False and week_complete("cryptoweek", week_of("2026-10-04"), date(2026, 10, 5)) is True
    # a lagging last bar: the week is over by the calendar but Sunday is missing - not cut until it arrives; a holiday Friday closes the shares week on Thursday
    notes = []
    d5, _ = weekly_returns(wk_days[:13], wk_matrix[:13], "cryptoweek", now=date(2026, 10, 5), log=notes.append)   # ends Saturday 2026-10-03
    assert d5 == ["2026-09-27"] and notes and "2026-10-03" in notes[0], (d5, notes)
    assert week_closed_by("cryptoweek", "2026-10-04") and not week_closed_by("cryptoweek", "2026-10-03")
    assert week_closed_by("sharesweek", "2026-10-09") and not week_closed_by("sharesweek", "2026-10-08") and week_closed_by("sharesweek", "2026-12-24") and week_closed_by("sharesweek", "2026-11-25") is False
    # the mid-week tail of a history is never a week (whatever `now` says)
    d6, _ = weekly_returns(wk_days, wk_matrix, "cryptoweek", now=date(2030, 1, 1))
    assert d6 == ["2026-09-27", "2026-10-04"], d6
    assert history_needed("cryptoweek") == MIN_WEEK_HISTORY and history_needed("crypto") == MIN_HISTORY and history_needed("cryptoweek", 7) == 7
    print("weeks: a week is its daily returns summed, dated on its last day, counted once it is over - Monday 00:00 UTC for crypto, Saturday for shares")

    # 2. Cutting a game is causal: a later return changes no earlier day.
    days = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(400)]
    matrix = rng.normal(0, 0.02, size=(400, 3))
    game = cut_game(days, matrix, k=10, min_history=250)
    assert len(game) == 150 and game[0]["date"] == days[250] and len(game[0]["bins"]) == 3 and len(game[0]["edges"][0]) == 9
    perturbed = matrix.copy()
    perturbed[300:, :] += 5.0
    game2 = cut_game(days, perturbed, k=10, min_history=250)
    assert all(a["bins"] == b["bins"] and a["edges"] == b["edges"] for a, b in zip(game[:50], game2[:50])), "a later return leaked into an earlier day"
    assert game2[60]["bins"] == [9, 9, 9], "a +5 return must land in the top bin"
    flat = np.array([[b for b in d["bins"]] for d in game])
    assert flat.min() >= 0 and flat.max() <= 9
    shares = np.bincount(flat.ravel(), minlength=10) / flat.size
    assert shares.min() > 0.04 and shares.max() < 0.18, shares
    print(f"cut: {len(game)} causal game days, later returns never move earlier bins, bins spread {shares.min():.2f}-{shares.max():.2f}")

    # 3. Through the store: synthetic random walks for the frozen crypto universe.
    conn = connect(":memory:")
    install_universe(conn)
    crypto = instruments(conn, "crypto")
    all_days = [f"{2024 + i // 360}-{1 + (i % 360) // 30:02d}-{1 + i % 30:02d}" for i in range(720)]
    for member in crypto:
        walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, size=len(all_days))))
        upsert_bars(conn, member["id"], [{"date": d, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}
                                          for d, c in zip(all_days, walk)], fetched_at="t")
    # one instrument misses one day: that day drops for all
    conn.execute("DELETE FROM bars WHERE instrument_id = ? AND date = ?", (crypto[2]["id"], all_days[400]))
    conn.commit()
    game_days, symbols = build_game(conn, "crypto", k=10, min_history=250)
    assert symbols == ["BTC", "ETH", "BNB", "XRP", "SOL"] and len(game_days) == 720 - 1 - 1 - 250, len(game_days)
    assert all(len(d["bins"]) == 5 for d in game_days) and all(all_days[400] != d["date"] for d in game_days)
    written = store_game_days(conn, "crypto", game_days, symbols)
    assert written == len(game_days) * 5
    stored = load_game_day(conn, "crypto", game_days[-1]["date"])
    assert stored["bins"] == game_days[-1]["bins"] and stored["edges"][3] == game_days[-1]["edges"][3] and stored["symbols"] == symbols
    assert latest_edges(conn, "crypto")["date"] == game_days[-1]["date"]
    assert store_game_days(conn, "crypto", game_days, symbols) == written, "rewriting is idempotent"
    assert conn.execute("SELECT COUNT(*) FROM game_days").fetchone()[0] == written
    # a cut over four instruments on the same dates leaves no fifth slot behind
    four = [dict(d, bins=d["bins"][:4], edges=d["edges"][:4], returns=d["returns"][:4]) for d in game_days]
    assert store_game_days(conn, "crypto", four, symbols[:4]) == len(four) * 4
    assert load_game_day(conn, "crypto", four[-1]["date"])["symbols"] == symbols[:4]
    assert conn.execute("SELECT COUNT(*) FROM game_days WHERE position >= 4").fetchone()[0] == 0
    assert store_game_days(conn, "crypto", game_days, symbols) == written
    print("store: game days cut from the aligned returns, a missing bar drops its day for all, edges stored and read back")
    # the week game: a store with real calendar dates (the fixture above uses 28-day months), 420 days of bars,
    # the complete weeks cut with a short history for the test and stored apart from the daily game
    conn_w = connect(":memory:")
    install_universe(conn_w)
    real_days = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(420)]
    for member in instruments(conn_w, "crypto"):
        walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, size=len(real_days))))
        upsert_bars(conn_w, member["id"], [{"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1.0} for d, c in zip(real_days, walk)], fetched_at="t")
    daily_w, symbols_w = build_game(conn_w, "crypto", k=10, min_history=250)
    store_game_days(conn_w, "crypto", daily_w, symbols_w)
    week_days, week_symbols = build_game(conn_w, "cryptoweek", k=10, min_history=20, now=date(2026, 2, 24))   # the newest fixture day: its own week is not over
    assert week_symbols == symbols_w and len(week_days) > 30 and all(date.fromisoformat(d["date"]).weekday() == 6 for d in week_days), \
        (len(week_days), [d["date"] for d in week_days[:3]])
    assert all(len(d["bins"]) == len(symbols_w) and len(d["edges"][0]) == 9 for d in week_days)
    # a week's return is the sum of its days' returns
    wd, wm, _ = game_returns(conn_w, "cryptoweek", now=date(2026, 2, 24))
    dd, dm, _ = game_returns(conn_w, "crypto")
    first_week = [i for i, d in enumerate(dd) if week_of(d) == week_of(wd[5])]
    assert np.allclose(wm[5], np.asarray(dm, dtype=float)[first_week].sum(axis=0)) and len(first_week) == 7
    store_game_days(conn_w, "cryptoweek", week_days, week_symbols)
    assert latest_edges(conn_w, "cryptoweek")["date"] == week_days[-1]["date"] and latest_edges(conn_w, "crypto")["date"] == daily_w[-1]["date"], "the two games keep separate game days"
    assert build_game(conn_w, "cryptoweek", k=10)[0] == [], "with the real history floor 420 days are too few weeks for a week game"
    print(f"week game: {len(week_days)} Sunday-dated weeks cut from the same bars, each the sum of its days, stored next to the daily game")

    # 4. The yearly CSV files: Predictor's shape, newest first, round trip, stable.
    with tempfile.TemporaryDirectory() as folder:
        files = write_game_csv(game_days, folder, "crypto")
        assert sorted(os.path.basename(f) for f in files) == ["crypto-gamedata-NL-2024.csv", "crypto-gamedata-NL-2025.csv"], files
        with open(files[0]) as handle:
            header = handle.readline().strip()
            first = handle.readline().strip().split(";")
        assert header == "Datum;Nummer 1;Nummer 2;Nummer 3;Nummer 4;Nummer 5" and len(first) == 6 and first[0].startswith("2024")
        back = read_game_csv(folder)
        assert [(d["date"], d["bins"]) for d in game_days] == back, "the CSV round trip must be exact"
        before = {f: open(f).read() for f in files}
        write_game_csv(game_days, folder, "crypto")
        assert all(open(f).read() == before[f] for f in files), "an unchanged year must produce an identical file"
        # the lottery loader's convention: newest first inside a file
        year_rows = [r for r in back if r[0].startswith("2025")]
        with open(files[1]) as handle:
            lines = handle.read().strip().splitlines()[1:]
        assert lines[0].split(";")[0] == year_rows[-1][0] and lines[-1].split(";")[0] == year_rows[0][0]
        # refresh does both, and writes the returns next to the bins
        game_again, files_again = refresh(conn, "crypto", folder, log=lambda *_: None)
        assert len(game_again) == len(game_days) and len(files_again) == 3 and files_again[-1].endswith("crypto-returns.tsv")
        days_back, matrix_back, symbols_back = read_returns_file(folder, "crypto")
        aligned_days, aligned_matrix, _ = aligned_returns(conn, "crypto")
        assert days_back == aligned_days and symbols_back == symbols and np.allclose(matrix_back, aligned_matrix, atol=1e-9)
        assert read_returns_file(folder) is not None and read_returns_file(os.path.join(folder, "nowhere")) is None
        assert read_game_csv(folder) == back, "the returns file must not be read as draws"
        # the bins in the files are the bins the returns file gives under the causal edges
        by_date = dict(zip(days_back, matrix_back))
        idx = {d: i for i, d in enumerate(days_back)}
        for day in game_days[-3:]:
            t = idx[day["date"]]
            assert [bin_of(matrix_back[t, pos], quantile_edges(matrix_back[:t, pos], 10)) for pos in range(5)] == day["bins"]
        # a shorter cut removes the year file and the stored days it no longer has
        shorter = [d for d in game_days if d["date"] >= "2025-01-01"]
        write_game_csv(shorter, folder, "crypto")
        assert sorted(os.listdir(folder)) == ["crypto-gamedata-NL-2025.csv", "crypto-returns.tsv"], os.listdir(folder)
        assert store_game_days(conn, "crypto", shorter, symbols) == len(shorter) * 5
        assert conn.execute("SELECT MIN(date) FROM game_days WHERE market = 'crypto'").fetchone()[0] == shorter[0]["date"]
        assert conn.execute("SELECT COUNT(DISTINCT date) FROM game_days WHERE market = 'crypto'").fetchone()[0] == len(shorter)
        # and refresh says so when the first day moved
        said = []
        refresh(conn, "crypto", folder, log=said.append)
        assert any("WARNING the first game day moved" in line for line in said), said
    print("csv: yearly files in the lottery shape, newest first, exact round trip, byte-identical when nothing changed, "
          "stale years and days removed, a moved first day announced; the returns file round-trips and re-cuts to the same bins")
    print("MarketGame self-check OK")


if __name__ == "__main__":
    _self_check()
