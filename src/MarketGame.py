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

try:
    from src.MarketData import aligned_returns, closes, instruments
except ImportError:  # imported from within src/
    from MarketData import aligned_returns, closes, instruments

K_BINS = 10          # digits 0..9, like pick3
MIN_HISTORY = 250    # returns an instrument needs before its first day can be binned (about a trading year)
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


def build_game(conn, market, k=K_BINS, min_history=MIN_HISTORY):
    """The market's game days from the store, with the instrument symbols in position order."""
    days, matrix, symbols = aligned_returns(conn, market)
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
    summaries = update_all(conn, market=market, log=log) if fetch else []
    folder = os.path.join(path, "data", "trainingData", market)
    game_days, _ = refresh(conn, market, folder, log=log)
    conn.close()
    return summaries, game_days


def refresh(conn, market, folder, k=K_BINS, min_history=MIN_HISTORY, log=print):
    """Store + CSV for one market. Returns (game days, files written)."""
    days, matrix, symbols = aligned_returns(conn, market)
    game_days = cut_game(days, matrix, k, min_history) if len(days) > min_history else []
    if not game_days:
        log(f"{market}: not enough aligned history to cut a game (need more than {min_history} common days)")
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
    print("store: game days cut from the aligned returns, a missing bar drops its day for all, edges stored and read back")

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
