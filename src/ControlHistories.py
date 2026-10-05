"""
Control histories: the data behind both Q0 (NullControls.py) and Q2
(RandomnessDiscrimination.py), in one place so the two experiments cannot
disagree about what "a fair draw" means.

Three kinds of history, all with the real calendar:

  real       what actually happened.
  synthetic  draws generated under the game's own rules, uniformly at
             random. No structure of any kind - not a trend, not a bias, not
             a repeat beyond chance.
  shuffled   the real draws, reordered. Every marginal is preserved exactly
             (number frequencies, co-occurrence, how hot a number looks) and
             only the time order is destroyed.

The synthetic rules come from HyperoptStatistics.GAME_CONFIG for the main
numbers (the authoritative range and draw size) and from the real history for
the trailing columns the config does not describe - the Euromillions stars,
the EuroDreams dream number, the VikingLotto super viking, the Joker+ zodiac
code, Lotto's unplayed bonus. Deriving those from the data is deliberate: it
is the same rule the real draws demonstrably followed.

CAVEAT the README's own interpretation list already names: a synthetic
history reproduces TODAY's rules over the whole calendar, so where a game's
rules changed mid-history the two differ for that reason alone. Any
discrimination result has to be checked against the rule-change dates before
it is called evidence of anything.

THE MARKET GAMES (README roadmap item 4, M3) have no drawing rule to
simulate: their draws are return bins cut from prices (src/MarketGame.py), so
their "fair process" is the efficient-market prior - a geometric random walk.
  synthetic  per instrument, independent log returns drawn from a Gaussian
             with the instrument's own mean and standard deviation over the
             real history (matched volatility, no clustering, no cross-
             section, no drift beyond the mean), cut into bins by the game's
             own causal quantile rule - so the bins are as equiprobable as the
             real ones and only the dynamics differ.
  shuffled   the real game days reordered, bins AND returns together: every
             bin frequency and the returns' marginal survive, volatility
             clustering and momentum die with the order.
Both carry their own <market>-returns.tsv next to the bins, so the rows that
model returns (src/MarketModels.py) run on a control as they run on the real
history.
"""

import os

import numpy as np
from dateutil.parser import parse as parse_date

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

from src.Helpers import Helpers
from src.MarketGame import MIN_HISTORY, K_BINS, cut_game, read_returns_file, write_returns_file
from HyperoptStatistics import GAME_CONFIG

helpers = Helpers()

CONTROL_ROOT = os.path.join(parent_dir, "data", "controls")


def training_path(game):
    return os.path.join(parent_dir, "data", "trainingData", game)


def _as_int(value):
    """CSV cell -> int, encoding Joker+'s zodiac name to its 0..11 code."""
    text = str(value).strip()
    return int(text) if text.lstrip("-").isdigit() else int(helpers.encode_zodiac(text))


def read_real_rows(game):
    """
    (header line, [(date string, [column values as written]), ...]) oldest
    first. The header is reused verbatim by the control files, so a control
    directory is indistinguishable in format from a real one.
    """
    path = training_path(game)
    rows, header = [], None
    for name in sorted(os.listdir(path)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(path, name), "r", encoding="utf-8-sig") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if not lines:
            continue
        if header is None:
            header = lines[0]
        for line in lines[1:]:
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                parse_date(parts[0])
            except Exception:
                continue
            rows.append((parts[0], parts[1:]))
    rows.sort(key=lambda row: parse_date(row[0]))
    return header, current_era(game, rows)


_ERA_NOTED = set()      # games whose dropped era this process has already reported

# The first draw under today's rules, for a game that once had other rules
# which the number range alone does not always betray: a 20-of-80 keno draw
# lies wholly inside 1-70 about one time in twenty, and two of the 58 old
# draws do. Rows dated before this are dropped together with the ones whose
# numbers fall outside the configured range.
ERA_START = {"keno": "2008-03-09"}


def current_era(game, rows):
    """
    The rows of today's game only. A draw dated before the game's ERA_START,
    or whose main numbers fall outside the configured range, belongs to an
    earlier version of the game and is dropped. Keno drew 20 of 1-80 until
    8 March 2008 and 20 of 1-70 since: 58 draws in the 2008 file, 56 of them
    with numbers above 70. Carried into a control, the old draws would get
    synthetic replacements generated under today's rules on their dates, and
    the shuffled control would scatter 1-80 draws across the window the rows
    are scored on. Says what it dropped, once per game and process; the other
    games lose nothing (checked 5 Oct 2026).
    """
    cfg = GAME_CONFIG.get(game)
    if not cfg:
        return rows
    low, high, size = cfg["min"], cfg["max"], cfg["draw_size"]
    since = parse_date(ERA_START[game]) if game in ERA_START else None
    kept, dropped = [], []
    for date, columns in rows:
        if since is not None and parse_date(date) < since:
            dropped.append(date)
            continue
        try:
            main = [_as_int(value) for value in columns[:size]]
        except Exception:
            kept.append((date, columns))      # not this function's business
            continue
        if main and all(low <= value <= high for value in main):
            kept.append((date, columns))
        else:
            dropped.append(date)
    if dropped and game not in _ERA_NOTED:
        _ERA_NOTED.add(game)
        rule = f"before {ERA_START[game]} or " if game in ERA_START else ""
        print(f"{game}: {len(dropped)} draw(s) {rule}with numbers outside {low}-{high} dropped as an earlier version "
              f"of the game ({dropped[0]} .. {dropped[-1]})")
    return kept


def column_ranges(rows, first, count):
    """Observed [min, max] per trailing column over the real history."""
    ranges = []
    for offset in range(count):
        values = []
        for _, columns in rows:
            if first + offset < len(columns):
                try:
                    values.append(_as_int(columns[first + offset]))
                except Exception:
                    continue
        ranges.append((min(values), max(values)) if values else (0, 0))
    return ranges


def synthetic_row(rng, cfg, positional, trailing, special_count, skip_last):
    """One draw under the game's own rules and nothing else."""
    low, high = cfg["min"], cfg["max"]
    if positional:
        # Digits drawn with replacement, in drawn order: repeats are normal
        # and the order is what pays.
        main = [int(v) for v in rng.integers(low, high + 1, size=cfg["draw_size"])]
    else:
        main = sorted(int(v) for v in rng.choice(range(low, high + 1), size=cfg["draw_size"], replace=False))

    extra = []
    if skip_last > 0:
        # Lotto's bonus: same pool, never one of the six.
        pool = [n for n in range(low, high + 1) if n not in set(main)]
        extra.extend(int(v) for v in rng.choice(pool, size=skip_last, replace=False))
    if special_count > 0:
        specials = trailing[skip_last:skip_last + special_count]
        if special_count > 1:
            # Euromillions' two stars: distinct, from one shared range.
            lo = min(r[0] for r in specials)
            hi = max(r[1] for r in specials)
            extra.extend(sorted(int(v) for v in rng.choice(range(lo, hi + 1), size=special_count, replace=False)))
        else:
            lo, hi = specials[0]
            extra.append(int(rng.integers(lo, hi + 1)))
    return main + extra


def control_market(game, mode, seed, recent=None, source=None):
    """
    A market game's control history: (dates, draws, returns days, returns
    matrix, symbols). The draws are the kept game days' bins (real, re-cut
    from a geometric random walk, or reordered), the returns are the
    matching <market>-returns.tsv content - every aligned day, the warm-up
    before the first game day included, so the market rows can be fitted on
    the control exactly as on the real history.
    """
    folder = source or training_path(game)
    _, rows = read_real_rows(game) if source is None else _read_rows_from(folder)
    found = read_returns_file(folder, game)
    if found is None:
        raise FileNotFoundError(f"{folder}: no {game}-returns.tsv - a market control needs the returns the bins were cut from")
    days, matrix, symbols = found
    matrix = np.asarray(matrix, dtype=float)
    if recent:
        rows = rows[-int(recent):]
    kept_dates = [date for date, _ in rows]
    rng = np.random.default_rng(seed)
    index = {d: i for i, d in enumerate(days)}

    if mode == "real":
        draws = [[_as_int(v) for v in columns] for _, columns in rows]
        return kept_dates, draws, list(days), matrix, list(symbols)

    if mode == "synthetic":
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0)
        synthetic = rng.normal(0.0, 1.0, size=matrix.shape) * sd + mean
        cut = {day["date"]: day["bins"] for day in cut_game(days, synthetic, k=K_BINS, min_history=MIN_HISTORY)}
        missing = [d for d in kept_dates if d not in cut]
        if missing:
            raise ValueError(f"{game}: {len(missing)} game day(s) have no cut under the returns file (first {missing[0]}) - "
                             "the game files and the returns file disagree; MarketsDaily.py --no-fetch rewrites both")
        return kept_dates, [list(cut[d]) for d in kept_dates], list(days), synthetic, list(symbols)

    if mode == "shuffled":
        order = rng.permutation(len(rows))
        draws = [[_as_int(v) for v in rows[i][1]] for i in order]
        shuffled = matrix.copy()
        for target_date, source_i in zip(kept_dates, order):
            source_date = rows[source_i][0]
            if target_date not in index or source_date not in index:
                raise ValueError(f"{game}: game day {target_date} or {source_date} is missing from the returns file")
            shuffled[index[target_date]] = matrix[index[source_date]]
        return kept_dates, draws, list(days), shuffled, list(symbols)

    raise ValueError(f"unknown control mode {mode}")


def _read_rows_from(folder):
    """read_real_rows for an explicit folder (the self-check's temporary market)."""
    rows, header = [], None
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(folder, name), "r", encoding="utf-8-sig") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if not lines:
            continue
        if header is None:
            header = lines[0]
        for line in lines[1:]:
            parts = line.split(";")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1:]))
    rows.sort(key=lambda row: parse_date(row[0]))
    return header, rows


def control_draws(game, mode, seed, recent=None):
    """
    (dates, draws) for one control history, in memory: every column, ints,
    oldest first, the real calendar. `mode` is "real", "synthetic" or
    "shuffled". A market game's history comes from control_market.

    `recent` keeps only the newest N draws, and it is applied BEFORE the mode,
    which is load-bearing for the shuffled control. Permuting the whole
    history and then taking its newest N gives a random sample of every era
    the game ever had - measured on lotto's last 600: per-number counts with
    a standard deviation of 13.5 against the real 9.1, chi-square p < 0.0001,
    i.e. a control that no longer shares the marginals it exists to preserve.
    Cutting first and shuffling inside the window keeps "same draws, same
    frequencies, different order" true.
    """
    if helpers.is_market_game(game):
        dates, draws, _, _, _ = control_market(game, mode, seed, recent=recent)
        return dates, draws
    cfg = GAME_CONFIG[game]
    _, rows = read_real_rows(game)
    if recent:
        rows = rows[-int(recent):]
    positional = helpers.is_positional_game(game)
    skip_last, special_count = cfg["skip_last_columns"], cfg["special_column_count"]
    trailing = column_ranges(rows, cfg["draw_size"], skip_last + special_count)
    rng = np.random.default_rng(seed)

    if mode == "synthetic":
        draws = [synthetic_row(rng, cfg, positional, trailing, special_count, skip_last) for _ in rows]
    elif mode == "shuffled":
        draws = [[_as_int(v) for v in rows[index][1]] for index in rng.permutation(len(rows))]
    elif mode == "real":
        draws = [[_as_int(v) for v in columns] for _, columns in rows]
    else:
        raise ValueError(f"unknown control mode {mode}")
    return [date for date, _ in rows], draws


def main_numbers(game, draws):
    """Just the played numbers - trailing bonus/special columns dropped."""
    return [draw[:GAME_CONFIG[game]["draw_size"]] for draw in draws]


def build_control_dataset(game, mode, seed, root=None, recent=None):
    """
    Writes one control history as CSVs that Helpers.load_data reads exactly
    like the real ones, and returns (directory, number of draws). Same dates
    in the same order: only the numbers are replaced (synthetic) or reordered
    (shuffled), so anything date-derived is identical and each control
    differs from the real history in exactly one respect.
    """
    header, rows = read_real_rows(game)
    if recent:
        rows = rows[-int(recent):]
    returns = None
    if helpers.is_market_game(game):
        _, draws, days, matrix, symbols = control_market(game, mode, seed, recent=recent)
        returns = (days, matrix, symbols)
    else:
        _, draws = control_draws(game, mode, seed, recent=recent)
    directory = os.path.join(root or CONTROL_ROOT, game, f"{mode}-seed{seed}")
    os.makedirs(directory, exist_ok=True)
    for stale in os.listdir(directory):
        os.remove(os.path.join(directory, stale))
    with open(os.path.join(directory, f"{game}-control.csv"), "w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for (date, _), draw in zip(rows, draws):
            handle.write(";".join([date] + [str(value) for value in draw]) + "\n")
    if returns is not None:
        # the market rows model these; the control carries its own
        write_returns_file(*returns, directory, game)
    return directory, len(rows)


if __name__ == "__main__":
    # Self-check on a temporary market (no real data touched): the synthetic
    # control is a random walk with the real volatility and equiprobable bins,
    # the shuffled control keeps bins and returns together, both round-trip
    # through the control folder with their returns file.
    #   python3 -m src.ControlHistories
    import tempfile
    from src.MarketGame import write_game_csv, read_game_csv, bin_of, quantile_edges
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        n_days, n_pos = 600, 4
        vol = np.where((np.arange(n_days) // 50) % 2 == 0, 0.01, 0.03)
        matrix = rng.standard_normal((n_days, n_pos)) * vol[:, None] + 0.0005
        from datetime import date, timedelta
        days = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(n_days)]
        source = os.path.join(tmp, "shares")
        game = cut_game(days, matrix, k=K_BINS, min_history=MIN_HISTORY)
        write_game_csv(game, source, "shares")
        write_returns_file(days, matrix, ["A", "B", "C", "D"], source, "shares")
        real_dates, real_draws, _, real_matrix, _ = control_market("shares", "real", 0, source=source)
        assert real_draws == [d["bins"] for d in game] and np.allclose(real_matrix, matrix)

        s_dates, s_draws, s_days, s_matrix, symbols = control_market("shares", "synthetic", 1, source=source)
        assert s_dates == real_dates and s_days == days and symbols == ["A", "B", "C", "D"]
        assert np.allclose(s_matrix.std(axis=0), matrix.std(axis=0), rtol=0.1) and np.allclose(s_matrix.mean(axis=0), matrix.mean(axis=0), atol=0.003)
        flat = np.array(s_draws)
        shares = np.bincount(flat.ravel(), minlength=10) / flat.size
        assert shares.min() > 0.05 and shares.max() < 0.16, shares
        # no volatility clustering: |r_t| and |r_{t-1}| are uncorrelated on the walk, correlated on the real history
        real_ac = np.corrcoef(np.abs(matrix[1:, 0]), np.abs(matrix[:-1, 0]))[0, 1]
        syn_ac = np.corrcoef(np.abs(s_matrix[1:, 0]), np.abs(s_matrix[:-1, 0]))[0, 1]
        assert real_ac > 0.15 and abs(syn_ac) < 0.1, (real_ac, syn_ac)
        # the synthetic bins are the synthetic returns under the causal edges
        idx = {d: i for i, d in enumerate(days)}
        t = idx[s_dates[-1]]
        assert [bin_of(s_matrix[t, p], quantile_edges(s_matrix[:t, p], 10)) for p in range(n_pos)] == s_draws[-1]
        assert control_market("shares", "synthetic", 1, source=source)[1] == s_draws and control_market("shares", "synthetic", 2, source=source)[1] != s_draws

        h_dates, h_draws, h_days, h_matrix, _ = control_market("shares", "shuffled", 3, source=source, recent=200)
        assert len(h_dates) == 200 and h_dates == real_dates[-200:]
        assert sorted(map(tuple, h_draws)) == sorted(map(tuple, real_draws[-200:])) and h_draws != real_draws[-200:]
        # bins and returns moved together: each shuffled day's returns are a real day's returns whose bins are that day's bins
        real_by_bins = {}
        for d, b in zip(real_dates, real_draws):
            real_by_bins.setdefault(tuple(b), []).append(idx[d])
        for target_date, draw in zip(h_dates, h_draws):
            assert any(np.allclose(h_matrix[idx[target_date]], matrix[i]) for i in real_by_bins[tuple(draw)])
        assert np.allclose(h_matrix[:idx[h_dates[0]]], matrix[:idx[h_dates[0]]]), "the warm-up rows stay in place"
        assert np.allclose(np.sort(h_matrix[:, 0]), np.sort(matrix[:, 0])), "the returns' marginal survives"

        # through the folder, as NullControls builds it
        directory = os.path.join(tmp, "controls", "shares", "synthetic-seed1")
        os.makedirs(directory)
        header, rows = _read_rows_from(source)
        with open(os.path.join(directory, "shares-control.csv"), "w") as handle:
            handle.write(header + "\n")
            for (date, _), draw in zip(rows, s_draws):
                handle.write(";".join([date] + [str(v) for v in draw]) + "\n")
        write_returns_file(s_days, s_matrix, symbols, directory, "shares")
        back = read_game_csv(directory)
        assert [b for _, b in back] == s_draws and read_returns_file(directory)[2] == symbols
        assert sorted(os.listdir(directory)) == ["shares-control.csv", "shares-returns.tsv"]
    print("ControlHistories self-check OK: a market's synthetic control is a matched random walk with equiprobable bins and no "
          "volatility clustering, its shuffled control moves bins and returns together, both carry their returns file")
