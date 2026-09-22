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
"""

import os

import numpy as np
from dateutil.parser import parse as parse_date

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

from src.Helpers import Helpers
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
    return header, rows


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


def control_draws(game, mode, seed, recent=None):
    """
    (dates, draws) for one control history, in memory: every column, ints,
    oldest first, the real calendar. `mode` is "real", "synthetic" or
    "shuffled".

    `recent` keeps only the newest N draws, and it is applied BEFORE the mode,
    which is load-bearing for the shuffled control. Permuting the whole
    history and then taking its newest N gives a random sample of every era
    the game ever had - measured on lotto's last 600: per-number counts with
    a standard deviation of 13.5 against the real 9.1, chi-square p < 0.0001,
    i.e. a control that no longer shares the marginals it exists to preserve.
    Cutting first and shuffling inside the window keeps "same draws, same
    frequencies, different order" true.
    """
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
    _, draws = control_draws(game, mode, seed, recent=recent)
    directory = os.path.join(root or CONTROL_ROOT, game, f"{mode}-seed{seed}")
    os.makedirs(directory, exist_ok=True)
    for stale in os.listdir(directory):
        os.remove(os.path.join(directory, stale))
    with open(os.path.join(directory, f"{game}-control.csv"), "w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for (date, _), draw in zip(rows, draws):
            handle.write(";".join([date] + [str(value) for value in draw]) + "\n")
    return directory, len(rows)
