import numpy as np
from collections import defaultdict


class Baselines:
    @staticmethod
    def random_ticket(min_number, max_number, draw_size):
        return sorted(
            np.random.choice(
                np.arange(min_number, max_number + 1),
                size=draw_size,
                replace=False
            ).astype(int).tolist()
        )

    @staticmethod
    def global_frequency_ticket(train_numbers, draw_size):
        freq = defaultdict(float)

        for draw in train_numbers:
            for n in draw:
                freq[int(n)] += 1

        ranked = sorted(freq, key=freq.get, reverse=True)
        return sorted(ranked[:draw_size])

    @staticmethod
    def column_frequency_ticket(train_numbers):
        num_columns = train_numbers.shape[1]
        ticket = []

        for col in range(num_columns):
            freq = defaultdict(float)

            for draw in train_numbers:
                freq[int(draw[col])] += 1

            best = max(freq, key=freq.get)
            ticket.append(best)

        # Remove duplicates while preserving order
        ticket = list(dict.fromkeys(ticket))

        # Fill missing values using global frequency
        if len(ticket) < num_columns:
            global_freq = defaultdict(float)

            for draw in train_numbers:
                for n in draw:
                    global_freq[int(n)] += 1

            ranked_global = sorted(
                global_freq,
                key=global_freq.get,
                reverse=True
            )

            for n in ranked_global:
                if n not in ticket:
                    ticket.append(n)

                if len(ticket) >= num_columns:
                    break

        # Not sorted - callers wanting a tidy display order (Keno) should sort
        # at the point of display, not here. Positional games must not use
        # this ticket at all: the dedupe and back-fill above move values out
        # of their column (positional_column_frequency_ticket).
        return ticket[:num_columns]

    @staticmethod
    def positional_column_frequency_ticket(train_numbers, draw_size=None):
        """
        Each column's own most frequent value (ties -> lowest), kept in its
        slot even when another column picked the same value: the ticket for a
        positional game (pick3, Joker+, the market games), where a repeated
        digit or bin is an ordinary draw. column_frequency_ticket's dedupe
        and pooled back-fill only make sense for a sorted set game. The one
        rule behind the Backtester's positional column_frequency baseline and
        the served OrderStatistics Baseline row (25 Oct 2026).
        """
        train_numbers = np.asarray(train_numbers)
        columns = train_numbers.shape[1] if draw_size is None else min(int(draw_size), train_numbers.shape[1])
        ticket = []
        for col in range(columns):
            counts = defaultdict(float)
            for draw in train_numbers:
                counts[int(draw[col])] += 1
            ticket.append(max(counts, key=lambda n: (counts[n], -n)))
        return ticket

    @staticmethod
    def column_frequency_subset(train_numbers, subset_size):
        """
        A Keno-style playable subset from column_frequency_ticket's full
        per-column-best ticket, ranked by each number's overall historical
        frequency (not column position, since a subset has no fixed slots).
        """
        full_ticket = Baselines.column_frequency_ticket(train_numbers)

        freq = defaultdict(float)
        for draw in train_numbers:
            for n in draw:
                freq[int(n)] += 1

        ranked = sorted(full_ticket, key=lambda n: freq.get(n, 0), reverse=True)
        return sorted(ranked[:subset_size])

class ColumnFrequencyBaseline:
    """
    The order-statistics baseline, as a model-shaped object so it can be
    served and tracked next to the real rows instead of only existing inside
    a backtest (README roadmap item 5).

    For a sorted set game the value drawn at position k is the k-th order
    statistic, and its historical mode is what any model must beat before its
    per-position skill means anything: position 1 of lotto is the minimum of
    six draws from 1-45, so "about 5" is right by arithmetic, not by
    prediction. For a positional game (pick3, Joker+, the market games) the
    same computation is the per-slot most frequent digit, which on an honest
    process is noise - exactly the comparison that makes a foundation model's
    per-position forecast readable. Until 25 Oct 2026 the positional ticket
    went through the set-game column_frequency_ticket, whose dedupe filled a
    slot sharing its mode with an earlier slot from the pooled ranking
    (pick3 served 804 for the mode 800); it is now the per-slot mode with
    repeats kept, the Backtester's rule.

    Same run() contract as the statistical models, so
    Helpers.run_model_with_special_column handles the special columns for it
    like for everything else.
    """

    def __init__(self):
        self.dataPath = ""
        self.recent_draws = 0        # 0 = the whole history
        self.sorted_prediction = True

    def setDataPath(self, dataPath): self.dataPath = dataPath
    def setRecentDraws(self, n): self.recent_draws = max(0, int(n))
    def setSortedPrediction(self, use): self.sorted_prediction = bool(use)
    def clear(self): pass

    def _numbers(self, skipRows, skipLastColumns, specialColumnCount):
        from Helpers import Helpers
        _, _, _, _, _, numbers, _, _ = Helpers().load_data(
            self.dataPath, skipRows=skipRows, skipLastColumns=skipLastColumns,
            specialColumnCount=specialColumnCount)
        numbers = np.array([[int(value) for value in draw] for draw in numbers])
        if self.recent_draws and len(numbers) > self.recent_draws:
            numbers = numbers[-self.recent_draws:]
        return numbers

    def run(self, generateSubsets=[], skipRows=0, skipLastColumns=0, specialColumnCount=0):
        numbers = self._numbers(skipRows, skipLastColumns, specialColumnCount)
        if len(numbers) == 0:
            return [], {}
        subsets = {size: Baselines.column_frequency_subset(numbers, size) for size in (generateSubsets or [])}
        if not self.sorted_prediction:
            return Baselines.positional_column_frequency_ticket(numbers), subsets
        return sorted(Baselines.column_frequency_ticket(numbers)), subsets

    def score_positions(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """Per-position historical distribution - the baseline's own scores."""
        numbers = self._numbers(skipRows, skipLastColumns, specialColumnCount)
        if len(numbers) == 0:
            return []
        labels = list(range(int(numbers.min()), int(numbers.max()) + 1))
        slots = []
        for column in range(numbers.shape[1]):
            counts = defaultdict(float)
            for value in numbers[:, column]:
                counts[int(value)] += 1.0
            total = sum(counts.values()) or 1.0
            slots.append({label: counts.get(label, 0.0) / total for label in labels})
        return slots

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        numbers = self._numbers(skipRows, skipLastColumns, specialColumnCount)
        if len(numbers) == 0:
            return {}
        counts = defaultdict(float)
        for draw in numbers:
            for value in draw:
                counts[int(value)] += 1.0
        total = sum(counts.values()) or 1.0
        return {number: count / total for number, count in counts.items()}


def _self_check():
    import os
    import sys
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # _numbers imports Helpers from src/

    # Slot modes 1, 1, 0 (slots 1 and 3 each tie two values -> the lowest): a positional
    # ticket keeps the repeated 1 in its slot; the set ticket drops it, which
    # shifts slot 3's 0 into slot 2 and back-fills the pooled runner-up 2.
    draws = [[1, 1, 0], [1, 1, 2], [3, 1, 2], [3, 2, 0], [5, 4, 9]]
    assert Baselines.positional_column_frequency_ticket(draws) == [1, 1, 0]
    assert Baselines.positional_column_frequency_ticket(draws, draw_size=2) == [1, 1]
    assert Baselines.column_frequency_ticket(np.array(draws)) == [1, 0, 2], "the set-game ticket is unchanged"

    # The served row on a pick3-shaped folder: positional = per-slot mode with
    # repeats (the 804-for-800 bug of 21 Sept - 25 Oct 2026), set = unchanged.
    with tempfile.TemporaryDirectory() as root:
        folder = os.path.join(root, "pick3")
        os.makedirs(folder)
        rows = [[8, 0, 0]] * 5 + [[8, 9, 3]] * 3 + [[4, 9, 0]] * 2 + [[4, 0, 3]] * 2
        with open(os.path.join(folder, "pick3-gamedata-NL-2026.csv"), "w") as handle:
            handle.write("Datum;Nummer 1;Nummer 2;Nummer 3\n")
            for day, draw in enumerate(rows, start=1):
                handle.write(f"2026-01-{day:02d};" + ";".join(map(str, draw)) + "\n")
        row = ColumnFrequencyBaseline()
        row.setDataPath(folder)
        row.setSortedPrediction(False)
        ticket, subsets = row.run()
        assert ticket == [8, 0, 0] and subsets == {}, ticket
        row.setSortedPrediction(True)
        assert row.run()[0] == sorted(Baselines.column_frequency_ticket(np.array(rows))), "set games keep the old ticket"
    print("Baselines self-check OK: positional tickets are the per-slot mode with repeats kept (ties -> lowest), "
          "set-game tickets unchanged")


if __name__ == "__main__":
    _self_check()
