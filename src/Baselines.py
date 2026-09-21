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

        # Not sorted - each entry's position corresponds to its column index,
        # which matters for positional games (Pick3). Callers wanting a tidy
        # display order (Keno) should sort at the point of display, not here.
        return ticket[:num_columns]

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
    prediction. For a positional game (pick3, Joker+) the same computation is
    the per-slot most frequent digit, which on an honest process is noise -
    exactly the comparison that makes a foundation model's per-position
    forecast readable.

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
        ticket = Baselines.column_frequency_ticket(numbers)
        subsets = {size: Baselines.column_frequency_subset(numbers, size) for size in (generateSubsets or [])}
        return (sorted(ticket) if self.sorted_prediction else ticket), subsets

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
