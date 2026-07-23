import numpy as np


class Metrics:
    @staticmethod
    def count_hits(predicted, actual):
        predicted_set = set(map(int, predicted))
        actual_set = set(map(int, actual))

        return len(predicted_set & actual_set)

    @staticmethod
    def matching_numbers(predicted, actual):
        predicted_set = set(map(int, predicted))
        actual_set = set(map(int, actual))

        return sorted(predicted_set & actual_set)

    @staticmethod
    def distribution(values):
        dist = {}

        for value in values:
            value = int(value)
            dist[value] = dist.get(value, 0) + 1

        return dict(sorted(dist.items()))

    @staticmethod
    def threshold_summary(values):
        """Rate (not raw count - recoverable via rate * runs) of draws with >= N hits, N=2..6."""
        total = len(values)

        if total == 0:
            return {}

        return {
            f"rate_{n}_or_more": sum(1 for v in values if v >= n) / total
            for n in range(2, 7)
        }

    @staticmethod
    def summarize_profit(values):
        if not values:
            return {}

        return {
            "total": float(np.sum(values)),
            "avg": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }

    @staticmethod
    def summarize(values, include_distribution=False):
        if not values:
            return {}

        summary = {
            "avg": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": int(np.max(values)),
            "min": int(np.min(values)),
            "std": float(np.std(values)),
            "thresholds": Metrics.threshold_summary(values)
        }

        if include_distribution:
            summary["distribution"] = Metrics.distribution(values)

        return summary