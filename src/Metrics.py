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
        total = len(values)

        if total == 0:
            return {}

        hits_2_or_more = sum(1 for v in values if v >= 2)
        hits_3_or_more = sum(1 for v in values if v >= 3)
        hits_4_or_more = sum(1 for v in values if v >= 4)
        hits_5_or_more = sum(1 for v in values if v >= 5)
        hits_6_or_more = sum(1 for v in values if v >= 6)

        return {
            "hits_2_or_more": hits_2_or_more,
            "hits_3_or_more": hits_3_or_more,
            "hits_4_or_more": hits_4_or_more,
            "hits_5_or_more": hits_5_or_more,
            "hits_6_or_more": hits_6_or_more,

            "rate_2_or_more": hits_2_or_more / total,
            "rate_3_or_more": hits_3_or_more / total,
            "rate_4_or_more": hits_4_or_more / total,
            "rate_5_or_more": hits_5_or_more / total,
            "rate_6_or_more": hits_6_or_more / total,
        }

    @staticmethod
    def summarize(values):
        if not values:
            return {}

        return {
            "avg": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": int(np.max(values)),
            "min": int(np.min(values)),
            "std": float(np.std(values)),
            "distribution": Metrics.distribution(values),
            "thresholds": Metrics.threshold_summary(values)
        }