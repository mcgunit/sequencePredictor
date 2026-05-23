import numpy as np
from Baselines import Baselines


class Backtester:
    def __init__(self, model):
        self.model = model

    def backtest(
        self,
        start_index=100,
        end_index=None,
        generate_subsets=None,
        skipRows=0,
        skipLastColumns=0,
        years_back=None,
        include_baselines=True,
        verbose=True
    ):
        if generate_subsets is None:
            generate_subsets = []

        numbers, _, _ = self.model.load_numbers(
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            years_back=years_back
        )

        if len(numbers) == 0:
            return []

        if end_index is None:
            end_index = len(numbers)

        results = []

        for i in range(start_index, end_index):
            train = numbers[:i]
            actual = list(map(int, numbers[i]))
            actual_set = set(actual)

            # -------------------------
            # Markov prediction
            # -------------------------
            self.model.build_markov_chain(train)

            history = train[-self.model.markov_order:]

            markov_prediction = self.model.predict_next_numbers(
                history,
                temperature=self.model.softMaxTemperature
            )

            markov_set = set(markov_prediction)

            row = {
                "index": i,
                "actual": sorted(actual),
                "markov_prediction": sorted(markov_prediction),
                "markov_hits": len(markov_set & actual_set),
            }

            voted_prediction, _ = self.model.generate_voted_ticket(
                history,
                n_tickets=500,
                ticket_size=train.shape[1],
                temperature=self.model.softMaxTemperature
            )

            row["markov_voted_prediction"] = voted_prediction
            row["markov_voted_hits"] = len(set(voted_prediction) & actual_set)

            # -------------------------
            # Markov subsets
            # -------------------------
            for subset_size in generate_subsets:
                subset = self.model.generate_best_subset(
                    markov_prediction,
                    subset_size
                )

                row[f"markov_subset_{subset_size}"] = subset
                row[f"markov_subset_{subset_size}_hits"] = len(
                    set(subset) & actual_set
                )

            # -------------------------
            # Baselines
            # -------------------------
            if include_baselines:
                draw_size = train.shape[1]

                random_pred = Baselines.random_ticket(
                    self.model.min_number,
                    self.model.max_number,
                    draw_size
                )

                global_freq_pred = Baselines.global_frequency_ticket(
                    train,
                    draw_size
                )

                column_freq_pred = Baselines.column_frequency_ticket(
                    train
                )

                row["random_prediction"] = random_pred
                row["random_hits"] = len(set(random_pred) & actual_set)

                row["global_frequency_prediction"] = global_freq_pred
                row["global_frequency_hits"] = len(
                    set(global_freq_pred) & actual_set
                )

                row["column_frequency_prediction"] = column_freq_pred
                row["column_frequency_hits"] = len(
                    set(column_freq_pred) & actual_set
                )

            results.append(row)

            if verbose and i % 100 == 0:
                print(f"Progress: {i}/{end_index}")

        return results

    def summarize(self, results, subset_sizes=None):
        if subset_sizes is None:
            subset_sizes = []

        if not results:
            return {}

        summary = {
            "runs": len(results)
        }

        metric_keys = [
            "markov_hits",
            "markov_voted_hits",
            "random_hits",
            "global_frequency_hits",
            "column_frequency_hits"
        ]

        for key in metric_keys:
            values = [r[key] for r in results if key in r]

            if not values:
                continue

            summary[key] = {
                "avg": float(np.mean(values)),
                "median": float(np.median(values)),
                "max": int(np.max(values)),
                "min": int(np.min(values)),
                "std": float(np.std(values)),
                "distribution": self._distribution(values)
            }

        for size in subset_sizes:
            key = f"markov_subset_{size}_hits"
            values = [r[key] for r in results if key in r]

            if values:
                summary[key] = {
                    "avg": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "max": int(np.max(values)),
                    "min": int(np.min(values)),
                    "std": float(np.std(values)),
                    "distribution": self._distribution(values)
                }

        return summary

    def _distribution(self, values):
        dist = {}

        for v in values:
            dist[int(v)] = dist.get(int(v), 0) + 1

        return dict(sorted(dist.items()))