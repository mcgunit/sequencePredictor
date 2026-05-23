import os, json
from Metrics import Metrics
from Baselines import Baselines


class Backtester:
    def __init__(self, data_loader_model):
        """
        data_loader_model should be one model that can load the full numbers array.

        In your case, using Markov as data_loader_model is fine because it has:
            load_numbers(...)
        """
        self.data_loader_model = data_loader_model
        self.models = {}

    def add_model(self, name, model):
        """
        The model should expose:
            run(generateSubsets=[], skipRows=0, skipLastColumns=0)

        Expected return:
            predicted_numbers, subsets
        """
        self.models[name] = model

    def backtest(
        self,
        start_index=100,
        end_index=None,
        generate_subsets=None,
        skipRows=0,
        skipLastColumns=0,
        years_back=None,
        include_baselines=True,
        verbose=True,
        save_results_path=None
    ):
        if generate_subsets is None:
            generate_subsets = []

        numbers, _, _ = self.data_loader_model.load_numbers(
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            years_back=years_back
        )

        if len(numbers) == 0:
            return []

        total_rows = len(numbers)

        if end_index is None:
            end_index = total_rows

        results = []

        for i in range(start_index, end_index):
            actual = list(map(int, numbers[i]))

            # Important:
            # skipRows tells each isolated model:
            # "ignore the last N rows, so prediction is made using data before row i"
            rows_to_skip = total_rows - i

            row = {
                "index": i,
                "actual": sorted(actual)
            }

            for model_name, model in self.models.items():
                try:
                    predicted_numbers, subsets = model.run(
                        generateSubsets=generate_subsets,
                        skipRows=rows_to_skip,
                        skipLastColumns=skipLastColumns
                    )

                    predicted_numbers = list(map(int, predicted_numbers))

                    row[f"{model_name}_prediction"] = sorted(predicted_numbers)
                    row[f"{model_name}_hits"] = Metrics.count_hits(
                        predicted_numbers,
                        actual
                    )
                    row[f"{model_name}_matching_numbers"] = Metrics.matching_numbers(
                        predicted_numbers,
                        actual
                    )

                    if subsets:
                        for subset_size, subset in subsets.items():
                            subset = list(map(int, subset))

                            row[f"{model_name}_subset_{subset_size}"] = sorted(subset)
                            row[f"{model_name}_subset_{subset_size}_hits"] = Metrics.count_hits(
                                subset,
                                actual
                            )
                            row[f"{model_name}_subset_{subset_size}_matching_numbers"] = Metrics.matching_numbers(
                                subset,
                                actual
                            )

                except Exception as e:
                    row[f"{model_name}_error"] = str(e)

            # -------------------------
            # Baselines
            # -------------------------
            if include_baselines:
                train_numbers = numbers[:i]
                draw_size = len(actual)

                random_prediction = Baselines.random_ticket(
                    self.data_loader_model.min_number,
                    self.data_loader_model.max_number,
                    draw_size
                )

                global_frequency_prediction = Baselines.global_frequency_ticket(
                    train_numbers,
                    draw_size
                )

                column_frequency_prediction = Baselines.column_frequency_ticket(
                    train_numbers
                )

                row["random_prediction"] = sorted(random_prediction)
                row["random_hits"] = Metrics.count_hits(
                    random_prediction,
                    actual
                )
                row["random_matching_numbers"] = Metrics.matching_numbers(
                    random_prediction,
                    actual
                )

                row["global_frequency_prediction"] = sorted(global_frequency_prediction)
                row["global_frequency_hits"] = Metrics.count_hits(
                    global_frequency_prediction,
                    actual
                )
                row["global_frequency_matching_numbers"] = Metrics.matching_numbers(
                    global_frequency_prediction,
                    actual
                )

                row["column_frequency_prediction"] = sorted(column_frequency_prediction)
                row["column_frequency_hits"] = Metrics.count_hits(
                    column_frequency_prediction,
                    actual
                )
                row["column_frequency_matching_numbers"] = Metrics.matching_numbers(
                    column_frequency_prediction,
                    actual
                )

            results.append(row)

            if verbose and i % 100 == 0:
                print(f"Progress: {i}/{end_index}")

        if save_results_path:
            self.save_results(results, save_results_path)

        return results

    def summarize(self, results):
        if not results:
            return {}

        summary = {
            "runs": len(results)
        }

        # Collect hit keys from ALL rows, not only first row
        hit_keys = sorted({
            key
            for row in results
            for key in row.keys()
            if key.endswith("_hits")
        })

        for key in hit_keys:
            values = [
                row[key]
                for row in results
                if key in row and isinstance(row[key], int)
            ]

            if values:
                summary[key] = Metrics.summarize(values)

        # Optional: collect model errors
        error_keys = sorted({
            key
            for row in results
            for key in row.keys()
            if key.endswith("_error")
        })

        if error_keys:
            summary["errors"] = {}

            for key in error_keys:
                errors = [
                    row[key]
                    for row in results
                    if key in row
                ]

                summary["errors"][key] = {
                    "count": len(errors),
                    "unique_errors": sorted(set(errors))[:10]
                }

        return summary

    def save_results(self, results, path):
        folder = os.path.dirname(path)

        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        with open(path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    import os, json

    from Markov import Markov
    from MarkovMonteCarlo import MarkovMonteCarlo
    from PoissonMonteCarlo import PoissonMonteCarlo
    from LaplaceMonteCarlo import LaplaceMonteCarlo
    
    print("Running global backtest")

    name = "lotto"
    generateSubsets = []

    path = os.getcwd()
    dataPath = os.path.join(
        os.path.abspath(os.path.join(path, os.pardir)),
        "test",
        "trainingData",
        name
    )

    # -------------------------
    # Markov
    # -------------------------
    markov = Markov()
    markov.setDataPath(dataPath)
    markov.setGameRange(1, 45)
    markov.setDrawSize(6)
    markov.setSoftMAxTemperature(0.45)
    markov.setAlpha(0.6)
    markov.setMinOccurrences(2)
    markov.setRecencyWeight(1.7)
    markov.setRecencyMode("constant")
    markov.setPairDecayFactor(1)
    markov.setSortedPrediction(True)
    markov.setUsePairScoring(False)
    markov.setMarkovOrder(2)

    # -------------------------
    # Markov Monte Carlo / voted-ticket Markov
    # -------------------------
    markov_mc = MarkovMonteCarlo(markov)
    markov_mc.setNumOfSimulations(250)

    # -------------------------
    # Poisson Monte Carlo
    # -------------------------
    poisson = PoissonMonteCarlo()
    poisson.setDataPath(dataPath)
    poisson.setNumOfSimulations(1000)
    poisson.setRecentDraws(500)
    poisson.setWeightFactor(1.0)

    # -------------------------
    # Laplace Monte Carlo
    # -------------------------
    laplace = LaplaceMonteCarlo()
    laplace.setDataPath(dataPath)
    laplace.setNumOfSimulations(1000)
    laplace.setRecentDraws(500)

    # -------------------------
    # Backtester
    # -------------------------
    backtester = Backtester(markov)

    backtester.add_model("markov", markov)
    backtester.add_model("markov_mc", markov_mc)
    backtester.add_model("poisson_mc", poisson)
    backtester.add_model("laplace_mc", laplace)

    results = backtester.backtest(
        start_index=200,
        skipLastColumns=1,
        generate_subsets=generateSubsets,
        include_baselines=True,
        verbose=True,
        save_results_path=os.path.join(
            os.path.abspath(os.path.join(path, os.pardir)),
            "test",
            "backtestResults",
            f"{name}_global_backtest.json"
        )
    )

    summary = backtester.summarize(results)

    print(json.dumps(summary, indent=4))