import os, sys, json, time, re, random
import numpy as np
from multiprocessing import Pool, cpu_count

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Metrics import Metrics
from Baselines import Baselines
from Helpers import Helpers

helpers = Helpers()

# Set once per backtest() call, right before the worker Pool is created, so
# forked workers inherit it via copy-on-write - avoids having to pickle model
# instances (several hold defaultdict(lambda: ...) attributes that plain
# pickle can't serialize) through Pool's task/result queues.
_worker_ctx = {}


def _backtest_single_day(i):
    """
    Runs every model + baseline for a single backtest day. Each day is fully
    independent (every model rebuilds itself from scratch using only data
    before `i`), so days can safely run in parallel across worker processes.
    """
    ctx = _worker_ctx
    data_loader_model = ctx["data_loader_model"]
    models = ctx["models"]
    numbers = ctx["numbers"]
    generate_subsets = ctx["generate_subsets"]
    skipLastColumns = ctx["skipLastColumns"]
    include_baselines = ctx["include_baselines"]
    game = ctx["game"]
    special_column_count = ctx["special_column_count"]

    total_rows = len(numbers)

    # Deterministic-but-distinct randomness per day, independent of which
    # worker/order actually processes it (forked workers otherwise start from
    # an identical inherited RNG state - the exact issue HyperoptStatistics.py's
    # "very important" fork/spawn comment documents). Both numpy's RNG and the
    # stdlib random module need reseeding: some models (e.g.
    # MarkovBayesianEnhanced.generate_crossover_combinations) use random.sample/
    # random.uniform directly instead of numpy.
    np.random.seed(i)
    random.seed(i)

    actual = list(map(int, numbers[i]))

    # Important:
    # skipRows tells each isolated model:
    # "ignore the last N rows, so prediction is made using data before row i"
    rows_to_skip = total_rows - i

    row = {
        "index": i,
        "actual": sorted(actual)
    }

    for model_name, model in models.items():
        try:
            predicted_numbers, subsets = helpers.run_model_with_special_column(
                model,
                generateSubsets=generate_subsets,
                skipRows=rows_to_skip,
                skipLastColumns=skipLastColumns,
                specialColumnCount=special_column_count
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

            if game == "pick3":
                profit = helpers.pick3_ticket_profit(predicted_numbers, actual)
                if profit is not None:
                    row[f"{model_name}_profit"] = profit

            if subsets:
                for subset_size, subset in subsets.items():
                    subset = list(map(int, subset))

                    if game == "keno":
                        profit = helpers.keno_ticket_profit(subset, actual)
                        if profit is not None:
                            row[f"{model_name}_subset_{subset_size}_profit"] = profit

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
            data_loader_model.min_number,
            data_loader_model.max_number,
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

    return row


def _print_progress(done, total, start_time, bar_width=30):
    fraction = done / total if total else 1.0
    filled = int(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)

    elapsed = time.time() - start_time
    per_item = elapsed / done if done else 0
    eta = per_item * (total - done)

    print(
        f"\r[{bar}] {done}/{total} ({fraction * 100:5.1f}%) "
        f"elapsed {elapsed:6.1f}s eta {eta:6.1f}s",
        end="",
        flush=True
    )

    if done >= total:
        print()


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
        save_results_path=None,
        game=None,
        special_column_count=0
    ):
        """
        game: "keno" or "pick3" enables profit calculation (in euro) alongside hit
        counts, reusing the same payout tables as Helpers.calculate_profit. Other
        games have no payout model yet, so only hit/matching-number stats apply.

        For "keno", profit is only computed for subsets of 5-10 numbers (in
        generate_subsets) since that's the playable range with real payouts.
        For "pick3", profit is computed on the full (positionally-ordered)
        prediction, since Pick3 payouts depend on digit order.

        special_column_count: for Euromillions (2 star columns), EuroDreams (1
        dream number), VikingLotto (1 super viking) - the trailing special
        column(s) have their own smaller range and must be modeled
        independently rather than mixed into the main numbers - see
        Helpers.run_model_with_special_column. When >0, skipLastColumns should
        be 0 so the special column(s) stay present in `numbers` for
        ground-truth comparison; each model's own run() is then internally
        split into a main-numbers call (skipLastColumns=special_column_count)
        and a specialColumnCount call, combined without merging/re-sorting them.
        """
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

        total_iterations = max(0, end_index - start_index)
        start_time = time.time()

        if total_iterations == 0:
            return []

        # Each backtest day is independent (every model rebuilds itself from
        # scratch using only data before that day), so days run in parallel
        # across a process Pool. Set _worker_ctx as a module global *before*
        # constructing the Pool so forked workers inherit it via copy-on-write -
        # this avoids pickling model instances (several use
        # defaultdict(lambda: ...) attributes plain pickle can't serialize).
        global _worker_ctx
        _worker_ctx = {
            "data_loader_model": self.data_loader_model,
            "models": self.models,
            "numbers": numbers,
            "generate_subsets": generate_subsets,
            "skipLastColumns": skipLastColumns,
            "include_baselines": include_baselines,
            "game": game,
            "special_column_count": special_column_count,
        }

        num_workers = max(1, min(cpu_count()-1, total_iterations))
        results = []

        with Pool(processes=num_workers) as pool:
            for iteration, row in enumerate(
                pool.imap(_backtest_single_day, range(start_index, end_index)),
                start=1
            ):
                results.append(row)

                if verbose:
                    _print_progress(iteration, total_iterations, start_time)

        if save_results_path:
            self.save_results(results, save_results_path)

        return results

    # {model}_subset_{size}_hits / {model}_subset_{size}_profit
    _SUBSET_KEY_RE = re.compile(r"^(?P<model>.+)_subset_(?P<size>\d+)_(?P<metric>hits|profit)$")
    # {model}_hits / {model}_profit / {model}_error (also matches baselines: random_hits, etc.)
    _PLAIN_KEY_RE = re.compile(r"^(?P<model>.+)_(?P<metric>hits|profit|error)$")

    def summarize(self, results, detailed=False, include_distribution=False):
        """
        Groups everything by model instead of spreading model x subset-size
        combinations across dozens of flat top-level keys.

        By default (detailed=False) each model gets just the numbers needed to
        compare models at a glance:
          - "hits_avg" / "profit_total": combined across the model's main
            prediction and all its subsets.
          - "main": {"hits_avg", "profit_total"} for the model's main prediction.
          - "subsets": {size: {"hits_avg", "profit_total"}} per subset size.
          - "errors": only present if that model raised errors.

        detailed=True restores the fuller per-key stats (median/max/min/std/
        thresholds) nested under "hits"/"profit" instead of the two plain
        numbers above - use this when you actually need to dig into one
        specific model/subset rather than compare across all of them.
        include_distribution: only relevant with detailed=True - also emits the
        full per-value hit-count histogram, the single biggest source of bloat.
        """
        if not results:
            return {}

        summary = {"runs": len(results), "models": {}}

        def get_model(name):
            return summary["models"].setdefault(name, {})

        def int_values(key):
            return [row[key] for row in results if key in row and isinstance(row[key], int)]

        def num_values(key):
            return [row[key] for row in results if key in row and isinstance(row[key], (int, float))]

        all_keys = sorted({key for row in results for key in row.keys()})

        hit_avgs = {}
        profit_totals = {}

        for key in all_keys:
            subset_match = self._SUBSET_KEY_RE.match(key)
            if subset_match:
                model_name = subset_match.group("model")
                size = subset_match.group("size")
                metric = subset_match.group("metric")
                model = get_model(model_name)
                subsets = model.setdefault("subsets", {}).setdefault(size, {})

                if metric == "hits":
                    values = int_values(key)
                    if values:
                        avg = float(np.mean(values))
                        subsets["hits"] = Metrics.summarize(values, include_distribution=include_distribution) if detailed else avg
                        hit_avgs.setdefault(model_name, []).append(avg)
                else:
                    values = num_values(key)
                    if values:
                        total = float(np.sum(values))
                        subsets["profit"] = Metrics.summarize_profit(values) if detailed else total
                        profit_totals[model_name] = profit_totals.get(model_name, 0) + total
                continue

            plain_match = self._PLAIN_KEY_RE.match(key)
            if not plain_match:
                continue

            model_name = plain_match.group("model")
            metric = plain_match.group("metric")
            model = get_model(model_name)

            if metric == "hits":
                values = int_values(key)
                if values:
                    avg = float(np.mean(values))
                    model["main"] = model.get("main", {})
                    model["main"]["hits"] = Metrics.summarize(values, include_distribution=include_distribution) if detailed else avg
                    hit_avgs.setdefault(model_name, []).append(avg)
            elif metric == "profit":
                values = num_values(key)
                if values:
                    total = float(np.sum(values))
                    model["main"] = model.get("main", {})
                    model["main"]["profit"] = Metrics.summarize_profit(values) if detailed else total
                    profit_totals[model_name] = profit_totals.get(model_name, 0) + total
            elif metric == "error":
                errors = [row[key] for row in results if key in row]
                model["errors"] = {
                    "count": len(errors),
                    "unique_errors": sorted(set(errors))[:10]
                }

        for model_name, model in summary["models"].items():
            avgs = hit_avgs.get(model_name)
            # Insert hits_avg/profit_total first so they read before "main"/"subsets"/"errors"
            model_ordered = {
                "hits_avg": float(np.mean(avgs)) if avgs else None,
                "profit_total": profit_totals.get(model_name)
            }
            model_ordered.update(model)
            summary["models"][model_name] = model_ordered

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
    from MarkovBayesian import MarkovBayesian
    from MarkovBayesianEnhanched import MarkovBayesianEnhanced
    from PoissonMonteCarlo import PoissonMonteCarlo
    from PoissonMarkov import PoissonMarkov
    from LaplaceMonteCarlo import LaplaceMonteCarlo
    from HybridStatisticalModel import HybridStatisticalModel


    import numpy as np
    np.random.seed(42)


    print("Running global backtest")

    name = "keno"
    generateSubsets = [5, 6, 7, 8, 9, 10]

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
    markov.setGameRange(1, 80)
    markov.setDrawSize(20)
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
    markov_mc_base = Markov()
    markov_mc_base.setDataPath(dataPath)
    markov_mc_base.setGameRange(1, 80)
    markov_mc_base.setDrawSize(20)
    markov_mc_base.setSoftMAxTemperature(0.45)
    markov_mc_base.setAlpha(0.6)
    markov_mc_base.setMinOccurrences(2)
    markov_mc_base.setRecencyWeight(1.7)
    markov_mc_base.setRecencyMode("constant")
    markov_mc_base.setPairDecayFactor(1)
    markov_mc_base.setSortedPrediction(True)
    markov_mc_base.setUsePairScoring(False)
    markov_mc_base.setMarkovOrder(2)

    markov_mc = MarkovMonteCarlo(markov_mc_base)
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
    # Markov Bayesian
    # -------------------------
    markov_bayesian = MarkovBayesian()
    markov_bayesian.setDataPath(dataPath)
    markov_bayesian.setSoftMAxTemperature(0.1)
    markov_bayesian.setAlpha(0.5)
    markov_bayesian.setMinOccurrences(5)

    # -------------------------
    # Markov Bayesian Enhanced
    # -------------------------
    markov_bayesian_enhanced = MarkovBayesianEnhanced()
    markov_bayesian_enhanced.setDataPath(dataPath)
    markov_bayesian_enhanced.setSoftMAxTemperature(0.1)
    markov_bayesian_enhanced.setAlpha(0.5)
    markov_bayesian_enhanced.setMinOccurrences(5)

    # -------------------------
    # Poisson-Markov (blended)
    # -------------------------
    poisson_markov = PoissonMarkov()
    poisson_markov.setDataPath(dataPath)
    poisson_markov.setWeights(poisson_weight=0.5, markov_weight=0.5)
    poisson_markov.setNumberOfSimulations(1000)

    # -------------------------
    # Hybrid Statistical Model
    # -------------------------
    hybrid = HybridStatisticalModel()
    hybrid.setDataPath(dataPath)
    hybrid.setSoftMaxTemperature(0.1)
    hybrid.setAlpha(0.5)
    hybrid.setMinOccurrences(5)
    hybrid.setNumberOfSimulations(5000)

    # -------------------------
    # Backtester
    # -------------------------
    backtester = Backtester(markov)

    backtester.add_model("markov", markov)
    backtester.add_model("markov_mc", markov_mc)
    backtester.add_model("markov_bayesian", markov_bayesian)
    backtester.add_model("markov_bayesian_enhanced", markov_bayesian_enhanced)
    backtester.add_model("poisson_mc", poisson)
    backtester.add_model("poisson_markov", poisson_markov)
    backtester.add_model("laplace_mc", laplace)
    backtester.add_model("hybrid_statistical", hybrid)

    # Backtest only the last ~60 draws (Keno has one draw/day, so 500+ draws
    # of history are available for every prediction well before that window)
    total_rows = len(markov.load_numbers()[0])
    start_index = max(500, total_rows - 60)

    results = backtester.backtest(
        start_index=start_index,
        end_index=total_rows,
        skipLastColumns=0,
        generate_subsets=generateSubsets,
        include_baselines=True,
        verbose=True,
        game=name,
        save_results_path=os.path.join(
            os.path.abspath(os.path.join(path, os.pardir)),
            "test",
            "backtestResults",
            f"{name}_global_backtest.json"
        )
    )

    summary = backtester.summarize(results)

    print(json.dumps(summary, indent=4))