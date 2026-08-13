import os, sys
import xgboost as xgb
import numpy as np
import joblib
from typing import List
from collections import defaultdict

# Dynamically adjust the import path for Helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

# Ensure Helpers can be imported
if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Helpers import Helpers

helpers = Helpers()


class XGBoostPredictor:
    """
    Gradient boosting (XGBoost) as a first-class prediction model, on the same
    interface every statistical model in src/ exposes, so Predictor.py,
    Backtester.py and HyperoptBoost.py can drive it exactly like Markov,
    PoissonMarkov, HybridStatisticalModel, ...:

        setDataPath / setSortedPrediction / clear
        run(generateSubsets=[], skipRows=0, skipLastColumns=0, specialColumnCount=0)
            -> (prediction, {subset_size: [numbers]})
        score_numbers(skipRows=0, skipLastColumns=0, specialColumnCount=0)
            -> {number: score}

    Modelling approach: one multiclass classifier per draw position, over a
    flattened window of the `n_previous_draws` preceding draws. That's
    unchanged - what changed is everything around it:

    - run() now accepts skipLastColumns/specialColumnCount/years_back, so it
      works with Helpers.run_model_with_special_column (Euromillions' 2 star
      columns, EuroDreams' dream number, VikingLotto's super viking are modeled
      independently from the main numbers, and Lotto's unplayed bonus number is
      dropped) instead of always training on every raw column.
    - subsets are returned as a {size: [numbers]} dict, like every other model,
      instead of a bare list.
    - score_numbers() exposes the per-number probability, so XGBoost can feed
      Backtester.collect_scores / the stacking meta-learners the same way the
      statistical models do.
    - setSortedPrediction() - non-positional games get a sorted ticket of
      *distinct* numbers (the old per-position argmax could - and for Keno
      routinely did - return the same number several times, wasting slots),
      while Pick3 keeps digits in drawn order, duplicates included.
    - labels are encoded through the game's actual observed label set instead
      of the old "subtract 1" offset trick, which silently assumed a 1..N
      contiguous range and mismatched train features (raw values) against
      predict features (values - 1).
    """

    def __init__(self):
        self.lengthOfDraw = 20
        self.dataPath = ""
        self.modelPath = ""
        self.n_previous_draws = 5
        self.n_estimators = 100
        self.max_depth = 5
        self.learning_rate = 0.1
        self.subsample = 1.0
        self.colsample_bytree = 1.0
        self.min_child_weight = 1.0
        self.reg_lambda = 1.0
        self.models = []
        self.labels = []
        # Memo of the last fit (see _ensure_fitted): Backtester with
        # collect_scores=True asks the same model for run() and score_numbers()
        # on the same day with the same parameters, and both need the same fit.
        self._fit_key = None
        self._fit_state = None
        self.top_k = 5
        self.force_nested = False
        self.subset_selection_mode = "softmax"
        self.subset_temperature = 0.5
        self.sorted_prediction = True  # Set False for positional games like Pick3
        self.save_models = False       # Opt-in: Backtester runs many days in parallel
        self.num_threads = 1           # 1 by default, see setNumThreads
        # Kept only for backwards compatibility with existing callers - label
        # encoding (see _encode/_decode) now handles any number range, including
        # Pick3's 0-9, so no offset is needed anywhere.
        self.offsetByOne = True

    # --- SETTERS ---
    def setDataPath(self, dataPath): self.dataPath = dataPath
    def setModelPath(self, modelPath): self.modelPath = modelPath
    def setPreviousDraws(self, nPreviousDraws): self.n_previous_draws = int(nPreviousDraws)
    def setEstimators(self, nEstimators): self.n_estimators = int(nEstimators)
    def setMaxDepth(self, maxDepth): self.max_depth = int(maxDepth)
    def setLearningRate(self, learningRate): self.learning_rate = float(learningRate)
    def setSubsample(self, subsample): self.subsample = float(subsample)
    def setColsampleByTree(self, colsample): self.colsample_bytree = float(colsample)
    def setMinChildWeight(self, minChildWeight): self.min_child_weight = float(minChildWeight)
    def setRegLambda(self, regLambda): self.reg_lambda = float(regLambda)
    def setTopK(self, topK): self.top_k = int(topK)
    def setForceNested(self, forceNested): self.force_nested = bool(forceNested)
    def setSubsetSelectionMode(self, mode): self.subset_selection_mode = mode
    def setSubsetTemperature(self, temperature): self.subset_temperature = float(temperature)
    def setSortedPrediction(self, use): self.sorted_prediction = bool(use)
    def setSaveModels(self, save): self.save_models = bool(save)
    def setNumThreads(self, numThreads): self.num_threads = max(1, int(numThreads))
    def setOffsetByOne(self, offset): self.offsetByOne = bool(offset)

    def setLengtOfDraw(self, lengthOfDraw):
        self.lengthOfDraw = int(lengthOfDraw)

    def clear(self):
        """Same contract as the statistical models' clear(): drop all fitted state."""
        self.models = []
        self.labels = []
        self._fit_key = None
        self._fit_state = None

    def _fit_signature(self, skipRows, skipLastColumns, specialColumnCount, years_back):
        """
        Everything that changes what a fit produces: the data slice plus every
        training hyperparameter. Deliberately excludes top_k / sorted_prediction
        / the subset knobs, which only affect how an existing fit is read out.
        Any setter call that matters therefore invalidates the cache by simply
        changing this key - no manual invalidation to forget.
        """
        return (
            self.dataPath, skipRows, skipLastColumns, specialColumnCount, years_back,
            self.n_previous_draws, self.n_estimators, self.max_depth, self.learning_rate,
            self.subsample, self.colsample_bytree, self.min_child_weight, self.reg_lambda,
            self.num_threads,
        )

    def _ensure_fitted(self, skipRows, skipLastColumns, specialColumnCount, years_back):
        """
        Loads + fits, or returns the previous (draws, window) if this exact
        request was just served. Backtester with collect_scores=True calls
        run() and score_numbers() back to back per day with identical
        parameters (twice more again for a special-column game's second
        range), and unlike the statistical models - whose fit is a cheap
        frequency/transition count - refitting XGBoost is the entire cost of
        the model. One cached entry is enough: callers work through one
        (day, range) at a time.
        """
        key = self._fit_signature(skipRows, skipLastColumns, specialColumnCount, years_back)
        if self._fit_key == key and self._fit_state is not None:
            return self._fit_state

        numbers, unique_labels = self.load_numbers(
            skipRows=skipRows, skipLastColumns=skipLastColumns,
            years_back=years_back, specialColumnCount=specialColumnCount)

        if len(numbers) == 0:
            self._fit_key, self._fit_state = key, None
            return None

        draws = [[int(n) for n in draw] for draw in numbers]
        _, window = self.fit(draws, unique_labels=unique_labels)

        self._fit_key = key
        self._fit_state = (draws, window)
        return self._fit_state

    # --- LABEL ENCODING ---
    def _build_labels(self, draws, unique_labels=None):
        """
        Class list for every per-position classifier: the game's own label set
        (Helpers.load_data's unique_labels, which covers the whole data folder)
        unioned with whatever actually appears in this training slice, so a
        value present in the slice can never fall outside the class space.
        """
        values = set(int(n) for draw in draws for n in draw)
        if unique_labels is not None:
            values |= set(int(label) for label in unique_labels)
        self.labels = sorted(values)

    def _encode(self, value):
        return self.labels.index(int(value))

    def _decode(self, index):
        return int(self.labels[int(index)])

    # --- DATA ---
    def load_numbers(self, skipRows=0, skipLastColumns=0, years_back=None, specialColumnCount=0):
        _, _, _, _, _, numbers, _, unique_labels = helpers.load_data(
            self.dataPath,
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            years_back=years_back,
            specialColumnCount=specialColumnCount
        )
        return numbers, unique_labels

    def _prepare_data(self, draws: List[List[int]], window: int):
        """
        X: the `window` preceding draws, flattened, as raw number values.
        Y[pos]: the encoded label drawn at position `pos`.

        Features are raw values here *and* in predict() - the previous version
        trained on raw values but predicted on value-1, so every live
        prediction was made on inputs shifted one step away from anything the
        model had ever seen.
        """
        X, Y = [], [[] for _ in range(self.lengthOfDraw)]
        for i in range(window, len(draws)):
            X.append(np.asarray(draws[i - window:i], dtype=float).flatten())
            for pos in range(self.lengthOfDraw):
                Y[pos].append(self._encode(draws[i][pos]))
        return np.array(X), [np.array(y) for y in Y]

    def _effective_window(self, draws):
        """
        n_previous_draws can be tuned well above what a short history (or a
        deep skipRows during a backtest) actually supports - clamp it instead
        of raising, so a model that simply has less history still predicts.
        """
        return max(1, min(self.n_previous_draws, len(draws) - 1))

    # --- TRAINING ---
    def fit(self, draws: List[List[int]], unique_labels=None):
        draws = [[int(n) for n in draw] for draw in draws]

        if len(draws) < 2:
            raise ValueError("Need at least 2 draws to train XGBoost.")

        self.setLengtOfDraw(len(draws[0]))
        self._build_labels(draws, unique_labels)

        window = self._effective_window(draws)
        X, Ys = self._prepare_data(draws, window)
        num_classes = len(self.labels)

        self.models = []
        for pos in range(self.lengthOfDraw):
            y_pos = Ys[pos]
            X_pos = X
            weights = np.ones(len(y_pos), dtype=float)

            # XGBoost's multi:softprob requires every class 0..num_classes-1 to
            # be present. Pad the missing ones with a copy of the first row at a
            # near-zero sample weight, so the class exists without meaningfully
            # polluting the fit (the previous version padded at full weight).
            missing_labels = sorted(set(range(num_classes)) - set(y_pos.tolist()))
            if missing_labels:
                X_pos = np.vstack([X_pos] + [X[0]] * len(missing_labels))
                y_pos = np.append(y_pos, missing_labels)
                weights = np.append(weights, np.full(len(missing_labels), 1e-6))

            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                eval_metric="mlogloss",
                # 1 by default: Backtester/Predictor already run many days (or
                # many Optuna trials) across a process Pool, and letting each
                # of those processes spawn cpu_count() XGBoost threads
                # oversubscribes the machine badly. Callers running XGBoost
                # single-process raise this via setNumThreads.
                n_jobs=self.num_threads
            )

            model.fit(X_pos, y_pos, sample_weight=weights)
            self.models.append(model)

        return X, window

    # --- INFERENCE ---
    def _position_probabilities(self, draws, window):
        """
        Per-position class-probability matrix for the next draw, shape
        (lengthOfDraw, num_classes).
        """
        features = np.asarray(draws[-window:], dtype=float).flatten().reshape(1, -1)
        return np.array([model.predict_proba(features)[0] for model in self.models])

    def _average_confidence(self, position_probabilities, top_k=None):
        """
        {number: average probability} across draw positions. With top_k set,
        only each position's top_k most probable numbers contribute (the tuned
        behaviour run() uses); without it, every number gets a score - needed by
        score_numbers(), whose consumers (the stacking meta-learners,
        Backtester.collect_scores) expect the full number range covered.
        """
        totals = defaultdict(lambda: {"total": 0.0, "count": 0})

        for probs in position_probabilities:
            if top_k:
                indices = np.argsort(probs)[-top_k:][::-1]
            else:
                indices = range(len(probs))

            for index in indices:
                number = self._decode(index)
                totals[number]["total"] += float(probs[index])
                totals[number]["count"] += 1

        return {number: value["total"] / value["count"] for number, value in totals.items()}

    def _build_ticket(self, position_probabilities, ranked_candidates):
        """
        Positional games (Pick3, sorted_prediction=False) keep the per-position
        argmax in drawn order, duplicates and all - that's a valid result there.

        Every other game needs `lengthOfDraw` *distinct* numbers: taking the
        per-position argmax alone routinely collided (several positions picking
        the same number), which silently shrank the real ticket. Collisions are
        refilled from the next-best candidates by average confidence.
        """
        picks = [self._decode(int(np.argmax(probs))) for probs in position_probabilities]

        if not self.sorted_prediction:
            return picks

        ticket, used = [], set()
        for number in picks:
            if number not in used:
                ticket.append(number)
                used.add(number)

        for number in ranked_candidates:
            if len(ticket) >= self.lengthOfDraw:
                break
            if number not in used:
                ticket.append(number)
                used.add(number)

        return sorted(ticket)

    def generate_best_subset(self, number_scores, ticket_numbers, nSubset):
        """
        Same shared subset generator every ensemble row uses
        (Helpers.generate_subset_from_scores) rather than XGBoost's own
        hand-rolled nesting loop. force_nested maps to mode="top": a
        deterministic top-N slice is inherently nested (the 5-subset is
        contained in the 6-subset, ...), which is exactly what that flag asked
        for.
        """
        mode = "top" if self.force_nested else self.subset_selection_mode
        return helpers.generate_subset_from_scores(
            number_scores, ticket_numbers, nSubset,
            mode=mode, temperature=self.subset_temperature)

    def predict(self, draws: List[List[int]]) -> List[int]:
        """Ticket only, from an already-fitted model."""
        window = self._effective_window(draws)
        position_probabilities = self._position_probabilities(draws, window)
        number_scores = self._average_confidence(position_probabilities, top_k=self.top_k)
        ranked = sorted(number_scores, key=number_scores.get, reverse=True)
        return self._build_ticket(position_probabilities, ranked)

    # --- PERSISTENCE ---
    def _model_folder(self, specialColumnCount=0):
        """
        Main and special-column models are two different fits over two
        different number ranges (see Helpers.run_model_with_special_column), so
        they must not overwrite each other on disk.
        """
        return os.path.join(self.modelPath, "special" if specialColumnCount > 0 else "main")

    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        for i, model in enumerate(self.models):
            joblib.dump(model, os.path.join(folder_path, f"model_pos_{i}.joblib"))
        joblib.dump(self.labels, os.path.join(folder_path, "labels.joblib"))

    def load(self, folder_path: str):
        """Loads however many positions were saved, instead of assuming 20 (Keno's draw size)."""
        self.labels = joblib.load(os.path.join(folder_path, "labels.joblib"))
        self.models = []
        for i in range(self.lengthOfDraw):
            model_path = os.path.join(folder_path, f"model_pos_{i}.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.models.append(joblib.load(model_path))

    # --- PUBLIC INTERFACE (matches every statistical model in src/) ---
    def run(self, generateSubsets=None, skipRows=0, skipLastColumns=0, specialColumnCount=0, years_back=None):
        if generateSubsets is None:
            generateSubsets = []

        fitted = self._ensure_fitted(skipRows, skipLastColumns, specialColumnCount, years_back)
        if fitted is None:
            return [], {}

        draws, window = fitted

        if self.save_models and self.modelPath:
            try:
                self.save(self._model_folder(specialColumnCount))
            except Exception as e:
                print("Failed to save XGBoost models: ", e)

        position_probabilities = self._position_probabilities(draws, window)
        number_scores = self._average_confidence(position_probabilities, top_k=self.top_k)
        ranked = sorted(number_scores, key=number_scores.get, reverse=True)
        prediction = self._build_ticket(position_probabilities, ranked)

        subsets = {}
        for subset_size in generateSubsets:
            subsets[subset_size] = self.generate_best_subset(number_scores, prediction, subset_size)

        return prediction, subsets

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0, years_back=None):
        """
        Per-number score for stacking (Phase 1) / Backtester.collect_scores:
        the average predicted probability each number receives across all draw
        positions - the same quantity run() ranks its ticket by, just returned
        in full instead of collapsed into one ticket, and without the top_k cut
        so every number in range gets a score.
        """
        fitted = self._ensure_fitted(skipRows, skipLastColumns, specialColumnCount, years_back)
        if fitted is None:
            return {}

        draws, window = fitted

        return self._average_confidence(self._position_probabilities(draws, window))

    def save_average_confidence_plot(self, avg_confidences, filename="average_confidence_per_number.png"):
        # matplotlib is imported here, not at module level: this file is
        # imported inside every Backtester worker process, and this debugging
        # helper is the only thing that ever needed it.
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.bar(list(avg_confidences.keys()), list(avg_confidences.values()), color='skyblue')
        plt.xlabel("Number")
        plt.ylabel("Average Confidence")
        plt.title("Average Confidence per Number")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


# Backwards-compatible alias: the class was named after Keno back when it only
# ever ran there. Existing imports keep working.
XGBoostKenoPredictor = XGBoostPredictor


if __name__ == "__main__":
    print("Trying XGBoost")

    xgboost = XGBoostPredictor()

    name = 'vikinglotto'
    generateSubsets = []
    specialColumnCount = {"euromillions": 2, "eurodreams": 1, "vikinglotto": 1}.get(name, 0)
    skipLastColumns = 1 if name == "lotto" else 0

    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", f"xgboost_{name}_models")

    xgboost.setDataPath(dataPath)
    xgboost.setModelPath(modelPath)
    xgboost.setSortedPrediction(not ("pick3" in name))

    if "keno" in name:
        generateSubsets = [6, 7]

    predicted_numbers, subsets = helpers.run_model_with_special_column(
        xgboost, generateSubsets=generateSubsets,
        skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

    print("Predicted Numbers: ", predicted_numbers)
    print("Generated Subsets: ", subsets)
