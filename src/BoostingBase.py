import os, sys
import numpy as np
import joblib
from typing import List
from collections import defaultdict

# Dynamically adjust the import path for Helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Helpers import Helpers

helpers = Helpers()


# Maps a bestParams/Optuna key suffix to the setter that consumes it, and the
# default used when the key is absent. Shared by Predictor.py (reading tuned
# values out of bestParams_<game>.json) and HyperoptBoost.py (writing them in
# from trial suggestions), so the two can never drift apart on a key name.
#
# Full key = f"{prefix}{suffix}", e.g. prefix "xgBoost" -> "xgBoostEstimators".
# Note "Maxdepth" (not "MaxDepth"): that's the spelling already persisted in
# every bestParams_<game>.json, and renaming it would orphan tuned values.
BOOSTING_PARAM_SUFFIXES = [
    ("Estimators", "setEstimators", 200),
    ("LearningRate", "setLearningRate", 0.1),
    ("Maxdepth", "setMaxDepth", 3),
    ("PreviousDraws", "setPreviousDraws", 11),
    ("TopK", "setTopK", 16),
    ("ForceNested", "setForceNested", True),
    ("Subsample", "setSubsample", 1.0),
    ("ColsampleByTree", "setColsampleByTree", 1.0),
    ("MinChildWeight", "setMinChildWeight", 1.0),
    ("RegLambda", "setRegLambda", 1.0),
    ("SubsetMode", "setSubsetSelectionMode", "softmax"),
    ("SubsetTemperature", "setSubsetTemperature", 0.5),
]


def apply_boosting_params(model, params, prefix):
    """
    Applies every tuned boosting parameter found under `prefix` to `model`,
    falling back to the shared default for any key the params dict doesn't
    have - so a bestParams_<game>.json written before a given knob existed
    doesn't raise, it just keeps that knob's default.
    """
    for suffix, setter, default in BOOSTING_PARAM_SUFFIXES:
        getattr(model, setter)(params.get(f"{prefix}{suffix}", default))
    return model


class BoostingPredictorBase:
    """
    Everything the gradient-boosting models share, so XGBoost / LightGBM /
    CatBoost are one implementation with a swappable backend rather than three
    near-identical files that drift apart.

    Subclasses supply only the library-specific parts:
        _make_classifier(num_classes)  -> an unfitted sklearn-API classifier
        _fit_classifier(model, X, y, sample_weight)
        library_name                   -> used in log messages

    Everything else - the same interface every statistical model in src/
    exposes - lives here:
        setDataPath / setSortedPrediction / clear
        run(generateSubsets, skipRows, skipLastColumns, specialColumnCount, years_back)
            -> (prediction, {subset_size: [numbers]})
        score_numbers(...) -> {number: score}

    Two formulations subclass this, and both are tracked as separate
    prediction rows so they can be compared directly:

    - PerPositionBoostingPredictor: one multiclass classifier per draw
      position. Models "which number lands in slot i", which genuinely fits a
      positional game (Pick3) but imposes slot identity on games that have
      none (Keno's 20 numbers are unordered).
    - MultiLabelBoostingPredictor: one binary classifier per number in the
      game's range, answering "is this number in the next draw". Matches the
      structure of a non-positional game directly, and is far cheaper: Keno
      goes from 20 multiclass fits over 80 classes to 80 binary fits.
    """

    library_name = "boosting"

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
        self._fit_key = None
        self._fit_state = None
        self.top_k = 5
        self.force_nested = False
        self.subset_selection_mode = "softmax"
        self.subset_temperature = 0.5
        self.sorted_prediction = True  # Set False for positional games like Pick3
        self.save_models = False       # Opt-in: Backtester runs many days in parallel
        self.num_threads = 1           # 1 by default, see setNumThreads

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

    def setLengtOfDraw(self, lengthOfDraw):
        self.lengthOfDraw = int(lengthOfDraw)

    def clear(self):
        """Same contract as the statistical models' clear(): drop all fitted state."""
        self.models = []
        self.labels = []
        self._fit_key = None
        self._fit_state = None

    # --- BACKEND HOOKS ---
    def _make_classifier(self, num_classes):
        raise NotImplementedError

    def _fit_classifier(self, model, X, y, sample_weight=None):
        model.fit(X, y, sample_weight=sample_weight)
        return model

    # --- LABEL ENCODING ---
    def _build_labels(self, draws, unique_labels=None):
        """
        Class list for the fit: the game's own label set (Helpers.load_data's
        unique_labels, which covers the whole data folder) unioned with
        whatever appears in this training slice, so a value present in the
        slice can never fall outside the class space. Derived from data rather
        than a hardcoded range, so Pick3's 0-9 and a special column's own small
        range work without special-casing.
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

    def _effective_window(self, draws):
        """
        n_previous_draws can be tuned well above what a short history (or a
        deep skipRows during a backtest) actually supports - clamp instead of
        raising, so a model with less history still predicts.
        """
        return max(1, min(self.n_previous_draws, len(draws) - 1))

    # --- FIT CACHE ---
    def _fit_signature(self, skipRows, skipLastColumns, specialColumnCount, years_back):
        """
        Everything that changes what a fit produces: the data slice plus every
        training hyperparameter. Excludes top_k / sorted_prediction / the
        subset knobs, which only affect how an existing fit is read out. Any
        setter call that matters invalidates the cache by changing this key -
        nothing to remember to invalidate by hand.
        """
        return (
            type(self).__name__, self.dataPath, skipRows, skipLastColumns, specialColumnCount, years_back,
            self.n_previous_draws, self.n_estimators, self.max_depth, self.learning_rate,
            self.subsample, self.colsample_bytree, self.min_child_weight, self.reg_lambda,
            self.num_threads,
        )

    def _ensure_fitted(self, skipRows, skipLastColumns, specialColumnCount, years_back):
        """
        Loads + fits, or returns the previous (draws, window) if this exact
        request was just served. Backtester with collect_scores=True calls
        run() and score_numbers() back to back per day with identical
        parameters (and twice more for a special-column game's second range).
        Unlike the statistical models - whose fit is a cheap frequency count -
        refitting a boosted ensemble is the model's entire cost.
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
        window = self.fit(draws, unique_labels=unique_labels)

        self._fit_key = key
        self._fit_state = (draws, window)
        return self._fit_state

    # --- SUBSETS ---
    def generate_best_subset(self, number_scores, ticket_numbers, nSubset):
        """
        The shared subset generator every ensemble row uses
        (Helpers.generate_subset_from_scores). force_nested maps to
        mode="top": a deterministic top-N slice is inherently nested (the
        5-subset is contained in the 6-subset, and so on), which is what that
        flag asked for.
        """
        mode = "top" if self.force_nested else self.subset_selection_mode
        return helpers.generate_subset_from_scores(
            number_scores, ticket_numbers, nSubset,
            mode=mode, temperature=self.subset_temperature)

    # --- PERSISTENCE ---
    def _model_folder(self, specialColumnCount=0):
        """
        Main and special-column models are two different fits over two
        different number ranges (see Helpers.run_model_with_special_column),
        so they must not overwrite each other on disk.
        """
        return os.path.join(self.modelPath, "special" if specialColumnCount > 0 else "main")

    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        for i, model in enumerate(self.models):
            joblib.dump(model, os.path.join(folder_path, f"model_{i}.joblib"))
        joblib.dump(self.labels, os.path.join(folder_path, "labels.joblib"))

    def load(self, folder_path: str):
        self.labels = joblib.load(os.path.join(folder_path, "labels.joblib"))
        self.models = []
        i = 0
        while True:
            model_path = os.path.join(folder_path, f"model_{i}.joblib")
            if not os.path.exists(model_path):
                break
            self.models.append(joblib.load(model_path))
            i += 1
        if not self.models:
            raise FileNotFoundError(f"No model files found in: {folder_path}")

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
                print(f"Failed to save {self.library_name} models: ", e)

        prediction, number_scores = self._predict(draws, window)

        subsets = {}
        for subset_size in generateSubsets:
            subsets[subset_size] = self.generate_best_subset(number_scores, prediction, subset_size)

        return prediction, subsets

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0, years_back=None):
        """
        Per-number score for stacking (Phase 1) / Backtester.collect_scores,
        covering the whole number range so the meta-learner sees every
        candidate.
        """
        fitted = self._ensure_fitted(skipRows, skipLastColumns, specialColumnCount, years_back)
        if fitted is None:
            return {}

        draws, window = fitted
        return self._score(draws, window)

    # --- FORMULATION HOOKS ---
    def fit(self, draws, unique_labels=None):
        raise NotImplementedError

    def _predict(self, draws, window):
        """Returns (ticket, {number: score})."""
        raise NotImplementedError

    def _score(self, draws, window):
        """Returns {number: score} over the full range."""
        raise NotImplementedError


class PerPositionBoostingPredictor(BoostingPredictorBase):
    """
    One multiclass classifier per draw position, over a flattened window of the
    n_previous_draws preceding draws (raw values). The original formulation -
    the only one that can represent a positional game like Pick3, where slot
    identity is real and digits may repeat.
    """

    def _prepare_data(self, draws: List[List[int]], window: int):
        X, Y = [], [[] for _ in range(self.lengthOfDraw)]
        for i in range(window, len(draws)):
            X.append(np.asarray(draws[i - window:i], dtype=float).flatten())
            for pos in range(self.lengthOfDraw):
                Y[pos].append(self._encode(draws[i][pos]))
        return np.array(X), [np.array(y) for y in Y]

    def fit(self, draws, unique_labels=None):
        draws = [[int(n) for n in draw] for draw in draws]

        if len(draws) < 2:
            raise ValueError(f"Need at least 2 draws to train {self.library_name}.")

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

            # Multiclass boosting requires every class 0..num_classes-1 to be
            # present. Pad missing ones with a copy of the first row at a
            # near-zero sample weight, so the class exists without
            # meaningfully polluting the fit.
            missing_labels = sorted(set(range(num_classes)) - set(y_pos.tolist()))
            if missing_labels:
                X_pos = np.vstack([X_pos] + [X[0]] * len(missing_labels))
                y_pos = np.append(y_pos, missing_labels)
                weights = np.append(weights, np.full(len(missing_labels), 1e-6))

            model = self._make_classifier(num_classes)
            self._fit_classifier(model, X_pos, y_pos, sample_weight=weights)
            self.models.append(model)

        return window

    def _position_probabilities(self, draws, window):
        """Per-position class-probability matrix, shape (lengthOfDraw, num_classes)."""
        features = np.asarray(draws[-window:], dtype=float).flatten().reshape(1, -1)
        return np.array([model.predict_proba(features)[0] for model in self.models])

    def _average_confidence(self, position_probabilities, top_k=None):
        """
        {number: average probability} across draw positions. With top_k set,
        only each position's top_k most probable numbers contribute (the tuned
        read-out run() uses); without it every number gets a score, which
        score_numbers() needs so the meta-learner sees the full range.
        """
        totals = defaultdict(lambda: {"total": 0.0, "count": 0})

        for probs in position_probabilities:
            indices = np.argsort(probs)[-top_k:][::-1] if top_k else range(len(probs))
            for index in indices:
                number = self._decode(index)
                totals[number]["total"] += float(probs[index])
                totals[number]["count"] += 1

        return {number: value["total"] / value["count"] for number, value in totals.items()}

    def _build_ticket(self, position_probabilities, ranked_candidates):
        """
        Positional games (sorted_prediction=False) keep the per-position argmax
        in drawn order, duplicates included - a valid result there.

        Every other game needs lengthOfDraw *distinct* numbers: the
        per-position argmax alone routinely collides (several positions picking
        the same number), silently shrinking the real ticket. Collisions are
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

    def _predict(self, draws, window):
        position_probabilities = self._position_probabilities(draws, window)
        number_scores = self._average_confidence(position_probabilities, top_k=self.top_k)
        ranked = sorted(number_scores, key=number_scores.get, reverse=True)
        return self._build_ticket(position_probabilities, ranked), number_scores

    def _score(self, draws, window):
        return self._average_confidence(self._position_probabilities(draws, window))

    def predict(self, draws: List[List[int]]) -> List[int]:
        """Ticket only, from an already-fitted model."""
        return self._predict(draws, self._effective_window(draws))[0]


class MultiLabelBoostingPredictor(BoostingPredictorBase):
    """
    One binary classifier per number in the game's range: "is this number in
    the next draw?". No slot identity is imposed, which matches how a
    non-positional game (Keno, Lotto, Euromillions, ...) actually works - see
    the README's "within-draw structure versus temporal predictability" note.

    Features are the multi-hot encoding of each draw in the window (one
    indicator per number per past draw) rather than the per-position model's
    raw sorted values. Raw sorted values encode "the 3rd smallest number was
    17", which is order-statistic structure, not membership; multi-hot encodes
    "17 was drawn 2 draws ago", which is what a membership question needs.

    Not meaningful for Pick3: digit order is part of the result and digits
    repeat, neither of which a set-membership model can represent. Callers
    should skip it there, the same way they skip WeightedEnsemble/MetaLearner.

    Cost note: this is much cheaper than the per-position formulation. Keno
    goes from 20 multiclass fits over 80 classes (which boost one tree group
    per class per round internally) to 80 plain binary fits.
    """

    def _multi_hot(self, draws_window):
        """One multi-hot vector per draw in the window, concatenated."""
        index_of = {label: i for i, label in enumerate(self.labels)}
        features = np.zeros(len(draws_window) * len(self.labels), dtype=float)
        for offset, draw in enumerate(draws_window):
            base = offset * len(self.labels)
            for number in draw:
                index = index_of.get(int(number))
                if index is not None:
                    features[base + index] = 1.0
        return features

    def _prepare_data(self, draws, window):
        X, Y = [], []
        for i in range(window, len(draws)):
            X.append(self._multi_hot(draws[i - window:i]))
            drawn = set(int(n) for n in draws[i])
            Y.append([1 if label in drawn else 0 for label in self.labels])
        return np.array(X), np.array(Y, dtype=int)

    def fit(self, draws, unique_labels=None):
        draws = [[int(n) for n in draw] for draw in draws]

        if len(draws) < 2:
            raise ValueError(f"Need at least 2 draws to train {self.library_name}.")

        self.setLengtOfDraw(len(draws[0]))
        self._build_labels(draws, unique_labels)

        window = self._effective_window(draws)
        X, Y = self._prepare_data(draws, window)

        self.models = []
        for index in range(len(self.labels)):
            y = Y[:, index]

            # A number that never appeared (or always appeared) in this slice
            # gives a single-class column, which no classifier can fit. Store
            # the constant rate instead of a model - _predict handles both.
            if len(set(y.tolist())) < 2:
                self.models.append(float(y.mean()))
                continue

            model = self._make_classifier(2)
            self._fit_classifier(model, X, y)
            self.models.append(model)

        return window

    def _number_probabilities(self, draws, window):
        features = self._multi_hot(draws[-window:]).reshape(1, -1)

        scores = {}
        for index, model in enumerate(self.models):
            number = self._decode(index)
            if isinstance(model, float):
                scores[number] = model
                continue
            probabilities = model.predict_proba(features)[0]
            # Column order follows model.classes_, so this stays correct even
            # if a backend orders them differently.
            classes = list(getattr(model, "classes_", [0, 1]))
            scores[number] = float(probabilities[classes.index(1)]) if 1 in classes else 0.0

        return scores

    def _predict(self, draws, window):
        number_scores = self._number_probabilities(draws, window)
        ranked = sorted(number_scores, key=number_scores.get, reverse=True)
        ticket = ranked[:self.lengthOfDraw]
        return (sorted(ticket) if self.sorted_prediction else ticket), number_scores

    def _score(self, draws, window):
        return self._number_probabilities(draws, window)
