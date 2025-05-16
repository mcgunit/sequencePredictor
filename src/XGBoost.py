import os, sys
import xgboost as xgb
import numpy as np
import os
import joblib
from itertools import combinations
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

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

class XGBoostKenoPredictor:
    def __init__(self, n_previous_draws=5, n_estimators=100, max_depth=5, learning_rate=0.1, lengthOfDraw=20):
        self.dataPath = ""
        self.modelPath = ""
        self.n_previous_draws = n_previous_draws
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = [None] * lengthOfDraw

    def setModelPath(self, modelPath):
        self.modelPath = modelPath
    
    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    def _prepare_data(self, draws: List[List[int]], lengthOfDraw):
        X, Y = [], [[] for _ in range(lengthOfDraw)]
        for i in range(self.n_previous_draws, len(draws)):
            window = draws[i - self.n_previous_draws:i]
            flat_features = np.array(window).flatten()
            X.append(flat_features)
            for pos in range(lengthOfDraw):
                Y[pos].append(draws[i][pos] - 1)
        return np.array(X), [np.array(y) for y in Y]

    def fit(self, draws: List[List[int]], num_classes, lengthOfDraw):
        X, Ys = self._prepare_data(draws, lengthOfDraw)

        for pos in range(lengthOfDraw):
            y_pos = Ys[pos].copy()
            X_pos = X.copy()

            missing_labels = set(range(num_classes)) - set(y_pos)

            if missing_labels:
                for label in missing_labels:
                    X_pos = np.vstack([X_pos, X[0]])       
                    y_pos = np.append(y_pos, label)         

            assert len(X_pos) == len(y_pos), f"X rows: {len(X_pos)} != labels: {len(y_pos)}"

            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                use_label_encoder=False,
                eval_metric="mlogloss"
            )

            print(f"Ys[{pos}]: min={Ys[pos].min()}, max={Ys[pos].max()}")
            model.fit(X_pos, y_pos)
            self.models[pos] = model

    def predict(self, recent_draws: List[List[int]]) -> List[int]:
        if len(recent_draws) < self.n_previous_draws:
            raise ValueError(f"Need at least {self.n_previous_draws} recent draws to predict.")

        # Flatten the most recent draws and convert to input features
        input_features = np.array(
            [[n - 1 for draw in recent_draws[-self.n_previous_draws:] for n in draw]]
        )  # shape: (1, features)

        predicted_draw = []
        for model in self.models:
            pred_label = model.predict(input_features)[0] 
            predicted_draw.append(int(pred_label) + 1)  

        return predicted_draw
    
    def predict_with_subsets(
        self,
        recent_draws: List[List[int]],
        draw_sizes: List[int],
        top_k: int = 3,
        force_nested: bool = False
    ) -> List[List[int]]:
        """
        Args:
            recent_draws: Past Keno draws used for prediction.
            draw_sizes: List of integers specifying the sizes of draws to predict.
            top_k: Number of most probable numbers to consider per position.
            force_nested: If True, ensures larger draws are supersets of smaller ones.

        Returns:
            predicted_draw: Top-1 most likely number for each position.
            subsets_by_draw_size: A list of draws for each requested draw size.
        """
        if len(recent_draws) < self.n_previous_draws:
            raise ValueError(f"Need at least {self.n_previous_draws} recent draws to predict.")

        input_features = np.array(
            [[n - 1 for draw in recent_draws[-self.n_previous_draws:] for n in draw]]
        )

        predicted_draw = []
        number_scores = defaultdict(lambda: {"total_prob": 0.0, "count": 0})

        for model in self.models:
            probs = model.predict_proba(input_features)[0]
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_numbers = [int(i + 1) for i in top_indices]

            # Top-1 prediction
            predicted_draw.append(top_numbers[0])

            # Track confidence for each number
            for i in top_indices:
                number = int(i + 1)
                number_scores[number]["total_prob"] += probs[i]
                number_scores[number]["count"] += 1

        # Compute average confidence per number
        average_confidences = {
            num: val["total_prob"] / val["count"]
            for num, val in number_scores.items()
        }

        save_average_confidence_plot(average_confidences)

        # Rank by average confidence (descending)
        sorted_candidates = sorted(average_confidences, key=average_confidences.get, reverse=True)

        # Generate draw subsets
        subsets_by_draw_size = []
        already_included = set()

        for size in draw_sizes:
            if force_nested:
                draw = list(already_included)
                for n in sorted_candidates:
                    if len(draw) >= size:
                        break
                    if n not in draw:
                        draw.append(n)
                already_included.update(draw)
            else:
                used = set()
                draw = []
                for n in sorted_candidates:
                    if len(draw) >= size:
                        break
                    if n not in used:
                        draw.append(n)
                        used.add(n)
            subsets_by_draw_size.append(draw)

        return predicted_draw, subsets_by_draw_size


    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        for i, model in enumerate(self.models):
            model_path = os.path.join(folder_path, f"model_pos_{i}.joblib")
            joblib.dump(model, model_path)

    def load(self, folder_path: str):
        self.models = []
        for i in range(20):
            model_path = os.path.join(folder_path, f"model_pos_{i}.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = joblib.load(model_path)
            self.models.append(model)

    def run(self, generateSubsets=[], skipRows=0):
        _, _, _, _, _, numbers, num_classes, _ = helpers.load_data(self.dataPath, skipRows=skipRows)

        #print("num classes: ", num_classes)

        self.fit(draws=numbers, num_classes=num_classes, lengthOfDraw=len(numbers[0]))

        # Save to disk
        self.save(self.modelPath)

        # to Load later
        #new_predictor = XGBoostKenoPredictor(n_previous_draws=5)
        #new_predictor.load(os.path.join(self.modelPath, "xgboost_keno_models"))
        subsets = {}

        if len(generateSubsets) > 0:
            predicted_numbers, subsets = self.predict_with_subsets(numbers[-5:], top_k=5, draw_sizes=generateSubsets, force_nested=False)
        else:
            predicted_numbers = self.predict(numbers[-5:])

        return predicted_numbers, subsets

def save_average_confidence_plot(avg_confidences, filename="average_confidence_per_number.png"):
    numbers = list(avg_confidences.keys())
    confidences = list(avg_confidences.values())

    plt.figure(figsize=(10, 4))
    plt.bar(numbers, confidences, color='skyblue')
    plt.xlabel("Number")
    plt.ylabel("Average Confidence")
    plt.title("Average Confidence per Number")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    print("Trying XGBoost")

    xgboost = XGBoostKenoPredictor(n_previous_draws=5)

    name = 'keno'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", f"xgboost_{name}_models")

    xgboost.setDataPath(dataPath)
    xgboost.setModelPath(modelPath)

    if "keno" in name:
        generateSubsets = [6, 7]

    print("Predicted Numbers: ", xgboost.run(generateSubsets=generateSubsets))
