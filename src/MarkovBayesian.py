import os
import sys
import numpy as np
import scipy.special
from collections import defaultdict
from Helpers import Helpers

# Dynamic path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')
for p in [current_dir, src_dir]:
    if p not in sys.path:
        sys.path.append(p)

helpers = Helpers()


class MarkovBayesian:
    def __init__(self):
        # Base hyperparameters
        self.dataPath = ""
        self.softMaxTemperature = 0.5
        self.alpha = 0.7
        self.min_occurrences = 5

        # Improvement-specific parameters
        self.decay_rate = 0.8        # (1) exponential recency decay
        self.dynamic_alpha = True    # (2) adaptive alpha based on entropy
        self.lambda_decay = 0.005    # (4) frequency time decay
        self.ngram_order = 2         # (5) use 1-gram or 2-gram Markov
        self.hybrid_blend = True     # (6) probability-based hybrid combination

        # Internal data
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.pair_counts = defaultdict(lambda: defaultdict(float))
        self.number_frequencies = defaultdict(float)
        self.bayesian_priors = defaultdict(lambda: 1)

    # ===== Setters =====
    def setDataPath(self, dataPath): self.dataPath = dataPath
    def setSoftMAxTemperature(self, t): self.softMaxTemperature = t
    def setAlpha(self, a): self.alpha = a
    def setMinOccurrences(self, n): self.min_occurrences = n
    def setDecayRate(self, r): self.decay_rate = r
    def setLambdaDecay(self, l): self.lambda_decay = l
    def setNgramOrder(self, n): self.ngram_order = n

    def clear(self):
        self.transition_matrix.clear()
        self.pair_counts.clear()
        self.number_frequencies.clear()
        self.bayesian_priors = defaultdict(lambda: 1)

    # ===== Core Methods =====
    def softmax_with_temperature(self, probabilities, temperature=1.0):
        """Applies softmax with temperature scaling — safely handles empty or zero arrays."""
        if probabilities is None or len(probabilities) == 0:
            return np.array([])

        probs = np.array(probabilities, dtype=float)
        if np.all(probs == 0):
            return np.ones_like(probs) / len(probs)

        probs = probs / max(float(temperature), 1e-8)
        try:
            return scipy.special.softmax(probs)
        except ValueError:
            # Catch edge cases like zero-size arrays
            return np.array([])

    def blended_probability(self, markov_probs, num_frequencies):
        freq_sum = sum(num_frequencies.values()) or 1
        base_blend = {}
        for num in set(markov_probs) | set(num_frequencies):
            base_blend[num] = (
                self.alpha * markov_probs.get(num, 0)
                + (1 - self.alpha) * (num_frequencies.get(num, 0) / freq_sum)
            )
        return base_blend

    def build_markov_chain(self, numbers):
        total_draws = len(numbers)
        for draw_index, draw in enumerate(numbers):
            # (1) exponential recency decay
            weight = np.exp(draw_index / total_draws * self.decay_rate)

            # (4) apply time decay to frequencies
            for num in draw:
                self.number_frequencies[num] += np.exp(-self.lambda_decay * (total_draws - draw_index))

            # (5) build n-gram transitions
            for i in range(len(draw) - self.ngram_order):
                prefix = tuple(draw[i:i + self.ngram_order])
                next_num = draw[i + self.ngram_order]
                self.transition_matrix[prefix][next_num] += weight

            # pair correlation
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    self.pair_counts[draw[i]][draw[j]] += 1
                    self.pair_counts[draw[j]][draw[i]] += 1

        # normalize and filter
        for prefix, transitions in self.transition_matrix.items():
            total = sum(transitions.values())
            if total == 0:
                continue
            self.transition_matrix[prefix] = {
                k: v / total for k, v in transitions.items() if v >= self.min_occurrences
            }

    def update_bayesian_model(self, drawn_numbers):
        for num in drawn_numbers:
            self.bayesian_priors[num] += 1

    def bayesian_prediction(self, n_predictions=20):
        total = sum(self.bayesian_priors.values())
        if total == 0:
            return {}
        probs = {num: c / total for num, c in self.bayesian_priors.items()}
        return probs

    def predict_next_numbers(self, previous_numbers, n_predictions=20):
        predictions = set()
        markov_probs = defaultdict(float)

        # build key for n-gram
        prefix = tuple(previous_numbers[-self.ngram_order:]) if len(previous_numbers) >= self.ngram_order else tuple(previous_numbers)
        if prefix in self.transition_matrix:
            next_nums = list(self.transition_matrix[prefix].keys())
            probs = list(self.transition_matrix[prefix].values())

            adjusted = self.softmax_with_temperature(probs, self.softMaxTemperature)

            if len(adjusted) == 0 or len(next_nums) == 0:
                return []  # gracefully skip empty predictions

            for num, p in zip(next_nums, adjusted):
                markov_probs[num] += p
        else:
            # fallback to frequency-based prediction if no transition found
            sorted_freq = sorted(self.number_frequencies, key=self.number_frequencies.get, reverse=True)
            return [int(x) for x in sorted_freq[:n_predictions]]

        # (2) dynamic alpha via entropy
        if self.dynamic_alpha and markov_probs:
            entropy = -sum(p * np.log(p + 1e-9) for p in markov_probs.values())
            self.alpha = 1 / (1 + np.exp(-entropy))

        # blended markov–frequency
        blended_probs = self.blended_probability(markov_probs, self.number_frequencies)
        sorted_nums = sorted(blended_probs, key=blended_probs.get, reverse=True)
        predictions.update(sorted_nums[:n_predictions])

        # (3) pair reinforcement
        pair_strength = defaultdict(float)
        for num in predictions:
            pair_strength[num] = sum(self.pair_counts[num][p] for p in predictions if p != num)
        for num, strength in pair_strength.items():
            blended_probs[num] = blended_probs.get(num, 0) + 0.01 * strength

        # select final predictions
        final_sorted = sorted(blended_probs, key=blended_probs.get, reverse=True)
        return [int(num) for num in final_sorted[:n_predictions]]

    def run(self, generateSubsets=[], skipRows=0):
        """
        Run the enhanced Markov–Bayesian++ model and return Python int results.
        """
        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath, skipRows=skipRows)

        # Build model and update with latest data
        self.build_markov_chain(numbers)
        last_draw = [int(x) for x in numbers[-1]]  # Ensure base draw is Python int
        self.update_bayesian_model(last_draw)

        n_predictions = len(last_draw)

        # Get predictions
        markov_pred_probs = [int(x) for x in self.predict_next_numbers(last_draw, n_predictions)]
        bayesian_probs = {int(k): float(v) for k, v in self.bayesian_prediction(n_predictions).items()}

        # (6) Hybrid combination
        if self.hybrid_blend:
            total = {}
            freq_sum = sum(self.number_frequencies.values()) or 1.0

            for num in set(markov_pred_probs) | set(bayesian_probs):
                num = int(num)
                total[num] = (
                    float(self.alpha) * bayesian_probs.get(num, 0.0)
                    + float(1 - self.alpha) * (float(self.number_frequencies.get(num, 0.0)) / freq_sum)
                )

            combined = sorted(total, key=total.get, reverse=True)[:n_predictions]
        else:
            combined = list(set(markov_pred_probs) | set(map(int, bayesian_probs.keys())))

        # Explicitly cast predictions to Python int
        combined = [int(x) for x in combined[:n_predictions]]

        # Generate subsets with pure Python ints
        subsets = {}
        for size in generateSubsets:
            subset = self.generate_best_subset(combined, size)
            subsets[int(size)] = [int(x) for x in subset]

        return combined, subsets

    # ===== Subset Selector =====
    def generate_best_subset(self, predicted_numbers, nSubset):
        unique_numbers = list(set(map(int, predicted_numbers)))
        if len(unique_numbers) < nSubset:
            return unique_numbers

        blended_probs = self.blended_probability({num: 1 for num in unique_numbers}, self.number_frequencies)
        sorted_nums = sorted(unique_numbers, key=lambda x: blended_probs.get(x, 0), reverse=True)
        return sorted(sorted_nums[:nSubset])


if __name__ == "__main__":
    print("Running New Markov-Bayesian Model")

    mb = MarkovBayesian()
    mb.setDataPath("../test/trainingData/keno")
    mb.setSoftMAxTemperature(0.3)
    mb.setAlpha(0.6)
    mb.setDecayRate(0.9)
    mb.setLambdaDecay(0.01)
    mb.setNgramOrder(2)

    preds, subsets = mb.run(generateSubsets=[6, 7])
    print("Predictions:", preds)
    print("Subsets:", subsets)
