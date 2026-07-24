import os
import sys
import random
from collections import defaultdict
from Helpers import Helpers
from MarkovBayesian import MarkovBayesian

# Dynamically adjust the import path for Helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

helpers = Helpers()

class MarkovBayesianEnhanced(MarkovBayesian):
    """
    Adds two genuine improvements over MarkovBayesian:
      - Exponential recency decay (decay_rate ** draw_index over the reversed
        draw history) instead of the parent's mild linear weighting.
      - A weighted ensemble_prediction that blends Markov-rank and
        Bayesian-rank (1.5x vs 1.0x) with tie-break jitter, instead of the
        parent's flat order-preserving union.

    (An earlier version also recombined random historical draws and picked
    whichever candidate scored best against a hardcoded target sum of 880 -
    that was Keno-specific (20 numbers, range 1-80) and silently produced
    biased-toward-high-numbers picks for every other game. Removed rather
    than fixed, since it was never a principled prediction mechanism.)
    """

    def build_markov_chain(self, numbers, decay_rate=0.98):
        for draw_index, draw in enumerate(reversed(numbers)):
            weight = decay_rate ** draw_index

            for i in range(len(draw) - 1):
                self.transition_matrix[draw[i]][draw[i + 1]] += weight

            for num in draw:
                self.number_frequencies[num] += weight

            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    self.pair_counts[draw[i]][draw[j]] += 1
                    self.pair_counts[draw[j]][draw[i]] += 1

        for k, v in self.transition_matrix.items():
            total = sum(v.values())
            self.transition_matrix[k] = {
                nk: nv / total for nk, nv in v.items() if nv >= self.min_occurrences
            }

    def ensemble_prediction(self, last_draw, n_predictions=20):
        markov_preds = self.predict_next_numbers(last_draw, n_predictions=n_predictions)
        bayes_preds = self.bayesian_prediction(n_predictions=n_predictions)
        scores = defaultdict(float)

        for i, num in enumerate(markov_preds):
            scores[int(num)] += (n_predictions - i) * 1.5
        for i, num in enumerate(bayes_preds):
            scores[int(num)] += (n_predictions - i) * 1.0

        for num in scores:
            scores[num] += random.uniform(0, 1)

        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:n_predictions]

    def run(self, generateSubsets=[], skipRows=0, skipLastColumns=0, specialColumnCount=0):
        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

        # Every run() must start from a clean slate: build_markov_chain
        # normalizes each transition_matrix entry from a defaultdict into a
        # plain dict, so a stale (unreset) matrix from a prior run() would
        # raise KeyError on the next call's `+=` for any newly-filtered key.
        self.clear()

        self.build_markov_chain(numbers)

        last_draw = numbers[-1]
        self.update_bayesian_model(last_draw)

        predicted_numbers = self.ensemble_prediction(last_draw, n_predictions=len(last_draw))
        predicted_numbers = list(dict.fromkeys(int(n) for n in predicted_numbers))

        if self.sorted_prediction:
            predicted_numbers = sorted(predicted_numbers)

        subsets = {}
        if generateSubsets:
            for subset_size in generateSubsets:
                best = self.generate_best_subset(predicted_numbers, subset_size)
                subsets[subset_size] = [int(n) for n in best]

        return predicted_numbers, subsets


if __name__ == "__main__":
    print("Trying Markov-Bayesian Enhanced Model")

    markovBayesian = MarkovBayesianEnhanced()

    name = 'keno'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    markovBayesian.setDataPath(dataPath)
    markovBayesian.setSoftMAxTemperature(0.1)
    markovBayesian.setAlpha(0.5)
    markovBayesian.setMinOccurrences(5)

    if "keno" in name:
        generateSubsets = [6, 7]

    print("Predicted Numbers: ", markovBayesian.run(generateSubsets=generateSubsets))
