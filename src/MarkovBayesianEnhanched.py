import os
import sys
import numpy as np
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
    def __init__(self):
        super().__init__()
        self.bigram_matrix = defaultdict(lambda: defaultdict(float))
        self.position_frequencies = defaultdict(lambda: defaultdict(float))

    def clear(self):
        super().clear()
        self.bigram_matrix = defaultdict(lambda: defaultdict(float))
        self.position_frequencies = defaultdict(lambda: defaultdict(float))

    def build_markov_chain(self, numbers, decay_rate=0.98):
        total_draws = len(numbers)
        for draw_index, draw in enumerate(reversed(numbers)):
            weight = decay_rate ** draw_index
            for i in range(len(draw) - 1):
                self.transition_matrix[draw[i]][draw[i + 1]] += weight
                if i < len(draw) - 2:
                    pair = (draw[i], draw[i + 1])
                    self.bigram_matrix[pair][draw[i + 2]] += weight
            for pos, num in enumerate(draw):
                self.position_frequencies[pos][num] += weight
                self.number_frequencies[num] += weight
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    self.pair_counts[draw[i]][draw[j]] += 1
                    self.pair_counts[draw[j]][draw[i]] += 1

        for transitions in [self.transition_matrix, self.bigram_matrix]:
            for k, v in transitions.items():
                total = sum(v.values())
                transitions[k] = {
                    nk: nv / total for nk, nv in v.items() if nv >= self.min_occurrences
                }

    def ensemble_prediction(self, last_draw, n_predictions=20):
        markov_preds = self.predict_next_numbers(last_draw, n_predictions=40, temperature=self.softMaxTemperature)
        bayes_preds = self.bayesian_prediction(n_predictions=40)
        scores = defaultdict(float)

        for i, num in enumerate(markov_preds):
            scores[int(num)] += (40 - i) * 1.5
        for i, num in enumerate(bayes_preds):
            scores[int(num)] += (40 - i) * 1.0

        for num in scores:
            scores[num] += random.uniform(0, 1)

        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:n_predictions]

    def generate_crossover_combinations(self, numbers, recent_draws=5, predictionLenght=20):
        if len(numbers) < recent_draws:
            return []

        draws = list(numbers[-recent_draws:])
        new_combos = []
        for _ in range(10):
            a, b = random.sample(draws, 2)
            a = list(a)
            b = list(b)
            try:
                mixed = list(set(random.sample(a, min(10, len(a))) + random.sample(b, min(10, len(b)))))
                if len(mixed) >= 10:
                    new_combos.append(sorted(mixed[:predictionLenght]))
            except ValueError:
                continue  # skip invalid combos where sample size is too large

        return new_combos


    def score_meta_features(self, subset):
        even_count = sum(1 for n in subset if n % 2 == 0)
        spread = max(subset) - min(subset)
        total = sum(subset)
        return (
            -abs(total - 880) * 0.5 +
            even_count * 2.0 +
            spread * 1.5
        )

    def best_scored_subset(self, candidate_sets, top_n=1):
        scored = sorted(candidate_sets, key=self.score_meta_features, reverse=True)
        return [sorted(set(int(n) for n in s)) for s in scored[:top_n]]

    def run(self, generateSubsets=[], skipRows=0):
        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath, skipRows=skipRows)
        self.numbers = numbers
        self.build_markov_chain(numbers)

        last_draw = numbers[-1]
        self.update_bayesian_model(last_draw)

        combined_predictions = self.ensemble_prediction(last_draw, n_predictions=len(last_draw))

        cross_combos = self.generate_crossover_combinations(numbers, 200, len(last_draw))
        candidate_sets = [combined_predictions] + cross_combos
        scored_subset = self.best_scored_subset(candidate_sets)

        subsets = {}
        if generateSubsets:
            # print("Creating subsets of:", generateSubsets)
            for subset_size in generateSubsets:
                best = self.generate_best_subset(combined_predictions, subset_size)
                subsets[subset_size] = [int(n) for n in best]

        return [int(n) for n in scored_subset[0]], subsets



if __name__ == "__main__":
    print("Trying Markov-Bayesian Enhanced Model")

    markovBayesian = MarkovBayesianEnhanced()

    name = 'lotto'
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
