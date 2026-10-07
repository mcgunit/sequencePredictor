import os, sys, random
import numpy as np
from collections import Counter, defaultdict
from PoissonMonteCarlo import PoissonMonteCarlo
from Markov import Markov

# Dynamically adjust the import path for Helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

# Ensure Helpers can be imported
if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

class PoissonMarkov:
    def __init__(self):
        self.poisson_model = PoissonMonteCarlo()
        self.markov_model = Markov()
        self.poisson_weight = 0.5  # Weight assigned to Poisson predictions
        self.markov_weight = 0.5   # Weight assigned to Markov predictions
        self.sorted_prediction = True  # Set False for positional games like Pick3
        self._last_number_scores = {}   # the blend's weight per number from the last run()
        self._last_tie_break = {}       # the sub-models' own scores, the second sort key of a subset

    def setSortedPrediction(self, use):
        """
        Disable for positional games (Pick3). Propagated to both sub-models so
        their own per-position order (see PoissonMonteCarlo/Markov) survives as
        far as possible before blend_predictions ranks by confidence - note that
        blend_predictions itself pools numbers into a flat weighted bag with no
        positional identity, so this alone does not make Pick3 output correct.
        """
        self.sorted_prediction = bool(use)
        self.poisson_model.setSortedPrediction(use)
        self.markov_model.setSortedPrediction(use)

    def setDataPath(self, dataPath):
        """Set data path for both models."""
        self.poisson_model.setDataPath(dataPath)
        self.markov_model.setDataPath(dataPath)

    def setWeights(self, poisson_weight=0.5, markov_weight=0.5):
        """Adjust the weight contributions of Poisson and Markov models."""
        total = poisson_weight + markov_weight
        self.poisson_weight = poisson_weight / total
        self.markov_weight = markov_weight / total

    def setNumberOfSimulations(self, n_simulations):
        self.poisson_model.setNumOfSimulations(n_simulations)

    def blend_predictions(self, poisson_numbers, markov_numbers, n_predictions=20):
        """Blend predictions from both models using weighted probability selection."""
        combined_counts = Counter()

        # Apply weights
        for num in poisson_numbers:
            combined_counts[int(num)] += self.poisson_weight
        for num in markov_numbers:
            combined_counts[int(num)] += self.markov_weight

        # The blend's weights take at most three values (one model's weight,
        # the other's, their sum), so a subset would be completed by number
        # order; the sub-models' own scores (the Poisson counts, the chain's
        # masses, each normalised) break the ties inside a weight class
        # instead - as a second sort key, so they can never outrank a
        # heavier class whatever the weights are (7 Oct 2026).
        poisson_scores = dict(getattr(self.poisson_model, "_last_number_scores", {}) or {})
        markov_scores = dict(getattr(self.markov_model, "_last_number_scores", {}) or {})
        p_max = max(poisson_scores.values(), default=0) or 1.0
        m_max = max(markov_scores.values(), default=0) or 1.0
        self._last_number_scores = {int(num): float(weight) for num, weight in combined_counts.items()}
        self._last_tie_break = {int(num): (poisson_scores.get(int(num), 0) / p_max + markov_scores.get(int(num), 0) / m_max) / 2
                                for num in combined_counts}
        # Random tie-breaking with consistent ordering
        unique_numbers = list(combined_counts.keys())
        random.shuffle(unique_numbers)

        # Sort based on weight
        sorted_numbers = sorted(unique_numbers, key=lambda x: combined_counts[x], reverse=True)

        return sorted_numbers[:n_predictions]

    def generate_best_subset(self, predicted_numbers, nSubset):
        """
        The keno subset: the ticket's numbers ranked by the blend's weight on
        them (both sub-models' votes), ties inside a weight class broken by
        the sub-models' own scores, the top nSubset. Until 7 Oct 2026 the
        weights were a linspace over set(ticket) - numeric order.
        """
        unique_numbers = list(dict.fromkeys(int(num) for num in predicted_numbers))
        if len(unique_numbers) <= nSubset:
            return sorted(unique_numbers)
        scores = self._last_number_scores
        ties = self._last_tie_break
        ranked = sorted(unique_numbers, key=lambda num: (-scores.get(num, 0.0), -ties.get(num, 0.0), num))
        return sorted(ranked[:nSubset])

    def run(self, generateSubsets=[], skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """Runs both models, blends predictions, and generates subsets if needed."""
        self.poisson_model.clear()
        self.markov_model.clear()
        poisson_numbers, _ = self.poisson_model.run(skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)
        #print("poisson numbers: ", poisson_numbers)
        markov_numbers, _ = self.markov_model.run(skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)
        #print("markov numbers: ", markov_numbers)

        # Flatten if returned as nested lists
        if isinstance(poisson_numbers[0], list):
            poisson_numbers = [int(num) for sublist in poisson_numbers for num in sublist]
        else:
            poisson_numbers = [int(num) for num in poisson_numbers]

        if isinstance(markov_numbers[0], list):
            markov_numbers = [int(num) for sublist in markov_numbers for num in sublist]
        else:
            markov_numbers = [int(num) for num in markov_numbers]

        hybrid_predictions = self.blend_predictions(poisson_numbers, markov_numbers, len(poisson_numbers))

        subsets = {}
        if generateSubsets:
            # print("Creating subsets of: ", generateSubsets)
            for subset_size in generateSubsets:
                subsets[subset_size] = self.generate_best_subset(hybrid_predictions, subset_size)

        return hybrid_predictions, subsets

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Per-number score for stacking (Phase 1): weighted blend of each
        sub-model's own per-number score (reusing PoissonMonteCarlo.score_numbers
        and Markov.score_numbers), analogous to how run() weight-blends their
        final tickets via blend_predictions.
        """
        poisson_scores = self.poisson_model.score_numbers(
            skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)
        markov_scores = self.markov_model.score_numbers(
            skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

        combined = defaultdict(float)
        for num, score in poisson_scores.items():
            combined[int(num)] += self.poisson_weight * score
        for num, score in markov_scores.items():
            combined[int(num)] += self.markov_weight * score

        return dict(combined)

if __name__ == "__main__":
    print("Running Hybrid Poisson-Markov Model")

    hybrid_model = PoissonMarkov()

    name = 'euromillions'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    hybrid_model.setDataPath(dataPath)
    hybrid_model.setWeights(poisson_weight=0.5, markov_weight=0.5)

    if "keno" in name:
        generateSubsets = [6, 7]

    predicted_numbers, subsets = hybrid_model.run(generateSubsets=generateSubsets)

    print("Predicted Numbers:", predicted_numbers)
    print("Generated Subsets:", subsets)
