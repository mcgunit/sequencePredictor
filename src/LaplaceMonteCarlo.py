import os, sys
import numpy as np
from scipy.stats import laplace
from scipy.special import softmax
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

class LaplaceMonteCarlo():
    def __init__(self):
        self.dataPath = ""
        self.num_simulations = 5000  # Fine-tuned Monte Carlo simulations
        self.recent_draws = 100  # Look-back window
        self.position_stats = defaultdict(lambda: [])
        self.sorted_prediction = True  # Set False for positional games like Pick3

    def clear(self):
        self.position_stats = defaultdict(lambda: [])

    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    def setNumOfSimulations(self, nSimulations):
        self.num_simulations = nSimulations

    def setRecentDraws(self, nRecentDraws):
        self.recent_draws = nRecentDraws

    def setSortedPrediction(self, use):
        """
        Disable for positional games (Pick3) so the returned digits keep their
        drawn (per-position) order instead of being reordered ascending by value.
        """
        self.sorted_prediction = bool(use)

    def generate_best_subset(self, predicted_numbers, nSubset):
        """Generate a unique subset using weighted probability selection."""
        unique_numbers = list(set(map(int, predicted_numbers)))  # Ensure standard integers

        if len(unique_numbers) < nSubset:
            return unique_numbers  # Fallback if not enough numbers

        # Assign probabilities (higher for top-ranked numbers)
        probabilities = np.linspace(1.0, 0.5, len(unique_numbers))
        probabilities /= probabilities.sum()  # Normalize

        # Randomly select numbers based on weighted probability
        best_subset = np.random.choice(unique_numbers, size=nSubset, replace=False, p=probabilities)

        return sorted(map(int, best_subset))

    def ensure_unique_prediction(self, predicted_numbers, n_predictions):
        unique_numbers = list(dict.fromkeys(map(int, predicted_numbers)))

        all_numbers = list(range(self.min_number, self.max_number + 1))
        remaining = [n for n in all_numbers if n not in unique_numbers]

        while len(unique_numbers) < n_predictions and remaining:
            chosen = int(np.random.choice(remaining))
            unique_numbers.append(chosen)
            remaining.remove(chosen)

        final_predictions = unique_numbers[:n_predictions]
        return sorted(final_predictions) if self.sorted_prediction else final_predictions
    
    def build_laplace_model(self, numbers):
        """Computes Laplace distribution parameters for each position."""

        self.min_number = min(min(draw) for draw in numbers)
        self.max_number = max(max(draw) for draw in numbers)

        for draw in numbers[-self.recent_draws:]:
            for pos, num in enumerate(draw):
                self.position_stats[pos].append(int(num))

        self.laplace_params = {}

        for pos, nums in self.position_stats.items():
            arr = np.array(nums, dtype=float)

            loc = np.mean(arr)
            scale = max(
                0.1,
                np.median(np.abs(arr - np.median(arr))) / np.sqrt(2)
            )

            self.laplace_params[pos] = (loc, scale)
    
    def monte_carlo_simulation(self, n_predictions=20):
        """Runs Monte Carlo simulations using Laplace distribution for each position."""
        predicted_numbers = []
        
        for pos in range(n_predictions):
            simulated_counts = defaultdict(int)
            
            for _ in range(self.num_simulations):
                if pos in self.laplace_params:
                    loc, scale = self.laplace_params[pos]
                    sampled_value = int(laplace.rvs(loc=loc, scale=scale))
                    sampled_value = max(self.min_number, min(self.max_number, sampled_value))
                    simulated_counts[sampled_value] += 1
            
            # Normalize and apply softmax filtering
            if simulated_counts:
                raw_values = np.array([simulated_counts[num] for num in simulated_counts])
                probabilities = softmax(raw_values)
                sorted_predictions = [num for _, num in sorted(zip(probabilities, simulated_counts.keys()), reverse=True)]
                predicted_numbers.append(int(np.random.choice(sorted_predictions[:3])))  # Pick from the top 3

        return self.ensure_unique_prediction(predicted_numbers, n_predictions)
    
    def run(self, generateSubsets=[], skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """Runs the Laplace Monte Carlo prediction process with optional subset generation."""

        self.clear()

        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

        self.setRecentDraws(min(self.recent_draws, len(numbers)))

        self.build_laplace_model(numbers)
        
        predicted_numbers = self.monte_carlo_simulation(n_predictions=len(numbers[-1]))
        
        subsets = {}
        if generateSubsets:
            # print("Creating subsets of: ", generateSubsets)
            for nPredictions in generateSubsets:
                subsets[nPredictions] = self.generate_best_subset(predicted_numbers, nPredictions)
        
        return predicted_numbers, subsets

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Per-number score for stacking (Phase 1): same Laplace Monte Carlo
        simulation build_laplace_model/monte_carlo_simulation already run,
        but merging the simulated counts across every position into one
        {number: score} dict instead of picking the top-3-per-position winner
        and discarding the rest.
        """
        self.clear()

        _, _, _, _, _, numbers, _, _ = helpers.load_data(
            self.dataPath, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

        if len(numbers) == 0:
            return {}

        self.setRecentDraws(min(self.recent_draws, len(numbers)))
        self.build_laplace_model(numbers)

        merged_counts = defaultdict(int)
        for pos, (loc, scale) in self.laplace_params.items():
            for _ in range(self.num_simulations):
                sampled_value = int(laplace.rvs(loc=loc, scale=scale))
                sampled_value = max(self.min_number, min(self.max_number, sampled_value))
                merged_counts[sampled_value] += 1

        return dict(merged_counts)

    def score_positions(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Per-position digit scores for the positional (Pick3) meta-learner: one
        {digit: probability} dict per drawn position, in drawn order, from the
        same build_laplace_model fit run() uses. Instead of drawing
        num_simulations Laplace samples per position and counting, this is
        the exact discretised pmf of that sampling step - int() truncation of
        the continuous sample followed by clipping to [min_number,
        max_number]: every sample below min_number+1 lands on min_number
        (truncation toward zero and the clip both push it there), every
        sample at or above max_number lands on max_number, and each digit in
        between owns the unit interval [digit, digit+1). Deterministic, so the
        feature carries the model's signal without sampling noise on top.
        Every digit of the game's label range is present (outside the fitted
        [min_number, max_number] -> 0.0) so the consumer can build fixed-width
        feature vectors without guarding keys. Empty history returns [] -
        same convention as score_numbers' {}.
        """
        self.clear()

        _, _, _, _, _, numbers, _, unique_labels = helpers.load_data(
            self.dataPath, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

        if len(numbers) == 0:
            return []

        self.setRecentDraws(min(self.recent_draws, len(numbers)))
        self.build_laplace_model(numbers)

        digits = [int(label) for label in unique_labels]
        low, high = int(self.min_number), int(self.max_number)

        position_scores = []
        for pos in range(len(numbers[-1])):
            if pos not in self.laplace_params:
                position_scores.append({digit: 0.0 for digit in digits})
                continue

            loc, scale = self.laplace_params[pos]
            scores = {}
            for digit in digits:
                if digit < low or digit > high:
                    mass = 0.0
                elif low == high:
                    mass = 1.0
                elif digit == low:
                    mass = laplace.cdf(low + 1, loc=loc, scale=scale)
                elif digit == high:
                    mass = 1.0 - laplace.cdf(high, loc=loc, scale=scale)
                else:
                    mass = laplace.cdf(digit + 1, loc=loc, scale=scale) - laplace.cdf(digit, loc=loc, scale=scale)
                # Floating-point cdf differences can dip a hair below zero.
                scores[digit] = float(max(0.0, mass))
            position_scores.append(scores)

        return position_scores

if __name__ == "__main__":
    print("Running Laplace Monte Carlo Simulation")
    
    model = LaplaceMonteCarlo()
    
    name = 'keno'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    
    model.setDataPath(dataPath)
    model.setNumOfSimulations(5000)
    model.setRecentDraws(2000)
    
    if "keno" in name:
        generateSubsets = [6, 7]
    
    print("Predicted Numbers: ", model.run(generateSubsets=generateSubsets))

