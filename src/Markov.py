import os, sys
import numpy as np
import scipy.special
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

class Markov():
    def __init__(self):
        self.dataPath = ""
        self.softMaxTemperature = 0.5
        self.alpha = 0.7
        self.min_occurrences = 5
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.number_frequencies = defaultdict(int)
    
    def clear(self):
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.number_frequencies = defaultdict(int)

    def setDataPath(self, dataPath):
        self.dataPath = dataPath
    
    def setSoftMAxTemperature(self, temperature):
        self.softMaxTemperature = temperature
    
    def setAlpha(self, nAlpha):
        self.alpha = nAlpha

    def setMinOccurrences(self, nMinOccurrences):
        self.min_occurrences = nMinOccurrences

    def generate_best_subset(self, predicted_numbers, nSubset):
        """Generate a subset of numbers using weighted selection based on Markov probabilities and frequencies."""
        unique_numbers = list(set(map(int, predicted_numbers)))  # Ensure unique standard integers

        if len(unique_numbers) < nSubset:
            return unique_numbers  # Fallback if not enough numbers

        # Compute blended probabilities using Markov and frequency data
        blended_probs = self.blended_probability({int(num): 1 for num in unique_numbers}, self.number_frequencies)

        # Sort numbers based on probability values
        sorted_numbers = sorted(unique_numbers, key=lambda x: blended_probs.get(int(x), 0), reverse=True)

        # Select the top `nSubset` numbers
        best_subset = sorted(map(int, sorted_numbers[:nSubset]))

        return best_subset

    def softmax_with_temperature(self, probabilities, temperature=1.0):
        """Applies temperature scaling to control randomness."""
        probs = np.array(probabilities) / temperature
        return scipy.special.softmax(probs)

    def blended_probability(self, markov_probs, num_frequencies):
        """Combines Markov transition probabilities with frequency analysis."""
        total_freq = sum(num_frequencies.values()) or 1  # Prevent division by zero
        return {
            int(num): (
                self.alpha * markov_probs.get(int(num), 0) +
                (1 - self.alpha) * (num_frequencies.get(int(num), 0) / total_freq)
            )
            for num in set(map(int, markov_probs)) | set(map(int, num_frequencies))
        }

    def build_markov_chain(self, numbers):
        """Creates the transition matrix with weighted recency and removes rare transitions."""
        total_draws = len(numbers)

        for draw_index, draw in enumerate(numbers):
            weight = 1 + (draw_index / total_draws)  # More weight to recent draws

            draw = list(map(int, draw))

            for i in range(len(draw) - 1):
                self.transition_matrix[draw[i]][draw[i + 1]] += weight
            
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    self.pair_counts[draw[i]][draw[j]] += 1
                    self.pair_counts[draw[j]][draw[i]] += 1
            
            for num in draw:
                self.number_frequencies[int(num)] += 1

        for number, transitions in self.transition_matrix.items():
            total = sum(transitions.values())
            self.transition_matrix[number] = {
                int(k): v / total for k, v in transitions.items() if v >= self.min_occurrences
            }

    def predict_next_numbers(self, previous_numbers, n_predictions=20, temperature=0.7):
        """Predicts the next numbers using Markov Chain with all fine-tuning strategies."""
        predictions = set()
        previous_numbers = list(map(int, previous_numbers))

        for num in previous_numbers:
            if num in self.transition_matrix and self.transition_matrix[num]:
                next_nums = list(self.transition_matrix[num].keys())
                probs = list(self.transition_matrix[num].values())

                adjusted_probs = self.softmax_with_temperature(probs, temperature=temperature)
                predicted_num = int(np.random.choice(next_nums, p=adjusted_probs))
                predictions.add(predicted_num)

        if len(predictions) < n_predictions:
            all_numbers = list(self.transition_matrix.keys())
            blended_probs = self.blended_probability(
                self.transition_matrix.get(num, {}),
                self.number_frequencies
            )

            sorted_freq = sorted(blended_probs, key=blended_probs.get, reverse=True)

            while len(predictions) < n_predictions and sorted_freq:
                next_best = int(sorted_freq.pop(0))
                if next_best not in predictions:
                    predictions.add(next_best)

        while len(predictions) < n_predictions:
            last_predicted = list(predictions)[-1]
            if last_predicted in self.pair_counts:
                next_best = max(
                    self.pair_counts[last_predicted],
                    key=self.pair_counts[last_predicted].get,
                    default=None
                )
                if next_best is not None and int(next_best) not in predictions:
                    predictions.add(int(next_best))

        return sorted(map(int, list(predictions)[:n_predictions]))

    def run(self, generateSubsets=[]):
        """
        Runs the Markov Chain prediction process with optional subset generation.
        
        Parameters:
        generateSubsets (list): List of subset sizes to generate, e.g., [6, 7] will generate subsets of size 6 and 7.
        """
        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath)

        self.build_markov_chain(numbers)

        last_draw = numbers[-1]
        predicted_numbers = self.predict_next_numbers(last_draw, n_predictions=len(last_draw), temperature=self.softMaxTemperature)

        subsets = {}
        if generateSubsets:
            print("Creating subsets of:", generateSubsets)
            for subset_size in generateSubsets:
                subsets[subset_size] = self.generate_best_subset(predicted_numbers, subset_size)

        return predicted_numbers, subsets


if __name__ == "__main__":
    print("Trying Markov with Fine-Tuning")

    markov = Markov()

    name = 'keno'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    markov.setDataPath(dataPath)
    markov.setSoftMAxTemperature(0.1)
    markov.setAlpha(0.5)
    markov.setMinOccurrences(5)

    if "keno" in name:
        generateSubsets = [6, 7]

    print("Predicted Numbers: ", markov.run(generateSubsets=generateSubsets))
