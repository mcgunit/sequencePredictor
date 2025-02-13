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
\
helpers = Helpers()

class Markov():
    def __init__(self):
        self.dataPath = ""
        self.softMaxTemperature = 0.5
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.number_frequencies = defaultdict(int)

    def setDataPath(self, dataPath):
        self.dataPath = dataPath
    
    def setSoftMAxTemperature(self, temperature):
        self.softMaxTemperature = temperature

    def softmax_with_temperature(self, probabilities, temperature=1.0):
        """Applies temperature scaling to control randomness."""
        probs = np.array(probabilities) / temperature
        return scipy.special.softmax(probs)

    def blended_probability(self, markov_probs, num_frequencies, alpha=0.7):
        """Combines Markov transition probabilities with frequency analysis."""
        return {num: (alpha * markov_probs.get(num, 0) + (1 - alpha) * (num_frequencies.get(num, 0) / sum(num_frequencies.values())))
            for num in set(markov_probs) | set(num_frequencies)}

    def build_markov_chain(self, numbers, min_occurrences=5):
        """Creates the transition matrix with weighted recency and removes rare transitions."""
        total_draws = len(numbers)

        for draw_index, draw in enumerate(numbers):
            weight = 1 + (draw_index / total_draws)  # More weight to recent draws

            for i in range(len(draw) - 1):
                self.transition_matrix[draw[i]][draw[i + 1]] += weight  # Apply weight to transitions
            
            # Count number pairs
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):  
                    self.pair_counts[draw[i]][draw[j]] += 1
                    self.pair_counts[draw[j]][draw[i]] += 1  # Ensure symmetry
            
            # Count individual number frequencies
            for num in draw:
                self.number_frequencies[num] += 1

        # Normalize transition probabilities and remove rare transitions
        for number, transitions in self.transition_matrix.items():
            total = sum(transitions.values())
            self.transition_matrix[number] = {k: v / total for k, v in transitions.items() if v >= min_occurrences}

    def predict_next_numbers(self, previous_numbers, n_predictions=20, temperature=0.7):
        """Predicts the next numbers using Markov Chain with all fine-tuning strategies."""
        predictions = set()

        for num in previous_numbers:
            if num in self.transition_matrix and self.transition_matrix[num]:
                next_nums = list(self.transition_matrix[num].keys())
                probs = list(self.transition_matrix[num].values())

                # Apply softmax temperature scaling
                adjusted_probs = self.softmax_with_temperature(probs, temperature=temperature)

                predicted_num = np.random.choice(next_nums, p=adjusted_probs)
                predictions.add(predicted_num)

        # If predictions are incomplete, use frequency-based fallback
        if len(predictions) < n_predictions:
            all_numbers = list(self.transition_matrix.keys())
            blended_probs = self.blended_probability(self.transition_matrix[num], self.number_frequencies)

            sorted_freq = sorted(blended_probs, key=blended_probs.get, reverse=True)

            while len(predictions) < n_predictions and sorted_freq:
                next_best = sorted_freq.pop(0)
                if next_best not in predictions:
                    predictions.add(next_best)

        # Ensure pair-based selection (if possible)
        while len(predictions) < n_predictions:
            last_predicted = list(predictions)[-1]
            if last_predicted in self.pair_counts:
                next_best = max(self.pair_counts[last_predicted], key=self.pair_counts[last_predicted].get, default=None)
                if next_best and next_best not in predictions:
                    predictions.add(next_best)

        return [int(num) for num in predictions][:n_predictions]

    def run(self):
        """Runs the Markov Chain prediction process."""
        
        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath)

        # Build the enhanced Markov Chain model
        self.build_markov_chain(numbers)

        # Get the last drawn numbers
        last_draw = numbers[-1]

        # Predict the next numbers
        predicted_numbers = self.predict_next_numbers(last_draw, n_predictions=len(numbers[0]), temperature=self.softMaxTemperature)
        #print("Predicted next numbers:", predicted_numbers)

        return predicted_numbers


if __name__ == "__main__":
    print("Trying Markov with Fine-Tuning")

    markov = Markov()

    name = 'keno'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    print("Most Frequent Numbers: ", helpers.count_number_frequencies(dataPath))

    markov.setDataPath(dataPath)
    markov.setSoftMAxTemperature(0.1)
    print("Predicted Numbers: ", markov.run())


