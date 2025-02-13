import os, sys
import numpy as np
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

    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    # Function to predict next likely numbers
    def predict_next_numbers(self, transition_matrix, previous_numbers, n_predictions=20):
        predictions = set()  # Use a set to ensure uniqueness
        
        for num in previous_numbers:
            if num in transition_matrix and transition_matrix[num]:
                next_nums = list(transition_matrix[num].keys())
                probs = list(transition_matrix[num].values())
                predicted_num = np.random.choice(next_nums, p=probs)

                # Add to predictions only if it's unique
                if predicted_num not in predictions:
                    predictions.add(predicted_num)

        # If we don't have enough predictions, fill in with most frequent numbers
        if len(predictions) < n_predictions:
            all_numbers = list(transition_matrix.keys())
            frequency = {num: sum(transition_matrix[num].values()) for num in all_numbers}
            sorted_freq = sorted(frequency, key=frequency.get, reverse=True)

            while len(predictions) < n_predictions and sorted_freq:
                next_best = sorted_freq.pop(0)
                if next_best not in predictions:
                    predictions.add(next_best)

        return [int(num) for num in predictions][:n_predictions]  # Convert np.int64 to int


    def run(self):
        # Load data using the helper function
        data_folder = self.dataPath 
        _, _, _, _, _, numbers, _, _ = helpers.load_data(data_folder)

        # Create a transition matrix
        transition_matrix = defaultdict(lambda: defaultdict(int))

        # Count transitions between numbers
        for draw in numbers:
            for i in range(len(draw) - 1):
                transition_matrix[draw[i]][draw[i + 1]] += 1

        # Convert counts to probabilities
        for number, transitions in transition_matrix.items():
            total = sum(transitions.values())
            for next_num in transitions:
                transition_matrix[number][next_num] /= total  # Normalize to probability

        # Get the last drawn numbers from the dataset
        last_draw = numbers[-1]

        # Predict the next numbers based on the last draw
        predicted_numbers = self.predict_next_numbers(transition_matrix, last_draw, n_predictions=len(numbers[0]))
        #print("Predicted next numbers:", predicted_numbers)



if __name__ == "__main__":
    print("Trying markov")

    markov = Markov()

    name = 'keno'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    markov.setDataPath(dataPath)

    markov.run()


