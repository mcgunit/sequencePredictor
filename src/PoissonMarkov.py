import os, sys
import numpy as np
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

    def setDataPath(self, dataPath):
        """Set data path for both models."""
        self.poisson_model.setDataPath(dataPath)
        self.markov_model.setDataPath(dataPath)

    def setWeights(self, poisson_weight=0.5, markov_weight=0.5):
        """Adjust the weight contributions of Poisson and Markov models."""
        total = poisson_weight + markov_weight
        self.poisson_weight = poisson_weight / total
        self.markov_weight = markov_weight / total

    def blend_predictions(self, poisson_numbers, markov_numbers, n_predictions=20):
        """Blend predictions from both models using weighted probability selection."""
        combined_counts = {}

        for num in poisson_numbers:
            combined_counts[num] = combined_counts.get(num, 0) + self.poisson_weight

        for num in markov_numbers:
            combined_counts[num] = combined_counts.get(num, 0) + self.markov_weight

        # Sort numbers based on combined probabilities
        sorted_numbers = sorted(combined_counts, key=combined_counts.get, reverse=True)

        return sorted_numbers[:n_predictions]  # Select top numbers

    def generate_best_subset(self, predicted_numbers, nSubset):
        """Generate a subset using probability-based selection."""
        unique_numbers = list(set(predicted_numbers))

        if len(unique_numbers) < nSubset:
            return unique_numbers  # Fallback if not enough numbers

        probabilities = np.linspace(1.0, 0.5, len(unique_numbers))
        probabilities /= probabilities.sum()

        best_subset = np.random.choice(unique_numbers, size=nSubset, replace=False, p=probabilities)

        return sorted(map(int, best_subset))

    def run(self, generateSubsets=[]):
        """Runs both models, blends predictions, and generates subsets if needed."""
        poisson_numbers, _ = self.poisson_model.run()  # Extract only the first element
        markov_numbers, _ = self.markov_model.run()  # Extract only the first element

        # Flatten nested lists if necessary
        if isinstance(poisson_numbers[0], list):
            poisson_numbers = [num for sublist in poisson_numbers for num in sublist]
        if isinstance(markov_numbers[0], list):
            markov_numbers = [num for sublist in markov_numbers for num in sublist]

        hybrid_predictions = self.blend_predictions(poisson_numbers, markov_numbers)

        subsets = {}
        if generateSubsets:
            for subset_size in generateSubsets:
                subsets[subset_size] = self.generate_best_subset(hybrid_predictions, subset_size)

        return hybrid_predictions, subsets

    

if __name__ == "__main__":
    print("Running Hybrid Poisson-Markov Model")

    hybrid_model = HybridPoissonMarkov()

    name = 'keno'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    hybrid_model.setDataPath(dataPath)
    hybrid_model.setWeights(poisson_weight=0.6, markov_weight=0.4) 

    if "keno" in name:
        generateSubsets = [6, 7]

    predicted_numbers, subsets = hybrid_model.run(generateSubsets=generateSubsets)

    print("Predicted Numbers:", predicted_numbers)
    print("Generated Subsets:", subsets)