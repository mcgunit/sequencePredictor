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

    def clear(self):
        self.position_stats = defaultdict(lambda: [])
    
    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    def setNumOfSimulations(self, nSimulations):
        self.num_simulations = nSimulations
    
    def setRecentDraws(self, nRecentDraws):
        self.recent_draws = nRecentDraws

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
    
    def build_laplace_model(self, numbers):
        """Computes Laplace distribution parameters for each position."""
        total_draws = len(numbers)
        
        self.min_number = min(min(draw) for draw in numbers)
        self.max_number = max(max(draw) for draw in numbers)
        
        for i, draw in enumerate(numbers[-self.recent_draws:]):  # Consider only recent draws
            for pos, num in enumerate(draw):
                self.position_stats[pos].append(num)
        
        # Compute Laplace parameters (location = mean, scale = MAD/sqrt(2))
        self.laplace_params = {
            pos: (np.mean(nums), max(0.1, np.median(np.abs(nums - np.median(nums))) / np.sqrt(2)))
            for pos, nums in self.position_stats.items()
        }
    
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

        return predicted_numbers
    
    def run(self, generateSubsets=[], skipRows=0):
        """Runs the Laplace Monte Carlo prediction process with optional subset generation."""
        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath, skipRows=skipRows)

        self.setRecentDraws(max(self.recent_draws, len(numbers)))

        self.build_laplace_model(numbers)
        
        predicted_numbers = self.monte_carlo_simulation(n_predictions=len(numbers[-1]))
        
        subsets = {}
        if generateSubsets:
            # print("Creating subsets of: ", generateSubsets)
            for nPredictions in generateSubsets:
                subsets[nPredictions] = self.generate_best_subset(predicted_numbers, nPredictions)
        
        return predicted_numbers, subsets

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

