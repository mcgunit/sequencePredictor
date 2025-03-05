import os
import sys
import numpy as np
import scipy.special
from collections import defaultdict
from scipy.stats import hypergeom
from Helpers import Helpers

# Dynamically adjust the import path for Helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

# Ensure Helpers can be imported
if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

helpers = Helpers()

class HybridStatisticalModel():
    def __init__(self):
        self.dataPath = ""
        self.softMaxTemperature = 0.5
        self.alpha = 0.7
        self.min_occurrences = 5
        self.numberOfSimulations = 5000
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.number_frequencies = defaultdict(int)
        self.bayesian_priors = defaultdict(lambda: 1)  # Prior counts for Bayesian model

    def clear(self):
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.number_frequencies = defaultdict(int)
        self.bayesian_priors = defaultdict(lambda: 1)  # Reset Bayesian priors

    def setDataPath(self, dataPath):
        self.dataPath = dataPath
    
    def setSoftMAxTemperature(self, temperature):
        self.softMaxTemperature = temperature
    
    def setAlpha(self, nAlpha):
        self.alpha = nAlpha

    def setMinOccurrences(self, nMinOccurrences):
        self.min_occurrences = nMinOccurrences
    
    def setNumberOfSimulations(self, nSimulations):
        self.numberOfSimulations = nSimulations

    def generate_best_subset(self, predicted_numbers, nSubset):
        """Generate a subset of numbers using weighted selection based on Markov probabilities and frequencies."""
        unique_numbers = list(set(map(int, predicted_numbers)))  # Ensure unique standard integers

        if len(unique_numbers) < nSubset:
            return unique_numbers  # Fallback if not enough numbers

        # Compute blended probabilities using Markov and frequency data
        blended_probs = self.blended_probability({num: 1 for num in unique_numbers}, self.number_frequencies)

        # Sort numbers based on probability values
        sorted_numbers = sorted(unique_numbers, key=lambda x: blended_probs.get(x, 0), reverse=True)

        # Select the top `nSubset` numbers
        best_subset = sorted_numbers[:nSubset]

        return sorted(map(int, best_subset))

    def softmax_with_temperature(self, probabilities, temperature=1.0):
        """Applies temperature scaling to control randomness."""
        probs = np.array(probabilities) / temperature
        return scipy.special.softmax(probs)

    def blended_probability(self, markov_probs, num_frequencies):
        """Combines Markov transition probabilities with frequency analysis."""
        return {num: (self.alpha * markov_probs.get(num, 0) + (1 - self.alpha) * (num_frequencies.get(num, 0) / sum(num_frequencies.values())))
            for num in set(markov_probs) | set(num_frequencies)}

    def build_markov_chain(self, numbers):
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
            self.transition_matrix[number] = {k: v / total for k, v in transitions.items() if v >= self.min_occurrences}

    def update_bayesian_model(self, drawn_numbers):
        """Update the Bayesian model with new drawn numbers."""
        for num in drawn_numbers:
            self.bayesian_priors[num] += 1  # Update the count for the drawn number

    def bayesian_prediction(self, n_predictions=20):
        """Predict the next numbers using Bayesian inference."""
        total_draws = sum(self.bayesian_priors.values())
        probabilities = {num: (count / total_draws) for num, count in self.bayesian_priors.items()}

        # Sort numbers based on their probabilities
        sorted_numbers = sorted(probabilities, key=probabilities.get, reverse=True)

        # Select the top `n_predictions` numbers
        return sorted_numbers[:n_predictions]

    def multinomial_prediction(self, n_draws=20):
        """Predict the next numbers using a multinomial distribution."""
        total_numbers = 80  # Total numbers in Keno
        probabilities = [self.number_frequencies[i] / sum(self.number_frequencies.values()) for i in range(1, total_numbers + 1)]
        
        # Draw n_draws numbers based on the multinomial distribution
        drawn_numbers = np.random.multinomial(n_draws, probabilities)
        
        # Get the numbers that were drawn
        predicted_numbers = [i + 1 for i, count in enumerate(drawn_numbers) if count > 0]
        
        return predicted_numbers

    def hypergeometric_prediction(self, n_draws=20, n_success=20, population_size=80):
        """Predict the next numbers using a hypergeometric distribution."""
        # Parameters for hypergeometric distribution
        M = population_size  # Total population size
        n = n_success  # Total number of success states in the population
        N = n_draws  # Number of draws

        # Draw numbers based on hypergeometric distribution
        drawn_numbers = hypergeom.rvs(M, n, N, size=1)

        # Get the drawn numbers
        predicted_numbers = [i for i in range(1, M + 1) if drawn_numbers[0] == i]
        
        return predicted_numbers
    
    def monte_carlo_simulation(self, n_simulations=1000, n_draws=20, total_numbers=80):
        """
        Perform Monte Carlo simulations to predict Keno outcomes.
        
        Parameters:
        n_simulations (int): Number of Monte Carlo simulations to run.
        n_draws (int): Number of draws to simulate in each run.
        
        Returns:
        list: A list of predicted numbers based on the simulations.
        """
        total_numbers = total_numbers  # Total numbers in Keno
        results = []

        for _ in range(n_simulations):
            # Simulate a draw based on the existing frequency distribution
            probabilities = [self.number_frequencies[i] / sum(self.number_frequencies.values()) for i in range(1, total_numbers + 1)]
            drawn_numbers = np.random.choice(range(1, total_numbers + 1), size=n_draws, replace=False, p=probabilities)
            results.append(drawn_numbers)

        # Aggregate results to find the most frequently drawn numbers
        flat_results = [num for sublist in results for num in sublist]
        unique, counts = np.unique(flat_results, return_counts=True)
        predicted_numbers = unique[np.argsort(-counts)][:n_draws]  # Get the top n_draws numbers

        return predicted_numbers.tolist()


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

    def run(self, generateSubsets=[]):
        """
        Runs the Markov Chain prediction process with optional subset generation.
        
        Parameters:
        generateSubsets (list): List of subset sizes to generate, e.g., [6, 7] will generate subsets of size 6 and 7.
        """
        _, _, _, _, _, numbers, _, numClasses = helpers.load_data(self.dataPath)

        # Build the enhanced Markov Chain model
        self.build_markov_chain(numbers)

        # Get the last drawn numbers
        last_draw = numbers[-1]

        # Update Bayesian model with the last drawn numbers
        self.update_bayesian_model(last_draw)

        # Predict n_predictions numbers first
        predicted_numbers = self.predict_next_numbers(last_draw, n_predictions=len(last_draw), temperature=self.softMaxTemperature)

        # Get predictions from the Bayesian model
        bayesian_predictions = self.bayesian_prediction(n_predictions=len(last_draw))

        # Get predictions from the multinomial model
        multinomial_predictions = self.multinomial_prediction(n_draws=len(last_draw))

        # Get predictions from the hypergeometric model
        hypergeometric_predictions = self.hypergeometric_prediction(n_draws=len(last_draw), population_size=len(numClasses))

        # Get predictions from the Monte Carlo simulation
        monte_carlo_predictions = self.monte_carlo_simulation(n_simulations=self.numberOfSimulations, n_draws=len(last_draw), total_numbers=len(numClasses))

        # Combine predictions from all models
        combined_predictions = list(set(predicted_numbers) | set(bayesian_predictions) | set(multinomial_predictions) | set(hypergeometric_predictions) | set(monte_carlo_predictions))

        # Convert combined predictions from np.int64 to Python int
        combined_predictions = [int(num) for num in combined_predictions]

        # Generate subsets if requested
        subsets = {}
        if generateSubsets:
            print("Creating subsets of:", generateSubsets)
            for subset_size in generateSubsets:
                subsets[subset_size] = self.generate_best_subset(combined_predictions, subset_size)

        # Convert subsets from np.int64 to Python int
        subsets = {size: [int(num) for num in numbers] for size, numbers in subsets.items()}

        return combined_predictions, subsets


if __name__ == "__main__":
    print("Trying Markov with Fine-Tuning, Bayesian Model, Multinomial, and Hypergeometric Distributions")

    hybridStatisticalModel = HybridStatisticalModel()

    name = 'keno'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    hybridStatisticalModel.setDataPath(dataPath)
    hybridStatisticalModel.setSoftMAxTemperature(0.1)
    hybridStatisticalModel.setAlpha(0.5)
    hybridStatisticalModel.setMinOccurrences(5)
    hybridStatisticalModel.setNumberOfSimulations(5000)

    if "keno" in name:
        generateSubsets = [6, 7]

    print("Predicted Numbers: ", hybridStatisticalModel.run(generateSubsets=generateSubsets))