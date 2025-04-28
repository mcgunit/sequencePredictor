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
        self.bayesian_priors = defaultdict(lambda: 1)
        self.positional_frequencies = defaultdict(lambda: defaultdict(int))  

    def clear(self):
        self.transition_matrix.clear()
        self.pair_counts.clear()
        self.number_frequencies.clear()
        self.bayesian_priors.clear()
        self.positional_frequencies.clear()

    # 🔹 Added setters to adjust model parameters dynamically
    def setDataPath(self, dataPath):
        self.dataPath = dataPath
    
    def setSoftMaxTemperature(self, temperature):
        self.softMaxTemperature = temperature
    
    def setAlpha(self, nAlpha):
        self.alpha = nAlpha

    def setMinOccurrences(self, nMinOccurrences):
        self.min_occurrences = nMinOccurrences
    
    def setNumberOfSimulations(self, nSimulations):
        self.numberOfSimulations = nSimulations

    def build_markov_chain(self, numbers):
        total_draws = len(numbers)

        for draw_index, draw in enumerate(numbers):
            weight = 1 + (draw_index / total_draws)

            for i in range(len(draw) - 1):
                self.transition_matrix[draw[i]][draw[i + 1]] += weight
            
            for position, number in enumerate(draw):
                self.positional_frequencies[position][number] += weight

            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):  
                    self.pair_counts[draw[i]][draw[j]] += 1
                    self.pair_counts[draw[j]][draw[i]] += 1

            for num in draw:
                self.number_frequencies[num] += 1

        for position, number in enumerate(draw):
            # Instead of adding weight directly, normalize across the total count
            self.positional_frequencies[position][number] += weight / (position + 1)

    def update_bayesian_model(self, drawn_numbers):
        for num in drawn_numbers:
            self.bayesian_priors[num] += 1  

    def bayesian_prediction(self, n_predictions=20):
        total_draws = sum(self.bayesian_priors.values())
        probabilities = {num: (count / total_draws) for num, count in self.bayesian_priors.items()}
        sorted_numbers = sorted(probabilities, key=probabilities.get, reverse=True)
        predicted_numbers = sorted_numbers[:n_predictions]

        # If there are not enough numbers, fill with the most probable numbers
        while len(predicted_numbers) < n_predictions:
            predicted_numbers.append(sorted_numbers[len(predicted_numbers) % len(sorted_numbers)])

        return predicted_numbers

    def multinomial_prediction(self, n_draws=20, total_numbers=80):
        # Calculate probabilities based on historical frequencies
        total_frequency = sum(self.number_frequencies.values())
        if total_frequency == 0:
            # If no historical data, fall back to uniform distribution
            probabilities = np.ones(total_numbers) / total_numbers
        else:
            probabilities = [self.number_frequencies[i] / total_frequency for i in range(1, total_numbers + 1)]

        # Draw numbers based on the multinomial distribution
        drawn_numbers = np.random.multinomial(n_draws, probabilities)
        predicted_numbers = [i + 1 for i, count in enumerate(drawn_numbers) if count > 0]

        # If there are not enough unique numbers, fill with the most probable numbers
        while len(predicted_numbers) < n_draws:
            additional_numbers = np.random.multinomial(n_draws, probabilities)
            for i, count in enumerate(additional_numbers):
                if count > 0 and (i + 1) not in predicted_numbers:
                    predicted_numbers.append(i + 1)
                if len(predicted_numbers) >= n_draws:
                    break

        # Ensure the predicted_numbers list has exactly n_draws elements
        if len(predicted_numbers) > n_draws:
            predicted_numbers = predicted_numbers[:n_draws]

        return predicted_numbers

    
    def hypergeometric_prediction(self, n_draws=20, n_success=20, population_size=80):
        # Use historical frequencies to determine the success parameter
        historical_frequencies = np.array([self.number_frequencies[i] for i in range(1, population_size + 1)])
        total_frequencies = historical_frequencies.sum()
        
        if total_frequencies == 0:
            # If no historical data, fall back to uniform distribution
            probabilities = np.ones(population_size) / population_size
        else:
            probabilities = historical_frequencies / total_frequencies

        # Draw numbers based on the hypergeometric distribution
        drawn_numbers = []
        while len(drawn_numbers) < n_draws:
            num = np.random.choice(np.arange(1, population_size + 1), p=probabilities)
            drawn_numbers.append(num)
        
        predicted_numbers = list(set(drawn_numbers))

        # If there are not enough unique numbers, fill with the most probable numbers
        while len(predicted_numbers) < n_draws:
            additional_numbers = np.random.choice(np.arange(1, population_size + 1), p=probabilities, size=n_draws)
            for num in additional_numbers:
                if num not in predicted_numbers:
                    predicted_numbers.append(num)
                if len(predicted_numbers) >= n_draws:
                    break

        # Ensure the predicted_numbers list has exactly n_draws elements
        if len(predicted_numbers) > n_draws:
            predicted_numbers = predicted_numbers[:n_draws]

        return predicted_numbers
    

    def monte_carlo_simulation(self, n_simulations=1000, n_draws=20):
        """Perform Monte Carlo simulations while ensuring probability normalization."""
        results = []
        
        for _ in range(n_simulations):
            drawn_numbers = []

            for position in range(n_draws):
                if position in self.positional_frequencies:
                    numbers, probs = zip(*self.positional_frequencies[position].items())

                    # Normalize probabilities to sum to 1
                    probs = np.array(probs, dtype=np.float64)
                    probs_sum = probs.sum()

                    if probs_sum == 0:
                        continue  # Skip this position if no valid probabilities

                    probs /= probs_sum  # Normalize

                    drawn_number = np.random.choice(numbers, p=probs)
                    drawn_numbers.append(drawn_number)

            if drawn_numbers:
                results.append(drawn_numbers)

        # Aggregate results
        flat_results = [num for sublist in results for num in sublist]
        
        if not flat_results:
            return []

        unique, counts = np.unique(flat_results, return_counts=True)
        predicted_numbers = unique[np.argsort(-counts)][:n_draws]  # Sort by frequency

        return predicted_numbers.tolist()


    def generate_best_subset(self, predicted_numbers, nSubset):
        unique_numbers = list(set(predicted_numbers))
        if len(unique_numbers) < nSubset:
            return unique_numbers

        blended_probs = self.blended_probability({num: 1 for num in unique_numbers}, self.number_frequencies)

        # New: Sort by probability, then alternate picking low/high numbers
        sorted_numbers = sorted(unique_numbers, key=lambda x: blended_probs.get(x, 0), reverse=True)

        best_subset = []
        while len(best_subset) < nSubset and sorted_numbers:
            if len(best_subset) % 2 == 0:  # Pick a high-value number
                best_subset.append(sorted_numbers.pop(0))
            else:  # Pick a lower-value number
                best_subset.append(sorted_numbers.pop(-1))

        return sorted(best_subset)

    def blended_probability(self, markov_probs, num_frequencies):
        return {num: (self.alpha * markov_probs.get(num, 0) + (1 - self.alpha) * (num_frequencies.get(num, 0) / sum(num_frequencies.values())))
            for num in set(markov_probs) | set(num_frequencies)}

    def run(self, generateSubsets=[], skipRows=0):
        _, _, _, _, _, numbers, _, numClasses = helpers.load_data(self.dataPath, skipRows=skipRows)

        self.build_markov_chain(numbers)
        last_draw = numbers[-1]
        self.update_bayesian_model(last_draw)

        n_predictions = len(last_draw)

        bayesian_predictions = self.bayesian_prediction(n_predictions=n_predictions)
        bayesian_predictions = [int(num) for num in bayesian_predictions]
        multinomial_predictions = self.multinomial_prediction(n_draws=n_predictions, total_numbers=len(numClasses))
        hypergeometric_predictions = self.hypergeometric_prediction(n_draws=n_predictions, n_success=n_predictions, population_size=len(numClasses))
        hypergeometric_predictions = [int(num) for num in hypergeometric_predictions]
        monte_carlo_predictions = self.monte_carlo_simulation(n_simulations=self.numberOfSimulations, n_draws=n_predictions)

        print("Bayesian Predictions: ", bayesian_predictions)
        print("Multinomial Predictions: ", multinomial_predictions)
        print("Hypergeometric Predictions: ", hypergeometric_predictions)
        print("Monte Carlo Predictions: ", monte_carlo_predictions)

        # Combine all predictions
        all_predictions = ( 
            bayesian_predictions + 
            multinomial_predictions + 
            hypergeometric_predictions + 
            monte_carlo_predictions
        )

        unique, counts = np.unique(all_predictions, return_counts=True)

        # 🔹 Sort first by frequency, then numerically in case of ties
        sorted_indices = np.lexsort((unique, -counts))  # Sort first by count (descending), then by number
        sorted_numbers = unique[sorted_indices][:n_predictions]  # Take the top `n_predictions`

        # Convert to Python int
        final_predictions = [int(num) for num in sorted_numbers]

        subsets = {}
        if generateSubsets:
            # print("Creating subsets of:", generateSubsets)
            for subset_size in generateSubsets:
                subsets[subset_size] = [int(num) for num in self.generate_best_subset(final_predictions, subset_size)]

        return final_predictions, subsets

if __name__ == "__main__":
    print("Trying Hybrid Statistical Model")

    hybridStatisticalModel = HybridStatisticalModel()
    name = 'pick3'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    hybridStatisticalModel.setDataPath(dataPath)
    hybridStatisticalModel.setSoftMaxTemperature(0.1)
    hybridStatisticalModel.setAlpha(0.5)
    hybridStatisticalModel.setMinOccurrences(5)
    hybridStatisticalModel.setNumberOfSimulations(5000)

    if "keno" in name:
        generateSubsets = [6, 7]

    print("Predicted Numbers: ", hybridStatisticalModel.run(generateSubsets=generateSubsets))
