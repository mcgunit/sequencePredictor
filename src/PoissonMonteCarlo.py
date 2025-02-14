import os
import numpy as np
from scipy.stats import poisson
from collections import defaultdict
from Helpers import Helpers

helpers = Helpers()

class PoissonMonteCarlo():
    def __init__(self):
        self.dataPath = ""
        self.num_simulations = 1000  # Number of Monte Carlo simulations
        self.number_frequencies = defaultdict(int)
    
    def setDataPath(self, dataPath):
        self.dataPath = dataPath
    
    def build_poisson_model(self, numbers):
        """Computes the Poisson distribution parameters based on historical data."""
        num_counts = defaultdict(int)
        total_draws = len(numbers)
        
        for draw in numbers:
            for num in draw:
                num_counts[num] += 1
                self.number_frequencies[num] += 1
        
        # Compute Poisson lambda (mean occurrences per draw)
        self.poisson_lambda = {num: count / total_draws for num, count in num_counts.items()}
    
    def monte_carlo_simulation(self, n_predictions=20):
        """Runs Monte Carlo simulations using Poisson distribution."""
        simulated_counts = defaultdict(int)
        
        for _ in range(self.num_simulations):
            simulated_draw = []
            for num in range(1, max(self.poisson_lambda.keys()) + 1):
                if num in self.poisson_lambda:
                    occurrence = poisson.rvs(self.poisson_lambda[num])
                    if occurrence > 0:
                        simulated_draw.append(num)
            
            for num in simulated_draw:
                simulated_counts[num] += 1
        
        # Select top predicted numbers
        sorted_predictions = sorted(simulated_counts, key=simulated_counts.get, reverse=True)
        return sorted_predictions[:n_predictions]
    
    def run(self):
        """Runs the Poisson Monte Carlo prediction process."""
        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath)
        self.build_poisson_model(numbers)
        predicted_numbers = self.monte_carlo_simulation(n_predictions=len(numbers[0]))
        return predicted_numbers

if __name__ == "__main__":
    print("Running Poisson Monte Carlo Simulation")
    
    model = PoissonMonteCarlo()
    
    name = 'keno'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    
    print("Most Frequent Numbers: ", helpers.count_number_frequencies(dataPath))
    
    model.setDataPath(dataPath)
    print("Predicted Numbers: ", model.run())
