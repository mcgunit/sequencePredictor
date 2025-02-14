import os,sys
import numpy as np
from scipy.stats import poisson
from scipy.special import softmax
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

class PoissonMonteCarlo():
    def __init__(self):
        self.dataPath = ""
        self.num_simulations = 5000  # Fine-tuned Monte Carlo simulations
        self.recent_weight_factor = 1.5  # Weighting for recent draws
        self.recent_draws = 100 # look back
        self.position_frequencies = defaultdict(lambda: defaultdict(int))

    def clear(self):
        self.position_frequencies = defaultdict(lambda: defaultdict(int))
    
    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    def setNumOfSimulations(self, nSimulations):
        self.num_simulations = nSimulations
    
    def setWeightFactor(self, nWeightFactor):
        self.recent_weight_factor = nWeightFactor

    def setRecentDraws(self, nRecentDraws):
        self.recent_draws = nRecentDraws
    
    def build_poisson_model(self, numbers):
        """Computes the Poisson distribution parameters for each position."""
        total_draws = len(numbers)
        
        for i, draw in enumerate(numbers):
            weight = self.recent_weight_factor if i >= total_draws - self.recent_draws else 1.0
            for pos, num in enumerate(draw):
                self.position_frequencies[pos][num] += weight
        
        # Compute Poisson lambda per position
        self.poisson_lambda = {
            pos: {num: count / total_draws for num, count in self.position_frequencies[pos].items()}
            for pos in self.position_frequencies
        }
    
    def monte_carlo_simulation(self, n_predictions=20):
        """Runs Monte Carlo simulations using Poisson distribution for each position."""
        predicted_numbers = []
        
        for pos in range(n_predictions):
            simulated_counts = defaultdict(int)
            
            for _ in range(self.num_simulations):
                for num, lam in self.poisson_lambda.get(pos, {}).items():
                    occurrence = poisson.rvs(lam)
                    if occurrence > 0:
                        simulated_counts[num] += 1
            
            # Normalize and apply softmax filtering
            raw_values = np.array([simulated_counts[num] for num in simulated_counts])
            probabilities = softmax(raw_values)
            sorted_predictions = [num for _, num in sorted(zip(probabilities, simulated_counts.keys()), reverse=True)]
            
            predicted_numbers.append(sorted_predictions[0] if sorted_predictions else np.random.randint(1, 81))
        
        return [int(num) for num in predicted_numbers]
    
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
    
    #print("Most Frequent Numbers: ", helpers.count_number_frequencies(dataPath))
    
    model.setDataPath(dataPath)
    model.setNumOfSimulations(5000)
    model.setRecentDraws(2000)
    model.setWeightFactor(0.1)
    
    print("Predicted Numbers: ", model.run())
