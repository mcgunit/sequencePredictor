import os, sys, json
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
        
        # We now need a list of transition matrices (one per position/column)
        # transition_matrices[0] stores transitions for Column 1, etc.
        self.transition_matrices = [] 
        
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.number_frequencies = defaultdict(int)

        self.recency_weight = 1.0
        self.recency_mode = "linear"
        self.pair_decay_factor = 0.9
        self.smoothing_factor = 0.01
        self.subset_selection_mode = "softmax"
        self.blend_mode = "linear"
    
    def clear(self):
        self.transition_matrices = []
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.number_frequencies = defaultdict(int)

    # ... Setters (Keep your existing setters) ...
    def setDataPath(self, dataPath): self.dataPath = dataPath
    def setSoftMAxTemperature(self, t): self.softMaxTemperature = t
    def setAlpha(self, a): self.alpha = a
    def setMinOccurrences(self, n): self.min_occurrences = n
    def setRecencyWeight(self, w): self.recency_weight = w
    def setRecencyMode(self, m): self.recency_mode = m
    def setPairDecayFactor(self, d): self.pair_decay_factor = d
    def setSmoothingFactor(self, s): self.smoothing_factor = s
    def setSubsetSelectionMode(self, m): self.subset_selection_mode = m
    def setBlendMode(self, m): self.blend_mode = m

    def softmax_with_temperature(self, probabilities, temperature=1.0):
        probs = np.array(probabilities)
        # Safety for 0 temperature
        if temperature < 1e-5:
            return probs / probs.sum()
        probs = probs / temperature
        return scipy.special.softmax(probs)

    def blended_probability(self, markov_probs, num_frequencies):
        """Combines Markov transition probabilities with frequency analysis."""
        total_freq = sum(num_frequencies.values()) or 1
        all_nums = set(map(int, markov_probs)) | set(map(int, num_frequencies))
        blended = {}

        for num in all_nums:
            mp = markov_probs.get(num, 0)
            freq = num_frequencies.get(num, 0) / total_freq

            if self.blend_mode == "log":
                blended[num] = np.log1p(mp) + np.log1p(freq)
            elif self.blend_mode == "harmonic":
                blended[num] = 2 * mp * freq / (mp + freq + 1e-8)
            else:  # linear
                blended[num] = self.alpha * mp + (1 - self.alpha) * freq
        return blended

    def build_markov_chain(self, numbers):
        """
        Builds TEMPORAL Markov chains for each column.
        numbers: List of draws, where each draw is [col1, col2, col3].
        Assumes 'numbers' is sorted chronologically (Oldest -> Newest).
        """
        self.clear()
        #if not numbers: return

        num_columns = len(numbers[0])
        # Initialize one dict per column
        self.transition_matrices = [defaultdict(lambda: defaultdict(int)) for _ in range(num_columns)]
        
        total_draws = len(numbers)

        for t in range(total_draws - 1):
            current_draw = numbers[t]
            next_draw = numbers[t + 1]

            # Calculate Weight based on T (Recency)
            if self.recency_mode == "linear":
                weight = 1 + (self.recency_weight * t / total_draws)
            elif self.recency_mode == "log":
                weight = 1 + np.log1p(t) * self.recency_weight
            else:
                weight = 1.0

            recency_factor = self.pair_decay_factor ** (total_draws - t)

            # 1. Temporal Transitions (Per Column)
            for col_idx in range(num_columns):
                u = int(current_draw[col_idx])
                v = int(next_draw[col_idx])
                self.transition_matrices[col_idx][u][v] += weight

            # 2. Pairwise Counts (Intra-draw, for 'Boxed' or global patterns)
            # We use the 'next_draw' for frequency counting to align with the weight
            for i in range(len(next_draw)):
                for j in range(i + 1, len(next_draw)):
                    n1, n2 = int(next_draw[i]), int(next_draw[j])
                    self.pair_counts[n1][n2] += weight * recency_factor
                    self.pair_counts[n2][n1] += weight * recency_factor
            
            # 3. Frequencies
            for num in next_draw:
                self.number_frequencies[int(num)] += weight

        # Normalize Matrices
        self._normalize_matrices()

    def _normalize_matrices(self):
        """Helper to normalize all transition matrices."""
        for col_idx in range(len(self.transition_matrices)):
            raw_matrix = self.transition_matrices[col_idx]
            cleaned = {}
            for number, transitions in raw_matrix.items():
                filtered = {k: v for k, v in transitions.items() if v >= self.min_occurrences}
                if not filtered: continue
                
                total = sum(filtered.values()) + self.smoothing_factor * len(filtered)
                cleaned[number] = {
                    int(k): (v + self.smoothing_factor) / total
                    for k, v in filtered.items()
                }
            self.transition_matrices[col_idx] = cleaned

    def predict_next_numbers(self, previous_draw, temperature=0.7):
        """
        Predicts the next number for EACH column specifically.
        previous_draw: [col1, col2, col3] from the last known draw.
        """
        predictions = []
        previous_draw = list(map(int, previous_draw))
        
        # We need to predict one number per column
        for col_idx, prev_num in enumerate(previous_draw):
            matrix = self.transition_matrices[col_idx] if col_idx < len(self.transition_matrices) else {}
            
            # Get Markov Probabilities for this column
            if prev_num in matrix:
                next_nums = list(matrix[prev_num].keys())
                probs = list(matrix[prev_num].values())
                
                # Blend with global frequencies to handle sparsity
                # (We treat the matrix prob as primary, frequency as fallback/smoothing)
                markov_probs_dict = dict(zip(next_nums, probs))
                blended = self.blended_probability(markov_probs_dict, self.number_frequencies)
                
                candidates = list(blended.keys())
                candidate_probs = list(blended.values())
            else:
                # Fallback: Just use global frequencies if no transition history for this number
                candidates = list(self.number_frequencies.keys())
                total_freq = sum(self.number_frequencies.values())
                candidate_probs = [self.number_frequencies[c]/total_freq for c in candidates]

            # Selection
            if not candidates:
                predictions.append(np.random.randint(0, 10)) # Absolute fallback
                continue

            # Apply Temperature Softmax
            adjusted_probs = self.softmax_with_temperature(candidate_probs, temperature=temperature)
            
            # Renormalize ensures sum is 1.0 explicitly
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            pred_num = int(np.random.choice(candidates, p=adjusted_probs))
            predictions.append(pred_num)

        return predictions

    def generate_best_subset(self, predicted_numbers, nSubset):
        # Keeps your original logic for generating subsets (e.g. for Keno)
        # Using the predicted numbers as seeds
        unique_numbers = list(set(map(int, predicted_numbers)))
        if len(unique_numbers) < nSubset:
            # Add top frequent numbers to fill
            sorted_freq = sorted(self.number_frequencies, key=self.number_frequencies.get, reverse=True)
            for f in sorted_freq:
                if f not in unique_numbers:
                    unique_numbers.append(f)
                if len(unique_numbers) >= nSubset: break
        
        return unique_numbers[:nSubset]

    def run(self, generateSubsets=[], skipRows=0):
        # Load Data
        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath, skipRows=skipRows)
        

        # Numbers is sorted oldest -> newest
        print("number: ", numbers[0])

        self.build_markov_chain(numbers)

        last_draw = numbers[-1] # The most recent draw
        predicted_numbers = self.predict_next_numbers(last_draw, temperature=self.softMaxTemperature)

        subsets = {}
        if generateSubsets:
            for subset_size in generateSubsets:
                subsets[subset_size] = self.generate_best_subset(predicted_numbers, subset_size)

        return predicted_numbers, subsets

if __name__ == "__main__":
    print("Trying Markov")

    markov = Markov()

    name = 'pick3'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    markov.setDataPath(dataPath)
    markov.setSoftMAxTemperature(0.45)
    markov.setAlpha(0.6)
    markov.setMinOccurrences(2) 
    markov.setRecencyWeight(1.7)

    markov.setSubsetSelectionMode("softmax")
    markov.setBlendMode("linear")
    markov.setRecencyMode("constant")
    #markov.setPairDecayFactor(1.2286344216885279)
    markov.setPairDecayFactor(1)

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-08-02.json")

    # Opening JSON file
    with open(sequenceToPredictFile, 'r') as openfile:
        sequenceToPredict = json.load(openfile)

    print("Real result: ", sequenceToPredict["realResult"])

    if "keno" in name:
        generateSubsets = [6, 7]
    for _ in range(1):
        print("Predicted Numbers: ", markov.run(generateSubsets=generateSubsets))
