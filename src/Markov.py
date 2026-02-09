import os, sys, json
import numpy as np
import scipy.special
from collections import defaultdict
import itertools

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
        
        # --- CONFIGURATION FLAGS ---
        self.markov_order = 1
        self.use_pair_scoring = False
        self.pair_scoring_weight = 1.0
        self.sorted_prediction = False # NEW: Replaces Deltas. Enforces X > Prev_X.
        
        # Data Structures
        self.transition_matrices = [] 
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        
        # NEW: Column-Specific Frequencies (Critical for Keno Ranges)
        # col_frequencies[0] stores freq for Col 1 (1-5 range)
        # col_frequencies[19] stores freq for Col 20 (60-70 range)
        self.col_frequencies = []
        
        # Global frequencies for Subset Generation
        self.global_frequencies = defaultdict(int)
        
        self.normalized_pairs = defaultdict(lambda: defaultdict(float))

        self.recency_weight = 1.0
        self.recency_mode = "linear"
        self.pair_decay_factor = 0.9
        self.smoothing_factor = 0.01
        self.subset_selection_mode = "softmax"
        self.blend_mode = "linear"
    
    def clear(self):
        self.transition_matrices = []
        self.col_frequencies = []
        self.global_frequencies = defaultdict(int)
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.normalized_pairs = defaultdict(lambda: defaultdict(float))

    # --- SETTERS ---
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
    def setMarkovOrder(self, order): self.markov_order = max(1, int(order))
    def setUsePairScoring(self, use): self.use_pair_scoring = bool(use)
    def setPairScoringWeight(self, w): self.pair_scoring_weight = float(w)
    
    def setSortedPrediction(self, use):
        """
        Enable for Keno, Lotto, EuroMillions.
        Enforces that the predicted sequence is strictly increasing.
        """
        self.sorted_prediction = bool(use)

    def softmax_with_temperature(self, probabilities, temperature=1.0):
        # FIX: Convert linear probabilities to logits before applying softmax
        probs = np.array(probabilities)
        # Add epsilon to avoid log(0)
        logits = np.log(probs + 1e-9)
        
        if temperature < 1e-5:
            idx = np.argmax(logits)
            p = np.zeros_like(probs)
            p[idx] = 1.0
            return p
            
        # Apply temperature to logits
        scaled_logits = logits / temperature
        return scipy.special.softmax(scaled_logits)

    def blended_probability(self, markov_probs, num_frequencies):
        # num_frequencies here is the COLUMN-SPECIFIC frequency
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
        self.clear()
        
        if len(numbers) <= self.markov_order: 
            return

        num_columns = len(numbers[0])
        self.transition_matrices = [defaultdict(lambda: defaultdict(int)) for _ in range(num_columns)]
        self.col_frequencies = [defaultdict(int) for _ in range(num_columns)]
        
        total_draws = len(numbers)

        for t in range(self.markov_order, total_draws):
            target_draw = numbers[t]
            
            if self.recency_mode == "linear":
                weight = 1 + (self.recency_weight * t / total_draws)
            elif self.recency_mode == "log":
                weight = 1 + np.log1p(t) * self.recency_weight
            else:
                weight = 1.0

            recency_factor = self.pair_decay_factor ** (total_draws - t)

            # 1. Transitions
            for col_idx in range(num_columns):
                # Context is the tuple of previous 'order' numbers in this specific column
                context = tuple(int(numbers[i][col_idx]) for i in range(t - self.markov_order, t))
                v = int(target_draw[col_idx])
                self.transition_matrices[col_idx][context][v] += weight
                
                # Update Column-Specific Frequency
                self.col_frequencies[col_idx][v] += weight

            # 2. Pairwise Counts 
            for i in range(len(target_draw)):
                for j in range(i + 1, len(target_draw)):
                    n1, n2 = int(target_draw[i]), int(target_draw[j])
                    k1, k2 = sorted((n1, n2))
                    self.pair_counts[k1][k2] += weight * recency_factor
            
            # 3. Global Frequencies (for Subset Generation)
            for num in target_draw:
                self.global_frequencies[int(num)] += weight

        self._normalize_matrices()

    def _normalize_matrices(self):
        for col_idx in range(len(self.transition_matrices)):
            raw_matrix = self.transition_matrices[col_idx]
            cleaned = {}
            for ctx, transitions in raw_matrix.items():
                filtered = {k: v for k, v in transitions.items() if v >= self.min_occurrences}
                if not filtered: continue
                total = sum(filtered.values()) + self.smoothing_factor * len(filtered)
                cleaned[ctx] = {
                    int(k): (v + self.smoothing_factor) / total
                    for k, v in filtered.items()
                }
            self.transition_matrices[col_idx] = cleaned
            
        total_pair_weight = sum(sum(d.values()) for d in self.pair_counts.values()) or 1
        for n1, d in self.pair_counts.items():
            for n2, w in d.items():
                self.normalized_pairs[n1][n2] = w / total_pair_weight

    def predict_next_numbers(self, history_draws, temperature=0.7):
        if len(history_draws) < self.markov_order:
            # Fallback
            width = len(history_draws[0]) if history_draws and len(history_draws[0]) > 0 else 3
            return [np.random.randint(1, 10) for _ in range(width)]

        relevant_history = history_draws[-self.markov_order:]
        num_columns = len(relevant_history[0])

        def get_col_probs(col_idx, min_val_constraint=None):
            context = tuple(int(draw[col_idx]) for draw in relevant_history)
            matrix = self.transition_matrices[col_idx] if col_idx < len(self.transition_matrices) else {}
            
            # Use Column-Specific Frequencies for blending
            col_freqs = self.col_frequencies[col_idx] if col_idx < len(self.col_frequencies) else defaultdict(int)
            
            if context in matrix:
                markov_dist = matrix[context]
                blended = self.blended_probability(markov_dist, col_freqs)
            else:
                # Fallback to column frequencies
                total = sum(col_freqs.values()) or 1
                blended = {k: v/total for k, v in col_freqs.items()}
            
            candidates = list(blended.keys())
            probs = list(blended.values())
            
            # --- FILTERING FOR SORTED PREDICTION ---
            if min_val_constraint is not None:
                # We need number > min_val_constraint
                filtered_cands = []
                filtered_probs = []
                for c, p in zip(candidates, probs):
                    if c > min_val_constraint:
                        filtered_cands.append(c)
                        filtered_probs.append(p)
                
                if not filtered_cands:
                    # Soft fallback: if no valid candidates, return empty to trigger hard fallback
                    return [], []
                
                candidates = filtered_cands
                probs = filtered_probs
                
                # Re-normalize sums to 1
                total_p = sum(probs)
                if total_p > 0:
                    probs = [p / total_p for p in probs]

            if not candidates: 
                 return [], []
            
            adj_probs = self.softmax_with_temperature(probs, temperature)
            return candidates, adj_probs

        # --- SAFETY SWITCH FOR PAIR SCORING ---
        local_use_pair_scoring = self.use_pair_scoring
        if local_use_pair_scoring and num_columns > 6:
            print(f"Warning: Disabling Pair Scoring. Too many columns ({num_columns}).")
            local_use_pair_scoring = False
            
        prediction = []
        last_pred_val = -1 # Keno numbers are > 0

        if not local_use_pair_scoring:
            # Independent Column Prediction
            for col in range(num_columns):
                constraint = last_pred_val if self.sorted_prediction else None
                
                cands, p = get_col_probs(col, min_val_constraint=constraint)
                
                if not cands:
                    # Hard Fallback: Last val + 1 (or 1 if first)
                    pred = last_pred_val + 1 if last_pred_val > 0 else 1
                else:
                    # Ensure sum 1.0
                    p = p / p.sum()
                    pred = int(np.random.choice(cands, p=p))
                
                prediction.append(pred)
                last_pred_val = pred
        else:
            # Pair Scoring (Pick3 only)
            # Note: Pair scoring with 'sorted_prediction' is complex. 
            # We assume Pair Scoring is only used for Pick3 where sorted_prediction=False.
            top_k = 4
            col_candidates = []
            for col in range(num_columns):
                cands, p = get_col_probs(col) # No constraint here, we score later? 
                # Actually for Pick3 we don't constrain.
                zipped = sorted(zip(cands, p), key=lambda x: x[1], reverse=True)
                col_candidates.append(zipped[:top_k])
            
            all_combinations = list(itertools.product(*[c[0] for c in col_candidates]))
            all_probs = list(itertools.product(*[c[1] for c in col_candidates]))
            
            final_scores = []
            
            for i, combo in enumerate(all_combinations):
                base_prob_score = np.sum(np.log(np.array(all_probs[i]) + 1e-9))
                pair_score = 0
                sorted_combo = sorted(combo)
                pairs = itertools.combinations(sorted_combo, 2)
                for p1, p2 in pairs:
                    w = self.normalized_pairs[p1].get(p2, 0)
                    pair_score += np.log(w + 1e-9)
                
                total = base_prob_score + (self.pair_scoring_weight * pair_score)
                final_scores.append(total)
            
            final_scores = np.array(final_scores)
            final_scores = final_scores - final_scores.max()
            final_probs = np.exp(final_scores)
            final_probs = final_probs / final_probs.sum()
            
            idx = np.random.choice(len(all_combinations), p=final_probs)
            prediction = list(all_combinations[idx])

        return prediction

    def generate_best_subset(self, predicted_numbers, nSubset):
        unique_numbers = list(dict.fromkeys(map(int, predicted_numbers)))
        
        if len(unique_numbers) < nSubset:
            # Fallback to global frequent numbers
            sorted_freq = sorted(self.global_frequencies, key=self.global_frequencies.get, reverse=True)
            for f in sorted_freq:
                if f not in unique_numbers:
                    unique_numbers.append(f)
                if len(unique_numbers) >= nSubset: break
                
            # Random fallback if still empty
            while len(unique_numbers) < nSubset:
                r = np.random.randint(1, 81)
                if r not in unique_numbers:
                    unique_numbers.append(r)
        
        return unique_numbers[:nSubset]

    def run(self, generateSubsets=[], skipRows=0):
        _, _, _, _, _, numbers, _, _ = helpers.load_data(self.dataPath, skipRows=skipRows)
        if len(numbers) == 0: return [], {}

        self.build_markov_chain(numbers)

        history_context = numbers[-self.markov_order:]
        predicted_numbers = self.predict_next_numbers(history_context, temperature=self.softMaxTemperature)

        subsets = {}
        if generateSubsets:
            for subset_size in generateSubsets:
                subsets[subset_size] = self.generate_best_subset(predicted_numbers, subset_size)

        return predicted_numbers, subsets

if __name__ == "__main__":
    print("Trying Markov")

    markov = Markov()
    name = 'keno' 
    generateSubsets = []
    
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    markov.setDataPath(dataPath)

    markov.setSoftMAxTemperature(0.45)
    markov.setAlpha(0.6)
    markov.setMinOccurrences(2) 
    markov.setRecencyWeight(1.7)
    markov.setRecencyMode("constant")
    markov.setPairDecayFactor(1)

    # --- GAME CONFIGURATION ---
    if "keno" in name.lower() or "lotto" in name.lower() or "euro" in name.lower():
        # Sorted Games: Use Sorted Prediction + Absolute Numbers
        markov.setSortedPrediction(True)
        markov.setUsePairScoring(False)
        markov.setMarkovOrder(2)
    else:
        # Positional Games (Pick3): Use Unsorted + Pair Scoring
        markov.setSortedPrediction(False)
        markov.setUsePairScoring(True)
        markov.setPairScoringWeight(0.1)
        markov.setMarkovOrder(2)

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-6-15.json")

    try:
        with open(sequenceToPredictFile, 'r') as openfile:
            sequenceToPredict = json.load(openfile)
        print("Real result: ", sequenceToPredict["realResult"])
    except:
        pass

    if "keno" in name.lower():
        generateSubsets = [6, 7]
        
    for _ in range(1):
        print("Predicted Numbers: ", markov.run(generateSubsets=generateSubsets))