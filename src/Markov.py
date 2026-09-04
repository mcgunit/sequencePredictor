import os, sys, json, itertools
import numpy as np
import scipy.special
from collections import defaultdict
from collections import Counter

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
        self.min_number = 1
        self.max_number = 80
        self.draw_size = None
        self.random_seed = None
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
    def setGameRange(self, min_number, max_number):
        self.min_number = int(min_number)
        self.max_number = int(max_number)

    def setDrawSize(self, draw_size):
        self.draw_size = int(draw_size)

    def setRandomSeed(self, seed):
        self.random_seed = seed
        np.random.seed(seed)
    
    def setSortedPrediction(self, use):
        """
        Enable for Keno, Lotto, EuroMillions.
        Enforces that the predicted sequence is strictly increasing.
        """
        self.sorted_prediction = bool(use)

    def load_numbers(self, skipRows=0, skipLastColumns=0, years_back=None, specialColumnCount=0):
        _, _, _, _, _, numbers, num_classes, unique_labels = helpers.load_data(
            self.dataPath,
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            years_back=years_back,
            specialColumnCount=specialColumnCount
        )
        return numbers, num_classes, unique_labels

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

    def _column_distribution(self, relevant_history, col_idx, temperature, min_val_constraint=None):
        """
        One column's next-value distribution: the Markov transition row for
        this column's context blended with the column's own frequencies (the
        frequencies alone when the context was never seen), optionally
        filtered to values above min_val_constraint (sorted games chain each
        slot on the previous one), then softmax-tempered. Returns
        (candidates, probabilities), or ([], []) when nothing is left to
        choose from so the caller can apply its own fallback. One
        implementation shared by predict_next_numbers (samples from it), the
        pair-scored joint (takes its top candidates) and score_positions
        (reads it out whole), so all three see the exact same distribution.
        """
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

    def _pair_scored_joint(self, relevant_history, temperature, top_k=4):
        """
        The pair-scored joint distribution over whole tickets (the Pick3
        path): each column's top_k candidates from _column_distribution,
        every cross-column combination scored by its summed log column
        probability plus pair_scoring_weight times the summed log pair
        affinity of its digits, softmaxed into probabilities. Returns
        (combinations, probabilities) in matching order. predict_next_numbers
        samples one ticket from it and score_positions marginalises it per
        slot - a single implementation so the distribution being scored is
        exactly the one the ticket is drawn from.
        """
        num_columns = len(relevant_history[0])

        col_candidates = []
        for col in range(num_columns):
            cands, p = self._column_distribution(relevant_history, col, temperature) # No constraint here, we score later?
            # Actually for Pick3 we don't constrain.
            zipped = sorted(zip(cands, p), key=lambda x: x[1], reverse=True)
            col_candidates.append(zipped[:top_k])

        candidate_nums = [[num for num, prob in col] for col in col_candidates]
        candidate_probs = [[prob for num, prob in col] for col in col_candidates]

        all_combinations = list(itertools.product(*candidate_nums))
        all_probs = list(itertools.product(*candidate_probs))

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

        return all_combinations, final_probs

    def predict_next_numbers(self, history_draws, temperature=0.7):
        if len(history_draws) < self.markov_order:
            # Fallback
            width = len(history_draws[0]) if history_draws and len(history_draws[0]) > 0 else 3
            return [np.random.randint(1, 10) for _ in range(width)]

        relevant_history = history_draws[-self.markov_order:]
        num_columns = len(relevant_history[0])

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
                
                cands, p = self._column_distribution(relevant_history, col, temperature, min_val_constraint=constraint)
                
                if not cands:
                    # Hard Fallback: Last val + 1 (or 1 if first)
                    remaining_slots = num_columns - col

                    if self.sorted_prediction:
                        min_allowed = last_pred_val + 1 if last_pred_val >= self.min_number else self.min_number
                        max_allowed = self.max_number - remaining_slots + 1

                        if min_allowed <= max_allowed:
                            pred = int(np.random.randint(min_allowed, max_allowed + 1))
                        else:
                            pred = min(self.max_number, last_pred_val + 1)
                    else:
                        pred = int(np.random.randint(self.min_number, self.max_number + 1))
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
            all_combinations, final_probs = self._pair_scored_joint(relevant_history, temperature)

            idx = np.random.choice(len(all_combinations), p=final_probs)
            prediction = list(all_combinations[idx])

        return prediction

    def generate_best_subset(self, predicted_numbers, nSubset):
        unique_numbers = list(dict.fromkeys(map(int, predicted_numbers)))
        
        # Rank the predicted numbers by their global historical frequency (highest first)
        ranked_prediction = sorted(unique_numbers, key=lambda x: self.global_frequencies.get(x, 0), reverse=True)
        
        if len(ranked_prediction) < nSubset:
            # Fallback to global frequent numbers
            sorted_freq = sorted(self.global_frequencies, key=self.global_frequencies.get, reverse=True)
            for f in sorted_freq:
                if f not in ranked_prediction:
                    ranked_prediction.append(f)
                if len(ranked_prediction) >= nSubset: 
                    break
                
            # Random fallback if still empty
            while len(ranked_prediction) < nSubset:
                r = np.random.randint(1, 81) # Assuming Keno max is 80
                if r not in ranked_prediction:
                    ranked_prediction.append(r)
        
        # Slice the top N most historically frequent numbers from our prediction
        best_subset = ranked_prediction[:nSubset]
        
        # Return them sorted numerically (standard for lottery tickets)
        return sorted(best_subset)
    
    def generate_candidate_tickets(self, history_draws, n_tickets=1000, temperature=None):
        if temperature is None:
            temperature = self.softMaxTemperature

        tickets = []

        for _ in range(n_tickets):
            ticket = self.predict_next_numbers(
                history_draws,
                temperature=temperature
            )

            if self.sorted_prediction:
                ticket = sorted(dict.fromkeys(map(int, ticket)))
            else:
                ticket = list(map(int, ticket))

            tickets.append(tuple(ticket))

        return tickets

    def rank_candidate_tickets(self, history_draws, n_tickets=5000, top_n=10, temperature=None):

        tickets = self.generate_candidate_tickets(
            history_draws,
            n_tickets=n_tickets,
            temperature=temperature
        )

        ranked = Counter(tickets).most_common(top_n)

        return [
            {
                "ticket": list(ticket),
                "count": count
            }
            for ticket, count in ranked
        ]
    def generate_voted_ticket(self, history_draws, n_tickets=10000, ticket_size=None, temperature=None):

        if temperature is None:
            temperature = self.softMaxTemperature

        if ticket_size is None:
            ticket_size = self.draw_size

        votes = defaultdict(float)

        tickets = self.generate_candidate_tickets(
            history_draws,
            n_tickets=n_tickets,
            temperature=temperature
        )

        for ticket in tickets:
            unique_ticket = set(ticket)
            for n in unique_ticket:
                votes[int(n)] += 1

        ranked_numbers = sorted(
            votes,
            key=votes.get,
            reverse=True
        )

        final_ticket = ranked_numbers[:ticket_size]

        return sorted(final_ticket), dict(votes)

    def run(self, generateSubsets=[], skipRows=0, skipLastColumns=0, specialColumnCount=0):
        numbers, _, _ = self.load_numbers(
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            specialColumnCount=specialColumnCount
        )

        if len(numbers) == 0:
            return [], {}

        self.build_markov_chain(numbers)

        history_context = numbers[-self.markov_order:]
        predicted_numbers = self.predict_next_numbers(
            history_context,
            temperature=self.softMaxTemperature
        )

        subsets = {}
        for subset_size in generateSubsets:
            subsets[subset_size] = self.generate_best_subset(predicted_numbers, subset_size)

        return predicted_numbers, subsets

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0, n_tickets=2000):
        """
        Per-number score for stacking (Phase 1): builds the same Monte Carlo
        voted-ticket distribution run() draws its single ticket from, but
        returns the full {number: votes} dict instead of collapsing it to one
        ticket - reuses generate_voted_ticket, no new prediction logic.
        """
        numbers, _, _ = self.load_numbers(
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            specialColumnCount=specialColumnCount
        )

        if len(numbers) == 0:
            return {}

        self.build_markov_chain(numbers)
        history_context = numbers[-self.markov_order:]

        _, votes = self.generate_voted_ticket(
            history_context,
            n_tickets=n_tickets,
            temperature=self.softMaxTemperature
        )

        return votes

    def score_positions(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Per-position digit scores for the positional (Pick3) meta-learner: one
        {digit: probability} dict per drawn position, in drawn order, summing
        to 1 per slot. Same load and chain build as run(), then the very
        distribution run()'s single ticket is sampled from, read out whole
        instead of collapsed to one draw:
          - pair-scoring path (how Pick3 is configured): the pair-scored joint
            over each slot's top candidates (_pair_scored_joint), marginalised
            per slot - digit d in slot p scores the summed probability of every
            combination carrying d in slot p. Digits outside a slot's top
            candidates get 0.0.
          - otherwise (pair scoring off, or too many columns for the joint -
            the same >6 cut-off predict_next_numbers applies, just without
            repeating its warning): each slot's blended, tempered distribution
            (_column_distribution) as-is. No sorted-prediction constraint:
            that chains on the previous slot's *sampled* value, which has no
            meaning when every slot is scored at once.
        Every digit of the game's label range is present so the consumer can
        build fixed-width feature vectors without guarding keys; a slot with
        nothing to choose from (chain too short) scores uniform rather than an
        all-zero row the consumer could not normalise - it is also what
        predict_next_numbers' own random fallback amounts to there. Empty
        history returns [] - same convention as score_numbers' {}.
        """
        numbers, _, unique_labels = self.load_numbers(
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            specialColumnCount=specialColumnCount
        )

        if len(numbers) == 0:
            return []

        self.build_markov_chain(numbers)
        history_context = numbers[-self.markov_order:]
        temperature = self.softMaxTemperature

        digits = [int(label) for label in unique_labels]
        num_columns = len(numbers[-1])
        uniform = {digit: 1.0 / len(digits) for digit in digits}

        if len(history_context) < self.markov_order:
            return [dict(uniform) for _ in range(num_columns)]

        column_dists = [
            self._column_distribution(history_context, col, temperature)
            for col in range(num_columns)
        ]

        # The joint needs every slot to have candidates (itertools.product
        # over an empty slot is empty); a slot can come up empty when the
        # chain is too short to have been built at all.
        use_joint = (
            self.use_pair_scoring
            and num_columns <= 6
            and all(len(cands) > 0 for cands, _ in column_dists)
        )

        position_scores = [{digit: 0.0 for digit in digits} for _ in range(num_columns)]

        if use_joint:
            combinations, probabilities = self._pair_scored_joint(history_context, temperature)
            for combo, prob in zip(combinations, probabilities):
                for pos, digit in enumerate(combo):
                    digit = int(digit)
                    if digit in position_scores[pos]:
                        position_scores[pos][digit] += float(prob)
        else:
            for pos, (cands, probs) in enumerate(column_dists):
                if not cands:
                    position_scores[pos] = dict(uniform)
                    continue
                for digit, prob in zip(cands, probs):
                    digit = int(digit)
                    if digit in position_scores[pos]:
                        position_scores[pos][digit] += float(prob)

        return position_scores

if __name__ == "__main__":
    print("Trying Markov")

    markov = Markov()
    name = 'lotto' 
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

    sequenceToPredict = None
    try:
        with open(sequenceToPredictFile, 'r') as openfile:
            sequenceToPredict = json.load(openfile)
        print("Real result: ", sequenceToPredict["realResult"])
    except:
        pass

    skipLastColumn = 0
    if "keno" in name.lower():
        markov.setGameRange(1, 80)
        markov.setDrawSize(20)
        generateSubsets = [6, 7]

    elif "lotto" in name.lower():
        markov.setGameRange(1, 45)
        markov.setDrawSize(6)
        skipLastColumn = 1

    elif "vikinglotto" in name.lower():
        markov.setGameRange(1, 48)
        markov.setDrawSize(6)

    elif "euro" in name.lower():
        markov.setGameRange(1, 50)
        markov.setDrawSize(5)

    elif "pick3" in name.lower():
        markov.setGameRange(0, 9)
        markov.setDrawSize(3)

    #####################
    # Single prediction #
    #####################
    predicted_numbers, subsets = markov.run(
        generateSubsets=generateSubsets,
        skipLastColumns=skipLastColumn
    )

    print("Predicted Numbers: ", predicted_numbers)
    if subsets:
        print("Subsets: ", subsets)

    if sequenceToPredict is not None:
        matches = set(predicted_numbers) & set(sequenceToPredict["realResult"])
        print("Real result: ", sequenceToPredict["realResult"])
        print("Numbers that matches: ", matches)

    ########################
    # Generate top tickets #
    ########################
    numbers, _, _ = markov.load_numbers(skipLastColumns=skipLastColumn)
    markov.build_markov_chain(numbers)

    history = numbers[-markov.markov_order:]

    ranked = markov.rank_candidate_tickets(
        history,
        n_tickets=10000,
        top_n=10
    )

    print("\nTop generated tickets:")
    print(json.dumps(ranked, indent=4))

    #####################
    # Top voted numbers #
    #####################

    voted_ticket, votes = markov.generate_voted_ticket(
        history,
        n_tickets=10000,
        ticket_size=markov.draw_size
    )

    print("Voted ticket:", voted_ticket)
