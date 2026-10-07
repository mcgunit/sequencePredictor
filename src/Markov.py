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
        # Nothing in the pipeline ever calls setGameRange, so these defaults
        # used to drive every random-fallback digit (e.g. an unseen Markov
        # context) for EVERY game - a lotto or pick3 fallback could emit 57.
        # build_markov_chain now derives the range from the data it is built
        # on unless a caller set it explicitly.
        self._game_range_explicit = False
        self.draw_size = None
        self.random_seed = None
        # --- CONFIGURATION FLAGS ---
        self.markov_order = 1
        self.use_pair_scoring = False
        self.pair_scoring_weight = 1.0
        self.sorted_prediction = False # NEW: Replaces Deltas. Enforces X > Prev_X.
        # Which transition the chain learns (markovTransitionMode, 5 Oct 2026):
        #   "column": per column, the number in that column in the previous
        #             draw(s) -> the number in the same column in the next
        #             draw (the definition since 3 Feb 2026);
        #   "within": from each number to the next number inside the same
        #             sorted draw, one table for the whole game (the original
        #             Markov row of 13 Feb 2025 - the gap structure of a
        #             ticket, not time).
        # Both tables are always built; the mode picks the prediction path.
        # Set games only: a positional game (pair scoring on) always takes
        # the column path and says so once.
        self.transition_mode = "column"
        self.within_matrix = {}
        self._within_note_printed = False
        # {number: mass} of the last run() - the ranking its keno subsets are cut by
        self._last_number_scores = None
        
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
        self.within_matrix = {}
        self._last_number_scores = None
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
    def setSubsetSelectionMode(self, m): self.subset_selection_mode = m   # accepted, not read since 7 Oct 2026 (see generate_best_subset)
    def setBlendMode(self, m): self.blend_mode = m
    def setMarkovOrder(self, order): self.markov_order = max(1, int(order))
    def setTransitionMode(self, mode):
        mode = str(mode)
        if mode not in ("column", "within"):
            raise ValueError(f"markovTransitionMode must be 'column' or 'within', not {mode!r}")
        self.transition_mode = mode
    def setUsePairScoring(self, use): self.use_pair_scoring = bool(use)
    def setPairScoringWeight(self, w): self.pair_scoring_weight = float(w)
    def setGameRange(self, min_number, max_number):
        self.min_number = int(min_number)
        self.max_number = int(max_number)
        self._game_range_explicit = True

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

        # Range of the columns actually loaded for this call: pick3 digits
        # 0-9, lotto mains 1-45, euromillions stars 1-12 on the special-only
        # call, ... - so the random fallbacks in predict_next_numbers /
        # generate_candidate_tickets stay inside the game's real range.
        if not self._game_range_explicit and len(numbers) > 0:
            arr = np.asarray(numbers)
            if arr.size > 0:
                self.min_number = int(arr.min())
                self.max_number = int(arr.max())

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

        # 4. Within-draw transitions (the 2025 definition): number -> the next
        #    number of the same draw, every draw, the same recency weight.
        within_raw = defaultdict(lambda: defaultdict(float))
        for t, draw in enumerate(numbers):
            if self.recency_mode == "linear":
                weight = 1 + (self.recency_weight * t / total_draws)
            elif self.recency_mode == "log":
                weight = 1 + np.log1p(t) * self.recency_weight
            else:
                weight = 1.0
            for i in range(len(draw) - 1):
                within_raw[int(draw[i])][int(draw[i + 1])] += weight
        self.within_matrix = self._normalize_table(within_raw)

        self._normalize_matrices()

    def _normalize_table(self, raw):
        """Prune transitions seen fewer than min_occurrences times, smooth, normalise - one table."""
        cleaned = {}
        for ctx, transitions in raw.items():
            filtered = {k: v for k, v in transitions.items() if v >= self.min_occurrences}
            if not filtered: continue
            total = sum(filtered.values()) + self.smoothing_factor * len(filtered)
            cleaned[ctx] = {
                int(k): (v + self.smoothing_factor) / total
                for k, v in filtered.items()
            }
        return cleaned

    def _normalize_matrices(self):
        for col_idx in range(len(self.transition_matrices)):
            self.transition_matrices[col_idx] = self._normalize_table(self.transition_matrices[col_idx])
            
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

        if self.transition_mode == "within":
            if self.use_pair_scoring:
                if not self._within_note_printed:
                    print("Markov: transition mode 'within' is for the sorted set games - a positional game takes the column path")
                    self._within_note_printed = True
            else:
                return self._within_prediction(relevant_history[-1], temperature)

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

    def number_scores_from_chain(self, history_draws, temperature):
        """
        {number: mass} the chain puts on every number for the next draw, in
        the mode in use - the ranking the keno subsets are cut by. Column
        mode sums each column's blended distribution (_column_distribution,
        no sorted constraint): over sorted positions that sum is the chain's
        probability that the number is drawn at all. Within mode sums the
        successor tables of the last draw's numbers and adds the blended
        frequency fill, as _within_prediction does. Untempered on purpose:
        the temperature shapes the sampled ticket, and a sharpened first
        column would put all its mass on 1 and rank 1 and 70 into every
        subset; `temperature` is accepted for the call sites and ignored.
        """
        if history_draws is None or len(history_draws) == 0 or len(history_draws[0]) == 0:   # a numpy slice, not a list
            return {}
        temperature = 1.0
        scores = defaultdict(float)
        if self.transition_mode == "within" and not self.use_pair_scoring:
            last = [int(v) for v in history_draws[-1]]
            for num in last:
                successors = self.within_matrix.get(num)
                if not successors:
                    continue
                probs = np.asarray(self.softmax_with_temperature(list(successors.values()), temperature), dtype=float)
                for cand, p in zip(successors.keys(), probs):
                    scores[int(cand)] += float(p)
            blended = self.blended_probability(self.within_matrix.get(last[-1], {}), self.global_frequencies)
            total = sum(blended.values()) or 1.0
            for cand, value in blended.items():
                scores[int(cand)] += float(value) / total
            return dict(scores)
        relevant = history_draws[-self.markov_order:]
        for col in range(len(relevant[0])):
            cands, probs = self._column_distribution(relevant, col, temperature)
            for cand, p in zip(cands, probs):
                scores[int(cand)] += float(p)
        return dict(scores)

    def _within_prediction(self, last_draw, temperature):
        """
        The 2025 Markov row's ticket: for every number of the last draw, a
        successor sampled (tempered) from that number's within-draw table;
        short tickets are filled, as then, from the last number's successors
        blended with the game's frequencies, best first, then from pair
        affinity (both directions of the stored pair table), then from the
        range. A sorted set of the draw's size.
        """
        n = len(last_draw)
        picks = []
        for num in map(int, last_draw):
            successors = self.within_matrix.get(num)
            if not successors:
                continue
            cands = list(successors.keys())
            probs = np.asarray(self.softmax_with_temperature(list(successors.values()), temperature), dtype=float)
            probs = probs / probs.sum()
            pick = int(np.random.choice(cands, p=probs))
            if pick not in picks:
                picks.append(pick)
        if len(picks) < n:
            blended = self.blended_probability(self.within_matrix.get(int(last_draw[-1]), {}), self.global_frequencies)
            for cand in sorted(blended, key=blended.get, reverse=True):
                if len(picks) >= n:
                    break
                cand = int(cand)
                if cand not in picks:
                    picks.append(cand)
        while len(picks) < n:
            partner = None
            if picks:
                # pair_counts keeps each pair once, under its smaller number;
                # the 2025 table was symmetric, so look both ways
                last = picks[-1]
                affinity = dict(self.pair_counts.get(last) or {})
                for lower, partners in self.pair_counts.items():
                    if last in partners:
                        affinity[lower] = affinity.get(lower, 0) + partners[last]
                if affinity:
                    partner = int(max(affinity, key=affinity.get))
            if partner is None or partner in picks:
                pool = [v for v in range(self.min_number, self.max_number + 1) if v not in picks]
                if not pool:
                    break
                partner = int(np.random.choice(pool))
            picks.append(partner)
        return sorted(picks[:n])

    def generate_best_subset(self, predicted_numbers, nSubset):
        """
        The keno subset: the ticket's numbers ranked by the chain's own mass
        on them (number_scores_from_chain, set by run()), the top nSubset.
        Until 7 Oct 2026 the ranking was the global frequency; the frequency
        stays as the fallback when no masses exist (a caller that built the
        chain without run()). The markovSubsetSelectionMode knob was tuned
        for a year and a half without ever being read; honouring its stored
        "softmax" would have made the served keno subset a near-uniform
        sample of the ticket (the chain's masses span 0.27-0.30 on the real
        history), so the subset is the top k and the tuner no longer searches
        the knob - the setter stays for the files that carry the key.
        """
        unique_numbers = list(dict.fromkeys(map(int, predicted_numbers)))
        scores = self._last_number_scores or {}
        if scores:
            ranked_prediction = sorted(unique_numbers, key=lambda n: (-scores.get(n, 0.0), n))
        else:
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
                r = np.random.randint(self.min_number, self.max_number + 1)   # the game's range (a 1-80 literal until 5 Oct 2026)
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
        self._last_number_scores = self.number_scores_from_chain(history_context, self.softMaxTemperature)
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
        build fixed-width feature vectors without guarding keys (the column
        view in either transition mode - this is the positional games' call,
        and they have only that path); a slot with
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
    # Self-check (python3 -m src.Markov, in npm test): the two transition
    # definitions on synthetic draws, no files touched.
    rng = np.random.default_rng(3)
    # 400 sorted draws of 5 from 1-30 that all contain the pair (7, 8): the
    # within-draw chain must learn 7 -> 8, which no column chain can express.
    others = [v for v in range(1, 31) if v not in (7, 8)]
    draws = [sorted([int(v) for v in rng.choice(others, size=3, replace=False)] + [7, 8]) for _ in range(400)]

    w = Markov(); w.setGameRange(1, 30); w.setDrawSize(5); w.setSortedPrediction(True); w.setMinOccurrences(2); w.setRandomSeed(1)
    w.setTransitionMode("within"); w.build_markov_chain(draws)
    assert w.within_matrix[7] and max(w.within_matrix[7], key=w.within_matrix[7].get) == 8, w.within_matrix.get(7)
    assert abs(sum(w.within_matrix[7].values()) - 1.0) < 1e-9
    ticket = w.predict_next_numbers(draws[-1:], temperature=0.5)
    assert len(ticket) == 5 and ticket == sorted(set(ticket)) and all(1 <= v <= 30 for v in ticket), ticket
    with_eight = sum(1 for _ in range(50) if 8 in w.predict_next_numbers(draws[-1:], temperature=0.5))
    assert with_eight >= 45, with_eight          # 7 is in every draw, so 8 follows nearly every time
    voted, votes = w.generate_voted_ticket(draws[-1:], n_tickets=200, ticket_size=5)
    assert len(voted) == 5 and votes.get(8, 0) >= 180, (voted, votes.get(8))
    w._last_number_scores = w.number_scores_from_chain(draws[-1:], 0.5)
    assert w._last_number_scores.get(8, 0) > 0 and 8 in w.generate_best_subset(ticket if 8 in ticket else ticket[:4] + [8], 2), "the subset must follow the chain's masses"
    w.setSubsetSelectionMode("softmax")      # the retired knob changes nothing: the subset stays the top k
    assert w.generate_best_subset(ticket if 8 in ticket else ticket[:4] + [8], 2) == sorted(sorted(ticket if 8 in ticket else ticket[:4] + [8], key=lambda n: (-w._last_number_scores.get(n, 0.0), n))[:2])

    # hand count on a tiny history: constant weights, no smoothing, no pruning
    t = Markov(); t.setMinOccurrences(1); t.setSmoothingFactor(0.0); t.setRecencyMode("constant"); t.setTransitionMode("within")
    t.build_markov_chain([[1, 2, 5], [1, 3, 5], [2, 3, 5]])
    assert t.within_matrix[1] == {2: 0.5, 3: 0.5} and t.within_matrix[2] == {3: 0.5, 5: 0.5} and t.within_matrix[3] == {5: 1.0}, t.within_matrix

    # the column path is the default and untouched; the within table exists in both modes
    c = Markov(); c.setGameRange(1, 30); c.setDrawSize(5); c.setSortedPrediction(True); c.setMinOccurrences(2); c.setRandomSeed(1)
    c.build_markov_chain(draws)
    assert c.transition_mode == "column" and c.within_matrix[7] and len(c.transition_matrices) == 5
    col_ticket = c.predict_next_numbers(draws[-1:], temperature=0.5)
    assert len(col_ticket) == 5 and all(1 <= v <= 30 for v in col_ticket), col_ticket

    # a positional game (pair scoring on) takes the column path whatever the knob says, and says so once
    p = Markov(); p.setGameRange(0, 9); p.setDrawSize(3); p.setUsePairScoring(True); p.setMinOccurrences(1); p.setTransitionMode("within")
    p.build_markov_chain([[int(v) for v in rng.integers(0, 10, size=3)] for _ in range(300)])
    pt = p.predict_next_numbers([[1, 2, 3]], temperature=0.5)
    assert len(pt) == 3 and all(0 <= v <= 9 for v in pt) and p._within_note_printed, pt

    try:
        Markov().setTransitionMode("sideways")
        raise AssertionError("an unknown mode must be refused")
    except ValueError:
        pass
    print(f"Markov self-check OK: within-draw chain learns 7 -> 8 (ticket carries 8 in {with_eight} of 50 draws, "
          f"{votes.get(8)} of 200 votes), hand-counted table matches, column path unchanged, positional games keep the column path")
