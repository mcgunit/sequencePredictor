import os


class MarkovMonteCarlo:
    """
    Thin voting wrapper around a Markov model: instead of Markov.run()'s single
    sample per column, repeatedly samples predict_next_numbers() (via the
    Markov model's own generate_candidate_tickets/rank_candidate_tickets/
    generate_voted_ticket) and keeps the numbers that win the most votes across
    simulations. The ticket-generation/voting logic itself lives on Markov,
    not here, to avoid maintaining two copies of the same code.
    """

    def __init__(self, markov_model):
        self.model = markov_model
        self.num_simulations = 1000

    def setNumOfSimulations(self, n):
        self.num_simulations = int(n)

    def generate_candidate_tickets(self, history_draws, n_tickets=None, temperature=None):
        return self.model.generate_candidate_tickets(
            history_draws,
            n_tickets=n_tickets if n_tickets is not None else self.num_simulations,
            temperature=temperature if temperature is not None else self.model.softMaxTemperature
        )

    def rank_candidate_tickets(self, history_draws, n_tickets=None, top_n=10, temperature=None):
        return self.model.rank_candidate_tickets(
            history_draws,
            n_tickets=n_tickets if n_tickets is not None else self.num_simulations,
            top_n=top_n,
            temperature=temperature if temperature is not None else self.model.softMaxTemperature
        )

    def generate_voted_ticket(self, history_draws, n_tickets=None, ticket_size=None, temperature=None):
        return self.model.generate_voted_ticket(
            history_draws,
            n_tickets=n_tickets if n_tickets is not None else self.num_simulations,
            ticket_size=ticket_size if ticket_size is not None else (self.model.draw_size or len(history_draws[0])),
            temperature=temperature if temperature is not None else self.model.softMaxTemperature
        )

    def run(self, generateSubsets=None, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        if generateSubsets is None:
            generateSubsets = []

        numbers, _, _ = self.model.load_numbers(
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            specialColumnCount=specialColumnCount
        )

        if len(numbers) == 0:
            return [], {}

        self.model.build_markov_chain(numbers)

        history = numbers[-self.model.markov_order:]

        predicted_numbers, votes = self.generate_voted_ticket(
            history,
            n_tickets=self.num_simulations,
            ticket_size=len(numbers[-1]),
            temperature=self.model.softMaxTemperature
        )

        subsets = {}

        for subset_size in generateSubsets:
            subsets[subset_size] = self.model.generate_best_subset(
                predicted_numbers,
                subset_size
            )

        return predicted_numbers, subsets

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Per-number score for stacking (Phase 1): same voting call run() makes,
        returning the full {number: votes} dict instead of the winning ticket.
        """
        numbers, _, _ = self.model.load_numbers(
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            specialColumnCount=specialColumnCount
        )

        if len(numbers) == 0:
            return {}

        self.model.build_markov_chain(numbers)
        history = numbers[-self.model.markov_order:]

        _, votes = self.generate_voted_ticket(
            history,
            n_tickets=self.num_simulations,
            ticket_size=len(numbers[-1]),
            temperature=self.model.softMaxTemperature
        )

        return votes

    def score_positions(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Per-position digit scores for the positional (Pick3) meta-learner: one
        {digit: votes} dict per drawn position, in drawn order. Same chain
        build and the same num_simulations sampled tickets run()'s voting
        draws, tallied per slot ("how often was digit d sampled in position
        p") instead of pooled over the whole ticket - the pooled tally is
        exactly what discards the slot identity a positional game pays out
        on. Only meaningful with sorted_prediction=False (how Pick3 is
        configured): the sorted path dedupes and re-orders each sampled
        ticket, so a shortened ticket is tallied only for the slots it still
        has. Every digit of the game's label range is present (never sampled
        -> 0.0) so the consumer can build fixed-width feature vectors without
        guarding keys; a sampled value outside that range (Markov's random
        fallbacks use its own min/max_number, which ModelFactory never sets
        for Pick3) is dropped rather than invent an extra key. Empty history
        returns [] - same convention as score_numbers' {}.
        """
        numbers, _, unique_labels = self.model.load_numbers(
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            specialColumnCount=specialColumnCount
        )

        if len(numbers) == 0:
            return []

        self.model.build_markov_chain(numbers)
        history = numbers[-self.model.markov_order:]

        tickets = self.generate_candidate_tickets(
            history,
            n_tickets=self.num_simulations,
            temperature=self.model.softMaxTemperature
        )

        digits = [int(label) for label in unique_labels]
        num_positions = len(numbers[-1])
        position_scores = [{digit: 0.0 for digit in digits} for _ in range(num_positions)]

        for ticket in tickets:
            for pos, digit in enumerate(ticket[:num_positions]):
                digit = int(digit)
                if digit in position_scores[pos]:
                    position_scores[pos][digit] += 1.0

        return position_scores

if __name__ == "__main__":
    from Markov import Markov

    name = 'lotto' 
    generateSubsets = []

    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    markov = Markov()
    markov.setDataPath(dataPath)
    markov.setGameRange(1, 45)
    markov.setDrawSize(6)
    markov.setSortedPrediction(True)
    markov.setMarkovOrder(2)

    markov_mc = MarkovMonteCarlo(markov)
    markov_mc.setNumOfSimulations(1000)

    print(markov_mc.run(skipLastColumns=1))