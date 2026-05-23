import os
from collections import Counter, defaultdict
import numpy as np


class MarkovMonteCarlo:
    def __init__(self, markov_model):
        self.model = markov_model
        self.num_simulations = 1000

    def setNumOfSimulations(self, n):
        self.num_simulations = int(n)

    def generate_candidate_tickets(self, history_draws, n_tickets=None, temperature=None):
        if n_tickets is None:
            n_tickets = self.num_simulations

        if temperature is None:
            temperature = self.model.softMaxTemperature

        tickets = []

        for _ in range(n_tickets):
            ticket = self.model.predict_next_numbers(
                history_draws,
                temperature=temperature
            )

            if self.model.sorted_prediction:
                ticket = sorted(dict.fromkeys(map(int, ticket)))
            else:
                ticket = list(map(int, ticket))

            tickets.append(tuple(ticket))

        return tickets

    def rank_candidate_tickets(self, history_draws, n_tickets=None, top_n=10, temperature=None):
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

    def generate_voted_ticket(self, history_draws, n_tickets=None, ticket_size=None, temperature=None):
        if n_tickets is None:
            n_tickets = self.num_simulations

        if temperature is None:
            temperature = self.model.softMaxTemperature

        if ticket_size is None:
            ticket_size = self.model.draw_size or len(history_draws[0])

        votes = defaultdict(float)

        tickets = self.generate_candidate_tickets(
            history_draws,
            n_tickets=n_tickets,
            temperature=temperature
        )

        for ticket in tickets:
            for number in set(ticket):
                votes[int(number)] += 1

        ranked_numbers = sorted(
            votes,
            key=votes.get,
            reverse=True
        )

        final_ticket = ranked_numbers[:ticket_size]

        return sorted(final_ticket), dict(votes)

    def run(self, generateSubsets=None, skipRows=0, skipLastColumns=0):
        if generateSubsets is None:
            generateSubsets = []

        numbers, _, _ = self.model.load_numbers(
            skipRows=skipRows,
            skipLastColumns=skipLastColumns
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