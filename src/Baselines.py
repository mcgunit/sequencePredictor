import numpy as np
from collections import defaultdict


class Baselines:
    @staticmethod
    def random_ticket(min_number, max_number, draw_size):
        return sorted(
            np.random.choice(
                np.arange(min_number, max_number + 1),
                size=draw_size,
                replace=False
            ).astype(int).tolist()
        )

    @staticmethod
    def global_frequency_ticket(train_numbers, draw_size):
        freq = defaultdict(float)

        for draw in train_numbers:
            for n in draw:
                freq[int(n)] += 1

        ranked = sorted(freq, key=freq.get, reverse=True)
        return sorted(ranked[:draw_size])

    @staticmethod
    def column_frequency_ticket(train_numbers):
        num_columns = train_numbers.shape[1]
        ticket = []

        for col in range(num_columns):
            freq = defaultdict(float)

            for draw in train_numbers:
                freq[int(draw[col])] += 1

            best = max(freq, key=freq.get)
            ticket.append(best)

        # Remove duplicates while preserving order
        ticket = list(dict.fromkeys(ticket))

        # Fill missing values using global frequency
        if len(ticket) < num_columns:
            global_freq = defaultdict(float)

            for draw in train_numbers:
                for n in draw:
                    global_freq[int(n)] += 1

            ranked_global = sorted(
                global_freq,
                key=global_freq.get,
                reverse=True
            )

            for n in ranked_global:
                if n not in ticket:
                    ticket.append(n)

                if len(ticket) >= num_columns:
                    break

        return sorted(ticket[:num_columns])