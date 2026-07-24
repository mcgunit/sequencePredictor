import os
import sys
from collections import Counter

# Dynamically adjust the import path for Helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Markov import Markov
from MarkovBayesian import MarkovBayesian
from PoissonMonteCarlo import PoissonMonteCarlo
from LaplaceMonteCarlo import LaplaceMonteCarlo
from PoissonMarkov import PoissonMarkov


class HybridStatisticalModel():
    """
    Meta-ensemble: runs Markov, MarkovBayesian, PoissonMonteCarlo,
    LaplaceMonteCarlo, and PoissonMarkov independently each round, then takes
    the most-voted number(s) across all of them.

    "Position" only has a genuine, fixed meaning for positional games (Pick3 -
    hundreds/tens/units digit). For non-positional games (Keno, Lotto,
    Euromillions, ...) numbers are an unordered set, so voting there is a
    plain frequency tally across every voter's predicted numbers rather than
    a per-index vote. This is driven by the same sorted_prediction flag used
    throughout the rest of the codebase: True (default, non-positional) means
    a frequency vote; False (Pick3) means a per-position vote.
    """

    def __init__(self):
        self.dataPath = ""
        self.sorted_prediction = True

        self.markov = Markov()
        self.markov_bayesian = MarkovBayesian()
        self.poisson_mc = PoissonMonteCarlo()
        self.laplace_mc = LaplaceMonteCarlo()
        self.poisson_markov = PoissonMarkov()

        self.voters = [
            self.markov,
            self.markov_bayesian,
            self.poisson_mc,
            self.laplace_mc,
            self.poisson_markov,
        ]

    def setDataPath(self, dataPath):
        self.dataPath = dataPath
        for voter in self.voters:
            voter.setDataPath(dataPath)

    def setSortedPrediction(self, use):
        """
        Disable for positional games (Pick3) so voting happens per-position
        instead of as a plain frequency tally - see class docstring.
        """
        self.sorted_prediction = bool(use)
        self.markov.setSortedPrediction(use)
        self.markov_bayesian.setSortedPrediction(use)
        self.poisson_mc.setSortedPrediction(use)
        self.laplace_mc.setSortedPrediction(use)
        self.poisson_markov.setSortedPrediction(use)

    # Shared hyperparameters applied across whichever voters actually expose
    # them - kept as the same setter names/signatures HyperoptStatistics.py's
    # objective_hybrid already tunes, so that file needs no changes.
    def setSoftMaxTemperature(self, value):
        self.markov.setSoftMAxTemperature(value)
        self.markov_bayesian.setSoftMAxTemperature(value)

    def setAlpha(self, value):
        self.markov.setAlpha(value)
        self.markov_bayesian.setAlpha(value)

    def setMinOccurrences(self, value):
        self.markov.setMinOccurrences(value)
        self.markov_bayesian.setMinOccurrences(value)

    def setNumberOfSimulations(self, value):
        self.poisson_mc.setNumOfSimulations(value)
        self.laplace_mc.setNumOfSimulations(value)
        self.poisson_markov.setNumberOfSimulations(value)

    def clear(self):
        # Each voter resets its own state internally at the start of its own
        # run() - nothing to clear here.
        pass

    def _vote_positional(self, voter_predictions, n_predictions):
        """One vote per voter per position; the majority digit wins each slot."""
        result = []
        for pos in range(n_predictions):
            votes = Counter(
                int(prediction[pos])
                for prediction in voter_predictions
                if pos < len(prediction)
            )
            if votes:
                result.append(votes.most_common(1)[0][0])
        return result

    def _vote_frequency(self, voter_predictions):
        """Tally of how often each number appears across all voters' predictions."""
        votes = Counter()
        for prediction in voter_predictions:
            for num in prediction:
                votes[int(num)] += 1
        return votes

    def generate_best_subset(self, vote_counts, predicted_numbers, nSubset):
        """Top-N most-voted numbers from this round's tally."""
        ranked = sorted(predicted_numbers, key=lambda n: (-vote_counts.get(n, 0), n))
        return sorted(ranked[:nSubset])

    def run(self, generateSubsets=[], skipRows=0, skipLastColumns=0, specialColumnCount=0):
        voter_predictions = []

        for voter in self.voters:
            try:
                prediction, _ = voter.run(
                    generateSubsets=[],
                    skipRows=skipRows,
                    skipLastColumns=skipLastColumns,
                    specialColumnCount=specialColumnCount
                )
                if prediction:
                    voter_predictions.append([int(num) for num in prediction])
            except Exception:
                # A single voter failing this round shouldn't sink the whole
                # ensemble - vote with whichever voters succeeded.
                continue

        if not voter_predictions:
            return [], {}

        n_predictions = len(voter_predictions[0])

        if self.sorted_prediction:
            vote_counts = self._vote_frequency(voter_predictions)
            ranked = sorted(vote_counts, key=lambda n: (-vote_counts[n], n))
            final_predictions = sorted(ranked[:n_predictions])
        else:
            final_predictions = self._vote_positional(voter_predictions, n_predictions)
            vote_counts = self._vote_frequency(voter_predictions)

        subsets = {}
        if generateSubsets:
            for subset_size in generateSubsets:
                subsets[subset_size] = self.generate_best_subset(vote_counts, final_predictions, subset_size)

        return final_predictions, subsets


if __name__ == "__main__":
    print("Trying Hybrid Statistical Model (meta-ensemble)")

    hybridStatisticalModel = HybridStatisticalModel()
    name = 'pick3'
    generateSubsets = []
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)

    hybridStatisticalModel.setDataPath(dataPath)
    hybridStatisticalModel.setSoftMaxTemperature(0.1)
    hybridStatisticalModel.setAlpha(0.5)
    hybridStatisticalModel.setMinOccurrences(5)
    hybridStatisticalModel.setNumberOfSimulations(500)

    if "pick3" in name:
        hybridStatisticalModel.setSortedPrediction(False)

    if "keno" in name:
        generateSubsets = [6, 7]

    print("Predicted Numbers: ", hybridStatisticalModel.run(generateSubsets=generateSubsets))
