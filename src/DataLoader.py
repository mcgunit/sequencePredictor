import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Helpers import Helpers

helpers = Helpers()


class DataLoader:
    """
    Minimal data source for Backtester: exposes just the game-range bookkeeping
    (min_number/max_number) and load_numbers(...) that Backtester needs to load
    ground-truth history and build its baselines, without requiring a full
    Markov (or other) model instance that isn't actually being backtested.
    """

    def __init__(self):
        self.dataPath = ""
        self.min_number = 1
        self.max_number = 80
        self.draw_size = None

    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    def setGameRange(self, min_number, max_number):
        self.min_number = int(min_number)
        self.max_number = int(max_number)

    def setDrawSize(self, draw_size):
        self.draw_size = int(draw_size)

    def load_numbers(self, skipRows=0, skipLastColumns=0, years_back=None, specialColumnCount=0):
        _, _, _, _, _, numbers, num_classes, unique_labels = helpers.load_data(
            self.dataPath,
            skipRows=skipRows,
            skipLastColumns=skipLastColumns,
            years_back=years_back,
            specialColumnCount=specialColumnCount
        )
        return numbers, num_classes, unique_labels
