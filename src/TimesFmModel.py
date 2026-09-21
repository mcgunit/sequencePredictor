# TimesFM Model - Google's pretrained time-series foundation model as a
# per-position predictor, the second entry of README roadmap item 5 and a
# control on the first: Chronos-2 and TimesFM-3 are different models from
# different labs trained on different corpora, so what they agree on is a
# property of the data and what only one of them sees is a property of that
# model. Run through the identical protocol (src/FoundationModel.py, same
# worker contract, same label mapping), which is the point - a second
# foundation model is a second reading, not a second pipeline.
#
# HOW TO READ THIS ROW. Exactly like the Chronos row (see src/ChronosModel.py):
# against the OrderStatistics Baseline for the sorted set games, and as a
# negative control on pick3/Joker+ where an honest process must look
# near-uniform. One difference is worth knowing when comparing the two: the
# checkpoint publishes a coarser quantile grid than Chronos-2 (deciles rather
# than 21 levels), so its label distribution is read off fewer knots. That
# makes its fine structure less trustworthy than Chronos's, never more.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from FoundationModel import FoundationModel


class TimesFmModel(FoundationModel):
    WORKER_SCRIPT = "timesfm_forecast.py"
    NAME = "TimesFM Model"
    PASSTHROUGH_ENV = ("TIMESFM_MODEL", "TIMESFM_DEVICE", "TIMESFM_THREADS")
