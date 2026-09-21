# Chronos Model - Amazon's pretrained time-series foundation model as a
# per-position predictor (README roadmap item 5). All of the machinery lives
# in src/FoundationModel.py, which TimesFM-3 shares; this file is the model's
# identity and the honest reading of its row.
#
# Every drawn position is treated as its own univariate series (position 1 of
# lotto over time, position 2, ...) and amazon/chronos-2 is asked, zero-shot,
# for the next value's distribution; that distribution over the game's labels
# is the per-position score shape every positional consumer here already
# understands, so this plugs in as one more tracked row and as a feature for
# the positional meta-learner (src/ModelFactory.py).
#
# HOW TO READ THIS ROW. For the sorted set games the per-position series are
# order statistics - position 1 is the minimum of the draw, so its
# distribution is genuinely predictable without the draw being predictable at
# all. The row must therefore be compared against the OrderStatistics
# Baseline row, never against chance. For the positional games (pick3,
# Joker+) an honest process gives near-uniform forecasts, which makes this
# row a useful negative control: measured on 2026-09-21 its pick3 digit
# masses ran 0.083-0.128 against 0.100 for uniform, peaking on whichever
# digit was over-represented in the last hundred draws - i.e. it behaves as a
# recency-weighted frequency estimator there, and the repeated-digit tickets
# that follow ([9, 9, 9], [8, 8, 8]) are that result, not a fault. A
# deviation that is both large and stable across weeks would be the finding.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from FoundationModel import FoundationModel


class ChronosModel(FoundationModel):
    WORKER_SCRIPT = "chronos_forecast.py"
    NAME = "Chronos Model"
    # The worker's own default is amazon/chronos-2; CHRONOS_MODEL overrides it
    # for an experiment without touching the code.
    PASSTHROUGH_ENV = ("CHRONOS_MODEL", "CHRONOS_DEVICE", "CHRONOS_THREADS")
