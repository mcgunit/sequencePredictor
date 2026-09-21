"""
Turning a continuous quantile forecast into a distribution over the discrete
labels of a game - the one place where a naive implementation invents signal,
so both foundation-model workers (chronos_forecast.py, timesfm_forecast.py)
share this single implementation.

Binning each quantile sample to its nearest label piles mass at the ends of
the range. Instead the quantile curve is read as an empirical CDF and each
label gets the mass between its half-open bounds [label - 0.5, label + 0.5].

THE TAILS ARE THE TRAP, and how they are handled depends on how coarse the
model's grid is. A quantile curve says nothing below its lowest level or
above its highest, so the mass outside has to go somewhere. Dumping it on
the first and last label - the obvious choice - is nearly harmless for
Chronos-2, whose outermost levels are 0.01 and 0.99 (measured 2026-09-21:
0.005 of mass per side, no argmax changed), and catastrophic for TimesFM-3,
which publishes only the nine deciles: it put a flat 0.100 on the lowest
label of every series, which made label 1 the argmax of four lotto positions
purely as an artefact.

So the curve is anchored at the bounds of the game's own support instead:
the points (first label - 0.5, 0.0) and (last label + 0.5, 1.0) are added to
it, and the interpolation spreads each tail uniformly over the stretch
between the outermost quantile and that bound. Uniform is the maximum-
entropy choice given the only thing the model actually told us about the
tail - how much mass is in it - and it invents no peak.

Measured on 2026-09-21: a grid of 199 in-range levels gives the same
distribution to three decimals as Chronos-2's own 21 native levels, so the
fine grid is interpolation, not resolution. Whatever a model publishes is
what it is read at; nothing here manufactures precision.
"""

import numpy as np


def label_probabilities(quantile_values, quantile_levels, labels):
    """
    Probability per label from one position's quantile curve.

    quantile_values must be non-decreasing (they come from the models that
    way); ties are handled by the interpolation below, which reads the curve
    as the inverse CDF and evaluates F at each label boundary.
    """
    lo, hi = float(labels[0]), float(labels[-1])
    floor, ceiling = lo - 0.5, hi + 0.5
    # The game's support is bounded, so a forecast below the lowest or above
    # the highest label is out-of-range mass that belongs inside: clip it in
    # before it is read, rather than letting it stretch the curve.
    # Sorted, not merely assumed sorted: np.interp needs an increasing x and
    # silently returns nonsense otherwise, and quantile crossing is a known
    # artefact of these models (Chronos-2 even has a fix_quantile_crossing
    # option that is off by default). Clipping is monotone, so the order of
    # the two operations does not matter.
    values = np.sort(np.clip(np.asarray(quantile_values, dtype=float), floor, ceiling))
    levels = np.asarray(quantile_levels, dtype=float)

    # F(x) by interpolating the inverse CDF, anchored at (floor, 0) and
    # (ceiling, 1) - see the header: that anchoring is what spreads each tail
    # over its stretch instead of spiking the edge label.
    xs = np.concatenate(([floor], values, [ceiling]))
    ys = np.concatenate(([0.0], levels, [1.0]))

    def cdf(x):
        return float(np.interp(x, xs, ys, left=0.0, right=1.0))

    masses = []
    for label in labels:
        left = max(floor, label - 0.5)
        right = min(ceiling, label + 0.5)
        masses.append(max(0.0, cdf(right) - cdf(left)))

    masses = np.asarray(masses, dtype=float)
    # Safety net for a degenerate curve (every quantile identical, so the
    # anchors carry everything): whatever the loop above did not account for
    # still has to land on a label rather than vanish.
    masses[0] += max(0.0, cdf(floor))
    masses[-1] += max(0.0, 1.0 - cdf(ceiling))

    total = masses.sum()
    if total <= 0:
        masses = np.full(len(labels), 1.0 / len(labels))   # no information -> uniform
    else:
        masses = masses / total
    return {str(int(label)): float(round(mass, 6)) for label, mass in zip(labels, masses)}


def levels_within(count, low, high):
    """
    `count` evenly spaced quantile levels inside [low, high] - the range the
    model was actually trained on. Asking outside it buys nothing: the
    libraries clamp to their nearest trained level and warn.
    """
    count = max(2, int(count))
    return [round(low + i * (high - low) / (count - 1), 6) for i in range(count)]
