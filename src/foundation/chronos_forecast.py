"""
Zero-shot per-position forecasts from a pretrained time-series foundation
model (amazon/chronos-2), for src/ChronosModel.py.

This script runs in its OWN interpreter environment, not the pipeline's: the
foundation stack wants numpy 2.x while the pipeline runs TensorFlow 2.16 on
numpy 1.26, so the libraries live in a separate directory (default
/root/.foundation-libs, override with FOUNDATION_LIBS) that only this process
puts on its path. Nothing here may be imported by the pipeline.

Protocol - one JSON object per line on stdin, one per line on stdout, so the
caller can keep one warm process for a whole backtest instead of paying the
model load per day:

  ->  {"series": [[float, ...], ...], "labels": [int, ...], "quantiles": 199}
  <-  {"ok": true, "scores": [{"1": 0.03, ...}, ...], "seconds": 0.16}
  <-  {"ok": false, "error": "..."}

`series[i]` is the history of drawn values at position i, oldest first.
`labels` is the game's value range. The reply gives, per position, a
probability for every label.

The mapping from a quantile curve to label probabilities lives in
quantile_labels.py, shared with the TimesFM worker so both models are read
exactly the same way.
"""

import json
import os
import sys
import time

LIBS = os.environ.get("FOUNDATION_LIBS", "/root/.foundation-libs")
if os.path.isdir(LIBS) and LIBS not in sys.path:
    sys.path.insert(0, LIBS)

from quantile_labels import label_probabilities, levels_within

MODEL = os.environ.get("CHRONOS_MODEL", "amazon/chronos-2")
DEVICE = os.environ.get("CHRONOS_DEVICE", "cpu")
# Longer contexts cost time and add nothing here: a lottery position series is
# stationary by construction, so a few hundred draws already carry whatever
# structure exists (for sorted games, the order statistics).
MAX_CONTEXT = int(os.environ.get("CHRONOS_CONTEXT") or os.environ.get("FOUNDATION_CONTEXT") or "512")
THREADS = int(os.environ.get("CHRONOS_THREADS", "4"))
# What amazon/chronos-2 was trained on; anything outside is clamped by the
# library to these ends.
TRAINED_QUANTILE_RANGE = (0.01, 0.99)

_pipeline = None


def pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from chronos import BaseChronosPipeline
        torch.set_num_threads(THREADS)
        _pipeline = BaseChronosPipeline.from_pretrained(MODEL, device_map=DEVICE)
    return _pipeline


def forecast(request):
    import torch

    series = request.get("series") or []
    labels = [int(v) for v in (request.get("labels") or [])]
    if not series or not labels:
        return {"ok": False, "error": "series and labels are required"}

    count = int(request.get("quantiles") or 199)
    count = max(9, min(count, 999))
    # Chronos-2 is trained on 21 quantile levels (0.01, 0.05 ... 0.95, 0.99).
    # A finer in-range grid is pure interpolation between those knots, kept
    # only because reading the curve at every label boundary is what the
    # caller needs - see quantile_labels.py.
    levels = levels_within(count, *TRAINED_QUANTILE_RANGE)

    contexts = []
    for values in series:
        window = [float(v) for v in values][-MAX_CONTEXT:]
        if len(window) < 8:
            return {"ok": False, "error": f"a position has only {len(window)} draws of history"}
        contexts.append(torch.tensor(window, dtype=torch.float32))

    started = time.time()
    quantiles, _ = pipeline().predict_quantiles(contexts, prediction_length=1, quantile_levels=levels)
    scores = [label_probabilities(quantiles[i][0, 0].numpy(), levels, labels) for i in range(len(contexts))]
    return {"ok": True, "scores": scores, "seconds": round(time.time() - started, 3),
            "model": MODEL, "context": min(MAX_CONTEXT, max(len(s) for s in series))}


def main():
    # Warm the model before the first reply so the caller's timeout applies to
    # the forecast, not to a 100 MB download on a fresh machine.
    if "--warm" in sys.argv:
        try:
            pipeline()
            print(json.dumps({"ok": True, "warm": True, "model": MODEL}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = forecast(json.loads(line))
        except Exception as exc:
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(reply), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
