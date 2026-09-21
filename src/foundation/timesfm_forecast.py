"""
Zero-shot per-position forecasts from Google's pretrained time-series
foundation model (google/timesfm-3.0-pytorch), for src/TimesFmModel.py.

Same contract as chronos_forecast.py - one JSON object per line in, one per
line out - so the two foundation models are interchangeable from the
pipeline's side and can be compared without a second code path:

  ->  {"series": [[float, ...], ...], "labels": [int, ...]}
  <-  {"ok": true, "scores": [{"1": 0.03, ...}, ...], "seconds": 0.21}
  <-  {"ok": false, "error": "..."}

Two honest differences from the Chronos worker, both reported in the reply:

  * the checkpoint publishes a FIXED grid of nine quantiles (the deciles), so
    the request's "quantiles" resolution is ignored - a label distribution
    here is read off nine knots instead of Chronos-2's twenty-one. Finer
    structure is therefore less trustworthy, never more.
  * it is a bigger model: ~2.8 GB resident against Chronos-2's ~0.8 GB, and
    the first run downloads ~1.2 GB of weights.

This script runs in its OWN interpreter environment (FOUNDATION_LIBS), not
the pipeline's - see src/FoundationModel.py. Nothing here may be imported by
the pipeline.
"""

import json
import os
import sys
import time

LIBS = os.environ.get("FOUNDATION_LIBS", "/root/.foundation-libs")
if os.path.isdir(LIBS) and LIBS not in sys.path:
    sys.path.insert(0, LIBS)

from quantile_labels import label_probabilities

MODEL = os.environ.get("TIMESFM_MODEL", "google/timesfm-3.0-pytorch")
DEVICE = os.environ.get("TIMESFM_DEVICE", "cpu")
# Same reasoning as the Chronos worker: a lottery position series is
# stationary by construction, so a few hundred draws carry whatever structure
# exists and a longer context only costs time.
MAX_CONTEXT = int(os.environ.get("TIMESFM_CONTEXT") or os.environ.get("FOUNDATION_CONTEXT") or "512")
THREADS = int(os.environ.get("TIMESFM_THREADS", "4"))

_forecaster = None


def forecaster():
    global _forecaster
    if _forecaster is None:
        import torch
        import timesfm
        torch.set_num_threads(THREADS)
        if DEVICE == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        _forecaster = timesfm.TimesFM3Forecaster.from_pretrained(MODEL)
    return _forecaster


def quantile_levels(count):
    """
    The levels behind the checkpoint's quantile outputs. TimesFM publishes
    evenly spaced interior quantiles - nine of them, i.e. the deciles - so
    (i + 1) / (count + 1) is the grid, derived from the array's own width
    rather than hardcoded, in case a future checkpoint widens it.
    """
    return [round((i + 1) / (count + 1), 6) for i in range(count)]


def forecast(request):
    import numpy as np

    series = request.get("series") or []
    labels = [int(v) for v in (request.get("labels") or [])]
    if not series or not labels:
        return {"ok": False, "error": "series and labels are required"}

    contexts = []
    for values in series:
        window = [float(v) for v in values][-MAX_CONTEXT:]
        if len(window) < 8:
            return {"ok": False, "error": f"a position has only {len(window)} draws of history"}
        contexts.append(np.asarray(window, dtype=np.float32))

    started = time.time()
    outputs = list(forecaster().predict_batch(contexts, horizon=1, return_quantiles=True))
    if len(outputs) != len(contexts):
        return {"ok": False, "error": f"asked for {len(contexts)} series, got {len(outputs)} forecasts"}

    scores = []
    levels = None
    for output in outputs:
        quantiles = np.asarray(output.quantiles, dtype=float)
        if quantiles.ndim != 2 or quantiles.shape[0] < 1:
            return {"ok": False, "error": f"unexpected quantile shape {quantiles.shape}"}
        row = np.sort(quantiles[0])          # the horizon-1 step, non-decreasing
        if levels is None:
            levels = quantile_levels(len(row))
        scores.append(label_probabilities(row, levels, labels))

    return {"ok": True, "scores": scores, "seconds": round(time.time() - started, 3),
            "model": MODEL, "levels": len(levels or []),
            "context": min(MAX_CONTEXT, max(len(s) for s in series))}


def main():
    # Warm the model before the first reply so the caller's timeout applies to
    # the forecast, not to a 1.2 GB download on a fresh machine.
    if "--warm" in sys.argv:
        try:
            forecaster()
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
