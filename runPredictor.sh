#!/bin/bash

cd /root/sequencePredictor/

# --ai on again (change of 2026-09-16): every deep learning training run is capped
# (--dl-model-seconds, default 240 s, stops like an early stop and keeps the
# best weights) and the per-day DL child has a deadline (--dl-timeout, auto),
# so the heavy LSTM/TCN/unified rows are tracked daily without their training
# blocking the prediction flow - the reason they were switched off in August.
python3 Predictor.py -a true >> /root/sequencePredictor/log/predictor.log 2>&1