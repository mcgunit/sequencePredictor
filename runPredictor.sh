#!/bin/bash

# The daily run. Since README roadmap item 8 the web server's own scheduler
# (jobs.js) starts this same command at 09:00 and the crontab entries are
# gone; this script stays as the hand-run path and as the fallback while the
# schedule is off (config/scheduler.disabled). Keep the two in step -
# test/jobs.test.js fails if they diverge.

cd /root/sequencePredictor/

# --ai on again (change of 2026-09-16): every deep learning training run is capped
# (--dl-model-seconds, default 240 s, stops like an early stop and keeps the
# best weights) and the per-day DL child has a deadline (--dl-timeout, auto),
# so the heavy LSTM/TCN/unified rows are tracked daily without their training
# blocking the prediction flow - the reason they were switched off in August.
python3 Predictor.py -a true >> /root/sequencePredictor/log/predictor.log 2>&1