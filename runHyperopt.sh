#!/bin/bash

cd /root/sequencePredictor/

python3 HyperoptStatistics.py >> /root/sequencePredictor/log/hyperopt.log 2>&1

# Retrain the Phase 1 stacking meta-learner on the freshly tuned bestParams_<game>.json
# files, so Predictor.py's MetaLearner Model always reflects the latest hyperopt run.
python3 TrainMetaLearner.py >> /root/sequencePredictor/log/hyperopt.log 2>&1