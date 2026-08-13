#!/bin/bash

cd /root/sequencePredictor/

python3 HyperoptStatistics.py >> /root/sequencePredictor/log/hyperoptStatistics.log 2>&1

# Tune the boosting model (XGBoost Model) the same way, into the same
# bestParams_<game>.json files. Runs after HyperoptStatistics.py because both
# take the shared process.lock.
python3 HyperoptBoost.py >> /root/sequencePredictor/log/hyperoptBoost.log 2>&1

# Retrain the Phase 1 stacking meta-learner on the freshly tuned bestParams_<game>.json
# files, so Predictor.py's MetaLearner Model always reflects the latest hyperopt run.
python3 TrainMetaLearner.py >> /root/sequencePredictor/log/TrainMetaLearner.log 2>&1