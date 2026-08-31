#!/bin/bash

cd /root/sequencePredictor/

python3 HyperoptStatistics.py >> /root/sequencePredictor/log/hyperoptStatistics.log 2>&1

# Tune the boosting model (XGBoost Model) the same way, into the same
# bestParams_<game>.json files. Runs after HyperoptStatistics.py because both
# take the shared process.lock.
python3 HyperoptBoost.py >> /root/sequencePredictor/log/hyperoptBoost.log 2>&1

# Tune the RL Ticket Model (pure numpy, minutes not hours) into the same
# bestParams_<game>.json files. Shares process.lock, so it must stay sequenced
# after the other tuners - and before TrainMetaLearner.py, which stays last.
python3 HyperoptRLTicket.py >> /root/sequencePredictor/log/hyperoptRLTicket.log 2>&1

# Tune the two quantum meta-learner variants (quantum-kernel SVC and VQC) into
# the same bestParams_<game>.json files. Shares process.lock, so it stays
# sequenced after the other tuners - and it MUST run before TrainMetaLearner.py:
# the whole point is that the weekly retrain trains the quantum artifacts on
# freshly tuned quantumKernel_*/quantumVqc_* params instead of week-old ones.
python3 HyperoptQuantum.py >> /root/sequencePredictor/log/hyperoptQuantum.log 2>&1

# Retrain the Phase 1 stacking meta-learner on the freshly tuned bestParams_<game>.json
# files, so Predictor.py's MetaLearner Model always reflects the latest hyperopt run.
python3 TrainMetaLearner.py >> /root/sequencePredictor/log/TrainMetaLearner.log 2>&1