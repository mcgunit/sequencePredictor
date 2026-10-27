#!/bin/bash

# The weekly tuning chain plus the four control experiments. Since README
# roadmap item 8 the web server's scheduler (jobs.js) runs these same commands,
# in this same order, as eleven separate jobs in two plans: the seven steps
# from HyperoptStatistics.py to the keno sub-selection tuning after
# TrainMetaLearner.py as the weekly chain, started when Saturday's predictor
# finishes, and RandomnessDiscrimination.py .. NullControls.py as the weekly
# controls plan, started when Sunday's predictor finishes - never at a fixed
# hour, each one visible with its own exit code on the Jobs page. This script
# stays as the hand-run path for both and as the fallback while the schedule
# is off; keep the commands, logs and order identical, test/jobs.test.js
# compares the two.

cd /root/sequencePredictor/

# Wait for the daily predictor to release process.lock instead of letting
# every tuner exit with "Another instance is already running": with the deep
# learning rows time-boxed back into the daily run (runPredictor.sh -a true)
# a Saturday run that also recovers a gap can still be busy at 14:00, and a
# skipped week of tuning would be silent. Waits at most 6 hours.
LOCK=/root/sequencePredictor/process.lock
waited=0
while [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; do
    if [ "$waited" -ge 21600 ]; then
        echo "$(date -u '+%F %T') runHyperopt.sh: process.lock still held by PID $(cat "$LOCK") after 6h - giving up this week" >> /root/sequencePredictor/log/hyperoptStatistics.log
        exit 1
    fi
    if [ "$waited" -eq 0 ]; then
        echo "$(date -u '+%F %T') runHyperopt.sh: waiting for process.lock (held by PID $(cat "$LOCK"))" >> /root/sequencePredictor/log/hyperoptStatistics.log
    fi
    sleep 60
    waited=$((waited + 60))
done

python3 HyperoptStatistics.py >> /root/sequencePredictor/log/hyperoptStatistics.log 2>&1

# Tune the boosting model (XGBoost Model) the same way, into the same
# bestParams_<game>.json files. Runs after HyperoptStatistics.py because both
# take the shared process.lock.
python3 HyperoptBoost.py >> /root/sequencePredictor/log/hyperoptBoost.log 2>&1

# Tune the RL Ticket Model (pure numpy, minutes not hours) into the same
# bestParams_<game>.json files. Shares process.lock, so it must stay sequenced
# after the other tuners - and before TrainMetaLearner.py.
python3 HyperoptRLTicket.py >> /root/sequencePredictor/log/hyperoptRLTicket.log 2>&1

# Select the rows the SubsetEnsemble Model votes over (README roadmap item 2).
# Enumerates the subsets of the tracked rows on the stored day JSONs only -
# seconds per game - and shares process.lock, so it stays sequenced with the
# others.
python3 HyperoptEnsemble.py >> /root/sequencePredictor/log/hyperoptEnsemble.log 2>&1

# Tune the two quantum meta-learner variants (quantum-kernel SVC and VQC) and
# their classical RBF-SVM control into the same bestParams_<game>.json files. Shares process.lock, so it stays
# sequenced after the other tuners - and it MUST run before TrainMetaLearner.py:
# the whole point is that the weekly retrain trains the quantum artifacts on
# freshly tuned quantumKernel_*/quantumVqc_*/classicalSvm_* params instead of week-old ones.
python3 HyperoptQuantum.py >> /root/sequencePredictor/log/hyperoptQuantum.log 2>&1

# Retrain the Phase 1 stacking meta-learner on the freshly tuned bestParams_<game>.json
# files, so Predictor.py's MetaLearner Model always reflects the latest hyperopt run.
python3 TrainMetaLearner.py >> /root/sequencePredictor/log/TrainMetaLearner.log 2>&1

# Tune how the weighted vote and the five meta-learner rows cut their keno
# ticket into the 5-10-number sub-selections (HyperoptStatistics.py's
# KenoSubsetTuning, out of its default run since 26 Oct 2026). Last on
# purpose: its day table scores the artifacts TrainMetaLearner.py has just
# written (the quantum ones on HyperoptQuantum.py's fresh parameters) and the
# weighted vote on this week's statistical and XGBoost parameters. Inside the
# first job it tuned on last weekend's artifacts, which the retrain then
# replaced. About a quarter of an hour, most of it the day table's backtest.
python3 HyperoptStatistics.py -g keno -s KenoSubsetTuning >> /root/sequencePredictor/log/kenoSubsetTuning.log 2>&1

# The control experiments (README "Null controls"), after everything that
# changes what is served: measurements of the rows, never inputs to them.
# Q2 first (about 80 minutes: 20 seeds for every game since 1 Nov 2026), then
# the irrelevant-feature control on the meta-learner table the chain just
# cached (refits only, about an hour), then Q0 (hours: a full backtest per
# control history). All write under data/controls/ for the History page's cards.
python3 RandomnessDiscrimination.py -g lotto,euromillions,eurodreams,vikinglotto,keno,pick3,jokerplus,crypto,shares,cryptoweek,sharesweek -w 10 -n 20 --seed weekly >> /root/sequencePredictor/log/randomnessDiscrimination.log 2>&1
python3 IrrelevantFeatureControl.py -g lotto,euromillions,eurodreams,vikinglotto,keno,pick3,jokerplus,crypto,shares,cryptoweek,sharesweek -r 3 -k 3 --seed weekly >> /root/sequencePredictor/log/featureControl.log 2>&1
python3 MarketRows.py -g crypto,shares,cryptoweek,sharesweek -d 250 >> /root/sequencePredictor/log/marketRows.log 2>&1
python3 NullControls.py -g lotto,euromillions,eurodreams,vikinglotto,keno,pick3,jokerplus,crypto,shares,cryptoweek,sharesweek -m both -n 3 --market-repetitions 5 --seed weekly -d 120 >> /root/sequencePredictor/log/nullControls.log 2>&1