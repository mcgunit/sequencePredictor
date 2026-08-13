from src.Markov import Markov
from src.MarkovMonteCarlo import MarkovMonteCarlo
from src.MarkovBayesian import MarkovBayesian
from src.MarkovBayesianEnhanched import MarkovBayesianEnhanced
from src.PoissonMonteCarlo import PoissonMonteCarlo
from src.PoissonMarkov import PoissonMarkov
from src.LaplaceMonteCarlo import LaplaceMonteCarlo
from src.XGBoost import XGBoostPredictor

# Ordered list of base-model display names fed into the meta-learner -
# HybridStatisticalModel is deliberately excluded: it's itself a vote-based
# ensemble of several of these models, so feeding it in too would be circular.
# The order here fixes the feature-vector column order persisted alongside
# the meta-learner, so Predictor.py must build vectors in this same order.
#
# XGBoost Model is appended last, deliberately: every meta-learner artifact
# stores its own feature_names and Predictor.py builds vectors from THAT list
# (skipping names it can't score), so an artifact trained before this change
# keeps working untouched - it simply never asks for the boosting feature.
# Appending rather than inserting also keeps the column order of every
# existing feature stable, so an old and a new artifact stay directly
# comparable. Re-run TrainMetaLearner.py to actually pick the new feature up.
BASE_MODEL_NAMES = [
    "Markov Model",
    "MarkovMonteCarlo Model",
    "MarkovBayesian Model",
    "MarkovBayesianEnhanched Model",
    "PoissonMonteCarlo Model",
    "PoissonMarkov Model",
    "LaplaceMonteCarlo Model",
    "XGBoost Model",
]

# Models with no per-position modeling of their own - excluded for Pick3,
# matching HyperoptStatistics.py's DISABLED_FOR_PICK3 / Predictor.py's
# "not pick3 in name" guards.
DISABLED_FOR_PICK3 = {"MarkovBayesian Model", "MarkovBayesianEnhanched Model", "PoissonMarkov Model"}


def build_models(dataPath, bestParams, is_pick3):
    """
    Instantiates the 7 base models configured with this game's already-tuned
    hyperopt params (bestParams_<game>.json), mirroring how Predictor.py's
    statisticalMethod() and Backtester.py's own __main__ configure the same
    models. Falls back to the same defaults Predictor.py uses when a param is
    missing (e.g. bestParams_<game>.json predates a given model). Shared by
    TrainMetaLearner.py and HyperoptStatistics.py's Keno subset-tuning
    objective, so both build these models identically instead of drifting
    apart over time.
    """
    models = {}

    markov = Markov()
    markov.setDataPath(dataPath)
    markov.setSoftMAxTemperature(bestParams.get("markovSoftMaxTemperature", 0.1))
    markov.setMinOccurrences(bestParams.get("markovMinOccurences", 9))
    markov.setAlpha(bestParams.get("markovAlpha", 0.2))
    markov.setRecencyWeight(bestParams.get("markovRecencyWeight", 1.0))
    markov.setRecencyMode(bestParams.get("markovRecencyMode", "constant"))
    markov.setPairDecayFactor(bestParams.get("markovPairDecayFactor", 0.3))
    markov.setSmoothingFactor(bestParams.get("markovSmoothingFactor", 0.6))
    markov.setSubsetSelectionMode(bestParams.get("markovSubsetSelectionMode", "softmax"))
    markov.setBlendMode(bestParams.get("markovBlendMode", "log"))
    markov.setMarkovOrder(bestParams.get("markovOrder", 1))
    markov.setSortedPrediction(not is_pick3)
    markov.setUsePairScoring(is_pick3)
    markov.setPairScoringWeight(bestParams.get("markovPairScoringWeight", 0.0))
    models["Markov Model"] = markov

    markovMcBase = Markov()
    markovMcBase.setDataPath(dataPath)
    markovMcBase.setSoftMAxTemperature(bestParams.get("markovMcSoftMaxTemperature", 0.1))
    markovMcBase.setMinOccurrences(bestParams.get("markovMcMinOccurences", 9))
    markovMcBase.setAlpha(bestParams.get("markovMcAlpha", 0.2))
    markovMcBase.setRecencyWeight(bestParams.get("markovMcRecencyWeight", 1.0))
    markovMcBase.setRecencyMode(bestParams.get("markovMcRecencyMode", "constant"))
    markovMcBase.setPairDecayFactor(bestParams.get("markovMcPairDecayFactor", 0.3))
    markovMcBase.setSmoothingFactor(bestParams.get("markovMcSmoothingFactor", 0.6))
    markovMcBase.setMarkovOrder(bestParams.get("markovMcOrder", 1))
    markovMcBase.setSortedPrediction(not is_pick3)
    markovMonteCarlo = MarkovMonteCarlo(markovMcBase)
    markovMonteCarlo.setNumOfSimulations(bestParams.get("markovMcNumSimulations", 1000))
    models["MarkovMonteCarlo Model"] = markovMonteCarlo

    if not is_pick3:
        markovBayesian = MarkovBayesian()
        markovBayesian.setDataPath(dataPath)
        markovBayesian.setSoftMAxTemperature(bestParams.get("markovBayesianSoftMaxTemperature", 0.24))
        markovBayesian.setAlpha(bestParams.get("markovBayesianAlpha", 0.15))
        markovBayesian.setMinOccurrences(bestParams.get("markovBayesianMinOccurences", 14))
        markovBayesian.setSortedPrediction(True)
        models["MarkovBayesian Model"] = markovBayesian

        markovBayesianEnhanced = MarkovBayesianEnhanced()
        markovBayesianEnhanced.setDataPath(dataPath)
        markovBayesianEnhanced.setSoftMAxTemperature(bestParams.get("markovBayesianEnhancedSoftMaxTemperature", 0.42))
        markovBayesianEnhanced.setAlpha(bestParams.get("markovBayesianEnhancedAlpha", 0.4))
        markovBayesianEnhanced.setMinOccurrences(bestParams.get("markovBayesianEnhancedMinOccurences", 19))
        markovBayesianEnhanced.setSortedPrediction(True)
        models["MarkovBayesianEnhanched Model"] = markovBayesianEnhanced

    poissonMonteCarlo = PoissonMonteCarlo()
    poissonMonteCarlo.setDataPath(dataPath)
    poissonMonteCarlo.setNumOfSimulations(bestParams.get("poissonMonteCarloNumberOfSimulations", 600))
    poissonMonteCarlo.setWeightFactor(bestParams.get("poissonMonteCarloWeightFactor", 0.8))
    poissonMonteCarlo.setSortedPrediction(not is_pick3)
    models["PoissonMonteCarlo Model"] = poissonMonteCarlo

    if not is_pick3:
        poissonMarkovWeight = bestParams.get("poissonMarkovWeight", 0.5)
        poissonMarkov = PoissonMarkov()
        poissonMarkov.setDataPath(dataPath)
        poissonMarkov.setWeights(poisson_weight=poissonMarkovWeight, markov_weight=1 - poissonMarkovWeight)
        poissonMarkov.setNumberOfSimulations(bestParams.get("poissonMarkovNumberOfSimulations", 100))
        poissonMarkov.setSortedPrediction(True)
        models["PoissonMarkov Model"] = poissonMarkov

    laplaceMonteCarlo = LaplaceMonteCarlo()
    laplaceMonteCarlo.setDataPath(dataPath)
    laplaceMonteCarlo.setNumOfSimulations(bestParams.get("laplaceMonteCarloNumberOfSimulations", 900))
    laplaceMonteCarlo.setSortedPrediction(not is_pick3)
    models["LaplaceMonteCarlo Model"] = laplaceMonteCarlo

    # Gradient boosting as a meta-learner feature: a boosted-tree score is a
    # genuinely different signal from the Markov/Poisson family, which is the
    # whole point of stacking. Same tuned xgBoost* params Predictor.py's
    # boostingMethod reads, so the feature the meta-learner is trained on is
    # the same one it gets served at prediction time.
    #
    # Note the cost: unlike the models above (whose "fit" is a frequency or
    # transition count), every XGBoost score is a real training run, so a
    # backtest collecting scores over N days trains N times. src/XGBoost.py
    # caches its fit per (data slice, hyperparameters) so run() and
    # score_numbers() on the same day don't train twice, and threads stay at 1
    # because Backtester already parallelises across days.
    xgboost = XGBoostPredictor()
    xgboost.setDataPath(dataPath)
    xgboost.setEstimators(bestParams.get("xgBoostEstimators", 200))
    xgboost.setLearningRate(bestParams.get("xgBoostLearningRate", 0.1))
    xgboost.setMaxDepth(bestParams.get("xgBoostMaxdepth", 3))
    xgboost.setPreviousDraws(bestParams.get("xgBoostPreviousDraws", 11))
    xgboost.setTopK(bestParams.get("xgBoostTopK", 16))
    xgboost.setForceNested(bestParams.get("xgBoostForceNested", True))
    xgboost.setSubsample(bestParams.get("xgBoostSubsample", 1.0))
    xgboost.setColsampleByTree(bestParams.get("xgBoostColsampleByTree", 1.0))
    xgboost.setMinChildWeight(bestParams.get("xgBoostMinChildWeight", 1.0))
    xgboost.setRegLambda(bestParams.get("xgBoostRegLambda", 1.0))
    xgboost.setSubsetSelectionMode(bestParams.get("xgBoostSubsetMode", "softmax"))
    xgboost.setSubsetTemperature(bestParams.get("xgBoostSubsetTemperature", 0.5))
    xgboost.setSortedPrediction(not is_pick3)
    xgboost.setNumThreads(1)
    xgboost.setSaveModels(False)
    models["XGBoost Model"] = xgboost

    return models
