from src.Markov import Markov
from src.MarkovMonteCarlo import MarkovMonteCarlo
from src.MarkovBayesian import MarkovBayesian
from src.MarkovBayesianEnhanched import MarkovBayesianEnhanced
from src.PoissonMonteCarlo import PoissonMonteCarlo
from src.PoissonMarkov import PoissonMarkov
from src.LaplaceMonteCarlo import LaplaceMonteCarlo
from src.XGBoost import XGBoostPredictor
from src.ChronosModel import ChronosModel
from src.TimesFmModel import TimesFmModel

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
    "Chronos Model",
    "TimesFM Model",
]

# The foundation models (README roadmap item 5) are the only base models that
# are not always available: they need their own library directory, which a
# machine may not have. They are also the only ones that cannot be run inside
# the Backtester's forked pool - see prepare_foundation_scores below, which
# every caller that collects a table must invoke. Chronos-2 is a feature by
# default; TimesFM-3 costs ~2.8 GB against Chronos's ~0.8 GB for a second
# reading of the same kind of signal, so it is opt-in per game.
FOUNDATION_MODELS = (
    ("Chronos Model", ChronosModel, "useChronosFeature", True),
    ("TimesFM Model", TimesFmModel, "useTimesFmFeature", False),
)

# Models with no per-position modeling of their own - excluded for the
# positional games (Pick3, Joker+: the draw is an ordered digit sequence, see
# Helpers.is_positional_game), matching HyperoptStatistics.py's
# DISABLED_FOR_PICK3 / Predictor.py's is_positional_game guards. The old
# name is kept as an alias so existing imports keep working.
DISABLED_FOR_POSITIONAL = {"MarkovBayesian Model", "MarkovBayesianEnhanched Model", "PoissonMarkov Model"}
DISABLED_FOR_PICK3 = DISABLED_FOR_POSITIONAL


def build_models(dataPath, bestParams, is_positional=False, is_pick3=None):
    """
    Instantiates the 7 base models configured with this game's already-tuned
    hyperopt params (bestParams_<game>.json), mirroring how Predictor.py's
    statisticalMethod() and Backtester.py's own __main__ configure the same
    models. Falls back to the same defaults Predictor.py uses when a param is
    missing (e.g. bestParams_<game>.json predates a given model). Shared by
    TrainMetaLearner.py and HyperoptStatistics.py's Keno subset-tuning
    objective, so both build these models identically instead of drifting
    apart over time.

    is_positional: True for a positional game (Pick3, Joker+ - see
    Helpers.is_positional_game): every model keeps the digits in drawn order
    (sorted False, duplicates allowed), Markov uses pair scoring, and the
    DISABLED_FOR_POSITIONAL models (no per-position modeling) are left out.
    is_pick3 is the historical name of the same flag, still accepted so
    callers written before Joker+ existed (TrainMetaLearner.py,
    HyperoptQuantum.py, HyperoptStatistics.py) keep working unchanged.
    """
    if is_pick3 is not None:
        is_positional = bool(is_pick3)
    is_positional = bool(is_positional)

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
    markov.setSortedPrediction(not is_positional)
    markov.setUsePairScoring(is_positional)
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
    markovMcBase.setSortedPrediction(not is_positional)
    markovMonteCarlo = MarkovMonteCarlo(markovMcBase)
    markovMonteCarlo.setNumOfSimulations(bestParams.get("markovMcNumSimulations", 1000))
    models["MarkovMonteCarlo Model"] = markovMonteCarlo

    if not is_positional:
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
    poissonMonteCarlo.setSortedPrediction(not is_positional)
    models["PoissonMonteCarlo Model"] = poissonMonteCarlo

    if not is_positional:
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
    laplaceMonteCarlo.setSortedPrediction(not is_positional)
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
    xgboost.setSortedPrediction(not is_positional)
    xgboost.setNumThreads(1)
    xgboost.setSaveModels(False)
    models["XGBoost Model"] = xgboost

    # Foundation models as meta-learner features: a zero-shot per-position
    # distribution from a model that has never seen a lottery is a genuinely
    # different signal from the Markov/Poisson family and from boosting -
    # which is the whole point of stacking. Appended last for the same reason
    # XGBoost was: existing artifacts store their own feature_names and never
    # ask for a column they were not trained with, so every older artifact
    # keeps working and the column order of every existing feature is stable.
    # Re-run TrainMetaLearner.py to actually pick the new feature up.
    for name, cls, flag, default in FOUNDATION_MODELS:
        if not bestParams.get(flag, default) or not cls.installed():
            continue
        model = cls()
        model.setDataPath(dataPath)
        model.setSortedPrediction(not is_positional)
        model.setRecentDraws(bestParams.get("foundationContext", 512))
        models[name] = model

    return models


def expected_model_names(dataPath, bestParams, is_positional=False, is_pick3=None):
    """
    The names build_models() will produce here, without constructing
    anything. A cached score table is only reusable if it was collected from
    the same SET of base models, and the callers check that before deciding
    whether to collect - so this has to answer before the models exist.
    `python3 -m src.ModelFactory` pins it against build_models().
    """
    if is_pick3 is not None:
        is_positional = bool(is_pick3)
    names = [name for name in BASE_MODEL_NAMES
             if not (is_positional and name in DISABLED_FOR_POSITIONAL)
             and name not in {n for n, _, _, _ in FOUNDATION_MODELS}]
    for name, cls, flag, default in FOUNDATION_MODELS:
        if bestParams.get(flag, default) and cls.installed():
            names.append(name)
    return names


def prepare_foundation_scores(models, start_index, total_rows, skipLastColumns=0,
                              specialColumnCount=0, label=""):
    """
    Forecast every day the backtest will ask for, here, in this process, and
    then put each foundation model into cache-only mode.

    src/Backtester.py forks its pool and shares the model objects
    copy-on-write. A forked child must not talk to the parent's worker (two
    conversations in one pipe) and must not start its own (fifteen children x
    0.8-2.8 GB is how a 16 GB box dies), so the parent does the work up front
    - one warm worker, ~0.2 s per day - and the children then read an
    inherited dictionary. Returns {model name: days forecast}.

    The keys mirror exactly what Backtester._backtest_day asks for: the main
    (or positional) call with the special column(s) dropped, plus the
    special-only call for the games that have one.
    """
    foundation = {name: model for name, model in models.items()
                  if getattr(model, "IS_FOUNDATION_MODEL", False)}
    if not foundation:
        return {}

    mainSkip = specialColumnCount if specialColumnCount > 0 else skipLastColumns
    keys = []
    for index in range(start_index, total_rows):
        skipRows = total_rows - index
        keys.append((skipRows, mainSkip, 0))
        if specialColumnCount > 0:
            keys.append((skipRows, 0, specialColumnCount))

    done = {}
    for name, model in foundation.items():
        print(f"{label}{name}: forecasting {len(keys)} day-slices before the backtest forks...")
        done[name] = model.precompute(keys, label=f"{label}{name}")
        if done[name] == 0:
            print(f"{label}{name}: no forecast succeeded - the column will be all zeros "
                  f"(is {name} installed? see README 'Foundation models')")
    return done


if __name__ == "__main__":
    # expected_model_names() duplicates build_models()'s inclusion rules, and
    # a drift between them would silently reuse a score table collected from
    # a different set of base models. This is the check that they agree.
    #
    #   python3 -m src.ModelFactory
    import itertools
    import os

    dataPath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "trainingData", "lotto")
    cases = [dict(pair) for pair in itertools.product(
        [("useChronosFeature", True), ("useChronosFeature", False)],
        [("useTimesFmFeature", True), ("useTimesFmFeature", False)])]
    failures = 0
    for is_positional in (False, True):
        for params in cases:
            built = list(build_models(dataPath, params, is_positional=is_positional).keys())
            expected = expected_model_names(dataPath, params, is_positional=is_positional)
            if built != expected:
                failures += 1
                print(f"MISMATCH positional={is_positional} {params}\n  built    {built}\n  expected {expected}")
    print(f"{2 * len(cases) - failures}/{2 * len(cases)} configurations agree"
          + ("" if not failures else " - FIX expected_model_names()"))
    raise SystemExit(1 if failures else 0)
