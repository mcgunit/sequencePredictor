import os, argparse, json, signal, sys

# Must be set before TensorFlow is imported (the src imports below pull it in):
# without it TF's first process grabs virtually the whole GPU at startup, so
# any second TF process - or a big trial on the 6GB card - dies with cuDNN
# RESOURCE_EXHAUSTED even when the card is otherwise idle. Allow-growth makes
# TF allocate only what it actually uses.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import gc
import time
import optuna
import numpy as np

from art import text2art
from datetime import datetime, timezone
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError


from src.TCN import TCNModel
from src.LSTM import LSTMModel
from src.UnifiedLstmTcn import UnifiedLstmTcnModel
from src.UnifiedLstmGruTcn import UnifiedLstmGruTcnModel
from src.TransformerModel import TransformerModel
from src.GNN import GNNModel
from src.AutoencoderAnomaly import AutoencoderAnomaly
from src.Markov import Markov
from src.MarkovBayesian import MarkovBayesian
from src.MarkovBayesianEnhanched import MarkovBayesianEnhanced
from src.PoissonMonteCarlo import PoissonMonteCarlo
from src.PoissonMarkov import PoissonMarkov
from src.LaplaceMonteCarlo import LaplaceMonteCarlo
from src.HybridStatisticalModel import HybridStatisticalModel
from src.XGBoost import XGBoostKenoPredictor
from src.Command import Command
from src.Helpers import Helpers
from src.DataFetcher import DataFetcher
from src.HyperoptRunner import fail_stale_running_trials
from src.TuningGate import challenge, make_defaults
from src.TuningScore import attrs_for_trial
from src import DeepLearningTuning as dlt



def cap_gpu_memory():
    """
    The GPU worker's cap (10 Nov 2026): with SP_GPU_MEMORY_MB set (the worker sets it), TensorFlow gets one logical
    GPU of that many MB, so the shared card keeps the rest for the other services on the machine. Runs at import, in
    the parent and in every spawned training child, before TensorFlow touches the GPU. Unset: no change (production).
    """
    mb = os.environ.get("SP_GPU_MEMORY_MB")
    if not mb:
        return
    import tensorflow as tf
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(mb))])
    except RuntimeError as e:   # the GPU was initialised already: the cap cannot be set any more
        print(f"GPU memory cap of {mb} MB not set: {e}")


cap_gpu_memory()

tcn = TCNModel()
lstm = LSTMModel()
unifiedLstmTcn = UnifiedLstmTcnModel()
unifiedLstmGruTcn = UnifiedLstmGruTcnModel()
transformer = TransformerModel()
gnn = GNNModel()
autoencoderAnomaly = AutoencoderAnomaly()
markov = Markov()
markovBayesian = MarkovBayesian()
markovBayesianEnhanced = MarkovBayesianEnhanced()
poissonMonteCarlo = PoissonMonteCarlo()
laplaceMonteCarlo = LaplaceMonteCarlo()
hybridStatisticalModel = HybridStatisticalModel()
poissonMarkov = PoissonMarkov()
xgboostPredictor = XGBoostKenoPredictor()
command = Command()
helpers = Helpers()
dataFetcher = DataFetcher()

LOCK_FILE = os.path.join(os.getcwd(), "process.lock")

# Module level (not just in the __main__ block): runPredictInChild's spawned
# children re-import this file as __mp_main__, which skips __main__ - but
# predict()/deepLearningMethod() read this global, so without it every trial
# dies in the child with NameError before any training happens.
path = os.getcwd()

# Prefix used for each new model's Optuna param names / bestParams_<game>.json
# keys - Predictor.py's runUnifiedDeepLearningModels() reads these same
# prefixed keys. Without a prefix, two independent per-model_type Optuna
# studies writing e.g. a bare "batchSize" would silently clobber each other
# (and LSTM's own already-tuned bare "batchSize") in bestParams_<game>.json -
# same reasoning as HyperoptBoost.suggest_boosting_params' per-model prefixes
# (BOOSTING_PARAM_SUFFIXES in src/BoostingBase.py).
MODEL_PARAM_PREFIX = {
    "tcn_model": "tcn",
    "unified_lstm_tcn_model": "unifiedLstmTcn",
    "unified_lstm_gru_tcn_model": "unifiedLstmGruTcn",
    "transformer_model": "transformer",
    "gnn_model": "gnn",
    "autoencoder_model": "autoencoder",
}

MODEL_DISPLAY_NAMES = {
    "lstm_model": "LSTM Base Model",
    "tcn_model": "TCN Base Model",
    "unified_lstm_tcn_model": "UnifiedLstmTcn Model",
    "unified_lstm_gru_tcn_model": "UnifiedLstmGruTcn Model",
    "transformer_model": "Transformer Model",
    "gnn_model": "GNN Model",
    "autoencoder_model": "Autoencoder Model",
}

# Trailing special/bonus column(s): Euromillions (2 star columns),
# EuroDreams (1 dream number), VikingLotto (1 super viking), Joker+ (1
# zodiac sign, stored as its 0..11 code - see Helpers.encode_zodiac) get
# modeled via a second output head (see LSTM.py/TCN.py/UnifiedLstmTcn.py/
# UnifiedLstmGruTcn.py create_model) instead of being lumped into the
# main numbers' output. Lotto's bonus number isn't modeled at all - it's
# simply dropped via skip_last_columns. Module level (was in __main__)
# because update_matching_numbers/process_single_history_entry need it to
# score matches per pool, and spawned children re-import this file as
# __mp_main__, which skips __main__. Mirrors Predictor.py / Helpers.
SPECIAL_COLUMN_COUNTS = {"euromillions": 2, "eurodreams": 1, "vikinglotto": 1, "jokerplus": 1}


def matchingSplitArgs(name, realResult):
    """
    (specialColumnCount, realMainCount) for find_best_matching_prediction, so
    matches are scored per pool (mains against mains, specials against
    specials) instead of one pooled set intersection - mirrors Predictor.py's
    helper of the same name. Lotto follows the real
    game's tiers: 6 played numbers score against the 6 drawn mains, and the
    7th (bonus) value only supplements a partial match - "5 (1)" is a high
    tier, "6 (0)" the jackpot - so lotto passes realMainCount=6 and
    find_best_matching_prediction matches the trailing bonus against the
    ticket itself into the special count ("vikinglotto" contains "lotto" but
    its viking is a dedicated special column, hence the exclusion).
    Keno/pick3 fall through with (0, None) - no split needed. Joker+ gets
    (1, None): its 7th value is the zodiac code; callers also pass game=name
    to find_best_matching_prediction so its rows are scored by the
    leading/trailing digit runs the game pays on instead of set hits.
    """
    specialColumnCount = next(
        (count for game, count in SPECIAL_COLUMN_COUNTS.items() if game in name), 0)
    realMainCount = None
    if "lotto" in name and "vikinglotto" not in name:
        realMainCount = len(realResult) - 1
    return specialColumnCount, realMainCount


def configure_lstm(model, modelParams):
    model.setBatchSize(modelParams["batchSize"])
    model.setEpochs(modelParams["epochs"])
    model.setNumberOfLSTMLayers(modelParams["num_lstm_layers"])
    model.setNumberOfLstmUnits(modelParams["lstm_units"])
    model.setNumberOfBidrectionalLayers(modelParams["num_bidirectional_layers"])
    model.setNumberOfBidirectionalLstmUnits(modelParams["bidirectional_lstm_units"])
    model.setDropout(modelParams["dropout"])
    model.setL2Regularization(modelParams["l2Regularization"])
    model.setEarlyStopPatience(modelParams["earlyStopPatience"])
    model.setReduceLearningRatePAience(modelParams["reduceLearningRatePatience"])
    model.setReducedLearningRateFactor(modelParams["reduceLearningRateFactor"])
    model.setUseFinalLSTMLayer(modelParams["useFinalLSTMLayer"])
    model.setOutpuActivation(modelParams["outputActivation"])
    model.setOptimizer(modelParams["optimizer_type"])
    model.setLearningRate(modelParams["learningRate"])
    model.setWindowSize(modelParams["windowSize"])
    model.setPredictionWindowSize(modelParams["windowSize"])
    model.setLabelSmoothing(modelParams["labelSmoothing"])


def configure_tcn(model, modelParams):
    model.setBatchSize(modelParams["batchSize"])
    model.setEpochs(modelParams["epochs"])
    model.setTcnUnits(modelParams["tcnUnits"])
    model.setNumTcnLayers(modelParams["numTcnLayers"])
    model.setDropout(modelParams["dropout"])
    model.setL2Regularization(modelParams["l2Regularization"])
    model.setEarlyStopPatience(modelParams["earlyStopPatience"])
    model.setReduceLearningRatePatience(modelParams["reduceLearningRatePatience"])
    model.setReducedLearningRateFactor(modelParams["reduceLearningRateFactor"])
    model.setLearningRate(modelParams["learningRate"])
    model.setWindowSize(modelParams["windowSize"])
    model.setPredictionWindowSize(modelParams["windowSize"])
    model.setLabelSmoothing(modelParams["labelSmoothing"])
    model.setNumHeads(modelParams["numHeads"])
    model.setKeyDim(modelParams["keyDim"])


def configure_unified_lstm_tcn(model, modelParams):
    model.setBatchSize(modelParams["batchSize"])
    model.setEpochs(modelParams["epochs"])
    model.setLstmUnits(modelParams["lstmUnits"])
    model.setTcnUnits(modelParams["tcnUnits"])
    model.setNumTcnLayers(modelParams["numTcnLayers"])
    model.setDropout(modelParams["dropout"])
    model.setL2Regularization(modelParams["l2Regularization"])
    model.setEarlyStopPatience(modelParams["earlyStopPatience"])
    model.setReduceLearningRatePatience(modelParams["reduceLearningRatePatience"])
    model.setReducedLearningRateFactor(modelParams["reduceLearningRateFactor"])
    model.setLearningRate(modelParams["learningRate"])
    model.setWindowSize(modelParams["windowSize"])
    model.setPredictionWindowSize(modelParams["windowSize"])
    model.setLabelSmoothing(modelParams["labelSmoothing"])
    model.setNumHeads(modelParams["numHeads"])
    model.setKeyDim(modelParams["keyDim"])


def configure_unified_lstm_gru_tcn(model, modelParams):
    configure_unified_lstm_tcn(model, modelParams)
    model.setGruUnits(modelParams["gruUnits"])


def configure_transformer(model, modelParams):
    model.setBatchSize(modelParams["batchSize"])
    model.setEpochs(modelParams["epochs"])
    model.setDModel(modelParams["dModel"])
    model.setNumLayers(modelParams["numLayers"])
    model.setNumHeads(modelParams["numHeads"])
    model.setKeyDim(modelParams["keyDim"])
    model.setFfnFactor(modelParams["ffnFactor"])
    model.setDropout(modelParams["dropout"])
    model.setL2Regularization(modelParams["l2Regularization"])
    model.setEarlyStopPatience(modelParams["earlyStopPatience"])
    model.setReduceLearningRatePatience(modelParams["reduceLearningRatePatience"])
    model.setReducedLearningRateFactor(modelParams["reduceLearningRateFactor"])
    model.setLearningRate(modelParams["learningRate"])
    model.setWindowSize(modelParams["windowSize"])
    model.setPredictionWindowSize(modelParams["windowSize"])
    model.setLabelSmoothing(modelParams["labelSmoothing"])


def configure_gnn(model, modelParams):
    # No setNumHeads/setKeyDim here - GNNModel has no attention block, its
    # architecture knobs are the GCN stack + co-occurrence graph decay.
    model.setBatchSize(modelParams["batchSize"])
    model.setEpochs(modelParams["epochs"])
    model.setGcnUnits(modelParams["gcnUnits"])
    model.setNumGcnLayers(modelParams["numGcnLayers"])
    model.setEmbeddingDim(modelParams["embeddingDim"])
    model.setDecay(modelParams["decay"])
    model.setDropout(modelParams["dropout"])
    model.setL2Regularization(modelParams["l2Regularization"])
    model.setEarlyStopPatience(modelParams["earlyStopPatience"])
    model.setReduceLearningRatePatience(modelParams["reduceLearningRatePatience"])
    model.setReducedLearningRateFactor(modelParams["reduceLearningRateFactor"])
    model.setLearningRate(modelParams["learningRate"])
    model.setWindowSize(modelParams["windowSize"])
    model.setPredictionWindowSize(modelParams["windowSize"])
    model.setLabelSmoothing(modelParams["labelSmoothing"])


def configure_autoencoder(model, modelParams):
    # No setNumHeads/setKeyDim here either - the autoencoder is a pure
    # Conv1D encoder / Dense decoder around the latent bottleneck.
    model.setBatchSize(modelParams["batchSize"])
    model.setEpochs(modelParams["epochs"])
    model.setLatentDim(modelParams["latentDim"])
    model.setEncoderUnits(modelParams["encoderUnits"])
    model.setNumEncoderLayers(modelParams["numEncoderLayers"])
    model.setDropout(modelParams["dropout"])
    model.setL2Regularization(modelParams["l2Regularization"])
    model.setEarlyStopPatience(modelParams["earlyStopPatience"])
    model.setReduceLearningRatePatience(modelParams["reduceLearningRatePatience"])
    model.setReducedLearningRateFactor(modelParams["reduceLearningRateFactor"])
    model.setLearningRate(modelParams["learningRate"])
    model.setWindowSize(modelParams["windowSize"])
    model.setPredictionWindowSize(modelParams["windowSize"])
    model.setLabelSmoothing(modelParams["labelSmoothing"])


# Maps model_type -> (module-level instance, its own configure(model,
# modelParams) function). Replaces the old `modelToUse = tcn if "lstm_model"
# not in model_type else lstm` + a single hardcoded LSTM-shaped setter block
# that would crash immediately on any non-LSTM model_type (TCNModel has no
# setNumberOfLSTMLayers etc.) - every dataset entry happened to be
# "lstm_model" so this never actually fired, but adding new model types
# safely needs each one calling only the setters it actually has.
MODEL_REGISTRY = {
    "lstm_model": {"instance": lstm, "configure": configure_lstm},
    "tcn_model": {"instance": tcn, "configure": configure_tcn},
    "unified_lstm_tcn_model": {"instance": unifiedLstmTcn, "configure": configure_unified_lstm_tcn},
    "unified_lstm_gru_tcn_model": {"instance": unifiedLstmGruTcn, "configure": configure_unified_lstm_gru_tcn},
    "transformer_model": {"instance": transformer, "configure": configure_transformer},
    "gnn_model": {"instance": gnn, "configure": configure_gnn},
    "autoencoder_model": {"instance": autoencoderAnomaly, "configure": configure_autoencoder},
}


def suggest_tcn_params(trial, prefix):
    """
    Dedicated (not suggest_fused_params) Optuna param space for the
    standalone TCN model - suggest_fused_params always includes an
    "lstmUnits" knob for the Unified models' LSTM branch, which TCN doesn't
    have (configure_tcn never reads it), so reusing it here would tune a
    meaningless hyperparameter and log a dead key into bestParams_<game>.json.
    """
    return {
        "batchSize": trial.suggest_categorical(f"{prefix}_batchSize", [4, 8, 16]),
        "epochs": trial.suggest_categorical(f"{prefix}_epochs", [1000]),
        "tcnUnits": trial.suggest_categorical(f"{prefix}_tcnUnits", [16, 32, 64, 128]),
        "numTcnLayers": trial.suggest_int(f"{prefix}_numTcnLayers", 1, 3),
        "dropout": trial.suggest_float(f"{prefix}_dropout", 0.1, 0.5, step=0.1),
        "l2Regularization": trial.suggest_float(f"{prefix}_l2Regularization", 0.0001, 0.01, step=0.0001),
        "learningRate": trial.suggest_float(f"{prefix}_learningRate", 0.00001, 0.001, step=0.00001),
        "earlyStopPatience": trial.suggest_int(f"{prefix}_earlyStopPatience", 10, 100, step=10),
        "reduceLearningRatePatience": trial.suggest_int(f"{prefix}_reduceLearningRatePatience", 10, 100, step=10),
        "reduceLearningRateFactor": trial.suggest_float(f"{prefix}_reduceLearningRateFactor", 0.1, 0.9, step=0.1),
        "windowSize": trial.suggest_int(f"{prefix}_windowSize", 2, 20, step=2),
        "labelSmoothing": trial.suggest_float(f"{prefix}_labelSmoothing", 0.01, 0.1, step=0.01),
        "numHeads": trial.suggest_categorical(f"{prefix}_numHeads", [2, 4, 8]),
        "keyDim": trial.suggest_categorical(f"{prefix}_keyDim", [16, 32, 64]),
        "yearsOfHistory": trial.suggest_categorical(f"{prefix}_yearsOfHistory", [10]),
    }


def suggest_fused_params(trial, prefix, include_gru):
    """
    Shared Optuna param space for UnifiedLstmTcn/UnifiedLstmGruTcn - every
    suggest name is prefixed (see MODEL_PARAM_PREFIX) so each model type's
    tuned values land under their own bestParams_<game>.json keys instead of
    clobbering LSTM's (or each other's).
    """
    params = {
        "batchSize": trial.suggest_categorical(f"{prefix}_batchSize", [4, 8, 16]),
        "epochs": trial.suggest_categorical(f"{prefix}_epochs", [1000]),
        "lstmUnits": trial.suggest_categorical(f"{prefix}_lstmUnits", [16, 32, 64, 128]),
        "tcnUnits": trial.suggest_categorical(f"{prefix}_tcnUnits", [16, 32, 64, 128]),
        "numTcnLayers": trial.suggest_int(f"{prefix}_numTcnLayers", 1, 3),
        "dropout": trial.suggest_float(f"{prefix}_dropout", 0.1, 0.5, step=0.1),
        "l2Regularization": trial.suggest_float(f"{prefix}_l2Regularization", 0.0001, 0.01, step=0.0001),
        "learningRate": trial.suggest_float(f"{prefix}_learningRate", 0.00001, 0.001, step=0.00001),
        "earlyStopPatience": trial.suggest_int(f"{prefix}_earlyStopPatience", 10, 100, step=10),
        "reduceLearningRatePatience": trial.suggest_int(f"{prefix}_reduceLearningRatePatience", 10, 100, step=10),
        "reduceLearningRateFactor": trial.suggest_float(f"{prefix}_reduceLearningRateFactor", 0.1, 0.9, step=0.1),
        "windowSize": trial.suggest_int(f"{prefix}_windowSize", 2, 20, step=2),
        "labelSmoothing": trial.suggest_float(f"{prefix}_labelSmoothing", 0.01, 0.1, step=0.01),
        "numHeads": trial.suggest_categorical(f"{prefix}_numHeads", [2, 4, 8]),
        "keyDim": trial.suggest_categorical(f"{prefix}_keyDim", [16, 32, 64]),
        "yearsOfHistory": trial.suggest_categorical(f"{prefix}_yearsOfHistory", [10]),
    }
    if include_gru:
        params["gruUnits"] = trial.suggest_categorical(f"{prefix}_gruUnits", [16, 32, 64, 128])
    return params


def suggest_transformer_params(trial, prefix):
    """
    Optuna param space for the Transformer model - every suggest name is
    prefixed (see MODEL_PARAM_PREFIX) so the tuned values land under the
    transformer_* keys Predictor.py's runUnifiedDeepLearningModels reads.
    """
    return {
        "batchSize": trial.suggest_categorical(f"{prefix}_batchSize", [4, 8, 16]),
        "epochs": trial.suggest_categorical(f"{prefix}_epochs", [1000]),
        "dModel": trial.suggest_categorical(f"{prefix}_dModel", [32, 64, 128]),
        "numLayers": trial.suggest_int(f"{prefix}_numLayers", 1, 3),
        "numHeads": trial.suggest_categorical(f"{prefix}_numHeads", [2, 4, 8]),
        "keyDim": trial.suggest_categorical(f"{prefix}_keyDim", [16, 32, 64]),
        "ffnFactor": trial.suggest_categorical(f"{prefix}_ffnFactor", [2, 4]),
        "dropout": trial.suggest_float(f"{prefix}_dropout", 0.1, 0.5, step=0.1),
        "l2Regularization": trial.suggest_float(f"{prefix}_l2Regularization", 0.0001, 0.01, step=0.0001),
        "learningRate": trial.suggest_float(f"{prefix}_learningRate", 0.00001, 0.001, step=0.00001),
        "earlyStopPatience": trial.suggest_int(f"{prefix}_earlyStopPatience", 10, 100, step=10),
        "reduceLearningRatePatience": trial.suggest_int(f"{prefix}_reduceLearningRatePatience", 10, 100, step=10),
        "reduceLearningRateFactor": trial.suggest_float(f"{prefix}_reduceLearningRateFactor", 0.1, 0.9, step=0.1),
        # Deliberately longer windows than the other DL models' 2-20 range:
        # self-attention can weight ANY draw in the window equally easily, so
        # long-range context is this model's whole point (see
        # TransformerModel.py's class comment) - a short window would reduce
        # it to an expensive TCN.
        "windowSize": trial.suggest_int(f"{prefix}_windowSize", 10, 40, step=5),
        "labelSmoothing": trial.suggest_float(f"{prefix}_labelSmoothing", 0.01, 0.1, step=0.01),
        "yearsOfHistory": trial.suggest_categorical(f"{prefix}_yearsOfHistory", [10]),
    }


def suggest_gnn_params(trial, prefix):
    """
    Optuna param space for the GNN model - prefixed gnn_* names to match
    Predictor.py's reads. No numHeads/keyDim: the model has no attention
    block (see configure_gnn).
    """
    return {
        "batchSize": trial.suggest_categorical(f"{prefix}_batchSize", [4, 8, 16]),
        "epochs": trial.suggest_categorical(f"{prefix}_epochs", [1000]),
        "gcnUnits": trial.suggest_categorical(f"{prefix}_gcnUnits", [16, 32, 64]),
        "numGcnLayers": trial.suggest_int(f"{prefix}_numGcnLayers", 1, 3),
        "embeddingDim": trial.suggest_categorical(f"{prefix}_embeddingDim", [8, 16, 32]),
        # Recency weight of the co-occurrence adjacency (a pair seen `age`
        # draws ago contributes decay^age): 0.99 keeps roughly the last ~70
        # draws relevant (half-life), 0.9999 makes the graph near-static over
        # years of history - the interesting regimes live between those.
        "decay": trial.suggest_float(f"{prefix}_decay", 0.99, 0.9999),
        "dropout": trial.suggest_float(f"{prefix}_dropout", 0.1, 0.5, step=0.1),
        "l2Regularization": trial.suggest_float(f"{prefix}_l2Regularization", 0.0001, 0.01, step=0.0001),
        "learningRate": trial.suggest_float(f"{prefix}_learningRate", 0.00001, 0.001, step=0.00001),
        "earlyStopPatience": trial.suggest_int(f"{prefix}_earlyStopPatience", 10, 100, step=10),
        "reduceLearningRatePatience": trial.suggest_int(f"{prefix}_reduceLearningRatePatience", 10, 100, step=10),
        "reduceLearningRateFactor": trial.suggest_float(f"{prefix}_reduceLearningRateFactor", 0.1, 0.9, step=0.1),
        "windowSize": trial.suggest_int(f"{prefix}_windowSize", 2, 20, step=2),
        "labelSmoothing": trial.suggest_float(f"{prefix}_labelSmoothing", 0.01, 0.1, step=0.01),
        "yearsOfHistory": trial.suggest_categorical(f"{prefix}_yearsOfHistory", [10]),
    }


def suggest_autoencoder_params(trial, prefix):
    """
    Optuna param space for the Autoencoder model - prefixed autoencoder_*
    names to match Predictor.py's reads. No numHeads/keyDim: pure Conv1D
    encoder / Dense decoder around the latent bottleneck.
    """
    return {
        "batchSize": trial.suggest_categorical(f"{prefix}_batchSize", [4, 8, 16]),
        "epochs": trial.suggest_categorical(f"{prefix}_epochs", [1000]),
        "latentDim": trial.suggest_categorical(f"{prefix}_latentDim", [4, 8, 16, 32]),
        "encoderUnits": trial.suggest_categorical(f"{prefix}_encoderUnits", [16, 32, 64, 128]),
        "numEncoderLayers": trial.suggest_int(f"{prefix}_numEncoderLayers", 1, 3),
        "dropout": trial.suggest_float(f"{prefix}_dropout", 0.1, 0.5, step=0.1),
        "l2Regularization": trial.suggest_float(f"{prefix}_l2Regularization", 0.0001, 0.01, step=0.0001),
        "learningRate": trial.suggest_float(f"{prefix}_learningRate", 0.00001, 0.001, step=0.00001),
        "earlyStopPatience": trial.suggest_int(f"{prefix}_earlyStopPatience", 10, 100, step=10),
        "reduceLearningRatePatience": trial.suggest_int(f"{prefix}_reduceLearningRatePatience", 10, 100, step=10),
        "reduceLearningRateFactor": trial.suggest_float(f"{prefix}_reduceLearningRateFactor", 0.1, 0.9, step=0.1),
        "windowSize": trial.suggest_int(f"{prefix}_windowSize", 2, 20, step=2),
        # Pinned to 0.0, NOT searched: the autoencoder's reconstruction NLL
        # doubles as the anomaly signal (see AutoencoderAnomaly.py's
        # computeAnomalyScores) and label smoothing puts a floor under every
        # class probability, which would bias that NLL floor and mask
        # predictability spikes. Kept as a suggest so the key still lands in
        # bestParams_<game>.json for Predictor.py to read.
        "labelSmoothing": trial.suggest_categorical(f"{prefix}_labelSmoothing", [0.0]),
        "yearsOfHistory": trial.suggest_categorical(f"{prefix}_yearsOfHistory", [10]),
    }


def suggest_lstm_params(trial):
    """The LSTM's search space. Unprefixed - matches Predictor.py's existing LSTM setter block, which reads these same
    bare keys. Don't add a prefix here, it would break that."""
    return {
        "yearsOfHistory": trial.suggest_categorical("yearsOfHistory", [10]),
        "epochs": trial.suggest_categorical("epochs", [1000]),
        "batchSize": trial.suggest_categorical("batchSize", [4]),
        "num_lstm_layers": trial.suggest_categorical("num_lstm_layers", [1]),
        "num_bidirectional_layers": trial.suggest_categorical("num_bidirectional_layers", [1]),
        "lstm_units": trial.suggest_categorical("lstm_units", [16, 32, 64, 128, 256]),
        "bidirectional_lstm_units": trial.suggest_categorical("bidirectional_lstm_units", [16, 32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "l2Regularization": trial.suggest_float("l2Regularization", 0.0001, 0.01, step=0.0001),
        "earlyStopPatience": trial.suggest_int("earlyStopPatience", 10, 100, step=10),
        "reduceLearningRatePatience": trial.suggest_int("reduceLearningRatePatience", 10, 100, step=10),
        "reduceLearningRateFactor": trial.suggest_float("reduceLearningRateFactor", 0.1, 0.9, step=0.1),
        "useFinalLSTMLayer": trial.suggest_categorical("useFinalLSTMLayer", [False]),
        "outputActivation": trial.suggest_categorical("outputActivation", ["softmax"]),  # keep fixed unless needed
        "optimizer_type": trial.suggest_categorical("optimizer_type", ["adam", "rmsprop", "adagrad", "nadam"]), # "sgd", does not work with categorical crossentropy
        "learningRate": trial.suggest_float("learningRate", 0.00001, 0.001, step=0.00001),
        "windowSize": trial.suggest_int("windowSize", 2, 20, step=2),
        "useTopPrediction": trial.suggest_categorical("useTopPrediction", [False]),
        "labelSmoothing": trial.suggest_float("labelSmoothing", 0.01, 0.1, step=0.01)
    }


def suggest_model_params(trial, model_type):
    """
    The settings one trial trains `model_type` with (moved out of the objective on 10 Nov 2026, so RunBacktest.py's
    deep-learning backtests ask for exactly the tuner's settings, answered with the served values).
    """
    if model_type == "lstm_model":
        return suggest_lstm_params(trial)
    if model_type == "tcn_model":
        return suggest_tcn_params(trial, MODEL_PARAM_PREFIX[model_type])
    if model_type == "transformer_model":
        return suggest_transformer_params(trial, MODEL_PARAM_PREFIX[model_type])
    if model_type == "gnn_model":
        return suggest_gnn_params(trial, MODEL_PARAM_PREFIX[model_type])
    if model_type == "autoencoder_model":
        return suggest_autoencoder_params(trial, MODEL_PARAM_PREFIX[model_type])
    return suggest_fused_params(trial, MODEL_PARAM_PREFIX[model_type], include_gru=(model_type == "unified_lstm_gru_tcn_model"))


def is_running():
    """
    Checks if another instance is running based on the lock file.

    The PID written into the lock is verified to still be alive: a lock whose
    owner is gone is stale - left behind by a crashed or killed run (a hung
    2026-08-17 run held the lock for two days and silently blocked every cron
    start after it). A stale lock is removed and treated as not running.
    """
    if not os.path.exists(LOCK_FILE):
        return False
    try:
        with open(LOCK_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # signal 0 = existence check only, nothing is sent
        return True
    except (ValueError, ProcessLookupError):
        print("Removing stale lock file (owner process no longer exists)")
        remove_lock()
        return False
    except PermissionError:
        # Process exists but belongs to another user - definitely running.
        return True

def create_lock():
    """Creates the lock file."""
    try:
        with open(LOCK_FILE, "x") as f:  # "x" mode creates the file, failing if it exists
            f.write(str(os.getpid()))  # Write the PID to the lock file (optional, but helpful for debugging)
        return True
    except FileExistsError:
        return False

def remove_lock():
    """Removes the lock file."""
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass  # It's okay if the lock file doesn't exist


def tuned_keno_subset_sizes(name):
    """
    The hyperopt-tuned use_5..use_10 subset choice for this game, matching
    Predictor.getKenoSubsetSizes - so the profit signal this script optimises
    against is measured on the subset sizes that will actually be played,
    instead of the hardcoded all-six range it used before.

    Falls back to all six sizes when the game has no tuned flags yet (a game
    whose statistical hyperopt hasn't run): returning nothing there would leave
    the DL profit signal permanently zero, which is worse than measuring on
    sizes that may later be disabled.
    """
    bestParams = {}
    bestParamsPath = os.path.join(os.getcwd(), f"bestParams_{name}.json")
    if os.path.exists(bestParamsPath):
        try:
            with open(bestParamsPath, "r") as infile:
                bestParams = json.load(infile)
        except Exception as e:
            print(f"Failed to read {bestParamsPath}, using all Keno subset sizes: ", e)

    if not any(f"use_{size}" in bestParams for size in (5, 6, 7, 8, 9, 10)):
        return list(range(5, 11))

    return [size for size in (5, 6, 7, 8, 9, 10) if bestParams.get(f"use_{size}")]


def print_intro():
    # Generate ASCII art with the text "LSTM"
    ascii_art = text2art("Predictor Hyperopt")
    # Print the introduction and ASCII art
    print("============================================================")
    print("Predictor Hyperopt")
    print("Licence : MIT License")
    print(ascii_art)
    print("Find best parameters for Predictor")

def update_matching_numbers(name, path):
    json_dir = os.path.join(path, "data", "hyperOptCache", name)
    if not os.path.exists(json_dir):
        print(f"Directory does not exist: {json_dir}")
        return

    # Step 1: Get all JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    # Step 2: Sort by date
    def parse_date(filename):
        try:
            name_part = filename.replace(".json", "")
            return datetime.strptime(name_part, "%Y-%m-%d")
        except ValueError:
            return datetime.max  # Skip improperly named files

    sorted_files = sorted(json_files, key=parse_date)

    # Step 3: Iterate through each pair (previous, current)
    for i in range(1, len(sorted_files)):
        prev_file = os.path.join(json_dir, sorted_files[i - 1])
        curr_file = os.path.join(json_dir, sorted_files[i])

        with open(prev_file, "r") as f_prev, open(curr_file, "r") as f_curr:
            prev_json = json.load(f_prev)
            curr_json = json.load(f_curr)

        curr_json["currentPredictionRaw"] = prev_json.get("newPredictionRaw", [])
        curr_json["currentPrediction"] = prev_json.get("newPrediction", [])

        specialColumnCount, realMainCount = matchingSplitArgs(name, curr_json["realResult"])
        best_match = helpers.find_best_matching_prediction(
            curr_json["realResult"], curr_json["currentPrediction"],
            specialColumnCount=specialColumnCount, realMainCount=realMainCount, game=name
        )
        curr_json["matchingNumbers"] = best_match

        # Save updated JSON
        with open(curr_file, "w") as f_curr_out:
            json.dump(curr_json, f_curr_out, indent=2)

    print(f"Updated matching numbers in {len(sorted_files) - 1} files.")


def process_single_history_entry(args):
    (historyIndex, historyEntry, historyData, name, model_type, dataPath, modelPath,
        skipLastColumns, years_back, previousJsonFilePath, path, modelParams, specialColumnCount, trainSeconds) = args

    modelToUse = MODEL_REGISTRY[model_type]["instance"]
    historyDate, historyResult = historyEntry
    jsonFileName = f"{historyDate.year}-{historyDate.month}-{historyDate.day}.json"
    jsonFilePath = os.path.join(path, "data", "hyperOptCache", name, jsonFileName)

    current_json_object = {
        "currentPredictionRaw": [],
        "currentPrediction": [],
        "realResult": historyResult,
        "newPrediction": [],
        "newPredictionRaw": [],
        "matchingNumbers": {},
        "labels": [],
        "numberFrequency": helpers.count_number_frequencies(dataPath)
    }

    if previousJsonFilePath and os.path.exists(previousJsonFilePath):
        with open(previousJsonFilePath, 'r') as openfile:
            previous_json_object = json.load(openfile)
        current_json_object["currentPredictionRaw"] = previous_json_object["newPredictionRaw"]
        current_json_object["currentPrediction"] = previous_json_object["newPrediction"]

    matchSpecialCount, matchRealMainCount = matchingSplitArgs(name, current_json_object["realResult"])
    best_matching_prediction = helpers.find_best_matching_prediction(
        current_json_object["realResult"], current_json_object["currentPrediction"],
        specialColumnCount=matchSpecialCount, realMainCount=matchRealMainCount, game=name)
    current_json_object["matchingNumbers"] = best_matching_prediction

    listOfDecodedPredictions = []
    unique_labels = []


    # Every rebuilt day builds a fresh Keras model; without clearing the
    # session first, the abandoned graphs of all previous days (and trials)
    # stay resident and RSS ratchets up until the 16GB memory cgroup
    # OOM-kills the whole study (observed 2026-08-28 at 16.0GB anon-rss,
    # trial 10 of a transformer study - the same disease Predictor.py's
    # runDeepLearningStepInChild exists for). Safe here: models warm-start
    # from the fingerprinted weights on disk, not from live objects.
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()

    modelToUse.setDataPath(dataPath)
    modelToUse.setModelPath(modelPath)
    modelToUse.setLoadModelWeights(True)
    MODEL_REGISTRY[model_type]["configure"](modelToUse, modelParams)
    # Production's budget per training run (6 Nov 2026, src/DeepLearningTuning.py E): the daily predictor trains
    # every DL row for --dl-model-seconds; None/0 trains unlimited, as before
    if hasattr(modelToUse, "setTrainSeconds"):
        modelToUse.setTrainSeconds(trainSeconds or None)

    # The day's tuner value is this run's learning check (src/LearningCheck.py excessLoss, 7 Nov 2026); never the
    # previous day's when this one has none
    modelToUse.last_learning_check = None

    # Perform training
    latest_raw_predictions, unique_labels = modelToUse.run(
        name, skipLastColumns, skipRows=len(historyData)-historyIndex, years_back=years_back, strict_val=False,
        specialColumnCount=specialColumnCount)

    # Set by run() on every model in MODEL_REGISTRY: the best val_loss across epochs (on the record only - it
    # carries the label smoothing and the L2 penalty, both tuned knobs) and the learning check, whose
    # smoothing-free excess loss over the training frequencies is the score (src/DeepLearningTuning.day_score).
    val_loss = getattr(modelToUse, "last_val_loss", float("inf"))
    check = getattr(modelToUse, "last_learning_check", None)

    predictedSequence = latest_raw_predictions.tolist()
    # TCN/Transformer/GNN/Autoencoder return unique_labels as the raw numpy
    # array from load_data (only LSTM/Unified* convert to a plain list
    # themselves) and json.dump below can't serialize ndarray - every such
    # trial would fail before scoring. Convert WITHOUT sorting: load_data's
    # ordering is the one-hot encoder's category order, i.e. the class-index
    # -> label decode mapping deepLearningMethod indexes into.
    if isinstance(unique_labels, np.ndarray):
        unique_labels = unique_labels.tolist()
    current_json_object["newPredictionRaw"] = predictedSequence
    listOfDecodedPredictions = deepLearningMethod(
        listOfDecodedPredictions, predictedSequence, unique_labels, 1, name, historyResult, jsonFilePath, modelParams,
        modelDisplayName=MODEL_DISPLAY_NAMES.get(model_type, "LSTM Base Model"))


    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)


    current_json_object["newPrediction"] = listOfDecodedPredictions
    current_json_object["labels"] = unique_labels

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)

    return jsonFilePath, val_loss, check



# Wall-clock budget per trial (CLI --trial-timeout; module level so the
# spawned children, which re-import this file, see a value too).
TRIAL_TIMEOUT_SECONDS = 7200
# How often the parent looks for a finished day while the child trains
PROGRESS_POLL_SECONDS = 15


def runPredictInChild(trial, progressPath, *args, onDay=None, **kwargs):
    """
    Runs one trial's predict() (the daysToRebuild training loop) in a
    one-shot spawned child process, for the same two hard-learned reasons as
    Predictor.py's runDeepLearningStepInChild:

    - Containment: the memory-cgroup OOM killer SIGKILLs the process when it
      outgrows the container's 16GB. In-process, that kills the WHOLE study
      mid-run (observed 2026-08-28: a pick3 transformer study died at 16GB
      anon-rss in trial 10, ~80 accumulated day-models in). In a child, the
      kill surfaces as BrokenProcessPool, which study.optimize's
      catch=(Exception,) records as one FAILED trial before moving on.
    - Memory hygiene: a fresh child per trial releases every TF allocation
      back to the OS at trial end, so RSS can't ratchet across trials.

    Costs one TF import + CUDA init per trial (~15-30s) - noise next to the
    minutes of training a trial performs.

    While the child trains, the parent reads each finished day's excess loss
    over the counts (7 Nov 2026; the validation loss until then) from progressPath (6 Nov 2026, src/DeepLearningTuning.py C, D): one
    progress line per day, and the running score reported to Optuna, whose
    pruner may stop the trial - the child is then killed and the trial
    recorded as pruned. trial may be a FixedTrial (the gate's references:
    never pruned) or None. onDay(days done), when given, is called at the start
    and after each finished day (the Hyperopts page's progress within a trial,
    11 Nov 2026); it must not raise.
    """
    try:
        os.remove(progressPath)
    except OSError:
        pass
    global CURRENT_EXECUTOR
    executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
    CURRENT_EXECUTOR = executor

    def killChild():
        for child in list(executor._processes.values()):
            child.kill()

    reported = 0
    started = time.time()
    if onDay:
        onDay(0)
    try:
        future = executor.submit(predict, *args, progressPath=progressPath, **kwargs)
        while True:
            result, done = None, False
            try:
                result, done = future.result(timeout=PROGRESS_POLL_SECONDS), True
            except FuturesTimeoutError:
                pass
            losses = dlt.read_progress(progressPath)
            while reported < len(losses):
                value = dlt.running_value(losses[:reported + 1])
                loss = losses[reported]
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] day {reported + 1}/{kwargs.get('daysToRebuild')}: "
                      f"excess loss over the counts {'n/a' if loss is None else f'{loss:+.4f}'}, running score {value:+.4f} "
                      f"({(time.time() - started) / 60:.1f} min)")
                if trial is not None:
                    trial.report(value, reported)
                    if not done and trial.should_prune():
                        killChild()
                        print(f"Pruned after day {reported + 1}: below the median of the earlier trials at this day")
                        raise optuna.TrialPruned()
                reported += 1
                if onDay:
                    onDay(reported)
            if done:
                # a trial that finished between two looks is scored in full, not pruned on its last day
                return result
            if time.time() - started > TRIAL_TIMEOUT_SECONDS:
                # A hung or runaway trial (a healthy-looking study was once killed by
                # hand after hours of silence) must not block the study: kill the
                # child - the pool has exactly this one process - and record the
                # trial as pruned, not as a score.
                killChild()
                print(f"Trial exceeded the {TRIAL_TIMEOUT_SECONDS}s budget - child killed, trial pruned")
                raise optuna.TrialPruned()
    finally:
        CURRENT_EXECUTOR = None
        executor.shutdown(wait=True, cancel_futures=True)


class ReferenceTrial(optuna.trial.FixedTrial):
    """
    The gate's references (the served parameters, Predictor.py's defaults) run through the objective like a trial.
    A served or default value outside a categorical's choices - LSTMModel's useFinalLSTMLayer True against the
    search space's [False] - makes a FixedTrial raise; here it is used as is, since scoring what is served is the
    point (TuningGate.evaluate_reference does the same for numeric ranges, which only warn).
    """
    def suggest_categorical(self, name, choices):
        if name in self._params and self._params[name] not in choices:
            return self._params[name]
        return super().suggest_categorical(name, choices)


# The training child's pool while a trial runs, for the interrupt handler below
CURRENT_EXECUTOR = None


def onInterrupt(signum, frame):
    """
    Ctrl+C or kill: the training child dies with the run (runPredictInChild's finally would otherwise wait for it),
    the lock goes, and the run's marker stays on disk - the same command resumes (src/DeepLearningTuning.py A).
    """
    if CURRENT_EXECUTOR is not None:
        for child in list(CURRENT_EXECUTOR._processes.values()):
            try:
                child.kill()
            except Exception:
                pass
    remove_lock()
    print("\nInterrupted - the finished trials are in db.sqlite3; run the same command again to resume this run "
          "(--fresh starts over)")
    sys.exit(130)


def clearFolder(folderPath):
    print("Clearing Folder: ", folderPath)
    try:
        for filename in os.listdir(folderPath):
            file_path = os.path.join(folderPath, filename)
        
            if os.path.isfile(file_path):
                os.remove(file_path)  
                #print(f"Deleted file: {filename}")
    except Exception as e:
        pass

def predict(name, model_type ,dataPath, modelPath, file, skipLastColumns=0, maxRows=0, years_back=None, daysToRebuild=31, modelParams={}, specialColumnCount=0,
            trainSeconds=None, progressPath=None):
    """
        Predicts the next sequence of numbers for a given dataset or rebuild the prediction for the last n months

        @param name: The name of the dataset
        @param model_type: The type of model to use
        @param dataPath: The path to the data
        @param modelPath: The path to the model
        @param file: The file to download
        @param skipLastColumns: The number of columns to skip
        @param maxRows: The maximum number of rows to use
        @param years_back: The number of years to go back
        @param daysToRebuild: The number of days to rebuild
        @param ai: To use ai tech to do predictions
        @param trainSeconds: the wall-clock budget of each training run (None = unlimited)
        @param progressPath: where each rebuilt day's excess loss over the counts is written as it is trained, for
            the parent's progress lines and pruning (src/DeepLearningTuning.py C, D)

        Returns (profit, avg_val_loss, dayExcess, dayLosses, verdicts, dayVariety), one entry per rebuilt day in the
        lists: the learning check's excessLoss (None without one), the val_loss (None for a non-finite one) and the
        learning check's verdict (None without one); dayVariety {"sameTickets": the check's sameTicket per day,
        "tickets": each day's newPrediction} (11 Nov 2026: the gate's variety rule, src/DeepLearningTuning.variety).
    """

    modelToUse = MODEL_REGISTRY[model_type]["instance"]
    modelToUse.setDataPath(dataPath)

    # Get the latest result out of the latest data so we can use it to check the previous prediction
    latestEntry, previousEntry = helpers.getLatestPrediction(dataPath)
    if latestEntry is not None:
        latestDate, latestResult = latestEntry

        folderPath = os.path.join(path, "data", "hyperOptCache", name)

        jsonFileName = f"{latestDate.year}-{latestDate.month}-{latestDate.day}.json"
        #print(jsonFileName, ":", latestResult)
        jsonFilePath = os.path.join(folderPath, jsonFileName)

        # Check if folder exists
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        else:
            # Clear the hyperOptCache
            clearFolder(folderPath)


        # Compare the latest result with the previous new prediction
        if not os.path.exists(jsonFilePath):

            print(f"Hyperopt -> Recreating {daysToRebuild} days of history")

            # Check if there is not a gap or so
            historyData = helpers.getLatestPrediction(dataPath, dateRange=daysToRebuild)
            print("History data: ", historyData)

            dateOffset = 0 # index of list entry

            print("Date to start from: ", historyData[dateOffset])

            previousJsonFilePath = ""

            # Search for existing history
            for index, historyEntry in enumerate(historyData):
                entryDate = historyEntry[0]
                entryResult = historyEntry[1]
                jsonFileName = f"{entryDate.year}-{entryDate.month}-{entryDate.day}.json"
                #print(jsonFileName, ":", entryResult)
                jsonFilePath = os.path.join(path, "data", "hyperOptCache", name, jsonFileName)
                #print("Does file exist: ", os.path.exists(jsonFilePath))
                if os.path.exists(jsonFilePath):
                    dateOffset = index
                    previousJsonFilePath = jsonFilePath
                    break
            
            # Remove all elements starting from dateOffset index
            #print("Date offset: ", dateOffset)
            historyData = historyData[dateOffset:]  # Keep elements after dateOffset because newer elements comes after the dateOffset index                
            #print("History to rebuild: ", historyData)

            argsList = [
                (historyIndex, historyEntry, historyData, name, model_type, dataPath,
                modelPath, skipLastColumns, years_back, previousJsonFilePath, path, modelParams, specialColumnCount,
                trainSeconds)
                for historyIndex, historyEntry in enumerate(historyData)
            ]


            val_losses = []
            dayLosses = []
            dayExcess = []
            verdicts = []
            sameTickets = []
            tickets = []
            for args in argsList:
                dayFile, val_loss, check = process_single_history_entry(args)
                if np.isfinite(val_loss):
                    val_losses.append(val_loss)
                dayLosses.append(float(val_loss) if np.isfinite(val_loss) else None)
                dayExcess.append(dlt.day_excess(check))
                verdicts.append(check.get("verdict") if isinstance(check, dict) else None)
                sameTickets.append(check.get("sameTicket") if isinstance(check, dict) else None)
                try:
                    with open(dayFile) as dayHandle:
                        tickets.append(json.load(dayHandle).get("newPrediction"))
                except Exception:   # the variety record only: a day file that cannot be read counts as no ticket
                    tickets.append(None)
                dlt.write_progress(progressPath, dayExcess)

            print("Finished rebuild of history entries.")

            # Find the matching numbers
            update_matching_numbers(name=name, path=path)

            # Calculate Profit - dispatched on the game name inside
            # calculate_profit (keno subsets, pick3 straight/box/pair, Joker+
            # left/right runs + sign on the 7-value row; main-pool hits for
            # the games without a payout table).
            profit =  helpers.calculate_profit(name=name, path=path)

            # avg_val_loss is backed by every retrain's held-out validation
            # windows (hundreds of samples) - far more reliable than profit,
            # which is backed by only `daysToRebuild` real draws. See
            # objective() for how the two are combined.
            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")

            return profit, avg_val_loss, dayExcess, dayLosses, verdicts, {"sameTickets": sameTickets, "tickets": tickets}
        else:
            print("Prediction already made")
    else:
        print("Did not found entries")


def deepLearningMethod(listOfDecodedPredictions, newPredictionRaw, labels, nOfPredictions, name, historyResult, jsonFilePath, modelParams, modelDisplayName="LSTM Base Model"):

    jsonDirPath = os.path.join(path, "data", "hyperOptCache", name)
    num_classes = len(labels)
    numbersLength = len(historyResult)

    nthPredictions = {
        "name": modelDisplayName,
        "predictions": []
    }
    # Decode prediction with nth highest probability
    predicted_indices = np.argmax(newPredictionRaw, axis=-1)
    predicted_digits = [int(labels[i]) for i in predicted_indices]

    print("Prediction: ", predicted_digits)
    nthPredictions["predictions"].append(predicted_digits)

    # Keno is the only game with playable sub-selections (5-10 numbers out of
    # the full 20-number ticket) - without this, calculate_profit's keno
    # branch (which only scores predictions of length 5-10, see
    # Helpers.keno_ticket_profit) never had anything to score for the DL
    # models, so their Keno hyperopt profit signal was always zero. Mirrors
    # the statistical models' score_numbers() + generate_best_subset, using
    # the DL prediction's per-number softmax probability as the score.
    if "keno" in name:
        try:
            number_scores = helpers.score_numbers_from_prediction(newPredictionRaw, labels)
            for subset_size in tuned_keno_subset_sizes(name):
                subset = helpers.generate_subset_from_scores(number_scores, predicted_digits, subset_size, mode="top")
                nthPredictions["predictions"].append(subset)
        except Exception as e:
            print("Failed to generate keno subsets: ", e)

    # useTopPrediction is an LSTM-only knob (see suggest_fused_params, which
    # doesn't set it) - .get() with a default so the other model types don't
    # crash here.
    if modelParams.get("useTopPrediction", False):
        try:
            predicted_digits = np.argmax(newPredictionRaw, axis=-1)
            top3_indices = np.argsort(newPredictionRaw, axis=-1)[:, -3:][:, ::-1]
            nthPredictions["predictions"].append(top3_indices[0].tolist())
        except Exception as e:
            print("Failed to parse the top prediction: ", e)

    listOfDecodedPredictions.append(nthPredictions)

    return listOfDecodedPredictions

if __name__ == "__main__":

    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)

    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)

    try:
        helpers.git_pull()
    except Exception as e:
        print("Failed to get latest changes")

    parser = argparse.ArgumentParser(
        prog='Sequence Predictor',
        description='Tries to predict a sequence of numbers',
        epilog='Check it out'
    )

    parser.add_argument('-r', '--rebuild_history', type=helpers.str2bool, default=False)
    parser.add_argument('-d', '--days', type=int, default=dlt.DAYS,
                        help='Rebuilt days per trial (default %(default)s since 7 Nov 2026, 8 until then: the '
                             'warm-started days barely differ, so more days mostly cost time)')
    parser.add_argument('-t', '--trials', type=int, default=15,
                        help='Trials per study in this run; a resumed run runs only the ones a study still lacks')
    parser.add_argument(
        '--trial-timeout', type=int, default=7200,
        help='Wall-clock budget in seconds per trial (the spawned training child is killed and the '
             'trial recorded as pruned). Default 2 hours - the unified LSTM/TCN trials are the long ones.')
    parser.add_argument('-s', '--save', type=helpers.str2bool, default=True,
                        help='Commit and push each study\'s bestParams when it ends, and the rest at the end of the run')
    parser.add_argument(
        '--train-seconds', type=int, default=dlt.TRAIN_SECONDS,
        help='Wall-clock budget of every training run, in seconds - Predictor.py\'s --dl-model-seconds, so the '
             'tuner scores parameters the way production trains them (6 Nov 2026). 0 = unlimited (up to 1,000 '
             'epochs, as before).')
    parser.add_argument(
        '--prune-after-days', type=int, default=dlt.PRUNE_AFTER_DAYS,
        help='Stop a trial whose running score after this many rebuilt days is below the median of the earlier '
             'trials at that day (Optuna MedianPruner; the first %d trials of a study are never pruned). '
             '0 = no pruning.' % dlt.PRUNE_STARTUP_TRIALS)
    parser.add_argument(
        '--gate', type=helpers.str2bool, default=True,
        help='Champion/challenger (src/TuningGate.py): this run\'s best trial is written only if it beats the '
             'served parameters and Predictor.py\'s defaults, re-scored on the same days. false = write it anyway.')
    parser.add_argument(
        '--gate-se-margin', type=float, default=dlt.GATE_SE_MARGIN,
        help='How many standard errors of its own score the challenger must win by (default %(default)s).')
    parser.add_argument(
        '--result-file', type=str, default=None,
        help='Also write each study\'s progress and the gate\'s verdict - with the parameters it would write and '
             'this run\'s best trial - to this JSON file (src/DeepLearningTuning.write_result). The Hyperopts page '
             'runs the tuner this way on the GPU worker with -s false: it shows the file and Apply writes on the '
             'main server (11 Nov 2026).')
    parser.add_argument(
        '--fresh', action='store_true',
        help='Start a new run even if an unfinished one with the same settings is on disk (default: resume it).')
    parser.add_argument(
        '-g', '--games',
        type=str,
        default="euromillions,lotto,eurodreams,jokerplus,keno,pick3,vikinglotto",
        help='Comma-separated list of games, e.g. "euromillions,lotto,..."'
    )
    parser.add_argument(
        '-m', '--models',
        type=str,
        default=",".join(MODEL_REGISTRY),
        help='Comma-separated list of model types to tune, e.g. '
             '"transformer_model,gnn_model". Valid: ' + ", ".join(MODEL_REGISTRY) + '. '
             'The legacy lstm/tcn/unified_* types are the expensive ones (heavy '
             'trainings, hours per study) - the transformer/gnn/autoencoder types '
             'are an order of magnitude cheaper and can be tuned on their own '
             'without paying for the heavy studies.'
    )
    args = parser.parse_args()

    print_intro()

    current_year = datetime.now().year
    print("Current Year:", current_year)

    daysToRebuild = int(args.days)
    rebuildHistory = bool(args.rebuild_history)
    n_trials = int(args.trials)
    TRIAL_TIMEOUT_SECONDS = int(args.trial_timeout)
    pushToGit = bool(args.save)
    trainSeconds = max(0, int(args.train_seconds))
    pruneAfterDays = max(0, int(args.prune_after_days))
    gateEnabled = bool(args.gate)
    gateSeMargin = float(args.gate_se_margin)
    resultFile = args.result_file or None

    print("Push to git: ", pushToGit)
    print("Running ", n_trials, "trials")
    print(f"Training budget: {trainSeconds or 'unlimited'} s per training run, pruning "
          f"{f'after day {pruneAfterDays}' if pruneAfterDays else 'off'}, gate "
          f"{f'on ({gateSeMargin:g} SE)' if gateEnabled else 'OFF'}")
    print(f"Lockbox: {dlt.LOCKBOX_NOTE}")

    # Convert the comma-separated string into a clean list
    games = [g.strip() for g in args.games.split(',') if g.strip()]

    print("Selected games:", games)


    # Every game gets its own independent study per model type, so
    # UnifiedLstmTcn Model / UnifiedLstmGruTcn Model / Transformer Model /
    # GNN Model / Autoencoder Model get tuned (and tracked) separately from
    # LSTM Base Model rather than sharing one set of params - same reasoning
    # as MODEL_PARAM_PREFIX above. Which types actually run comes from
    # --models (default: all of them).
    dl_model_types = [m.strip() for m in args.models.split(',') if m.strip()]
    unknown_model_types = [m for m in dl_model_types if m not in MODEL_REGISTRY]
    if unknown_model_types:
        print(f"Unknown model type(s) in --models: {', '.join(unknown_model_types)}")
        print(f"Valid model types: {', '.join(MODEL_REGISTRY)}")
        # The lock was already taken above - leaving it behind on this early
        # exit would block every later (cron) run until the stale-lock check
        # kicks in.
        remove_lock()
        sys.exit(1)
    # SPECIAL_COLUMN_COUNTS moved to module level (see the definition near
    # MODEL_DISPLAY_NAMES) so the match-scoring helpers can use it too. Lotto's
    # bonus number isn't modeled at all - it's simply dropped via
    # skip_last_columns (was incorrectly 0 here at one point, unlike
    # Predictor.py's own dataset list, which already drops it).
    datasets = [
        # (dataset_name, model_type, skip_last_columns, special_column_count)
        (dataset_name, model_type, skip_last_columns, SPECIAL_COLUMN_COUNTS.get(dataset_name, 0))
        for dataset_name, skip_last_columns in [
            ("euromillions", 0),
            ("lotto", 1),
            ("eurodreams", 0),
            # skip 0, not 1: the trailing zodiac column is Joker+'s special
            # column (SPECIAL_COLUMN_COUNTS) and must stay present so the DL
            # models' special head can model it - dropping it would turn the
            # game into six plain digits with no sign to predict.
            ("jokerplus", 0),
            ("keno", 0),
            ("pick3", 0),
            ("vikinglotto", 0),
        ]
        for model_type in dl_model_types
    ]

    # The run's marker (src/DeepLearningTuning.py A): the same command after a kill resumes it
    runFolder = os.path.join(path, "data", "hyperOptCache")
    runMarker, resumed = dlt.open_run(runFolder, dlt.run_settings(
        games, dl_model_types, n_trials, daysToRebuild,
        trainSeconds, gateEnabled), fresh=args.fresh)
    studyKeys = [dlt.study_name(d[0], d[1]) for d in datasets if d[0] in games]
    signal.signal(signal.SIGINT, onInterrupt)
    signal.signal(signal.SIGTERM, onInterrupt)

    for dataset_name, model_type, skip_last_columns, special_column_count in datasets:
        if dataset_name in games:
            try:
                print(f"\n{dataset_name.capitalize()}")
                # the score's own studies (src/DeepLearningTuning.STUDY_SUFFIX, 7 Nov 2026): the old ones hold
                # minus-val_loss values, whose scale would mislead the sampler and the pruner's median
                studyKey = dlt.study_name(dataset_name, model_type)
                if studyKey in runMarker["done"]:
                    print(f"{studyKey}: finished earlier in this run - skipped")
                    continue
                progressPath = os.path.join(runFolder, f"progress-{studyKey}.json")
                dlt.write_result(resultFile, studyKey, game=dataset_name, model=model_type,
                                 row=MODEL_DISPLAY_NAMES[model_type], status="running", wanted=n_trials,
                                 days=daysToRebuild, trainSeconds=trainSeconds or None, gate=gateEnabled,
                                 startedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                modelPath = os.path.join(path, "data", "hyperOptCache", "models", model_type)
                dataPath = os.path.join(path, "data", "trainingData", dataset_name)
                file = f"{dataset_name}-gamedata-NL-{current_year}.csv"

                # The new model types' hyperOptCache/models folders don't
                # exist on a fresh checkout, and GNNModel.run (unlike the
                # Transformer/Autoencoder run()s) doesn't create its own -
                # ModelCheckpoint's first save mid-training would crash the
                # trial otherwise.
                os.makedirs(modelPath, exist_ok=True)

                # To prevent the hyperopt failing for loading an old model
                clearFolder(os.path.join(path, "data", "hyperOptCache", "models", model_type))

                kwargs_wget = {
                    "folder": dataPath,
                    "file": file
                }

                if os.path.exists(os.path.join(dataPath, file)):
                    print("Starting data fetcher")
                    filePath = os.path.join(dataPath, file)
                    dataFetcher.startDate = dataFetcher.calculate_start_date(filePath)
                    gameName = ""
                    if "euromillions" in dataset_name:
                        gameName = "Euro+Millions"
                    if "lotto" in dataset_name:
                        gameName = "Lotto"
                    if "eurodreams" in dataset_name:
                        gameName = "EuroDreams"
                    if "jokerplus" in dataset_name:
                        gameName = "Joker%2B"
                    if "keno" in dataset_name:
                        gameName = "Keno"
                    if "pick3" in dataset_name:
                        gameName = "Pick3"
                    if "vikinglotto" in dataset_name:
                        gameName = "Viking+Lotto"
                    # A transient API failure here must not cost the whole
                    # model-type study (observed 2026-08-30: the autoencoder
                    # study for pick3 never ran because this fetch - the
                    # THIRD one of the same run, the CSV already fresh from
                    # the transformer/gnn entries seconds earlier - threw
                    # into the per-dataset except). The CSV on disk is a
                    # perfectly good fallback: at worst it misses the very
                    # newest draw.
                    try:
                        dataFetcher.getLatestData(gameName, filePath)
                    except Exception as e:
                        print(f"Data fetch failed for {dataset_name} - continuing with the existing CSV: {e}")
                    #os.remove(os.path.join(dataPath, file))
                #command.run("wget -P {folder} https://prdlnboppreportsst.blob.core.windows.net/legal-reports/{file}".format(**kwargs_wget), verbose=False)


                # The page's progress within a trial (11 Nov 2026): which trial or gate reference trains now, and how
                # many of its days are done. Only with --result-file; a failed write never stops the study.
                currentRun = {"role": "trial"}

                def onDay(daysDone, trial):
                    if not resultFile:
                        return
                    try:
                        dlt.write_result(resultFile, studyKey, current={
                            "role": currentRun["role"],
                            "trial": trial.number if currentRun["role"] == "trial" else None,
                            "day": daysDone, "days": daysToRebuild,
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    except Exception as e:
                        print(f"{studyKey}: could not write the result file ({e})")

                def objective(trial):
                    numOfRepeats = 1 # To average out the rusults before continueing to the next result
                    totalProfit = 0
                    results = [] # Intermediate results

                    modelParams = suggest_model_params(trial, model_type)

                    # Optuna only logs when a trial FINISHES, and a healthy
                    # trial can be silent for minutes at a stretch (XLA
                    # recompiles each rebuilt day's first epoch in the fresh
                    # child, and SelectiveProgbarLogger prints only every 50
                    # epochs) - without a start line, a slow trial is
                    # indistinguishable from a hang (a healthy 2026-08-30
                    # study was manually killed for exactly that reason).
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Trial {trial.number} started "
                          f"({dataset_name}-{model_type}) - long quiet stretches are normal, "
                          f"progress prints every 50 epochs per rebuilt day")

                    for _ in range(numOfRepeats):
                        result = runPredictInChild(trial, progressPath, f"{dataset_name}", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=modelParams["yearsOfHistory"], daysToRebuild=daysToRebuild, modelParams=modelParams, specialColumnCount=special_column_count, trainSeconds=trainSeconds,
                                                   onDay=lambda n, trial=trial: onDay(n, trial))
                        if result is not None:
                            results.append(result)

                    clearFolder(os.path.join(path, "data", "hyperOptCache", "models", model_type))

                    if not results:
                        # No prediction could be made this trial (e.g. no new
                        # history to rebuild) - fail the trial rather than
                        # silently scoring it as a tie with real trials.
                        return -float("inf")

                    # The score (src/DeepLearningTuning.day_score, 7 Nov 2026): minus the mean over the
                    # rebuilt days of the learning check's excess loss - the row's plain cross-entropy on its
                    # validation windows above the training frequencies', with no label smoothing and no L2
                    # on either, so the tuned smoothing/L2 can no longer move the yardstick. 0 = as good as
                    # serving the counts. The hits (calculate_profit over the days) are on the record only:
                    # 0.05 a hit was ~100 standard errors of the old score (the 6 Nov lotto run).
                    profit, _, dayExcess, dayLosses, verdicts, dayVariety = results[0]
                    tuning = dlt.day_score(dayExcess, hits=profit, val_losses=dayLosses, verdicts=verdicts,
                                           same_tickets=dayVariety["sameTickets"], tickets=dayVariety["tickets"])
                    trial.set_user_attr("tuning", attrs_for_trial(tuning))
                    return tuning["score"]

                # Write best params to json
                jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
                existingData = {}
                if os.path.exists(jsonBestParamsFilePath):
                    with open(jsonBestParamsFilePath, "r") as infile:
                        existingData = json.load(infile)

                # Create an Optuna study object. The pruner (D) is not stored with
                # the study: every run sets it.
                study = optuna.create_study(
                    direction='maximize',
                    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                    study_name=studyKey,
                    load_if_exists=True,
                    pruner=(optuna.pruners.MedianPruner(n_startup_trials=dlt.PRUNE_STARTUP_TRIALS,
                                                        n_warmup_steps=pruneAfterDays - 1)
                            if pruneAfterDays else optuna.pruners.NopPruner())
                )
                # Trials left RUNNING by a killed run (Ctrl+C, OOM, reboot).
                fail_stale_running_trials(study)

                # A (resume): the trials the study had before this run, recorded
                # the first time the run opened it, and how many it still needs.
                known = dlt.known_trials(runFolder, runMarker, studyKey,
                                         [t.number for t in study.get_trials(deepcopy=False)])
                remaining = dlt.remaining_trials(
                    [(t.number, t.state.name) for t in study.get_trials(deepcopy=False)], known, n_trials)
                print(f"{studyKey}: {n_trials - remaining} of {n_trials} trials done in this run, {remaining} to go")

                # Run the automatic tuning process. catch: a single failed
                # trial (e.g. a GPU-OOM on an oversized lstmUnits/tcnUnits
                # combination - observed 2026-08-19 on pick3, where trial 0
                # OOMing killed the entire study) is recorded as a FAILED
                # trial and the study moves on to the next suggestion instead
                # of aborting the whole game.
                def reportTrial(study, trial, studyKey=studyKey, known=known):
                    try:
                        dlt.write_result(resultFile, studyKey, progress=dlt.trial_progress(
                            study.get_trials(deepcopy=False), known))
                    except Exception as e:  # the page's view only: never stops the study
                        print(f"{studyKey}: could not write the result file ({e})")

                if resultFile:
                    reportTrial(study, None)
                if remaining:
                    study.optimize(objective, n_trials=remaining, catch=(Exception,),
                                   callbacks=[reportTrial] if resultFile else None)
                if resultFile:
                    dlt.write_result(resultFile, studyKey, status="gate" if gateEnabled else "writing", current=None)

                # Champion/challenger (src/TuningGate.py, 6 Nov 2026): this run's
                # best completed trial is written only if it beats the served
                # parameters and Predictor.py's defaults - both trained and scored
                # here on the same days with the same budget - by gateSeMargin of
                # its own standard errors; the defaults are served back when they
                # score above the served ones. Never study.best_params: older
                # trials (other budgets, other days) are sampler history only.
                displayName = MODEL_DISPLAY_NAMES[model_type]

                def evaluateReference(params, role, objective=objective, studyKey=studyKey):
                    print(f"{studyKey}: scoring the {role} (a full trial, never pruned)")
                    currentRun["role"] = role
                    reference = ReferenceTrial(dict(params))
                    try:
                        value = objective(reference)
                    except optuna.TrialPruned:
                        print(f"{studyKey}: the {role} timed out - it cannot block")
                        return None, None
                    except Exception as e:
                        print(f"{studyKey}: the {role} could not be scored ({e}) - it cannot block")
                        return None, None
                    return value, reference.user_attrs.get("tuning")

                outcome = challenge(
                    study, known, objective, existingData,
                    make_defaults(dlt.predictor_defaults(MODEL_PARAM_PREFIX.get(model_type), existingData), existingData),
                    timeout_seconds=TRIAL_TIMEOUT_SECONDS, label=studyKey, enabled=gateEnabled,
                    window_days=daysToRebuild, evaluate=evaluateReference, se_margin=gateSeMargin,
                    revert_margin=dlt.GATE_REVERT_MARGIN, variety=True)
                print(outcome["summary"])
                outcome["record"]["lockbox"] = dlt.LOCKBOX_NOTE
                outcome["record"]["train_seconds"] = trainSeconds or None
                existingData.setdefault("tuningGate", {})[displayName] = outcome["record"]
                if outcome["params"] is not None:
                    existingData.update(outcome["params"])
                if resultFile:
                    challengerParams = None
                    if outcome.get("trial") is not None:
                        try:
                            challengerParams = dict(study.trials[outcome["trial"]].params)
                        except Exception:
                            challengerParams = None
                    dlt.write_result(resultFile, studyKey, status="done", current=None, decision=outcome["decision"],
                                     reason=outcome["reason"], params=outcome["params"], challengerParams=challengerParams,
                                     record=outcome["record"], summary=outcome["summary"],
                                     progress=dlt.trial_progress(study.get_trials(deepcopy=False), known),
                                     finishedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"))

                # B: written and committed now, not at the end of the run
                with open(jsonBestParamsFilePath, "w+") as outfile:
                    json.dump(existingData, outfile, indent=4)
                dlt.mark_done(runFolder, runMarker, studyKey)
                if pushToGit:
                    try:
                        helpers.git_push(commit_message=f"Deep learning hyperopt: {studyKey} {outcome['decision']}")
                    except Exception as e:
                        print(f"Could not commit {studyKey}: {e} - the file is written, the end of the run tries again")

                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))
                clearFolder(os.path.join(path, "data", "hyperOptCache", "models", model_type))

            except Exception as e:
                print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")
                try:
                    dlt.write_result(resultFile, dlt.study_name(dataset_name, model_type), status="failed", current=None, error=str(e)[:500])
                except Exception:
                    pass

    try:
        for filename in os.listdir(os.getcwd()):
            if 'wget' in filename:
                file_path = os.path.join(os.getcwd(), filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
    except Exception as e:
        print("Failed to cleanup folder")

    if all(key in runMarker["done"] for key in studyKeys):
        dlt.finish_run(runFolder, runMarker)
    else:
        print("Studies not finished (see the errors above): "
              + ", ".join(key for key in studyKeys if key not in runMarker["done"])
              + " - the same command runs only those")

    try:
        if pushToGit:
            helpers.git_push(commit_message="Saving latest deep learning hyperopt")
    except Exception as e:
        print("Failed to push latest predictions:", e)
    finally:
        remove_lock()  # Ensure the lock is removed even if an error occurs
    

    
