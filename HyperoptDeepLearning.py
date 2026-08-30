import os, argparse, json, sys

# Must be set before TensorFlow is imported (the src imports below pull it in):
# without it TF's first process grabs virtually the whole GPU at startup, so
# any second TF process - or a big trial on the 6GB card - dies with cuDNN
# RESOURCE_EXHAUSTED even when the card is otherwise idle. Allow-growth makes
# TF allocate only what it actually uses.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import gc
import optuna
import numpy as np

from art import text2art
from datetime import datetime
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor


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
# same reasoning as HyperoptStatistics.py's suggest_keno_subset docstring.
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

        best_match = helpers.find_best_matching_prediction(
            curr_json["realResult"], curr_json["currentPrediction"]
        )
        curr_json["matchingNumbers"] = best_match

        # Save updated JSON
        with open(curr_file, "w") as f_curr_out:
            json.dump(curr_json, f_curr_out, indent=2)

    print(f"Updated matching numbers in {len(sorted_files) - 1} files.")


def process_single_history_entry(args):
    (historyIndex, historyEntry, historyData, name, model_type, dataPath, modelPath,
        skipLastColumns, years_back, previousJsonFilePath, path, modelParams, specialColumnCount) = args

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

    best_matching_prediction = helpers.find_best_matching_prediction(
        current_json_object["realResult"], current_json_object["currentPrediction"])
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

    # Perform training
    latest_raw_predictions, unique_labels = modelToUse.run(
        name, skipLastColumns, skipRows=len(historyData)-historyIndex, years_back=years_back, strict_val=False,
        specialColumnCount=specialColumnCount)

    # Set by run() on every model in MODEL_REGISTRY (best val_loss across
    # epochs) - used by predict()/objective() below as the primary hyperopt
    # signal instead of relying solely on the tiny real-draw profit sample.
    val_loss = getattr(modelToUse, "last_val_loss", float("inf"))

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

    return jsonFilePath, val_loss



def runPredictInChild(*args, **kwargs):
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
    """
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
        return executor.submit(predict, *args, **kwargs).result()


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

def predict(name, model_type ,dataPath, modelPath, file, skipLastColumns=0, maxRows=0, years_back=None, daysToRebuild=31, modelParams={}, specialColumnCount=0):
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
                modelPath, skipLastColumns, years_back, previousJsonFilePath, path, modelParams, specialColumnCount)
                for historyIndex, historyEntry in enumerate(historyData)
            ]


            val_losses = []
            for args in argsList:
                _, val_loss = process_single_history_entry(args)
                if np.isfinite(val_loss):
                    val_losses.append(val_loss)

            print("Finished rebuild of history entries.")

            # Find the matching numbers
            update_matching_numbers(name=name, path=path)

            # Calculate Profit
            profit =  helpers.calculate_profit(name=name, path=path)

            # avg_val_loss is backed by every retrain's held-out validation
            # windows (hundreds of samples) - far more reliable than profit,
            # which is backed by only `daysToRebuild` real draws. See
            # objective() for how the two are combined.
            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")

            return profit, avg_val_loss
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
                subset = helpers.generate_subset_from_scores(number_scores, predicted_digits, subset_size)
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
    parser.add_argument('-d', '--days', type=int, default=8)
    parser.add_argument('-t', '--trials', type=int, default=15)
    parser.add_argument('-s', '--save', type=helpers.str2bool, default=True)
    parser.add_argument(
        '-g', '--games',
        type=str,
        default="euromillions,lotto,eurodreams,keno,pick3,vikinglotto",
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
    pushToGit = bool(args.save)

    print("Push to git: ", pushToGit)
    print("Running ", n_trials, "trials")

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
    # Trailing special/bonus column(s): Euromillions (2 star columns),
    # EuroDreams (1 dream number), VikingLotto (1 super viking) get modeled
    # via a second output head (see LSTM.py/TCN.py/UnifiedLstmTcn.py/
    # UnifiedLstmGruTcn.py create_model) instead of being lumped into the
    # main numbers' output. Lotto's bonus number isn't modeled at all - it's
    # simply dropped via skip_last_columns (was incorrectly 0 here, unlike
    # Predictor.py's own dataset list, which already drops it).
    SPECIAL_COLUMN_COUNTS = {"euromillions": 2, "eurodreams": 1, "vikinglotto": 1}
    datasets = [
        # (dataset_name, model_type, skip_last_columns, special_column_count)
        (dataset_name, model_type, skip_last_columns, SPECIAL_COLUMN_COUNTS.get(dataset_name, 0))
        for dataset_name, skip_last_columns in [
            ("euromillions", 0),
            ("lotto", 1),
            ("eurodreams", 0),
            #("jokerplus", 1),
            ("keno", 0),
            ("pick3", 0),
            ("vikinglotto", 0),
        ]
        for model_type in dl_model_types
    ]

    for dataset_name, model_type, skip_last_columns, special_column_count in datasets:
        if dataset_name in games:
            try:
                print(f"\n{dataset_name.capitalize()}")
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
                    dataFetcher.getLatestData(gameName, filePath)
                    #os.remove(os.path.join(dataPath, file))
                #command.run("wget -P {folder} https://prdlnboppreportsst.blob.core.windows.net/legal-reports/{file}".format(**kwargs_wget), verbose=False)


                def objective(trial):
                    numOfRepeats = 1 # To average out the rusults before continueing to the next result
                    totalProfit = 0
                    results = [] # Intermediate results

                    if model_type == "lstm_model":
                        # Unprefixed - matches Predictor.py's existing LSTM
                        # setter block, which reads these same bare keys.
                        # Don't add a prefix here, it would break that.
                        modelParams = {
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
                    elif model_type == "tcn_model":
                        modelParams = suggest_tcn_params(trial, MODEL_PARAM_PREFIX[model_type])
                    elif model_type == "transformer_model":
                        modelParams = suggest_transformer_params(trial, MODEL_PARAM_PREFIX[model_type])
                    elif model_type == "gnn_model":
                        modelParams = suggest_gnn_params(trial, MODEL_PARAM_PREFIX[model_type])
                    elif model_type == "autoencoder_model":
                        modelParams = suggest_autoencoder_params(trial, MODEL_PARAM_PREFIX[model_type])
                    else:
                        modelParams = suggest_fused_params(
                            trial, MODEL_PARAM_PREFIX[model_type],
                            include_gru=(model_type == "unified_lstm_gru_tcn_model"))

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
                        result = runPredictInChild(f"{dataset_name}", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=modelParams["yearsOfHistory"], daysToRebuild=daysToRebuild, modelParams=modelParams, specialColumnCount=special_column_count)
                        if result is not None:
                            results.append(result)

                    clearFolder(os.path.join(path, "data", "hyperOptCache", "models", model_type))

                    if not results:
                        # No prediction could be made this trial (e.g. no new
                        # history to rebuild) - fail the trial rather than
                        # silently scoring it as a tie with real trials.
                        return -float("inf")

                    avgProfit = sum(profit for profit, _ in results) / len(results)
                    avgValLoss = sum(val_loss for _, val_loss in results) / len(results)

                    # val_loss is the primary signal - it's backed by ~hundreds
                    # of held-out validation windows per trial, versus profit's
                    # `daysToRebuild` real draws (often just 1-2), which is far
                    # too small a sample to reliably separate a genuinely better
                    # model from a lucky one. Profit is kept only as a small
                    # tie-breaker between models with similar val_loss.
                    #
                    # The tie-breaker is CLIPPED: at pick3/keno payout scale a
                    # single lucky mid-tier hit in the scored week adds +2.75
                    # or more to the score, an order of magnitude above the
                    # val_loss spread between trials (~0.1-0.3) - a 2026-08-30
                    # pick3 transformer study had its "best" trial selected by
                    # exactly one such hit (+0.34 vs ~-3.7 for every other
                    # trial). The cap keeps profit a tie-breaker instead of a
                    # lottery on which trial got lucky.
                    PROFIT_WEIGHT = 0.05
                    PROFIT_CONTRIBUTION_CAP = 0.5
                    profitTerm = max(-PROFIT_CONTRIBUTION_CAP,
                                     min(PROFIT_CONTRIBUTION_CAP, PROFIT_WEIGHT * avgProfit))
                    return -avgValLoss + profitTerm
                
                # Write best params to json
                jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
                existingData = {}
                if os.path.exists(jsonBestParamsFilePath):
                    with open(jsonBestParamsFilePath, "r") as infile:
                        existingData = json.load(infile)

                # Create an Optuna study object
                study = optuna.create_study(
                    direction='maximize',
                    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                    study_name=f"{dataset_name}-{model_type}",
                    load_if_exists=True
                )

                # Run the automatic tuning process. catch: a single failed
                # trial (e.g. a GPU-OOM on an oversized lstmUnits/tcnUnits
                # combination - observed 2026-08-19 on pick3, where trial 0
                # OOMing killed the entire study) is recorded as a FAILED
                # trial and the study moves on to the next suggestion instead
                # of aborting the whole game.
                study.optimize(objective, n_trials=n_trials, catch=(Exception,))

                # Output the best hyperparameters and score
                print("Best Parameters: ", study.best_params)
                print("Best Score: ", study.best_value)

                existingData.update(study.best_params)

                with open(jsonBestParamsFilePath, "w+") as outfile:
                    json.dump(existingData, outfile, indent=4)
                
                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))
                clearFolder(os.path.join(path, "data", "hyperOptCache", "models", model_type))

            except Exception as e:
                print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")

    try:
        for filename in os.listdir(os.getcwd()):
            if 'wget' in filename:
                file_path = os.path.join(os.getcwd(), filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
    except Exception as e:
        print("Failed to cleanup folder")

    try:
        if pushToGit:
            helpers.git_push(commit_message="Saving latest deep learning hyperopt")
    except Exception as e:
        print("Failed to push latest predictions:", e)
    finally:
        remove_lock()  # Ensure the lock is removed even if an error occurs
    

    
