import os, argparse, json, sys, time

# Must be set before TensorFlow is imported (the src imports below pull it in):
# without it TF's first process grabs virtually the whole GPU at startup, so
# any second TF process - or a big trial on the 6GB card - dies with cuDNN
# RESOURCE_EXHAUSTED even when the card is otherwise idle. Allow-growth makes
# TF allocate only what it actually uses.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import numpy as np
import subprocess
import joblib

from art import text2art
from datetime import datetime
from multiprocessing import Pool, cpu_count, get_context
from concurrent.futures import ProcessPoolExecutor

from src.TCN import TCNModel
from src.LSTM import LSTMModel
from src.UnifiedLstmTcn import UnifiedLstmTcnModel
from src.UnifiedLstmGruTcn import UnifiedLstmGruTcnModel
from src.Markov import Markov
from src.MarkovMonteCarlo import MarkovMonteCarlo
from src.MarkovBayesian import MarkovBayesian
from src.MarkovBayesianEnhanched import MarkovBayesianEnhanced
from src.PoissonMonteCarlo import PoissonMonteCarlo
from src.PoissonMarkov import PoissonMarkov
from src.LaplaceMonteCarlo import LaplaceMonteCarlo
from src.HybridStatisticalModel import HybridStatisticalModel
from src.TransformerModel import TransformerModel
from src.GNN import GNNModel
from src.AutoencoderAnomaly import AutoencoderAnomaly
from src.RLTicketModel import RLTicketModel
from src.XGBoost import XGBoostPredictor, XGBoostMultiLabelPredictor
from src.LightGBM import LightGBMPredictor, LightGBMMultiLabelPredictor
from src.CatBoost import CatBoostPredictor, CatBoostMultiLabelPredictor
from src.BoostingBase import apply_boosting_params
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
rlTicket = RLTicketModel()
markov = Markov()
markovMcBase = Markov()
markovMonteCarlo = MarkovMonteCarlo(markovMcBase)
markovBayesian = MarkovBayesian()
markovBayesianEnhanced = MarkovBayesianEnhanced()
poissonMonteCarlo = PoissonMonteCarlo()
laplaceMonteCarlo = LaplaceMonteCarlo()
hybridStatisticalModel = HybridStatisticalModel()
poissonMarkov = PoissonMarkov()
xgboostPredictor = XGBoostPredictor()
command = Command()
helpers = Helpers()
dataFetcher = DataFetcher()

LOCK_FILE = os.path.join(os.getcwd(), "process.lock")

# Euromillions (2 star columns), EuroDreams (1 dream number), and
# VikingLotto (1 super viking) are drawn from their own, smaller range -
# model them independently from the main numbers (see
# Helpers.run_model_with_special_column) instead of mixing them into the
# same pool/sort. Lotto's bonus number is not modeled at all (it isn't
# played), so it keeps being fully dropped via skipLastColumns.
SPECIAL_COLUMN_COUNTS = {"euromillions": 2, "eurodreams": 1, "vikinglotto": 1}


def getKenoSubsetSizes(name, bestParams_json_object):
    """
    Keno is the only game with sub-selections (playable 5-10-number tickets
    out of the full 20). Shared by statisticalMethod (individual models),
    addWeightedEnsemblePrediction, and the MetaLearner block, so all three
    respect the same use_5..use_10 hyperopt-tuned toggles instead of each
    reimplementing this lookup.
    """
    if "keno" not in name:
        return []
    return [size for size in (5, 6, 7, 8, 9, 10) if bestParams_json_object.get(f"use_{size}")]


# Caches loaded meta_learner.joblib artifacts (see TrainMetaLearner.py) by
# path, so a rebuild loop over many history days doesn't reload the same
# artifact from disk on every single day.
metaLearnerCache = {}


def print_intro():
    # Generate ASCII art with the text "LSTM"
    ascii_art = text2art("Predictor")
    # Print the introduction and ASCII art
    print("============================================================")
    print("Predictor")
    print("Licence : MIT License")
    print(ascii_art)
    print("Prediction artificial intelligence")

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

def update_matching_numbers(name, path):
    json_dir = os.path.join(path, "data", "database", name)
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
        
        #print("prev_json: ", prev_json)
        #print("curr_json: ", curr_json)

        curr_json["currentPredictionRaw"] = prev_json.get("newPredictionRaw", [])
        curr_json["currentPrediction"] = prev_json.get("newPrediction", [])
        curr_json["currentNumberFrequency"] = prev_json.get("numberFrequency", {})

        best_match = helpers.find_best_matching_prediction(
            curr_json["realResult"], curr_json["currentPrediction"]
        )
        curr_json["matchingNumbers"] = best_match

        # Save updated JSON
        with open(curr_file, "w") as f_curr_out:
            json.dump(curr_json, f_curr_out, indent=2)

    print(f"Updated matching numbers in {len(sorted_files) - 1} files.")


def process_single_history_entry_first_step(args):
    """
    First step to prepare the database and perform the statistical method.
    In this step we can process multible files.
    """
    
    (historyIndex, historyEntry, historyData, name, dataPath, previousJsonFilePath, path, skipLastColumns) = args

    historyDate, historyResult = historyEntry
    jsonFileName = f"{historyDate.year}-{historyDate.month}-{historyDate.day}.json"
    jsonFilePath = os.path.join(path, "data", "database", name, jsonFileName)

    current_json_object = {
        "currentPredictionRaw": [],
        "currentPrediction": [],
        "currentNumberFrequency": {},
        "realResult": historyResult,
        "newPrediction": [],
        "newPredictionRaw": [],
        "matchingNumbers": {},
        "labels": [],
        "numberFrequency": []
    }

    try:
        # Check the previous prediction with the real result
        if previousJsonFilePath and os.path.exists(previousJsonFilePath):
            with open(previousJsonFilePath, 'r') as openfile:
                print("openfile: ", openfile)
                previous_json_object = json.load(openfile)
            current_json_object["currentPredictionRaw"] = previous_json_object["newPredictionRaw"]
            current_json_object["currentPrediction"] = previous_json_object["newPrediction"]
            current_json_object["currentNumberFrequency"] = previous_json_object.get("numberFrequency", {})

        best_matching_prediction = helpers.find_best_matching_prediction(
            current_json_object["realResult"], current_json_object["currentPrediction"])
        current_json_object["matchingNumbers"] = best_matching_prediction

        

        with open(jsonFilePath, "w+") as outfile:
            json.dump(current_json_object, outfile, indent=2)
    except Exception as e:
        print("Failed to check previous json: ", e)

    try: 
        listOfDecodedPredictions = []

        listOfDecodedPredictions = statisticalMethod(
           listOfDecodedPredictions, dataPath, path, name, skipRows=len(historyData)-historyIndex
           , skipLastColumns=skipLastColumns
        )

        current_json_object["newPrediction"] = listOfDecodedPredictions
    except Exception as e:
        print("Failed to perform statistical method: ", e)

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile, indent=2)

    return jsonFilePath


def deepLearningStep(name, dataPath, modelPath, skipLastColumns, bestParams_json_object,
                     years_back, specialColumnCount, skipRows, repoPath, fullAi=True):
    """
    Trains and predicts every deep learning model for one day: LSTM Base
    Model plus the UNIFIED_DL_MODELS rows. Module-level and fully
    argument-driven so it can run in a spawned child process (see
    runDeepLearningStepInChild). Returns (newPredictionRaw, unique_labels,
    dlRows, anomalyWatch) - newPredictionRaw/unique_labels are None if the
    LSTM failed, anomalyWatch is None unless the Autoencoder Model ran (it
    must be computed in this same child - the trained model dies with it).
    """
    dlRows = []
    predictedSequence = None
    unique_labels = None

    # fullAi=False (the cron default, --ai off) skips the heavy legacy DL
    # models (LSTM base + the requiresFullAi registry rows) but still runs
    # the lightweight research models per their use<Prefix> toggles.
    if not fullAi or not bestParams_json_object.get("useLstm", True):
        dlRows, anomalyWatch = runUnifiedDeepLearningModels(
            dlRows, repoPath, name, dataPath, skipLastColumns, bestParams_json_object,
            skipRows=skipRows, years_back=years_back, specialColumnCount=specialColumnCount,
            fullAi=fullAi)
        return predictedSequence, unique_labels, dlRows, anomalyWatch

    modelToUse = lstm
    modelToUse.setDataPath(dataPath)
    modelToUse.setModelPath(modelPath)
    modelToUse.setBatchSize(bestParams_json_object["batchSize"])
    modelToUse.setEpochs(bestParams_json_object["epochs"])
    modelToUse.setNumberOfLSTMLayers(bestParams_json_object["num_lstm_layers"])
    modelToUse.setNumberOfLstmUnits(bestParams_json_object["lstm_units"])
    modelToUse.setNumberOfBidrectionalLayers(bestParams_json_object["num_bidirectional_layers"])
    modelToUse.setNumberOfBidirectionalLstmUnits(bestParams_json_object["bidirectional_lstm_units"])
    modelToUse.setOptimizer(bestParams_json_object["optimizer_type"])
    modelToUse.setLearningRate(bestParams_json_object["learningRate"])
    modelToUse.setDropout(bestParams_json_object["dropout"]) # 0.2 - 0.5
    modelToUse.setL2Regularization(bestParams_json_object["l2Regularization"]) # 0.0001 - 0.001
    modelToUse.setUseFinalLSTMLayer(bestParams_json_object["useFinalLSTMLayer"])
    modelToUse.setEarlyStopPatience(bestParams_json_object["earlyStopPatience"])
    modelToUse.setReduceLearningRatePAience(bestParams_json_object["reduceLearningRatePatience"])
    modelToUse.setReducedLearningRateFactor(bestParams_json_object["reduceLearningRateFactor"])
    modelToUse.setWindowSize(bestParams_json_object["windowSize"]) # 50 - 100
    modelToUse.setPredictionWindowSize(modelToUse.window_size)
    modelToUse.setLabelSmoothing(bestParams_json_object["labelSmoothing"])

    # Own try/except (like every other model): the LSTM raising - e.g.
    # training went NaN with no healthy checkpoint (see LSTM.run) - must cost
    # only its own row, not the unified DL rows that follow.
    try:
        latest_raw_predictions, unique_labels = modelToUse.run(
            name, skipLastColumns, skipRows=skipRows, years_back=years_back,
            specialColumnCount=specialColumnCount)
        predictedSequence = latest_raw_predictions.tolist()
        dlRows = deepLearningMethod(
            dlRows, predictedSequence, unique_labels, gameName=name,
            kenoSubsetSizes=getKenoSubsetSizes(name, bestParams_json_object))
    except Exception as e:
        print("Failed to perform LSTM Base Model prediction: ", e)

    dlRows, anomalyWatch = runUnifiedDeepLearningModels(
        dlRows, repoPath, name, dataPath, skipLastColumns, bestParams_json_object,
        skipRows=skipRows, years_back=years_back, specialColumnCount=specialColumnCount,
        fullAi=fullAi)

    return predictedSequence, unique_labels, dlRows, anomalyWatch


def runDeepLearningStepInChild(*args):
    """
    Runs deepLearningStep in a one-shot spawned child process, for two
    reasons learned the hard way:

    - Containment: the memory-cgroup OOM killer SIGKILLs the process when it
      outgrows the container's 16GB (observed 2026-08-20 at 16.7GB RSS and
      2026-08-22 at 10.3GB) - no Python traceback, and previously the entire
      Predictor died mid-pipeline. In a child, the kill surfaces here as
      BrokenProcessPool: the day loses only its DL rows (the half-built file
      is auto-repaired by the completeness-aware rebuild) and every other
      method still runs.
    - Memory hygiene: those 10-16GB were TF/Keras allocations accumulating
      across the ~4 model trainings per day x games in one long-lived
      process. A fresh child per day releases everything back to the OS on
      exit, so RSS can't ratchet up in the first place.

    Costs one TF import + CUDA init per day (~15-30s) - small next to the
    minutes of training it wraps.
    """
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            return executor.submit(deepLearningStep, *args).result()
    except Exception as e:
        print("Deep learning step died (likely OOM-killed) - skipping DL rows for this day: ", e)
        return None, None, [], None


def process_single_history_entry_second_step(args):
    """
    Second step to perform methods where we can not process multible files at the same time
    """
    
    (historyIndex, historyEntry, historyData, name, model_type, dataPath, modelPath,
     skipLastColumns, years_back, ai, previousJsonFilePath, path, boost, bestParams_json_object,
     specialColumnCount) = args

    historyDate, historyResult = historyEntry
    jsonFileName = f"{historyDate.year}-{historyDate.month}-{historyDate.day}.json"
    jsonFilePath = os.path.join(path, "data", "database", name, jsonFileName)

    current_json_object = {}

    # We need the file of the first step to continue
    if jsonFilePath and os.path.exists(jsonFilePath):
        with open(jsonFilePath, 'r') as openfile:
            current_json_object = json.load(openfile)
    else:
        print("File of first step not found")
        exit()

    listOfDecodedPredictions = current_json_object["newPrediction"]
    unique_labels = []

    if ai or lightDlModelsEnabled(bestParams_json_object):
        # All DL training runs in a one-shot spawned child - see
        # runDeepLearningStepInChild for why (OOM containment + per-day
        # memory release). With ai (--ai) off, the child still runs the
        # lightweight research models (Transformer/GNN/Autoencoder) - only
        # the heavy LSTM/TCN/unified trainings stay behind the flag.
        predictedSequence, dl_labels, dlRows, anomalyWatch = runDeepLearningStepInChild(
            name, dataPath, modelPath, skipLastColumns, bestParams_json_object,
            years_back, specialColumnCount, len(historyData)-historyIndex, path, ai)

        if predictedSequence is not None:
            current_json_object["newPredictionRaw"] = predictedSequence
        if anomalyWatch is not None:
            current_json_object["anomalyWatch"] = anomalyWatch
        listOfDecodedPredictions.extend(dlRows)
        unique_labels = dl_labels

    if unique_labels is None or len(unique_labels) == 0:
        # ai disabled, or the DL child died: labels are still needed for the
        # json's "labels" field (and to mark the file complete - see
        # entryIsComplete), so load them straight from the data.
        _, _, _, _, _, _, _, unique_labels = helpers.load_data(
            dataPath, skipLastColumns, years_back=years_back)
        unique_labels = unique_labels.tolist()

    if boost:
       listOfDecodedPredictions = boostingMethod(
           listOfDecodedPredictions, dataPath, path, name,
           skipRows=(len(historyData)-historyIndex), skipLastColumns=skipLastColumns)

    
    current_json_object["newPrediction"] = listOfDecodedPredictions
    current_json_object["labels"] = unique_labels

    # Calculate the frequent numbers in prediction in last step
    try:
        current_json_object["numberFrequency"] = helpers.count_number_frequencies_from_new_prediction(
            current_json_object, model_scores=bestParams_json_object.get("modelScores"))
        addWeightedEnsemblePrediction(current_json_object, name, model_scores=bestParams_json_object.get("modelScores"), bestParams_json_object=bestParams_json_object)
    except Exception as e:
        print("Failed to calculate the number frequencies: ", e)

    addRLTicketPrediction(listOfDecodedPredictions, dataPath, path, name,
                          specialColumnCount, bestParams_json_object,
                          # historyResult still carries non-modeled trailing
                          # columns (lotto's bonus via skipLastColumns) on top
                          # of the special columns - drop both.
                          fallbackDrawSize=len(historyResult) - specialColumnCount - skipLastColumns,
                          cutoffDate=historyDate)

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile, indent=2)



    return jsonFilePath


def predict(name, model_type ,dataPath, modelPath, skipLastColumns=0, daysToRebuild=31, ai=False, boost=False, forceRebuild=False):
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
        @param forceRebuild: Re-generate (overwrite) the last daysToRebuild
               draws even when their files already exist - the -r flag. Use
               after history corruption. Without it, existing files are never
               overwritten; only missing draws are built.
    """

    # Get the hyperopted parameters 
    bestParams_json_object = {
        "yearsOfHistory": 20,   
    }

    try:
        # Load hyperopt parameters if exists
        hyperoptParamsJsonFile = os.path.join(path, f"bestParams_{name}.json")
        if hyperoptParamsJsonFile and os.path.exists(hyperoptParamsJsonFile):
            with open(hyperoptParamsJsonFile, 'r') as openfile:
                # Merge over the in-code defaults rather than replacing them
                # outright - an existing bestParams_<game>.json predating a
                # newer default key (e.g. useMarkovMonteCarlo) would otherwise
                # crash every lookup of that key with a KeyError.
                bestParams_json_object.update(json.load(openfile))
    except Exception as e:
        print("Failed to parse parameter file: ", e)

    # model_type is always "lstm_model" here (see the __main__ datasets list) -
    # TCN now runs as its own additional row via UNIFIED_DL_MODELS instead of
    # this either/or.
    modelToUse = lstm
    modelToUse.setDataPath(dataPath)

    # Get the latest result out of the latest data so we can use it to check the previous prediction
    latestEntry, previousEntry = helpers.getLatestPrediction(dataPath)
    if latestEntry is not None:
        latestDate, latestResult = latestEntry

        
        jsonFileName = f"{latestDate.year}-{latestDate.month}-{latestDate.day}.json"
        #print(jsonFileName, ":", latestResult)
        jsonFilePath = os.path.join(path, "data", "database", name, jsonFileName)

        # Check if folder exists
        if not os.path.exists(os.path.join(path, "data", "database", name)):
            os.makedirs(os.path.join(path, "data", "database", name), exist_ok=True)


        # Compare the latest result with the previous new prediction.
        # forceRebuild (-r) skips both the compare path and the
        # already-made short-circuit - it exists to re-generate history
        # whose files are present but corrupt.
        if forceRebuild or not os.path.exists(jsonFilePath):
            if not forceRebuild:
                print("New result detected. Lets compare with a prediction from previous entry")

            current_json_object = {
                "currentPredictionRaw": [],
                "currentPrediction": [],
                "currentNumberFrequency": {},
                "realResult": latestResult,
                "newPrediction": [],      # Decoded prediction with help of labels
                "newPredictionRaw": [],   # Raw prediction that contains the statistical data
                "matchingNumbers": {},
                "labels": [],             # Needed for decoding the raw predictions
                "numberFrequency": {}
            }

            doNewPrediction = True

            # First find the json file containing the prediction for this result
            if not forceRebuild and previousEntry is not None:
                previousDate, previousResult = previousEntry
                jsonPreviousFileName = f"{previousDate.year}-{previousDate.month}-{previousDate.day}.json"
                print(jsonPreviousFileName, ":", latestResult)
                jsonPreviousFilePath = os.path.join(path, "data", "database", name, jsonPreviousFileName)
                print(jsonPreviousFilePath)
                if os.path.exists(jsonPreviousFilePath):
                    doNewPrediction = False
                    print("previous json file found lets compare")
                    # Opening JSON file
                    with open(jsonPreviousFilePath, 'r') as openfile:
                    
                        # Reading from json file
                        previous_json_object = json.load(openfile)
                    
                    #print(previous_json_object)
                    #print(type(previous_json_object))

                    # The current prediction is the new prediction from the previous one
                    current_json_object["currentPredictionRaw"] = previous_json_object["newPredictionRaw"]
                    current_json_object["currentPrediction"] = previous_json_object["newPrediction"]
                    current_json_object["currentNumberFrequency"] = previous_json_object.get("numberFrequency", {})

                    # Check on prediction with nth highest probability
                    print("find matching numbers")
                    best_matching_prediction = helpers.find_best_matching_prediction(current_json_object["realResult"], current_json_object["currentPrediction"])

                    current_json_object["matchingNumbers"] = best_matching_prediction

                    print("matching_numbers: ", current_json_object["matchingNumbers"]["matching_numbers"])

                    listOfDecodedPredictions = []
                    unique_labels = []

                    yearsOfHistory = bestParams_json_object['yearsOfHistory']
                    if ai or lightDlModelsEnabled(bestParams_json_object):
                        try:
                            # Train and do a new prediction - in a one-shot
                            # spawned child (see runDeepLearningStepInChild:
                            # OOM containment + per-day memory release). This
                            # is the daily cron path, where a single process
                            # doing 4 DL trainings x 6 games used to grow past
                            # the container's 16GB cgroup limit and get
                            # SIGKILLed with no trace. With ai (--ai) off the
                            # child still runs the lightweight research models
                            # (Transformer/GNN/Autoencoder); only the heavy
                            # LSTM/TCN/unified trainings stay behind the flag.
                            specialColumnCount = SPECIAL_COLUMN_COUNTS.get(name, 0)

                            predictedSequence, dl_labels, dlRows, anomalyWatch = runDeepLearningStepInChild(
                                name, dataPath, modelPath, skipLastColumns, bestParams_json_object,
                                yearsOfHistory, specialColumnCount, 0, path, ai)

                            if predictedSequence is not None:
                                current_json_object["newPredictionRaw"] = predictedSequence
                                current_json_object["labels"] = dl_labels
                                unique_labels = dl_labels
                            if anomalyWatch is not None:
                                current_json_object["anomalyWatch"] = anomalyWatch
                            listOfDecodedPredictions.extend(dlRows)
                        except Exception as e:
                            print("Failed to perform deep learning method: ", e)
                    if unique_labels is None or len(unique_labels) == 0:
                        # ai off with the LSTM skipped (or the DL child died):
                        # labels are still needed for the json's "labels"
                        # field and the entryIsComplete check.
                        _, _, _, _, _, _, _, unique_labels = helpers.load_data(dataPath, skipLastColumns, years_back=yearsOfHistory)
                        unique_labels = unique_labels.tolist()

                    # Always record labels - with ai disabled (or the DL child
                    # dead) this used to stay [], which the completeness check
                    # (entryIsComplete) would read as a half-built file and
                    # pointlessly rebuild the day on every following run.
                    current_json_object["labels"] = unique_labels

                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile, indent=2)
                    
                    listOfDecodedPredictions = statisticalMethod(listOfDecodedPredictions, dataPath, path, name, skipLastColumns=skipLastColumns)
                    
                    if boost:
                        listOfDecodedPredictions = boostingMethod(
                            listOfDecodedPredictions, dataPath, path, name, skipLastColumns=skipLastColumns)

                    current_json_object["newPrediction"] = listOfDecodedPredictions

                    # Calculate the frequent numbers in prediction
                    try:
                        current_json_object["numberFrequency"] = helpers.count_number_frequencies_from_new_prediction(
                            current_json_object, model_scores=bestParams_json_object.get("modelScores"))
                        addWeightedEnsemblePrediction(current_json_object, name, model_scores=bestParams_json_object.get("modelScores"), bestParams_json_object=bestParams_json_object)
                    except Exception as e:
                        print("Failed to calculate the number frequencies: ", e)

                    addRLTicketPrediction(listOfDecodedPredictions, dataPath, path, name,
                                          SPECIAL_COLUMN_COUNTS.get(name, 0), bestParams_json_object)

                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile, indent=2)

                    #return predictedSequence
                
            if doNewPrediction:
                # Which draws need (re)building is decided from the actual
                # draw history (getLatestPrediction counts draw rows, so games
                # that only play twice a week are handled correctly), not a
                # fixed calendar window:
                #
                # - Normal gap recovery: anchor on the NEWEST draw that
                #   already has a database json and rebuild everything after
                #   it - however large that gap is (a multi-day outage used to
                #   be silently truncated to daysToRebuild draws). Existing
                #   files are never overwritten; the old logic anchored on the
                #   OLDEST existing file inside the window and re-generated
                #   every valid day after it.
                # - Interior holes (e.g. corrupted files that were deleted)
                #   within the last daysToRebuild draws are rebuilt too. Each
                #   day only needs the draw data before it (skipRows), and
                #   update_matching_numbers() re-links every file pair at the
                #   end, so filling holes out of sequence is safe.
                # - forceRebuild (-r): explicitly re-generate (overwrite) the
                #   last daysToRebuild draws - the only path that overwrites.
                #
                # A huge dateRange returns the entire draw history; skipRows
                # semantics stay exact because the list still ends at the
                # newest draw: len(fullHistory) - absoluteIndex is the same
                # value the old truncated-list arithmetic produced.
                fullHistory = helpers.getLatestPrediction(dataPath, dateRange=10**9)

                def entryJsonPath(entry):
                    entryDate = entry[0]
                    return os.path.join(path, "data", "database", name,
                                        f"{entryDate.year}-{entryDate.month}-{entryDate.day}.json")

                def entryIsComplete(jsonPath):
                    """
                    Step 2 (process_single_history_entry_second_step) always
                    writes a non-empty "labels" list - in both its ai and
                    non-ai branches - so an existing file without one is a
                    half-built leftover of a run that died between step 1 and
                    step 2 (only the statistical rows present; no DL/boosting
                    rows, no numberFrequency, no matching-number links).
                    Those used to count as valid history forever; treat them
                    like holes so they get rebuilt.
                    """
                    if not os.path.exists(jsonPath):
                        return False
                    try:
                        with open(jsonPath, "r") as openfile:
                            return bool(json.load(openfile).get("labels"))
                    except Exception:
                        return False

                existingIndices = [i for i, entry in enumerate(fullHistory) if entryIsComplete(entryJsonPath(entry))]
                windowStart = max(0, len(fullHistory) - daysToRebuild)

                if forceRebuild or not existingIndices:
                    # -r, or a brand-new game with no database at all: build
                    # the last daysToRebuild draws.
                    rebuildIndices = list(range(windowStart, len(fullHistory)))
                    print(f"Rebuilding the last {len(rebuildIndices)} draws"
                          + (" (forced by -r, existing files are overwritten)" if forceRebuild else " (no existing history found)"))
                else:
                    anchorIdx = existingIndices[-1]
                    existingSet = set(existingIndices)
                    holeIndices = [i for i in range(windowStart, anchorIdx) if i not in existingSet]
                    rebuildIndices = holeIndices + list(range(anchorIdx + 1, len(fullHistory)))
                    print(f"Newest complete prediction: {os.path.basename(entryJsonPath(fullHistory[anchorIdx]))} - "
                          f"building {len(rebuildIndices)} missing/incomplete draw(s) "
                          f"({len(holeIndices)} interior, {len(fullHistory) - anchorIdx - 1} after the newest complete file)")

                if rebuildIndices:
                    print("Date to start from: ", fullHistory[rebuildIndices[0]])

                # Only the first rebuilt entry gets a pre-existing "previous"
                # file (the nearest existing json older than it, to seed the
                # prediction-vs-result chain). Every other entry's true
                # previous file is produced by a sibling worker in this same
                # parallel batch - reading it here would race against that
                # worker's own concurrent write to the exact same path (and
                # update_matching_numbers() re-links every consecutive file
                # pair afterwards anyway).
                previousJsonFilePath = ""
                if rebuildIndices:
                    for i in reversed(range(rebuildIndices[0])):
                        candidate = entryJsonPath(fullHistory[i])
                        if entryIsComplete(candidate):
                            previousJsonFilePath = candidate
                            break

                argsList = [
                    (absoluteIndex, fullHistory[absoluteIndex], fullHistory, name, dataPath,
                    previousJsonFilePath if position == 0 else "", path, skipLastColumns)
                    for position, absoluteIndex in enumerate(rebuildIndices)
                ]

                #print("Argslist: ", len(argsList))

                if len(argsList) > 0:
                    # spawn (not the default fork) + ProcessPoolExecutor, for
                    # two hard-learned reasons. (1) By the time a second/third
                    # game reaches this rebuild, the parent has already run
                    # LSTM training (CUDA) and XGBoost (OpenMP) for the
                    # previous game's second step - forking such a parent hands
                    # every worker corrupted thread/lock state, and they
                    # segfault in libc on first use (observed 2026-08-17:
                    # identical-address libc segfaults in all workers minutes
                    # after start). spawn starts clean interpreters instead.
                    # (2) multiprocessing.Pool.map hangs FOREVER if a worker
                    # dies - that same 08-17 run sat futex-waiting for two days
                    # holding process.lock, silently blocking every cron run
                    # after it. ProcessPoolExecutor raises BrokenProcessPool
                    # instead, so a dead worker becomes a loud failed game, not
                    # a dead pipeline.
                    # Capped at 8: spawn workers each import TensorFlow
                    # (~600MB RSS) since they re-import this module from
                    # scratch, and this box has 16GB - cpu_count()-1 workers
                    # (15) would spend ~9GB on imports alone before doing any
                    # work. The per-day statistical work is seconds each, so
                    # the pool was never the bottleneck anyway.
                    with ProcessPoolExecutor(
                        max_workers=min(8, (cpu_count()-1), len(argsList)),
                        mp_context=get_context("spawn")
                    ) as executor:
                        results = list(executor.map(process_single_history_entry_first_step, argsList))

                    print("Finished first step: multiprocessing rebuild of history entries and statistical method.")

                    # Checkpoint: link matching numbers now, so the statistical
                    # rows' matches/profit are recorded on disk even if a later
                    # step (DL training, boosting) crashes or gets OOM-killed.
                    # Runs again after the second step to link the full rows.
                    update_matching_numbers(name=name, path=path)

                    yearsOfHistory = bestParams_json_object['yearsOfHistory']

                    specialColumnCount = SPECIAL_COLUMN_COUNTS.get(name, 0)

                    argsList = [
                        (absoluteIndex, fullHistory[absoluteIndex], fullHistory, name, model_type, dataPath, modelPath,
                            skipLastColumns, yearsOfHistory, ai, previousJsonFilePath, path, boost, bestParams_json_object,
                            specialColumnCount)
                        for absoluteIndex in rebuildIndices
                    ]

                    # Deliberately not a Pool: this step trains/uses the GPU
                    # (TCN/LSTM) and only ever ran with a single worker anyway
                    # (no parallelism gained), but forking a worker after the
                    # parent has touched CUDA leaves the child with a broken,
                    # non-reinitializable CUDA context - every GPU op in the
                    # fork then fails with CUDA_ERROR_NOT_INITIALIZED. Running
                    # directly in the parent avoids the fork entirely.
                    results = [process_single_history_entry_second_step(args) for args in argsList]

                    print("Finished second step: single process rebuild of history entries and ai or boosting method.")

                    # Find the matching numbers
                    update_matching_numbers(name=name, path=path)
                else:
                    print("No entries to process for: ", name)

                #return predictedSequence
        else:
            print("Prediction already made")
    else:
        print("Did not found entries")


def addRLTicketPrediction(listOfDecodedPredictions, dataPath, path, name,
                          specialColumnCount, bestParams_json_object, fallbackDrawSize=None,
                          cutoffDate=None):
    """
    Appends the RL Ticket Model row (README "Strategic Optimization"): learns
    ticket CONSTRUCTION from the interaction between the other models' rows
    and the payout structure (REINFORCE over the stored day JSONs, real
    keno/pick3 payouts as reward, hit count elsewhere). Called last, after
    the ensemble row, so today's full vote is part of its features - and
    deliberately after numberFrequency so its own row doesn't feed back into
    the votes it consumes. Pure numpy, ~1-2s, wall-clock capped, and its
    run() never raises (it degrades to vote-share ranking) - the try/except
    is for the surrounding plumbing. Shared by both places a day's rows are
    finalized (fresh daily prediction and history rebuild's second step).
    """
    if not bestParams_json_object.get("useRlTicket", True):
        print("RL Ticket Model disabled via useRlTicket - skipping")
        return listOfDecodedPredictions
    try:
        rlTicket.setModelPath(os.path.join(path, "data", "models", "rl_model"))
        rlTicket.setLearningRate(bestParams_json_object.get("rlTicketLearningRate", 0.05))
        rlTicket.setEpochs(bestParams_json_object.get("rlTicketEpochs", 30))
        rlTicket.setSamplesPerDay(bestParams_json_object.get("rlTicketSamplesPerDay", 32))
        rlTicket.setTrainDays(bestParams_json_object.get("rlTicketTrainDays", 120))
        rlTicket.setMaxTrainSeconds(bestParams_json_object.get("rlTicketMaxTrainSeconds", 60))

        mainLabels = helpers.get_unique_labels(dataPath)
        # get_unique_labels has no eurodreams branch and falls through to
        # 1-50, but the game draws mains from 1-40 - candidates 41-50 would
        # be forever-unseen numbers with a maximal staleness feature, which
        # noisy REINFORCE can happily rank into the published ticket. Fixed
        # here rather than in get_unique_labels: changing num_classes there
        # would invalidate every saved DL weight fingerprint for the game.
        if "eurodreams" in name:
            mainLabels = np.arange(1, 41)
        # Main-ticket size from an existing row (rows append special columns
        # after the mains); fall back to the caller's draw-derived size. The
        # RL row predicts main numbers only - specials have their own range
        # and payout logic that ticket construction can't arbitrage.
        rowWithTicket = next((row for row in listOfDecodedPredictions if row.get("predictions") and row["predictions"][0]), None)
        if rowWithTicket is not None:
            drawSize = len(rowWithTicket["predictions"][0]) - specialColumnCount
        elif fallbackDrawSize:
            drawSize = fallbackDrawSize
        else:
            print("RL Ticket Model skipped: no rows to derive the ticket size from")
            return listOfDecodedPredictions
        gameConfig = {
            "numberRange": (int(min(mainLabels)), int(max(mainLabels))) if "jokerplus" not in name else (0, 9),
            "drawSize": drawSize,
            "kenoSubsetSizes": getKenoSubsetSizes(name, bestParams_json_object),
            "isPick3": "pick3" in name,
            "perPositionClasses": 10,
            # During a history rebuild, day files AFTER the day being rebuilt
            # already exist on disk with realResults (step 1 writes them all
            # first) - the cutoff keeps the RL training window strictly
            # before the day, like skipRows does for every other model.
            "cutoffDate": cutoffDate,
        }
        rlRow = rlTicket.run(name, listOfDecodedPredictions,
                             os.path.join(path, "data", "database", name), gameConfig)
        if rlRow and rlRow.get("predictions"):
            listOfDecodedPredictions.append(rlRow)
    except Exception as e:
        print("Failed to perform RL Ticket Model prediction: ", e)
    return listOfDecodedPredictions


def addWeightedEnsemblePrediction(current_json_object, name, model_scores=None, bestParams_json_object=None):
    """
    Appends the score-weighted vote as its own ticket/row in newPrediction (so
    it shows up in the Model table next to every individual model's own
    prediction), instead of only existing as a separate chart. Skipped for
    Pick3, since it's positional and a frequency vote across models has no
    notion of position.

    For games with special columns (Euromillions star numbers, EuroDreams
    dream number, VikingLotto super viking - see SPECIAL_COLUMN_COUNTS), the
    main and special numbers are voted on and picked separately, then
    concatenated - the same way every individual model already keeps them
    separate via Helpers.run_model_with_special_column. A single flat vote
    over the whole row would let main-range numbers (a much bigger pool)
    crowd out the special slot(s), producing an out-of-range special number.

    For Keno, also generates the same use_5..use_10 sub-selections every
    individual model produces (previously missing entirely for this row),
    using Helpers.generate_subset_from_scores over the already-computed main
    vote so a subset is just "which of these 20 numbers", not a fresh vote.
    """
    if "pick3" in name:
        return

    bestParams_json_object = bestParams_json_object or {}

    predictions = current_json_object.get("newPrediction", [])
    ticket_size = next((len(model["predictions"][0]) for model in predictions if model.get("predictions")), 0)
    if ticket_size == 0:
        return

    specialColumnCount = next((count for game, count in SPECIAL_COLUMN_COUNTS.items() if game in name), 0)
    mainCount = ticket_size - specialColumnCount

    mainFrequencies, specialFrequencies = helpers.count_number_frequencies_by_position(
        current_json_object, mainCount, model_scores=model_scores)

    mainTicket = helpers.build_weighted_ensemble_prediction(mainFrequencies, mainCount)
    if not mainTicket:
        return

    ticketNumbers = mainTicket["predictions"][0]

    # Subsets (Keno only) are drawn from the main-number vote before any
    # special-column concatenation below - Keno has no special columns
    # (mutually exclusive with SPECIAL_COLUMN_COUNTS), so this ordering never
    # actually interacts with the special-column branch in practice.
    ensemblePredictions = [ticketNumbers] + [
        helpers.generate_subset_from_scores(
            mainFrequencies, ticketNumbers, subsetSize,
            mode=bestParams_json_object.get("weightedEnsembleSubsetMode", "softmax"),
            temperature=bestParams_json_object.get("weightedEnsembleSubsetTemperature", 0.5))
        for subsetSize in getKenoSubsetSizes(name, bestParams_json_object)
    ]

    if specialColumnCount > 0:
        specialTicket = helpers.build_weighted_ensemble_prediction(specialFrequencies, specialColumnCount)
        if not specialTicket:
            return
        ensemblePredictions[0] = ticketNumbers + specialTicket["predictions"][0]

    predictions.append({"name": "WeightedEnsemble Model", "predictions": ensemblePredictions})


def deepLearningMethod(listOfDecodedPredictions, newPredictionRaw, unique_labels, modelDisplayName="LSTM Base Model",
                       gameName="", kenoSubsetSizes=None):

    try:
        nthPredictions = {
            "name": modelDisplayName,
            "predictions": []
        }

        predicted_indices = np.argmax(newPredictionRaw, axis=-1)
        predicted_digits = [int(unique_labels[i]) for i in predicted_indices]

        nthPredictions["predictions"].append(predicted_digits)

        # Keno is the only game with playable sub-selections (5-10 numbers
        # out of the full 20-number ticket) - mirrors
        # HyperoptDeepLearning.py's deepLearningMethod, reusing the DL
        # prediction's per-number softmax probability as the subset score.
        #
        # kenoSubsetSizes comes from getKenoSubsetSizes (the hyperopt-tuned
        # use_5..use_10 flags), the same choice every statistical model and
        # XGBoost respects. This used to be a hardcoded range(5, 11), so the
        # DL rows placed a bet at every subset size regardless of which ones
        # tuning had actually selected - e.g. all six for Keno while
        # bestParams_keno.json enables only use_10, making those rows'
        # tracked results non-comparable with every other model's.
        if "keno" in gameName:
            try:
                number_scores = helpers.score_numbers_from_prediction(newPredictionRaw, unique_labels)
                for subset_size in (kenoSubsetSizes if kenoSubsetSizes is not None else range(5, 11)):
                    subset = helpers.generate_subset_from_scores(number_scores, predicted_digits, subset_size)
                    nthPredictions["predictions"].append(subset)
            except Exception as e:
                print("Failed to generate keno subsets: ", e)

        listOfDecodedPredictions.append(nthPredictions)

    except Exception as e:
        print("Failed to perform nth prediction: ", e)

    return listOfDecodedPredictions


# (model instance, bestParams key prefix, display name, model_type folder) -
# shared by both places the LSTM deep learning model runs, so
# TCN Base Model / UnifiedLstmTcn Model / UnifiedLstmGruTcn Model show up as
# their own additional rows next to LSTM Base Model, tracked independently in
# real-life results. Each model gets its own isolated try/except (like every
# other model in this file) so one failing doesn't take down the others.
# The fifth element holds per-model default overrides for the shared keys the
# loop reads with .get(): the Transformer defaults to a longer window (long-
# range attention is its whole point) and the Autoencoder defaults to zero
# label smoothing (its reconstruction NLL doubles as the anomaly signal -
# smoothing would bias the NLL floor and mask predictability spikes).
UNIFIED_DL_MODELS = [
    (tcn, "tcn", "TCN Base Model", "tcn_model", {}),
    (unifiedLstmTcn, "unifiedLstmTcn", "UnifiedLstmTcn Model", "unified_lstm_tcn_model", {}),
    (unifiedLstmGruTcn, "unifiedLstmGruTcn", "UnifiedLstmGruTcn Model", "unified_lstm_gru_tcn_model", {}),
    (transformer, "transformer", "Transformer Model", "transformer_model", {"windowSize": 30, "requiresFullAi": False}),
    (gnn, "gnn", "GNN Model", "gnn_model", {"requiresFullAi": False}),
    (autoencoderAnomaly, "autoencoder", "Autoencoder Model", "autoencoder_model", {"labelSmoothing": 0.0, "dropout": 0.2, "requiresFullAi": False}),
]


def lightDlModelsEnabled(bestParams_json_object):
    """
    True if at least one of the lightweight research models (the registry
    entries with requiresFullAi=False) is enabled by its use<Prefix> key.
    The cron runs with --ai off because the LSTM/TCN/unified trainings are
    what blow the memory/time budget - the Transformer/GNN/Autoencoder rows
    are an order of magnitude cheaper, so they get their own gate and run
    (in the same one-shot spawned child) even without --ai. Callers use this
    to skip spawning the child (one TF import, ~15-30s) when nothing in it
    would run anyway.
    """
    return any(
        bestParams_json_object.get("use" + prefix[0].upper() + prefix[1:], True)
        for _, prefix, _, _, modelDefaults in UNIFIED_DL_MODELS
        if not modelDefaults.get("requiresFullAi", True))


def runUnifiedDeepLearningModels(listOfDecodedPredictions, path, name, dataPath, skipLastColumns,
                                  bestParams_json_object, skipRows=0, years_back=None, specialColumnCount=0,
                                  fullAi=True):
    """
    Runs UnifiedLstmTcn Model and UnifiedLstmGruTcn Model (see
    src/UnifiedLstmTcn.py / src/UnifiedLstmGruTcn.py) alongside the existing
    LSTM Base Model row. Uses bestParams_json_object.get(..., default) - not
    bracket indexing - since these are new keys that won't exist in any
    bestParams_<game>.json written before this feature landed (the project
    already hit a real KeyError crash once from assuming a new key always
    exists).
    """
    autoencoderRan = False
    for model, prefix, displayName, modelTypeFolder, modelDefaults in UNIFIED_DL_MODELS:
        # Heavy rows (TCN/unified*) stay behind the --ai flag; the lightweight
        # research rows only need their own use<Prefix> toggle (default on).
        if modelDefaults.get("requiresFullAi", True) and not fullAi:
            continue
        if not bestParams_json_object.get("use" + prefix[0].upper() + prefix[1:], True):
            print(f"{displayName} disabled via use{prefix[0].upper() + prefix[1:]} - skipping")
            continue
        try:
            modelPath = os.path.join(path, "data", "models", modelTypeFolder)
            model.setDataPath(dataPath)
            model.setModelPath(modelPath)
            model.setLoadModelWeights(True)
            model.setBatchSize(bestParams_json_object.get(f"{prefix}_batchSize", 16))
            model.setEpochs(bestParams_json_object.get(f"{prefix}_epochs", 1000))
            # Architecture-specific setters are all hasattr-guarded: the
            # registry mixes TCN-, LSTM-, attention-, graph- and
            # autoencoder-shaped models, and an unconditional call to a
            # setter a model doesn't have would land in this try/except and
            # silently cost that model's row for the day.
            if hasattr(model, "setLstmUnits"):
                model.setLstmUnits(bestParams_json_object.get(f"{prefix}_lstmUnits", 64))
            if hasattr(model, "setTcnUnits"):
                model.setTcnUnits(bestParams_json_object.get(f"{prefix}_tcnUnits", 64))
            if hasattr(model, "setNumTcnLayers"):
                model.setNumTcnLayers(bestParams_json_object.get(f"{prefix}_numTcnLayers", 2))
            if hasattr(model, "setGruUnits"):
                model.setGruUnits(bestParams_json_object.get(f"{prefix}_gruUnits", 64))
            if hasattr(model, "setDModel"):
                model.setDModel(bestParams_json_object.get(f"{prefix}_dModel", 64))
            if hasattr(model, "setNumLayers"):
                model.setNumLayers(bestParams_json_object.get(f"{prefix}_numLayers", 2))
            if hasattr(model, "setFfnFactor"):
                model.setFfnFactor(bestParams_json_object.get(f"{prefix}_ffnFactor", 4))
            if hasattr(model, "setGcnUnits"):
                model.setGcnUnits(bestParams_json_object.get(f"{prefix}_gcnUnits", 32))
            if hasattr(model, "setNumGcnLayers"):
                model.setNumGcnLayers(bestParams_json_object.get(f"{prefix}_numGcnLayers", 2))
            if hasattr(model, "setEmbeddingDim"):
                model.setEmbeddingDim(bestParams_json_object.get(f"{prefix}_embeddingDim", 16))
            if hasattr(model, "setDecay"):
                model.setDecay(bestParams_json_object.get(f"{prefix}_decay", 0.999))
            if hasattr(model, "setLatentDim"):
                model.setLatentDim(bestParams_json_object.get(f"{prefix}_latentDim", 16))
            if hasattr(model, "setEncoderUnits"):
                model.setEncoderUnits(bestParams_json_object.get(f"{prefix}_encoderUnits", 64))
            if hasattr(model, "setNumEncoderLayers"):
                model.setNumEncoderLayers(bestParams_json_object.get(f"{prefix}_numEncoderLayers", 2))
            model.setDropout(bestParams_json_object.get(f"{prefix}_dropout", modelDefaults.get("dropout", 0.3)))
            model.setL2Regularization(bestParams_json_object.get(f"{prefix}_l2Regularization", 0.0005))
            model.setLearningRate(bestParams_json_object.get(f"{prefix}_learningRate", 0.001))
            model.setEarlyStopPatience(bestParams_json_object.get(f"{prefix}_earlyStopPatience", 20))
            model.setReduceLearningRatePatience(bestParams_json_object.get(f"{prefix}_reduceLearningRatePatience", 5))
            model.setReducedLearningRateFactor(bestParams_json_object.get(f"{prefix}_reduceLearningRateFactor", 0.5))
            model.setWindowSize(bestParams_json_object.get(f"{prefix}_windowSize", modelDefaults.get("windowSize", 20)))
            model.setPredictionWindowSize(model.window_size)
            model.setLabelSmoothing(bestParams_json_object.get(f"{prefix}_labelSmoothing", modelDefaults.get("labelSmoothing", 0.05)))
            if hasattr(model, "setNumHeads"):
                model.setNumHeads(bestParams_json_object.get(f"{prefix}_numHeads", 4))
            if hasattr(model, "setKeyDim"):
                model.setKeyDim(bestParams_json_object.get(f"{prefix}_keyDim", 32))

            latest_raw_predictions, unique_labels = model.run(
                name, skipLastColumns, skipRows=skipRows, years_back=years_back, specialColumnCount=specialColumnCount)

            listOfDecodedPredictions = deepLearningMethod(
                listOfDecodedPredictions, latest_raw_predictions.tolist(), unique_labels, modelDisplayName=displayName,
                gameName=name, kenoSubsetSizes=getKenoSubsetSizes(name, bestParams_json_object))

            if model is autoencoderAnomaly:
                autoencoderRan = True
        except Exception as e:
            print(f"Failed to perform {displayName} prediction: ", e)

    # Security layer (README "Unsupervised Anomaly Detection"): the
    # autoencoder's reconstruction NLL on each REAL draw, computed on the
    # model just trained above (same args -> cached, no second training).
    # A strongly negative rolling z (the real draw suddenly became easy to
    # reconstruct) is a predictability spike - the alert condition. Must run
    # here, inside the spawned DL child, because the trained model dies with
    # the child process.
    anomalyWatch = None
    if autoencoderRan:
        try:
            scores = autoencoderAnomaly.computeAnomalyScores(
                name, skipLastColumns, skipRows=skipRows, years_back=years_back,
                specialColumnCount=specialColumnCount)
            if scores:
                recent = [entry["z"] for entry in scores[-30:] if entry.get("z") is not None]
                anomalyWatch = {
                    "model": "Autoencoder Model",
                    "latest_z": scores[-1].get("z"),
                    "min_z_recent": round(min(recent), 3) if recent else None,
                    "alert": bool(recent and min(recent) < -3.0),
                    "scores": scores[-120:],
                }
        except Exception as e:
            print("Failed to compute autoencoder anomaly scores: ", e)

    return listOfDecodedPredictions, anomalyWatch


def statisticalMethod(listOfDecodedPredictions, dataPath, path, name, skipRows=0, skipLastColumns=0):

    bestParams_json_object = {
        "use_5":True,
        "use_6":True,
        "use_7":True,
        "use_8":True,
        "use_9":True,
        "use_10":True,
        "yearsOfHistory": 20,
        "useMarkov":False,
        "useMarkovMonteCarlo":False,
        "useMarkovBayesian":True,
        "usevMarkovBayesianEnhanced":True,
        "usePoissonMonteCarlo":False,
        "usePoissonMarkov":True,
        "useLaplaceMonteCarlo":False,
        "useHybridStatisticalModel":True,
        "markovSoftMaxTemperature":0.10002049510925136,
        "markovMinOccurences":9,
        "markovAlpha":0.20682688936213361,
        "markovRecencyWeight":1.591825953176242,
        "markovRecencyMode":"constant",
        "markovPairDecayFactor":0.34980042438509473,
        "markovSmoothingFactor":0.6342058116675424,
        "markovSubsetSelectionMode":"softmax",
        "markovBlendMode":"log",
        "markovOrder": 1,             
        "markovPairScoringWeight": 0, 
        "markovSortedPrediction": False, 
        "markovUsePairScoring": False,
        "markovMcSoftMaxTemperature":0.1,
        "markovMcMinOccurences":9,
        "markovMcAlpha":0.2,
        "markovMcRecencyWeight":1.0,
        "markovMcRecencyMode":"constant",
        "markovMcPairDecayFactor":0.3,
        "markovMcSmoothingFactor":0.6,
        "markovMcOrder": 1,
        "markovMcNumSimulations": 1000,
        "markovBayesianSoftMaxTemperature":0.24235148017270242,
        "markovBayesianMinOccurences":14,
        "markovBayesianAlpha":0.1452615422969012,
        "markovBayesianEnhancedSoftMaxTemperature":0.4244268734953605,
        "markovBayesianEnhancedAlpha":0.4015984866176651,
        "markovBayesianEnhancedMinOccurences":19,
        "poissonMonteCarloNumberOfSimulations":600,
        "poissonMonteCarloWeightFactor":0.836053158339262,
        "poissonMarkovWeight":0.48068822894893704,
        "poissonMarkovNumberOfSimulations":100,
        "laplaceMonteCarloNumberOfSimulations":900,
        "hybridStatisticalModelSoftMaxTemperature":0.918188590362822,
        "hybridStatisticalModelAlpha":0.7874157368729954,
        "hybridStatisticalModelMinOcurrences": 19,
        "hybridStatisticalModelNumberOfSimulations": 900,
        "batchSize": 8,
        "epochs": 1000,
        "num_lstm_layers": 1,
        "lstm_units": 32,
        "num_bidirectional_layers": 1,
        "bidirectional_lstm_units": 16,
        "optimizer_type": "adam",
        "learningRate": 0.005,
        "dropout": 0.2,
        "l2Regularization": 0.0001,
        "useFinalLSTMLayer": False,
        "earlyStopPatience": 100,
        "reduceLearningRatePatience": 50,
        "reduceLearningRateFactor": 0.9,
        "windowSize": 14,
        "labelSmoothing": 0.05
    }

    try:
        # Load hyperopt parameters if exists
        hyperoptParamsJsonFile = os.path.join(path, f"bestParams_{name}.json")
        if hyperoptParamsJsonFile and os.path.exists(hyperoptParamsJsonFile):
            with open(hyperoptParamsJsonFile, 'r') as openfile:
                # Merge over the in-code defaults rather than replacing them
                # outright - an existing bestParams_<game>.json predating a
                # newer default key (e.g. useMarkovMonteCarlo) would otherwise
                # crash every lookup of that key with a KeyError.
                bestParams_json_object.update(json.load(openfile))
    except Exception as e:
        print("Failed to parse parameter file: ", e)

    specialColumnCount = next((count for game, count in SPECIAL_COLUMN_COUNTS.items() if game in name), 0)

    # Pick3 is positional (digit order matters for straight/box/pair payouts), so
    # every model must return digits in drawn order instead of ascending-sorted.
    # Explicitly set every run (not just for pick3) since these model instances
    # are module-level singletons reused sequentially across games/history days.
    sortedPrediction = not ("pick3" in name)

    subsets = getKenoSubsetSizes(name, bestParams_json_object)


    
    
    if bestParams_json_object["useMarkov"]:
        try:
            # Markov
            #print("Performing Markov Prediction")
            markov.setDataPath(dataPath)
            markov.setSoftMAxTemperature(bestParams_json_object["markovSoftMaxTemperature"]) 
            markov.setMinOccurrences(bestParams_json_object["markovMinOccurences"]) 
            markov.setAlpha(bestParams_json_object["markovAlpha"])
            markov.setRecencyWeight(bestParams_json_object["markovRecencyWeight"])
            markov.setRecencyMode(bestParams_json_object["markovRecencyMode"])
            markov.setPairDecayFactor(bestParams_json_object["markovPairDecayFactor"])
            markov.setSmoothingFactor(bestParams_json_object["markovSmoothingFactor"])
            markov.setSubsetSelectionMode(bestParams_json_object["markovSubsetSelectionMode"])
            markov.setBlendMode(bestParams_json_object["markovBlendMode"])
            markov.setMarkovOrder(bestParams_json_object["markovOrder"])
            markov.setSortedPrediction(bestParams_json_object["markovSortedPrediction"])
            markov.setUsePairScoring(bestParams_json_object["markovUsePairScoring"])
            markov.setPairScoringWeight(bestParams_json_object["markovPairScoringWeight"])
            markov.clear()

            markovPrediction = {
                "name": "Markov Model",
                "predictions": []
            }

            markovSequence, markovSubsets = helpers.run_model_with_special_column(markov, generateSubsets=subsets, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)
            
            markovPrediction["predictions"].append(markovSequence)
            for key in markovSubsets:
                markovPrediction["predictions"].append(markovSubsets[key])

            listOfDecodedPredictions.append(markovPrediction)
        except Exception as e:
            print("Failed to perform Markov: ", e)

    if bestParams_json_object["useMarkovMonteCarlo"]:
        try:
            # Markov Monte Carlo
            #print("Performing Markov Monte Carlo Prediction")
            markovMcBase.setDataPath(dataPath)
            markovMcBase.setSoftMAxTemperature(bestParams_json_object["markovMcSoftMaxTemperature"])
            markovMcBase.setMinOccurrences(bestParams_json_object["markovMcMinOccurences"])
            markovMcBase.setAlpha(bestParams_json_object["markovMcAlpha"])
            markovMcBase.setRecencyWeight(bestParams_json_object["markovMcRecencyWeight"])
            markovMcBase.setRecencyMode(bestParams_json_object["markovMcRecencyMode"])
            markovMcBase.setPairDecayFactor(bestParams_json_object["markovMcPairDecayFactor"])
            markovMcBase.setSmoothingFactor(bestParams_json_object["markovMcSmoothingFactor"])
            markovMcBase.setMarkovOrder(bestParams_json_object["markovMcOrder"])
            markovMcBase.setSortedPrediction(sortedPrediction)
            markovMonteCarlo.setNumOfSimulations(bestParams_json_object["markovMcNumSimulations"])

            markovMcPrediction = {
                "name": "MarkovMonteCarlo Model",
                "predictions": []
            }

            markovMcSequence, markovMcSubsets = helpers.run_model_with_special_column(markovMonteCarlo, generateSubsets=subsets, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

            markovMcPrediction["predictions"].append(markovMcSequence)
            for key in markovMcSubsets:
                markovMcPrediction["predictions"].append(markovMcSubsets[key])

            listOfDecodedPredictions.append(markovMcPrediction)
        except Exception as e:
            print("Failed to perform Markov Monte Carlo: ", e)

    if not "pick3" in name and bestParams_json_object["useMarkovBayesian"]:
        try:
            # Markov Bayesian
            #print("Performing Markov Bayesian Prediction")
            markovBayesian.setDataPath(dataPath)
            markovBayesian.setSoftMAxTemperature(bestParams_json_object["markovBayesianSoftMaxTemperature"])
            markovBayesian.setAlpha(bestParams_json_object["markovBayesianAlpha"] )
            markovBayesian.setMinOccurrences(bestParams_json_object["markovBayesianMinOccurences"])
            markovBayesian.setSortedPrediction(sortedPrediction)
            markovBayesian.clear()

            markovBayesianPrediction = {
                "name": "MarkovBayesian Model",
                "predictions": []
            }

            markovBayesianSequence, markovBayesianSubsets = helpers.run_model_with_special_column(markovBayesian, generateSubsets=subsets, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)
            markovBayesianPrediction["predictions"].append(markovBayesianSequence)
            for key in markovBayesianSubsets:
                markovBayesianPrediction["predictions"].append(markovBayesianSubsets[key])

            listOfDecodedPredictions.append(markovBayesianPrediction)
        except Exception as e:
            print("Failed to perform Markov Bayesian: ", e)

    if not "pick3" in name and bestParams_json_object["usevMarkovBayesianEnhanced"]:
        try:
            # Markov Bayesian Enhanced
            #print("Performing Markov Bayesian Enhanced Prediction")
            markovBayesianEnhanced.setDataPath(dataPath)
            markovBayesianEnhanced.setSoftMAxTemperature(bestParams_json_object["markovBayesianEnhancedSoftMaxTemperature"])
            markovBayesianEnhanced.setAlpha(bestParams_json_object["markovBayesianEnhancedAlpha"])
            markovBayesianEnhanced.setMinOccurrences(bestParams_json_object["markovBayesianEnhancedMinOccurences"])
            markovBayesianEnhanced.setSortedPrediction(sortedPrediction)
            markovBayesianEnhanced.clear()

            markovBayesianEnhancedPrediction = {
                "name": "MarkovBayesianEnhanched Model",
                "predictions": []
            }

            markovBayesianEnhancedSequence, markovBayesianEnhancedSubsets = helpers.run_model_with_special_column(markovBayesianEnhanced, generateSubsets=subsets, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)
            markovBayesianEnhancedPrediction["predictions"].append(markovBayesianEnhancedSequence)
            for key in markovBayesianEnhancedSubsets:
                markovBayesianEnhancedPrediction["predictions"].append(markovBayesianEnhancedSubsets[key])

            listOfDecodedPredictions.append(markovBayesianEnhancedPrediction)
        except Exception as e:
            print("Failed to perform Markov Bayesian Enhanced: ", e)

    if bestParams_json_object["usePoissonMonteCarlo"]:
        try:
            # Poisson Distribution with Monte Carlo Analysis
            #print("Performing Poisson Monte Carlo Prediction")
            poissonMonteCarlo.setDataPath(dataPath)
            poissonMonteCarlo.setNumOfSimulations(bestParams_json_object["poissonMonteCarloNumberOfSimulations"])
            poissonMonteCarlo.setWeightFactor(bestParams_json_object["poissonMonteCarloWeightFactor"])
            poissonMonteCarlo.setSortedPrediction(sortedPrediction)
            poissonMonteCarlo.clear()

            poissonMonteCarloPrediction = {
                "name": "PoissonMonteCarlo Model",
                "predictions": []
            }

            poissonMonteCarloSequence, poissonMonteCarloSubsets = helpers.run_model_with_special_column(poissonMonteCarlo, generateSubsets=subsets, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

            poissonMonteCarloPrediction["predictions"].append(poissonMonteCarloSequence)
            for key in poissonMonteCarloSubsets:
                poissonMonteCarloPrediction["predictions"].append(poissonMonteCarloSubsets[key])

            listOfDecodedPredictions.append(poissonMonteCarloPrediction)    
        except Exception as e:
            print("Failed to perform Poisson Distribution with Monte Carlo Analysis: ", e)

    if not "pick3" in name and bestParams_json_object["usePoissonMarkov"]:
        try:
            # Poisson-Markov Distribution
            #print("Performing Poisson-Markov Prediction")
            poissonMarkov.setDataPath(dataPath)
            poissonMarkov.setWeights(poisson_weight=bestParams_json_object["poissonMarkovWeight"], markov_weight=(1-bestParams_json_object["poissonMarkovWeight"]))
            poissonMarkov.setNumberOfSimulations(bestParams_json_object["poissonMarkovNumberOfSimulations"])
            poissonMarkov.setSortedPrediction(sortedPrediction)

            poissonMarkovPrediction = {
                "name": "PoissonMarkov Model",
                "predictions": []
            }

            poissonMarkovSequence, poissonMarkovSubsets = helpers.run_model_with_special_column(poissonMarkov, generateSubsets=subsets, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

            poissonMarkovPrediction["predictions"].append(poissonMarkovSequence)
            for key in poissonMarkovSubsets:
                poissonMarkovPrediction["predictions"].append(poissonMarkovSubsets[key])

            listOfDecodedPredictions.append(poissonMarkovPrediction)    
        except Exception as e:
            print("Failed to perform Poisson-Markov Distribution: ", e)

    if bestParams_json_object["useLaplaceMonteCarlo"]:
        try:
            # Laplace Distribution with Monte Carlo Analysis
            #print("Performing Laplace Monte Carlo Prediction")
            laplaceMonteCarlo.setDataPath(dataPath)
            laplaceMonteCarlo.setNumOfSimulations(bestParams_json_object["laplaceMonteCarloNumberOfSimulations"])
            laplaceMonteCarlo.setSortedPrediction(sortedPrediction)
            laplaceMonteCarlo.clear()

            laplaceMonteCarloPrediction = {
                "name": "LaplaceMonteCarlo Model",
                "predictions": []
            }


            laplaceMonteCarloSequence, laplaceMonteCarloSubsets = helpers.run_model_with_special_column(laplaceMonteCarlo, generateSubsets=subsets, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)
            laplaceMonteCarloPrediction["predictions"].append(laplaceMonteCarloSequence)
            for key in laplaceMonteCarloSubsets:
                laplaceMonteCarloPrediction["predictions"].append(laplaceMonteCarloSubsets[key])
            
            listOfDecodedPredictions.append(laplaceMonteCarloPrediction)
        except Exception as e:
            print("Failed to perform Laplace Distribution with Monte Carlo Analysis: ", e)

    if not "pick3" in name and bestParams_json_object["useHybridStatisticalModel"]:
        try:
            # Hybrid Statistical Model
            #print("Performing Hybrid Statistical Model Prediction")
            hybridStatisticalModel.setDataPath(dataPath)
            hybridStatisticalModel.setSoftMaxTemperature(bestParams_json_object["hybridStatisticalModelSoftMaxTemperature"])
            hybridStatisticalModel.setAlpha(bestParams_json_object["hybridStatisticalModelAlpha"])
            hybridStatisticalModel.setMinOccurrences(bestParams_json_object["hybridStatisticalModelMinOcurrences"])
            hybridStatisticalModel.setNumberOfSimulations(bestParams_json_object["hybridStatisticalModelNumberOfSimulations"])
            hybridStatisticalModel.setSortedPrediction(sortedPrediction)
            hybridStatisticalModel.clear()

            hybridStatisticalModelPrediction = {
                "name": "HybridStatisticalModel",
                "predictions": []
            }

            hybridStatisticalModelSequence, hybridStatisticalModelSubsets = helpers.run_model_with_special_column(hybridStatisticalModel, generateSubsets=subsets, skipRows=skipRows, skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)
            hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSequence)
            for key in hybridStatisticalModelSubsets:
                hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSubsets[key])
            
            listOfDecodedPredictions.append(hybridStatisticalModelPrediction)
        except Exception as e:
            print("Failed to perform Hybrid Statistical Model: ", e)

    # Phase 1 stacking meta-learner(s) (see TrainMetaLearner.py): blends each
    # base model's own per-number score into one learned P(drawn), instead of
    # the flat/weighted vote WeightedEnsemble Model uses. Skipped gracefully
    # if this game hasn't been trained yet (no meta_learner*.joblib), and
    # always skipped for Pick3 for the same reason WeightedEnsemble Model is
    # (positional game, a per-number score ranking has no notion of digit
    # order).
    #
    # MetaLearnerV2 Model (lens diversity, see README) reuses the exact same
    # base-model scores as MetaLearner Model - only the trained model class
    # differs (GradientBoostingClassifier vs LogisticRegression, see
    # TrainMetaLearner.py) - so both are computed from one shared score pass
    # instead of scoring every base model twice.
    if not "pick3" in name:
        metaLearnerPath = os.path.join(path, "data", "models", name, "meta_learner.joblib")
        metaLearnerV2Path = os.path.join(path, "data", "models", name, "meta_learner_v2.joblib")
        # Quantum meta-learners (README's quantum research track, trained by
        # the same TrainMetaLearner.py run): identical artifact shape, so they
        # are served by the exact same runMetaLearnerVariant below - only the
        # trained model class inside differs (see src/QuantumModels.py).
        quantumMetaLearnerPath = os.path.join(path, "data", "models", name, "quantum_meta_learner.joblib")
        quantumVqcMetaLearnerPath = os.path.join(path, "data", "models", name, "quantum_vqc_meta_learner.joblib")

        if any(os.path.exists(p) for p in (metaLearnerPath, metaLearnerV2Path, quantumMetaLearnerPath, quantumVqcMetaLearnerPath)):
            try:
                # Re-apply this game's tuned params to every base model
                # feeding the meta-learner(s), independent of whether that
                # model's own useX flag happens to be enabled above - these
                # are module-level singletons reused across games/history
                # days, so without this they could still hold a previous
                # game's config.
                markov.setDataPath(dataPath)
                markov.setSoftMAxTemperature(bestParams_json_object["markovSoftMaxTemperature"])
                markov.setMinOccurrences(bestParams_json_object["markovMinOccurences"])
                markov.setAlpha(bestParams_json_object["markovAlpha"])
                markov.setRecencyWeight(bestParams_json_object["markovRecencyWeight"])
                markov.setRecencyMode(bestParams_json_object["markovRecencyMode"])
                markov.setPairDecayFactor(bestParams_json_object["markovPairDecayFactor"])
                markov.setSmoothingFactor(bestParams_json_object["markovSmoothingFactor"])
                markov.setSubsetSelectionMode(bestParams_json_object["markovSubsetSelectionMode"])
                markov.setBlendMode(bestParams_json_object["markovBlendMode"])
                markov.setMarkovOrder(bestParams_json_object["markovOrder"])
                markov.setSortedPrediction(bestParams_json_object["markovSortedPrediction"])
                markov.setUsePairScoring(bestParams_json_object["markovUsePairScoring"])
                markov.setPairScoringWeight(bestParams_json_object["markovPairScoringWeight"])

                markovMcBase.setDataPath(dataPath)
                markovMcBase.setSoftMAxTemperature(bestParams_json_object["markovMcSoftMaxTemperature"])
                markovMcBase.setMinOccurrences(bestParams_json_object["markovMcMinOccurences"])
                markovMcBase.setAlpha(bestParams_json_object["markovMcAlpha"])
                markovMcBase.setRecencyWeight(bestParams_json_object["markovMcRecencyWeight"])
                markovMcBase.setRecencyMode(bestParams_json_object["markovMcRecencyMode"])
                markovMcBase.setPairDecayFactor(bestParams_json_object["markovMcPairDecayFactor"])
                markovMcBase.setSmoothingFactor(bestParams_json_object["markovMcSmoothingFactor"])
                markovMcBase.setMarkovOrder(bestParams_json_object["markovMcOrder"])
                markovMcBase.setSortedPrediction(sortedPrediction)
                markovMonteCarlo.setNumOfSimulations(bestParams_json_object["markovMcNumSimulations"])

                markovBayesian.setDataPath(dataPath)
                markovBayesian.setSoftMAxTemperature(bestParams_json_object["markovBayesianSoftMaxTemperature"])
                markovBayesian.setAlpha(bestParams_json_object["markovBayesianAlpha"])
                markovBayesian.setMinOccurrences(bestParams_json_object["markovBayesianMinOccurences"])
                markovBayesian.setSortedPrediction(sortedPrediction)

                markovBayesianEnhanced.setDataPath(dataPath)
                markovBayesianEnhanced.setSoftMAxTemperature(bestParams_json_object["markovBayesianEnhancedSoftMaxTemperature"])
                markovBayesianEnhanced.setAlpha(bestParams_json_object["markovBayesianEnhancedAlpha"])
                markovBayesianEnhanced.setMinOccurrences(bestParams_json_object["markovBayesianEnhancedMinOccurences"])
                markovBayesianEnhanced.setSortedPrediction(sortedPrediction)

                poissonMonteCarlo.setDataPath(dataPath)
                poissonMonteCarlo.setNumOfSimulations(bestParams_json_object["poissonMonteCarloNumberOfSimulations"])
                poissonMonteCarlo.setWeightFactor(bestParams_json_object["poissonMonteCarloWeightFactor"])
                poissonMonteCarlo.setSortedPrediction(sortedPrediction)

                poissonMarkov.setDataPath(dataPath)
                poissonMarkov.setWeights(poisson_weight=bestParams_json_object["poissonMarkovWeight"], markov_weight=(1 - bestParams_json_object["poissonMarkovWeight"]))
                poissonMarkov.setNumberOfSimulations(bestParams_json_object["poissonMarkovNumberOfSimulations"])
                poissonMarkov.setSortedPrediction(sortedPrediction)

                laplaceMonteCarlo.setDataPath(dataPath)
                laplaceMonteCarlo.setNumOfSimulations(bestParams_json_object["laplaceMonteCarloNumberOfSimulations"])
                laplaceMonteCarlo.setSortedPrediction(sortedPrediction)

                # Same configuration boostingMethod applies (and the same
                # ModelFactory.build_models applies when TrainMetaLearner.py
                # builds the training data), so the boosting feature the
                # meta-learner is served here is the one it was trained on.
                # .get() with defaults: these keys won't exist in a
                # bestParams_<game>.json written before HyperoptBoost.py ran.
                # Note the meta-learner scores main and special columns in two
                # separate passes, so this fits more than once per prediction -
                # src/XGBoost.py's fit cache keeps that to one fit per range.
                apply_boosting_params(xgboostPredictor, bestParams_json_object, "xgBoost")
                xgboostPredictor.setDataPath(dataPath)
                xgboostPredictor.setSortedPrediction(sortedPrediction)
                # Measured on this 16-core box (eurodreams, 50 nested binary
                # fits, estimators=50): 1 thread 2.8s, 4 threads 6.7s, 15
                # threads did not finish in 9+ minutes - the per-number
                # binary fits are far too small for OpenMP, whose spin-wait
                # overhead dominates and oversubscribes against everything
                # else. Same reasoning as ModelFactory.build_models'
                # setNumThreads(1).
                xgboostPredictor.setNumThreads(1)
                xgboostPredictor.setSaveModels(False)

                modelInstances = {
                    "Markov Model": markov,
                    "MarkovMonteCarlo Model": markovMonteCarlo,
                    "MarkovBayesian Model": markovBayesian,
                    "MarkovBayesianEnhanched Model": markovBayesianEnhanced,
                    "PoissonMonteCarlo Model": poissonMonteCarlo,
                    "PoissonMarkov Model": poissonMarkov,
                    "LaplaceMonteCarlo Model": laplaceMonteCarlo,
                    "XGBoost Model": xgboostPredictor,
                }

                def scoreNumbersFor(featureNames, skipLast, special):
                    return {
                        featureName: modelInstances[featureName].score_numbers(
                            skipRows=skipRows, skipLastColumns=skipLast, specialColumnCount=special)
                        for featureName in featureNames if featureName in modelInstances
                    }

                def rankByModel(model, featureNames, perModelScores, numberRange):
                    featureMatrix = [
                        [perModelScores.get(featureName, {}).get(number, 0.0) for featureName in featureNames]
                        for number in numberRange
                    ]
                    probabilities = model.predict_proba(featureMatrix)[:, 1]
                    return probabilities, [n for _, n in sorted(zip(probabilities, numberRange), reverse=True)]

                # Cached by feature-name tuple so MetaLearner Model and
                # MetaLearnerV2 Model (same 7 base models, different trained
                # classifier - see TrainMetaLearner.py) don't score everything
                # twice when both artifacts exist and share the same features.
                mainScoresByFeatures = {}
                specialScoresByFeatures = {}

                def runMetaLearnerVariant(artifactPath, displayName, subsetModeKey, subsetTemperatureKey):
                    if not os.path.exists(artifactPath):
                        return
                    try:
                        artifact = metaLearnerCache.get(artifactPath)
                        if artifact is None:
                            artifact = joblib.load(artifactPath)
                            metaLearnerCache[artifactPath] = artifact

                        featureNames = tuple(artifact["feature_names"])

                        # Main numbers: same main-only call (drops the special
                        # column(s) via skipLastColumns) every individual
                        # model's own run() makes via
                        # Helpers.run_model_with_special_column - see
                        # Backtester.py's collect_scores split for why this
                        # must not also pass specialColumnCount in the same
                        # call.
                        if featureNames not in mainScoresByFeatures:
                            mainScoresByFeatures[featureNames] = scoreNumbersFor(
                                featureNames, specialColumnCount if specialColumnCount > 0 else skipLastColumns, 0)
                        perModelScores = mainScoresByFeatures[featureNames]

                        minNumber = artifact["min_number"]
                        maxNumber = artifact["max_number"]
                        numberRange = list(range(minNumber, maxNumber + 1))
                        probabilities, rankedNumbers = rankByModel(artifact["model"], featureNames, perModelScores, numberRange)
                        ticket = sorted(rankedNumbers[:artifact["draw_size"]])

                        if specialColumnCount > 0 and "special_model" in artifact:
                            if featureNames not in specialScoresByFeatures:
                                specialScoresByFeatures[featureNames] = scoreNumbersFor(featureNames, 0, specialColumnCount)
                            perModelSpecialScores = specialScoresByFeatures[featureNames]
                            specialNumberRange = list(range(artifact["special_min_number"], artifact["special_max_number"] + 1))
                            _, rankedSpecialNumbers = rankByModel(artifact["special_model"], featureNames, perModelSpecialScores, specialNumberRange)
                            ticket = ticket + sorted(rankedSpecialNumbers[:artifact["special_draw_size"]])

                        predictions = [ticket]
                        mainScoreByNumber = dict(zip(numberRange, probabilities))
                        for subsetSize in subsets:
                            predictions.append(helpers.generate_subset_from_scores(
                                mainScoreByNumber, ticket, subsetSize,
                                mode=bestParams_json_object.get(subsetModeKey, "softmax"),
                                temperature=bestParams_json_object.get(subsetTemperatureKey, 0.5)))

                        listOfDecodedPredictions.append({"name": displayName, "predictions": predictions})
                    except Exception as e:
                        print(f"Failed to perform {displayName} prediction: ", e)

                runMetaLearnerVariant(metaLearnerPath, "MetaLearner Model", "metaLearnerSubsetMode", "metaLearnerSubsetTemperature")
                runMetaLearnerVariant(metaLearnerV2Path, "MetaLearnerV2 Model", "metaLearnerV2SubsetMode", "metaLearnerV2SubsetTemperature")
                # Quantum rows share the classical rows' feature names, so the
                # score caches above mean all four variants cost one scoring
                # pass; runMetaLearnerVariant's artifact-missing early-return
                # and per-variant try/except give each row a graceful skip and
                # failure isolation (an artifact that fails to unpickle - e.g.
                # src/QuantumModels.py missing on an older checkout - only
                # loses its own row).
                runMetaLearnerVariant(quantumMetaLearnerPath, "QuantumMetaLearner Model", "quantumMetaLearnerSubsetMode", "quantumMetaLearnerSubsetTemperature")
                runMetaLearnerVariant(quantumVqcMetaLearnerPath, "QuantumVQC Model", "quantumVqcSubsetMode", "quantumVqcSubsetTemperature")
            except Exception as e:
                print("Failed to perform Meta-Learner prediction: ", e)

    return listOfDecodedPredictions

# (instance, bestParams key prefix, display name, use-flag key, is_multi_label).
#
# Three gradient-boosting libraries x two formulations, each its own tracked
# row so the comparison is like-for-like: all six share the identical feature
# window, ticket construction and subset generator (see src/BoostingBase.py),
# so a difference between rows is attributable to the library or the
# formulation, not to incidental plumbing differences.
#
# XGBoost Model keeps the unprefixed-by-history "xgBoost" prefix and the
# "useBoost" flag it already has in every bestParams_<game>.json, so its
# tracked history stays continuous. Each model gets its own isolated
# try/except (like every other model in this file) so one failing - including
# one whose library isn't installed - doesn't take down the others.
BOOSTING_MODELS = [
    # Deliberately the module-level xgboostPredictor singleton, not a fresh
    # instance: statisticalMethod's meta-learner block scores the very same
    # object for this game/day, so sharing it lets BoostingBase's fit cache
    # serve both from one training run.
    (xgboostPredictor, "xgBoost", "XGBoost Model", "useBoost", False),
    (XGBoostMultiLabelPredictor(), "xgBoostMl", "XGBoostMultiLabel Model", "useXgBoostMultiLabel", True),
    (LightGBMPredictor(), "lightGbm", "LightGBM Model", "useLightGbm", False),
    (LightGBMMultiLabelPredictor(), "lightGbmMl", "LightGBMMultiLabel Model", "useLightGbmMultiLabel", True),
    (CatBoostPredictor(), "catBoost", "CatBoost Model", "useCatBoost", False),
    (CatBoostMultiLabelPredictor(), "catBoostMl", "CatBoostMultiLabel Model", "useCatBoostMultiLabel", True),
]


def boostingMethod(listOfDecodedPredictions, dataPath, path, name, skipRows=0, skipLastColumns=0):
    """
    Every gradient-boosting model as its own tracked prediction row, driven
    exactly like the statistical models above: the same
    Helpers.run_model_with_special_column call (so Euromillions' star columns /
    EuroDreams' dream number / VikingLotto's super viking are modeled
    independently, and Lotto's unplayed bonus column is dropped), the same
    getKenoSubsetSizes subset choice, and the same {size: subset} dict.

    The multi-label models are skipped for Pick3: they model set membership
    ("is this number in the next draw"), which cannot represent digit order or
    the repeated digits Pick3 routinely produces - the same reason
    WeightedEnsemble/MetaLearner are skipped there.
    """
    bestParams_json_object = {
        "use_5": True, "use_6": True, "use_7": True,
        "use_8": True, "use_9": True, "use_10": True,
    }
    # Every model runs unless its flag is explicitly disabled - hyperopt tunes
    # each model but never turns one off, so their real-life results stay
    # comparable over time.
    bestParams_json_object.update({useKey: True for _, _, _, useKey, _ in BOOSTING_MODELS})

    try:
        # Load hyperopt parameters if exists
        hyperoptParamsJsonFile = os.path.join(path, f"bestParams_{name}.json")
        if hyperoptParamsJsonFile and os.path.exists(hyperoptParamsJsonFile):
            with open(hyperoptParamsJsonFile, 'r') as openfile:
                # Merged over the in-code defaults (not replacing them) for the
                # same reason statisticalMethod does it: a
                # bestParams_<game>.json written before a newer key exists
                # would otherwise KeyError on every lookup of that key.
                bestParams_json_object.update(json.load(openfile))
    except Exception as e:
        print("Failed to parse parameter file in boost method: ", e)

    subsets = getKenoSubsetSizes(name, bestParams_json_object)
    specialColumnCount = next((count for game, count in SPECIAL_COLUMN_COUNTS.items() if game in name), 0)
    isPick3 = "pick3" in name

    for model, prefix, displayName, useKey, isMultiLabel in BOOSTING_MODELS:
        if not bestParams_json_object.get(useKey, True):
            continue

        if isPick3 and isMultiLabel:
            continue

        try:
            apply_boosting_params(model, bestParams_json_object, prefix)
            model.setDataPath(dataPath)
            model.setModelPath(os.path.join(path, "data", "models", f"{prefix.lower()}_{name}_models"))
            # Pick3 is positional - digits stay in drawn order, duplicates
            # included. Every other game gets a sorted ticket of distinct
            # numbers. Set explicitly each run: these are module-level
            # singletons reused across games/history days.
            model.setSortedPrediction(not isPick3)
            # This step runs single-process, but giving the library the
            # machine's cores was measured to be actively harmful on this
            # box: the per-number binary fits are so small that OpenMP
            # spin-wait overhead dominates (eurodreams, estimators=50:
            # 1 thread 2.8s, 4 threads 6.7s, 15 threads >9 minutes). One
            # thread, matching ModelFactory.build_models.
            model.setNumThreads(1)
            model.setSaveModels(True)
            # No clear() here (unlike the statistical models, whose clear()
            # drops accumulated counts): BoostingBase keys its fit cache on the
            # data slice *and* every training hyperparameter, so a stale fit
            # can never be reused. Keeping it means the main-numbers fit the
            # meta-learner block already did for this game/day is reused rather
            # than retrained.

            sequence, modelSubsets = helpers.run_model_with_special_column(
                model, generateSubsets=subsets, skipRows=skipRows,
                skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

            prediction = {"name": displayName, "predictions": [sequence]}
            for key in modelSubsets:
                prediction["predictions"].append(modelSubsets[key])

            listOfDecodedPredictions.append(prediction)
        except Exception as e:
            print(f"Failed to perform {displayName}: ", e)

    return listOfDecodedPredictions


if __name__ == "__main__":
    
    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)

    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)

    try:
        time.sleep(100)
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
        parser.add_argument(
            '-a', '--ai', type=helpers.str2bool, default=False,
            help='Enable the HEAVY deep learning models (LSTM/TCN/Unified). '
                 'Off by default for now: their training is what grows past '
                 'the container memory limit and gets OOM-killed. The '
                 'lightweight research models (Transformer/GNN/Autoencoder) '
                 'run regardless, gated by their own useTransformer/useGnn/'
                 'useAutoencoder keys in bestParams_<game>.json. ANDed with '
                 'each game\'s own ai flag in the datasets list.')
        parser.add_argument(
            '-b', '--boost', type=helpers.str2bool, default=True,
            help='Enable the gradient boosting models. ANDed with each '
                 'game\'s own boost flag in the datasets list.')
        parser.add_argument('-d', '--days', type=int, default=31)
        parser.add_argument('-s', '--save', type=helpers.str2bool, default=True)
        parser.add_argument(
            '-g', '--games',
            type=str,
            default="euromillions,lotto,eurodreams,keno,pick3,vikinglotto",
            help='Comma-separated list of games, e.g. "euromillions,lotto,..."'
        )
        args = parser.parse_args()

        print_intro()

        current_year = datetime.now().year
        print("Current Year:", current_year)

        daysToRebuild = int(args.days)
        rebuildHistory = bool(args.rebuild_history)
        pushToGit = bool(args.save)
        aiEnabled = bool(args.ai)
        boostEnabled = bool(args.boost)

        print("Deep learning enabled: ", aiEnabled)
        print("Boosting enabled: ", boostEnabled)

        print("Push to git: ", pushToGit)

        # Convert the comma-separated string into a clean list
        games = [g.strip() for g in args.games.split(',') if g.strip()]

        print("Selected games:", games)

        path = os.getcwd()

        # Here we can force disable ai and boost methods. If enabled here we let hyperopt decide
        # (boost is on for every game now that XGBoost Model runs on the same
        # interface as the statistical/DL models - it gets its own tracked
        # prediction row, tuned by HyperoptBoost.py, so its real-life results
        # can be compared against every other method.)
        datasets = [
            # (dataset_name, model_type, skip_last_columns, ai, xgboost)
            ("euromillions", "lstm_model", 0, True, True),
            ("lotto", "lstm_model", 1, True, True),
            ("eurodreams", "lstm_model", 0, True, True),
            #("jokerplus", "lstm_model", 1, False, True),
            ("keno", "lstm_model", 0, True, True),    # DL models now generate Keno subsets too (see deepLearningMethod)
            ("pick3", "lstm_model", 0, True, True),
            ("vikinglotto", "lstm_model", 0, True, True),
        ]

        for dataset_name, model_type, skip_last_columns, ai, boost in datasets:
            try:
                if dataset_name in games:
                    print(f"\n{dataset_name.capitalize()}")
                    modelPath = os.path.join(path, "data", "models", model_type)
                    dataPath = os.path.join(path, "data", "trainingData", dataset_name)
                    file = f"{dataset_name}-gamedata-NL-{current_year}.csv"

                    kwargs_wget = {
                        "folder": dataPath,
                        "file": file
                    }

                    # Lets check if file exists
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
                    else:
                        command.run("wget -P {folder} https://prdlnboppreportsst.blob.core.windows.net/legal-reports/{file}".format(**kwargs_wget), verbose=False)

                    # Predict with hyperopt params
                    predict(dataset_name, model_type, dataPath, modelPath, skipLastColumns=skip_last_columns, daysToRebuild=daysToRebuild, ai=(ai and aiEnabled), boost=(boost and boostEnabled), forceRebuild=rebuildHistory)
                else:
                    pass

            except Exception as e:
                print(f"Failed to predict {dataset_name.capitalize()}: {e}")

        
        print("Finished with predictions")

        # Per-game/per-model performance summary over all scored history -
        # rendered by the web UI's History page (see server.js /database) and
        # committed alongside the prediction jsons below.
        try:
            helpers.generate_model_performance_report(os.path.join(path, "data", "database"))
        except Exception as e:
            print("Failed to generate model performance report: ", e)

        # try:
        #     helpers.generatePredictionTextFile(os.path.join(path, "data", "database"))
        # except Exception as e:
        #     print("Failed to generate txt file:", e)

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
                helpers.git_push()
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()  # Ensure the lock is removed even if an error occurs
    
    

    
