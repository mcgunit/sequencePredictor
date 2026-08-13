import os, argparse, json, sys, time
import numpy as np
import subprocess
import joblib

from art import text2art
from datetime import datetime
from multiprocessing import Pool, cpu_count

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
from src.XGBoost import XGBoostKenoPredictor
from src.Command import Command
from src.Helpers import Helpers
from src.DataFetcher import DataFetcher

tcn = TCNModel()
lstm = LSTMModel()
unifiedLstmTcn = UnifiedLstmTcnModel()
unifiedLstmGruTcn = UnifiedLstmGruTcnModel()
markov = Markov()
markovMcBase = Markov()
markovMonteCarlo = MarkovMonteCarlo(markovMcBase)
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
    """Checks if another instance is running based on the lock file."""
    return os.path.exists(LOCK_FILE)

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


def process_single_history_entry_second_step(args):
    """
    Second step to perform methods where we can not process multible files at the same time
    """
    
    (historyIndex, historyEntry, historyData, name, model_type, dataPath, modelPath,
     skipLastColumns, years_back, ai, previousJsonFilePath, path, boost, bestParams_json_object,
     specialColumnCount) = args

    # model_type is always "lstm_model" here (see the __main__ datasets list) -
    # TCN now runs as its own additional row via UNIFIED_DL_MODELS instead of
    # this either/or.
    modelToUse = lstm
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

    if ai:
        # Set the fundation for deepLearningMethod
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

        latest_raw_predictions, unique_labels = modelToUse.run(
            name, skipLastColumns, skipRows=len(historyData)-historyIndex, years_back=years_back,
            specialColumnCount=specialColumnCount)

        predictedSequence = latest_raw_predictions.tolist()
        #unique_labels = unique_labels.tolist()
        current_json_object["newPredictionRaw"] = predictedSequence
        listOfDecodedPredictions = deepLearningMethod(listOfDecodedPredictions, predictedSequence, unique_labels, gameName=name)
        listOfDecodedPredictions = runUnifiedDeepLearningModels(
            listOfDecodedPredictions, path, name, dataPath, skipLastColumns, bestParams_json_object,
            skipRows=len(historyData) - historyIndex, years_back=years_back, specialColumnCount=specialColumnCount)
    else:
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

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile, indent=2)



    return jsonFilePath


def predict(name, model_type ,dataPath, modelPath, skipLastColumns=0, daysToRebuild=31, ai=False, boost=False):
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


        # Compare the latest result with the previous new prediction
        if not os.path.exists(jsonFilePath):
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
            if previousEntry is not None:
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

                    if ai:
                        try:
                            # Train and do a new prediction
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
                            yearsOfHistory = bestParams_json_object['yearsOfHistory']
                            specialColumnCount = SPECIAL_COLUMN_COUNTS.get(name, 0)
                            latest_raw_predictions, unique_labels = modelToUse.run(
                                name, skipLastColumns, years_back=yearsOfHistory, specialColumnCount=specialColumnCount)

                            predictedSequence = latest_raw_predictions.tolist()


                            # Save the current prediction as newPrediction
                            current_json_object["newPredictionRaw"] = predictedSequence
                            current_json_object["labels"] = unique_labels


                            listOfDecodedPredictions = deepLearningMethod(listOfDecodedPredictions, current_json_object["newPredictionRaw"], unique_labels, gameName=name)
                            listOfDecodedPredictions = runUnifiedDeepLearningModels(
                                listOfDecodedPredictions, path, name, dataPath, skipLastColumns, bestParams_json_object,
                                years_back=yearsOfHistory, specialColumnCount=specialColumnCount)
                        except Exception as e:
                            print("Failed to perform deep learning method: ", e)
                    else:
                        yearsOfHistory = bestParams_json_object['yearsOfHistory']
                        _, _, _, _, _, _, _, unique_labels = helpers.load_data(dataPath, skipLastColumns, years_back=yearsOfHistory)
                        unique_labels = unique_labels.tolist()


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

                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile, indent=2)

                    #return predictedSequence
                
            if doNewPrediction:
                print(f"No previous prediction file found, Cannot compare. Recreating {daysToRebuild} days of history")

                # Check if there is not a gap or so
                historyData = helpers.getLatestPrediction(dataPath, dateRange=daysToRebuild)
                #print("History data: ", historyData)

                dateOffset = 0 # index of last entry

                print("Date to start from: ", historyData[dateOffset])
                

                previousJsonFilePath = ""

                # Search for existing history
                for index, historyEntry in enumerate(historyData):
                    entryDate = historyEntry[0]
                    entryResult = historyEntry[1]
                    jsonFileName = f"{entryDate.year}-{entryDate.month}-{entryDate.day}.json"
                    #print(jsonFileName, ":", entryResult)
                    jsonFilePath = os.path.join(path, "data", "database", name, jsonFileName)
                    #print("Does file exist: ", os.path.exists(jsonFilePath))
                    if os.path.exists(jsonFilePath):
                        dateOffset = index
                        previousJsonFilePath = jsonFilePath
                        break
                
                # Remove all elements starting from dateOffset index
                #print("Date offset: ", dateOffset)
                historyData = historyData[dateOffset:]  # Keep elements after dateOffset because newer elements comes after the dateOffset index                
                #print("History to rebuild: ", historyData)

                # Only historyIndex 0 has a legitimate pre-existing "previous"
                # file (the one found above, right before the rebuild range).
                # Every other entry's true previous file is produced by a
                # sibling worker in this same parallel batch - reading it here
                # would race against that worker's own concurrent write to
                # the exact same path (and is logically wrong regardless,
                # since it'd compare every day against one fixed file instead
                # of the correct rolling previous day).
                argsList = [
                    (historyIndex, historyEntry, historyData, name, dataPath,
                    previousJsonFilePath if historyIndex == 0 else "", path, skipLastColumns)
                    for historyIndex, historyEntry in enumerate(historyData)
                ]

                #print("Argslist: ", len(argsList))

                if len(argsList) > 0:
                    #print("Numbers of cpu needed: ", min(cpu_count() - 1, len(argsList)))
                    with Pool(processes=min((cpu_count()-1), len(argsList))) as pool:
                        results = pool.map(process_single_history_entry_first_step, argsList)

                    print("Finished first step: multiprocessing rebuild of history entries and statistical method.")

                    yearsOfHistory = bestParams_json_object['yearsOfHistory']

                    specialColumnCount = SPECIAL_COLUMN_COUNTS.get(name, 0)

                    argsList = [
                        (historyIndex, historyEntry, historyData, name, model_type, dataPath, modelPath,
                            skipLastColumns, yearsOfHistory, ai, previousJsonFilePath, path, boost, bestParams_json_object,
                            specialColumnCount)
                        for historyIndex, historyEntry in enumerate(historyData)
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


def deepLearningMethod(listOfDecodedPredictions, newPredictionRaw, unique_labels, modelDisplayName="LSTM Base Model", gameName=""):

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
        if "keno" in gameName:
            try:
                number_scores = helpers.score_numbers_from_prediction(newPredictionRaw, unique_labels)
                for subset_size in range(5, 11):
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
UNIFIED_DL_MODELS = [
    (tcn, "tcn", "TCN Base Model", "tcn_model"),
    (unifiedLstmTcn, "unifiedLstmTcn", "UnifiedLstmTcn Model", "unified_lstm_tcn_model"),
    (unifiedLstmGruTcn, "unifiedLstmGruTcn", "UnifiedLstmGruTcn Model", "unified_lstm_gru_tcn_model"),
]


def runUnifiedDeepLearningModels(listOfDecodedPredictions, path, name, dataPath, skipLastColumns,
                                  bestParams_json_object, skipRows=0, years_back=None, specialColumnCount=0):
    """
    Runs UnifiedLstmTcn Model and UnifiedLstmGruTcn Model (see
    src/UnifiedLstmTcn.py / src/UnifiedLstmGruTcn.py) alongside the existing
    LSTM Base Model row. Uses bestParams_json_object.get(..., default) - not
    bracket indexing - since these are new keys that won't exist in any
    bestParams_<game>.json written before this feature landed (the project
    already hit a real KeyError crash once from assuming a new key always
    exists).
    """
    for model, prefix, displayName, modelTypeFolder in UNIFIED_DL_MODELS:
        try:
            modelPath = os.path.join(path, "data", "models", modelTypeFolder)
            model.setDataPath(dataPath)
            model.setModelPath(modelPath)
            model.setLoadModelWeights(True)
            model.setBatchSize(bestParams_json_object.get(f"{prefix}_batchSize", 16))
            model.setEpochs(bestParams_json_object.get(f"{prefix}_epochs", 1000))
            if hasattr(model, "setLstmUnits"):
                model.setLstmUnits(bestParams_json_object.get(f"{prefix}_lstmUnits", 64))
            model.setTcnUnits(bestParams_json_object.get(f"{prefix}_tcnUnits", 64))
            model.setNumTcnLayers(bestParams_json_object.get(f"{prefix}_numTcnLayers", 2))
            if hasattr(model, "setGruUnits"):
                model.setGruUnits(bestParams_json_object.get(f"{prefix}_gruUnits", 64))
            model.setDropout(bestParams_json_object.get(f"{prefix}_dropout", 0.3))
            model.setL2Regularization(bestParams_json_object.get(f"{prefix}_l2Regularization", 0.0005))
            model.setLearningRate(bestParams_json_object.get(f"{prefix}_learningRate", 0.001))
            model.setEarlyStopPatience(bestParams_json_object.get(f"{prefix}_earlyStopPatience", 20))
            model.setReduceLearningRatePatience(bestParams_json_object.get(f"{prefix}_reduceLearningRatePatience", 5))
            model.setReducedLearningRateFactor(bestParams_json_object.get(f"{prefix}_reduceLearningRateFactor", 0.5))
            model.setWindowSize(bestParams_json_object.get(f"{prefix}_windowSize", 20))
            model.setPredictionWindowSize(model.window_size)
            model.setLabelSmoothing(bestParams_json_object.get(f"{prefix}_labelSmoothing", 0.05))
            model.setNumHeads(bestParams_json_object.get(f"{prefix}_numHeads", 4))
            model.setKeyDim(bestParams_json_object.get(f"{prefix}_keyDim", 32))

            latest_raw_predictions, unique_labels = model.run(
                name, skipLastColumns, skipRows=skipRows, years_back=years_back, specialColumnCount=specialColumnCount)

            listOfDecodedPredictions = deepLearningMethod(
                listOfDecodedPredictions, latest_raw_predictions.tolist(), unique_labels, modelDisplayName=displayName, gameName=name)
        except Exception as e:
            print(f"Failed to perform {displayName} prediction: ", e)

    return listOfDecodedPredictions


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

        if os.path.exists(metaLearnerPath) or os.path.exists(metaLearnerV2Path):
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
                xgboostPredictor.setDataPath(dataPath)
                xgboostPredictor.setEstimators(bestParams_json_object.get("xgBoostEstimators", 200))
                xgboostPredictor.setLearningRate(bestParams_json_object.get("xgBoostLearningRate", 0.1))
                xgboostPredictor.setMaxDepth(bestParams_json_object.get("xgBoostMaxdepth", 3))
                xgboostPredictor.setPreviousDraws(bestParams_json_object.get("xgBoostPreviousDraws", 11))
                xgboostPredictor.setTopK(bestParams_json_object.get("xgBoostTopK", 16))
                xgboostPredictor.setSubsample(bestParams_json_object.get("xgBoostSubsample", 1.0))
                xgboostPredictor.setColsampleByTree(bestParams_json_object.get("xgBoostColsampleByTree", 1.0))
                xgboostPredictor.setMinChildWeight(bestParams_json_object.get("xgBoostMinChildWeight", 1.0))
                xgboostPredictor.setRegLambda(bestParams_json_object.get("xgBoostRegLambda", 1.0))
                xgboostPredictor.setSortedPrediction(sortedPrediction)
                xgboostPredictor.setNumThreads(max(1, cpu_count() - 1))
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
            except Exception as e:
                print("Failed to perform Meta-Learner prediction: ", e)

    return listOfDecodedPredictions

def boostingMethod(listOfDecodedPredictions, dataPath, path, name, skipRows=0, skipLastColumns=0):
    """
    XGBoost Model as its own tracked prediction row, driven exactly like every
    statistical model above: the same Helpers.run_model_with_special_column
    call (so Euromillions' star columns / EuroDreams' dream number /
    VikingLotto's super viking are modeled independently, and Lotto's unplayed
    bonus column is dropped), the same getKenoSubsetSizes subset choice, and
    the same {size: subset} dict handling - previously it ignored
    skipLastColumns/special columns entirely and returned subsets as a bare
    list.
    """
    try:
        bestParams_json_object = {
            "use_5":True,
            "use_6":True,
            "use_7":True,
            "use_8":True,
            "use_9":True,
            "use_10":True,
            "useBoost": True,
            "xgBoostEstimators": 500,
            "xgBoostLearningRate": 0.7014495252508934,
            "xgBoostMaxdepth": 3,
            "xgBoostPreviousDraws": 81,
            "xgBoostTopK": 31,
            "xgBoostForceNested": True,
            "xgBoostSubsample": 1.0,
            "xgBoostColsampleByTree": 1.0,
            "xgBoostMinChildWeight": 1.0,
            "xgBoostRegLambda": 1.0,
            "xgBoostSubsetMode": "softmax",
            "xgBoostSubsetTemperature": 0.5
        }

        try:
            # Load hyperopt parameters if exists
            hyperoptParamsJsonFile = os.path.join(path, f"bestParams_{name}.json")
            if hyperoptParamsJsonFile and os.path.exists(hyperoptParamsJsonFile):
                with open(hyperoptParamsJsonFile, 'r') as openfile:
                    # Merged over the in-code defaults (not replacing them) for
                    # the same reason statisticalMethod does it: a
                    # bestParams_<game>.json written before a newer key exists
                    # would otherwise KeyError on every lookup of that key.
                    bestParams_json_object.update(json.load(openfile))
        except Exception as e:
            print("Failed to parse parameter file in boost method: ", e)

        subsets = getKenoSubsetSizes(name, bestParams_json_object)
        specialColumnCount = next((count for game, count in SPECIAL_COLUMN_COUNTS.items() if game in name), 0)

        if bestParams_json_object["useBoost"]:
            #print("Performing XGBoost Prediction")
            xgboostPredictor.setDataPath(dataPath)
            xgboostPredictor.setModelPath(modelPath=os.path.join(path, "data", "models", f"xgboost_{name}_models"))
            xgboostPredictor.setEstimators(bestParams_json_object["xgBoostEstimators"])
            xgboostPredictor.setLearningRate(bestParams_json_object["xgBoostLearningRate"])
            xgboostPredictor.setMaxDepth(bestParams_json_object["xgBoostMaxdepth"])
            xgboostPredictor.setPreviousDraws(bestParams_json_object["xgBoostPreviousDraws"])
            xgboostPredictor.setTopK(bestParams_json_object["xgBoostTopK"])
            xgboostPredictor.setForceNested(bestParams_json_object["xgBoostForceNested"])
            xgboostPredictor.setSubsample(bestParams_json_object["xgBoostSubsample"])
            xgboostPredictor.setColsampleByTree(bestParams_json_object["xgBoostColsampleByTree"])
            xgboostPredictor.setMinChildWeight(bestParams_json_object["xgBoostMinChildWeight"])
            xgboostPredictor.setRegLambda(bestParams_json_object["xgBoostRegLambda"])
            xgboostPredictor.setSubsetSelectionMode(bestParams_json_object["xgBoostSubsetMode"])
            xgboostPredictor.setSubsetTemperature(bestParams_json_object["xgBoostSubsetTemperature"])
            # Pick3 is positional - digits stay in drawn order, duplicates
            # included. Every other game gets a sorted ticket of distinct
            # numbers. Set explicitly each run: this is a module-level
            # singleton reused across games/history days.
            xgboostPredictor.setSortedPrediction(not ("pick3" in name))
            # This step runs single-process (see process_single_history_entry_second_step),
            # unlike the Backtester's day-parallel Pool, so XGBoost can use the
            # machine's cores itself here.
            xgboostPredictor.setNumThreads(max(1, cpu_count() - 1))
            xgboostPredictor.setSaveModels(True)
            # No clear() here (unlike the statistical models, whose clear()
            # drops accumulated counts): src/XGBoost.py keys its fit cache on
            # the data slice *and* every training hyperparameter, so a stale
            # fit can never be reused. Keeping it means the main-numbers fit
            # the meta-learner block just did for this same game/day is reused
            # instead of retrained - one full XGBoost training saved per
            # prediction.

            xgboostPrediction = {
                "name": "XGBoost Model",
                "predictions": []
            }

            xgboostSequence, xgboostSubsets = helpers.run_model_with_special_column(
                xgboostPredictor, generateSubsets=subsets, skipRows=skipRows,
                skipLastColumns=skipLastColumns, specialColumnCount=specialColumnCount)

            xgboostPrediction["predictions"].append(xgboostSequence)
            for key in xgboostSubsets:
                xgboostPrediction["predictions"].append(xgboostSubsets[key])

            listOfDecodedPredictions.append(xgboostPrediction)
    except Exception as e:
        print("Failed to perform XGBoost: ", e)

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
                    predict(dataset_name, model_type, dataPath, modelPath, skipLastColumns=skip_last_columns, daysToRebuild=daysToRebuild, ai=ai, boost=boost)
                else:
                    pass

            except Exception as e:
                print(f"Failed to predict {dataset_name.capitalize()}: {e}")

        
        print("Finished with predictions")

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
    
    

    
