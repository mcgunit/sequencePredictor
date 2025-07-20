import os, argparse, json, sys, time
import numpy as np
import subprocess

from art import text2art
from datetime import datetime
from multiprocessing import Pool, cpu_count

from src.TCN import TCNModel
from src.LSTM import LSTMModel
from src.LSTM_ARIMA_Model import LSTM_ARIMA_Model
from src.RefinemePrediction import RefinePrediction
from src.TopPrediction import TopPrediction
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
lstmArima = LSTM_ARIMA_Model()
refinePrediction = RefinePrediction()
topPredictor = TopPrediction()
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


def process_single_history_entry_first_step(args):
    """
    First step to prepare the database and perform the statistical method.
    In this step we can process multible files.
    """
    
    (historyIndex, historyEntry, historyData, name, dataPath, previousJsonFilePath, path) = args

    historyDate, historyResult = historyEntry
    jsonFileName = f"{historyDate.year}-{historyDate.month}-{historyDate.day}.json"
    jsonFilePath = os.path.join(path, "data", "database", name, jsonFileName)

    current_json_object = {
        "currentPredictionRaw": [],
        "currentPrediction": [],
        "realResult": historyResult,
        "newPrediction": [],
        "newPredictionRaw": [],
        "matchingNumbers": {},
        "labels": [],
        "numberFrequency": []
    }

    try:
        current_json_object["numberFrequency"] = helpers.count_number_frequencies(dataPath)
    except Exception as e:
        print("Failed to calculate the number frequencies: ", e)
    
    try:
        # Check the previous prediction with the real result
        if previousJsonFilePath and os.path.exists(previousJsonFilePath):
            with open(previousJsonFilePath, 'r') as openfile:
                previous_json_object = json.load(openfile)
            current_json_object["currentPredictionRaw"] = previous_json_object["newPredictionRaw"]
            current_json_object["currentPrediction"] = previous_json_object["newPrediction"]

        best_matching_prediction = helpers.find_best_matching_prediction(
            current_json_object["realResult"], current_json_object["currentPrediction"])
        current_json_object["matchingNumbers"] = best_matching_prediction

        

        with open(jsonFilePath, "w+") as outfile:
            json.dump(current_json_object, outfile)
    except Exception as e:
        print("Failed to check previous json: ", e)

    try: 
        listOfDecodedPredictions = []

        listOfDecodedPredictions = statisticalMethod(
            listOfDecodedPredictions, dataPath, path, name, skipRows=len(historyData)-historyIndex)

        current_json_object["newPrediction"] = listOfDecodedPredictions
    except Exception as e:
        print("Failed to perform statistical method: ", e)

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)

    return jsonFilePath


def process_single_history_entry_second_step(args):
    """
    Second step to perform methods where we can not process multible files at the same time
    """
    
    (historyIndex, historyEntry, historyData, name, model_type, dataPath, modelPath,
     skipLastColumns, years_back, ai, previousJsonFilePath, path, boost) = args

    modelToUse = tcn if "lstm_model" not in model_type else lstm
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
        modelToUse.setDataPath(dataPath)
        modelToUse.setModelPath(modelPath)
        modelToUse.setBatchSize(16)
        modelToUse.setEpochs(1000)
        latest_raw_predictions, unique_labels = modelToUse.run(
            name, skipLastColumns, skipRows=len(historyData)-historyIndex, years_back=years_back)
        predictedSequence = latest_raw_predictions.tolist()
        unique_labels = unique_labels.tolist()
        current_json_object["newPredictionRaw"] = predictedSequence
        listOfDecodedPredictions = deepLearningMethod(listOfDecodedPredictions, predictedSequence, unique_labels, 2)
    else:
        _, _, _, _, _, _, _, unique_labels = helpers.load_data(
            dataPath, skipLastColumns, years_back=years_back)
        unique_labels = unique_labels.tolist()

    if boost:
       listOfDecodedPredictions = boostingMethod(listOfDecodedPredictions, dataPath, path, name, skipRows=(len(historyData)-historyIndex))

    
    current_json_object["newPrediction"] = listOfDecodedPredictions
    current_json_object["labels"] = unique_labels

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)



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
        "yearsOfHistory": 2,   
    }

    try:
        # Load hyperopt parameters if exists
        hyperoptParamsJsonFile = os.path.join(path, f"bestParams_{name}.json")
        if hyperoptParamsJsonFile and os.path.exists(hyperoptParamsJsonFile):
            with open(hyperoptParamsJsonFile, 'r') as openfile:
                bestParams_json_object = json.load(openfile)
    except Exception as e:
        print("Failed to parse parameter file: ", e)

    modelToUse = tcn
    if "lstm_model" in model_type:
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
                "realResult": latestResult,
                "newPrediction": [],      # Decoded prediction with help of labels
                "newPredictionRaw": [],   # Raw prediction that contains the statistical data
                "matchingNumbers": {},
                "labels": [],             # Needed for decoding the raw predictions
                "numberFrequency": helpers.count_number_frequencies(dataPath)
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

                    # Check on prediction with nth highest probability
                    print("find matching numbers")
                    best_matching_prediction = helpers.find_best_matching_prediction(current_json_object["realResult"], current_json_object["currentPrediction"])

                    current_json_object["matchingNumbers"] = best_matching_prediction

                    print("matching_numbers: ", current_json_object["matchingNumbers"]["matching_numbers"])

                    listOfDecodedPredictions = []
                    unique_labels = []

                    if ai:
                        # Train and do a new prediction
                        modelToUse.setModelPath(modelPath)
                        modelToUse.setBatchSize(16)
                        modelToUse.setEpochs(1000)
                        latest_raw_predictions, unique_labels = modelToUse.run(name, skipLastColumns, years_back=bestParams_json_object['yearsOfHistory'])
                        
                        predictedSequence = latest_raw_predictions.tolist()

                
                        # Save the current prediction as newPrediction
                        current_json_object["newPredictionRaw"] = predictedSequence
                        current_json_object["labels"] = unique_labels.tolist()

            
                        listOfDecodedPredictions = deepLearningMethod(listOfDecodedPredictions, current_json_object["newPredictionRaw"], current_json_object["labels"], 2, current_json_object["realResult"], unique_labels, jsonFilePath, name)
                    else:
                        _, _, _, _, _, _, _, unique_labels = helpers.load_data(dataPath, skipLastColumns, years_back=bestParams_json_object['yearsOfHistory'])
                        unique_labels = unique_labels.tolist()


                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile)
                    
                    listOfDecodedPredictions = statisticalMethod(listOfDecodedPredictions, dataPath, path, name)
                    
                    if boost:
                        listOfDecodedPredictions = boostingMethod(listOfDecodedPredictions, dataPath, path, name)

                    current_json_object["newPrediction"] = listOfDecodedPredictions

                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile)

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

                argsList = [
                    (historyIndex, historyEntry, historyData, name, dataPath,
                    previousJsonFilePath, path)
                    for historyIndex, historyEntry in enumerate(historyData)
                ]

                #print("Argslist: ", len(argsList))

                if len(argsList) > 0:
                    #print("Numbers of cpu needed: ", min(cpu_count() - 1, len(argsList)))
                    with Pool(processes=min((cpu_count()-3), len(argsList))) as pool:
                        results = pool.map(process_single_history_entry_first_step, argsList)

                    print("Finished first step: multiprocessing rebuild of history entries and statistical method.")

                    argsList = [
                        (historyIndex, historyEntry, historyData, name, model_type, dataPath, modelPath,
                            skipLastColumns, bestParams_json_object['yearsOfHistory'], ai, previousJsonFilePath, path, boost)
                        for historyIndex, historyEntry in enumerate(historyData)
                    ]

                    with Pool(processes=1) as pool:
                        results = pool.map(process_single_history_entry_second_step, argsList)

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


def deepLearningMethod(listOfDecodedPredictions, newPredictionRaw, labels, nOfPredictions, historyResult, unique_labels, jsonFilePath, name):
    
    try:
        nthPredictions = {
            "name": "LSTM Base Model",
            "predictions": []
        }
        # Decode prediction with nth highest probability
        for i in range(nOfPredictions):
            prediction_nth_indices = helpers.decode_predictions(newPredictionRaw, labels, i)
            nthPredictions["predictions"].append(prediction_nth_indices)
        
        listOfDecodedPredictions.append(nthPredictions)

        return listOfDecodedPredictions
    except Exception as e:
        print("Failed to perform nth prediction: ", e)


    jsonDirPath = os.path.join(path, "data", "database", name)
    num_classes = len(unique_labels)
    numbersLength = len(historyResult)


    try:
        # Refine predictions
        #print("Refine predictions")
        refinePrediction.trainRefinePredictionsModel(name, jsonDirPath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)
        refined_prediction_raw = refinePrediction.refinePrediction(name=name, pathToLatestPredictionFile=jsonFilePath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)

        #print("refined_prediction_raw: ", refined_prediction_raw)
        refinedPredictions = {
            "name": "LSTM Refined Model",
            "predictions": []
        }

        for i in range(2):
            prediction_highest_indices = helpers.decode_predictions(refined_prediction_raw[0], unique_labels, nHighestProb=i)
            #print("Refined Prediction with ", i+1 ,"highest probs: ", prediction_highest_indices)
            refinedPredictions["predictions"].append(prediction_highest_indices)

        listOfDecodedPredictions.append(refinedPredictions)
    except Exception as e:
        print("Was not able to run refine prediction model: ", e)

    try:
        # Top prediction
        #print("Performing a Top Prediction")
        topPredictor.trainTopPredictionsModel(name, jsonDirPath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)
        top_prediction_raw = topPredictor.topPrediction(name=name, pathToLatestPredictionFile=jsonFilePath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)
        topPrediction = helpers.getTopPredictions(top_prediction_raw, unique_labels, num_top=numbersLength)

        topPrediction = {
            "name": "LSTM Top Predictor",
            "predictions": []
        }

        # Print Top prediction
        for i, prediction in enumerate(topPrediction):
            topHighestProbPrediction = [int(num) for num in prediction]
            #print(f"Top Prediction {i+1}: {sorted(topHighestProbPrediction)}")
            topPrediction["predictions"].append(topHighestProbPrediction)
        
        listOfDecodedPredictions.append(topPrediction)
    except Exception as e:
        print("Was not able to run top prediction model: ", e)

    try:
        # Arima prediction
        #print("Performing ARIMA Prediction")
        lstmArima.setModelPath(os.path.join(path, "data", "models", "lstm_arima_model"))
        lstmArima.setDataPath(dataPath)
        lstmArima.setBatchSize(8)
        lstmArima.setEpochs(1000)

        arimaPrediction = {
            "name": "ARIMA Model",
            "predictions": []
        }

        predicted_arima_sequence = lstmArima.run(name)
        arimaPrediction["predictions"].append(predicted_arima_sequence)
        listOfDecodedPredictions.append(arimaPrediction)

    except Exception as e:
        print("Failed to perform ARIMA: ", e)


def statisticalMethod(listOfDecodedPredictions, dataPath, path, name, skipRows=0):

    bestParams_json_object = {
        "use_5":True,
        "use_6":True,
        "use_7":True,
        "use_8":True,
        "use_9":True,
        "use_10":True,
        "yearsOfHistory": 2,
        "useMarkov":False,
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
        "hybridStatisticalModelNumberOfSimulations": 900
    }

    try:
        # Load hyperopt parameters if exists
        hyperoptParamsJsonFile = os.path.join(path, f"bestParams_{name}.json")
        if hyperoptParamsJsonFile and os.path.exists(hyperoptParamsJsonFile):
            with open(hyperoptParamsJsonFile, 'r') as openfile:
                bestParams_json_object = json.load(openfile)
    except Exception as e:
        print("Failed to parse parameter file: ", e)

    subsets = []
    if "keno" in name:
        if bestParams_json_object["use_5"]:
            subsets.append(5)
        if bestParams_json_object["use_6"]:
            subsets.append(6)
        if bestParams_json_object["use_7"]:
            subsets.append(7)
        if bestParams_json_object["use_8"]:
            subsets.append(8)
        if bestParams_json_object["use_9"]:
            subsets.append(9)
        if bestParams_json_object["use_10"]:
            subsets.append(10)
        

    
    
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
            markov.clear()

            markovPrediction = {
                "name": "Markov Model",
                "predictions": []
            }

            markovSequence, markovSubsets = markov.run(generateSubsets=subsets, skipRows=skipRows)
            
            markovPrediction["predictions"].append(markovSequence)
            for key in markovSubsets:
                markovPrediction["predictions"].append(markovSubsets[key])

            listOfDecodedPredictions.append(markovPrediction)
        except Exception as e:
            print("Failed to perform Markov: ", e)

    if bestParams_json_object["useMarkovBayesian"]:
        try:
            # Markov Bayesian
            #print("Performing Markov Bayesian Prediction")
            markovBayesian.setDataPath(dataPath)
            markovBayesian.setSoftMAxTemperature(bestParams_json_object["markovBayesianSoftMaxTemperature"])
            markovBayesian.setAlpha(bestParams_json_object["markovBayesianAlpha"] )
            markovBayesian.setMinOccurrences(bestParams_json_object["markovBayesianMinOccurences"])
            markovBayesian.clear()

            markovBayesianPrediction = {
                "name": "MarkovBayesian Model",
                "predictions": []
            }

            markovBayesianSequence, markovBayesianSubsets = markovBayesian.run(generateSubsets=subsets, skipRows=skipRows)
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
            markovBayesianEnhanced.clear()

            markovBayesianEnhancedPrediction = {
                "name": "MarkovBayesianEnhanched Model",
                "predictions": []
            }

            markovBayesianEnhancedSequence, markovBayesianEnhancedSubsets = markovBayesianEnhanced.run(generateSubsets=subsets, skipRows=skipRows)
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
            poissonMonteCarlo.clear()

            poissonMonteCarloPrediction = {
                "name": "PoissonMonteCarlo Model",
                "predictions": []
            }

            poissonMonteCarloSequence, poissonMonteCarloSubsets = poissonMonteCarlo.run(generateSubsets=subsets, skipRows=skipRows)

            poissonMonteCarloPrediction["predictions"].append(poissonMonteCarloSequence)
            for key in poissonMonteCarloSubsets:
                poissonMonteCarloPrediction["predictions"].append(poissonMonteCarloSubsets[key])

            listOfDecodedPredictions.append(poissonMonteCarloPrediction)    
        except Exception as e:
            print("Failed to perform Poisson Distribution with Monte Carlo Analysis: ", e)

    if bestParams_json_object["usePoissonMarkov"]:
        try:
            # Poisson-Markov Distribution
            #print("Performing Poisson-Markov Prediction")
            poissonMarkov.setDataPath(dataPath)
            poissonMarkov.setWeights(poisson_weight=bestParams_json_object["poissonMarkovWeight"], markov_weight=(1-bestParams_json_object["poissonMarkovWeight"]))
            poissonMarkov.setNumberOfSimulations(bestParams_json_object["poissonMarkovNumberOfSimulations"])

            poissonMarkovPrediction = {
                "name": "PoissonMarkov Model",
                "predictions": []
            }

            poissonMarkovSequence, poissonMarkovSubsets = poissonMarkov.run(generateSubsets=subsets, skipRows=skipRows)

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
            laplaceMonteCarlo.clear()

            laplaceMonteCarloPrediction = {
                "name": "LaplaceMonteCarlo Model",
                "predictions": []
            }


            laplaceMonteCarloSequence, laplaceMonteCarloSubsets = laplaceMonteCarlo.run(generateSubsets=subsets, skipRows=skipRows)
            laplaceMonteCarloPrediction["predictions"].append(laplaceMonteCarloSequence)
            for key in laplaceMonteCarloSubsets:
                laplaceMonteCarloPrediction["predictions"].append(laplaceMonteCarloSubsets[key])
            
            listOfDecodedPredictions.append(laplaceMonteCarloPrediction)
        except Exception as e:
            print("Failed to perform Laplace Distribution with Monte Carlo Analysis: ", e)

    if bestParams_json_object["useHybridStatisticalModel"]:
        try:
            # Hybrid Statistical Model
            #print("Performing Hybrid Statistical Model Prediction")
            hybridStatisticalModel.setDataPath(dataPath)
            hybridStatisticalModel.setSoftMaxTemperature(bestParams_json_object["hybridStatisticalModelSoftMaxTemperature"])
            hybridStatisticalModel.setAlpha(bestParams_json_object["hybridStatisticalModelAlpha"])
            hybridStatisticalModel.setMinOccurrences(bestParams_json_object["hybridStatisticalModelMinOcurrences"])
            hybridStatisticalModel.setNumberOfSimulations(bestParams_json_object["hybridStatisticalModelNumberOfSimulations"])
            hybridStatisticalModel.clear()

            hybridStatisticalModelPrediction = {
                "name": "HybridStatisticalModel",
                "predictions": []
            }

            hybridStatisticalModelSequence, hybridStatisticalModelSubsets = hybridStatisticalModel.run(generateSubsets=subsets, skipRows=skipRows)
            hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSequence)
            for key in hybridStatisticalModelSubsets:
                hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSubsets[key])
            
            listOfDecodedPredictions.append(hybridStatisticalModelPrediction)
        except Exception as e:
            print("Failed to perform Hybrid Statistical Model: ", e)


    return listOfDecodedPredictions

def boostingMethod(listOfDecodedPredictions, dataPath, path, name, skipRows=0):
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
            "xgBoostForceNested": True
        }

        try:
            # Load hyperopt parameters if exists
            hyperoptParamsJsonFile = os.path.join(path, f"bestParams_{name}.json")
            if hyperoptParamsJsonFile and os.path.exists(hyperoptParamsJsonFile):
                with open(hyperoptParamsJsonFile, 'r') as openfile:
                    bestParams_json_object = json.load(openfile)
        except Exception as e:
            print("Failed to parse parameter file in boost method: ", e)

        subsets = []
        if "keno" in name:
            if bestParams_json_object["use_5"]:
                subsets.append(5)
            if bestParams_json_object["use_6"]:
                subsets.append(6)
            if bestParams_json_object["use_7"]:
                subsets.append(7)
            if bestParams_json_object["use_8"]:
                subsets.append(8)
            if bestParams_json_object["use_9"]:
                subsets.append(9)
            if bestParams_json_object["use_10"]:
                subsets.append(10)

        if bestParams_json_object["useBoost"]:
            #print("Performing XGBoost Prediction")
            if "pick3" in name:
                xgboostPredictor.setOffsetByOne(False)
            else:
                xgboostPredictor.setOffsetByOne(True)
            xgboostPredictor.setDataPath(dataPath)
            xgboostPredictor.setModelPath(modelPath=os.path.join(path, "data", "models", f"xgboost_{name}_models"))
            xgboostPredictor.setEstimators(bestParams_json_object["xgBoostEstimators"])
            xgboostPredictor.setLearningRate(bestParams_json_object["xgBoostLearningRate"])
            xgboostPredictor.setMaxDepth(bestParams_json_object["xgBoostMaxdepth"])
            xgboostPredictor.setPreviousDraws(bestParams_json_object["xgBoostPreviousDraws"])
            xgboostPredictor.setTopK(bestParams_json_object["xgBoostTopK"])
            xgboostPredictor.setForceNested(bestParams_json_object["xgBoostForceNested"])

            xgboostPrediction = {
                "name": "xgboost",
                "predictions": []
            }

            xgboostSequence, xgboostSubsets = xgboostPredictor.run(generateSubsets=subsets, skipRows=skipRows)
            xgboostPrediction["predictions"].append(xgboostSequence)
            for item in xgboostSubsets:
                xgboostPrediction["predictions"].append(item)
            
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

        parser.add_argument('-r', '--rebuild_history', type=bool, default=False)
        parser.add_argument('-d', '--days', type=int, default=7)
        args = parser.parse_args()

        print_intro()

        current_year = datetime.now().year
        print("Current Year:", current_year)

        daysToRebuild = int(args.days)
        rebuildHistory = bool(args.rebuild_history)


        path = os.getcwd()

        # Here we can force disable ai and boost methods. If enabled here we let hyperopt decide
        datasets = [
            # (dataset_name, model_type, skip_last_columns, ai, xgboost)
            ("euromillions", "tcn_model", 0, True, True),
            ("lotto", "lstm_model", 0, True, True),
            ("eurodreams", "lstm_model", 0, True, True),
            #("jokerplus", "lstm_model", 1, False, True),
            ("keno", "lstm_model", 0, False, True),    # For Keno subsets are need to ceated for ai
            ("pick3", "lstm_model", 0, True, True),
            ("vikinglotto", "lstm_model", 0, True, True),
        ]

        for dataset_name, model_type, skip_last_columns, ai, boost in datasets:
            try:
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
            

                #command.run("wget -P {folder} https://prdlnboppreportsst.blob.core.windows.net/legal-reports/{file}".format(**kwargs_wget), verbose=False)

                # Predict with hyperopt params
                predict(dataset_name, model_type, dataPath, modelPath, skipLastColumns=skip_last_columns, daysToRebuild=daysToRebuild, ai=ai, boost=boost)

                # Predict for current year
                #predict(f"{dataset_name}_currentYear", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=1, daysToRebuild=daysToRebuild, ai=ai, boost=boost)

                # Predict for current year + last year
                #predict(f"{dataset_name}_twoYears", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=2, daysToRebuild=daysToRebuild, ai=ai, boost=boost)

                # Predict for current year + last two years
                #predict(f"{dataset_name}_threeYears", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=3, daysToRebuild=daysToRebuild, ai=ai, boost=boost)

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
            helpers.git_push()
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()  # Ensure the lock is removed even if an error occurs
    
    

    
