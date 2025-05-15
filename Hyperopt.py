import os, argparse, json
import numpy as np
import subprocess
import optuna

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
        skipLastColumns, years_back, ai, boost, previousJsonFilePath, path, modelParams) = args

    modelToUse = tcn if "lstm_model" not in model_type else lstm
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

    if ai:
        modelToUse.setDataPath(dataPath)
        modelToUse.setModelPath(modelPath)
        modelToUse.setBatchSize(16)
        modelToUse.setEpochs(1000)
        latest_raw_predictions, unique_labels = modelToUse.run(
            name, skipLastColumns, skipRows=len(historyData)-historyIndex, years_back=years_back)
        predictedSequence = latest_raw_predictions.tolist()
        unique_labels = unique_labels.tolist()
        current_json_object["newPredictionRaw"] = predictedSequence
        listOfDecodedPredictions = deepLearningMethod(listOfDecodedPredictions, predictedSequence, unique_labels, 2, modelParams)
    else:
        _, _, _, _, _, _, _, unique_labels = helpers.load_data(
            dataPath, skipLastColumns, years_back=years_back)
        unique_labels = unique_labels.tolist()

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)

    listOfDecodedPredictions = statisticalMethod(
        listOfDecodedPredictions, dataPath, path, name, historyResult,
        unique_labels, jsonFilePath, ai, modelParams, skipRows=len(historyData)-historyIndex)
    
    if boost:
        listOfDecodedPredictions = boostingMethod(listOfDecodedPredictions, dataPath, path, name, modelParams, skipRows=len(historyData)-historyIndex)

    current_json_object["newPrediction"] = listOfDecodedPredictions
    current_json_object["labels"] = unique_labels

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)

    return jsonFilePath

def calculate_profit(name, path):
    json_dir = os.path.join(path, "data", "hyperOptCache", name)
    if not os.path.exists(json_dir):
        print(f"Directory does not exist: {json_dir}")
        return

    payoutTableKeno = {
        10: { 0: 3, 5: 1, 6: 4, 7: 10, 8: 200, 9: 2000, 10: 250000 },
        9: { 0: 3, 5: 2, 6: 5, 7: 50, 8: 500, 9: 50000 },
        8: { 0: 3, 5: 4, 6: 10, 7: 100, 8: 10000 },
        7: { 0: 3, 5: 3, 6: 30, 7: 3000 },
        6: { 3: 1, 4: 4, 5: 20, 6: 200 },
        5: { 3: 2, 4: 5, 5: 150 },
        4: { 2: 1, 3: 2, 4: 30 },
        3: { 2: 1, 3: 16 },
        2: { 2: 6.50 },
        "lost": -1
    }

    total_profit = 0

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, "r") as file:
                data = json.load(file)

                real_result = set(data.get("realResult", []))
                model_predictions = data.get("currentPrediction", [])

                for model in model_predictions:
                    for prediction in model["predictions"]:
                        played = len(prediction)
                        if played < 4 or played > 10:
                            continue

                        matches = len(set(prediction) & real_result)
                        profit = payoutTableKeno.get(played, {}).get(matches, payoutTableKeno["lost"])
                        total_profit += profit
    
    return total_profit


def clearFolder(folderPath):
    try:
        for filename in os.listdir(folderPath):
            file_path = os.path.join(folderPath, filename)
        
            if os.path.isfile(file_path):
                os.remove(file_path)  
                #print(f"Deleted file: {filename}")
    except Exception as e:
        pass

def predict(name, model_type ,dataPath, modelPath, file, skipLastColumns=0, maxRows=0, years_back=None, daysToRebuild=31, ai=False, boost=False, modelParams={}):
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

    modelToUse = tcn
    if "lstm_model" in model_type:
        modelToUse = lstm
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
            #print("History data: ", historyData)

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
            historyData = historyData[dateOffset:]  # Keep elements afer dateOffset because newer elements comes after the dateOffset index                
            #print("History to rebuild: ", historyData)

            argsList = [
                (historyIndex, historyEntry, historyData, name, model_type, dataPath,
                modelPath, skipLastColumns, years_back, ai, boost, previousJsonFilePath, path, modelParams)
                for historyIndex, historyEntry in enumerate(historyData)
            ]

            numberOfProcesses = min((cpu_count()-1), len(argsList))
            if boost or ai:
                numberOfProcesses = 1

            with Pool(processes=numberOfProcesses) as pool:
                results = pool.map(process_single_history_entry, argsList)

            print("Finished multiprocessing rebuild of history entries.")

            # Find the matching numbers
            update_matching_numbers(name=name, path=path)

            # Calculate Profit
            profit =  calculate_profit(name=name, path=path)

            return profit
        else:
            print("Prediction already made")
    else:
        print("Did not found entries")


def deepLearningMethod(listOfDecodedPredictions, newPredictionRaw, labels, nOfPredictions, modelParams):
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


def statisticalMethod(listOfDecodedPredictions, dataPath, path, name, historyResult, unique_labels, jsonFilePath, ai, modelParams, skipRows=0):
    #####################
    # Start refinements #
    #####################
    jsonDirPath = os.path.join(path, "data", "hyperOptCache", name)
    num_classes = len(unique_labels)
    numbersLength = len(historyResult)
    
    if ai:
        try:
            # Refine predictions
            print("Refine predictions")
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
            print("Performing a Top Prediction")
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
            print("Performing ARIMA Prediction")
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

    if modelParams["useMarkov"] == True:
        try:
            # Markov
            # print("Performing Markov Prediction")
            markov.setDataPath(dataPath)
            markov.setSoftMAxTemperature(modelParams["markovSoftMaxTemperature"])
            markov.setMinOccurrences(modelParams["markovMinOccurences"])
            markov.setAlpha(modelParams["markovAlpha"])
            markov.setRecencyWeight(modelParams["markovRecencyWeight"])
            markov.setRecencyMode(modelParams["markovRecencyMode"])
            markov.setPairDecayFactor(modelParams["markovPairDecayFactor"])
            markov.setSmoothingFactor(modelParams["markovSmoothingFactor"])
            markov.setSubsetSelectionMode(modelParams["markovSubsetSelectionMode"])
            markov.setBlendMode(modelParams["markovBlendMode"])
            markov.clear()

            markovPrediction = {
                "name": "Markov Model",
                "predictions": []
            }

            subsets = []
            if "keno" in name:
                subsets = modelParams["kenoSubset"]

            markovSequence, markovSubsets = markov.run(generateSubsets=subsets, skipRows=skipRows)
            
            markovPrediction["predictions"].append(markovSequence)
            for key in markovSubsets:
                markovPrediction["predictions"].append(markovSubsets[key])

            listOfDecodedPredictions.append(markovPrediction)
        except Exception as e:
            print("Failed to perform Markov: ", e)

    
    if modelParams["useMarkovBayesian"] == True:
        try:
            # Markov Bayesian
            #print("Performing Markov Bayesian Prediction")
            markovBayesian.setDataPath(dataPath)
            markovBayesian.setSoftMAxTemperature(modelParams["markovBayesianSoftMaxTemperature"])
            markovBayesian.setAlpha(modelParams["markovBayesianAlpha"])
            markovBayesian.setMinOccurrences(modelParams["markovBayesianMinOccurences"])
            markovBayesian.clear()

            markovBayesianPrediction = {
                "name": "MarkovBayesian Model",
                "predictions": []
            }

            subsets = []
            if "keno" in name:
                subsets = modelParams["kenoSubset"]

            markovBayesianSequence, markovBayesianSubsets = markovBayesian.run(generateSubsets=subsets)
            markovBayesianPrediction["predictions"].append(markovBayesianSequence)
            for key in markovBayesianSubsets:
                markovBayesianPrediction["predictions"].append(markovBayesianSubsets[key])

            listOfDecodedPredictions.append(markovBayesianPrediction)
        except Exception as e:
            print("Failed to perform Markov Bayesian: ", e)

    
    if not "pick3" in name and modelParams["usevMarkovBayesianEnhanced"] == True:
        try:
            # Markov Bayesian Enhanced
            #print("Performing Markov Bayesian Enhanced Prediction")
            markovBayesianEnhanced.setDataPath(dataPath)
            markovBayesianEnhanced.setSoftMAxTemperature(modelParams["markovBayesianEnhancedSoftMaxTemperature"])
            markovBayesianEnhanced.setAlpha(modelParams["markovBayesianEnhancedAlpha"])
            markovBayesianEnhanced.setMinOccurrences(modelParams["markovBayesianEnhancedMinOccurences"])
            markovBayesianEnhanced.clear()

            markovBayesianEnhancedPrediction = {
                "name": "MarkovBayesianEnhanched Model",
                "predictions": []
            }

            subsets = []
            if "keno" in name:
                subsets = modelParams["kenoSubset"]

            markovBayesianEnhancedSequence, markovBayesianEnhancedSubsets = markovBayesianEnhanced.run(generateSubsets=subsets)
            markovBayesianEnhancedPrediction["predictions"].append(markovBayesianEnhancedSequence)
            for key in markovBayesianEnhancedSubsets:
                markovBayesianEnhancedPrediction["predictions"].append(markovBayesianEnhancedSubsets[key])

            listOfDecodedPredictions.append(markovBayesianEnhancedPrediction)
        except Exception as e:
            print("Failed to perform Markov Bayesian Enhanced: ", e)

    if modelParams["usePoissonMonteCarlo"] == True:
        try:
            # Poisson Distribution with Monte Carlo Analysis
            #print("Performing Poisson Monte Carlo Prediction")
            poissonMonteCarlo.setDataPath(dataPath)
            poissonMonteCarlo.setNumOfSimulations(modelParams["poissonMonteCarloNumberOfSimulations"])
            poissonMonteCarlo.setWeightFactor(modelParams["poissonMonteCarloWeightFactor"])
            poissonMonteCarlo.clear()

            poissonMonteCarloPrediction = {
                "name": "PoissonMonteCarlo Model",
                "predictions": []
            }

            subsets = []
            if "keno" in name:
                subsets = modelParams["kenoSubset"]

            poissonMonteCarloSequence, poissonMonteCarloSubsets = poissonMonteCarlo.run(generateSubsets=subsets)

            poissonMonteCarloPrediction["predictions"].append(poissonMonteCarloSequence)
            for key in poissonMonteCarloSubsets:
                poissonMonteCarloPrediction["predictions"].append(poissonMonteCarloSubsets[key])

            listOfDecodedPredictions.append(poissonMonteCarloPrediction)    
        except Exception as e:
            print("Failed to perform Poisson Distribution with Monte Carlo Analysis: ", e)

    if modelParams["usePoissonMarkov"] == True:
        try:
            # Poisson-Markov Distribution
            #print("Performing Poisson-Markov Prediction")
            poissonMarkov.setDataPath(dataPath)
            poissonMarkov.setWeights(poisson_weight=modelParams["poissonMarkovWeight"], markov_weight=(1-modelParams["poissonMarkovWeight"]))
            poissonMarkov.setNumberOfSimulations(modelParams["poissonMarkovNumberOfSimulations"])

            poissonMarkovPrediction = {
                "name": "PoissonMarkov Model",
                "predictions": []
            }

            subsets = []
            if "keno" in name:
                subsets = modelParams["kenoSubset"]

            poissonMarkovSequence, poissonMarkovSubsets = poissonMarkov.run(generateSubsets=subsets)

            poissonMarkovPrediction["predictions"].append(poissonMarkovSequence)
            for key in poissonMarkovSubsets:
                poissonMarkovPrediction["predictions"].append(poissonMarkovSubsets[key])

            listOfDecodedPredictions.append(poissonMarkovPrediction)    
        except Exception as e:
            print("Failed to perform Poisson-Markov Distribution: ", e)

    if modelParams["useLaplaceMonteCarlo"] == True:
        try:
            # Laplace Distribution with Monte Carlo Analysis
            #print("Performing Laplace Monte Carlo Prediction")
            laplaceMonteCarlo.setDataPath(dataPath)
            laplaceMonteCarlo.setNumOfSimulations(modelParams["laplaceMonteCarloNumberOfSimulations"])
            laplaceMonteCarlo.clear()

            laplaceMonteCarloPrediction = {
                "name": "LaplaceMonteCarlo Model",
                "predictions": []
            }

            subsets = []
            if "keno" in name:
                subsets = modelParams["kenoSubset"]

            laplaceMonteCarloSequence, laplaceMonteCarloSubsets = laplaceMonteCarlo.run(generateSubsets=subsets)
            laplaceMonteCarloPrediction["predictions"].append(laplaceMonteCarloSequence)
            for key in laplaceMonteCarloSubsets:
                laplaceMonteCarloPrediction["predictions"].append(laplaceMonteCarloSubsets[key])
            
            listOfDecodedPredictions.append(laplaceMonteCarloPrediction)
        except Exception as e:
            print("Failed to perform Laplace Distribution with Monte Carlo Analysis: ", e)

    if modelParams["useHybridStatisticalModel"] == True:
        try:
            # Hybrid Statistical Model
            #print("Performing Hybrid Statistical Model Prediction")
            hybridStatisticalModel.setDataPath(dataPath)
            hybridStatisticalModel.setSoftMaxTemperature(modelParams["hybridStatisticalModelSoftMaxTemperature"])
            hybridStatisticalModel.setAlpha(modelParams["hybridStatisticalModelAlpha"])
            hybridStatisticalModel.setMinOccurrences(modelParams["hybridStatisticalModelMinOcurrences"])
            hybridStatisticalModel.setNumberOfSimulations(modelParams["hybridStatisticalModelNumberOfSimulations"])
            hybridStatisticalModel.clear()

            hybridStatisticalModelPrediction = {
                "name": "HybridStatisticalModel",
                "predictions": []
            }

            subsets = []
            if "keno" in name:
                subsets = modelParams["kenoSubset"]

            hybridStatisticalModelSequence, hybridStatisticalModelSubsets = hybridStatisticalModel.run(generateSubsets=subsets)
            hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSequence)
            for key in hybridStatisticalModelSubsets:
                hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSubsets[key])
            
            listOfDecodedPredictions.append(hybridStatisticalModelPrediction)
        except Exception as e:
            print("Failed to perform Hybrid Statistical Model: ", e)

    return listOfDecodedPredictions

def boostingMethod(listOfDecodedPredictions, dataPath, path, name, modelParams, skipRows=0):
    try:
        # Hybrid Statistical Model
        #print("Performing XGBoost Prediction")
        xgboostPredictor.setDataPath(dataPath)
        xgboostPredictor.setModelPath(modelPath=os.path.join(path, "data", "models", f"xgboost_{name}_models"))

        xgboostPrediction = {
            "name": "xgboost",
            "predictions": []
        }

        subsets = []
        if "keno" in name:
            subsets = modelParams["kenoSubset"]

        xgboostSequence, xgboostSubsets = xgboostPredictor.run(generateSubsets=subsets, skipRows=skipRows)
        xgboostPrediction["predictions"].append(xgboostSequence)
        for item in xgboostSubsets:
            xgboostPrediction["predictions"].append(item)
        
        listOfDecodedPredictions.append(xgboostPrediction)
    except Exception as e:
        print("Failed to perform XGBoost: ", e)

    return listOfDecodedPredictions


if __name__ == "__main__":

    try:
        cmd = ['pgrep', '-f', 'python.*Predictor.py']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        my_pid, err = process.communicate()

        if err:
            print(f"Error running pgrep: {err.decode('utf-8')}")

        pid_list = my_pid.decode('utf-8').strip().splitlines()
        num_pids = len(pid_list)

        if num_pids >= 2:
            print("Multiple instances running. Exiting.")
            exit()
        else:
            print("No instances running. Continuing.")

    except Exception as e:
        print(f"Failed to check if running: {e}")

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
    parser.add_argument('-d', '--days', type=int, default=31)
    args = parser.parse_args()

    print_intro()

    current_year = datetime.now().year
    print("Current Year:", current_year)

    daysToRebuild = int(args.days)
    rebuildHistory = bool(args.rebuild_history)


    path = os.getcwd()

    datasets = [
        # (dataset_name, model_type, skip_last_columns, ai)
        #("euromillions", "tcn_model", 0, False),
        #("lotto", "lstm_model", 0, False),
        #("eurodreams", "lstm_model", 0, False),
        #("jokerplus", "lstm_model", 1, False),
        ("keno", "lstm_model", 0, False),
        #("pick3", "lstm_model", 0, False),
        #("vikinglotto", "lstm_model", 0, False),
    ]

    for dataset_name, model_type, skip_last_columns, ai in datasets:
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
                os.remove(os.path.join(dataPath, file))
            command.run("wget -P {folder} https://prdlnboppreportsst.blob.core.windows.net/legal-reports/{file}".format(**kwargs_wget), verbose=False)

            # Predict for complete data
            #predict(dataset_name, model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, daysToRebuild=daysToRebuild, ai=ai)

            # Predict for current year
            #predict(f"{dataset_name}_currentYear", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=1, daysToRebuild=daysToRebuild, ai=ai)

            # Predict for current year + last year
            def objective(trial):
                numOfRepeats = 2 # To average out the rusults before continueing to the next result
                totalProfit = 0
                results = [] # Intermediate results

                all_values = [5, 6, 7, 8, 9, 10]
                MIN_LEN = 1
                MAX_LEN = 6

                # Binary inclusion mask for each value
                inclusion_mask = [trial.suggest_categorical(f"use_{v}", [True, False]) for v in all_values]
                
                # Build the subset from the mask
                subset = [v for v, include in zip(all_values, inclusion_mask) if include]

                # Enforce length constraints
                if not (MIN_LEN <= len(subset) <= MAX_LEN):
                    return float("-inf")  # Or float("inf") if minimizing

                modelParams = {
                    'useMarkov': trial.suggest_categorical("useMarkov", [True, False]),
                    'useMarkovBayesian': trial.suggest_categorical("useMarkovBayesian", [True, False]),
                    'usevMarkovBayesianEnhanced': trial.suggest_categorical("usevMarkovBayesianEnhanced", [True, False]),
                    'usePoissonMonteCarlo': trial.suggest_categorical("usePoissonMonteCarlo", [True, False]),
                    'usePoissonMarkov': trial.suggest_categorical("usePoissonMarkov", [True, False]),
                    'useLaplaceMonteCarlo': trial.suggest_categorical("useLaplaceMonteCarlo", [True, False]),
                    'useHybridStatisticalModel': trial.suggest_categorical("useHybridStatisticalModel", [True, False]),
                    'useBoost': trial.suggest_categorical("useBoost", [True, False]),
                    'kenoSubset': subset,
                    'markovSoftMaxTemperature': trial.suggest_float('markovSoftMaxTemperature', 0.1, 1.0),
                    'markovMinOccurences': trial.suggest_int('markovMinOccurences', 1, 20),
                    'markovAlpha': trial.suggest_float('markovAlpha', 0.1, 1.0),
                    'markovRecencyWeight': trial.suggest_float('markovRecencyWeight', 0.1, 2.0),
                    'markovRecencyMode': trial.suggest_categorical("markovRecencyMode", ["linear", "log", "constant"]),
                    'markovPairDecayFactor': trial.suggest_float('markovPairDecayFactor', 0.1, 2.0),
                    'markovSmoothingFactor': trial.suggest_float('markovSmoothingFactor', 0.01, 1.0),
                    'markovSubsetSelectionMode': trial.suggest_categorical("markovSubsetSelectionMode", ["top", "softmax"]),
                    'markovBlendMode': trial.suggest_categorical("markovBlendMode", ["linear", "harmonic", "log"]),
                    'markovBayesianSoftMaxTemperature': trial.suggest_float('markovBayesianSoftMaxTemperature', 0.1, 1.0),
                    'markovBayesianMinOccurences': trial.suggest_int('markovBayesianMinOccurences', 1, 20),
                    'markovBayesianAlpha': trial.suggest_float('markovBayesianAlpha', 0.1, 1.0),
                    'markovBayesianEnhancedSoftMaxTemperature': trial.suggest_float('markovBayesianEnhancedSoftMaxTemperature', 0.1, 1.0),
                    'markovBayesianEnhancedAlpha': trial.suggest_float('markovBayesianEnhancedAlpha', 0.1, 1.0),
                    'markovBayesianEnhancedMinOccurences': trial.suggest_int('markovBayesianEnhancedMinOccurences', 1, 20),
                    'poissonMonteCarloNumberOfSimulations': trial.suggest_int('poissonMonteCarloNumberOfSimulations', 100, 1000, step=100),
                    'poissonMonteCarloWeightFactor': trial.suggest_float('poissonMonteCarloWeightFactor', 0.1, 1.0),
                    'poissonMarkovWeight': trial.suggest_float('poissonMarkovWeight', 0.1, 1.0),
                    'poissonMarkovNumberOfSimulations': trial.suggest_int('poissonMarkovNumberOfSimulations', 100, 1000, step=100),
                    'laplaceMonteCarloNumberOfSimulations': trial.suggest_int('laplaceMonteCarloNumberOfSimulations', 100, 1000, step=100),
                    'hybridStatisticalModelSoftMaxTemperature': trial.suggest_float('hybridStatisticalModelSoftMaxTemperature', 0.1, 1.0),
                    'hybridStatisticalModelAlpha': trial.suggest_float('hybridStatisticalModelAlpha', 0.1, 1.0),
                    'hybridStatisticalModelMinOcurrences': trial.suggest_int('hybridStatisticalModelMinOcurrences', 1, 20),
                    'hybridStatisticalModelNumberOfSimulations': trial.suggest_int('hybridStatisticalModelNumberOfSimulations', 100, 1000, step=100),
                }
                for _ in range(numOfRepeats):
                    profit = predict(f"{dataset_name}_twoYears", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=2, daysToRebuild=daysToRebuild, ai=ai, boost=modelParams["useBoost"], modelParams=modelParams)
                    #print("Profit: ", profit)
                    results.append(profit)

                totalProfit = sum(results) / len(results)

                return totalProfit

            # Create an Optuna study object
            study = optuna.create_study(
                direction='maximize',
                storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                study_name=f"Sequence-Predictor-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )

            # Run the automatic tuning process
            study.optimize(objective, n_trials=1000)

            # Output the best hyperparameters and score
            print("Best Parameters: ", study.best_params)
            print("Best Score: ", study.best_value)

            # Write best params to json
            jsonBestParamsFilePath = os.path.join(path, "bestParams.json")
            with open(jsonBestParamsFilePath, "w+") as outfile:
                json.dump(study.best_params, outfile)
            
            clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}_twoYears"))

            # Predict for current year + last two years
            #predict(f"{dataset_name}_threeYears", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=3, daysToRebuild=daysToRebuild, ai=ai)

        except Exception as e:
            print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")

    # try:
    #     helpers.generatePredictionTextFile(os.path.join(path, "data", "hyperOptCache"))
    # except Exception as e:
    #     print("Failed to generate txt file:", e)

    # try:
    #     helpers.git_push()
    # except Exception as e:
    #     print("Failed to push latest predictions:", e)
    
    

    
