import os, argparse, json, sys, time
import numpy as np
import subprocess
import optuna

from art import text2art
from datetime import datetime
from multiprocessing import Pool, cpu_count

from src.Markov import Markov
from src.MarkovBayesian import MarkovBayesian
from src.MarkovBayesianEnhanched import MarkovBayesianEnhanced
from src.PoissonMonteCarlo import PoissonMonteCarlo
from src.PoissonMarkov import PoissonMarkov
from src.LaplaceMonteCarlo import LaplaceMonteCarlo
from src.HybridStatisticalModel import HybridStatisticalModel
from src.Command import Command
from src.Helpers import Helpers
from src.DataFetcher import DataFetcher


markov = Markov()
markovBayesian = MarkovBayesian()
markovBayesianEnhanced = MarkovBayesianEnhanced()
poissonMonteCarlo = PoissonMonteCarlo()
laplaceMonteCarlo = LaplaceMonteCarlo()
hybridStatisticalModel = HybridStatisticalModel()
poissonMarkov = PoissonMarkov()
command = Command()
helpers = Helpers()
dataFetcher = DataFetcher()

LOCK_FILE = os.path.join(os.getcwd(), "process.lock")

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
    (historyIndex, historyEntry, historyData, name, dataPath,
        skipLastColumns, years_back, previousJsonFilePath, path, modelParams) = args

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

    _, _, _, _, _, _, _, unique_labels = helpers.load_data(
        dataPath, skipLastColumns, years_back=years_back)
    unique_labels = unique_labels.tolist()

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)

    listOfDecodedPredictions = statisticalMethod(
        listOfDecodedPredictions, dataPath, name, modelParams, skipRows=len(historyData)-historyIndex)
    
    current_json_object["newPrediction"] = listOfDecodedPredictions
    current_json_object["labels"] = unique_labels

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)

    return jsonFilePath


def clearFolder(folderPath):
    try:
        for filename in os.listdir(folderPath):
            file_path = os.path.join(folderPath, filename)
        
            if os.path.isfile(file_path):
                os.remove(file_path)  
                #print(f"Deleted file: {filename}")
    except Exception as e:
        pass

def predict(name, dataPath, skipLastColumns=0, years_back=None, daysToRebuild=31, modelParams={}):
    """
        Predicts the next sequence of numbers for a given dataset or rebuild the prediction for the last n months

        @param name: The name of the dataset
        @param dataPath: The path to the data 
        @param skipLastColumns: The number of columns to skip
        @param years_back: The number of years to go back
        @param daysToRebuild: The number of days to rebuild
    """

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
                (historyIndex, historyEntry, historyData, name, dataPath,
                    skipLastColumns, years_back, previousJsonFilePath, path, modelParams)
                for historyIndex, historyEntry in enumerate(historyData)
            ]

            numberOfProcesses = min((cpu_count()-1), len(argsList))

            with Pool(processes=numberOfProcesses) as pool:
                results = pool.map(process_single_history_entry, argsList)

            print("Finished multiprocessing rebuild of history entries.")

            # Find the matching numbers
            update_matching_numbers(name=name, path=path)

            # Calculate Profit
            profit = helpers.calculate_profit(name=name, path=path)

            return profit
        else:
            print("Prediction already made")
    else:
        print("Did not found entries")


def statisticalMethod(listOfDecodedPredictions, dataPath, name, modelParams, skipRows=0):

    subsets = []
    if "keno" in name:
        subsets = modelParams["kenoSubset"]
    
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

            hybridStatisticalModelSequence, hybridStatisticalModelSubsets = hybridStatisticalModel.run(generateSubsets=subsets)
            hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSequence)
            for key in hybridStatisticalModelSubsets:
                hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSubsets[key])
            
            listOfDecodedPredictions.append(hybridStatisticalModelPrediction)
        except Exception as e:
            print("Failed to perform Hybrid Statistical Model: ", e)

    return listOfDecodedPredictions

if __name__ == "__main__":

    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)

    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)

    try:

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
        parser.add_argument('-d', '--days', type=int, default=61)
        parser.add_argument('-t', '--trials', type=int, default=500)
        args = parser.parse_args()

        print_intro()

        current_year = datetime.now().year
        print("Current Year:", current_year)

        daysToRebuild = int(args.days)
        rebuildHistory = bool(args.rebuild_history)
        n_trials = int(args.trials)


        path = os.getcwd()

        datasets = [
            # (dataset_name, model_type, skip_last_columns)
            ("euromillions", "tcn_model", 0),
            ("lotto", "lstm_model", 0),
            ("eurodreams", "lstm_model", 0),
            #("jokerplus", "lstm_model", 0),
            ("keno", "lstm_model", 0),
            ("pick3", "lstm_model", 0),
            ("vikinglotto", "lstm_model", 0),
        ]

        for dataset_name, model_type, skip_last_columns in datasets:
            try:
                print(f"\n{dataset_name.capitalize()}")
                modelPath = os.path.join(path, "data", "models", model_type)
                dataPath = os.path.join(path, "data", "trainingData", dataset_name)
                file = f"{dataset_name}-gamedata-NL-{current_year}.csv"

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


                defautParams = {
                    'yearsOfHistory': None, # None = all available data
                    'useMarkov': False,
                    'useMarkovBayesian': False,
                    'usevMarkovBayesianEnhanced': False,
                    'usePoissonMonteCarlo': False,
                    'usePoissonMarkov': False,
                    'useLaplaceMonteCarlo': False,
                    'useHybridStatisticalModel': False,
                    'markovSoftMaxTemperature': 1,
                    'markovMinOccurences': 1,
                    'markovAlpha': 1,
                    'markovRecencyWeight': 1,
                    'markovRecencyMode': 1,
                    'markovPairDecayFactor': 1,
                    'markovSmoothingFactor': 1,
                    'markovSubsetSelectionMode': 1,
                    'markovBlendMode': 1,
                    'markovBayesianSoftMaxTemperature': 1,
                    'markovBayesianMinOccurences': 1,
                    'markovBayesianAlpha': 1,
                    'markovBayesianEnhancedSoftMaxTemperature': 1,
                    'markovBayesianEnhancedAlpha': 1,
                    'markovBayesianEnhancedMinOccurences': 1,
                    'poissonMonteCarloNumberOfSimulations': 1,
                    'poissonMonteCarloWeightFactor': 1,
                    'poissonMarkovWeight': 1,
                    'poissonMarkovNumberOfSimulations': 1,
                    'laplaceMonteCarloNumberOfSimulations': 1,
                    'hybridStatisticalModelSoftMaxTemperature': 1,
                    'hybridStatisticalModelAlpha': 1,
                    'hybridStatisticalModelMinOcurrences': 1,
                    'hybridStatisticalModelNumberOfSimulations': 1
                }

                def objectivePoissonMonteCarlo(trial):
                    numOfRepeats = 1 # To average out the rusults before continueing to the next result
                    totalProfit = 0
                    results = [] # Intermediate results

                    # this is needed to reset values to default for preventing non used parameters high jacking the hyperopt
                    modelParams = defautParams

                    modelParams['usePoissonMonteCarlo'] = True
                    modelParams["poissonMonteCarloNumberOfSimulations"] = trial.suggest_int('poissonMonteCarloNumberOfSimulations', 100, 1000, step=100)
                    modelParams["poissonMonteCarloWeightFactor"] = trial.suggest_float('poissonMonteCarloWeightFactor', 0.1, 1.0)
                    

                    if "keno" in dataset_name:
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
                        
                        modelParams["kenoSubset"] = subset

                    #print("Params: ", modelParams)

                    for _ in range(numOfRepeats):
                        profit = predict(f"{dataset_name}", dataPath, skipLastColumns=skip_last_columns, years_back=modelParams['yearsOfHistory'], daysToRebuild=daysToRebuild, modelParams=modelParams)
                        #print("Profit: ", profit)
                        results.append(profit)

                    totalProfit = sum(results) / len(results)

                    return totalProfit
                
                def objectiveMarkov(trial):
                    numOfRepeats = 1 # To average out the rusults before continueing to the next result
                    totalProfit = 0
                    results = [] # Intermediate results

                    # this is needed to reset values to default for preventing non used parameters high jacking the hyperopt
                    modelParams = defautParams

                    modelParams['useMarkov'] = True
                    modelParams['markovSoftMaxTemperature'] = trial.suggest_float('markovSoftMaxTemperature', 0.1, 1.0)
                    modelParams['markovMinOccurences'] = trial.suggest_int('markovMinOccurences', 1, 20)
                    modelParams['markovAlpha'] = trial.suggest_float('markovAlpha', 0.1, 1.0)
                    modelParams['markovRecencyWeight'] = trial.suggest_float('markovRecencyWeight', 0.1, 2.0)
                    modelParams['markovRecencyMode'] = trial.suggest_categorical("markovRecencyMode", ["linear", "log", "constant"])
                    modelParams['markovPairDecayFactor'] = trial.suggest_float('markovPairDecayFactor', 0.1, 2.0)
                    modelParams['markovSmoothingFactor'] = trial.suggest_float('markovSmoothingFactor', 0.01, 1.0)
                    modelParams['markovSubsetSelectionMode'] = trial.suggest_categorical("markovSubsetSelectionMode", ["top", "softmax"])
                    modelParams['markovBlendMode'] = trial.suggest_categorical("markovBlendMode", ["linear", "harmonic", "log"])

                    if "keno" in dataset_name:
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
                        
                        modelParams["kenoSubset"] = subset

                    #print("Params: ", modelParams)

                    for _ in range(numOfRepeats):
                        profit = predict(f"{dataset_name}", dataPath, skipLastColumns=skip_last_columns, years_back=modelParams['yearsOfHistory'], daysToRebuild=daysToRebuild, modelParams=modelParams)
                        #print("Profit: ", profit)
                        results.append(profit)

                    totalProfit = sum(results) / len(results)

                    return totalProfit

                def objectiveMarkovBayesian(trial):
                    numOfRepeats = 1 # To average out the rusults before continueing to the next result
                    totalProfit = 0
                    results = [] # Intermediate results

                    # this is needed to reset values to default for preventing non used parameters high jacking the hyperopt
                    modelParams = defautParams

                    modelParams['useMarkovBayesian'] = True
                    modelParams['markovBlendMode'] = trial.suggest_categorical("markovBlendMode", ["linear", "harmonic", "log"])
                    modelParams['markovBayesianSoftMaxTemperature'] = trial.suggest_float('markovBayesianSoftMaxTemperature', 0.1, 1.0)
                    modelParams['markovBayesianMinOccurences'] = trial.suggest_int('markovBayesianMinOccurences', 1, 20)
                    modelParams['markovBayesianAlpha'] = trial.suggest_float('markovBayesianAlpha', 0.1, 1.0)
                    modelParams['markovBayesianEnhancedSoftMaxTemperature'] = trial.suggest_float('markovBayesianEnhancedSoftMaxTemperature', 0.1, 1.0)
                    modelParams['markovBayesianEnhancedAlpha'] = trial.suggest_float('markovBayesianEnhancedAlpha', 0.1, 1.0)
                    modelParams['markovBayesianEnhancedMinOccurences'] = trial.suggest_int('markovBayesianEnhancedMinOccurences', 1, 20)

                    if "keno" in dataset_name:
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
                        
                        modelParams["kenoSubset"] = subset

                    #print("Params: ", modelParams)

                    for _ in range(numOfRepeats):
                        profit = predict(f"{dataset_name}", dataPath, skipLastColumns=skip_last_columns, years_back=modelParams['yearsOfHistory'], daysToRebuild=daysToRebuild, modelParams=modelParams)
                        #print("Profit: ", profit)
                        results.append(profit)

                    totalProfit = sum(results) / len(results)

                    return totalProfit
                
                def objectiveMarkovBayesianEnhanced(trial):
                    numOfRepeats = 1 # To average out the rusults before continueing to the next result
                    totalProfit = 0
                    results = [] # Intermediate results

                    # this is needed to reset values to default for preventing non used parameters high jacking the hyperopt
                    modelParams = defautParams

                    modelParams['usevMarkovBayesianEnhanced'] = True
                    modelParams['markovBayesianEnhancedSoftMaxTemperature'] = trial.suggest_float('markovBayesianEnhancedSoftMaxTemperature', 0.1, 1.0)
                    modelParams['markovBayesianEnhancedAlpha'] = trial.suggest_float('markovBayesianEnhancedAlpha', 0.1, 1.0)
                    modelParams['markovBayesianEnhancedMinOccurences'] = trial.suggest_int('markovBayesianEnhancedMinOccurences', 1, 20)

                    if "keno" in dataset_name:
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
                        
                        modelParams["kenoSubset"] = subset

                    #print("Params: ", modelParams)

                    for _ in range(numOfRepeats):
                        profit = predict(f"{dataset_name}", dataPath, skipLastColumns=skip_last_columns, years_back=modelParams['yearsOfHistory'], daysToRebuild=daysToRebuild, modelParams=modelParams)
                        #print("Profit: ", profit)
                        results.append(profit)

                    totalProfit = sum(results) / len(results)

                    return totalProfit
                
                def objectivePoissonMarkov(trial):
                    numOfRepeats = 1 # To average out the rusults before continueing to the next result
                    totalProfit = 0
                    results = [] # Intermediate results

                    # this is needed to reset values to default for preventing non used parameters high jacking the hyperopt
                    modelParams = defautParams

                    modelParams['usePoissonMarkov'] = True
                    modelParams['poissonMarkovWeight'] = trial.suggest_float('poissonMarkovWeight', 0.1, 1.0)
                    modelParams['poissonMarkovNumberOfSimulations'] = trial.suggest_int('poissonMarkovNumberOfSimulations', 100, 1000, step=100)

                    if "keno" in dataset_name:
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
                        
                        modelParams["kenoSubset"] = subset

                    #print("Params: ", modelParams)

                    for _ in range(numOfRepeats):
                        profit = predict(f"{dataset_name}", dataPath, skipLastColumns=skip_last_columns, years_back=modelParams['yearsOfHistory'], daysToRebuild=daysToRebuild, modelParams=modelParams)
                        #print("Profit: ", profit)
                        results.append(profit)

                    totalProfit = sum(results) / len(results)

                    return totalProfit
                
                def objectiveLaPlaceMonteCarlo(trial):
                    numOfRepeats = 1 # To average out the rusults before continueing to the next result
                    totalProfit = 0
                    results = [] # Intermediate results

                    # this is needed to reset values to default for preventing non used parameters high jacking the hyperopt
                    modelParams = defautParams

                    modelParams['useLaplaceMonteCarlo'] = True
                    modelParams['laplaceMonteCarloNumberOfSimulations'] = trial.suggest_int('laplaceMonteCarloNumberOfSimulations', 100, 1000, step=100)

                    if "keno" in dataset_name:
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
                        
                        modelParams["kenoSubset"] = subset

                    #print("Params: ", modelParams)

                    for _ in range(numOfRepeats):
                        profit = predict(f"{dataset_name}", dataPath, skipLastColumns=skip_last_columns, years_back=modelParams['yearsOfHistory'], daysToRebuild=daysToRebuild, modelParams=modelParams)
                        #print("Profit: ", profit)
                        results.append(profit)

                    totalProfit = sum(results) / len(results)

                    return totalProfit
                
                def objectiveHybridStatistical(trial):
                    numOfRepeats = 1 # To average out the rusults before continueing to the next result
                    totalProfit = 0
                    results = [] # Intermediate results

                    # this is needed to reset values to default for preventing non used parameters high jacking the hyperopt
                    modelParams = defautParams

                    modelParams['useHybridStatisticalModel'] = True
                    modelParams['hybridStatisticalModelSoftMaxTemperature'] = trial.suggest_float('hybridStatisticalModelSoftMaxTemperature', 0.1, 1.0)
                    modelParams['hybridStatisticalModelAlpha'] = trial.suggest_float('hybridStatisticalModelAlpha', 0.1, 1.0)
                    modelParams['hybridStatisticalModelMinOcurrences'] = trial.suggest_int('hybridStatisticalModelMinOcurrences', 1, 20)
                    modelParams['hybridStatisticalModelNumberOfSimulations'] = trial.suggest_int('hybridStatisticalModelNumberOfSimulations', 100, 1000, step=100)
                    

                    if "keno" in dataset_name:
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
                        
                        modelParams["kenoSubset"] = subset

                    #print("Params: ", modelParams)

                    for _ in range(numOfRepeats):
                        profit = predict(f"{dataset_name}", dataPath, skipLastColumns=skip_last_columns, years_back=modelParams['yearsOfHistory'], daysToRebuild=daysToRebuild, modelParams=modelParams)
                        #print("Profit: ", profit)
                        results.append(profit)

                    totalProfit = sum(results) / len(results)

                    return totalProfit

                # Write best params to json
                jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
                existingData = {}
                if os.path.exists(jsonBestParamsFilePath):
                    with open(jsonBestParamsFilePath, "r") as infile:
                        existingData = json.load(infile)

                totalProfitPoissonMonteCarlo = 0
                totalProfitMarkov = 0
                totalProfitMarkovBayesian = 0
                totalProfitMarkovBayesianEnhanced = 0
                totalProfitPoissonMarkov = 0
                totalProfitLaPlaceMonteCarlo = 0
                totalProfitHybridStatistical = 0
                
                # Create an Optuna study object
                #studyName = f"Sequence-Predictor-Statistical-{dataset_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                studyName = f"{dataset_name}-PoissonMonteCarlo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                study = optuna.create_study(
                    direction='maximize',
                    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                    study_name=studyName,
                    load_if_exists=True
                )

                # Run the automatic tuning process
                study.optimize(objectivePoissonMonteCarlo, n_trials=n_trials)

                # Output the best hyperparameters and score
                print("Best Parameters for Poisson MonteCarlo: ", study.best_params)
                print("Best Score for Poisson MonteCarlo: ", study.best_value)

                totalProfitPoissonMonteCarlo = study.best_value
                # save params
                existingData.update(study.best_params)

                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))

                studyName = f"{dataset_name}-Markov_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                study = optuna.create_study(
                    direction='maximize',
                    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                    study_name=studyName,
                    load_if_exists=True
                )

                # Run the automatic tuning process
                study.optimize(objectiveMarkov, n_trials=n_trials)

                # Output the best hyperparameters and score
                print("Best Parameters for Markov: ", study.best_params)
                print("Best Score for Markov: ", study.best_value)

                totalProfitMarkov = study.best_value
                # save params
                existingData.update(study.best_params)

                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))

                studyName = f"{dataset_name}-MarkovBayesian_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                study = optuna.create_study(
                    direction='maximize',
                    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                    study_name=studyName,
                    load_if_exists=True
                )

                # Run the automatic tuning process
                study.optimize(objectiveMarkovBayesian, n_trials=n_trials)

                # Output the best hyperparameters and score
                print("Best Parameters for Markov Bayesian: ", study.best_params)
                print("Best Score for Markov Bayesian: ", study.best_value)

                totalProfitMarkovBayesian = study.best_value
                # save params
                existingData.update(study.best_params)

                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))

                studyName = f"{dataset_name}-MarkovBayesianEnhanced_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                study = optuna.create_study(
                    direction='maximize',
                    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                    study_name=studyName,
                    load_if_exists=True
                )

                # Run the automatic tuning process
                study.optimize(objectiveMarkovBayesianEnhanced, n_trials=n_trials)

                # Output the best hyperparameters and score
                print("Best Parameters for Markov Bayesian Enhanced: ", study.best_params)
                print("Best Score for Markov Bayesian Enhanced: ", study.best_value)

                totalProfitMarkovBayesianEnhanced = study.best_value
                # save params
                existingData.update(study.best_params)

                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))

                studyName = f"{dataset_name}-PoissonMarkov_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                study = optuna.create_study(
                    direction='maximize',
                    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                    study_name=studyName,
                    load_if_exists=True
                )

                # Run the automatic tuning process
                study.optimize(objectivePoissonMarkov, n_trials=n_trials)

                # Output the best hyperparameters and score
                print("Best Parameters for Poisson Markov: ", study.best_params)
                print("Best Score for Poisson Markov: ", study.best_value)

                totalProfitPoissonMarkov = study.best_value
                # save params
                existingData.update(study.best_params)

                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))

                studyName = f"{dataset_name}-LaPlaceMonteCarlo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                study = optuna.create_study(
                    direction='maximize',
                    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                    study_name=studyName,
                    load_if_exists=True
                )

                # Run the automatic tuning process
                study.optimize(objectiveLaPlaceMonteCarlo, n_trials=n_trials)

                # Output the best hyperparameters and score
                print("Best Parameters for LaPlace MonteCarlo: ", study.best_params)
                print("Best Score for LaPlace MonteCarlo: ", study.best_value)

                totalProfitLaPlaceMonteCarlo = study.best_value
                # save params
                existingData.update(study.best_params)

                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))

                studyName = f"{dataset_name}-HybridStatiscal_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                study = optuna.create_study(
                    direction='maximize',
                    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                    study_name=studyName,
                    load_if_exists=True
                )

                # Run the automatic tuning process
                study.optimize(objectiveHybridStatistical, n_trials=n_trials)

                # Output the best hyperparameters and score
                print("Best Parameters for Hybrid Statistical: ", study.best_params)
                print("Best Score for Hybrid Statistical: ", study.best_value)

                totalProfitHybridStatistical = study.best_value
                # save params
                existingData.update(study.best_params)

                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))

                

                # Determine which strategy is best based on profit
                profits = {
                    'usePoissonMonteCarlo': totalProfitPoissonMonteCarlo,
                    'useMarkov': totalProfitMarkov,
                    'useMarkovBayesian': totalProfitMarkovBayesian,
                    'usevMarkovBayesianEnhanced': totalProfitMarkovBayesianEnhanced,
                    'usePoissonMarkov': totalProfitPoissonMarkov,
                    'useLaplaceMonteCarlo': totalProfitLaPlaceMonteCarlo,
                    'useHybridStatisticalModel': totalProfitHybridStatistical
                }

                # Find the strategy with the maximum profit
                best_strategy = max(profits, key=profits.get)

                # Create booleans for each strategy, only the best is set to True
                strategy_flags = {k: (k == best_strategy) for k in profits}

                print("Strategy outcome: ", strategy_flags)

                existingData.update(strategy_flags)
                
                with open(jsonBestParamsFilePath, "w+") as outfile:
                    json.dump(existingData, outfile, indent=4)
                
                clearFolder(os.path.join(path, "data", "hyperOptCache", f"{dataset_name}"))

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
            helpers.git_push(commit_message="Saving latest statistical hyperopt")
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()  # Ensure the lock is removed even if an error occurs
    
    

    
