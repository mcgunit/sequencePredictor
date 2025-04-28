import os, argparse, json
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
command = Command()
helpers = Helpers()


def print_intro():
    # Generate ASCII art with the text "LSTM"
    ascii_art = text2art("Predictor")
    # Print the introduction and ASCII art
    print("============================================================")
    print("Predictor")
    print("Licence : MIT License")
    print(ascii_art)
    print("Prediction artificial intelligence")

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


def process_single_history_entry(args):
    (historyIndex, historyEntry, historyData, name, model_type, dataPath, modelPath,
     skipLastColumns, years_back, ai, previousJsonFilePath, path) = args

    modelToUse = tcn if "lstm_model" not in model_type else lstm
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
        listOfDecodedPredictions = firstStage(listOfDecodedPredictions, predictedSequence, unique_labels, 2)
    else:
        _, _, _, _, _, _, _, unique_labels = helpers.load_data(
            dataPath, skipLastColumns, years_back=years_back)
        unique_labels = unique_labels.tolist()

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)

    listOfDecodedPredictions = secondStage(
        listOfDecodedPredictions, dataPath, path, name, historyResult,
        unique_labels, jsonFilePath, ai)

    current_json_object["newPrediction"] = listOfDecodedPredictions
    current_json_object["labels"] = unique_labels

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)

    return jsonFilePath



def predict(name, model_type ,dataPath, modelPath, file, skipLastColumns=0, maxRows=0, years_back=None, daysToRebuild=31, ai=False):
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

    kwargs_wget = {
        "folder": dataPath,
        "file": file
    }

    # Lets check if file exists
    if os.path.exists(os.path.join(dataPath, file)):
        os.remove(os.path.join(dataPath, file))
    command.run("wget -P {folder} https://prdlnboppreportsst.blob.core.windows.net/legal-reports/{file}".format(**kwargs_wget), verbose=False)

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
                        latest_raw_predictions, unique_labels = modelToUse.run(name, skipLastColumns, years_back=years_back)
                        
                        predictedSequence = latest_raw_predictions.tolist()

                
                        # Save the current prediction as newPrediction
                        current_json_object["newPredictionRaw"] = predictedSequence
                        current_json_object["labels"] = unique_labels.tolist()

            
                        listOfDecodedPredictions = firstStage(listOfDecodedPredictions, current_json_object["newPredictionRaw"], current_json_object["labels"], 2)
                    else:
                        _, _, _, _, _, _, _, unique_labels = helpers.load_data(dataPath, skipLastColumns, years_back=years_back)
                        unique_labels = unique_labels.tolist()


                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile)
                    
                    listOfDecodedPredictions = secondStage(listOfDecodedPredictions, dataPath, path, name, current_json_object["realResult"], unique_labels, jsonFilePath, ai)


                    current_json_object["newPrediction"] = listOfDecodedPredictions

                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile)

                    #return predictedSequence
                
            if doNewPrediction:
                print(f"No previous prediction file found, Cannot compare. Recreating {daysToRebuild} days of history")

                # Check if there is not a gap or so
                historyData = helpers.getLatestPrediction(dataPath, dateRange=daysToRebuild)
                #print("History data: ", historyData)

                dateOffset = len(historyData)-1 # index of list entry

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
                historyData = historyData[:dateOffset]  # Keep elements before dateOffset because older elements comes after the dateOffset index                
                #print("History to rebuild: ", historyData)

                # Now lets iterate in reversed order to start with the older entries
                reversedHistory = list(reversed(historyData))

                argsList = [
                    (historyIndex, historyEntry, reversedHistory, name, model_type, dataPath,
                    modelPath, skipLastColumns, years_back, ai, previousJsonFilePath, path)
                    for historyIndex, historyEntry in enumerate(reversedHistory)
                ]

                #print("Argslist: ", len(argsList))

                if len(argsList) > 0:
                    #print("Numbers of cpu needed: ", min(cpu_count() - 1, len(argsList)))
                    with Pool(processes=min((cpu_count()-1), len(argsList))) as pool:
                        results = pool.map(process_single_history_entry, argsList)

                    print("Finished multiprocessing rebuild of history entries.")

                    # Find the matching numbers
                    update_matching_numbers(name=name, path=path)
                else:
                    print("No entries to process for: ", name)
                

                #return predictedSequence
        else:
            print("Prediction already made")
    else:
        print("Did not found entries")


def firstStage(listOfDecodedPredictions, newPredictionRaw, labels, nOfPredictions):
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


def secondStage(listOfDecodedPredictions, dataPath, path, name, historyResult, unique_labels, jsonFilePath, ai):
    #####################
    # Start refinements #
    #####################
    jsonDirPath = os.path.join(path, "data", "database", name)
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
    
    try:
        # Markov
        print("Performing Markov Prediction")
        markov.setDataPath(dataPath)
        markov.setSoftMAxTemperature(0.5971260746885267) # Determined with hyperopt
        markov.setMinOccurrences(2) # Determined with hyperopt
        markov.setAlpha(1.11636426149007) # Determined with hyperopt
        markov.clear()

        markovPrediction = {
            "name": "Markov Model",
            "predictions": []
        }

        subsets = []
        if "keno" in name:
            subsets = [5, 6, 7, 8, 9, 10]

        markovSequence, markovSubsets = markov.run(generateSubsets=subsets)
        
        markovPrediction["predictions"].append(markovSequence)
        for key in markovSubsets:
            markovPrediction["predictions"].append(markovSubsets[key])

        listOfDecodedPredictions.append(markovPrediction)
    except Exception as e:
        print("Failed to perform Markov: ", e)

    try:
        # Markov Bayesian
        print("Performing Markov Bayesian Prediction")
        markovBayesian.setDataPath(dataPath)
        markovBayesian.setSoftMAxTemperature(0.1)
        markovBayesian.setAlpha(0.3)
        markovBayesian.setMinOccurrences(10)
        markovBayesian.clear()

        markovBayesianPrediction = {
            "name": "MarkovBayesian Model",
            "predictions": []
        }

        subsets = []
        if "keno" in name:
            subsets = [5, 6, 7, 8, 9, 10]

        markovBayesianSequence, markovBayesianSubsets = markovBayesian.run(generateSubsets=subsets)
        markovBayesianPrediction["predictions"].append(markovBayesianSequence)
        for key in markovBayesianSubsets:
            markovBayesianPrediction["predictions"].append(markovBayesianSubsets[key])

        listOfDecodedPredictions.append(markovBayesianPrediction)
    except Exception as e:
        print("Failed to perform Markov Bayesian: ", e)

    if not "pick3" in name:
        try:
            # Markov Bayesian Enhanced
            print("Performing Markov Bayesian Enhanced Prediction")
            markovBayesianEnhanced.setDataPath(dataPath)
            markovBayesianEnhanced.setSoftMAxTemperature(0.1)
            markovBayesianEnhanced.setAlpha(0.7)
            markovBayesianEnhanced.setMinOccurrences(10)
            markovBayesianEnhanced.clear()

            markovBayesianEnhancedPrediction = {
                "name": "MarkovBayesianEnhanched Model",
                "predictions": []
            }

            subsets = []
            if "keno" in name:
                subsets = [5, 6, 7, 8, 9, 10]

            markovBayesianEnhancedSequence, markovBayesianEnhancedSubsets = markovBayesianEnhanced.run(generateSubsets=subsets)
            markovBayesianEnhancedPrediction["predictions"].append(markovBayesianEnhancedSequence)
            for key in markovBayesianSubsets:
                markovBayesianEnhancedPrediction["predictions"].append(markovBayesianEnhancedSubsets[key])

            listOfDecodedPredictions.append(markovBayesianEnhancedPrediction)
        except Exception as e:
            print("Failed to perform Markov Bayesian Enhanced: ", e)

    try:
        # Poisson Distribution with Monte Carlo Analysis
        print("Performing Poisson Monte Carlo Prediction")
        poissonMonteCarlo.setDataPath(dataPath)
        poissonMonteCarlo.setNumOfSimulations(1000)
        poissonMonteCarlo.setWeightFactor(0.1)
        poissonMonteCarlo.clear()

        poissonMonteCarloPrediction = {
            "name": "PoissonMonteCarlo Model",
            "predictions": []
        }

        subsets = []
        if "keno" in name:
            subsets = [5, 6, 7, 8, 9, 10]

        poissonMonteCarloSequence, poissonMonteCarloSubsets = poissonMonteCarlo.run(generateSubsets=subsets)

        poissonMonteCarloPrediction["predictions"].append(poissonMonteCarloSequence)
        for key in poissonMonteCarloSubsets:
            poissonMonteCarloPrediction["predictions"].append(poissonMonteCarloSubsets[key])

        listOfDecodedPredictions.append(poissonMonteCarloPrediction)    
    except Exception as e:
        print("Failed to perform Poisson Distribution with Monte Carlo Analysis: ", e)

    try:
        # Poisson-Markov Distribution
        print("Performing Poisson-Markov Prediction")
        poissonMarkov.setDataPath(dataPath)
        poissonMarkov.setWeights(poisson_weight=0.3, markov_weight=0.7)
        poissonMarkov.setNumberOfSimulations(1000)

        poissonMarkovPrediction = {
            "name": "PoissonMarkov Model",
            "predictions": []
        }

        subsets = []
        if "keno" in name:
            subsets = [5, 6, 7, 8, 9, 10]

        poissonMarkovSequence, poissonMarkovSubsets = poissonMarkov.run(generateSubsets=subsets)

        poissonMarkovPrediction["predictions"].append(poissonMarkovSequence)
        for key in poissonMarkovSubsets:
            poissonMarkovPrediction["predictions"].append(poissonMarkovSubsets[key])

        listOfDecodedPredictions.append(poissonMarkovPrediction)    
    except Exception as e:
        print("Failed to perform Poisson-Markov Distribution: ", e)


    try:
        # Laplace Distribution with Monte Carlo Analysis
        print("Performing Laplace Monte Carlo Prediction")
        laplaceMonteCarlo.setDataPath(dataPath)
        laplaceMonteCarlo.setNumOfSimulations(1000)
        laplaceMonteCarlo.clear()

        laplaceMonteCarloPrediction = {
            "name": "LaplaceMonteCarlo Model",
            "predictions": []
        }

        subsets = []
        if "keno" in name:
            subsets = [5, 6, 7, 8, 9, 10]

        laplaceMonteCarloSequence, laplaceMonteCarloSubsets = laplaceMonteCarlo.run(generateSubsets=subsets)
        laplaceMonteCarloPrediction["predictions"].append(laplaceMonteCarloSequence)
        for key in laplaceMonteCarloSubsets:
            laplaceMonteCarloPrediction["predictions"].append(laplaceMonteCarloSubsets[key])
        
        listOfDecodedPredictions.append(laplaceMonteCarloPrediction)
    except Exception as e:
        print("Failed to perform Laplace Distribution with Monte Carlo Analysis: ", e)

    try:
        # Hybrid Statistical Model
        print("Performing Hybrid Statistical Model Prediction")
        hybridStatisticalModel.setDataPath(dataPath)
        hybridStatisticalModel.setSoftMaxTemperature(0.1)
        hybridStatisticalModel.setAlpha(0.7)
        hybridStatisticalModel.setMinOccurrences(10)
        hybridStatisticalModel.setNumberOfSimulations(1000)
        hybridStatisticalModel.clear()

        hybridStatisticalModelPrediction = {
            "name": "HybridStatisticalModel",
            "predictions": []
        }

        subsets = []
        if "keno" in name:
            subsets = [5, 6, 7, 8, 9, 10]

        hybridStatisticalModelSequence, hybridStatisticalModelSubsets = hybridStatisticalModel.run(generateSubsets=subsets)
        hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSequence)
        for key in hybridStatisticalModelSubsets:
            hybridStatisticalModelPrediction["predictions"].append(hybridStatisticalModelSubsets[key])
        
        listOfDecodedPredictions.append(hybridStatisticalModelPrediction)
    except Exception as e:
        print("Failed to perform Hybrid Statistical Model: ", e)


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
    parser.add_argument('-d', '--days', type=int, default=14)
    args = parser.parse_args()

    print_intro()

    current_year = datetime.now().year
    print("Current Year:", current_year)

    daysToRebuild = int(args.days)
    rebuildHistory = bool(args.rebuild_history)


    path = os.getcwd()

    datasets = [
        # (dataset_name, model_type, skip_last_columns, ai)
        ("euromillions", "tcn_model", 0, False),
        ("lotto", "lstm_model", 0, False),
        ("eurodreams", "lstm_model", 0, False),
        #("jokerplus", "lstm_model", 1, False),
        ("keno", "lstm_model", 0, False),
        ("pick3", "lstm_model", 0, False),
        ("vikinglotto", "lstm_model", 0, False),
    ]

    for dataset_name, model_type, skip_last_columns, ai in datasets:
        try:
            print(f"\n{dataset_name.capitalize()}")
            modelPath = os.path.join(path, "data", "models", model_type)
            dataPath = os.path.join(path, "data", "trainingData", dataset_name)
            file = f"{dataset_name}-gamedata-NL-{current_year}.csv"

            # Predict for complete data
            predict(dataset_name, model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, daysToRebuild=daysToRebuild, ai=ai)

            # Predict for current year
            predict(f"{dataset_name}_currentYear", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=1, daysToRebuild=daysToRebuild, ai=ai)

            # Predict for current year + last year
            predict(f"{dataset_name}_twoYears", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=2, daysToRebuild=daysToRebuild, ai=ai)

            # Predict for current year + last two years
            predict(f"{dataset_name}_threeYears", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=3, daysToRebuild=daysToRebuild, ai=ai)

        except Exception as e:
            print(f"Failed to predict {dataset_name.capitalize()}: {e}")

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

    # try:
    #     helpers.git_push()
    # except Exception as e:
    #     print("Failed to push latest predictions:", e)
    
    

    
