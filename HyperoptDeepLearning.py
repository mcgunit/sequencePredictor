import os, argparse, json, sys
import optuna
import numpy as np

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
    (historyIndex, historyEntry, historyData, name, model_type, dataPath, modelPath,
        skipLastColumns, years_back, previousJsonFilePath, path, modelParams) = args

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


    modelToUse.setDataPath(dataPath)
    modelToUse.setModelPath(modelPath)
    modelToUse.setLoadModelWeights(True) 
    modelToUse.setBatchSize(modelParams["batchSize"])
    modelToUse.setEpochs(modelParams["epochs"])
    modelToUse.setNumberOfLSTMLayers(modelParams["num_lstm_layers"])
    modelToUse.setNumberOfBidrectionalLayers(modelParams["num_bidirectional_layers"])
    modelToUse.setNumberOfLstmUnits(modelParams["lstm_units"])
    modelToUse.setNumberOfBidirectionalLstmUnits(modelParams["bidirectional_lstm_units"])
    modelToUse.setDropout(modelParams["dropout"])
    modelToUse.setL2Regularization(modelParams["l2Regularization"])
    modelToUse.setEarlyStopPatience(modelParams["earlyStopPatience"])
    modelToUse.setReduceLearningRatePAience(modelParams["reduceLearningRatePatience"])
    modelToUse.setReducedLearningRateFactor(modelParams["reduceLearningRateFactor"])
    modelToUse.setUseFinalLSTMLayer(modelParams["useFinalLSTMLayer"])
    modelToUse.setOutpuActivation(modelParams["outputActivation"])
    modelToUse.setOptimizer(modelParams["optimizer"])
    modelToUse.setLearningRate(modelParams["learningRate"])
    modelToUse.setWindowSize(modelParams["windowSize"])
    modelToUse.setPredictionWindowSize(modelParams["windowSize"])
    modelToUse.setMarkovAlpha(modelParams["lstmMarkovAlpha"])
    modelToUse.setLabelSmoothing(modelParams["labelSmoothing"])

    # Perform training
    latest_raw_predictions, unique_labels = modelToUse.run(
        name, skipLastColumns, skipRows=len(historyData)-historyIndex, years_back=years_back)
    
    predictedSequence = latest_raw_predictions.tolist()
    unique_labels = unique_labels.tolist()
    current_json_object["newPredictionRaw"] = predictedSequence
    listOfDecodedPredictions = deepLearningMethod(listOfDecodedPredictions, predictedSequence, unique_labels, 1, name, historyResult, jsonFilePath, modelParams)


    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)


    current_json_object["newPrediction"] = listOfDecodedPredictions
    current_json_object["labels"] = unique_labels

    with open(jsonFilePath, "w+") as outfile:
        json.dump(current_json_object, outfile)

    return jsonFilePath



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

def predict(name, model_type ,dataPath, modelPath, file, skipLastColumns=0, maxRows=0, years_back=None, daysToRebuild=31, modelParams={}):
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
                modelPath, skipLastColumns, years_back, previousJsonFilePath, path, modelParams)
                for historyIndex, historyEntry in enumerate(historyData)
            ]

            
            numberOfProcesses = 1

            with Pool(processes=numberOfProcesses) as pool:
                results = pool.map(process_single_history_entry, argsList)

            print("Finished rebuild of history entries.")

            # Find the matching numbers
            update_matching_numbers(name=name, path=path)

            # Calculate Profit
            profit =  helpers.calculate_profit(name=name, path=path)

            return profit
        else:
            print("Prediction already made")
    else:
        print("Did not found entries")


def deepLearningMethod(listOfDecodedPredictions, newPredictionRaw, labels, nOfPredictions, name, historyResult, jsonFilePath, modelParams):

    jsonDirPath = os.path.join(path, "data", "hyperOptCache", name)
    num_classes = len(labels)
    numbersLength = len(historyResult)

    nthPredictions = {
        "name": "LSTM Base Model",
        "predictions": []
    }
    # Decode prediction with nth highest probability
    predicted_digits = np.argmax(newPredictionRaw, axis=-1)
    print("Prediction: ", predicted_digits.tolist())
    nthPredictions["predictions"].append(predicted_digits.tolist())

    
    if modelParams["useTopPrediction"]:
        try:
            predicted_digits = np.argmax(newPredictionRaw, axis=-1) 
            top3_indices = np.argsort(newPredictionRaw, axis=-1)[:, -3:][:, ::-1]
            nthPredictions["predictions"].append(top3_indices[0].tolist())
        except Exception as e:
            print("Failed to parse the top prediction: ", e)

    if modelParams["useLstmMarkovPrediction"]:
        try:
            top3_indices_lstm_markov = np.argsort(lstm.getLstmMArkov(), axis=-1)[:, -3:][:, ::-1]
            nthPredictions["predictions"].append(top3_indices_lstm_markov[0].tolist())
        except Exception as e:
            print("Failed to parse lstm+markov: ", e)
    
    listOfDecodedPredictions.append(nthPredictions)

    """
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
            prediction_highest_indices = helpers.decode_predictions(refined_prediction_raw[0], labels, nHighestProb=i)
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
        topPrediction = helpers.getTopPredictions(top_prediction_raw, labels, num_top=numbersLength)

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
    """

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
        # (dataset_name, model_type, skip_last_columns)
        #("euromillions", "lstm_model", 0),
        #("lotto", "lstm_model", 0),
        #("eurodreams", "lstm_model", 0),
        #("jokerplus", "lstm_model", 1),
        #("keno", "lstm_model", 0),
        ("pick3", "lstm_model", 0),
        #("vikinglotto", "lstm_model", 0),
    ]

    for dataset_name, model_type, skip_last_columns in datasets:
        try:
            print(f"\n{dataset_name.capitalize()}")
            modelPath = os.path.join(path, "data", "hyperOptCache", "models", model_type)
            dataPath = os.path.join(path, "data", "trainingData", dataset_name)
            file = f"{dataset_name}-gamedata-NL-{current_year}.csv"

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


                modelParams =  {
                    "yearsOfHistory": trial.suggest_categorical("yearsOfHistory", [20]),
                    "epochs": trial.suggest_categorical("epochs", [1000]),
                    "batchSize": trial.suggest_categorical("batchSize", [8]),
                    "num_lstm_layers": trial.suggest_categorical("num_lstm_layers", [1, 2, 3]),
                    "num_bidirectional_layers": trial.suggest_categorical("num_bidirectional_layers", [1, 2, 3]),
                    "lstm_units": trial.suggest_categorical("lstm_units", [16, 32, 64, 128]),
                    "bidirectional_lstm_units": trial.suggest_categorical("bidirectional_lstm_units", [16, 32, 64, 128]),
                    "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
                    "l2Regularization": trial.suggest_float("l2Regularization", 0.0001, 0.01, step=0.0001),
                    "earlyStopPatience": trial.suggest_int("earlyStopPatience", 10, 100, step=10),
                    "reduceLearningRatePatience": trial.suggest_int("reduceLearningRatePatience", 1, 100, step=10),
                    "reduceLearningRateFactor": trial.suggest_float("reduceLearningRateFactor", 0.1, 0.9, step=0.1),
                    "useFinalLSTMLayer": trial.suggest_categorical("useFinalLSTMLayer", [True, False]),
                    "outputActivation": trial.suggest_categorical("outputActivation", ["softmax"]),  # keep fixed unless needed
                    "optimizer": trial.suggest_categorical("optimizer_type", ["adam", "rmsprop", "adagrad", "nadam"]), # "sgd", does not work with categorical crossentropy
                    "learningRate": trial.suggest_float("learningRate", 0.00001, 0.001, step=0.00001),
                    "windowSize": trial.suggest_int("windowSize", 5, 100, step=5),
                    "lstmMarkovAlpha": trial.suggest_float("lstmMarkovAlpha", 0.01, 1.0, step=0.01),
                    "useLstmMarkovPrediction": trial.suggest_categorical("useLstmMarkovPrediction", [True, False]),
                    "useTopPrediction": trial.suggest_categorical("useTopPrediction", [True, False]),
                    "labelSmoothing": trial.suggest_float("labelSmoothing", 0.01, 0.1, step=0.01)
                }

                for _ in range(numOfRepeats):
                    profit = predict(f"{dataset_name}", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=modelParams["yearsOfHistory"], daysToRebuild=daysToRebuild, modelParams=modelParams)
                    #print("Profit: ", profit)
                    results.append(profit)

                totalProfit = sum(results) / len(results)


                clearFolder(os.path.join(path, "data", "hyperOptCache", "models", model_type))

                return totalProfit
            
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
                study_name=f"{dataset_name}-LSTM",
                load_if_exists=True
            )

            # Run the automatic tuning process
            study.optimize(objective, n_trials=500)

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
        helpers.git_push(commit_message="Saving latest deep learning hyperopt")
    except Exception as e:
        print("Failed to push latest predictions:", e)
    finally:
        remove_lock()  # Ensure the lock is removed even if an error occurs
    

    
