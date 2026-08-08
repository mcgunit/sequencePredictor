import os, argparse, json, sys
import optuna
import numpy as np

from art import text2art
from datetime import datetime


from src.TCN import TCNModel
from src.LSTM import LSTMModel
from src.UnifiedLstmTcn import UnifiedLstmTcnModel
from src.UnifiedLstmGruTcn import UnifiedLstmGruTcnModel
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

# Prefix used for each new model's Optuna param names / bestParams_<game>.json
# keys - Predictor.py's runUnifiedDeepLearningModels() reads these same
# prefixed keys. Without a prefix, two independent per-model_type Optuna
# studies writing e.g. a bare "batchSize" would silently clobber each other
# (and LSTM's own already-tuned bare "batchSize") in bestParams_<game>.json -
# same reasoning as HyperoptStatistics.py's suggest_keno_subset docstring.
MODEL_PARAM_PREFIX = {
    "unified_lstm_tcn_model": "unifiedLstmTcn",
    "unified_lstm_gru_tcn_model": "unifiedLstmGruTcn",
}

MODEL_DISPLAY_NAMES = {
    "lstm_model": "LSTM Base Model",
    "tcn_model": "TCN Base Model",
    "unified_lstm_tcn_model": "UnifiedLstmTcn Model",
    "unified_lstm_gru_tcn_model": "UnifiedLstmGruTcn Model",
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
    model.setMarkovAlpha(modelParams["lstmMarkovAlpha"])
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
    model.setMarkovAlpha(modelParams["lstmMarkovAlpha"])


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
    unique_labels = unique_labels
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
            for subset_size in range(5, 11):
                subset = helpers.generate_subset_from_scores(number_scores, predicted_digits, subset_size)
                nthPredictions["predictions"].append(subset)
        except Exception as e:
            print("Failed to generate keno subsets: ", e)

    # useTopPrediction/useLstmMarkovPrediction are LSTM-only knobs (see
    # suggest_fused_params, which doesn't set them) - .get() with a default
    # so the two new model types don't crash here.
    if modelParams.get("useTopPrediction", False):
        try:
            predicted_digits = np.argmax(newPredictionRaw, axis=-1)
            top3_indices = np.argsort(newPredictionRaw, axis=-1)[:, -3:][:, ::-1]
            nthPredictions["predictions"].append(top3_indices[0].tolist())
        except Exception as e:
            print("Failed to parse the top prediction: ", e)

    if modelParams.get("useLstmMarkovPrediction", False):
        try:
            top3_indices_lstm_markov = np.argsort(lstm.getLstmMArkov(), axis=-1)[:, -3:][:, ::-1]
            nthPredictions["predictions"].append(top3_indices_lstm_markov[0].tolist())
        except Exception as e:
            print("Failed to parse lstm+markov: ", e)

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
    parser.add_argument('-t', '--trials', type=int, default=150)
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
    n_trials = int(args.trials)
    pushToGit = bool(args.save)

    print("Push to git: ", pushToGit)
    print("Running ", n_trials, "trials")

    # Convert the comma-separated string into a clean list
    games = [g.strip() for g in args.games.split(',') if g.strip()]

    print("Selected games:", games)

    path = os.getcwd()

    # Every game gets its own independent study per model type, so
    # UnifiedLstmTcn Model / UnifiedLstmGruTcn Model get tuned (and tracked)
    # separately from LSTM Base Model rather than sharing one set of params -
    # same reasoning as MODEL_PARAM_PREFIX above.
    dl_model_types = ["lstm_model", "unified_lstm_tcn_model", "unified_lstm_gru_tcn_model"]
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
                            "lstmMarkovAlpha": trial.suggest_float("lstmMarkovAlpha", 0.01, 0.1, step=0.01),
                            "useLstmMarkovPrediction": trial.suggest_categorical("useLstmMarkovPrediction", [False]),
                            "useTopPrediction": trial.suggest_categorical("useTopPrediction", [False]),
                            "labelSmoothing": trial.suggest_float("labelSmoothing", 0.01, 0.1, step=0.01)
                        }
                    else:
                        modelParams = suggest_fused_params(
                            trial, MODEL_PARAM_PREFIX[model_type],
                            include_gru=(model_type == "unified_lstm_gru_tcn_model"))

                    for _ in range(numOfRepeats):
                        result = predict(f"{dataset_name}", model_type, dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=modelParams["yearsOfHistory"], daysToRebuild=daysToRebuild, modelParams=modelParams, specialColumnCount=special_column_count)
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
                    PROFIT_WEIGHT = 0.05
                    return -avgValLoss + PROFIT_WEIGHT * avgProfit
                
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

                # Run the automatic tuning process
                study.optimize(objective, n_trials=n_trials)

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
    

    
