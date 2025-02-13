import os, argparse, json
import numpy as np

from art import text2art
from datetime import datetime

from src.TCN import TCNModel
from src.LSTM import LSTMModel
from src.LSTM_ARIMA_Model import LSTM_ARIMA_Model
from src.RefinemePrediction import RefinePrediction
from src.TopPrediction import TopPrediction
from src.Markov import Markov
from src.Command import Command
from src.Helpers import Helpers

tcn = TCNModel()
lstm = LSTMModel()
lstmArima = LSTM_ARIMA_Model()
refinePrediction = RefinePrediction()
topPredictor = TopPrediction()
markov = Markov()
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



def predict(name, dataPath, modelPath, file, skipLastColumns=0, maxRows=0, years_back=None):

    modelToUse = tcn
    if "lotto" in name or "eurodreams" in name or "jokerplus" in name or "keno" in name or "pick3" in name or "vikinglotto" in name:
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
    latestEntry, previousEntry = helpers.getLatestPrediction(os.path.join(dataPath, file))
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

                    listOfMatching = []
                    # Check on prediction with nth highest probability
                    for i in range(len(current_json_object["currentPrediction"])):
                        matching_numbers = helpers.find_matching_numbers(current_json_object["realResult"], current_json_object["currentPrediction"][i])
                        #print("Matching Numbers with ", i+1 ,"highest probs: ", matching_numbers)
                        listOfMatching.append({
                            "index": i,
                            "matchingSequence": current_json_object["currentPrediction"][i],
                            "matchingNumbers": matching_numbers
                        })
                    
                    # Use max with a key to find the dictionary with the largest 'matchingNumbers' list
                    largest_matching_object = max(listOfMatching, key=lambda x: len(x['matchingNumbers']))


                    current_json_object["matchingNumbers"] = {
                        "bestMatchIndex": largest_matching_object["index"],
                        "bestMatchSequence": largest_matching_object["matchingSequence"],
                        "matchingNumbers": largest_matching_object["matchingNumbers"]
                    }

                    print("matching_numbers: ", current_json_object["matchingNumbers"]["matchingNumbers"])

                    # Train and do a new prediction
                    modelToUse.setModelPath(modelPath)
                    modelToUse.setBatchSize(16)
                    modelToUse.setEpochs(1000)
                    latest_raw_predictions, unique_labels = modelToUse.run(name, skipLastColumns, years_back=years_back)
                    
                    predictedSequence = latest_raw_predictions.tolist()

            
                    # Save the current prediction as newPrediction
                    current_json_object["newPredictionRaw"] = predictedSequence
                    current_json_object["labels"] = unique_labels.tolist()

                    listOfDecodedPredictions = []
                    # Decode prediction with nth highest probability
                    for i in range(2):
                        prediction_nth_indices = helpers.decode_predictions(current_json_object["newPredictionRaw"], current_json_object["labels"], i)
                        listOfDecodedPredictions.append(prediction_nth_indices)


                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile)
                    
                    listOfDecodedPredictions = secondStage(listOfDecodedPredictions, dataPath, path, name, historyResult, unique_labels, jsonFilePath)


                    current_json_object["newPrediction"] = listOfDecodedPredictions

                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile)

                    #return predictedSequence
                
            if doNewPrediction:
                print("No previous prediction file found, Cannot compare. Recreating one month of history")

                # Check if there is not a gap or so
                historyData = helpers.getLatestPrediction(os.path.join(dataPath, file), dateRange=1)
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
                for historyIndex, historyEntry in enumerate(reversed(historyData)):
                    historyDate = historyEntry[0]
                    historyResult = historyEntry[1]
                    jsonFileName = f"{historyDate.year}-{historyDate.month}-{historyDate.day}.json"
                    #print(jsonFileName, ":", historyResult)
                    jsonFilePath = os.path.join(path, "data", "database", name, jsonFileName)

                    if historyIndex == 0:
                        print("oldest entry: ", historyEntry)
                        
                        current_json_object = {
                            "currentPredictionRaw": [],
                            "currentPrediction": [],
                            "realResult": historyResult,
                            "newPrediction": [],    # Decoded prediction according to formula in decode_prediction
                            "newPredictionRaw": [], # Raw prediction that contains the statistical data
                            "matchingNumbers": [],
                            "labels": [],
                            "numberFrequency": helpers.count_number_frequencies(dataPath)
                        }

                        # Connect the history
                        if previousJsonFilePath:
                            print("Starting from: ", previousJsonFilePath)
                            # Opening JSON file
                            with open(previousJsonFilePath, 'r') as openfile:
                                # Reading from json file
                                previous_json_object = json.load(openfile)

                            current_json_object["currentPredictionRaw"] = previous_json_object["newPredictionRaw"]
                            current_json_object["currentPrediction"] = previous_json_object["newPrediction"]

                            listOfMatchings = []
                            # Check on prediction with nth highest probability
                            for i in range(len(current_json_object["currentPrediction"])):
                                matching_numbers = helpers.find_matching_numbers(current_json_object["realResult"], current_json_object["currentPrediction"][i])
                                #print("Matching Numbers with ", i+1 ,"highest probs: ", matching_numbers)
                                listOfMatchings.append({
                                    "index": i,
                                    "matchingSequence": current_json_object["currentPrediction"][i],
                                    "matchingNumbers": matching_numbers
                                })
                            

                            # Use max with a key to find the dictionary with the largest 'matchingNumbers' list
                            largest_matching_object = max(listOfMatchings, key=lambda x: len(x['matchingNumbers']))


                            current_json_object["matchingNumbers"] = {
                                "bestMatchIndex": largest_matching_object["index"],
                                "bestMatchSequence": largest_matching_object["matchingSequence"],
                                "matchingNumbers": largest_matching_object["matchingNumbers"]
                            }

                            print("matching_numbers: ", current_json_object["matchingNumbers"]["matchingNumbers"])


                        # Train and do a new prediction
                        modelToUse.setDataPath(dataPath)
                        
                        modelToUse.setModelPath(modelPath)
                        modelToUse.setBatchSize(16)
                        modelToUse.setEpochs(1000)
                        latest_raw_predictions, unique_labels = modelToUse.run(name, skipLastColumns, skipRows=len(historyData)-historyIndex , years_back=years_back)

                        predictedSequence = latest_raw_predictions.tolist()
                        unique_labels = unique_labels.tolist()

                        listOfDecodedPredictions = []
                        for i in range(2):
                            prediction_nth_indices = helpers.decode_predictions(predictedSequence, unique_labels, i)
                            listOfDecodedPredictions.append(prediction_nth_indices)
                
                        # Save the current prediction as newPrediction
                        current_json_object["newPredictionRaw"] = predictedSequence

                        with open(jsonFilePath, "w+") as outfile:
                            json.dump(current_json_object, outfile)
                        
                        listOfDecodedPredictions = secondStage(listOfDecodedPredictions, dataPath, path, name, historyResult, unique_labels, jsonFilePath)

                        current_json_object["newPrediction"] = listOfDecodedPredictions
                        current_json_object["labels"] = unique_labels

                        # store the decoded and refined predictions
                        with open(jsonFilePath, "w+") as outfile:
                            json.dump(current_json_object, outfile)
                    else:
                        # The previous file should be created at index 0

                        # Opening JSON file
                        with open(previousJsonFilePath, 'r') as openfile:
                        
                            # Reading from json file
                            previous_json_object = json.load(openfile)

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
                        
                        #print(previous_json_object)
                        #print(type(previous_json_object))

                        # The current prediction is the new prediction from the previous one
                        current_json_object["currentPredictionRaw"] = previous_json_object["newPredictionRaw"]
                        current_json_object["currentPrediction"] = previous_json_object["newPrediction"]

                        #print(current_json_object["currentPredictionRaw"])

                        listOfMatchings = []
                        # Compare decoded and refined predictions stored in currentPrediction with the real result (drawing)
                        for i in range(len(current_json_object["currentPrediction"])):
                            matching_numbers = helpers.find_matching_numbers(current_json_object["realResult"], current_json_object["currentPrediction"][i])
                            print("Matching Numbers with ", i+1 , matching_numbers)
                            listOfMatchings.append({
                                "index": i,
                                "matchingSequence": current_json_object["currentPrediction"][i],
                                "matchingNumbers": matching_numbers
                            })
                        

                        # Use max with a key to find the dictionary with the largest 'matchingNumbers' list
                        largest_matching_object = max(listOfMatchings, key=lambda x: len(x['matchingNumbers']))


                        current_json_object["matchingNumbers"] = {
                            "bestMatchIndex": largest_matching_object["index"],
                            "bestMatchSequence": largest_matching_object["matchingSequence"],
                            "matchingNumbers": largest_matching_object["matchingNumbers"]
                        }

                        print("matching_numbers: ", current_json_object["matchingNumbers"]["matchingNumbers"])

                        # Train and do a new prediction
                        modelToUse.setModelPath(modelPath)
                        modelToUse.setBatchSize(16)
                        modelToUse.setEpochs(1000)
                        latest_raw_predictions, unique_labels = modelToUse.run(name, skipLastColumns, skipRows=len(historyData)-historyIndex, years_back=years_back)

                        predictedSequence = latest_raw_predictions.tolist()

                        # Save the current prediction as newPrediction
                        current_json_object["newPredictionRaw"] = predictedSequence
                        current_json_object["labels"] = unique_labels.tolist()
                        
                        listOfDecodedPredictions = []
                        # Decode prediction with nth highest probability
                        for i in range(2):
                            prediction_nth_indices = helpers.decode_predictions(current_json_object["newPredictionRaw"], current_json_object["labels"], i)
                            listOfDecodedPredictions.append(prediction_nth_indices)

                        with open(jsonFilePath, "w+") as outfile:
                            json.dump(current_json_object, outfile)

                        listOfDecodedPredictions = secondStage(listOfDecodedPredictions, dataPath, path, name, historyResult, unique_labels, jsonFilePath)

                        # store the decoded and refined predictions
                        current_json_object["newPrediction"] = listOfDecodedPredictions

                        with open(jsonFilePath, "w+") as outfile:
                            json.dump(current_json_object, outfile)

                    previousJsonFilePath = jsonFilePath

                #return predictedSequence
        else:
            print("Prediction already made")
    else:
        print("Did not found entries")


def secondStage(listOfDecodedPredictions, dataPath, path, name, historyResult, unique_labels, jsonFilePath):
    #####################
    # Start refinements #
    #####################
    jsonDirPath = os.path.join(path, "data", "database", name)
    num_classes = len(unique_labels)
    numbersLength = len(historyResult)

    try:
        # Refine predictions
        refinePrediction.trainRefinePredictionsModel(name, jsonDirPath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)
        refined_prediction_raw = refinePrediction.refinePrediction(name=name, pathToLatestPredictionFile=jsonFilePath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)

        #print("refined_prediction_raw: ", refined_prediction_raw)

        for i in range(2):
            prediction_highest_indices = helpers.decode_predictions(refined_prediction_raw[0], unique_labels, nHighestProb=i)
            #print("Refined Prediction with ", i+1 ,"highest probs: ", prediction_highest_indices)
            listOfDecodedPredictions.append(prediction_highest_indices)
    except Exception as e:
        print("Was not able to run refine prediction model: ", e)

    try:
        # Top prediction
        topPredictor.trainTopPredictionsModel(name, jsonDirPath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)
        top_prediction_raw = topPredictor.topPrediction(name=name, pathToLatestPredictionFile=jsonFilePath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)
        topPrediction = helpers.getTopPredictions(top_prediction_raw, unique_labels, num_top=numbersLength)

        # Print Top prediction
        for i, prediction in enumerate(topPrediction):
            topHighestProbPrediction = [int(num) for num in prediction]
            #print(f"Top Prediction {i+1}: {sorted(topHighestProbPrediction)}")
            listOfDecodedPredictions.append(topHighestProbPrediction)
    except Exception as e:
        print("Was not able to run top prediction model: ", e)

    try:
        # Arima prediction
        lstmArima.setModelPath(os.path.join(path, "data", "models", "lstm_arima_model"))
        lstmArima.setDataPath(dataPath)
        lstmArima.setBatchSize(8)
        lstmArima.setEpochs(1000)

        predicted_arima_sequence = lstmArima.run(name)
        listOfDecodedPredictions.append(predicted_arima_sequence)

    except Exception as e:
        print("Failed to perform ARIMA: ", e)

    try:
        # Markov
        markov.setDataPath(dataPath)
        markovSequence = markov.run() 
        listOfDecodedPredictions.append(markovSequence)
    except Exception as e:
        print("Failed to perform Markov: ", e)
        exit()

    return listOfDecodedPredictions


if __name__ == "__main__":
    try:
        helpers.git_pull()
    except Exception as e:
        print("Failed to get latest changes")

    parser = argparse.ArgumentParser(
        prog='LSTM Sequence Predictor',
        description='Tries to predict a sequence of numbers',
        epilog='Check it out'
    )
    parser.add_argument('-d', '--data', default="euromillions")
    args = parser.parse_args()
    print(args.data)

    print_intro()

    current_year = datetime.now().year
    print("Current Year:", current_year)

    path = os.getcwd()

    datasets = [
        # (dataset_name, model_type, skip_last_columns)
        #("euromillions", "tcn_model", 0),
        #("lotto", "lstm_model", 1),
        #("eurodreams", "lstm_model", 0),
        #("jokerplus", "lstm_model", 1),
        ("keno", "lstm_model", 0),
        #("pick3", "lstm_model", 0),
        #("vikinglotto", "lstm_model", 0),
    ]

    for dataset_name, model_type, skip_last_columns in datasets:
        try:
            print(f"\n{dataset_name.capitalize()}")
            modelPath = os.path.join(path, "data", "models", model_type)
            dataPath = os.path.join(path, "data", "trainingData", dataset_name)
            file = f"{dataset_name}-gamedata-NL-{current_year}.csv"

            # Predict for complete data
            predict(dataset_name, dataPath, modelPath, file, skipLastColumns=skip_last_columns)

            # Predict for current year
            predict(f"{dataset_name}_currentYear", dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=1)

            # Predict for current year + last two years
            predict(f"{dataset_name}_threeYears", dataPath, modelPath, file, skipLastColumns=skip_last_columns, years_back=3)

        except Exception as e:
            print(f"Failed to predict {dataset_name.capitalize()}: {e}")

    try:
        helpers.generatePredictionTextFile(os.path.join(path, "data", "database"))
    except Exception as e:
        print("Failed to generate txt file:", e)

    try:
        helpers.git_push()
    except Exception as e:
        print("Failed to push latest predictions:", e)
    
    

    
