import os, argparse, json
import numpy as np

from art import text2art
from datetime import datetime

from src.TCN import TCNModel
from src.LSTM import LSTMModel
from src.Command import Command
from src.Helpers import Helpers

tcn = TCNModel()
lstm = LSTMModel()
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
                "labels": []              # Needed for decoding the raw predictions
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
                    listOfDecodedPredictions = []

                    # Check on prediction with nth highest probability
                    for i in range(10):
                        prediction_nth_indices = helpers.decode_predictions(current_json_object["currentPredictionRaw"], previous_json_object["labels"], i)
                        #print("Prediction with ", i+1 ,"highest probs: ", prediction_nth_indices)
                        matching_numbers = helpers.find_matching_numbers(current_json_object["realResult"], prediction_nth_indices)
                        #print("Matching Numbers with ", i+1 ,"highest probs: ", matching_numbers)
                        listOfMatching.append({
                            "index": i,
                            "matchingSequence": prediction_nth_indices,
                            "matchingNumbers": matching_numbers
                        })

                        listOfDecodedPredictions.append(prediction_nth_indices)
                    

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

                    decodedRawPredictions = []
                    # Decode prediction with nth highest probability
                    for i in range(10):
                        prediction_nth_indices = helpers.decode_predictions(current_json_object["newPredictionRaw"], current_json_object["labels"], i)
                        decodedRawPredictions.append(prediction_nth_indices)

                    current_json_object["newPrediction"] = decodedRawPredictions

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

                # Search for existing history
                for index, historyEntry in enumerate(historyData):
                    entryDate = historyEntry[0]
                    entryResult = historyEntry[1]
                    jsonFileName = f"{entryDate.year}-{entryDate.month}-{entryDate.day}.json"
                    print(jsonFileName, ":", entryResult)
                    jsonFilePath = os.path.join(path, "data", "database", name, jsonFileName)
                    print("Does file exist: ", os.path.exists(jsonFilePath))
                    if os.path.exists(jsonFilePath):
                        dateOffset = index
                
                # Remove all elements starting from dateOffset index
                historyData = historyData[:dateOffset]  # Keep elements before dateOffset because older elements comes after the dateOffset index
                
                #print("History to rebuild: ", historyData)

                previousJsonFilePath = ""

                # Now lets iterate in reversed order to start with the older entries
                for historyIndex, historyEntry in enumerate(reversed(historyData)):
                    historyDate = historyEntry[0]
                    historyResult = historyEntry[1]
                    jsonFileName = f"{historyDate.year}-{historyDate.month}-{historyDate.day}.json"
                    print(jsonFileName, ":", historyResult)
                    jsonFilePath = os.path.join(path, "data", "database", name, jsonFileName)

                    if historyIndex == 0:
                        print("oldest entry: ", historyEntry)
                        
                        current_json_object = {
                            "currentPredictionRaw": [],
                            "currentPrediction": [],
                            "realResult": [],
                            "newPrediction": [],    # Decoded prediction according to formula in decode_prediction
                            "newPredictionRaw": [], # Raw prediction that contains the statistical data
                            "matchingNumbers": [],
                            "labels": []
                        }

                        # Train and do a new prediction
                        modelToUse.setDataPath(dataPath)
                        
                        modelToUse.setModelPath(modelPath)
                        modelToUse.setBatchSize(16)
                        modelToUse.setEpochs(1000)
                        latest_raw_predictions, unique_labels = modelToUse.run(name, skipLastColumns, skipRows=len(historyData)-historyIndex , years_back=years_back)

                        predictedSequence = latest_raw_predictions.tolist()
                        unique_labels = unique_labels.tolist()

                        listOfDecodedPredictions = []
                        for i in range(10):
                            prediction_nth_indices = helpers.decode_predictions(predictedSequence, unique_labels, i)
                            listOfDecodedPredictions.append(prediction_nth_indices)
                
                        # Save the current prediction as newPrediction
                        current_json_object["newPredictionRaw"] = predictedSequence
                        current_json_object["newPrediction"] = listOfDecodedPredictions
                        current_json_object["labels"] = unique_labels

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
                            "labels": []
                        }
                        
                        #print(previous_json_object)
                        #print(type(previous_json_object))

                        # The current prediction is the new prediction from the previous one
                        current_json_object["currentPredictionRaw"] = previous_json_object["newPredictionRaw"]
                        current_json_object["currentPrediction"] = previous_json_object["newPrediction"]

                        #print(current_json_object["currentPredictionRaw"])

                        listOfDecodedPredictions = []
                        listOfMatchings = []

                        # Check on prediction with nth highest probability
                        for i in range(10):
                            prediction_nth_indices = helpers.decode_predictions(current_json_object["currentPredictionRaw"], previous_json_object["labels"], i)
                            #print("Prediction with ", i+1 ,"highest probs: ", prediction_nth_indices)
                            matching_numbers = helpers.find_matching_numbers(current_json_object["realResult"], prediction_nth_indices)
                            #print("Matching Numbers with ", i+1 ,"highest probs: ", matching_numbers)
                            listOfMatchings.append({
                                "index": i,
                                "matchingSequence": prediction_nth_indices,
                                "matchingNumbers": matching_numbers
                            })

                            listOfDecodedPredictions.append(prediction_nth_indices)
                        

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
                        
                        decodedRawPredictions = []
                        # Decode prediction with nth highest probability
                        for i in range(10):
                            prediction_nth_indices = helpers.decode_predictions(current_json_object["newPredictionRaw"], current_json_object["labels"], i)
                            decodedRawPredictions.append(prediction_nth_indices)

                        current_json_object["newPrediction"] = decodedRawPredictions

                        with open(jsonFilePath, "w+") as outfile:
                            json.dump(current_json_object, outfile)

                    previousJsonFilePath = jsonFilePath

                #return predictedSequence
        else:
            print("Prediction already made")
    else:
        print("Did not found entries")



if __name__ == "__main__":

    try:
        helpers.git_pull()
    except Exception as e:
        print("Failed to get latest changes")

    parser = argparse.ArgumentParser(
                    prog='LSTM Sequence Predictor',
                    description='Tries to predict a sequence of numbers',
                    epilog='Check it uit')  
    parser.add_argument('-d', '--data', default="euromillions")
    
    args = parser.parse_args()
    print(args.data)

    print_intro()

    # Get the current date and time
    current_datetime = datetime.now()

    # Access the year attribute to get the current year
    current_year = current_datetime.year
    #current_year = 2024


    # Print the result
    print("Current Year:", current_year)

    path = os.getcwd()
    
    
    try:
        #####################
        #   Euromillions    #
        #####################
        print("Euromillions")
        modelPath = os.path.join(path, "data", "models", "tcn_model")
        dataPath = os.path.join(path, "data", "trainingData", "euromillions")
        file = "euromillions-gamedata-NL-{0}.csv".format(current_year)

        name = 'euromillions'
        # Do also a training on the complete data
        predict(name, dataPath, modelPath, file)

        print("Euromillions current year")
        name = 'euromillions_currentYear'
        predict(name, dataPath, modelPath, file, years_back=1)


        print("Euromillions current + last two years")
        name = 'euromillions_threeYears'
        predict(name, dataPath, modelPath, file, years_back=3)

    except Exception as e:
        print("Failed to predict Euromillions", e)

    try:
        #####################
        #       Lotto       #
        #####################
        print("Lotto")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        dataPath = os.path.join(path, "data", "trainingData", "lotto")
        file = "lotto-gamedata-NL-{0}.csv".format(current_year)


        name = 'lotto'
        # With skipLastColumns we only going to use 6 numbers because number 7 is the bonus number
        # do a training on the complete data
        predict(name, dataPath, modelPath, file, skipLastColumns=1)


        print("Lotto current year")
        name = 'lotto_currentYear'
        # With skipLastColumns we only going to use 6 numbers because number 7 is the bonus number
        predict(name, dataPath, modelPath, file, skipLastColumns=1, years_back=1)
        


        print("Lotto current year + last two years")
        name = 'lotto_threeYears'
        # With skipLastColumns we only going to use 6 numbers because number 7 is the bonus number
        predict(name, dataPath, modelPath, file, skipLastColumns=1, years_back=3)

    except Exception as e:
        print("Failed to predict Lotto", e)

    try:
        #####################
        #     euroDreams    #
        #####################
        print("euroDreams")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        dataPath = os.path.join(path, "data", "trainingData", 'eurodreams')
        file = "eurodreams-gamedata-NL-{0}.csv".format(current_year)

        name = 'eurodreams'
        predict(name, dataPath, modelPath, file, skipLastColumns=0)

        print("euroDreams Three Years")
        name = 'eurodreams_threeYears'
        predict(name, dataPath, modelPath, file, skipLastColumns=0, years_back=3)

        print("euroDreams Current Year")
        name = 'eurodreams_currentYear'

        predict(name, dataPath, modelPath, file, skipLastColumns=0, years_back=1)
    except Exception as e:
        print("Failed to predict euroDreams", e)
    
    
    """
    try:
        #####################
        #     joker plus    #
        #####################
        print("Joker Plus")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'jokerplus'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "jokerplus-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=1)
    except Exception as e:
        print("Failed to predict Joker plus", e)

    """

    
    try:
        #####################
        #        keno       #
        #####################
        print("Keno")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        dataPath = os.path.join(path, "data", "trainingData", "keno")
        file = "keno-gamedata-NL-{0}.csv".format(current_year)

        name = 'keno'
        predict(name, dataPath, modelPath, file, skipLastColumns=0)

        print("Keno Three Years")
        name = 'keno_threeYears'
        predict(name, dataPath, modelPath, file, skipLastColumns=0, years_back=3)

        print("Keno Current Year")
        name = 'keno_currentYear'

        predict(name, dataPath, modelPath, file, skipLastColumns=0, years_back=1)
    except Exception as e:
        print("Failed to predict Keno", e)
    
    
    try:
        #####################
        #        Pick3      #
        #####################
        print("Pick3")
        modelPath = os.path.join(path, "data", "models", "tcn_model")
        dataPath = os.path.join(path, "data", "trainingData", "pick3")
        file = "pick3-gamedata-NL-{0}.csv".format(current_year)

        name = 'pick3'
        predict(name, dataPath, modelPath, file, skipLastColumns=0)

        print("Pick3 Three Years")
        name = 'pick3_threeYears'
        predict(name, dataPath, modelPath, file, skipLastColumns=0, years_back=3)

        name = 'pick3_currentYear'
        predict(name, dataPath, modelPath, file, skipLastColumns=0, years_back=1)
    except Exception as e:
        print("Failed to predict Pick3", e)

    
    try:
        #####################
        #    Viking Lotto   #
        #####################
        print("Viking Lotto")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        dataPath = os.path.join(path, "data", "trainingData", "vikinglotto")
        file = "vikinglotto-gamedata-NL-{0}.csv".format(current_year)
        
        name = 'vikinglotto'
        predict(name, dataPath, modelPath, file, skipLastColumns=0)

        print("Viking Lotto Three Years")
        name = 'vikinglotto_threeYears'
        predict(name, dataPath, modelPath, file, skipLastColumns=0, years_back=3)

        print("Viking Lotto Current Year")
        name = 'vikinglotto_currentYear'
        predict(name, dataPath, modelPath, file, skipLastColumns=0, years_back=1)
    except Exception as e:
        print("Failed to predict Viking Lotto", e)

    try:
        helpers.generatePredictionTextFile(os.path.join(path, "data", "database"))
    except Exception as e:
        print("Failed to generate txt file", e)
    
    try:
        helpers.git_push()
    except Exception as e:
        print("Failed to push latest predictions")
    
    
    
    

    
