import os, argparse, json
import numpy as np

from art import text2art
from datetime import datetime

from src.LSTM import LSTM
from src.GRU import GRU
from src.Command import Command
from src.Helpers import Helpers

lstm = LSTM()
gru = GRU()
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



def predict(dataPath, modelPath, file):

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
    if latestEntry is not None and previousEntry is not None:
        latestDate, latestResult = latestEntry

        
        jsonFileName = f"{latestDate.year}-{latestDate.month}-{latestDate.day}.json"
        #print(jsonFileName, ":", latestResult)
        jsonFilePath = os.path.join(path, "data", "database", data, jsonFileName)

        # Check if folder exists
        if not os.path.exists(os.path.join(path, "data", "database", data)):
            os.mkdir(os.path.join(path, "data", "database", data))

        # Compare the latest result with the previous new prediction
        if not os.path.exists(jsonFilePath):
            print("New result detected. Lets compare with a prediction from previous entry")

            current_json_object = {
                "currentPrediction": [],
                "realResult": latestResult,
                "newPrediction": [],
                "matchingNumbers": {}
            }

            # First find the json file containing the prediction for this result
            previousDate, previousResult = previousEntry
            jsonPreviousFileName = f"{previousDate.year}-{previousDate.month}-{previousDate.day}.json"
            print(jsonPreviousFileName, ":", latestResult)
            jsonPreviousFilePath = os.path.join(path, "data", "database", data, jsonPreviousFileName)
            print(jsonPreviousFilePath)
            if os.path.exists(jsonPreviousFilePath):
                print("previous json file found lets compare")
                # Opening JSON file
                with open(jsonPreviousFilePath, 'r') as openfile:
                
                    # Reading from json file
                    previous_json_object = json.load(openfile)
                
                #print(previous_json_object)
                #print(type(previous_json_object))

                # The current prediction is the new prediction from the previous one
                current_json_object["currentPrediction"] = previous_json_object["newPrediction"]

                # Check the matching numbers
                best_match_index, best_match_sequence, matching_numbers = helpers.find_matching_numbers(current_json_object["realResult"], current_json_object["currentPrediction"])
                current_json_object["matchingNumbers"] = {
                    "bestMatchIndex": best_match_index,
                    "bestMatchSequence": best_match_sequence,
                    "matchingNumbers": matching_numbers
                }

                # Train and do a new prediction
                lstm.setDataPath(dataPath)
                lstm.setModelPath(modelPath)
                lstm.setBatchSize(4)
                lstm.setEpochs(1)
                predictedNumbers = lstm.run(data)
                predictedSequence = predictedNumbers.tolist()

        
                # Save the current prediction as newPrediction
                current_json_object["newPrediction"] = predictedSequence

                with open(jsonFilePath, "w+") as outfile:
                    json.dump(current_json_object, outfile)

                return predictedSequence
            else:
                print("No previous prediction file found, Cannot compare. Creating from scratch")
                current_json_object = {
                    "currentPrediction": [],
                    "realResult": [],
                    "newPrediction": [],
                    "matchingNumbers": []
                }

                # Train and do a new prediction
                lstm.setDataPath(dataPath)
                lstm.setModelPath(modelPath)
                lstm.setBatchSize(4)
                lstm.setEpochs(1)
                predictedNumbers = lstm.run(data)
                predictedSequence = predictedNumbers.tolist()

        
                # Save the current prediction as newPrediction
                current_json_object["newPrediction"] = predictedSequence

                with open(jsonFilePath, "w+") as outfile:
                    json.dump(current_json_object, outfile)

                return predictedSequence
        else:
            print("Prediction already made")
    else:
        print("Did not found entries")



if __name__ == "__main__":

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

    # Print the result
    print("Current Year:", current_year)

    path = os.getcwd()
    modelPath = os.path.join(path, "data", "models", "lstm_model")

    #####################
    #   Euromillions    #
    #####################
    print("Euromillions")
    # First get latest data
    data = 'euromillions'
    dataPath = os.path.join(path, "data", "trainingData", data)
    file = "euromillions-gamedata-NL-{0}.csv".format(current_year)
    
    predict(dataPath, modelPath, file)

    #################################
    #   Euromillions_currentYear    #
    #################################
    print("Euromillions current year")
    # First get latest data
    data = 'euromillions_currentYear'
    dataPath = os.path.join(path, "data", "trainingData", data)
    file = "euromillions-gamedata-NL-{0}.csv".format(current_year)
    
    predict(dataPath, modelPath, file)

    ####################################
    #   Euromillions_hreeYears         #
    ####################################
    print("Euromillions current + last two years")
    # First get latest data
    data = 'euromillions_threeYears'
    dataPath = os.path.join(path, "data", "trainingData", data)
    file = "euromillions-gamedata-NL-{0}.csv".format(current_year)
    
    predict(dataPath, modelPath, file)
    
    #####################
    #       Lotto       #
    #####################
    print("Lotto")
    # First get latest data
    data = 'lotto'
    dataPath = os.path.join(path, "data", "trainingData", data)
    file = "lotto-gamedata-NL-{0}.csv".format(current_year)
    kwargs_wget = {
        "folder": dataPath,
        "file": file
    }

    predict(dataPath, modelPath, file)

    ##############################
    #       Lotto currentYear    #
    ##############################
    print("Lotto current year")
    # First get latest data
    data = 'lotto_currentYear'
    dataPath = os.path.join(path, "data", "trainingData", data)
    file = "lotto-gamedata-NL-{0}.csv".format(current_year)
    kwargs_wget = {
        "folder": dataPath,
        "file": file
    }

    predict(dataPath, modelPath, file)
    

    ################################
    #       Lotto threeYears       #
    ################################
    print("Lotto current year + last two years")
    # First get latest data
    data = 'lotto_threeYears'
    dataPath = os.path.join(path, "data", "trainingData", data)
    file = "lotto-gamedata-NL-{0}.csv".format(current_year)
    kwargs_wget = {
        "folder": dataPath,
        "file": file
    }

    predict(dataPath, modelPath, file)
    
    
    

    