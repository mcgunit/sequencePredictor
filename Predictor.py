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



def predict(dataPath, modelPath, file, data, skipLastColumns=0, doTraining=True, maxRows=0):

    modelToUse = tcn
    if "lotto" in data or "eurodreams" in data or "jokerplus" in data or "keno" in data or "pick3" in data or "vikinglotto" in data:
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

            doNewPrediction = True

            # First find the json file containing the prediction for this result
            if previousEntry is not None:
                previousDate, previousResult = previousEntry
                jsonPreviousFileName = f"{previousDate.year}-{previousDate.month}-{previousDate.day}.json"
                print(jsonPreviousFileName, ":", latestResult)
                jsonPreviousFilePath = os.path.join(path, "data", "database", data, jsonPreviousFileName)
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
                    current_json_object["currentPrediction"] = previous_json_object["newPrediction"]

                    # Check the matching numbers
                    best_match_index, best_match_sequence, matching_numbers_array = helpers.find_matching_numbers(current_json_object["realResult"], current_json_object["currentPrediction"])
                    current_json_object["matchingNumbers"] = {
                        "bestMatchIndex": best_match_index,
                        "bestMatchSequence": best_match_sequence,
                        "matchingNumbers": matching_numbers_array
                    }

                    print("matching_numbers: ", matching_numbers_array)

                    # Train and do a new prediction
                    if doTraining:
                        modelToUse.setModelPath(modelPath)
                        modelToUse.setBatchSize(16)
                        modelToUse.setEpochs(1000)
                        predictedNumbers = modelToUse.run(data, skipLastColumns)
                    else:
                        model = os.path.join(modelPath, "model_euromillions.keras")
                        if "lotto" in data and not "vikinglotto" in data:
                            model = os.path.join(modelPath, "model_lotto.keras")
                        if "eurodreams" in data:
                            model = os.path.join(modelPath, "model_eurodreams.keras")
                        if "jokerplus" in data:
                            model = os.path.join(modelPath, "model_jokerplus.keras")
                        if "keno" in data:
                            model = os.path.join(modelPath, "model_keno.keras")
                        if "pick3" in data:
                            model = os.path.join(modelPath, "model_pick3.keras")
                        if "vikinglotto" in data:
                            model = os.path.join(modelPath, "model_vikinglotto.keras")
                        predictedNumbers = modelToUse.doPrediction(model, skipLastColumns, maxRows=maxRows)

                    predictedSequence = predictedNumbers.tolist()

            
                    # Save the current prediction as newPrediction
                    current_json_object["newPrediction"] = predictedSequence

                    with open(jsonFilePath, "w+") as outfile:
                        json.dump(current_json_object, outfile)

                    return predictedSequence
                
            if doNewPrediction:
                print("No previous prediction file found, Cannot compare. Creating from scratch")
                
                current_json_object = {
                    "currentPrediction": [],
                    "realResult": [],
                    "newPrediction": [],
                    "matchingNumbers": []
                }

                # Train and do a new prediction
                modelToUse.setDataPath(dataPath)
                
                if doTraining:
                    modelToUse.setModelPath(modelPath)
                    modelToUse.setBatchSize(16)
                    modelToUse.setEpochs(1000)
                    predictedNumbers = modelToUse.run(data, skipLastColumns)
                else:
                    model = os.path.join(modelPath, "model_euromillions.keras")
                    if "lotto" in data and not "vikinglotto" in data:
                        model = os.path.join(modelPath, "model_lotto.keras")
                    if "eurodreams" in data:
                        model = os.path.join(modelPath, "model_eurodreams.keras")
                    if "jokerplus" in data:
                        model = os.path.join(modelPath, "model_jokerplus.keras")
                    if "keno" in data:
                        model = os.path.join(modelPath, "model_keno.keras")
                    if "pick3" in data:
                        model = os.path.join(modelPath, "model_pick3.keras")
                    if "vikinglotto" in data:
                        model = os.path.join(modelPath, "model_vikinglotto.keras")
                    predictedNumbers = modelToUse.doPrediction(model, skipLastColumns, maxRows=maxRows)

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
        # First get latest data
        data = 'euromillions'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "euromillions-gamedata-NL-{0}.csv".format(current_year)
        
        # Do also a training on the complete data
        predict(dataPath, modelPath, file, data)

        #################################
        #   Euromillions_currentYear    #
        #################################
        print("Euromillions current year")
        # First get latest data
        data = 'euromillions_currentYear'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "euromillions-gamedata-NL-{0}.csv".format(current_year)
        
        predict(dataPath, modelPath, file, data)

        ####################################
        #   Euromillions_threeYears         #
        ####################################
        print("Euromillions current + last two years")
        # First get latest data
        data = 'euromillions_threeYears'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "euromillions-gamedata-NL-{0}.csv".format(current_year)
        
        predict(dataPath, modelPath, file, data)

    except Exception as e:
        print("Failed to predict Euromillions", e)

    try:
        #####################
        #       Lotto       #
        #####################
        print("Lotto")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'lotto'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "lotto-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        # With skipLastColumns we only going to use 6 numbers because number 7 is the bonus number
        # do a training on the complete data
        predict(dataPath, modelPath, file, data, skipLastColumns=1)

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

        # With skipLastColumns we only going to use 6 numbers because number 7 is the bonus number
        predict(dataPath, modelPath, file, data, skipLastColumns=1)
        

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

        # With skipLastColumns we only going to use 6 numbers because number 7 is the bonus number
        predict(dataPath, modelPath, file, data, skipLastColumns=1)

    except Exception as e:
        print("Failed to predict Lotto", e)

    try:
        #####################
        #     euroDreams    #
        #####################
        print("euroDreams")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'eurodreams'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "eurodreams-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)

        #####################
        #     euroDreams    #
        #####################
        print("euroDreams Three Years")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'eurodreams_threeYears'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "eurodreams-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)

        #####################
        #     euroDreams    #
        #####################
        print("euroDreams Current Year")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'eurodreams_currentYear'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "eurodreams-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)
    except Exception as e:
        print("Failed to predict euroDreams", e)

    '''
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

    '''
    try:
        #####################
        #        keno       #
        #####################
        print("Keno")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'keno'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "keno-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)

        #####################
        #        keno       #
        #####################
        print("Keno Three Years")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'keno_threeYears'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "keno-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)

        #####################
        #        keno       #
        #####################
        print("Keno Current Year")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'keno_currentYear'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "keno-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)
    except Exception as e:
        print("Failed to predict Keno", e)


    try:
        #####################
        #        Pick3      #
        #####################
        print("Pick3")
        modelPath = os.path.join(path, "data", "models", "tcn_model")
        # First get latest data
        data = 'pick3'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "pick3-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)

        #####################
        #        Pick3      #
        #####################
        print("Pick3 Three Years")
        modelPath = os.path.join(path, "data", "models", "tcn_model")
        # First get latest data
        data = 'pick3_threeYears'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "pick3-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)

        #####################
        #        Pick3      #
        #####################
        print("Pick3 Current Year")
        modelPath = os.path.join(path, "data", "models", "tcn_model")
        # First get latest data
        data = 'pick3_currentYear'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "pick3-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)
    except Exception as e:
        print("Failed to predict Pick3", e)


    try:
        #####################
        #    Viking Lotto   #
        #####################
        print("Viking Lotto")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'vikinglotto'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "vikinglotto-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)

        #####################
        #    Viking Lotto   #
        #####################
        print("Viking Lotto Three Years")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'vikinglotto_threeYears'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "vikinglotto-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)

        #####################
        #    Viking Lotto   #
        #####################
        print("Viking Lotto Current Year")
        modelPath = os.path.join(path, "data", "models", "lstm_model")
        # First get latest data
        data = 'vikinglotto_currentYear'
        dataPath = os.path.join(path, "data", "trainingData", data)
        file = "vikinglotto-gamedata-NL-{0}.csv".format(current_year)
        kwargs_wget = {
            "folder": dataPath,
            "file": file
        }

        predict(dataPath, modelPath, file, data, skipLastColumns=0)
    except Exception as e:
        print("Failed to predict Viking Lotto", e)

    try:
        helpers.generatePredictionTextFile(os.path.join(path, "data", "database"))
    except Exception as e:
        print("Failed to generate txt file", e)
    
    
    

    
