import os, argparse

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

    # First get latest data
    data = 'euromillions'
    path = os.getcwd()
    dataPath = os.path.join(path, "data", data)
    file = "euromillions-gamedata-NL-{0}.csv".format(current_year)
    kwargs_wget = {
        "folder": dataPath,
        "file": file
    }
    command.run("wget -P {folder} https://prdlnboppreportsst.blob.core.windows.net/legal-reports/{file}".format(**kwargs_wget), verbose=True)

    # Get the latest result out of the latest data so we can use it to check the previous prediction
    latestResult = helpers.getLatestPrediction(os.path.join(dataPath, file))

    print("Latest result: ", latestResult)
    
    # Train and predict
    #lstm.setDataPath(dataPath)

    