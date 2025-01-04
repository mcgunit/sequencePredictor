import os, json, collections
import numpy as np

from dateutil.parser import parse
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from collections import Counter


class Helpers():

    def getLatestPrediction(self, csvFile):
        # Initialize an empty list to hold the data
        data = []

        print(f"Getting latest prediction from file: {csvFile}")
        try:
            # Construct full file path
            file_path = csvFile
            
            # Load data from the file
            csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1)

            if not isinstance(csvData[0], (list, np.ndarray)):
                print("Need to reform loaded latest prediction data")
                csvData = [csvData.tolist()]

            print("CSV DATA: ", csvData)
            
            # Append each entry to the data list
            for entry in csvData:
                # Attempt to parse the date
                date_str = entry[0]
                try:
                    # Use dateutil.parser to parse the date
                    date = parse(date_str)
                except Exception as e:
                    print(f"Date parsing error for entry '{date_str}': {e}")
                    continue  # Skip this entry if date parsing fails
                
                # Convert the rest to integers
                try:
                    numbers = list(map(int, entry[1:]))  # Convert the rest to integers
                except ValueError as ve:
                    print(f"Number conversion error for entry '{entry[1:]}': {ve}")
                    continue  # Skip this entry if number conversion fails
                
                data.append((date, numbers))  # Store as a tuple (date, number1, number2, ...)

        except Exception as e:
            print(f"Error processing file {csvFile}: {e}")

        # If data is not empty, find the most recent entry
        if data:
            if len(data) == 1:
                # Sort data by date (the first element of the tuple)
                data.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order
                previous_entry = None
                latest_entry = data[0]  # Get the most recent entry
                return (latest_entry, previous_entry)  # Return the most recent entry
            else:
                # Sort data by date (the first element of the tuple)
                data.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order
                previous_entry = data[1] # needed to find the previous prediction to compare with the latest entry
                latest_entry = data[0]  # Get the most recent entry
                return (latest_entry, previous_entry)  # Return the most recent entry
        else:
            print("No data found.")
            return None  # Return None if no data was found
        

    def find_matching_numbers(self, sequence, sequence_list):
        # Convert the input sequence to a tuple for hashing
        sequence = tuple(sequence)

        # Calculate similarity scores and matching numbers
        results = [
            (len(set(sequence).intersection(set(tuple(np.array(seq).flatten())))), set(sequence).intersection(set(tuple(np.array(seq).flatten()))))
            for seq in sequence_list
        ]

        # Find the best match
        best_match_index = max(range(len(results)), key=lambda i: results[i][0])
        best_match_sequence = sequence_list[best_match_index]
        matching_numbers = results[best_match_index][1]
        matching_numbers_array = [int(x) for x in matching_numbers]

        return (best_match_index, best_match_sequence, matching_numbers_array)
    
    def decode_predictions(self, raw_predictions):
        # Get the indices of the maximum probabilities for each of the 7 positions
        predicted_indices = np.argmax(raw_predictions, axis=-1)  # Shape will be (1796, 7)
        
        # Convert indices to numbers (1-based indexing)
        predicted_numbers = predicted_indices + 1  # Add 1 to convert from 0-based to 1-based
        
        return predicted_numbers
    
    def predict_numbers(self, model, input_data):
        # Get raw predictions from the model
        raw_predictions = model.predict(input_data)
        #print("Raw Predictions Shape:", raw_predictions)
        #print("Raw Predictions (First Sample):", raw_predictions[0])
        #print("Raw Predictions (Second Sample):", raw_predictions[1])

        # Decode raw predictions into numbers
        predicted_numbers = self.decode_predictions(raw_predictions)
        #print("Decoded Predictions Shape:", predicted_numbers.shape)
        #print("Decoded Predictions Example (First Sample):", predicted_numbers[0])
        #print("Decoded Predictions Example (Second Sample):", predicted_numbers[1])

        return predicted_numbers

    # Function to print the predicted numbers
    def print_predicted_numbers(self, predicted_numbers):

        #print("Predicted Numbers Shape:", predicted_numbers.shape)
        #print("Predicted Numbers Type:", type(predicted_numbers))
        
        print("============================================================")
        for i in range(10):
            print(f"Predicted Numbers {i}: {', '.join(map(str, predicted_numbers[i]))}")
        print("============================================================")
        
        
    def load_data(self, dataPath, skipLastColumns=0, nth_row=5, maxRows=0):
        # Initialize an empty list to hold the data
        data = []

        for csvFile in os.listdir(dataPath):
            if csvFile.endswith(".csv"):
                try:
                    # Construct full file path
                    file_path = os.path.join(dataPath, csvFile)

                    # Load data from the file
                    if maxRows > 0:
                        csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1, max_rows=maxRows)
                    else:
                        csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1)

                
                    if not isinstance(csvData[0], (list, np.ndarray)):
                        print("Need to reform loaded csv data")
                        csvData = [csvData.tolist()]
                        
                    
                    # Skip last number of columns by slicing (if required)
                    if skipLastColumns > 0:
                        csvData = csvData[:, :-skipLastColumns]

                    #print("csv data: ", csvData)

                    # Append each entry to the data list
                    for entry in csvData:
                        # Attempt to parse the date
                        date_str = entry[0]
                        #print("Date: ", date_str)
                        try:
                            # Use dateutil.parser to parse the date
                            date = parse(date_str)
                        except Exception as e:
                            print(f"Date parsing error for entry '{date_str}': {e}")
                            continue  # Skip this entry if date parsing fails

                        # Convert the rest to integers
                        try:
                            numbers = list(map(int, entry[1:]))  # Convert the rest to integers
                        except ValueError as ve:
                            print(f"Number conversion error for entry '{entry[1:]}': {ve}")
                            continue  # Skip this entry if number conversion fails

                        data.append((date, *numbers))  # Store as a tuple (date, number1, number2, ...)

                except Exception as e:
                    print(f"Error processing file {csvFile}: {e}")

        # Sort the data by date
        data.sort(key=lambda x: x[0])  # Sort by the date (first element of the tuple)

        #print("Data: ", data)

        # Convert the sorted data into a NumPy array
        sorted_data = np.array(data)

        # If you want to separate the date and numbers into different arrays
        dates = sorted_data[:, 0]  # Dates
        numbers = sorted_data[:, 1:].astype(int)  # Numbers as integers (multi-label data)

        # Replace all -1 values with 0 (or you can remove them if it's not needed)
        numbers[numbers == -1] = 0

        # Unique labels for one-hot encoding
        # Euromillions are 50 numbers, Lotto are 45 numbers
        unique_labels = np.arange(1, 51)  # This should create an array [1, 2, ..., 50]
        if "lotto" in dataPath:
            unique_labels = np.arange(1, 46)  # This should create an array [1, 2, ..., 45]
        if "keno" in dataPath:
            unique_labels = np.arange(1, 81)  # This should create an array [1, 2, ..., 80]
        if "vikinglotto" in dataPath:
            unique_labels = np.arange(1, 49)  # This should create an array [1, 2, ..., 49]
        if "pick3" in dataPath:
            unique_labels = np.arange(0, 10)  # This should create an array [0, 2, ..., 9]
        if "jokerplus" in dataPath:
            unique_labels = np.arange(0, 10).tolist()
            unique_labels.append("Boogschutter")
            unique_labels.append("Kreeft")
            unique_labels.append("Weegschaal")
            unique_labels.append("Schorpioen")
            unique_labels.append("Stier")
            unique_labels.append("Leeuw")
            unique_labels.append("Maagd")
            unique_labels.append("Ram")
            unique_labels.append("Waterman")
            unique_labels.append("Vissen")
            unique_labels.append("Steenbok")
            unique_labels.append("Tweeling")




        encoder = OneHotEncoder(categories=[unique_labels], sparse_output=False)

        # Reshape numbers array to a single column for encoding, then reshape back
        one_hot_labels = encoder.fit_transform(numbers.flatten().reshape(-1, 1))

        one_hot_labels = one_hot_labels.reshape(numbers.shape[0], numbers.shape[1], -1)

        # Number of classes (the unique labels we have after one-hot encoding)
        num_classes = one_hot_labels.shape[2] 

        print("Num classes: ", num_classes)

        # Prepare training and validation sets
        train_indices = [i for i in range(len(numbers)) if i % nth_row != 0]  # Indices for training data
        val_indices = [i for i in range(len(numbers)) if i % nth_row == 0]    # Indices for validation data

        train_data = numbers[train_indices]
        val_data = numbers[val_indices]

        print("length of train data: ", len(train_data))
        print("length of val_data: ", len(val_data))

        train_labels = one_hot_labels[train_indices]
        val_labels = one_hot_labels[val_indices]

        print("Train data shape: ", train_data.shape)       # (samples, sequence_length) -> (n_samples, 7)
        print("Train labels shape: ", train_labels.shape)   # (samples, sequence

        #print("Train Labels Example:", train_labels[:5])  # Corresponding one-hot encoded labels

        # Get the maximum value in the data (for scaling purposes, if needed)
        max_value = np.max(numbers)

        return train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes





    
    def generatePredictionTextFile(self, path):
        print("Generating text file with latest predictions")
        latestPredictionFile = os.path.join(os.getcwd(), "latestPrediction.txt")

        if os.path.exists(latestPredictionFile):
            os.remove(latestPredictionFile)

        for folder in os.listdir(path):
            #print(folder)
            folder_path = os.path.join(path, folder)
            # Get all JSON files in the folder
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            
            # Parse dates from the filenames and find the latest
            latest_file = None
            latest_date = None
            
            for file in files:
                try:
                    # Extract the date from the filename and parse it
                    date_part = file.split('.')[0]  # Assuming format: YYYY-MM-DD.json
                    file_date = datetime.strptime(date_part, "%Y-%m-%d")
                    
                    # Update the latest file if this one is more recent
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = file
                except ValueError:
                    # Ignore files with invalid date formats
                    continue

            if latest_file is not None:
                # Opening JSON file
                with open(os.path.join(path, folder, latest_file), 'r') as openfile:
                
                    # Reading from json file
                    predictionFObject = json.load(openfile)

                with open(latestPredictionFile, "a+") as myfile:
                    myfile.write("{}:\n".format(folder))
                    myfile.write("{}\n".format(predictionFObject["newPrediction"]))
                    myfile.write("\n")
                    myfile.write("\n")

    def mostFrequentNumbers(self, array, numbers=7):
        # Flatten the 2D array into a 1D array
        flat_array = array.flatten()
        
        # Count the occurrences of each number
        counts = Counter(flat_array)
        
        # Get the 6 most common numbers
        most_common = counts.most_common(numbers)
        
        # Extract the numbers from the most_common list
        top_numbers = [num for num, _ in most_common]
        
        return np.array(top_numbers)

            
            