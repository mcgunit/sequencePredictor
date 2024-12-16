import os, json
import numpy as np

from dateutil.parser import parse
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder


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
            # Sort data by date (the first element of the tuple)
            data.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order
            previous_entry = data[1] # needed to find the previous prediction to compare with the latest entry
            latest_entry = data[0]  # Get the most recent entry
            return (latest_entry, previous_entry)  # Return the most recent entry
        else:
            print("No data found.")
            return None  # Return None if no data was found
        

    def find_matching_numbers(self, sequence, sequence_list):
        # Calculate similarity scores and matching numbers
        results = [
            (len(set(sequence).intersection(seq)), set(sequence).intersection(seq))
            for seq in sequence_list
        ]

        # Find the best match
        best_match_index = max(range(len(results)), key=lambda i: results[i][0])
        best_match_sequence = sequence_list[best_match_index]
        matching_numbers = results[best_match_index][1]


        return (best_match_index, best_match_sequence, sorted(matching_numbers))
    
    def decode_predictions(self, raw_predictions, top_k=7):
        # Initialize an empty list for the final predictions
        final_predictions = []

        # Process each row of raw predictions
        for row in raw_predictions:
            # Get indices of the top `top_k` probabilities
            top_indices = np.argsort(-row)[:top_k]

            # Convert indices to numbers (1-based indexing)
            top_numbers = (top_indices + 1).tolist()  # Convert to list for consistent output

            final_predictions.append(top_numbers)  # Ensure exactly top_k numbers are added

        return np.array(final_predictions, dtype=int)
    
    # Function to predict numbers using the trained model
    def predict_numbers(self, model, input_data, num_choices=7, value_range=(1, 50)):
        
        # Get the model's raw predictions (probabilities)
        raw_predictions = model.predict(input_data)
        print("Raw Predictions: ", raw_predictions)

        # Decode raw predictions into unique numbers
        predicted_numbers = self.decode_predictions(raw_predictions)
        print("Predicted Numbers: ", predicted_numbers)
        return predicted_numbers

    # Function to print the predicted numbers
    def print_predicted_numbers(self, predicted_numbers):
        # Print a separator line and "Predicted Numbers:"
        
        print("============================================================")
        # Print number of rows
        for i in range(10):
            print("Predicted Numbers {}:".format(i))
            print(', '.join(map(str, predicted_numbers[i])))
        print("============================================================")
        

    
    

    def load_data(self, dataPath, skipLastColumns=0, nth_row=5):
        # Initialize an empty list to hold the data
        data = []

        for csvFile in os.listdir(dataPath):
            if csvFile.endswith(".csv"):
                try:
                    # Construct full file path
                    file_path = os.path.join(dataPath, csvFile)

                    # Load data from the file
                    csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1)

                    # Skip last number of columns by slicing (if required)
                    if skipLastColumns > 0:
                        csvData = csvData[:, :-skipLastColumns]

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

                        data.append((date, *numbers))  # Store as a tuple (date, number1, number2, ...)

                except Exception as e:
                    print(f"Error processing file {csvFile}: {e}")

        # Sort the data by date
        data.sort(key=lambda x: x[0])  # Sort by the date (first element of the tuple)

        # Convert the sorted data into a NumPy array
        sorted_data = np.array(data)

        # If you want to separate the date and numbers into different arrays
        dates = sorted_data[:, 0]  # Dates
        numbers = sorted_data[:, 1:].astype(int)  # Numbers as integers (multi-label data)

        # Replace all -1 values with 0 (or you can remove them if it's not needed)
        numbers[numbers == -1] = 0

        # Unique labels for one-hot encoding
        unique_labels = np.arange(1, 51)

        # One-hot encode all numbers with a fixed range (1–50)
        encoder = OneHotEncoder(categories=[unique_labels], sparse_output=False)

        # Reshape numbers array to a single column for encoding, then reshape back
        one_hot_labels = encoder.fit_transform(numbers.flatten().reshape(-1, 1))

        # Reshape back into the original format (rows x 7 x 50)
        one_hot_labels = one_hot_labels.reshape(numbers.shape[0], numbers.shape[1], -1)

        # Combine the one-hot encoded vectors for all 7 numbers in each row
        one_hot_labels = one_hot_labels.sum(axis=1)  # Sum across the 7 numbers per row

        # Prepare training and validation sets
        train_indices = [i for i in range(len(numbers)) if i % nth_row != 0]  # Indices for training data
        val_indices = [i for i in range(len(numbers)) if i % nth_row == 0]    # Indices for validation data

        train_data = numbers[train_indices]
        val_data = numbers[val_indices]

        print("length of train data: ", len(train_data))
        print("length of val_data: ", len(val_data))

        train_labels = one_hot_labels[train_indices]
        val_labels = one_hot_labels[val_indices]

        # Get the maximum value in the data (for scaling purposes, if needed)
        max_value = np.max(numbers)

        # Number of classes (the unique labels we have after one-hot encoding)
        num_classes = one_hot_labels.shape[1]  # Should now be 50

        return train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes


    
    def generatePredictionTextFile(self, path):
        print("Generating text file with latest predictions")
        latestPredictionFile = os.path.join(os.getcwd(), "latestPrediction.txt")

        if os.path.exists(latestPredictionFile):
            os.remove(latestPredictionFile)

        for folder in os.listdir(path):
            print(folder)
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

            
            