import os
import numpy as np

from dateutil.parser import parse


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

        return (best_match_index, best_match_sequence, matching_numbers)
    
    # Function to predict numbers using the trained model
    def predict_numbers(self, model, val_data, num_features):
        # Predict on the validation data using the model
        predictions = model.predict(val_data)
        # Get the indices of the top 'num_features' predictions for each sample in validation data
        indices = np.argsort(predictions, axis=1)[:, -num_features:]
        # Get the predicted numbers using these indices from validation data
        predicted_numbers = np.take_along_axis(val_data, indices, axis=1)
        # Return the first 10 predictions
        return predicted_numbers[:10]

    # Function to print the predicted numbers
    def print_predicted_numbers(self, predicted_numbers):
        # Print a separator line and "Predicted Numbers:"
        print("============================================================")
        # Print number of rows
        for i in range(len(predicted_numbers)):
            print("Predicted Numbers {}:".format(i))
            print(', '.join(map(str, predicted_numbers[i])))
        print("============================================================")

    
    # Function to load data from a file and preprocess it
    def load_data(self, dataPath):
        # Initialize an empty list to hold the data
        data = []

        for csvFile in os.listdir(dataPath):
            if csvFile.endswith(".csv"):
                #print(f"Processing file: {csvFile}")
                try:
                    # Construct full file path
                    file_path = os.path.join(dataPath, csvFile)
                    
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
                        
                        data.append((date, *numbers))  # Store as a tuple (date, number1, number2, ...)

                except Exception as e:
                    print(f"Error processing file {csvFile}: {e}")

        # Sort the data by date
        data.sort(key=lambda x: x[0])  # Sort by the date (first element of the tuple)

        # Convert the sorted data into a NumPy array
        sorted_data = np.array(data)

        # If you want to separate the date and numbers into different arrays
        dates = sorted_data[:, 0]  # Dates
        numbers = sorted_data[:, 1:].astype(int)  # Numbers as integers

        # Replace all -1 values with 0
        numbers[numbers == -1] = 0
        # Split data into training and validation sets
        train_data = numbers[:int(0.8*len(numbers))]
        val_data = numbers[int(0.8*len(numbers)):]
        # Get the maximum value in the data
        max_value = np.max(numbers)
        return train_data, val_data, max_value