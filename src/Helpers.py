import os, json, subprocess, collections
import numpy as np
import asciichartpy

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from collections import Counter



class Helpers():

    def getLatestPrediction(self, csvFile, dateRange=None):
        """
            Get latest result from csv file.
            If dateRange is provided it will return a list containing multiple results
            If dateRange is None (default) it will return the current and previous result
        """
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

            #print("CSV DATA: ", csvData)
            
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
            if dateRange is None:
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
                # Calculate the cutoff date based on the dateRange
                cutoff_date = datetime.now() - relativedelta(months=dateRange)
                # Filter data to include only entries within the date range
                filtered_data = [entry for entry in data if entry[0] >= cutoff_date]
                return filtered_data
        else:
            print("No data found.")
            return None  # Return None if no data was found
        

    def find_matching_numbers(self, sequence, predictedSequence):
        #print("Sequence: ", sequence)
        #print("Predicted Sequence: ", predictedSequence)
        # Convert the input sequence to a set for efficient matching
        sequence_set = set(sequence)

        # Find the matching numbers between sequence and highest_indices
        matching_numbers = list(sequence_set.intersection(predictedSequence))

        # Convert to a list of integers
        matching_numbers = [int(x) for x in matching_numbers]

        return matching_numbers
    
    def decode_predictions(self, raw_predictions, labels, nHighestProb=0):
        """
        Decode the prediction based on probability and match with corresponding labels.

        Parameters
        ----------
        raw_predictions : list or np.ndarray
            List of raw predictions.
        labels : list or np.ndarray
            List of labels corresponding to the classes.
        nHighestProb : int, optional
            Rank of probability to consider. 0 means highest probability, 1 means second-highest, etc.

        Returns
        -------
        list
            Decoded predictions as per the provided labels.
        """

        # Ensure raw_predictions is a numpy array for easy processing
        raw_predictions = np.array(raw_predictions)
        labels = np.array(labels)

        # Get indices of the top nHighestProb probability
        highest_indices = np.argsort(raw_predictions, axis=1)[:, -(nHighestProb + 1)]

        # Map indices to labels
        decoded_predictions = labels[highest_indices]

        return decoded_predictions.tolist()
    
    def predict_numbers(self, model, input_data):
        # Get raw predictions from the model
        raw_predictions = model.predict(input_data)

        latest_raw_predictions = raw_predictions[::-1] # reverse

        latest_raw_predictions = latest_raw_predictions[0] # take the the latest
        #print("Latest raw prediction: ", latest_raw_predictions)

        """
            Structure of latest_raw_prediction is a list of each position a class and each class is a list of probabilities of what class it will be.
            Fow example when the model has to predict a set of 3 numbers and each number can be 1, 2 or 3 you can get something like this:
            [[0.1,0.2, 0.7], [0.8, 0.1, 0.1], [0.0, 0.9, 0.1]] --> This will be a prediction of a set of 3 numbers being [3, 1, 2].
            The index of the highest probability in the list determines the predicted number (index+1) 
        """

        return latest_raw_predictions

    # Function to print the predicted numbers
    def print_predicted_numbers(self, predicted_numbers):

        #print("Predicted Numbers Shape:", predicted_numbers.shape)
        #print("Predicted Numbers Type:", type(predicted_numbers))
        """
        print("============================================================")
        for i in range(len(predicted_numbers)):
            print(f"Predicted Numbers {i}: {', '.join(map(str, predicted_numbers[i]))}")
        print("============================================================")
        """

        for i, sublist in enumerate(predicted_numbers):
            chart = asciichartpy.plot(sublist, {'height': 10})
            print(f"Graph for Sublist {i+1}:\n{chart}\n")


        
        
    def load_data(self, dataPath, skipLastColumns=0, nth_row=5, maxRows=0, skipRows=0, years_back=None):
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

                    # Append each entry to the data list
                    for entry in csvData:
                        date_str = entry[0]
                        try:
                            date = parse(date_str)
                        except Exception as e:
                            print(f"Date parsing error for entry '{date_str}': {e}")
                            continue

                        try:
                            numbers = list(map(int, entry[1:]))
                        except ValueError as ve:
                            print(f"Number conversion error for entry '{entry[1:]}': {ve}")
                            continue

                        data.append((date, *numbers))

                except Exception as e:
                    print(f"Error processing file {csvFile}: {e}")

        # Sort the data by date
        data.sort(key=lambda x: x[0], reverse=False)  # Oldest to newest

        # Convert to NumPy array
        sorted_data = np.array(data)

        # Filter data for a relative range of years
        if years_back is not None:
            most_recent_date = sorted_data[-1, 0]  # Most recent date in the array
            cutoff_date = most_recent_date.replace(year=most_recent_date.year - years_back)
            filtered_data = [entry for entry in sorted_data if entry[0] >= cutoff_date]
            sorted_data = np.array(filtered_data)

        # Continue processing
        dates = sorted_data[:, 0]
        numbers = sorted_data[:, 1:].astype(int)

        # Replace all -1 values with 0
        numbers[numbers == -1] = 0

        # Remove the last n elements in case of history building
        if skipRows > 0:
            print("Skipping Rows: ", skipRows)
            print("Length of data before skipping rows: ", len(numbers))
            numbers = numbers[:-skipRows]
            
    

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

        #print("unique_labels: ", unique_labels)

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

        print("Length of data: ", len(numbers))

        return train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels



    def load_prediction_data(self, dataPath, skipLastColumns=0, maxRows=0):
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
        data.sort(key=lambda x: x[0], reverse=False)  # Sort by the date (first element of the tuple)

        #print("Data: ", data)

        # Convert the sorted data into a NumPy array
        sorted_data = np.array(data)

        print("Sorted data: ", sorted_data)

        # If you want to separate the date and numbers into different arrays
        dates = sorted_data[:, 0]  # Dates
        numbers = sorted_data[:, 1:].astype(int)  # Numbers as integers (multi-label data)

        # Replace all -1 values with 0 (or you can remove them if it's not needed)
        numbers[numbers == -1] = 0

        print("length of data: ", len(numbers))
        print("shape of data: ", numbers.shape)


        return numbers

    
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

    
    def git_push(self, commit_message="saving last predictions"):
        try:
            # Stage all changes
            subprocess.run(["git", "add", "-A"], check=True)

            # Commit changes
            subprocess.run(["git", "commit", "-m", f"{commit_message}"], check=True)

            # Push changes
            subprocess.run(["git", "push"], check=True)

            print("Changes have been pushed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing Git commands: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def git_pull(self):
        try:
            subprocess.run(["git", "fetch"], check=True)

            subprocess.run(["git", "pull"], check=True)

            print("Got latest changes")
        except Exception as e:
            print("Failed to get latest changes")

    def extractFeaturesFromJsonForRefinement(self, jsonFileOrDir):
        """
            Function to extract features for training a refinement model
            Can be a folder containing json files or a single file
        """
        X = []
        y = []
        if os.path.isdir(jsonFileOrDir):
            for file in sorted(os.listdir(jsonFileOrDir)):
                if file.endswith(".json"):
                    with open(os.path.join(jsonFileOrDir, file), "r") as f:
                        data = json.load(f)

                    if "currentPredictionRaw" not in data:
                        print(f"⚠ Warning: No 'currentPredictionRaw' in {file}")
                        continue
                    
                    raw_probs = np.array(data["currentPredictionRaw"])
                    
                    if raw_probs.size == 0:
                        print(f"⚠ Warning: Empty probability array in {file}")
                        continue

                    # Debug: Print shape of `raw_probs`
                    print(f"Processing {file}, shape: {raw_probs.shape}")

                    # Ensure raw_probs has expected shape (20, 80)
                    if raw_probs.shape[0] != 20 or raw_probs.shape[1] != 80:
                        print(f"⚠ Unexpected shape {raw_probs.shape} in {file}")
                        continue

                    # Feature Extraction
                    mean_probs = np.mean(raw_probs, axis=0)  # Average probability per number
                    max_probs = np.max(raw_probs, axis=0)    # Maximum probability per number
                    sum_probs = np.sum(raw_probs, axis=0)    # Sum of probabilities per number

                    # Combine features
                    features = np.concatenate([mean_probs, max_probs, sum_probs])

                    # Ensure actual result exists
                    if "realResult" not in data or len(data["realResult"]) == 0:
                        print(f"⚠ Warning: No realResult in {file}")
                        continue
                    
                    actual_result = data["realResult"]  # This is a list of actual drawn numbers

                    # Convert realResult into a one-hot encoded vector (shape: (80,))
                    real_result_vector = np.zeros(80)  # 80 possible numbers
                    for num in actual_result:
                        real_result_vector[num - 1] = 1  # Convert numbers (1-80) to index (0-79)

                    X.append(features)
                    y.append(real_result_vector)  # Now y is a probability-like distribution
        elif os.path.isfile(jsonFileOrDir):
            if jsonFileOrDir.endswith(".json"):
                with open(os.path.join(jsonFileOrDir), "r") as f:
                    data = json.load(f)

                if "currentPredictionRaw" not in data:
                    print(f"⚠ Warning: No 'currentPredictionRaw' in {jsonFileOrDir}")

                
                raw_probs = np.array(data["currentPredictionRaw"])
                
                if raw_probs.size == 0:
                    print(f"⚠ Warning: Empty probability array in {jsonFileOrDir}")


                # Debug: Print shape of `raw_probs`
                print(f"Processing {jsonFileOrDir}, shape: {raw_probs.shape}")

                # Ensure raw_probs has expected shape (20, 80)
                if raw_probs.shape[0] != 20 or raw_probs.shape[1] != 80:
                    print(f"⚠ Unexpected shape {raw_probs.shape} in {jsonFileOrDir}")


                # Feature Extraction
                mean_probs = np.mean(raw_probs, axis=0)  # Average probability per number
                max_probs = np.max(raw_probs, axis=0)    # Maximum probability per number
                sum_probs = np.sum(raw_probs, axis=0)    # Sum of probabilities per number

                # Combine features
                features = np.concatenate([mean_probs, max_probs, sum_probs])

                # Ensure actual result exists
                if "realResult" not in data or len(data["realResult"]) == 0:
                    print(f"⚠ Warning: No realResult in {jsonFileOrDir}")
                
                actual_result = data["realResult"]  # This is a list of actual drawn numbers

                # Convert realResult into a one-hot encoded vector (shape: (80,))
                real_result_vector = np.zeros(80)  # 80 possible numbers
                for num in actual_result:
                    real_result_vector[num - 1] = 1  # Convert numbers (1-80) to index (0-79)

                X.append(features)
                y.append(real_result_vector)  # Now y is a probability-like distribution

        return np.array(X), np.array(y)  # Both X and y now have compatible shapes
    

    def get_top_predictions(self, predictions, labels, num_top=20):
        """
        Extracts the top N most probable numbers from the model output.

        :param predictions: Model output (probability distribution of shape (batch_size, 80)).
        :param labels: The corresponding numbers (1-80 for Keno).
        :param num_top: Number of top predictions to extract.
        :return: List of top N numbers for each prediction.
        """
        top_numbers = []
        
        for prediction in predictions:
            # Get indices of top N probabilities
            top_indices = np.argsort(prediction)[-num_top:]  # Indices of top 20 numbers
            top_numbers.append([labels[i] for i in top_indices])  # Convert indices to numbers

        return top_numbers