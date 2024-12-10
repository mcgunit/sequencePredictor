# Import necessary libraries
import os, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from dateutil.parser import parse
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers, regularizers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
from art import text2art

path = os.getcwd()
dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "euromillions")

# Function to print the introduction of the program
def print_intro():
    # Generate ASCII art with the text "gru"
    ascii_art = text2art("GRU")
    # Print the introduction and ASCII art
    print("============================================================")
    print("GRU")
    print("Licence : MIT License")
    print(ascii_art)
    print("Prediction artificial intelligence")

# Function to load data from a file and preprocess it
def load_data():
    # Initialize an empty list to hold the data
    data = []

    for csvFile in os.listdir(dataPath):
        if csvFile.endswith(".csv"):
            print(f"Processing file: {csvFile}")
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

# Function to create the model
def create_model(num_features, max_value):
    model = models.Sequential()

    # Add an Embedding layer
    model.add(layers.Embedding(input_dim=max_value + 1, output_dim=64, input_length=None))
    
    # First GRU layer
    model.add(layers.GRU(128, return_sequences=True,
                         kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))

    # Second GRU layer
    model.add(layers.GRU(64, return_sequences=True, 
                         kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))

    # Third GRU layer
    model.add(layers.GRU(32, return_sequences=False, 
                         kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))

    # Dense output layer
    model.add(layers.Dense(num_features, activation='linear'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

    return model

# Function to train the model
def train_model(model, train_data, val_data, modelName):
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-3)
    checkpoint = ModelCheckpoint(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "gru_model", "model_{0}_checkpoint.keras".format(modelName)), save_best_only=True)

    # Fit the model on the training data and validate on the validation data for 100 epochs
    history = model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=1000, batch_size=4, callbacks=[early_stopping, reduce_lr, checkpoint])
    
    return history
    


# Function to predict numbers using the trained model
def predict_numbers(model, val_data, num_features):
    # Predict on the validation data using the model
    predictions = model.predict(val_data)
    # Get the indices of the top 'num_features' predictions for each sample in validation data
    indices = np.argsort(predictions, axis=1)[:, -num_features:]
    # Get the predicted numbers using these indices from validation data
    predicted_numbers = np.take_along_axis(val_data, indices, axis=1)
    return predicted_numbers

# Function to print the predicted numbers
def print_predicted_numbers(predicted_numbers):
    # Print a separator line and "Predicted Numbers:"
    print("============================================================")
    print("Predicted Numbers:")
    # Print only the first row of predicted numbers
    print(', '.join(map(str, predicted_numbers[0])))
    print("============================================================")

# Main function to run everything   
def main(args):
    data = args.data

    # Print introduction of program 
    print_intro()
    
    # Load and preprocess data 
    train_data, val_data, max_value = load_data()
    
    # Get number of features from training data 
    num_features = train_data.shape[1]

    if os.path.exists(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "gru_model", "model_{0}.keras".format(data))):
        model = load_model(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "gru_model", "model_{0}.keras".format(data)))
    elif os.path.exists(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "gru_model", "model_{0}_checkpoint.keras".format(data))):
        model = load_model(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "gru_model", "model_{0}_checkpoint.keras".format(data)))
    else:
        # Create and compile model 
        model = create_model(num_features, max_value)
   
   
   
    # Train model 
    history = train_model(model, train_data, val_data, modelName=data)
    
    # Predict numbers using trained model 
    predicted_numbers = predict_numbers(model, val_data, num_features)
    
    # Print predicted numbers 
    print_predicted_numbers(predicted_numbers)

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.savefig('model_{0}_performance.png'.format(data))

    model.save(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "gru_model", "model_{0}.keras".format(data)))

    if(os.path.exists(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "gru_model", "model_{0}_checkpoint.keras".format(data)))):
        os.remove(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "gru_model", "model_{0}_checkpoint.keras".format(data)))

# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(
                    prog='gru Sequence Predictor',
                    description='Tries to predict a sequence of numbers',
                    epilog='Check it uit')  
    parser.add_argument('-d', '--data', default="euromillions")
    
    args = parser.parse_args()
    print(args.data)

    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", args.data)

    main(args)
