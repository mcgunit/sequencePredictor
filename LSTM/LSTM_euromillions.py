# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
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
    # Generate ASCII art with the text "LSTM"
    ascii_art = text2art("LSTM")
    # Print the introduction and ASCII art
    print("============================================================")
    print("LSTM")
    print("Licence : MIT License")
    print(ascii_art)
    print("Lottery prediction artificial intelligence")

# Function to load data from a file and preprocess it
def load_data():
    # Initialize an empty list to hold the data
    data = []

    # Iterate through files in the directory
    for csvFile in os.listdir(dataPath):
        if csvFile.endswith(".csv"):
            print(f"Processing file: {csvFile}")
            try:
                # Construct full file path
                file_path = os.path.join(dataPath, csvFile)
                
                # Load data from the file
                csvData = np.genfromtxt(file_path, usecols=range(1, 7), delimiter=';', dtype=int, skip_header=1)
                
                # Append each entry to the data list
                for entry in csvData:
                    #print("Entry: ", entry)
                    data.append(entry)
            except Exception as e:
                print(f"Error processing file {csvFile}: {e}")

    # Convert the data list to a NumPy array for easier manipulation
    data = np.array(data)
    
    # Replace all -1 values with 0
    data[data == -1] = 0
    # Split data into training and validation sets
    train_data = data[:int(0.8*len(data))]
    val_data = data[int(0.8*len(data)):]
    # Get the maximum value in the data
    max_value = np.max(data)
    return train_data, val_data, max_value

# Function to create the model
def create_model(num_features, max_value):
    # Create a sequential model
    model = models.Sequential()
    
    # Add an Embedding layer
    model.add(layers.Embedding(input_dim=max_value + 1, output_dim=64, input_length=None))

    # Add the first LSTM layer with L2 regularization
    model.add(layers.LSTM(256, return_sequences=True, 
                          kernel_regularizer=regularizers.l2(0.1), 
                          recurrent_regularizer=regularizers.l2(0.1)))
    
    # Add a Dropout layer
    model.add(layers.Dropout(0.1))

    # Add the first LSTM layer with L2 regularization
    model.add(layers.LSTM(128, return_sequences=True, 
                          kernel_regularizer=regularizers.l2(0.1), 
                          recurrent_regularizer=regularizers.l2(0.1)))
    
    # Add a Dropout layer
    model.add(layers.Dropout(0.1))
    
    # Add the first LSTM layer with L2 regularization
    model.add(layers.LSTM(64, return_sequences=True, 
                          kernel_regularizer=regularizers.l2(0.1), 
                          recurrent_regularizer=regularizers.l2(0.1)))
    
    # Add a Dropout layer
    model.add(layers.Dropout(0.1))
    
    # Add a second LSTM layer with L2 regularization
    model.add(layers.LSTM(32, return_sequences=True,
                          kernel_regularizer=regularizers.l2(0.1), 
                          recurrent_regularizer=regularizers.l2(0.1)))
    
    # Add another Dropout layer
    model.add(layers.Dropout(0.1))
    
    # Add a second LSTM layer with L2 regularization
    model.add(layers.LSTM(16, 
                          kernel_regularizer=regularizers.l2(0.1), 
                          recurrent_regularizer=regularizers.l2(0.1)))
    
    # Add another Dropout layer
    model.add(layers.Dropout(0.1))
    
    # Add a Dense layer for output
    model.add(layers.Dense(num_features, activation='softmax'))
    
    # Compile the model with categorical crossentropy loss, adam optimizer, and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])
    
    return model

# Function to train the model
def train_model(model, train_data, val_data):
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "lstm_model", "model_euromillions_checkpoint.keras"), save_best_only=True)

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
def main():
   # Print introduction of program 
   print_intro()
   
   # Load and preprocess data 
   train_data, val_data, max_value = load_data()
   
   # Get number of features from training data 
   num_features = train_data.shape[1]

   if os.path.exists(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "lstm_model", "model_euromillions.keras")):
       model = load_model(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "lstm_model", "model_euromillions.keras"))
   else:
        # Create and compile model 
       model = create_model(num_features, max_value)
   
   
   
   # Train model 
   history = train_model(model, train_data, val_data)
   
   # Predict numbers using trained model 
   predicted_numbers = predict_numbers(model, val_data, num_features)
   
   # Print predicted numbers 
   print_predicted_numbers(predicted_numbers)

   plt.plot(history.history['accuracy'])
   plt.plot(history.history['val_accuracy'])
   plt.title('model accuracy')
   plt.ylabel('accuracy')
   plt.xlabel('epoch')
   plt.legend(['train', 'val'], loc='upper left')
   plt.savefig('model_accuracy.png')

   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.title('model loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'val'], loc='upper left')
   plt.savefig('model_loss.png')

   model.save(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "lstm_model", "model_euromillions.keras"))

   if(os.path.exists(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "lstm_model", "model_euromillions_checkpoint.keras"))):
       os.remove(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "lstm_model", "model_euromillions_checkpoint.keras"))

# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":
   main()
