# Import necessary libraries
import os, sys, json
import pandas as pd

from matplotlib import pyplot as plt
#from tensorflow import keras
from keras import layers, regularizers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Dynamically adjust the import path for Helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

# Ensure Helpers can be imported
if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Helpers import Helpers

helpers = Helpers()

class LSTM():
    dataPath = ""
    modelPath = ""
    epochs = 1000
    batchSize = 4

    

    def setDataPath(self, dataPath):
        self.dataPath = dataPath
    
    def setModelPath(self, modelPath):
        self.modelPath = modelPath
    
    def setEpochs(self, epochs):
        self.epochs = epochs
    
    def setBatchSize(self, batchSize):
        self.batchSize = batchSize

    

    # Function to create the model
    def create_model(self, num_features, max_value):
        num_layers = 10
        num_lstm_layers = 50
        num_deep_layers = 50
        embedding_output_dimension = 64
        lstm_units = 64
        dense_units = 64
        # Create a sequential model
        model = models.Sequential()
        
        # Add an Embedding layer
        model.add(layers.Embedding(input_dim=max_value + 1, output_dim=embedding_output_dimension, input_length=None))
        
        for _ in range(num_lstm_layers):
            # Add number of LSTM layer with L2 regularization
            model.add(layers.LSTM(lstm_units, return_sequences=True, 
                                kernel_regularizer=regularizers.l2(0.001)))
        
        # Add a Dropout layer
        model.add(layers.Dropout(0.2))

        model.add(layers.LSTM(lstm_units, return_sequences=False, 
                                kernel_regularizer=regularizers.l2(0.001)))
        
        
        for _ in range(num_deep_layers):
            # Add a Dense layer
            model.add(layers.Dense(dense_units, activation='relu'))  # First Dense layer
        
    
        #model.add(layers.Dropout(0.2))  # Optional Dropout layer

        # Add a final Dense layer for output
        model.add(layers.Dense(num_features, activation='softmax'))
        
        # Compile the model with mean_squared_error loss, adam optimizer, and mae metric
        model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mae'])

        #print(model.summary())
        
        return model

    # Function to train the model
    def train_model(self, model, train_data, val_data, modelName):
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-4)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, "model_{0}_checkpoint.keras".format(modelName)), save_best_only=True)

        # Fit the model on the training data and validate on the validation data for 100 epochs
        history = model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=self.epochs, batch_size=self.batchSize, callbacks=[early_stopping, reduce_lr, checkpoint])
        
        return history
        


    # Main function to run the complete lstm flow  
    def run(self, data='euromillions', skipLastColumns=0):
        
        # Load and preprocess data 
        train_data, val_data, max_value = helpers.load_data(self.dataPath, skipLastColumns)
        
        # Get number of features from training data 
        num_features = train_data.shape[1]

        if os.path.exists(os.path.join(self.modelPath, "model_{0}.keras".format(data))):
            model = load_model(os.path.join(self.modelPath, "model_{0}.keras".format(data)))
        elif os.path.exists(os.path.join(self.modelPath, "model_{0}_checkpoint.keras".format(data))):
            model = load_model(os.path.join(self.modelPath, "model_{0}_checkpoint.keras".format(data)))
        else:
            # Create and compile model 
            model = self.create_model(num_features, max_value)
    
    
        # Train model 
        history = self.train_model(model, train_data, val_data, modelName=data)
        
        # Predict numbers using trained model 
        predicted_numbers = helpers.predict_numbers(model, val_data, num_features)

        pd.DataFrame(history.history).plot(figsize=(8,5))
        plt.savefig(os.path.join(self.modelPath, 'model_{0}_performance.png'.format(data)))

        model.save(os.path.join(self.modelPath, "model_{0}.keras".format(data)))

        if(os.path.exists(os.path.join(self.modelPath, "model_{0}_checkpoint.keras".format(data)))):
            os.remove(os.path.join(self.modelPath, "model_{0}_checkpoint.keras".format(data)))
        
        return predicted_numbers

# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":
    lstm = LSTM()

    data = 'euromillions'
    #data = 'lotto'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", data)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "lstm_model")

    lstm.setModelPath(modelPath)
    lstm.setDataPath(dataPath)
    lstm.setBatchSize(4)
    lstm.setEpochs(10)
    predictedNumbers = lstm.run(data)
    
    helpers.print_predicted_numbers(predictedNumbers)

    # Opening JSON file
    sequenceToPredictFile = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "sequenceToPredict_euromillions.json")
    with open(sequenceToPredictFile, 'r') as openfile:
    
        # Reading from json file
        sequenceToPredict = json.load(openfile)

    best_match_index, best_match_sequence, matching_numbers = helpers.find_matching_numbers(sequenceToPredict["sequenceToPredict"], predictedNumbers)

    print("Best Matching Index: ", best_match_index)
    print("Best Matching Sequence: ", best_match_sequence)
    print("Matching Numbers: ", matching_numbers)
