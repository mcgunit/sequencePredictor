# Import necessary libraries
import os
import numpy as np
import pandas as pd
from dateutil.parser import parse
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers, regularizers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from tensorflow.keras.models import load_model

from Helpers import Helpers

helpers = Helpers()

class GRU():
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
    def train_model(self, model, train_data, val_data, modelName):
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-3)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, "model_{0}_checkpoint.keras".format(modelName)), save_best_only=True)

        # Fit the model on the training data and validate on the validation data for 100 epochs
        history = model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=self.epochs, batch_size=self.batchSize, callbacks=[early_stopping, reduce_lr, checkpoint])
        
        return history
        

    # Main function to run the full GRU
    def run(self, data):

        # Load and preprocess data 
        train_data, val_data, max_value = helpers.load_data(self.dataPath)
        
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
        plt.savefig('model_{0}_performance.png'.format(data))

        model.save(os.path.join(self.modelPath, "model_{0}.keras".format(data)))

        if(os.path.exists(os.path.join(self.modelPath, "model_{0}_checkpoint.keras".format(data)))):
            os.remove(os.path.join(self.modelPath, "model_{0}_checkpoint.keras".format(data)))

        return predicted_numbers

# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":
   
    gru = GRU()

    #data = 'euromillions'
    data = 'lotto'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "trainingData", data)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "models", "gru_model")

    gru.setModelPath(modelPath)
    gru.setDataPath(dataPath)
    gru.setBatchSize(4)
    gru.setEpochs(1)

    predictedNumbers = gru.run(data)
    
    print(predictedNumbers.tolist())
    helpers.print_predicted_numbers(predictedNumbers)
