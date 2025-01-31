# Import necessary libraries
import os, sys, json
import pandas as pd
import tensorflow as tf

from tcn import TCN  # Import the TCN layer
from matplotlib import pyplot as plt
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

class TCNModel:
    def __init__(self):
        self.dataPath = ""
        self.modelPath = ""
        self.epochs = 1000
        self.batchSize = 4

    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    def setModelPath(self, modelPath):
        self.modelPath = modelPath

    def setEpochs(self, epochs):
        self.epochs = epochs

    def setBatchSize(self, batchSize):
        self.batchSize = batchSize


    """
    If training loss is high: The model is underfitting. Increase complexity or train for more epochs.
    If validation loss diverges from training loss: The model is overfitting. Add more regularization (dropout, L2).
    """
    def create_model(self, max_value, num_classes=50):
        embedding_output_dimension = 64
        tcn_units = 64              # Number of filters in TCN layers
        num_tcn_layers = 2          # Number of TCN layers
        num_dense_layers = 2        # Numbers of Dense layers
        dense_units = 64            # Number of units in dense layersW
        dropout_rate = 0.3
        l2_lambda = 0.001           # Set the L2 regularization factor

        model = models.Sequential()

        # Embedding layer for input sequences
        model.add(layers.Embedding(input_dim=max_value + 1, output_dim=embedding_output_dimension))

        # Add TCN layers
        for _ in range(num_tcn_layers):
            model.add(TCN(nb_filters=tcn_units, return_sequences=True, dropout_rate=dropout_rate, kernel_size=3))

        # Dense layers for processing TCN outputs
        for _ in range(num_dense_layers):  # Add a couple of dense layers
            model.add(layers.TimeDistributed(layers.Dense(dense_units, activation='relu', 
                                                       kernel_regularizer=regularizers.l2(l2_lambda))))
            model.add(layers.Dropout(dropout_rate))

        # Output layer with softmax activation
        model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='softmax')))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

        return model


    def train_model(self, model, train_data, train_labels, val_data, val_labels, model_name):
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, f"model_{model_name}_checkpoint.keras"), save_best_only=True)

        history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                            epochs=self.epochs, batch_size=self.batchSize, callbacks=[early_stopping, reduce_lr, checkpoint])
        return history

    
    def run(self, name='euromillions', skipLastColumns=0, maxRows=0, skipRows=0, years_back=None):
        """
        Train and perform a prediction
        """
        # Load and preprocess data
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(self.dataPath, skipLastColumns, maxRows=maxRows, skipRows=skipRows, years_back=years_back)

        model_path = os.path.join(self.modelPath, f"model_{name}.keras")
        checkpoint_path = os.path.join(self.modelPath, f"model_{name}_checkpoint.keras")

        if os.path.exists(model_path):
            model = load_model(model_path)
        elif os.path.exists(checkpoint_path):
            model = load_model(checkpoint_path)
        else:
            model = self.create_model(max_value, num_classes)

        # Train the model
        history = self.train_model(model, train_data, train_labels, val_data, val_labels, model_name=name)

        # Predict numbers
        latest_raw_predictions = helpers.predict_numbers(model, numbers)

        # Plot training history
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.savefig(os.path.join(self.modelPath, f'model_{name}_performance.png'))

        # Save model
        model.save(model_path)

        # Remove checkpoint if exists
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return latest_raw_predictions, unique_labels
    
    def doPrediction(self, modelPath, skipLastColumns, maxRows=0):
        """
        Do only a prediction. modelPath is the absolute path to the model
        """

        numbers = helpers.load_prediction_data(self.dataPath, skipLastColumns, maxRows=maxRows)

        model = load_model(modelPath)

        # Predict numbers
        latest_raw_predictions = helpers.predict_numbers(model, numbers)

        return latest_raw_predictions


# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":
    tcn_model = TCNModel()

    name = 'pick3'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "tcn_model")

    tcn_model.setModelPath(modelPath)
    tcn_model.setDataPath(dataPath)
    tcn_model.setBatchSize(16)
    tcn_model.setEpochs(1000)
    
    latest_raw_predictions, unique_labels = tcn_model.run(name, years_back=1)

    #helpers.print_predicted_numbers(latest_raw_predictions)

    # Opening JSON file
    sequenceToPredictFile = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "sequenceToPredict_{0}.json".format(name))
    with open(sequenceToPredictFile, 'r') as openfile:
        sequenceToPredict = json.load(openfile)

    # Generate set of predictions
    print("Raw predictions: ", latest_raw_predictions)

    # Check on prediction with nth highest probability
    for i in range(10):
        prediction_highest_indices = helpers.decode_predictions(latest_raw_predictions, unique_labels, nHighestProb=i)
        print("Prediction with ", i+1 ,"highest probs: ", prediction_highest_indices)
        matching_numbers = helpers.find_matching_numbers(sequenceToPredict["sequenceToPredict"], prediction_highest_indices)
        print("Matching Numbers with ", i+1 ,"highest probs: ", matching_numbers)
