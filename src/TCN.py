# Import necessary libraries
import os, sys, json
import pandas as pd

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
from SelectiveProgbarLogger import SelectiveProgbarLogger

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
    def create_model(self, max_value, num_classes=50, model_path=""):
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

        model.build(input_shape=(None, None))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])

        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}")
            model.load_weights(model_path)

        return model


    def train_model(self, model, train_data, train_labels, val_data, val_labels, model_name):
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, f"model_{model_name}_checkpoint.keras"), save_best_only=True)

        history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                            epochs=self.epochs, batch_size=self.batchSize, callbacks=[early_stopping, reduce_lr, checkpoint, SelectiveProgbarLogger(verbose=1, epoch_interval=self.epochs/2)])
        return history

    
    def run(self, name='euromillions', skipLastColumns=0, maxRows=0, skipRows=0, years_back=None):
        """
        Train and perform a prediction
        """
        # Load and preprocess data
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(self.dataPath, skipLastColumns, maxRows=maxRows, skipRows=skipRows, years_back=years_back)

        model_path = os.path.join(self.modelPath, f"model_{name}.keras")
        checkpoint_path = os.path.join(self.modelPath, f"model_{name}_checkpoint.keras")

        model = self.create_model(max_value, num_classes=num_classes, model_path=model_path)

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
    from RefinemePrediction import RefinePrediction
    from TopPrediction import TopPrediction

    tcn_model = TCNModel()
    refinePrediction = RefinePrediction()
    topPrediction = TopPrediction()

    name = 'keno'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "tcn_model")

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    pathToLatestJsonFile = os.path.join(jsonDirPath, "2025-1-31.json")
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-2-1.json")

    # Opening JSON file
    with open(sequenceToPredictFile, 'r') as openfile:
        sequenceToPredict = json.load(openfile)

    numbersLength = len(sequenceToPredict["realResult"])

    tcn_model.setModelPath(modelPath)
    tcn_model.setDataPath(dataPath)
    tcn_model.setBatchSize(16)
    tcn_model.setEpochs(1000)

    latest_raw_predictions, unique_labels = tcn_model.run(name, years_back=1)
    num_classes = len(unique_labels)


    # Check on prediction with nth highest probability
    for i in range(2):
        prediction_highest_indices = helpers.decode_predictions(latest_raw_predictions, unique_labels, nHighestProb=i)
        print("Prediction with ", i+1 ,"highest probs: ", prediction_highest_indices)
        matching_numbers = helpers.find_matching_numbers(sequenceToPredict["realResult"], prediction_highest_indices)
        print("Matching Numbers with ", i+1 ,"highest probs: ", matching_numbers)
    


    # Refine predictions
    refinePrediction.trainRefinePredictionsModel(name, jsonDirPath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)
    refined_prediction_raw = refinePrediction.refinePrediction(name=name, pathToLatestPredictionFile=pathToLatestJsonFile, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)

    #print("refined_prediction_raw: ", refined_prediction_raw)

    # Print refined predictions
    for i in range(2):
        prediction_highest_indices = helpers.decode_predictions(refined_prediction_raw[0], unique_labels, nHighestProb=i)
        print("Prediction with ", i+1 ,"highest probs: ", prediction_highest_indices)

        matching_numbers = helpers.find_matching_numbers(sequenceToPredict["realResult"], prediction_highest_indices)
        print("Matching Numbers with ", i+1 ,"highest probs: ", matching_numbers)


    
    # Top prediction
    topPrediction.trainTopPredictionsModel(name, jsonDirPath, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)
    top_prediction_raw = topPrediction.topPrediction(name=name, pathToLatestPredictionFile=pathToLatestJsonFile, modelPath=modelPath, num_classes=num_classes, numbersLength=numbersLength)
    topPrediction = helpers.getTopPredictions(top_prediction_raw, unique_labels, num_top=numbersLength)

    # Print Top prediction
    for i, prediction in enumerate(topPrediction):
        prediction = [int(num) for num in prediction]
        print(f"Top Prediction {i+1}: {sorted(prediction)}")

        # Check for matching numbers    
        matchingNumbers = helpers.find_matching_numbers(prediction, sequenceToPredict["realResult"])
        print("Matching Numbers: ", matchingNumbers)
