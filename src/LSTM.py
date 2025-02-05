# Import necessary libraries
import os, sys, json
import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from keras import layers, regularizers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam


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

@register_keras_serializable()
def multi_label_accuracy(y_true, y_pred):
    """
    Computes the accuracy for multi-label classification.
    It calculates how many of the predicted numbers match the actual numbers.
    """
    threshold = 0.5  # Convert probabilities to binary (0 or 1)
    y_pred = K.cast(y_pred > threshold, dtype='float32')  # Convert predictions to 0s & 1s
    correct_preds = K.sum(y_true * y_pred, axis=-1)  # Count matching 1s
    total_preds = K.sum(y_true, axis=-1)  # Count total 1s in actual result
    return correct_preds / total_preds  # Percentage of correctly predicted numbers


class SelectiveProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, verbose, epoch_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = epoch_interval
    
    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.verbose = (
            0 
                if epoch % self.epoch_interval != 0 
                else self.default_verbose
        )
        super().on_epoch_begin(epoch, *args, **kwargs)

class LSTMModel:
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
        num_lstm_layers = 1
        num_dense_layers = 1
        num_bidirectional_layers = 1
        embedding_output_dimension = 16
        lstm_units = 16
        bidirectional_lstm_units = 16
        dense_units = 16
        dropout = 0.2
        l2Regularization = 0.0001

        model = models.Sequential()

        # Embedding layer
        model.add(layers.Embedding(input_dim=max_value + 1, output_dim=embedding_output_dimension))

        # LSTM+GRU layers
        for _ in range(num_lstm_layers):
            model.add(layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2Regularization)))
            model.add(layers.GRU(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2Regularization)))
            model.add(layers.Dropout(dropout))

        model.add(layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2Regularization)))
        model.add(layers.Dropout(dropout))
        
        for _ in range(num_bidirectional_layers):
            model.add(layers.Bidirectional(layers.LSTM(bidirectional_lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2Regularization))))
            model.add(layers.Dropout(dropout))
        
        for _ in range(num_dense_layers):
            # Dense layer to process sequence outputs
            model.add(layers.TimeDistributed(layers.Dense(dense_units, activation='relu')))
            model.add(layers.Dropout(dropout))

        # Output layer
        model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='softmax')))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])

        return model

    def train_model(self, model, train_data, train_labels, val_data, val_labels, model_name):
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, f"model_{model_name}_checkpoint.keras"), save_best_only=True)

        history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                            epochs=self.epochs, batch_size=self.batchSize, callbacks=[early_stopping, reduce_lr, checkpoint])
        return history

    def run(self, name='euromillions', skipLastColumns=0, maxRows=0, skipRows=0, years_back=None):
        # Load and preprocess data
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(self.dataPath, skipLastColumns, maxRows=maxRows, skipRows=skipRows, years_back=years_back)

        num_features = train_data.shape[1]

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
    

    def trainRefinePredictionsModel(self, name, path_to_json_folder, num_classes=80):
        """
        Create a neural network to refine predictions.
        Ensures output has the same shape as the original raw predictions (20, 80).
        """
        epochs = 100
        model_path = os.path.join(self.modelPath, f"refine_prediction_model_{name}.keras")

        X_train, y_train = helpers.extractFeaturesFromJsonForRefinement(path_to_json_folder)

        inputShape = (20, num_classes)  # Ensure input shape is consistent with raw predictions

        model = models.Sequential([
            layers.Input(shape=inputShape),
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=[multi_label_accuracy])
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=4,
            verbose=False,
            callbacks=[SelectiveProgbarLogger(verbose=1, epoch_interval=epochs//10)]
        )
        
        model.save(model_path)
        
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.savefig(os.path.join(self.modelPath, f'refine_prediction_model_{name}_performance.png'))
        print(f"Refine Prediction AI Model {name} Trained and Saved!")


    def refinePrediction(self, name, pathToLatestPredictionFile):
        """
        Refine the predictions while keeping the (20, 80) shape.
        """
        model_path = os.path.join(self.modelPath, f"refine_prediction_model_{name}.keras")
        second_model = load_model(model_path, custom_objects={"multi_label_accuracy": multi_label_accuracy})

        X_new, _ = helpers.extractFeaturesFromJsonForRefinement(pathToLatestPredictionFile)
        refined_prediction = second_model.predict(X_new)
        
        refined_prediction = refined_prediction.reshape(-1, 20, 80)  # Ensure correct shape
        print("Refined Prediction Shape:", refined_prediction.shape)
        return refined_prediction
    
    def trainTopPredictionsModel(self, name, path_to_json_folder, num_classes):
        """
        Create a neural network to get the top prediction.
        @num_classes: How many numbers to predict.
        """
        epochs = 1000
        model_path = os.path.join(self.modelPath, f"top_prediction_model_{name}.keras")

        X_train, y_train = helpers.extractFeaturesFromJsonForDetermineTopPrediction(path_to_json_folder)

        inputShape = (X_train.shape[1],)

        model = models.Sequential([
            layers.Input(shape=inputShape), 
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.6),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.6),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=[multi_label_accuracy])

        #print("X_train:", X_train)  # Should be (num_samples, 240)
        #print("y_train:", y_train)  # Should be (num_samples, 80) if one-hot encoded in case of keno
        
        # Create and train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=4, verbose=False, callbacks=[SelectiveProgbarLogger(verbose=1, epoch_interval=epochs/10)])

        # Save model for future use
        model.save(model_path)

        # Plot training history
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.savefig(os.path.join(self.modelPath, f'top_prediction_model_{name}_performance.png'))

        print(f"Refine Prediction AI Model {name} Trained and Saved!")
    
    def topPrediction(self, name, pathToLatestPredictionFile):
        """
            Get top prediction with an AI
        """

        model_path = os.path.join(self.modelPath, f"top_prediction_model_{name}.keras")

        second_model = load_model(model_path)

        # Get new prediction features
        new_json = pathToLatestPredictionFile
        X_new, _ = helpers.extractFeaturesFromJsonForDetermineTopPrediction(new_json)

        # Get refined prediction
        refined_prediction = second_model.predict(X_new)

        print("Top Prediction: ", refined_prediction)
        return refined_prediction
    

    
    

# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":
    lstm_model = LSTMModel()

    name = 'keno'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "lstm_model")

    lstm_model.setModelPath(modelPath)
    lstm_model.setDataPath(dataPath)
    lstm_model.setBatchSize(16)
    lstm_model.setEpochs(1000)

    """
    latest_raw_predictions, unique_labels = lstm_model.run(name, years_back=1)

    #helpers.print_predicted_numbers(latest_raw_predictions)

    # Opening JSON file
    sequenceToPredictFile = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "sequenceToPredict_{0}.json".format(name))
    with open(sequenceToPredictFile, 'r') as openfile:
        sequenceToPredict = json.load(openfile)

    # Generate set of predictions
    print(len(latest_raw_predictions[0]))

    # Check on prediction with nth highest probability
    for i in range(10):
        prediction_highest_indices = helpers.decode_predictions(latest_raw_predictions, unique_labels, nHighestProb=i)
        print("Prediction with ", i+1 ,"highest probs: ", prediction_highest_indices)
        matching_numbers = helpers.find_matching_numbers(sequenceToPredict["sequenceToPredict"], prediction_highest_indices)
        print("Matching Numbers with ", i+1 ,"highest probs: ", matching_numbers)
    """

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    pathToLatestJsonFile = os.path.join(jsonDirPath, "2025-1-31.json")
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-2-1.json")
    labels = np.arange(1, 81) # for testing but we can extract the labels from the run

    """
    # Refine predictions
    lstm_model.trainRefinePredictionsModel(name, jsonDirPath, num_classes=80)
    refined_prediction_raw = lstm_model.refinePrediction(name=name, pathToLatestPredictionFile=pathToLatestJsonFile)

    print("refined_prediction_raw: ", refined_prediction_raw)

    # Print refined predictions
    for i in range(10):
        prediction_highest_indices = helpers.decode_predictions(refined_prediction_raw[0], labels, nHighestProb=i)
        print("Prediction with ", i+1 ,"highest probs: ", prediction_highest_indices)

        # Opening JSON file
        with open(sequenceToPredictFile, 'r') as openfile:
            sequenceToPredict = json.load(openfile)

        matching_numbers = helpers.find_matching_numbers(sequenceToPredict["realResult"], prediction_highest_indices)
        print("Matching Numbers with ", i+1 ,"highest probs: ", matching_numbers)
    """

    lstm_model.trainTopPredictionsModel(name, jsonDirPath, num_classes=80)
    top_prediction_raw = lstm_model.topPrediction(name=name, pathToLatestPredictionFile=pathToLatestJsonFile)
    topPrediction = helpers.getTopPredictions(top_prediction_raw, labels, num_top=20)

    # Print Top prediction
    for i, prediction in enumerate(topPrediction):
        prediction = [int(num) for num in prediction]
        print(f"Top Prediction {i+1}: {sorted(prediction)}")

        # Opening JSON file
        with open(sequenceToPredictFile, 'r') as openfile:
            sequenceToPredict = json.load(openfile)

        # Check for matching numbers    
        matchingNumbers = helpers.find_matching_numbers(prediction, sequenceToPredict["realResult"])
        print("Matching Numbers: ", matchingNumbers)

