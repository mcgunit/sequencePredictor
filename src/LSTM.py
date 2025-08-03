# Import necessary libraries
import os, sys, json
import pandas as pd

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
from SelectiveProgbarLogger import SelectiveProgbarLogger

helpers = Helpers()


class LSTMModel:
    def __init__(self):
        self.dataPath = ""
        self.modelPath = ""
        self.epochs = 1000
        self.batchSize = 4
        self.num_lstm_layers = 1
        self.num_dense_layers = 1
        self.num_bidirectional_layers = 1
        self.embedding_output_dimension = 16
        self.lstm_units = 16
        self.bidirectional_lstm_units = 16
        self.dense_units = 16
        self.dropout = 0.2
        self.l2Regularization = 0.0001
        self.earlyStopPatience = 20
        self.reduceLearningRatePatience = 5
        self.reduceLearningRateFactor = 0.9
        self.useFinalLSTMLayer = True
        self.useGRU = True
        self.denseActivation = "relu"
        self.outputActivation = "softmax"

    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    def setModelPath(self, modelPath):
        self.modelPath = modelPath

    def setEpochs(self, epochs):
        self.epochs = epochs

    def setBatchSize(self, batchSize):
        self.batchSize = batchSize

    def setNumberOfLSTMLayers(self, nLayers):
        self.num_lstm_layers = nLayers
    
    def setNumberOfDenseLayers(self, nLayers):
        self.num_dense_layers = nLayers

    def setNumberOfBidrectionalLayers(self, nLayers):
        self.num_bidirectional_layers = nLayers

    def setNumberOfEmbeddingOutputDimension(self, dimension):
        self.embedding_output_dimension = dimension

    def setNumberOfLstmUnits(self, units):
        self.lstm_units = units
    
    def setNumberOfBidirectionalLstmUnits(self, units):
        self.bidirectional_lstm_units = units

    def setNumberOfDenseUnits(self, units):
        self.dense_units = units

    def setDropout(self, dropout):
        self.dropout = dropout
    
    def setL2Regularization(self, value):
        self.l2Regularization = value
    
    def setEarlyStopPatience(self, value):
        self.earlyStopPatience = value
    
    def setReduceLearningRatePAience(self, value):
        self.reduceLearningRatePatience = value
    
    def setReducedLearningRateFactor(self, value):
        self.reduceLearningRateFactor = value
    
    def setUseFinalLSTMLayer(self, value):
        self.useFinalLSTMLayer = value
    
    def setUseGRU(self, value):
        self.useGRU = value

    def setDenseActivation(self, value):
        self.denseActivation = value

    def setOutpuActivation(self, value):
        self.outputActivation = value

    """
    If training loss is high: The model is underfitting. Increase complexity or train for more epochs.
    If validation loss diverges from training loss: The model is overfitting. Add more regularization (dropout, L2).
    """
    def create_model(self, max_value, num_classes=50, model_path=""):
        num_lstm_layers = self.num_lstm_layers
        num_dense_layers = self.num_dense_layers
        num_bidirectional_layers = self.num_bidirectional_layers
        embedding_output_dimension = self.embedding_output_dimension
        lstm_units = self.lstm_units
        bidirectional_lstm_units = self.bidirectional_lstm_units
        dense_units = self.dense_units
        dropout = self.dropout
        l2Regularization = self.l2Regularization

        model = models.Sequential()

        # Embedding layer
        model.add(layers.Embedding(input_dim=max_value + 1, output_dim=embedding_output_dimension))

        # LSTM+GRU layers
        for _ in range(num_lstm_layers):
            model.add(layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2Regularization)))
            if self.useGRU:
                model.add(layers.GRU(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2Regularization)))
            model.add(layers.Dropout(dropout))
        
        if self.useFinalLSTMLayer:
            model.add(layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2Regularization)))
            model.add(layers.Dropout(dropout))
        
        for _ in range(num_bidirectional_layers):
            model.add(layers.Bidirectional(layers.LSTM(bidirectional_lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2Regularization))))
            model.add(layers.Dropout(dropout))
        
        for _ in range(num_dense_layers):
            # Dense layer to process sequence outputs
            model.add(layers.TimeDistributed(layers.Dense(dense_units, activation=self.denseActivation)))
            model.add(layers.Dropout(dropout))

        # Output layer
        model.add(layers.TimeDistributed(layers.Dense(num_classes, activation=self.outputActivation)))

        model.build(input_shape=(None, None))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])

        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}")
            model.load_weights(model_path)

        return model

    def train_model(self, model, train_data, train_labels, val_data, val_labels, model_name):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.earlyStopPatience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.reduceLearningRateFactor, patience=self.reduceLearningRatePatience)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, f"model_{model_name}_checkpoint.keras"), save_best_only=True)

        history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                            epochs=self.epochs, batch_size=self.batchSize, verbose=False, callbacks=[early_stopping, reduce_lr, checkpoint, SelectiveProgbarLogger(verbose=1, epoch_interval=self.epochs/2)])
        return history

    def run(self, name='euromillions', skipLastColumns=0, maxRows=0, skipRows=0, years_back=None):
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

    lstm_model = LSTMModel()
    refinePrediction = RefinePrediction()
    topPrediction = TopPrediction()

    name = 'keno'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "lstm_model")

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    pathToLatestJsonFile = os.path.join(jsonDirPath, "2025-1-31.json")
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-2-1.json")

    # Opening JSON file
    with open(sequenceToPredictFile, 'r') as openfile:
        sequenceToPredict = json.load(openfile)

    numbersLength = len(sequenceToPredict["realResult"])

    lstm_model.setModelPath(modelPath)
    lstm_model.setDataPath(dataPath)
    lstm_model.setBatchSize(16)
    lstm_model.setEpochs(1000)

    latest_raw_predictions, unique_labels = lstm_model.run(name, years_back=1)
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

