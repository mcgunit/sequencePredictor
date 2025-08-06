# Import necessary libraries
import os, sys, json
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from keras import layers, regularizers, models, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical


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
from Markov import Markov 

helpers = Helpers()
markov = Markov()


class LSTMModel:
    def __init__(self):
        self.dataPath = ""
        self.modelPath = ""
        self.epochs = 1000
        self.batchSize = 4
        self.num_lstm_layers = 1
        self.num_bidirectional_layers = 1
        self.lstm_units = 16
        self.bidirectional_lstm_units = 16
        self.dropout = 0.2
        self.l2Regularization = 0.0001
        self.earlyStopPatience = 20
        self.reduceLearningRatePatience = 5
        self.reduceLearningRateFactor = 0.9
        self.useFinalLSTMLayer = True
        self.useGRU = True
        self.outputActivation = "softmax"
        self.optimizer_type = 'adam'
        self.learning_rate = 0.005
        self.loadModelWeights = True
        self.window_size = 10
        self.predictionWindowSize = 20

        # LSTM + Markov
        self.markovAlpha = 0.5
        self.lstmMarkov = None

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

    def setNumberOfBidrectionalLayers(self, nLayers):
        self.num_bidirectional_layers = nLayers

    def setNumberOfLstmUnits(self, units):
        self.lstm_units = units
    
    def setNumberOfBidirectionalLstmUnits(self, units):
        self.bidirectional_lstm_units = units

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

    def setOutpuActivation(self, value):
        self.outputActivation = value

    def setOptimizer(self, optimizer): 
        self.optimizer_type = optimizer.lower()

    def setLearningRate(self, value):
        self.learning_rate = value

    def setLoadModelWeights(self, value):
        self.loadModelWeights = value

    def setWindowSize(self, value):
        self.window_size = value
    
    def setPredictionWindowSize(self, value):
        self.predictionWindowSize = value

    def setMarkovAlpha(self, value):
        self.markovAlpha = value

    def getLstmMArkov(self):
        return self.lstmMarkov


    """
    If training loss is high: The model is underfitting. Increase complexity or train for more epochs.
    If validation loss diverges from training loss: The model is overfitting. Add more regularization (dropout, L2).
    """
    def create_model(self, max_value, num_classes=50, model_path="", digitsPerDraw=7):
        num_lstm_layers = self.num_lstm_layers
        num_bidirectional_layers = self.num_bidirectional_layers
        lstm_units = self.lstm_units
        bidirectional_lstm_units = self.bidirectional_lstm_units
        dropout = self.dropout
        l2Regularization = self.l2Regularization
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        if self.optimizer_type == "adam":
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_type == "sgd":
            optimizer = optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        elif self.optimizer_type == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer_type == "adagrad":
            optimizer = optimizers.Adagrad(learning_rate=self.learning_rate)
        elif self.optimizer_type == "nadam":
            optimizer = optimizers.Nadam(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        model = models.Sequential()
        model.add(layers.Input(shape=(digitsPerDraw, num_classes)))  # 3 features (e.g., digits in draw)

        # LSTM layers
        for _ in range(num_lstm_layers):
            model.add(layers.LSTM(lstm_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(l2Regularization)))
            if self.useGRU:
                model.add(layers.GRU(lstm_units, return_sequences=True,
                                    kernel_regularizer=regularizers.l2(l2Regularization)))
            model.add(layers.Dropout(dropout))

        # Final LSTM if needed
        if self.useFinalLSTMLayer:
            model.add(layers.LSTM(lstm_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(l2Regularization)))
            model.add(layers.Dropout(dropout))

        # Bidirectional layers — must come while input is still 3D
        for _ in range(num_bidirectional_layers):
            model.add(layers.Bidirectional(layers.LSTM(
                bidirectional_lstm_units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2Regularization)
            )))
            model.add(layers.Dropout(dropout))

        # Final LSTM layer — collapse time dimension to (batch, features)
        model.add(layers.LSTM(lstm_units, return_sequences=False,
                            kernel_regularizer=regularizers.l2(l2Regularization)))
        model.add(layers.Dropout(dropout))

        # Dense output
        model.add(layers.Dense(digitsPerDraw * num_classes, activation=self.outputActivation))
        model.add(layers.Reshape((digitsPerDraw, num_classes)))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        if self.loadModelWeights:
            if os.path.exists(f"{model_path}.weights.h5"):
                print(f"Loading weights from {model_path}.weights.h5")
                model.load_weights(f"{model_path}.weights.h5")

        return model

    def train_model(self, model, train_data, train_labels, val_data, val_labels, model_name):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.earlyStopPatience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.reduceLearningRateFactor, patience=self.reduceLearningRatePatience)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, f"model_{model_name}_checkpoint.keras"), save_best_only=True)

        history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                            epochs=self.epochs, batch_size=self.batchSize, verbose=False, callbacks=[early_stopping, reduce_lr, checkpoint, SelectiveProgbarLogger(verbose=1, epoch_interval=int(50))])
        return history

    def run(self, name='euromillions', skipLastColumns=0, maxRows=0, skipRows=0, years_back=None):
        # Load and preprocess data
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(self.dataPath, skipLastColumns, maxRows=maxRows, skipRows=skipRows, years_back=years_back)

        model_path = os.path.join(self.modelPath, f"model_{name}.keras")
        checkpoint_path = os.path.join(self.modelPath, f"model_{name}_checkpoint.keras")

        # Use all numbers (raw data) instead of train_data. Because train_data is only 80 procent of all data 
        X, y = helpers.create_sequences(numbers, window_size=self.window_size)
        val_data_seq, val_labels_seq = helpers.create_sequences(numbers, window_size=self.window_size)

        #print("X shape: ", X.shape)  # (n_samples - 10, 10, 3)
        #print("y shape: ", y.shape)  # (n_samples - 10, 3)

        #print("val data shape: ", val_data_seq.shape)
        #print("val label shape: ", val_labels_seq.shape)

        y = np.array([to_categorical(draw, num_classes=num_classes) for draw in y])
        val_labels_seq = np.array([to_categorical(draw, num_classes=num_classes) for draw in val_labels_seq])
        
        model = self.create_model(max_value, num_classes=num_classes, model_path=model_path, digitsPerDraw=X.shape[2])

        # Train the model
        history = self.train_model(model, X, y, val_data_seq, val_labels_seq, model_name=name)

        # Predict numbers
        latest_raw_predictions = helpers.predict_numbers(model, numbers, window_size=self.predictionWindowSize)

        try:
            markov.build_markov_chain(numbers)
            markovChain = markov.getTransformationMatrix()
            lastDraw = numbers[len(numbers)-1]
            #print("last draw: ", lastDraw)
            markov_probs = self.get_markov_probs_for_last_draw(markovChain, lastDraw, num_classes)
            self.lstmMarkov = self.markovAlpha * latest_raw_predictions + (1 - self.markovAlpha) * markov_probs

        except Exception as e:
            print("Failed to build Markov Chain: ", e)

        # Plot training history
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.savefig(os.path.join(self.modelPath, f'model_{name}_performance.png'))

        # Save weights
        model.save_weights(f"{model_path}.weights.h5")
        model.save(model_path)

        # Remove checkpoint if exists
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return latest_raw_predictions, unique_labels

    # def doPrediction(self, modelPath, skipLastColumns, maxRows=0):
    #     """
    #     Do only a prediction. modelPath is the absolute path to the model
    #     """
    #     numbers = helpers.load_prediction_data(self.dataPath, skipLastColumns, maxRows=maxRows)

    #     model = load_model(modelPath, compile=True)

    #     # Predict numbers
    #     latest_raw_predictions = helpers.predict_numbers(model, numbers, window_size=self.predictionWindowSize)

    #     return latest_raw_predictions
    
    def get_markov_probs_for_last_draw(self, transition_matrix, last_draw, num_classes):
        markov_probs = np.zeros((len(last_draw), num_classes))

        for i, from_number in enumerate(last_draw):
            transitions = transition_matrix.get(from_number, {})
            for to_number, prob in transitions.items():
                markov_probs[i, to_number] = prob

        return markov_probs
    


# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":

    lstm_model = LSTMModel()

    name = 'pick3'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "lstm_model")

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-7-31.json")

    # Opening JSON file
    with open(sequenceToPredictFile, 'r') as openfile:
        sequenceToPredict = json.load(openfile)

    numbersLength = len(sequenceToPredict["realResult"])

    lstm_model.setLoadModelWeights(False)
    lstm_model.setModelPath(modelPath)
    lstm_model.setDataPath(dataPath)
    lstm_model.setBatchSize(64)
    lstm_model.setEpochs(10000)
    lstm_model.setNumberOfLSTMLayers(1)
    lstm_model.setNumberOfLstmUnits(64)
    lstm_model.setUseGRU(False)
    lstm_model.setNumberOfBidrectionalLayers(1)
    lstm_model.setNumberOfBidirectionalLstmUnits(64)
    lstm_model.setOptimizer("adam")
    lstm_model.setLearningRate(0.001)
    lstm_model.setDropout(0.3)
    lstm_model.setL2Regularization(0.0002)
    lstm_model.setEarlyStopPatience(10000)
    lstm_model.setReduceLearningRatePAience(500)
    lstm_model.setReducedLearningRateFactor(0.9)
    lstm_model.setWindowSize(50)
    lstm_model.setMarkovAlpha(0.6)
    lstm_model.setPredictionWindowSize(10)

    latest_raw_predictions, unique_labels = lstm_model.run(name, years_back=20)
    num_classes = len(unique_labels)

    latest_raw_predictions = latest_raw_predictions.tolist()

    print("Raw predictions: ", latest_raw_predictions)

    predicted_digits = np.argmax(latest_raw_predictions, axis=-1) 

    top3_indices = np.argsort(latest_raw_predictions, axis=-1)[:, -3:][:, ::-1]

    print(f"Position top prediction: {top3_indices[0].tolist()}")

    top3_indices_lstm_markov = np.argsort(lstm_model.getLstmMArkov(), axis=-1)[:, -3:][:, ::-1]

    print(f"lstm+markov prediction: {top3_indices_lstm_markov[0].tolist()}")

    print("Prediction: ", predicted_digits.tolist())
    print("Real result: ", sequenceToPredict["realResult"])

    

