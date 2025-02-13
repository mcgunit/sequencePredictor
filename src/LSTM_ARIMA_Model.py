#https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
import os, sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from statsmodels.tsa.arima.model import ARIMA

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

class LSTM_ARIMA_Model:
    def __init__(self):
        self.dataPath = ""
        self.modelPath = ""
        self.epochs = 50
        self.batchSize = 8
        self.order = (5,1,0)  # (5,1,0) Default ARIMA order test: (3,1,0), and (2,1,2)
        self.lookback = 10  # LSTM lookback window
        self.lstm_units = 8
        self.dense_units = 8

    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    def setModelPath(self, modelPath):
        self.modelPath = modelPath

    def setEpochs(self, epochs):
        self.epochs = epochs

    def setBatchSize(self, batchSize):
        self.batchSize = batchSize


    def apply_arima(self, data):
        arima_predictions = []
        residuals = []
        
        print("Making arima raw predictions")

        for i in range(data.shape[1]):
            series = data[:, i]
            model = ARIMA(series, order=self.order)
            model_fit = model.fit()
            pred = model_fit.predict(start=0, end=len(series)-1)
            arima_predictions.append(pred)
            residuals.append(series - pred)

        return np.array(arima_predictions).T, np.array(residuals).T

    def prepare_lstm_data(self, residuals):
        X, y = [], []
        for i in range(len(residuals) - self.lookback):
            X.append(residuals[i:i + self.lookback, :])
            y.append(residuals[i + self.lookback, :])
        return np.array(X), np.array(y)

    def create_lstm_model(self, name, input_shape, numbersToPredict=20):
        print("Creating lstm model for arima")
        model = Sequential([
            Input(input_shape),
            LSTM(self.lstm_units, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(self.lstm_units, activation='relu'),
            Dense(numbersToPredict)  # Predicting residuals for numbersToPredict
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse') # loss: mse, mae, huber_loss

        if os.path.exists(os.path.join(self.modelPath, f"model_{name}.keras")):
            model.load_weights(os.path.join(self.modelPath, f"model_{name}.keras"))

        return model

    def train_lstm(self, name, train_data, train_labels, numbersToPredict=20):
        model = self.create_lstm_model(name, (train_data.shape[1], train_data.shape[2]), numbersToPredict=numbersToPredict)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, f"model_{name}_checkpoint.keras"), save_best_only=True)
        
        model.fit(train_data, train_labels, epochs=self.epochs, batch_size=self.batchSize, validation_split=0.2, callbacks=[early_stopping, reduce_lr, checkpoint])
        model.save(os.path.join(self.modelPath, f"model_{name}.keras"))
        return model
    
    def predict_next_sequence(self, model, residuals, arima_predictions, unique_labels):
        last_residuals = residuals[-self.lookback:].reshape(1, self.lookback, residuals.shape[1])
        lstm_residual_pred = model.predict(last_residuals)[0]
        arima_pred_next = arima_predictions[-1]
        
        final_prediction = np.round(arima_pred_next + lstm_residual_pred).astype(int)

        # Ensure predictions are within unique_labels
        final_prediction = np.array([min(unique_labels, key=lambda x: abs(x - num)) for num in final_prediction])
        return final_prediction

    def run(self, name):
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(self.dataPath)
        arima_predictions, residuals = self.apply_arima(numbers)
        print("Arima predictions: ", len(arima_predictions))
        X_train, y_train = self.prepare_lstm_data(residuals)
        model = self.train_lstm(name, X_train, y_train, numbersToPredict=len(numbers[0]))
        next_sequence = self.predict_next_sequence(model, residuals, arima_predictions, unique_labels)
        #print("Predicted Next Sequence:", next_sequence)
        return next_sequence.tolist()


if __name__ == "__main__":

    name = 'keno'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "lstm_arima_model")

    hybrid_model = LSTM_ARIMA_Model()

    hybrid_model.setModelPath(modelPath)
    hybrid_model.setDataPath(dataPath)
    hybrid_model.setBatchSize(8)
    hybrid_model.setEpochs(1000)

    predicted_sequence = hybrid_model.run(name)

    print("Predicted Sequence:", predicted_sequence)
