#https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
import os
import numpy as np
import pandas as pd
import pmdarima as pm
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from statsmodels.tsa.arima.model import ARIMA
from Helpers import Helpers

helpers = Helpers()

class LSTM_ARIMA_Model:
    def __init__(self):
        self.dataPath = ""
        self.modelPath = ""
        self.epochs = 50
        self.batchSize = 16
        self.order = None  # Auto-detect best ARIMA order
        self.lookback = 30  # LSTM lookback window
        self.lstm_units = 16

    def setDataPath(self, dataPath):
        self.dataPath = dataPath

    def setModelPath(self, modelPath):
        self.modelPath = modelPath

    def setEpochs(self, epochs):
        self.epochs = epochs

    def setBatchSize(self, batchSize):
        self.batchSize = batchSize

    def find_best_arima_order(self, series):
        model = pm.auto_arima(series, 
                              start_p=0, max_p=5, 
                              start_q=0, max_q=5, 
                              d=None, seasonal=False, 
                              stepwise=True, trace=True)
        return model.order

    def apply_arima(self, data):
        arima_predictions = []
        residuals = []
        
        for i in range(data.shape[1]):
            series = data[:, i]
            
            if self.order is None:
                self.order = self.find_best_arima_order(series)
            
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

    def create_lstm_model(self, input_shape, numbersToPredict=20):
        model = Sequential([
            Input(input_shape),
            LSTM(self.lstm_units, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(self.lstm_units, activation='relu'),
            Dense(numbersToPredict)
        ])
        model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
        return model

    def train_lstm(self, name, train_data, train_labels, numbersToPredict=20):
        model = self.create_lstm_model((train_data.shape[1], train_data.shape[2]), numbersToPredict=numbersToPredict)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
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
        final_prediction = np.array([min(unique_labels, key=lambda x: abs(x - num)) for num in final_prediction])
        return final_prediction

    def run(self, name='hybrid_model'):
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(self.dataPath)
        arima_predictions, residuals = self.apply_arima(numbers)
        print("Arima predictions: ", arima_predictions)
        X_train, y_train = self.prepare_lstm_data(residuals)
        model = self.train_lstm(name, X_train, y_train, numbersToPredict=len(numbers[0]))
        next_sequence = self.predict_next_sequence(model, residuals, arima_predictions, unique_labels)
        print("Predicted Next Sequence:", next_sequence)
        return next_sequence



if __name__ == "__main__":

    name = 'keno'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "lstm_arima_model")

    hybrid_model = LSTM_ARIMA_Model()

    hybrid_model.setModelPath(modelPath)
    hybrid_model.setDataPath(dataPath)
    hybrid_model.setBatchSize(16)
    hybrid_model.setEpochs(100)

    predicted_sequence = hybrid_model.run(name=name)
