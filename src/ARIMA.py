#https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class LSTM_ARIMA_Model:
    def __init__(self, dataPath, modelPath, order=(5,1,0), lstm_units=16, epochs=50, batch_size=16, lookback=10):
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.order = order  # ARIMA order (p,d,q)
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.lookback = lookback  # Number of past steps used for LSTM prediction
        self.scaler = MinMaxScaler(feature_range=(0,1))

    def load_data(self):
        """
        Load and preprocess the time-series sequence data.
        """
        data = pd.read_csv(self.dataPath)
        self.data = data[['Number1', 'Number2', 'Number3']].values  # Adjust columns as needed

    def apply_arima(self):
        """
        Fit ARIMA model to capture trends and seasonality for each sequence.
        """
        self.arima_predictions = []
        self.residuals = []
        
        for i in range(self.data.shape[1]):  # Loop over each number sequence
            series = self.data[:, i]
            model = ARIMA(series, order=self.order)
            model_fit = model.fit()
            pred = model_fit.predict(start=0, end=len(series)-1)
            self.arima_predictions.append(pred)

            # Compute residuals (errors)
            self.residuals.append(series - pred)

        self.arima_predictions = np.array(self.arima_predictions).T  # Convert back to original shape
        self.residuals = np.array(self.residuals).T

    def prepare_lstm_data(self):
        """
        Prepare LSTM training data using ARIMA residuals.
        """
        dataX, dataY = [], []
        for i in range(len(self.residuals) - self.lookback):
            dataX.append(self.residuals[i:i + self.lookback, :])
            dataY.append(self.residuals[i + self.lookback, :])
        
        return np.array(dataX), np.array(dataY)

    def create_lstm_model(self, input_shape):
        """
        Build and compile an LSTM model for residual prediction.
        """
        model = Sequential([
            LSTM(self.lstm_units, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.lstm_units, activation='relu'),
            Dense(3)  # Predicting residuals for 3 numbers
        ])
        model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
        return model

    def train_lstm(self):
        """
        Train the LSTM model using ARIMA residuals.
        """
        X_train, y_train = self.prepare_lstm_data()
        self.lstm_model = self.create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5)

        self.lstm_model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, 
                            verbose=1, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

    def predict_next_sequence(self):
        """
        Predict next sequence using LSTM for residuals, then combine with ARIMA forecast.
        """
        last_residuals = self.residuals[-self.lookback:]
        last_residuals = last_residuals.reshape(1, last_residuals.shape[0], last_residuals.shape[1])
        
        lstm_residual_pred = self.lstm_model.predict(last_residuals)[0]
        arima_pred_next = self.arima_predictions[-1]  # Last ARIMA prediction
        
        final_prediction = arima_pred_next + lstm_residual_pred  # Combine ARIMA and LSTM outputs
        return final_prediction

    def run(self):
        """
        Run the full pipeline: Load data, apply ARIMA, train LSTM, and predict next sequence.
        """
        self.load_data()
        self.apply_arima()
        self.train_lstm()
        next_sequence = self.predict_next_sequence()
        print("Predicted Next Sequence:", next_sequence)
        return next_sequence

if __name__ == "__main__":
    # Usage Example
    dataPath = "your_data.csv"  # Replace with actual path
    modelPath = "your_model_path"

    hybrid_model = LSTM_ARIMA_Model(dataPath, modelPath)
    predicted_sequence = hybrid_model.run()
