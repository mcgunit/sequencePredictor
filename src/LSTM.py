# Import necessary libraries
import os, sys, json
import pandas as pd
import tensorflow as tf

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

# Custom Attention Layer (If you need attention layer for future use, you can re-enable this part)
class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.dense = layers.Dense(1, activation='tanh')

    def call(self, inputs):
        attention_scores = self.dense(inputs)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)
        weighted_inputs = inputs * attention_scores
        return tf.reduce_sum(weighted_inputs, axis=1)

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
        num_bidirectional_layers = 2
        embedding_output_dimension = 64
        lstm_units = 128
        bidirectional_lstm_units = 128
        dense_units = 128
        dropout = 0.3
        l2Regularization = 0.001

        model = models.Sequential()

        # Embedding layer
        model.add(layers.Embedding(input_dim=max_value + 1, output_dim=embedding_output_dimension))

        # 1D Convolutional layer
        #model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(None, embedding_output_dimension)))
        #model.add(layers.MaxPooling1D(pool_size=2))

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
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, f"model_{model_name}_checkpoint.keras"), save_best_only=True)

        history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                            epochs=self.epochs, batch_size=self.batchSize, callbacks=[early_stopping, reduce_lr, checkpoint])
        return history

    def run(self, data='euromillions', skipLastColumns=0):
        # Load and preprocess data
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes = helpers.load_data(self.dataPath, skipLastColumns)

        num_features = train_data.shape[1]

        model_path = os.path.join(self.modelPath, f"model_{data}.keras")
        checkpoint_path = os.path.join(self.modelPath, f"model_{data}_checkpoint.keras")

        if os.path.exists(model_path):
            model = load_model(model_path)
        elif os.path.exists(checkpoint_path):
            model = load_model(checkpoint_path)
        else:
            model = self.create_model(max_value, num_classes)

        # Train the model
        history = self.train_model(model, train_data, train_labels, val_data, val_labels, model_name=data)

        # Predict numbers
        predicted_numbers = helpers.predict_numbers(model, numbers)

        # Plot training history
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.savefig(os.path.join(self.modelPath, f'model_{data}_performance.png'))

        # Save model
        model.save(model_path)

        # Remove checkpoint if exists
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return predicted_numbers

    def doPrediction(self, modelPath, skipLastColumns):
        """
        Do only a prediction. modelPath is the absolute path to the model
        """
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes = helpers.load_data(self.dataPath, skipLastColumns)

        model = load_model(modelPath)

        # Predict numbers
        predicted_numbers = helpers.predict_numbers(model, numbers)

        return predicted_numbers

# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":
    lstm_model = LSTMModel()

    data = 'lotto'
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", data)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "lstm_model")

    lstm_model.setModelPath(modelPath)
    lstm_model.setDataPath(dataPath)
    lstm_model.setBatchSize(16)
    lstm_model.setEpochs(1000)

    predicted_numbers = lstm_model.run(data)

    print("Top six numbers: ", helpers.mostFrequentNumbers(predicted_numbers, numbers=6))

    helpers.print_predicted_numbers(predicted_numbers)

    # Opening JSON file
    sequenceToPredictFile = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "sequenceToPredict_{0}.json".format(data))
    with open(sequenceToPredictFile, 'r') as openfile:
        sequenceToPredict = json.load(openfile)

    best_match_index, best_match_sequence, matching_numbers = helpers.find_matching_numbers(sequenceToPredict["sequenceToPredict"], predicted_numbers)

    print("Best Matching Index: ", best_match_index)
    print("Best Matching Sequence: ", best_match_sequence)
    print("Matching Numbers: ", matching_numbers)
