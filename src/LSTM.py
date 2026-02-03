# Import necessary libraries
import os, sys, json
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from keras import layers, regularizers, models, optimizers, losses
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

class SelfAttentionBlock(layers.Layer):
    def __init__(self, num_heads=4, key_dim=32, ffn_factor=4, dropout=0.0):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_factor = ffn_factor
        self.dropout = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        d_model = input_shape[-1]  # dynamically match input dim (e.g. 128)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ffn_factor * d_model, activation='relu'),
            layers.Dense(d_model),   # project back to same dim
        ])

    def call(self, x, training=None):
        attn_out = self.mha(query=x, key=x, value=x, training=training)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + ffn_out)
        return x


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
        self.outputActivation = "softmax"
        self.optimizer_type = 'adam'
        self.learning_rate = 0.005
        self.loadModelWeights = True
        self.window_size = 10
        self.predictionWindowSize = 20
        self.labelSmoothing = 0.1
        self.num_heads = 2
        self.key_dim = 16

        # LSTM + Markov
        self.markovAlpha = 0.5
        self.lstmMarkov = None

    """
        Setters
    """

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

    def setLabelSmoothing(self, value):
        self.labelSmoothing = value

    def setNumHeads(self, value):
        self.num_heads = value

    def setKeyDim(self, value):
        self.key_dim = value


    """
        Getters
    """

    def getLstmMArkov(self):
        return self.lstmMarkov
    

    """
        Custom Metric functions
    """

    # 1. Per-digit accuracy
    # on average % of digits guessed correctly.
    def digit_accuracy(self, y_true, y_pred):
        # y_true, y_pred: (batch, 3, 10)
        y_true_labels = tf.argmax(y_true, axis=-1)   # (batch, 3)
        y_pred_labels = tf.argmax(y_pred, axis=-1)   # (batch, 3)
        matches = tf.cast(tf.equal(y_true_labels, y_pred_labels), tf.float32)
        return tf.reduce_mean(matches)  # average over digits and batch

    # 2. Any-digit hit rate
    # in % of draws, it guessed at least one digit right.
    def any_digit_hit(self, y_true, y_pred):
        y_true_labels = tf.argmax(y_true, axis=-1)   # (batch, 3)
        y_pred_labels = tf.argmax(y_pred, axis=-1)   # (batch, 3)
        # For each draw, check if ANY of the 3 digits are correct
        correct_any = tf.reduce_any(tf.equal(y_true_labels, y_pred_labels), axis=-1)
        return tf.reduce_mean(tf.cast(correct_any, tf.float32))  # average over batch

    # 3. Full-draw accuracy (redundant with categorical_accuracy, but explicit)
    # got the exact 3-digit combo % of the time (better than random 0.1%).
    def full_draw_accuracy(self, y_true, y_pred):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        correct_all = tf.reduce_all(tf.equal(y_true_labels, y_pred_labels), axis=-1)
        return tf.reduce_mean(tf.cast(correct_all, tf.float32))
    


    def create_model(self, max_value, num_classes=10, model_path="", digitsPerDraw=3):
        # --- Parameters ---
        # A smaller embedding dim is usually fine for 10 digits
        embedding_dim = 8  
        lstm_units = self.lstm_units     # Enough capacity without massive overfitting
        dropout = self.dropout       # Higher dropout due to noisy data
        learning_rate = self.learning_rate

        # --- Architecture ---
        # Input Shape: (Window_Size, 3 Digits)
        input_layer = layers.Input(shape=(self.window_size, digitsPerDraw))

        # 1. Flatten to treating the stream as a single sequence of symbols
        #    If window is 5 and digits is 3, we flatten to a sequence of length 15.
        #    This often helps the LSTM see the relationship between Digit 1 and Digit 2 
        #    within the same draw, not just across time.
        x = layers.Reshape((self.window_size * digitsPerDraw,))(input_layer)

        # 2. Embedding Layer
        #    Converts integer 0-9 into a vector of size 16. 
        #    This removes the "magnitude" bias.
        x = layers.Embedding(input_dim=num_classes, output_dim=embedding_dim)(x)

        # 3. Bidirectional LSTM
        #    We use one strong layer. 'return_sequences=False' because we only 
        #    care about the final prediction after seeing the whole window.
        x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
        x = layers.Dropout(dropout)(x)

        # 4. Dense Intermediary (Optional "Thinking" layer)
        x = layers.Dense(lstm_units, activation='relu')(x)
        
        # 5. Output Head (The Logic We Fixed)
        #    Linear projection -> Reshape -> Independent Softmax
        x = layers.Dense(digitsPerDraw * num_classes)(x)
        x = layers.Reshape((digitsPerDraw, num_classes))(x)
        output_layer = layers.Softmax(axis=-1)(x)

        model = models.Model(inputs=input_layer, outputs=output_layer)

        # --- Compilation ---
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        loss = losses.CategoricalCrossentropy() # from_logits=False is default, which matches Softmax output
        
        metrics = [
            'accuracy', 
            # We can keep your custom metrics here
            self.digit_accuracy,
            self.any_digit_hit
        ]
        
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        # Load weights logic...
        if self.loadModelWeights:
            try:
                if os.path.exists(f"{model_path}.weights.h5"):
                    print(f"Loading weights from {model_path}.weights.h5")
                    model.load_weights(f"{model_path}.weights.h5")
            except Exception:
                pass

        return model
    
    def train_model(self, model, train_data, train_labels, val_data, val_labels, model_name):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.earlyStopPatience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.reduceLearningRateFactor, patience=self.reduceLearningRatePatience)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, f"model_{model_name}_checkpoint.keras"), save_best_only=True)

        history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                            epochs=self.epochs, batch_size=self.batchSize, verbose=False, callbacks=[early_stopping, reduce_lr, checkpoint, SelectiveProgbarLogger(verbose=1, epoch_interval=int(50))])
        return history

    def run(self, name='pick3', skipLastColumns=0, maxRows=0, skipRows=0, years_back=None, strict_val=False):
        # Load and preprocess data
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(
            self.dataPath, skipLastColumns, maxRows=maxRows, skipRows=skipRows, years_back=years_back
        )

        print("Numclasses: ", num_classes)

        # -------------------------------------------------------
        # Normalize labels to 0-based (works for lotto + pick3)
        # -------------------------------------------------------
        min_label = np.min(unique_labels)       # pick3 => 0, lotto => 1
        numbers = numbers - min_label           # normalize
        # Now everything is 0..num_classes-1 internally
        # -------------------------------------------------------

        model_path = os.path.join(self.modelPath, f"model_{name}.keras")
        checkpoint_path = os.path.join(self.modelPath, f"model_{name}_checkpoint.keras")

        # -------- TIME-BASED SPLIT --------
        n = len(numbers)
        split_idx = int(n * 0.8)  # 80/20 split by time

        # Strict validation (no training history inside val windows)
        if strict_val:
            train_numbers = numbers[:split_idx]
            val_numbers   = numbers[split_idx:]            # validation windows come only from val period
            X, y = helpers.create_sequences(train_numbers, window_size=self.window_size)
            X_val, y_val = helpers.create_sequences(val_numbers, window_size=self.window_size)
        else:
            # Forecast-style validation (first val input uses training history)
            X, y = helpers.create_sequences(numbers[:split_idx], window_size=self.window_size)
            start = max(0, split_idx - self.window_size)
            X_val, y_val = helpers.create_sequences(numbers[start:], window_size=self.window_size)

            keep = np.where(np.arange(start + self.window_size, start + self.window_size + len(y_val)) >= split_idx)[0]
            X_val, y_val = X_val[keep], y_val[keep]

        # One-hot labels to shape (batch, digitsPerDraw, num_classes)
        y = np.array([to_categorical(draw, num_classes=num_classes) for draw in y])
        y_val = np.array([to_categorical(draw, num_classes=num_classes) for draw in y_val])

        print("X shape: ", X.shape)
        print("y shape: ", y.shape)
        print("X_val shape: ", X_val.shape)
        print("y_val shape: ", y_val.shape)

        # Build & train
        model = self.create_model(max_value, num_classes=num_classes, model_path=model_path, digitsPerDraw=X.shape[2])
        history = self.train_model(model, X, y, X_val, y_val, model_name=name)

        # Predict next draw using all history available
        latest_raw_predictions = helpers.predict_numbers(model, numbers, window_size=self.predictionWindowSize)

        # Optional Markov blend
        try:
            markov.build_markov_chain(numbers)
            markovChain = markov.getTransformationMatrix()
            lastDraw = numbers[-1]
            markov_probs = self.get_markov_probs_for_last_draw(markovChain, lastDraw, num_classes)
            self.lstmMarkov = self.markovAlpha * latest_raw_predictions + (1 - self.markovAlpha) * markov_probs
        except Exception as e:
            print("Failed to build Markov Chain: ", e)

        # -------------------------------------------------------
        # DENORMALIZE predictions back to original labels
        # -------------------------------------------------------
        # latest_raw_predictions shape: (digits, num_classes)
        # Convert output probabilities for each digit

        # convert indices (0..classes-1 → real labels)
        denorm_predictions = np.zeros_like(latest_raw_predictions)
        for digit in range(latest_raw_predictions.shape[0]):
            for cls in range(latest_raw_predictions.shape[1]):
                # same probability, but class shifted back
                denorm_predictions[digit, cls] = latest_raw_predictions[digit, cls]

        # BUT: your main script uses argmax → so we only need to shift indices
        # easiest:
        unique_labels_sorted = sorted(unique_labels)
        unique_labels_sorted = [int(v) for v in unique_labels_sorted]
        # unique_labels_sorted[i] gives original label for class i

        # Example usage later:
        # predicted_digits = unique_labels_sorted[np.argmax(latest_raw_predictions[digit])]
        # -------------------------------------------------------

        # Plot, save, cleanup
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.savefig(os.path.join(self.modelPath, f'model_{name}_performance.png'))

        model.save_weights(f"{model_path}.weights.h5")
        model.save(model_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # Return normalized predictions + the actual labels map
        return latest_raw_predictions, unique_labels_sorted


    def doPrediction(self, modelPath, skipLastColumns, maxRows=0):
        """
        Do only a prediction. modelPath is the absolute path to the model
        """
        numbers = helpers.load_prediction_data(self.dataPath, skipLastColumns, maxRows=maxRows)

        model = load_model(modelPath, compile=True)

        # Predict numbers
        latest_raw_predictions = helpers.predict_numbers(model, numbers, window_size=self.predictionWindowSize)

        return latest_raw_predictions
    
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
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-08-02.json")

    # Opening JSON file
    with open(sequenceToPredictFile, 'r') as openfile:
        sequenceToPredict = json.load(openfile)

    numbersLength = len(sequenceToPredict["realResult"])

    lstm_model.setLoadModelWeights(False)
    lstm_model.setModelPath(modelPath)
    lstm_model.setDataPath(dataPath)
    lstm_model.setBatchSize(4)
    lstm_model.setEpochs(500)
    lstm_model.setNumberOfLSTMLayers(1)
    lstm_model.setNumberOfLstmUnits(32)
    lstm_model.setNumberOfBidrectionalLayers(1)
    lstm_model.setNumberOfBidirectionalLstmUnits(16)
    lstm_model.setOptimizer("adagrad")
    lstm_model.setLearningRate(0.0005)
    lstm_model.setDropout(0.5) # 0.2 - 0.5
    lstm_model.setL2Regularization(0.001) #0.001 - 0.00005
    lstm_model.setUseFinalLSTMLayer(False)
    lstm_model.setEarlyStopPatience(1000)
    lstm_model.setReduceLearningRatePAience(30)
    lstm_model.setReducedLearningRateFactor(0.9)
    lstm_model.setWindowSize(7)
    lstm_model.setMarkovAlpha(0.01)
    lstm_model.setPredictionWindowSize(lstm_model.window_size)
    lstm_model.setLabelSmoothing(0.01)
    lstm_model.setNumHeads(2)
    lstm_model.setKeyDim(64)

    latest_raw_predictions, unique_labels = lstm_model.run(name, years_back=10, strict_val=False)
    num_classes = len(unique_labels)

    #print("Labels: ", unique_labels)
    latest_raw_predictions = latest_raw_predictions.tolist()

    predicted_indices = np.argmax(latest_raw_predictions, axis=-1)
    predicted_digits = [int(unique_labels[i]) for i in predicted_indices]

    # top7_indices = np.argsort(latest_raw_predictions, axis=-1)[:, -numbersLength:][:, ::-1]
    # top7_labels = [[int(unique_labels[i]) for i in row] for row in top7_indices]
    # #print(f"Position top prediction: {top7_labels[0]}")

    # lstm_markov = lstm_model.getLstmMArkov()
    # top_indices_markov = np.argsort(lstm_markov, axis=-1)[:, -numbersLength:][:, ::-1]
    # top_labels_markov = [[int(unique_labels[i]) for i in row] for row in top_indices_markov]
    # #print(f"lstm+markov prediction: {top_labels_markov[0]}")

    matches = set(predicted_digits) & set(sequenceToPredict["realResult"])


    print("Prediction: ", predicted_digits)
    print("Real result: ", sequenceToPredict["realResult"])
    print("Numbers that matches: ", matches)


    

