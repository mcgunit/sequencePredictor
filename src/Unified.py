# Unified.py
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
from tcn import TCN

# Dynamically adjust the import path for Helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Helpers import Helpers
from SelectiveProgbarLogger import SelectiveProgbarLogger
from Markov import Markov

helpers = Helpers()
markov = Markov()


# ---------------------------
# Self-Attention block
# ---------------------------
class SelfAttentionBlock(layers.Layer):
    def __init__(self, num_heads=4, key_dim=32, ffn_factor=4, dropout=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_factor = ffn_factor
        self.dropout = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = None

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.ffn = models.Sequential([
            layers.Dense(self.ffn_factor * d_model, activation="relu"),
            layers.Dense(d_model)
        ])

    def call(self, x, training=None):
        attn_out = self.mha(query=x, key=x, value=x, training=training)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + ffn_out)


# ---------------------------
# Unified Model Class
# ---------------------------
class UnifiedModel:
    def __init__(self):
        self.arch = "gru"  # "lstm", "gru", "tcn"
        # paths & training config
        self.dataPath = ""
        self.modelPath = ""
        self.epochs = 1000
        self.batchSize = 32
        self.dropout = 0.3
        self.l2Regularization = 0.0005
        self.earlyStopPatience = 20
        self.reduceLearningRatePatience = 5
        self.reduceLearningRateFactor = 0.5
        self.learning_rate = 0.001
        self.window_size = 20
        self.predictionWindowSize = 20
        self.labelSmoothing = 0.05
        self.num_heads = 4
        self.key_dim = 32
        self.markovAlpha = 0.5
        # model sizes
        self.lstm_units = 64
        self.gru_units = 64
        self.tcn_units = 64
        self.num_tcn_layers = 2
        # flags
        self.loadModelWeights = False
        self.hybridMarkov = None

    # ---------------------------
    # Setters (added)
    # ---------------------------
    def setArch(self, arch): self.arch = arch
    def setDataPath(self, dataPath): self.dataPath = dataPath
    def setModelPath(self, modelPath): self.modelPath = modelPath
    def setEpochs(self, epochs): self.epochs = epochs
    def setBatchSize(self, batchSize): self.batchSize = batchSize
    def setDropout(self, dropout): self.dropout = dropout
    def setL2Regularization(self, value): self.l2Regularization = value
    def setEarlyStopPatience(self, value): self.earlyStopPatience = value
    def setReduceLearningRatePatience(self, value): self.reduceLearningRatePatience = value
    def setReducedLearningRateFactor(self, value): self.reduceLearningRateFactor = value
    def setLearningRate(self, value): self.learning_rate = value
    def setWindowSize(self, value): self.window_size = value
    def setPredictionWindowSize(self, value): self.predictionWindowSize = value
    def setLabelSmoothing(self, value): self.labelSmoothing = value
    def setNumHeads(self, value): self.num_heads = value
    def setKeyDim(self, value): self.key_dim = value
    def setMarkovAlpha(self, value): self.markovAlpha = value
    def setLoadModelWeights(self, value): self.loadModelWeights = value
    def setLstmUnits(self, value): self.lstm_units = value
    def setGruUnits(self, value): self.gru_units = value
    def setTcnUnits(self, value): self.tcn_units = value
    def setNumTcnLayers(self, value): self.num_tcn_layers = value

    # ---------------------------
    # Metrics
    # ---------------------------
    def digit_accuracy(self, y_true, y_pred):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        matches = tf.cast(tf.equal(y_true_labels, y_pred_labels), tf.float32)
        return tf.reduce_mean(matches)

    def any_digit_hit(self, y_true, y_pred):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        correct_any = tf.reduce_any(tf.equal(y_true_labels, y_pred_labels), axis=-1)
        return tf.reduce_mean(tf.cast(correct_any, tf.float32))

    def full_draw_accuracy(self, y_true, y_pred):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        correct_all = tf.reduce_all(tf.equal(y_true_labels, y_pred_labels), axis=-1)
        return tf.reduce_mean(tf.cast(correct_all, tf.float32))

    # ---------------------------
    # Model creation
    # ---------------------------
    def create_model(self, max_value, num_classes=50, model_path="", digitsPerDraw=3):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.window_size, digitsPerDraw)))

        if self.arch == "lstm":
            # simple stacked LSTM block (one layer here; you can increase)
            model.add(layers.LSTM(self.lstm_units, return_sequences=True,
                                  kernel_regularizer=regularizers.l2(self.l2Regularization)))
            model.add(layers.Dropout(self.dropout))

        elif self.arch == "gru":
            model.add(layers.GRU(self.gru_units, return_sequences=True,
                                 kernel_regularizer=regularizers.l2(self.l2Regularization)))
            model.add(layers.Dropout(self.dropout))

        elif self.arch == "tcn":
            for _ in range(self.num_tcn_layers):
                model.add(TCN(nb_filters=self.tcn_units,
                              kernel_size=3,
                              return_sequences=True,
                              dropout_rate=self.dropout))

        # Attention blocks (shared across architectures)
        model.add(SelfAttentionBlock(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout))
        model.add(SelfAttentionBlock(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout))

        # Pooling + final dense (per-digit softmax)
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(digitsPerDraw * num_classes, activation="softmax",
                               kernel_regularizer=regularizers.l2(self.l2Regularization)))
        model.add(layers.Reshape((digitsPerDraw, num_classes)))

        # Compile the model
        loss = losses.CategoricalCrossentropy(label_smoothing=self.labelSmoothing)
        metrics = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            self.digit_accuracy,
            self.any_digit_hit,
            self.full_draw_accuracy
        ]
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        if self.loadModelWeights and model_path and os.path.exists(f"{model_path}.weights.h5"):
            print(f"Loading weights from {model_path}.weights.h5")
            model.load_weights(f"{model_path}.weights.h5")

        return model

    # ---------------------------
    # Training + Ensemble Predict
    # ---------------------------
    def train_and_predict(self, name, years_back=1, strict_val=True):
        # load_data returns: train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(
            self.dataPath, 0, maxRows=0, skipRows=0, years_back=years_back
        )

        # prepare sequences (time-based split like before)
        n = len(numbers)
        split_idx = int(n * 0.8)

        if strict_val:
            train_numbers = numbers[:split_idx]
            val_numbers = numbers[split_idx:]
            X, y = helpers.create_sequences(train_numbers, window_size=self.window_size)
            X_val, y_val = helpers.create_sequences(val_numbers, window_size=self.window_size)
        else:
            X, y = helpers.create_sequences(numbers[:split_idx], window_size=self.window_size)
            start = max(0, split_idx - self.window_size)
            X_val, y_val = helpers.create_sequences(numbers[start:], window_size=self.window_size)
            keep = np.where(np.arange(start + self.window_size, start + self.window_size + len(y_val)) >= split_idx)[0]
            X_val, y_val = X_val[keep], y_val[keep]

        # One-hot labels
        y = np.array([to_categorical(draw, num_classes=num_classes) for draw in y])
        y_val = np.array([to_categorical(draw, num_classes=num_classes) for draw in y_val])

        # ---------------------------
        # Train all 3 architectures
        # ---------------------------
        archs = ["lstm", "gru", "tcn"]
        preds = []

        for arch in archs:
            print(f"\n--- Training {arch.upper()} model ---")
            self.arch = arch  # set arch dynamically

            model_path = os.path.join(self.modelPath, f"model_{name}_{arch}.keras")
            checkpoint_path = os.path.join(self.modelPath, f"model_{name}_{arch}_checkpoint.keras")

            model = self.create_model(max_value, num_classes=num_classes, model_path=model_path, digitsPerDraw=X.shape[2])

            # Ensure modelPath exists
            os.makedirs(self.modelPath, exist_ok=True)

            history = model.fit(
                X, y,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batchSize,
                verbose=False,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=self.earlyStopPatience, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=self.reduceLearningRateFactor, patience=self.reduceLearningRatePatience),
                    ModelCheckpoint(checkpoint_path, save_best_only=True),
                    SelectiveProgbarLogger(verbose=1, epoch_interval=50)
                ]
            )

            # Save performance plot, weights and model
            try:
                pd.DataFrame(history.history).plot(figsize=(8, 5))
                plt.savefig(os.path.join(self.modelPath, f"model_{name}_{arch}_performance.png"))
                plt.close()
            except Exception:
                pass

            model.save_weights(f"{model_path}.weights.h5")
            model.save(model_path)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            # Get predictions from this model
            pred = helpers.predict_numbers(model, numbers, window_size=self.predictionWindowSize)
            preds.append(pred)

        # ---------------------------
        # Combine predictions (average ensemble)
        # ---------------------------
        combined_pred = np.mean(preds, axis=0)

        return combined_pred, unique_labels


# ---------------------------
# Main Comparison
# ---------------------------
if __name__ == "__main__":
    name = "pick3"
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "compare_models")

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-8-3.json")
    with open(sequenceToPredictFile, "r") as openfile:
        sequenceToPredict = json.load(openfile)

    # initialize unified ensemble model
    model = UnifiedModel()
    model.setModelPath(modelPath)
    model.setDataPath(dataPath)
    model.setBatchSize(8)
    model.setEpochs(1000)   # adjust for quicker runs if needed
    model.setWindowSize(30)
    model.setPredictionWindowSize(30)
    model.setLearningRate(0.0001)
    model.setDropout(0.3)
    model.setL2Regularization(0.0005)
    model.setEarlyStopPatience(50)
    model.setReduceLearningRatePatience(15)
    model.setReducedLearningRateFactor(0.5)
    model.setLabelSmoothing(0.05)
    model.setNumHeads(4)
    model.setKeyDim(32)
    model.setLoadModelWeights(False)
    model.setLstmUnits(16)
    model.setNumTcnLayers(1)
    model.setTcnUnits(16)
    model.setGruUnits(16)

    # train all three models and get combined predictions
    raw_pred, unique_labels = model.train_and_predict(name, years_back=20, strict_val=True)

    # convert probabilities into predictions
    predicted_digits = np.argmax(raw_pred, axis=-1)
    top3_indices = np.argsort(raw_pred, axis=-1)[:, -3:][:, ::-1]

    print("\n=== Ensemble Prediction ===")
    print("Real result:", sequenceToPredict["realResult"])
    print(" Predicted digits:", predicted_digits.tolist())
    print(" Top3 per digit:", top3_indices[0].tolist())


