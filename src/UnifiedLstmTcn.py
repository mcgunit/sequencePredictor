# UnifiedLstmTcn.py
# Real fusion (not ensemble-averaging like the old Unified.py prototype): a
# BiLSTM branch and a TCN branch process the same embedded input window and
# get concatenated *before* the shared attention/output head, so the network
# learns from both representations jointly instead of blending two
# independently-trained models' outputs after the fact.
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

helpers = Helpers()


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
# UnifiedLstmTcn Model Class
# ---------------------------
class UnifiedLstmTcnModel:
    def __init__(self):
        self.dataPath = ""
        self.modelPath = ""
        self.epochs = 1000
        self.batchSize = 16
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
        self.lstm_units = 64
        self.tcn_units = 64
        self.num_tcn_layers = 2
        self.loadModelWeights = True

    # ---------------------------
    # Setters
    # ---------------------------
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
    def setLoadModelWeights(self, value): self.loadModelWeights = value
    def setLstmUnits(self, value): self.lstm_units = value
    def setTcnUnits(self, value): self.tcn_units = value
    def setNumTcnLayers(self, value): self.num_tcn_layers = value

    # ---------------------------
    # Custom metrics
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
    def create_model(self, max_value, num_classes=50, model_path="", digitsPerDraw=3, specialColumnCount=0, specialNumClasses=0):
        embedding_dim = 8

        # Trailing specialColumnCount positions (stars/dream number/viking
        # number) have their own, smaller number range than the main
        # positions - see the second output head below.
        mainDigits = digitsPerDraw - specialColumnCount

        input_layer = layers.Input(shape=(self.window_size, digitsPerDraw))

        # Shared embedding: keeps window_size as the time dimension (unlike
        # LSTM.py, which flattens to window_size*digitsPerDraw) so the LSTM
        # and TCN branches below produce sequences of the same length and can
        # be concatenated along the feature axis.
        embedded = layers.Embedding(input_dim=num_classes, output_dim=embedding_dim)(input_layer)
        embedded = layers.Reshape((self.window_size, digitsPerDraw * embedding_dim))(embedded)

        lstm_branch = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True,
                        kernel_regularizer=regularizers.l2(self.l2Regularization))
        )(embedded)
        lstm_branch = layers.Dropout(self.dropout)(lstm_branch)

        tcn_branch = embedded
        for _ in range(self.num_tcn_layers):
            tcn_branch = TCN(
                nb_filters=self.tcn_units,
                kernel_size=3,
                return_sequences=True,
                dropout_rate=self.dropout
            )(tcn_branch)

        fused = layers.Concatenate(axis=-1)([lstm_branch, tcn_branch])

        # Attention blocks (over the fused representation)
        x = SelfAttentionBlock(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout)(fused)
        x = SelfAttentionBlock(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout)(x)

        x = layers.GlobalAveragePooling1D()(x)

        # Output head(s). Softmax was previously applied inside the Dense
        # layer before Reshape, which normalizes over the whole flattened
        # digitsPerDraw*num_classes vector instead of per position - fixed
        # here to Dense (linear) -> Reshape -> Softmax(axis=-1) so each
        # position's probabilities actually sum to 1 independently.
        main_logits = layers.Dense(mainDigits * num_classes,
                                    kernel_regularizer=regularizers.l2(self.l2Regularization))(x)
        main_logits = layers.Reshape((mainDigits, num_classes))(main_logits)
        main_output = layers.Softmax(axis=-1, name="main_output")(main_logits)

        if specialColumnCount > 0:
            special_logits = layers.Dense(specialColumnCount * specialNumClasses,
                                           kernel_regularizer=regularizers.l2(self.l2Regularization))(x)
            special_logits = layers.Reshape((specialColumnCount, specialNumClasses))(special_logits)
            special_output = layers.Softmax(axis=-1, name="special_output")(special_logits)

            model = models.Model(inputs=input_layer, outputs=[main_output, special_output])
            loss = {
                "main_output": losses.CategoricalCrossentropy(label_smoothing=self.labelSmoothing),
                "special_output": losses.CategoricalCrossentropy(label_smoothing=self.labelSmoothing),
            }
            per_output_metrics = [
                "accuracy",
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                self.digit_accuracy,
                self.any_digit_hit,
                self.full_draw_accuracy
            ]
            metrics = {"main_output": per_output_metrics, "special_output": per_output_metrics}
        else:
            model = models.Model(inputs=input_layer, outputs=main_output)
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
    # Training
    # ---------------------------
    def train_model(self, model, train_data, train_labels, val_data, val_labels, model_name):
        early_stopping = EarlyStopping(monitor="val_loss", patience=self.earlyStopPatience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=self.reduceLearningRateFactor, patience=self.reduceLearningRatePatience)
        checkpoint = ModelCheckpoint(os.path.join(self.modelPath, f"model_{model_name}_checkpoint.keras"), save_best_only=True)

        history = model.fit(train_data, train_labels,
                            validation_data=(val_data, val_labels),
                            epochs=self.epochs,
                            batch_size=self.batchSize,
                            verbose=False,
                            callbacks=[early_stopping, reduce_lr, checkpoint, SelectiveProgbarLogger(verbose=1, epoch_interval=50)])
        return history

    # ---------------------------
    # Run Training + Prediction
    # ---------------------------
    def run(self, name="pick3", skipLastColumns=0, maxRows=0, skipRows=0, years_back=None, strict_val=True, specialColumnCount=0):
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(
            self.dataPath, skipLastColumns, maxRows=maxRows, skipRows=skipRows, years_back=years_back
        )

        # Trailing specialColumnCount positions (stars/dream number/viking
        # number) get their own num_classes/label range - see create_model's
        # second output head.
        mainDigits = numbers.shape[1] - specialColumnCount
        special_num_classes = 0
        special_unique_labels = None
        if specialColumnCount > 0:
            special_unique_labels = helpers.get_unique_labels(self.dataPath, special=True)
            special_num_classes = len(special_unique_labels)

        # Normalize labels to 0-based (Embedding requires indices in
        # [0, num_classes) - matches LSTM.py's normalization, needed here
        # because unlike TCN.py this model embeds its input).
        min_label = np.min(unique_labels)
        numbers = numbers - min_label

        os.makedirs(self.modelPath, exist_ok=True)
        model_path = os.path.join(self.modelPath, f"model_{name}.keras")
        checkpoint_path = os.path.join(self.modelPath, f"model_{name}_checkpoint.keras")

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

        y_main = np.array([to_categorical(draw, num_classes=num_classes) for draw in y[:, :mainDigits]])
        y_val_main = np.array([to_categorical(draw, num_classes=num_classes) for draw in y_val[:, :mainDigits]])

        if specialColumnCount > 0:
            y_special = np.array([to_categorical(draw, num_classes=special_num_classes) for draw in y[:, mainDigits:]])
            y_val_special = np.array([to_categorical(draw, num_classes=special_num_classes) for draw in y_val[:, mainDigits:]])
            train_targets = {"main_output": y_main, "special_output": y_special}
            val_targets = {"main_output": y_val_main, "special_output": y_val_special}
        else:
            train_targets = y_main
            val_targets = y_val_main

        print("X shape: ", X.shape)
        print("y shape: ", y_main.shape)
        print("X_val shape: ", X_val.shape)
        print("y_val shape: ", y_val_main.shape)

        model = self.create_model(max_value, num_classes=num_classes, model_path=model_path, digitsPerDraw=X.shape[2],
                                   specialColumnCount=specialColumnCount, specialNumClasses=special_num_classes)
        history = self.train_model(model, X, train_targets, X_val, val_targets, model_name=name)

        # Best (not last) val_loss - EarlyStopping uses restore_best_weights=True,
        # so the epoch history ends on may not be the epoch the saved weights
        # came from. Read by HyperoptDeepLearning.py as a hyperopt signal.
        self.last_val_loss = float(min(history.history.get("val_loss", [float("inf")])))

        latest_raw_predictions = helpers.predict_numbers(model, numbers, window_size=self.predictionWindowSize)

        if specialColumnCount > 0:
            main_prediction, special_prediction = latest_raw_predictions
            latest_raw_predictions, combined_labels = helpers.combine_special_prediction(
                main_prediction, special_prediction, unique_labels, special_unique_labels)
            unique_labels = combined_labels

        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.savefig(os.path.join(self.modelPath, f"model_{name}_performance.png"))
        plt.close()

        model.save_weights(f"{model_path}.weights.h5")
        model.save(model_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # NOTE: must NOT re-sort here when specialColumnCount>0 - unique_labels
        # is combined_labels at that point, whose ordering (main labels then
        # special labels, each already sorted within their own segment) is
        # what makes each position's class index decode correctly. Sorting
        # the combined list globally would scramble that index->label mapping.
        unique_labels_sorted = unique_labels if specialColumnCount > 0 else sorted(int(v) for v in unique_labels)
        return latest_raw_predictions, unique_labels_sorted

    # ---------------------------
    # Prediction only
    # ---------------------------
    def doPrediction(self, modelPath, skipLastColumns, maxRows=0):
        numbers = helpers.load_prediction_data(self.dataPath, skipLastColumns, maxRows=maxRows)
        model = load_model(modelPath, compile=True)
        return helpers.predict_numbers(model, numbers, window_size=self.predictionWindowSize)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    model = UnifiedLstmTcnModel()

    name = "pick3"
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "unified_lstm_tcn_model")

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-8-3.json")

    with open(sequenceToPredictFile, "r") as openfile:
        sequenceToPredict = json.load(openfile)

    model.setLoadModelWeights(False)
    model.setModelPath(modelPath)
    model.setDataPath(dataPath)
    model.setBatchSize(64)
    model.setEpochs(2000)
    model.setLearningRate(0.001)
    model.setDropout(0.3)
    model.setL2Regularization(0.0005)
    model.setEarlyStopPatience(300)
    model.setReduceLearningRatePatience(15)
    model.setReducedLearningRateFactor(0.5)
    model.setWindowSize(20)
    model.setPredictionWindowSize(20)
    model.setLabelSmoothing(0.05)
    model.setNumHeads(4)
    model.setKeyDim(32)
    model.setLstmUnits(32)
    model.setTcnUnits(32)
    model.setNumTcnLayers(2)

    latest_raw_predictions, unique_labels = model.run(name, years_back=20, strict_val=True)

    latest_raw_predictions = latest_raw_predictions.tolist()
    predicted_digits = np.argmax(latest_raw_predictions, axis=-1)

    print("Prediction: ", predicted_digits.tolist())
    print("Real result: ", sequenceToPredict["realResult"])
