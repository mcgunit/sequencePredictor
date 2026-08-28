# Import necessary libraries
import os, sys, json
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from keras import layers, regularizers, models, optimizers, losses
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN
from keras.utils import to_categorical

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
# Sinusoidal positional encoding
# ---------------------------
class SinusoidalPositionalEncoding(layers.Layer):
    # Sinusoidal (fixed) rather than learned: it carries no trainable
    # weights, so it never has to be fingerprinted separately - the saved
    # weights stay valid for any run whose window_size/d_model (already in
    # the fingerprint) match. Without SOME positional signal, self-attention
    # plus GlobalAveragePooling is order-blind and the model couldn't tell
    # the most recent draw from the oldest one in the window.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = None

    def build(self, input_shape):
        seq_len = int(input_shape[1])
        d_model = int(input_shape[2])
        positions = np.arange(seq_len)[:, np.newaxis]
        # Even indices get sin, odd get cos, with the classic 10000^(2i/d)
        # geometric wavelength progression (Vaswani et al.).
        div_term = np.power(10000.0, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angles = positions / div_term
        encoding = np.zeros((seq_len, d_model), dtype=np.float32)
        encoding[:, 0::2] = np.sin(angles[:, 0::2])
        encoding[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.constant(encoding[np.newaxis, :, :])

    def call(self, x):
        return x + self.pos_encoding


# ---------------------------
# Pre-LN Transformer encoder block
# ---------------------------
class TransformerEncoderBlock(layers.Layer):
    # Pre-LN (normalize BEFORE attention/FFN, residual around both) instead
    # of TCN.py SelfAttentionBlock's post-LN: pre-LN keeps gradient scale
    # stable without a warmup schedule, which matters here because training
    # runs with a plain Adam + ReduceLROnPlateau setup shared across games.
    def __init__(self, num_heads=4, key_dim=32, ffn_factor=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ffn_factor = ffn_factor
        self.dropout_rate = dropout
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.ffn = None

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.ffn = models.Sequential([
            layers.Dense(self.ffn_factor * d_model, activation="relu"),
            layers.Dense(d_model)
        ])

    def call(self, x, training=None):
        attn_in = self.norm1(x)
        attn_out = self.mha(query=attn_in, key=attn_in, value=attn_in, training=training)
        x = x + self.dropout1(attn_out, training=training)
        ffn_out = self.ffn(self.norm2(x), training=training)
        return x + self.dropout2(ffn_out, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ffn_factor": self.ffn_factor,
            "dropout": self.dropout_rate,
        })
        return config


# ---------------------------
# Transformer Model Class
# ---------------------------
class TransformerModel:
    # Pure self-attention encoder over the draw window. Rationale (see the
    # README research section): attention can weight ANY historical draw in
    # the window equally easily, so long-range temporal context isn't
    # squeezed through a recency-biased recurrence (LSTM) or a limited
    # receptive field (TCN). windowSize therefore defaults longer (30) than
    # TCN's 20 - the long-range view is the whole point of this model.
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
        self.window_size = 30
        self.predictionWindowSize = 30
        self.labelSmoothing = 0.05
        self.d_model = 64
        self.num_layers = 2
        self.num_heads = 4
        self.key_dim = 32
        self.ffn_factor = 4
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
    def setDModel(self, value): self.d_model = value
    def setNumLayers(self, value): self.num_layers = value
    def setNumHeads(self, value): self.num_heads = value
    def setKeyDim(self, value): self.key_dim = value
    def setFfnFactor(self, value): self.ffn_factor = value
    def setLoadModelWeights(self, value): self.loadModelWeights = value

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
    # Model Creation
    # ---------------------------
    def create_model(self, max_value, num_classes=50, model_path="", digitsPerDraw=3, specialColumnCount=0, specialNumClasses=0):
        # Trailing specialColumnCount positions (stars/dream number/viking
        # number) have their own, smaller number range than the main
        # positions - see the second output head below (same dual-head split
        # as TCN.py/LSTM.py).
        mainDigits = digitsPerDraw - specialColumnCount

        input_layer = layers.Input(shape=(self.window_size, digitsPerDraw))

        # Dense embedding lifts the raw digitsPerDraw-wide draw vector into
        # d_model dimensions - MultiHeadAttention needs enough width to split
        # across heads, and 3-7 raw features is far too narrow for that.
        x = layers.Dense(self.d_model)(input_layer)
        x = SinusoidalPositionalEncoding()(x)
        x = layers.Dropout(self.dropout)(x)

        # --- Pre-LN Transformer encoder stack ---
        for _ in range(self.num_layers):
            x = TransformerEncoderBlock(num_heads=self.num_heads, key_dim=self.key_dim,
                                        ffn_factor=self.ffn_factor, dropout=self.dropout)(x)

        # Pre-LN blocks leave the residual stream un-normalized on exit, so
        # close the stack with one final LayerNormalization (standard pre-LN
        # practice) before pooling.
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # --- Global pooling ---
        x = layers.GlobalAveragePooling1D()(x)

        # --- Output head(s) ---
        # Dense (linear) -> Reshape -> Softmax(axis=-1), NOT softmax inside
        # the Dense: softmax before the Reshape normalizes over the whole
        # flattened digitsPerDraw*num_classes vector instead of per position
        # (the bug TCN.py's comment documents) - this way each position's
        # probabilities sum to 1 independently.
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

        # clipnorm bounds the gradient's global L2 norm per update - the
        # standard defense against exploding gradients pushing loss to
        # inf/nan (seen in practice with some hyperopt-sampled learning
        # rate/dropout/l2 combos on other DL models in this repo).
        optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # Architecture fingerprint: only the parameters that shape the
        # weights or the input/output dimensions. Saved next to the weights;
        # on load, a mismatch (e.g. hyperopt just tuned different
        # d_model/window) triggers a fresh retrain instead of crashing on a
        # shape mismatch or silently keeping the old architecture's weights.
        # Non-shape parameters (dropout rate, learning rate, l2, ...) are
        # deliberately excluded so warm-starting keeps saving training time
        # when only those change. The sinusoidal positional encoding is
        # fixed (no weights) and fully determined by window_size + d_model,
        # both already fingerprinted.
        self._weights_fingerprint = {
            "max_value": int(max_value),
            "num_classes": int(num_classes),
            "digitsPerDraw": int(digitsPerDraw),
            "specialColumnCount": int(specialColumnCount),
            "specialNumClasses": int(specialNumClasses),
            "d_model": int(self.d_model),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "key_dim": int(self.key_dim),
            "ffn_factor": int(self.ffn_factor),
            "window_size": int(self.window_size),
        }

        if self.loadModelWeights:
            helpers.load_weights_if_fingerprint_matches(model, model_path, self._weights_fingerprint)

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
                            callbacks=[early_stopping, reduce_lr, checkpoint, TerminateOnNaN(), SelectiveProgbarLogger(verbose=1, epoch_interval=50)])
        return history

    # ---------------------------
    # Run Training + Prediction
    # ---------------------------
    def run(self, name="pick3", skipLastColumns=0, maxRows=0, skipRows=0, years_back=None, strict_val=True, specialColumnCount=0):
        train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels = helpers.load_data(
            self.dataPath, skipLastColumns, maxRows=maxRows, skipRows=skipRows, years_back=years_back
        )

        # This model gets a brand-new weights folder (transformer_model), so
        # unlike the pre-existing model folders it won't exist on a fresh
        # checkout/prod deploy - without this, the very first ModelCheckpoint
        # save would crash the run.
        os.makedirs(self.modelPath, exist_ok=True)

        # Trailing specialColumnCount positions (stars/dream number/viking
        # number) get their own num_classes/label range - see create_model's
        # second output head.
        mainDigits = numbers.shape[1] - specialColumnCount
        special_num_classes = 0
        special_unique_labels = None
        if specialColumnCount > 0:
            special_unique_labels = helpers.get_unique_labels(self.dataPath, special=True)
            special_num_classes = len(special_unique_labels)

        # Normalize labels to 0-based - without this, a raw value equal to
        # num_classes (e.g. Euromillions' 50 against a 50-class one-hot,
        # valid indices 0-49) crashes to_categorical with an out-of-bounds
        # index. Matches LSTM.py/TCN.py's normalization.
        min_label = np.min(unique_labels)
        numbers = numbers - min_label

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

        # Exploding gradients (some hyperopt-sampled hyperparameter combos)
        # can leave the model's live weights NaN/Inf even after training -
        # if the very first epoch is already non-finite, EarlyStopping's
        # restore_best_weights has no earlier "best" epoch to fall back to.
        # Force the worst possible score and skip saving below so a
        # corrupted run doesn't get persisted and then reloaded (and
        # re-corrupted) by every subsequent retrain step.
        weights_are_finite = helpers.model_weights_are_finite(model)
        if not weights_are_finite:
            # ModelCheckpoint monitors val_loss with save_best_only, and NaN
            # epochs never rank as "best" - so if any epoch before the
            # blow-up was healthy, the checkpoint still holds finite weights.
            # (TerminateOnNaN stops training at the first NaN batch, but
            # EarlyStopping only restores best weights when IT is the one
            # that stops training, so the live weights are corrupt here.)
            if os.path.exists(checkpoint_path):
                try:
                    model.load_weights(checkpoint_path)
                    weights_are_finite = helpers.model_weights_are_finite(model)
                except Exception as e:
                    print(f"Could not restore best checkpoint after non-finite training: {e}")
            if weights_are_finite:
                print(f"Warning: {name} training went non-finite (NaN/Inf) - recovered the best earlier epoch from the checkpoint.")
        if not weights_are_finite:
            self.last_val_loss = float("inf")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            # NaN from the very first batch: no healthy weights exist for this
            # run at all. A prediction from NaN weights is pure garbage
            # (np.argmax over an all-NaN row returns index 0 for every
            # position), so refuse to emit one - callers catch per-model and
            # skip this row for the day instead of recording noise.
            raise RuntimeError(
                f"{name}: training produced non-finite weights and no healthy "
                f"checkpoint exists - skipping this model's prediction")

        latest_raw_predictions = helpers.predict_numbers(model, numbers, window_size=self.predictionWindowSize)

        if specialColumnCount > 0:
            main_prediction, special_prediction = latest_raw_predictions
            latest_raw_predictions, combined_labels = helpers.combine_special_prediction(
                main_prediction, special_prediction, unique_labels, special_unique_labels)
            unique_labels = combined_labels

        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.savefig(os.path.join(self.modelPath, f"model_{name}_performance.png"))

        if weights_are_finite:
            helpers.save_weights_with_fingerprint(
                model, model_path, getattr(self, "_weights_fingerprint", None))
            model.save(model_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return latest_raw_predictions, unique_labels

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
    transformer_model = TransformerModel()

    name = "pick3"
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "transformer_model")

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-8-3.json")

    with open(sequenceToPredictFile, "r") as openfile:
        sequenceToPredict = json.load(openfile)

    transformer_model.setLoadModelWeights(False)
    transformer_model.setModelPath(modelPath)
    transformer_model.setDataPath(dataPath)
    transformer_model.setBatchSize(16)
    transformer_model.setEpochs(1000)
    transformer_model.setLearningRate(0.001)
    transformer_model.setDropout(0.3)
    transformer_model.setL2Regularization(0.0005)
    transformer_model.setEarlyStopPatience(20)
    transformer_model.setReduceLearningRatePatience(5)
    transformer_model.setReducedLearningRateFactor(0.5)
    transformer_model.setWindowSize(30)
    transformer_model.setPredictionWindowSize(30)
    transformer_model.setLabelSmoothing(0.05)
    transformer_model.setDModel(64)
    transformer_model.setNumLayers(2)
    transformer_model.setNumHeads(4)
    transformer_model.setKeyDim(32)
    transformer_model.setFfnFactor(4)

    latest_raw_predictions, unique_labels = transformer_model.run(name, years_back=20, strict_val=True)
    num_classes = len(unique_labels)

    latest_raw_predictions = latest_raw_predictions.tolist()
    predicted_digits = np.argmax(latest_raw_predictions, axis=-1)
    top3_indices = np.argsort(latest_raw_predictions, axis=-1)[:, -3:][:, ::-1]

    print(f"Top prediction per digit: {top3_indices[0].tolist()}")

    print("Prediction: ", predicted_digits.tolist())
    print("Real result: ", sequenceToPredict["realResult"])
