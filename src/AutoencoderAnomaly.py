# Import necessary libraries
import os, sys, json
import pandas as pd
import numpy as np
import tensorflow as tf

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

# Width of the learned per-number embedding that replaces one-hot encoding of
# the input window. Kept as a module constant rather than a setter/fingerprint
# entry: the fingerprint deliberately tracks only tunable architecture
# parameters, and if this constant is ever edited in code the saved weights
# simply fail to load on shape mismatch and trigger one fresh retrain (see
# helpers.load_weights_if_fingerprint_matches' exception path).
EMBED_DIM = 8


# ---------------------------
# Conditional Autoencoder for Anomaly Detection
# ---------------------------
class AutoencoderAnomaly:
    """
    Conditional (sequence-to-next) autoencoder for the README's "Security &
    Randomness Detection" research track: an unsupervised monitor for
    "predictability spikes". The encoder squeezes the recent draw window
    through a deliberately narrow latent bottleneck - the information
    bottleneck IS the anomaly detector: a truly random game leaves nothing
    compressible, so the decoder's per-draw negative log-likelihood of the
    real next draw should hover around the entropy floor. If that
    reconstruction error drops significantly on real draws (a strongly
    negative rolling z-score, see computeAnomalyScores), non-random structure
    is present - a game-integrity alert.

    The decoder emits standard per-position softmax heads for the NEXT draw,
    so run() satisfies the same contract as TCN.py/LSTM.py and the model
    doubles as a tracked prediction row in Predictor.py.
    """

    def __init__(self):
        self.dataPath = ""
        self.modelPath = ""
        self.epochs = 1000
        self.batchSize = 16
        self.dropout = 0.2
        self.l2Regularization = 0.0005
        self.earlyStopPatience = 20
        self.reduceLearningRatePatience = 5
        self.reduceLearningRateFactor = 0.5
        self.learning_rate = 0.001
        self.window_size = 20
        self.predictionWindowSize = 20
        # 0.0 by default (unlike the prediction-first DL models): anomaly
        # detection needs honest likelihoods - label smoothing puts a floor
        # under every class probability and would bias the NLL used as the
        # reconstruction-error signal.
        self.labelSmoothing = 0.0
        self.latent_dim = 16
        self.encoder_units = 64
        self.num_encoder_layers = 2
        self.loadModelWeights = True
        # Trained-state cache for computeAnomalyScores: scoring must reuse
        # the exact model/data run() produced in this process instead of
        # retraining or reloading.
        self._anomaly_cache = None

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
    def setLatentDim(self, value): self.latent_dim = value
    def setEncoderUnits(self, value): self.encoder_units = value
    def setNumEncoderLayers(self, value): self.num_encoder_layers = value
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
        # positions - see the second output head below, same dual-head
        # pattern as TCN.py/LSTM.py.
        mainDigits = digitsPerDraw - specialColumnCount

        input_layer = layers.Input(shape=(self.window_size, digitsPerDraw))

        # --- Encoder ---
        # The window arrives as raw 0-based integers (helpers.predict_numbers
        # feeds them straight to model.predict), so the categorical encoding
        # has to live inside the model. A learned embedding plays the "one-
        # hot" role compactly; after normalization every special column's
        # value range is a subset of the main range, so one shared table of
        # num_classes entries covers all positions.
        x = layers.Embedding(input_dim=num_classes, output_dim=EMBED_DIM)(input_layer)
        x = layers.Reshape((self.window_size, digitsPerDraw * EMBED_DIM))(x)

        for _ in range(self.num_encoder_layers):
            x = layers.Conv1D(self.encoder_units, kernel_size=3, padding="same", activation="relu",
                              kernel_regularizer=regularizers.l2(self.l2Regularization))(x)
            x = layers.Dropout(self.dropout)(x)

        x = layers.GlobalAveragePooling1D()(x)

        # The bottleneck: deliberately narrow (default 16) and linear. Its
        # narrowness is the whole point - the decoder can only reconstruct
        # the next draw well if the window actually carries compressible
        # structure, which a fair game should not have.
        latent = layers.Dense(self.latent_dim, name="latent_bottleneck")(x)

        # --- Decoder ---
        # No dropout after the bottleneck: the anomaly signal is the NLL the
        # decoder assigns at inference, and extra stochastic capacity-loss on
        # the decode side would only blur it.
        d = layers.Dense(self.encoder_units, activation="relu",
                         kernel_regularizer=regularizers.l2(self.l2Regularization))(latent)

        # Per-position softmax over each position's own class axis - Dense
        # (linear) -> Reshape -> Softmax(axis=-1), never softmax over the
        # flattened digitsPerDraw*num_classes vector (see the fix note in
        # TCN.py: flattened softmax normalizes across positions and breaks
        # both the probabilities and the NLL).
        main_logits = layers.Dense(mainDigits * num_classes,
                                   kernel_regularizer=regularizers.l2(self.l2Regularization))(d)
        main_logits = layers.Reshape((mainDigits, num_classes))(main_logits)
        main_output = layers.Softmax(axis=-1, name="main_output")(main_logits)

        if specialColumnCount > 0:
            special_logits = layers.Dense(specialColumnCount * specialNumClasses,
                                          kernel_regularizer=regularizers.l2(self.l2Regularization))(d)
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
        # inf/nan (see TCN.py, which hit this in practice with some
        # hyperopt-sampled parameter combos).
        optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # Architecture fingerprint: only the parameters that shape the
        # weights or the input/output dimensions. Saved next to the weights;
        # on load, a mismatch (e.g. hyperopt just tuned different
        # units/window) triggers a fresh retrain instead of crashing on a
        # shape mismatch or silently keeping the old architecture's weights.
        # Non-shape parameters (dropout rate, learning rate, l2, label
        # smoothing, ...) are deliberately excluded so warm-starting keeps
        # saving training time when only those change.
        self._weights_fingerprint = {
            "num_classes": int(num_classes),
            "digitsPerDraw": int(digitsPerDraw),
            "specialColumnCount": int(specialColumnCount),
            "specialNumClasses": int(specialNumClasses),
            "latent_dim": int(self.latent_dim),
            "encoder_units": int(self.encoder_units),
            "num_encoder_layers": int(self.num_encoder_layers),
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

        # Predictor.py creates the per-model-type folders lazily elsewhere;
        # creating it here keeps standalone/smoke runs self-sufficient.
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
        # came from.
        self.last_val_loss = float(min(history.history.get("val_loss", [float("inf")])))

        # Exploding gradients can leave the model's live weights NaN/Inf even
        # after training - if the very first epoch is already non-finite,
        # EarlyStopping's restore_best_weights has no earlier "best" epoch to
        # fall back to. Force the worst possible score and skip saving below
        # so a corrupted run doesn't get persisted and then reloaded (and
        # re-corrupted) by every subsequent retrain step. Same recovery
        # ladder as TCN.run.
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
            # skip this row for the day instead of recording noise. An NLL
            # from NaN weights would be equally useless, so the anomaly cache
            # is not populated either.
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
        # Free the figure - computeAnomalyScores runs in the same process, so
        # leaked figures would pile up across the per-game loop.
        plt.close("all")

        if weights_are_finite:
            helpers.save_weights_with_fingerprint(
                model, model_path, getattr(self, "_weights_fingerprint", None))
            model.save(model_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # Keep the healthy trained model and its exact (normalized) data in
        # memory for computeAnomalyScores - scoring must see the same model
        # state that produced this run's prediction, not a retrain.
        self._anomaly_cache = {
            "model": model,
            "numbers": numbers,
            "mainDigits": mainDigits,
            "specialColumnCount": specialColumnCount,
            "args": (name, skipLastColumns, skipRows, years_back, specialColumnCount),
        }

        return latest_raw_predictions, unique_labels

    # ---------------------------
    # Anomaly scoring (the actual "security" output)
    # ---------------------------
    def computeAnomalyScores(self, name, skipLastColumns=0, skipRows=0, years_back=None, specialColumnCount=0):
        """
        Slides over the last min(200, available) draws and, for each draw t,
        computes the negative log-likelihood of the REAL draw t under the
        model's prediction from the window ending at t-1 - the autoencoder's
        reconstruction error - plus a rolling z-score of that NLL against the
        (up to) 60 preceding NLLs.

        A strongly NEGATIVE z means the real draw was suddenly much easier to
        reconstruct than its recent past - the "predictability spike" /
        game-integrity alert condition from the README. Returns
        [{"index": i_from_end, "nll": float, "z": float or None}] oldest ->
        newest, where index counts back from the newest draw (newest = 0).
        """
        # Reuse the model run() trained in this process. If run() hasn't
        # happened yet (or ran with different data-selection arguments, so
        # its cached draws aren't the ones being scored), run it now.
        cache = self._anomaly_cache
        wanted_args = (name, skipLastColumns, skipRows, years_back, specialColumnCount)
        if cache is None or cache["args"] != wanted_args:
            self.run(name, skipLastColumns, skipRows=skipRows, years_back=years_back,
                     specialColumnCount=specialColumnCount)
            cache = self._anomaly_cache

        model = cache["model"]
        numbers = cache["numbers"]  # already 0-based normalized, oldest -> newest
        mainDigits = cache["mainDigits"]

        n = len(numbers)
        available = n - self.window_size
        if available <= 0:
            print(f"{name}: not enough draws ({n}) for a {self.window_size}-draw scoring window")
            return []

        num_scored = min(200, available)
        first_t = n - num_scored  # first scored draw index (>= window_size)

        # One stacked predict call instead of num_scored separate ones -
        # numbers[t - window_size : t] is the window ending at t-1 for each
        # scored draw t.
        windows = np.array([numbers[t - self.window_size:t] for t in range(first_t, n)])
        raw = model.predict(windows, verbose=0)

        # Dual-head models return [main_probs, special_probs]; the draw's
        # reconstruction error is the joint NLL over ALL positions, so both
        # heads contribute.
        if isinstance(raw, (list, tuple)):
            prob_blocks = [(np.asarray(raw[0]), 0), (np.asarray(raw[1]), mainDigits)]
        else:
            prob_blocks = [(np.asarray(raw), 0)]

        # Clip away exact zeros before the log: softmax underflow on a
        # confident wrong class would otherwise turn one position into
        # -log(0) = inf and poison the rolling statistics.
        eps = 1e-12
        nlls = np.zeros(num_scored)
        for probs, col_offset in prob_blocks:
            positions = probs.shape[1]
            actual = numbers[first_t:, col_offset:col_offset + positions]  # (num_scored, positions)
            picked = np.take_along_axis(probs, actual[:, :, None], axis=-1)[:, :, 0]
            nlls += -np.log(np.clip(picked, eps, 1.0)).sum(axis=1)

        # Rolling z against the preceding (up to) 60 NLLs. A z estimated from
        # only a handful of samples is noise, not evidence - require at least
        # 10 preceding scores (and non-degenerate spread) before emitting one,
        # otherwise report None so consumers can't mistake warm-up for signal.
        scores = []
        for i in range(num_scored):
            preceding = nlls[max(0, i - 60):i]
            z = None
            if len(preceding) >= 10:
                std = float(np.std(preceding))
                if std > 0 and np.isfinite(std):
                    z = float((nlls[i] - np.mean(preceding)) / std)
            scores.append({
                "index": int(n - 1 - (first_t + i)),
                "nll": float(nlls[i]),
                "z": z,
            })

        return scores


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    autoencoder_model = AutoencoderAnomaly()

    name = "pick3"
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "autoencoder_model")

    autoencoder_model.setLoadModelWeights(False)
    autoencoder_model.setModelPath(modelPath)
    autoencoder_model.setDataPath(dataPath)
    autoencoder_model.setBatchSize(16)
    autoencoder_model.setEpochs(1000)
    autoencoder_model.setLearningRate(0.001)
    autoencoder_model.setDropout(0.2)
    autoencoder_model.setL2Regularization(0.0005)
    autoencoder_model.setEarlyStopPatience(20)
    autoencoder_model.setReduceLearningRatePatience(5)
    autoencoder_model.setReducedLearningRateFactor(0.5)
    autoencoder_model.setWindowSize(20)
    autoencoder_model.setPredictionWindowSize(20)
    autoencoder_model.setLabelSmoothing(0.0)
    autoencoder_model.setLatentDim(16)
    autoencoder_model.setEncoderUnits(64)
    autoencoder_model.setNumEncoderLayers(2)

    latest_raw_predictions, unique_labels = autoencoder_model.run(name, years_back=20, strict_val=True)

    predicted_indices = np.argmax(latest_raw_predictions, axis=-1)
    predicted_digits = [int(unique_labels[i]) for i in predicted_indices]
    print("Prediction: ", predicted_digits)

    anomaly_scores = autoencoder_model.computeAnomalyScores(name, years_back=20)
    alerts = [s for s in anomaly_scores if s["z"] is not None and s["z"] < -3]
    print(f"Scored {len(anomaly_scores)} draws, predictability spikes (z < -3): {len(alerts)}")
    for s in anomaly_scores[-5:]:
        print(s)
