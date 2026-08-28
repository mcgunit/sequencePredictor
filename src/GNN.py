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
# Graph readout layer (GCN stack + window gather)
# ---------------------------
@tf.keras.utils.register_keras_serializable(package="gnn")
class GraphWindowReadout(layers.Layer):
    # One layer owns the whole "graph side" of the model: the GCN stack over
    # the co-occurrence graph plus the gather that conditions on the recent
    # window. The adjacency and static node features are DATA, not weights -
    # they are recomputed in numpy on every run() from the training draws and
    # baked in here as tf.constants. Baking them into the graph (rather than
    # making them a second model input) is deliberate: helpers.predict_numbers
    # feeds the model exactly one input, the (1, window_size, digitsPerDraw)
    # window tensor, so a single-input model keeps the shared prediction path
    # working unchanged for this model too.
    def __init__(self, adjacency=None, node_features=None, main_digits=3,
                 gcn_units=32, num_gcn_layers=2, embedding_dim=16,
                 special_num_classes=0, l2=0.0005, **kwargs):
        super().__init__(**kwargs)
        # Constants, not add_weight: they must never train and never end up
        # in the fingerprint - only their SHAPE (num_classes, feature count)
        # shapes the trainable weights below, and those dims are
        # fingerprinted by create_model.
        self.adjacency = None if adjacency is None else tf.constant(np.asarray(adjacency), dtype=tf.float32)
        self.node_features = None if node_features is None else tf.constant(np.asarray(node_features), dtype=tf.float32)
        self.main_digits = main_digits
        self.gcn_units = gcn_units
        self.num_gcn_layers = num_gcn_layers
        self.embedding_dim = embedding_dim
        self.special_num_classes = special_num_classes
        self.l2 = l2

    def build(self, input_shape):
        num_classes = int(self.adjacency.shape[0])
        static_feature_count = int(self.node_features.shape[1])

        # Trainable per-number embedding, concatenated with the static
        # (frequency/recency/degree) features - gives the GCN something to
        # learn per node beyond the handcrafted statistics.
        self.node_embedding = self.add_weight(
            name="node_embedding",
            shape=(num_classes, self.embedding_dim),
            initializer="glorot_uniform",
            regularizer=regularizers.l2(self.l2))

        # GCN kernels: H' = relu(A_norm @ H @ W + b), stacked. Two hops let a
        # number's representation absorb its community (numbers drawn
        # together more often than chance), which is the whole point of this
        # model vs the purely sequential ones.
        in_dim = static_feature_count + self.embedding_dim
        self.gcn_kernels = []
        self.gcn_biases = []
        for i in range(self.num_gcn_layers):
            self.gcn_kernels.append(self.add_weight(
                name=f"gcn_kernel_{i}",
                shape=(in_dim, self.gcn_units),
                initializer="glorot_uniform",
                regularizer=regularizers.l2(self.l2)))
            self.gcn_biases.append(self.add_weight(
                name=f"gcn_bias_{i}",
                shape=(self.gcn_units,),
                initializer="zeros"))
            in_dim = self.gcn_units

    def call(self, window):
        # The node embeddings are batch-independent, so the GCN runs once per
        # forward pass over all num_classes nodes - cheap even for Keno
        # (80x80 matmuls), regardless of batch size.
        h = tf.concat([self.node_features, self.node_embedding], axis=-1)
        for kernel, bias in zip(self.gcn_kernels, self.gcn_biases):
            h = tf.nn.relu(tf.matmul(self.adjacency, tf.matmul(h, kernel)) + bias)

        # Readout conditioning on the recent window: look up the GCN
        # embedding of every MAIN number in every draw of the window and
        # mean-pool per timestep. Only the main columns index the node table -
        # the trailing special columns (stars/dream/viking number) live in
        # their own smaller label space, so their 0-based indices would
        # collide with main numbers 0..specialNumClasses-1.
        idx = tf.cast(window[:, :, :self.main_digits], tf.int32)
        gathered = tf.gather(h, idx)                       # (batch, T, mainDigits, units)
        step_embeddings = tf.reduce_mean(gathered, axis=2)  # (batch, T, units)

        # Special-column games: append the (scaled) raw special history to
        # each timestep so the special head isn't predicting blind from main
        # -number context alone. Kept raw (no own graph) - with only 1-2
        # special digits per draw there is barely any co-occurrence structure
        # to convolve over.
        if self.main_digits < window.shape[-1]:
            special_scale = float(max(self.special_num_classes, 1))
            step_embeddings = tf.concat(
                [step_embeddings, window[:, :, self.main_digits:] / special_scale], axis=-1)

        # Global mean node embedding = "state of the whole graph" this run,
        # tiled per batch so it can be concatenated after the GRU.
        global_embedding = tf.reduce_mean(h, axis=0)
        global_embedding = tf.tile(global_embedding[None, :], [tf.shape(window)[0], 1])

        return step_embeddings, global_embedding

    def get_config(self):
        # Keras 3 refuses to serialize a layer whose __init__ took
        # non-serializable args (the checkpoint callback saves the full
        # .keras model every best epoch, so this runs during training, not
        # just at the final model.save). The graph constants are shipped as
        # plain nested lists - a few KB for most games, ~100KB for Keno's
        # 80x80 adjacency, both fine inside the zipped .keras archive.
        config = super().get_config()
        config.update({
            "adjacency": None if self.adjacency is None else self.adjacency.numpy().tolist(),
            "node_features": None if self.node_features is None else self.node_features.numpy().tolist(),
            "main_digits": self.main_digits,
            "gcn_units": self.gcn_units,
            "num_gcn_layers": self.num_gcn_layers,
            "embedding_dim": self.embedding_dim,
            "special_num_classes": self.special_num_classes,
            "l2": self.l2,
        })
        return config


# ---------------------------
# GNN Model Class
# ---------------------------
class GNNModel:
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
        self.gcn_units = 32
        self.num_gcn_layers = 2
        self.embedding_dim = 16
        # Exponential recency weight for the co-occurrence adjacency: a pair
        # seen `age` draws ago contributes decay^age. 0.999 keeps years of
        # history relevant (half-life ~690 draws) while still letting the
        # graph drift with the recent regime.
        self.decay = 0.999
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
    def setGcnUnits(self, value): self.gcn_units = value
    def setNumGcnLayers(self, value): self.num_gcn_layers = value
    def setEmbeddingDim(self, value): self.embedding_dim = value
    def setDecay(self, value): self.decay = value
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
    # Graph construction (numpy, per run)
    # ---------------------------
    def build_graph_inputs(self, main_draws, num_classes):
        # The graph is DATA, not weights: recomputed here from the TRAINING
        # portion of the (0-based, ascending) draw history on every run, then
        # handed to create_model as constants. Built from the training
        # portion only so the graph never encodes the validation draws the
        # model is scored against.
        n_draws = len(main_draws)

        # Co-occurrence adjacency with exponential recency weighting: every
        # pair of numbers appearing in the same draw strengthens their edge,
        # recent draws counting more. Communities of numbers drawn together
        # more often than chance become dense blocks the GCN can detect -
        # structure a pairwise decay statistic can't represent.
        adjacency = np.zeros((num_classes, num_classes), dtype=np.float64)
        for t, draw in enumerate(main_draws):
            w = self.decay ** (n_draws - 1 - t)
            rows, cols = np.meshgrid(draw, draw)
            # np.add.at instead of fancy-index += : repeated digits inside
            # one draw (possible for pick3) would otherwise be silently
            # collapsed by numpy's buffered assignment.
            np.add.at(adjacency, (rows.ravel(), cols.ravel()), w)
        # A number co-occurring with itself carries no pair information.
        np.fill_diagonal(adjacency, 0.0)

        # Degree centrality from the raw (pre-normalization) adjacency - how
        # connected a number is overall.
        degree = adjacency.sum(axis=1)
        degree_feature = degree / degree.max() if degree.max() > 0 else degree

        # Symmetric normalization D^-1/2 (A+I) D^-1/2 (the standard GCN
        # propagation matrix): self-loops keep each node's own features in
        # the mix, and the degree scaling stops repeated convolutions from
        # blowing up high-degree nodes' activations.
        a_hat = adjacency + np.eye(num_classes)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(a_hat.sum(axis=1), 1e-12))
        a_norm = (a_hat * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]

        # Static node features, all scaled to [0, 1]:
        # long-run frequency, recent-window frequency, draws-since-last-seen,
        # degree centrality. The trainable embedding is added inside the
        # layer (it's a weight; these are data).
        counts = np.bincount(main_draws.ravel(), minlength=num_classes).astype(np.float64)
        long_freq = counts / counts.max() if counts.max() > 0 else counts

        recent_n = min(n_draws, max(self.window_size, 50))
        recent_counts = np.bincount(main_draws[-recent_n:].ravel(), minlength=num_classes).astype(np.float64)
        recent_freq = recent_counts / recent_counts.max() if recent_counts.max() > 0 else recent_counts

        # Never-seen numbers get the maximum possible age.
        last_seen_age = np.full(num_classes, n_draws, dtype=np.float64)
        for t, draw in enumerate(main_draws):
            # Iterating oldest->newest, later draws overwrite with a smaller
            # age, so what survives is the age at the last occurrence.
            last_seen_age[draw] = n_draws - 1 - t
        last_seen_feature = last_seen_age / max(n_draws, 1)

        node_features = np.stack(
            [long_freq, recent_freq, last_seen_feature, degree_feature], axis=1)

        return a_norm.astype(np.float32), node_features.astype(np.float32)

    # ---------------------------
    # Model Creation
    # ---------------------------
    def create_model(self, max_value, num_classes=50, model_path="", digitsPerDraw=3,
                     specialColumnCount=0, specialNumClasses=0, adjacency=None, node_features=None):
        # Trailing specialColumnCount positions (stars/dream number/viking
        # number) have their own, smaller number range than the main
        # positions - see the second output head below.
        mainDigits = digitsPerDraw - specialColumnCount

        window_input = layers.Input(shape=(self.window_size, digitsPerDraw))

        # GCN over the co-occurrence graph + gather of the window's numbers -
        # see GraphWindowReadout for why the graph is baked in as constants.
        step_embeddings, global_embedding = GraphWindowReadout(
            adjacency=adjacency,
            node_features=node_features,
            main_digits=mainDigits,
            gcn_units=self.gcn_units,
            num_gcn_layers=self.num_gcn_layers,
            embedding_dim=self.embedding_dim,
            special_num_classes=specialNumClasses,
            l2=self.l2Regularization)(window_input)

        # Small GRU over the per-timestep pooled node embeddings: this is
        # what turns the static graph into a predictor - the trajectory of
        # "which communities have been firing lately" instead of a single
        # snapshot. Kept at gcn_units (32 by default) so even Keno's
        # 20-digit/80-class windows stay cheap on the 6GB prod GPU.
        x = layers.GRU(self.gcn_units)(step_embeddings)
        x = layers.Concatenate()([x, global_embedding])
        x = layers.Dense(2 * self.gcn_units, activation="relu",
                         kernel_regularizer=regularizers.l2(self.l2Regularization))(x)
        x = layers.Dropout(self.dropout)(x)

        # --- Output head(s) ---
        # Dense (linear) -> Reshape -> Softmax(axis=-1), NOT softmax inside
        # the Dense: softmax before the reshape would normalize over the
        # whole flattened digitsPerDraw*num_classes vector instead of per
        # position (the bug TCN.py's head comment documents).
        main_logits = layers.Dense(mainDigits * num_classes,
                                   kernel_regularizer=regularizers.l2(self.l2Regularization))(x)
        main_logits = layers.Reshape((mainDigits, num_classes))(main_logits)
        main_output = layers.Softmax(axis=-1, name="main_output")(main_logits)

        if specialColumnCount > 0:
            special_logits = layers.Dense(specialColumnCount * specialNumClasses,
                                          kernel_regularizer=regularizers.l2(self.l2Regularization))(x)
            special_logits = layers.Reshape((specialColumnCount, specialNumClasses))(special_logits)
            special_output = layers.Softmax(axis=-1, name="special_output")(special_logits)

            model = models.Model(inputs=window_input, outputs=[main_output, special_output])
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
            model = models.Model(inputs=window_input, outputs=main_output)
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
        # inf/nan (seen in practice with some hyperopt-sampled parameter
        # combos, especially on Keno's large 20-position, 80-class output).
        optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # Architecture fingerprint: ONLY the parameters that shape the
        # weights or input/output dimensions - a mismatch triggers a fresh
        # retrain instead of a shape-mismatch crash or silently stale
        # architecture (see load_weights_if_fingerprint_matches). The
        # adjacency/node feature VALUES are deliberately excluded: they
        # change with every new draw, but never change any weight shape -
        # only num_classes and the feature COUNT do, and those are here.
        # Non-shape parameters (dropout, learning rate, l2, decay) are also
        # excluded so warm-starting keeps saving training time when only
        # those change.
        self._weights_fingerprint = {
            "num_classes": int(num_classes),
            "digitsPerDraw": int(digitsPerDraw),
            "specialColumnCount": int(specialColumnCount),
            "specialNumClasses": int(specialNumClasses),
            "gcn_units": int(self.gcn_units),
            "num_gcn_layers": int(self.num_gcn_layers),
            "embedding_dim": int(self.embedding_dim),
            "window_size": int(self.window_size),
            "node_feature_count": int(node_features.shape[1]),
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
        # Like TransformerModel/AutoencoderAnomaly: the weights folder (e.g.
        # data/models/gnn_model) doesn't exist on a fresh checkout, and the
        # ModelCheckpoint/savefig calls below would fail without it.
        os.makedirs(self.modelPath, exist_ok=True)

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
        # index. Also required here for the graph: the 0-based values double
        # as node indices into the adjacency/embedding tables.
        min_label = np.min(unique_labels)
        numbers = numbers - min_label

        model_path = os.path.join(self.modelPath, f"model_{name}.keras")
        checkpoint_path = os.path.join(self.modelPath, f"model_{name}_checkpoint.keras")

        n = len(numbers)
        split_idx = int(n * 0.8)

        # Graph built from the training portion only (both strict_val
        # branches train on numbers[:split_idx]) so validation draws never
        # leak into the adjacency the model conditions on.
        a_norm, node_features = self.build_graph_inputs(numbers[:split_idx, :mainDigits], num_classes)

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
                                  specialColumnCount=specialColumnCount, specialNumClasses=special_num_classes,
                                  adjacency=a_norm, node_features=node_features)
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
# Main
# ---------------------------
if __name__ == "__main__":
    gnn_model = GNNModel()

    name = "pick3"
    path = os.getcwd()
    dataPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "trainingData", name)
    modelPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "models", "gnn_model")

    jsonDirPath = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "test", "database", name)
    sequenceToPredictFile = os.path.join(jsonDirPath, "2025-8-3.json")

    with open(sequenceToPredictFile, "r") as openfile:
        sequenceToPredict = json.load(openfile)

    gnn_model.setLoadModelWeights(False)
    gnn_model.setModelPath(modelPath)
    gnn_model.setDataPath(dataPath)
    gnn_model.setBatchSize(16)
    gnn_model.setEpochs(2000)
    gnn_model.setLearningRate(0.001)
    gnn_model.setDropout(0.3)
    gnn_model.setL2Regularization(0.0005)
    gnn_model.setEarlyStopPatience(300)
    gnn_model.setReduceLearningRatePatience(15)
    gnn_model.setReducedLearningRateFactor(0.5)
    gnn_model.setWindowSize(20)
    gnn_model.setPredictionWindowSize(20)
    gnn_model.setLabelSmoothing(0.05)
    gnn_model.setGcnUnits(32)
    gnn_model.setNumGcnLayers(2)
    gnn_model.setEmbeddingDim(16)
    gnn_model.setDecay(0.999)

    latest_raw_predictions, unique_labels = gnn_model.run(name, years_back=20, strict_val=True)
    num_classes = len(unique_labels)

    latest_raw_predictions = latest_raw_predictions.tolist()
    predicted_digits = np.argmax(latest_raw_predictions, axis=-1)
    top3_indices = np.argsort(latest_raw_predictions, axis=-1)[:, -3:][:, ::-1]

    print(f"Top prediction per digit: {top3_indices[0].tolist()}")

    print("Prediction: ", predicted_digits.tolist())
    print("Real result: ", sequenceToPredict["realResult"])
