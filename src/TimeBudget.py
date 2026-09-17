import time
import tensorflow as tf


class TimeBudgetCallback(tf.keras.callbacks.Callback):
    """
    Stops a training run once `seconds` of wall clock have passed since
    fit() began - checked after every batch, so one long epoch cannot overrun
    the budget by a whole epoch.

    It stops training the way EarlyStopping and TerminateOnNaN do (setting
    model.stop_training), so the rest of the callback stack behaves exactly
    as for an early stop: Keras still runs the epoch's validation and
    on_epoch_end, EarlyStopping(restore_best_weights=True) puts the best
    epoch's weights back in its on_train_end, and ModelCheckpoint has already
    saved the best epoch. The model therefore still predicts, with the best
    weights it reached - a slow training day costs quality, never the row.
    That is what lets the daily pipeline run the heavy deep learning models
    again (Predictor.py --dl-model-seconds) without a training run blocking
    the prediction flow. `triggered` tells the caller the budget cut the run.
    """

    def __init__(self, seconds, label=""):
        super().__init__()
        self.seconds = float(seconds) if seconds and seconds > 0 else None
        self.label = label
        self.started = None
        self.triggered = False
        self.epochs_run = 0

    def on_train_begin(self, logs=None):
        self.started = time.time()
        self.triggered = False
        self.epochs_run = 0

    def _check(self):
        if self.seconds is None or self.triggered or self.started is None:
            return
        elapsed = time.time() - self.started
        if elapsed >= self.seconds:
            self.triggered = True
            self.model.stop_training = True
            where = f" for {self.label}" if self.label else ""
            print(f"Time budget of {self.seconds:.0f}s reached{where} after {self.epochs_run} completed "
                  f"epoch(s) ({elapsed:.0f}s) - stopping training, keeping the best weights so far")

    def on_train_batch_end(self, batch, logs=None):
        self._check()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_run = epoch + 1
        self._check()
