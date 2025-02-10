import os, sys
import pandas as pd

from matplotlib import pyplot as plt
from keras import layers, models
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
from SelectiveProgbarLogger import SelectiveProgbarLogger

helpers = Helpers()

class RefinePrediction():
    def trainRefinePredictionsModel(self, name, path_to_json_folder, modelPath, num_classes=80, numbersLength=20):
        """
        Create a neural network to refine predictions.
        Ensures output has the same shape as the original raw predictions (numbersLength, num_classes).
        """
        epochs = 1000
        model_path = os.path.join(modelPath, f"refine_prediction_model_{name}.keras")

        X_train, y_train = helpers.extractFeaturesFromJsonForRefinement(path_to_json_folder, num_classes=num_classes, numbersLength=numbersLength)

        if len(X_train) > 0 and len(y_train) > 0:

            inputShape = (numbersLength, num_classes)  # Ensure input shape is consistent with raw predictions

            model = models.Sequential([
                layers.Input(shape=inputShape),
                layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
                layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.3),
                layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))
            ])

            model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])

            if os.path.exists(model_path):
                model.load_weights(model_path)
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=8,
                verbose=False,
                callbacks=[SelectiveProgbarLogger(verbose=1, epoch_interval=epochs/2)]
            )
            
            model.save(model_path)
            
            pd.DataFrame(history.history).plot(figsize=(8, 5))
            plt.savefig(os.path.join(modelPath, f'refine_prediction_model_{name}_performance.png'))
            print(f"Refine Prediction AI Model {name} Trained and Saved!")


    def refinePrediction(self, name, pathToLatestPredictionFile, modelPath, num_classes=80, numbersLength=20):
        """
        Refine the predictions while keeping the (numbersLength, num_classes) shape.
        """
        model_path = os.path.join(modelPath, f"refine_prediction_model_{name}.keras")
        second_model = load_model(model_path)

        X_new, _ = helpers.extractFeaturesFromJsonForRefinement(pathToLatestPredictionFile, num_classes=num_classes, numbersLength=numbersLength)

        if len(X_new) > 0:
            refined_prediction = second_model.predict(X_new)
            
            refined_prediction = refined_prediction.reshape(-1, numbersLength, num_classes)  # Ensure correct shape
            print("Refined Prediction Shape:", refined_prediction.shape)
            return refined_prediction
        else:
            return []
    