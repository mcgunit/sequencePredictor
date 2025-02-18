import os, sys
import pandas as pd
import tensorflow.keras.backend as K

from matplotlib import pyplot as plt
from keras import layers, models
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable


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

@register_keras_serializable()
def multi_label_accuracy(y_true, y_pred):
    """
    Computes the accuracy for multi-label classification.
    It calculates how many of the predicted numbers match the actual numbers.
    """
    threshold = 0.5  # Convert probabilities to binary (0 or 1)
    y_pred = K.cast(y_pred > threshold, dtype='float32')  # Convert predictions to 0s & 1s
    correct_preds = K.sum(y_true * y_pred, axis=-1)  # Count matching 1s
    total_preds = K.sum(y_true, axis=-1)  # Count total 1s in actual result
    return correct_preds / total_preds  # Percentage of correctly predicted numbers

class TopPrediction():
    def trainTopPredictionsModel(self, name, path_to_json_folder, modelPath, num_classes=80, numbersLength=20):
        """
        Create a neural network to get the top prediction.
        @num_classes: How many numbers to predict.
        """
        epochs = 1000
        model_path = os.path.join(modelPath, f"top_prediction_model_{name}.keras")

        X_train, y_train = helpers.extractFeaturesFromJsonForDetermineTopPrediction(path_to_json_folder, num_classes=num_classes, numbersLength=numbersLength)

        if len(X_train) > 0 and len(y_train) > 0:
            inputShape = (X_train.shape[1],)

            model = models.Sequential([
                layers.Input(shape=inputShape), 
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.6),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.6),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes, activation='sigmoid')
            ])

            #model.build(input_shape=inputShape)

            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=[multi_label_accuracy])

            if os.path.exists(model_path):
                model.load_weights(model_path)
            
            # Create and train the model
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=8, verbose=False, callbacks=[SelectiveProgbarLogger(verbose=1, epoch_interval=epochs/2)])

            # Save model for future use
            model.save(model_path)

            # Plot training history
            pd.DataFrame(history.history).plot(figsize=(8, 5))
            plt.savefig(os.path.join(modelPath, f'top_prediction_model_{name}_performance.png'))

            print(f"Refine Prediction AI Model {name} Trained and Saved!")
    
    def topPrediction(self, name, pathToLatestPredictionFile, modelPath, num_classes=80, numbersLength=20):
        """
            Get top prediction with an AI
        """

        model_path = os.path.join(modelPath, f"top_prediction_model_{name}.keras")

        second_model = load_model(model_path, custom_objects={"multi_label_accuracy": multi_label_accuracy})

        # Get new prediction features
        new_json = pathToLatestPredictionFile
        X_new, _ = helpers.extractFeaturesFromJsonForDetermineTopPrediction(new_json, num_classes=num_classes, numbersLength=numbersLength)

        if len(X_new) > 0:
            # Get refined prediction
            refined_prediction = second_model.predict(X_new)

            #print("Top Prediction: ", refined_prediction)
            return refined_prediction
        else:
            return []