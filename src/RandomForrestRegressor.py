import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from random import randint

i = 0 # Number of predictions to make
while i < 10: 
    # Load the data from Excel file
    path = os.getcwd()
    data = pd.read_excel(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "data", "euromillions-gamedata-NL-2024.xlsx"))

    # Split the data into features (X) and target (y)
    X = data[['Nummer 1', 'Nummer 2', 'Nummer 3', 'Nummer 4', 'Nummer 5', 'Ster 1', 'Ster 2']]
    y = data.iloc[:, 1:]

    # Train a Random Forest Regression model
    model = RandomForestRegressor(n_estimators=10000, random_state=None)
    model.fit(X, y)

    # Generate a new set of random features for prediction
    new_data = pd.DataFrame({
        "Nummer 1": [randint(1, 70) for _ in range(100)],
        "Nummer 2": [randint(1, 70) for _ in range(100)],
        "Nummer 3": [randint(1, 70) for _ in range(100)],
        "Nummer 4": [randint(1, 70) for _ in range(100)],
        "Nummer 5": [randint(1, 70) for _ in range(100)],
        "Ster 1": [randint(1, 25) for _ in range(100)],
        "Ster 2": [randint(1, 25) for _ in range(100)]
    })

    # Use the trained model to predict the next 6 numbers for each set of features
    predictions = model.predict(new_data)

    # Get the most likely set of numbers based on the predictions
    most_likely_set = predictions[0]
    for p in predictions:
        if p[0] > most_likely_set[0]:
            most_likely_set = p

    # Convert most_likely_set to whole numbers
    rounded_most_likely_set = [round(x) for x in most_likely_set]

    # Print the most likely set of numbers
    print(str(f"{i+1:02d}") + ". The most likely set of numbers is:", rounded_most_likely_set)
    i += 1
