# Build this file with: docker build -t sequence_predicor .
FROM python:3.11.11-bookworm

# Install needed dependancies
RUN python3 -m pip install numpy tensorflow==2.18 keras art pandas scikit-learn matplotlib keras-tcn==3.1.2 asciichartpy==1.5.25 pmdarima==2.0.4 statsmodels==0.14.4 optuna==4.3.0 xgboost=3.0.0


# You can run this docker after building:
    # docker run --rm -it -v {absolute path to folder sequencePredictor}:/opt/sequencePredictor sequence_predictor /bin/bash
    