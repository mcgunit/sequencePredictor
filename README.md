
# Sequence Predictor

## Installation

To install, you will need to have Python 3.x and the following libraries installed:
- numpy
- tensorflow
- keras
- art

You can install these libraries using pip by running the following command:

For CPU only: 
```
    python3 -m pip install numpy tensorflow keras art pandas scikit-learn openpyxl
```

For GPU enabled:

```
    python3 -m pip install numpy tensorflow[and-cuda] keras art pandas scikit-learn openpyxl
```

## How to run

To run the complete flow run:

```
    python3 Predictor.py
```

To test model specific for example LSTM run:

```
    python3 LSTM.py
```

Check the __main__ section of the LSTM.py or GRU.py for pointing to data and set parameters for testing.


## Disclaimer

The code within this repository comes with no guarantee, the use of this code is your responsibility. I take NO responsibility and/or liability for how you choose to use any of the source code available here. By using any of the files available in this repository, you understand that you are AGREEING TO USE AT YOUR OWN RISK. Once again, ALL files available here are for EDUCATION and/or RESEARCH purposes ONLY.
Please keep in mind that while LSTM.py uses advanced machine learning techniques to predict lottery numbers, there is no guarantee that its predictions will be accurate. Lottery results are inherently random and unpredictable, so it is important to use LSTM responsibly and not rely solely on its predictions.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this project as long as you give attribution to the original author.
