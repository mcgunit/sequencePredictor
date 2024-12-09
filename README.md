
# Lotto Predictor

## Random Forest Regressor

This model is located in the randomForestRegressor folder.

To use this model, you will need to have Python installed on your computer, as well as the following libraries:

-   pandas
-   scikit-learn
-   openpyxl

To install the libraries, run the following command:

Copy code

`python3 -m pip install pandas scikit-learn openpyxl` 

### Usage

1.  Download the previous winning lottery numbers and save them in an Excel file.
2.  Run the `Predictor.py` file in the randomForestRegressor folder, which will train a Random Forest Regression model on the previous winning numbers and generate a set of predicted numbers.
3.  The program will output the most likely set of numbers for the next drawing.

### Example output

```
01. The most likely set of numbers is: [18, 30, 33, 36, 39, 42, 14]
02. The most likely set of numbers is: [18, 29, 32, 35, 39, 42, 14]
03. The most likely set of numbers is: [18, 30, 33, 35, 39, 42, 14]
04. The most likely set of numbers is: [19, 30, 33, 37, 40, 43, 15]
05. The most likely set of numbers is: [19, 30, 33, 37, 40, 42, 14]
06. The most likely set of numbers is: [18, 24, 28, 31, 37, 41, 31]
07. The most likely set of numbers is: [18, 29, 32, 35, 39, 42, 15]
08. The most likely set of numbers is: [18, 30, 33, 37, 40, 42, 14]
09. The most likely set of numbers is: [18, 25, 29, 32, 37, 42, 30]
10. The most likely set of numbers is: [18, 26, 30, 33, 38, 42, 30]
```

## LotteryAi

LotteryAi is a lottery prediction artificial intelligence that uses machine learning to predict the winning numbers of a any lottery game.

### Installation

To install LotteryAi, you will need to have Python 3.x and the following libraries installed:
- numpy
- tensorflow
- keras
- art

You can install these libraries using pip by running the following command:

'''
    python3 -m pip install numpy tensorflow keras art
'''

### Usage

To use LotteryAi, you will need to have a data file containing past lottery results. This file should be in a comma-separated format, with each row representing a single draw and the numbers in ascending order, rows are in new line without comma. Dont use white spaces. Last row number must have nothing after last number.

Once you have the data file, you can run the `LotteryAi.py` script to train the model and generate predictions. The script will print the generated ASCII art and the first ten rows of predicted numbers to the console.

## Disclaimer

The code within this repository comes with no guarantee, the use of this code is your responsibility. I take NO responsibility and/or liability for how you choose to use any of the source code available here. By using any of the files available in this repository, you understand that you are AGREEING TO USE AT YOUR OWN RISK. Once again, ALL files available here are for EDUCATION and/or RESEARCH purposes ONLY.
Please keep in mind that while LotteryAi.py uses advanced machine learning techniques to predict lottery numbers, there is no guarantee that its predictions will be accurate. Lottery results are inherently random and unpredictable, so it is important to use LotteryAi responsibly and not rely solely on its predictions.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this project as long as you give attribution to the original author.
