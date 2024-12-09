
# Lottery Predictor

This project is a machine learning model designed to predict the most likely set of lottery numbers for the next drawing based on previous winning numbers.

## Getting Started

To use this model, you will need to have Python installed on your computer, as well as the following libraries:

-   pandas
-   scikit-learn

To install the libraries, run the following command:

Copy code

`pip install pandas scikit-learn` 

## Usage

1.  Download the previous winning lottery numbers from your state's lottery website and save them in an Excel file.
2.  Run the `Predictor.py` file, which will train a Random Forest Regression model on the previous winning numbers and generate a set of predicted numbers.
3.  The program will output the most likely set of numbers for the next drawing.

## Example output

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

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this project as long as you give attribution to the original author.
