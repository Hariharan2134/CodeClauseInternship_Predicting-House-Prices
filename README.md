Housing Price Prediction Project
This project involves predicting house prices based on the number of bedrooms and the area (in square feet) using a linear regression model. The dataset used for this project is Housing.csv from Kaggle.

Table of Contents
Introduction
Dataset
Project Setup
Model Training
Prediction
Evaluation
Usage
Introduction
The goal of this project is to develop a simple linear regression model to predict house prices based on two features:

Number of bedrooms
Area in square feet
Dataset
The dataset Housing.csv contains information about various houses, including the number of bedrooms, the area in square feet, and the price. The dataset was sourced from Kaggle.

Dataset Information
File: Housing.csv
Features:
bedrooms: Number of bedrooms in the house
area: Area of the house in square feet
price: Price of the house
Summary Statistics
sql
Copy code
# Displayed using pandas describe() method
First Few Rows
bash
Copy code
# Displayed using pandas head() method
Project Setup
To set up the project, follow these steps:

Clone the repository:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Install the required packages:

Copy code
pip install pandas scikit-learn
Place the Housing.csv file in the project directory.

Model Training
The model is trained using the following steps:

Load the dataset using pandas.
Display dataset information, summary statistics, and the first few rows.
Split the dataset into features (X) and target (y).
Split the data into training and testing sets using train_test_split from sklearn.
Train a linear regression model using the training data.
Prediction
To predict the price of a house:

The user is prompted to enter the number of bedrooms and the area in square feet.
The model predicts the price based on the input features.
Evaluation
The model's performance is evaluated using the following metrics:

Mean Squared Error (MSE): Measures the average of the squares of the errors.
R-squared Score: Indicates how well the regression model fits the data.
python
Copy code
from sklearn.metrics import mean_squared_error, r2_score

# Predictions on the test set
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R-squared Score
r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')
Execute the script:
python housing_price_prediction.py
Enter the required inputs when prompted:

Number of bedrooms
Area in square feet
The script will output the predicted price of the house:

Predicted Price for <bedrooms> bedrooms and <area> sq.ft. area: $<predicted_price>
