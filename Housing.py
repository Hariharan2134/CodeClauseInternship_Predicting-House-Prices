import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('Housing.csv')
print("Dataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nFirst Few Rows:")
print(df.head())
X = df[['bedrooms', 'area']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
bedrooms = int(input("Enter the count of bedrooms needed in the house: "))
area = int(input("Enter the square feet of the area required for the house: "))
predicted_price = model.predict([[bedrooms, area]])

print(f'\nPredicted Price for {bedrooms} bedrooms and {area} sq.ft. area: ${predicted_price[0]:,.2f}')
