import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



data = {
    'SquareFootage': [1500, 1800,2400, 3000, 3500, 4000, 4500],
    'Price': [200000, 250000, 300000, 350000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

print(df.head())


x = df[['SquareFootage']]
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training Data: {x_train.shape}, {y_train.shape}")
print(f"Testing Data: {x_test.shape}, {y_test.shape}")


model = LinearRegression()

model.fit(x_train, y_train)

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

y_prep = model.predict(x_test)

print(f"Predicted Prices: {y_prep}")
print("Actual prices:", y_test.values)


mse = mean_squared_error(y_test, y_prep)

r2 = r2_score(y_test, y_prep)

print(f"Mean Square Error: {mse}")
print(f"R-square: {r2}")

plt.scatter(x_test, y_test,color='blue', label='Actual Data')
plt.plot(x_test, y_prep, color='red', label='Regression Line')

plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices VS Square Footage')
plt.legend()

plt.show()