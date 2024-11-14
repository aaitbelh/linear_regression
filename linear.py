import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

# Normalize using MinMaxScaler
scaler = MinMaxScaler()

# Features (mileage) and target (price)
X = data.iloc[:, 0].values.reshape(-1, 1)  # Assuming mileage is in the first column
Y = data.iloc[:, 1].values.reshape(-1, 1)  # Assuming price is in the second column

# Scale the features and target
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)

# Hyperparameters
epochs = 500
n = len(data)
learning_rate = 0.1

# Initialize theta (coefficients)
theta1 = 0
theta0 = 0

# Gradient Descent
for i in range(epochs):
    sumVal_theta1 = 0
    sumVal_theta0 = 0
    for j in range(n):
        x = X_scaled[j]
        y = Y_scaled[j]
        estimated_price = (theta1 * x) + theta0
        sumVal_theta1 += (estimated_price - y) * x
        sumVal_theta0 += (estimated_price - y)
    
    # Update theta values
    theta1 = theta1 - (learning_rate * (1/n) * sumVal_theta1)
    theta0 = theta0 - (learning_rate * (1/n) * sumVal_theta0)

# Predicting prices for all the scaled mileage values
# De-normalize predicted values to original scale
predicted_Y = scaler.inverse_transform(predicted_Y_scaled.reshape(-1, 1))

# Now calculate the precision of the model

# Mean Squared Error (MSE)
mse = np.mean((predicted_Y - Y)**2)
print(f"Mean Squared Error (MSE): {mse}")

# R-squared (R²) Score Calculation
ss_total = np.sum((Y - np.mean(Y))**2)  # Total sum of squares
ss_residual = np.sum((Y - predicted_Y)**2)  # Residual sum of squares
r_squared = 1 - (ss_residual / ss_total)
print(f"R-squared (R²): {r_squared}")
