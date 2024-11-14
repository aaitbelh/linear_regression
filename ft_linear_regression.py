import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

data = pd.read_csv("data.csv")
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
# plt.scatter(X, Y)
# plt.show()

epochs = 500
n = len(data)
learning_rate = 0.1
theta1 = 0
theta0 = 0
for i in range(epochs):
    sumVal_theta1 = 0
    sumVal_theta0 = 0
    for j in range(n):
        x = ((X[j] - X.min()) / (X.max() - X.min()))
        y = ((Y[j] - Y.min()) / (Y.max() - Y.min()))
        estimatade_price = (theta1 * x) + theta0
        sumVal_theta1 += (estimatade_price - y) * x
        sumVal_theta0 += estimatade_price - y
    theta1 = theta1 - (learning_rate * (1/n) * sumVal_theta1)
    theta0 = theta0 - (learning_rate * (1/n) * sumVal_theta0)
print(theta1, theta0)
pre_km = 76025
predicted_price = (theta1 * ((pre_km - data.km.min()) / (data.km.max() - data.km.min())) + theta0) * (data.price.max() - data.price.min()) + data.price.min()
print(predicted_price)
#calculat the precision of my model
mse = 0
mae = 0
ss_total = 0
ss_residual = 0
mean_y = Y.mean()

for i in range(n):
    x = ((X[i] - X.min()) / (X.max() - X.min()))
    y = ((Y[i] - Y.min()) / (Y.max() - Y.min()))
    predicted_y = theta1 * x + theta0
    mse += (y - predicted_y) ** 2
    mae += abs(y - predicted_y)
    
    # For R-squared calculation
    ss_total += (y - mean_y) ** 2
    ss_residual += (y - predicted_y) ** 2

# Final calculations
mse = mse / n
rmse = math.sqrt(mse)
mae = mae / n
r_squared = 1 - (ss_residual / ss_total)

# Display the results
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-Squared (R²):", r_squared)