import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Example data: actual values and predicted values
actual = np.array([-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2])
predicted = np.array([14247.34, 211851.0, 4430.84, 50.71, 2.5, 45.45, 22.99, 3.5, 0, 50, 10, 1, 1, 14.8, 6.45])
predicted = np.log10(predicted + 1e-2)

# Mean Squared Error
mse = mean_squared_error(actual, predicted)

# Mean Absolute Error
mae = mean_absolute_error(actual, predicted)

# R-squared
r2 = r2_score(actual, predicted)

# Mean Absolute Percentage Error
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")