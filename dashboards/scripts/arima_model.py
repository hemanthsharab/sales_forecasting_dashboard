import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np
import os

path = os.getcwd()
print(path)
# Load the dataset
df = pd.read_csv(f'{path}/data/sales_data.csv', parse_dates=['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)

# Split into train and test
train_size = int(len(df) * 0.8)
train, test = df['sales'][:train_size], df['sales'][train_size:]

# Fit ARIMA model
arima_order = (5, 1, 0)  # You might need to adjust this based on AIC/BIC
arima_model = ARIMA(train, order=arima_order)
arima_result = arima_model.fit()

# Forecast
forecast = arima_result.forecast(steps=len(test))
test_forecast = forecast.values

# Evaluation
rmse = np.sqrt(mean_squared_error(test, test_forecast))
print(f'ARIMA RMSE: {rmse}')

# Save the model
with open(f'{path}/models/arima_model.pkl', 'wb') as pkl:
    pickle.dump(arima_result, pkl)
