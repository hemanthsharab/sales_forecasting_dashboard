import pandas as pd
import joblib
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np

import os
path = os.getcwd()
# Load XGBoost model
xgb_model = joblib.load(f'{path}/models/xgboost_model.pkl')

# Load ARIMA model
with open(f'{path}/models/arima_model.pkl', 'rb') as pkl:
    arima_model = pickle.load(pkl)

# Load test data
X_test = pd.read_csv(f'{path}/data/X_test.csv')
y_test = pd.read_csv(f'{path}/data/y_test.csv').values.ravel()

# XGBoost Predictions
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'XGBoost RMSE: {rmse_xgb}')

# ARIMA Predictions
df = pd.read_csv(f'{path}/data/sales_data.csv', parse_dates=['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)
train_size = int(len(df) * 0.8)
test = df['sales'][train_size:]
forecast = arima_model.forecast(steps=len(test))
y_pred_arima = forecast.values
rmse_arima = np.sqrt(mean_squared_error(test, y_pred_arima))
print(f'ARIMA RMSE: {rmse_arima}')

# Decide which model to use or combine
if rmse_xgb < rmse_arima:
    selected_model = 'XGBoost'
else:
    selected_model = 'ARIMA'

print(f'Selected Model: {selected_model}')
