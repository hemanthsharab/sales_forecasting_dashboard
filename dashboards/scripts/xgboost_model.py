import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import os
path = os.getcwd()
# Load the processed data
X_train = pd.read_csv(f'{path}/data/X_train.csv')
X_test = pd.read_csv(f'{path}/data/X_test.csv')
y_train = pd.read_csv(f'{path}/data/y_train.csv').values.ravel()
y_test = pd.read_csv(f'{path}/data/y_test.csv').values.ravel()

# Initialize XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'XGBoost RMSE: {rmse}')

# Save the model
joblib.dump(xgb_model, f'{path}/models/xgboost_model.pkl')
