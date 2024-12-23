import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

path = os.getcwd()


# Load the dataset
df = pd.read_csv(f'{path}/data/sales_data.csv', parse_dates=['date'])

# Sort by date
df = df.sort_values('date')

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Feature Engineering
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Lag Features
df['sales_lag_1'] = df['sales'].shift(1)
df['sales_lag_7'] = df['sales'].shift(7)
df['sales_lag_30'] = df['sales'].shift(30)

# Rolling Features
df['sales_roll_mean_7'] = df['sales'].rolling(window=7).mean()
df['sales_roll_std_7'] = df['sales'].rolling(window=7).std()

# Drop rows with NaN values after feature engineering
df.dropna(inplace=True)

# Define features and target
X = df.drop(['date', 'sales'], axis=1)
y = df['sales']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, f'{path}/models/scaler.pkl')

# Save the processed data for modeling
X_train.to_csv(f'{path}/data/X_train.csv', index=False)
X_test.to_csv(f'{path}/data/X_test.csv', index=False)
y_train.to_csv(f'{path}/data/y_train.csv', index=False)
y_test.to_csv(f'{path}/data/y_test.csv', index=False)
