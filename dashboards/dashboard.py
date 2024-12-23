import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os

path = os.getcwd()
# Set page configuration
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("Sales Forecasting Dashboard")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv(f'{path}/data/sales_data.csv', parse_dates=['date'])
    df = df.sort_values('date')
    return df

df = load_data()

# Display raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Sales Data")
    st.write(df)

# Visualization of Sales Over Time
st.subheader("Sales Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x='date', y='sales', data=df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Feature Importance from XGBoost
@st.cache_resource
def load_xgb_model():
    model = joblib.load(f'{path}/models/xgboost_model.pkl')
    return model

xgb_model = load_xgb_model()
feature_importances = pd.Series(xgb_model.feature_importances_, index=pd.read_csv(f'{path}/data/X_train.csv').columns)
feature_importances = feature_importances.sort_values(ascending=False)

st.subheader("Feature Importances (XGBoost)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax)
st.pyplot(fig)

# Forecasting

model_selection = st.selectbox("Select Model for Forecasting", ("XGBoost", "ARIMA"))

if model_selection == "XGBoost":
    # Load scaler
    scaler = joblib.load(f'{path}/models/scaler.pkl')
    
    # Load model
    model = xgb_model
    
    # Prepare latest data for prediction
    latest_data = pd.read_csv(f'{path}/data/X_test.csv').tail(1)
    latest_data_scaled = scaler.transform(latest_data)
    
    # Predict
    forecast = model.predict(latest_data_scaled)[0]
    
    st.write(f"**Next Day Forecasted Sales (XGBoost):** {forecast:.2f}")
    
    # Plot Forecast
    forecast_df = df.copy()
    forecast_date = df['date'].iloc[-1] + pd.Timedelta(days=1)
    forecast_df = forecast_df.append({'date': forecast_date, 'sales': forecast}, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='date', y='sales', data=forecast_df, ax=ax, label='Actual Sales')
    sns.scatterplot(x='date', y='sales', data=forecast_df.tail(1), color='red', ax=ax, label='Forecasted Sales')
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif model_selection == "ARIMA":
    # Load ARIMA model
    with open(f'{path}/models/arima_model.pkl', 'rb') as pkl_file:
        arima_model = pickle.load(pkl_file)
    
    # Forecast
    forecast = arima_model.forecast(steps=1)[0]
    
    st.write(f"**Next Day Forecasted Sales (ARIMA):** {forecast:.2f}")
    
    # Plot Forecast
    forecast_df = df.copy()
    forecast_date = df.index[-1] + pd.Timedelta(days=1)
    forecast_df = forecast_df.append({'date': forecast_date, 'sales': forecast}, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='date', y='sales', data=forecast_df, ax=ax, label='Actual Sales')
    sns.scatterplot(x='date', y='sales', data=forecast_df.tail(1), color='red', ax=ax, label='Forecasted Sales')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Comparison of Models
st.subheader("Model Performance Comparison")

# Load test data and predictions
X_test = pd.read_csv(f'{path}/data/X_test.csv')
y_test = pd.read_csv(f'{path}/data/y_test.csv').values.ravel()

# XGBoost Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Load ARIMA model
with open(f'{path}/models/arima_model.pkl', 'rb') as pkl_file:
    arima_model = pickle.load(pkl_file)

# ARIMA Predictions
df_full = pd.read_csv(f'{path}/data/sales_data.csv', parse_dates=['date'])
df_full = df_full.sort_values('date')
df_full.set_index('date', inplace=True)
train_size = int(len(df_full) * 0.8)
test = df_full['sales'][train_size:]
forecast_arima = arima_model.forecast(steps=len(test))
y_pred_arima = forecast_arima.values

# Calculate RMSE
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
rmse_arima = np.sqrt(mean_squared_error(test, y_pred_arima))

performance_data = {
    'Model': ['XGBoost', 'ARIMA'],
    'RMSE': [rmse_xgb, rmse_arima]
}
performance_df = pd.DataFrame(performance_data)
st.write(performance_df)
