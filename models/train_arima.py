import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_FOLDER   = r"C:\Users\HP\Desktop\DA\Data"
PREP_FOLDER   = os.path.join(DATA_FOLDER, "preprocessed")
OUTPUT_FOLDER = r"C:\Users\HP\Desktop\DA\models\models_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#Load ARIMA data
print("ARIMA Baseline")

with open(os.path.join(PREP_FOLDER, "arima_data.pkl"), "rb") as f:
    arima_data = pickle.load(f)

train_series = arima_data["train"]["avg_temp"]
test_series  = arima_data["test"]["avg_temp"]
city         = arima_data["city"]

print(f"\n  City  : {city}")
print(f"  Train : {len(train_series)} months")
print(f"  Test  : {len(test_series)} months")

#Check stationarity
print("\n  Checking stationarity (ADF test)...")
adf_result = adfuller(train_series)
print(f"  ADF Statistic : {adf_result[0]:.4f}")
print(f"  p-value       : {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("  Series is stationary (p < 0.05) — no differencing needed")
    d = 0
else:
    print("  Series is non-stationary — will use d=1 differencing in ARIMA")
    d = 1

#Train SARIMA model
# SARIMA(p,d,q)(P,D,Q,s) where s=12 for monthly seasonality
# Order (1,d,1)(1,1,1,12) is well-established for monthly climate series
print("\n  Training SARIMA(1,d,1)(1,1,1,12) ...")

model = SARIMAX(
    train_series,
    order=(1, d, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fitted_model = model.fit(disp=False)
print(f"  AIC: {fitted_model.aic:.2f}  |  BIC: {fitted_model.bic:.2f}")

# Generate predictions 
print("\n  Generating predictions on test set...")

# Forecast for the test period
forecast = fitted_model.forecast(steps=len(test_series))
forecast.index = test_series.index

# Evaluate
y_true = test_series.values
y_pred = forecast.values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - y_true.mean()) ** 2)
r2   = 1 - (ss_res / ss_tot)

print(f"ARIMA Results — {city} Monthly Temp")
print(f"  RMSE : {rmse:.4f} °C")
print(f"  MAE  : {mae:.4f} °C")
print(f"  R²   : {r2:.4f}")

# Also get in-sample fitted values for training period
train_fitted = fitted_model.fittedvalues

arima_results = {
    "model":         fitted_model,
    "city":          city,
    "train_series":  train_series,
    "test_series":   test_series,
    "train_fitted":  train_fitted,
    "test_forecast": forecast,
    "metrics": {
        "rmse": rmse,
        "mae":  mae,
        "r2":   r2,
        "aic":  fitted_model.aic,
        "bic":  fitted_model.bic,
    },
    "model_order": f"SARIMA(1,{d},1)(1,1,1,12)",
}

save_path = os.path.join(OUTPUT_FOLDER, "arima_results.pkl")
with open(save_path, "wb") as f:
    pickle.dump(arima_results, f)

print(f"\n  Saved → {save_path}")
print(f"\n  ARIMA training complete!")
print(f"  Expected: LSTM RMSE will be lower than ARIMA RMSE of {rmse:.4f} °C")