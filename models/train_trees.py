import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics  import (mean_squared_error, mean_absolute_error,
                               accuracy_score, precision_score,
                               recall_score, f1_score, confusion_matrix)
import xgboost as xgb

DATA_FOLDER   = r"C:\Users\HP\Desktop\DA\Data"
PREP_FOLDER   = os.path.join(DATA_FOLDER, "preprocessed")
OUTPUT_FOLDER = r"C:\Users\HP\Desktop\DA\models\models_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load master dataset 
print("Training Random Forest + XGBoost (leakage-free)")

master_path = os.path.join(DATA_FOLDER, "master_gujarat_climate.csv")
df = pd.read_csv(master_path)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

with open(os.path.join(PREP_FOLDER, "city_label_encoder.pkl"), "rb") as f:
    city_le = pickle.load(f)

season_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Post_Monsoon": 3}
df["season_enc"] = df["season"].map(season_map)
df["city_enc"]   = city_le.transform(df["city"])

#Chronological split
SPLIT_DATE = "2018-12-31"
train_df = df[df["date"] <= SPLIT_DATE].copy().reset_index(drop=True)
test_df  = df[df["date"] >  SPLIT_DATE].copy().reset_index(drop=True)
print(f"\n  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

# Leakage-free
TEMP_FEATURES = [
    "max_temp", "min_temp",
    "humidity", "pressure",
    "wind_speed", "wind_speed_max",
    "dewpoint", "rainfall_mm",
    "lag1_temp", "lag7_temp",
    "lag1_rain",
    "month", "day", "year",
    "season_enc", "city_enc",
]


RAIN_FEATURES = [
    "max_temp", "min_temp", "avg_temp",
    "humidity", "pressure",
    "wind_speed", "wind_speed_max",
    "dewpoint",
    "lag1_temp", "lag7_temp",
    "lag1_rain",
    "roll7_temp", "roll30_temp",
    "month", "day", "year",
    "season_enc", "city_enc",
]

WET_FEATURES = RAIN_FEATURES.copy()

print(f"  Temperature features : {len(TEMP_FEATURES)} (leakage-free)")
print(f"  Rainfall features    : {len(RAIN_FEATURES)} (leakage-free)")

# Build X, y for each target
X_tr_temp = train_df[TEMP_FEATURES];  X_te_temp = test_df[TEMP_FEATURES]
y_tr_temp = train_df["avg_temp"];     y_te_temp = test_df["avg_temp"]

X_tr_rain = train_df[RAIN_FEATURES];  X_te_rain = test_df[RAIN_FEATURES]
y_tr_rain = train_df["rainfall_mm"];  y_te_rain = test_df["rainfall_mm"]

X_tr_wet  = train_df[WET_FEATURES];   X_te_wet  = test_df[WET_FEATURES]
y_tr_wet  = train_df["wet_day"];      y_te_wet  = test_df["wet_day"]

neg = (y_tr_wet == 0).sum();  pos = (y_tr_wet == 1).sum()
spw = neg / pos
print(f"  Wet day ratio        : {neg:,} dry / {pos:,} wet  (spw={spw:.1f})")

# Metric helpers
def reg_metrics(yt, yp, label):
    yt, yp = np.array(yt), np.array(yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    r2   = 1 - np.sum((yt-yp)**2) / np.sum((yt-yt.mean())**2)
    print(f"    {label:30s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def cls_metrics(yt, yp, label):
    yt = np.array(yt)
    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)
    f1   = f1_score(yt, yp, zero_division=0)
    cm   = confusion_matrix(yt, yp)
    print(f"    {label:30s}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print(f"      Confusion: TN={cm[0,0]:,} FP={cm[0,1]:,} | FN={cm[1,0]:,} TP={cm[1,1]:,}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


# RANDOM FOREST
print("\nTraining Random Forest...")

print("\n  [1/3] RF Temperature (~2 min)...")
rf_temp = RandomForestRegressor(n_estimators=200, max_depth=15,
    min_samples_split=5, min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_temp.fit(X_tr_temp, y_tr_temp)
rf_tp = rf_temp.predict(X_te_temp)
rf_tm = reg_metrics(y_te_temp, rf_tp, "RF Temperature")

print("\n  [2/3] RF Rainfall...")
rf_rain = RandomForestRegressor(n_estimators=200, max_depth=15,
    min_samples_split=5, min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_rain.fit(X_tr_rain, y_tr_rain)
rf_rp = np.clip(rf_rain.predict(X_te_rain), 0, None)
rf_rm = reg_metrics(y_te_rain, rf_rp, "RF Rainfall")

print("\n  [3/3] RF Wet Day...")
rf_wet = RandomForestClassifier(n_estimators=200, max_depth=15,
    class_weight="balanced", n_jobs=-1, random_state=42)
rf_wet.fit(X_tr_wet, y_tr_wet)
rf_wp = rf_wet.predict(X_te_wet)
rf_wm = cls_metrics(y_te_wet, rf_wp, "RF Wet Day")

rf_imp = pd.Series(rf_temp.feature_importances_, index=TEMP_FEATURES).sort_values(ascending=False)
print(f"\n  RF Top 10 features (temperature):")
for feat, imp in rf_imp.head(10).items():
    print(f"    {feat:22s} {imp:.4f}  {'█'*int(imp*60)}")


# XGBOOST
print("\nTraining XGBoost...")

print("\n  [1/3] XGB Temperature...")
xgb_temp = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbosity=0)
xgb_temp.fit(X_tr_temp, y_tr_temp, verbose=False)
xgb_tp = xgb_temp.predict(X_te_temp)
xgb_tm = reg_metrics(y_te_temp, xgb_tp, "XGB Temperature")

print("\n  [2/3] XGB Rainfall...")
xgb_rain = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbosity=0)
xgb_rain.fit(X_tr_rain, y_tr_rain, verbose=False)
xgb_rp = np.clip(xgb_rain.predict(X_te_rain), 0, None)
xgb_rm = reg_metrics(y_te_rain, xgb_rp, "XGB Rainfall")

print("\n  [3/3] XGB Wet Day...")
xgb_wet = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    n_jobs=-1, random_state=42, verbosity=0, eval_metric="logloss")
xgb_wet.fit(X_tr_wet, y_tr_wet, verbose=False)
xgb_wp = xgb_wet.predict(X_te_wet)
xgb_wm = cls_metrics(y_te_wet, xgb_wp, "XGB Wet Day")

xgb_imp = pd.Series(xgb_temp.feature_importances_, index=TEMP_FEATURES).sort_values(ascending=False)
print(f"\n  XGB Top 10 features (temperature):")
for feat, imp in xgb_imp.head(10).items():
    print(f"    {feat:22s} {imp:.4f}  {'█'*int(imp*60)}")


# COMPARISON TABLE
print(f"\nFINAL MODEL COMPARISON")

print(f"\n  Temperature Regression:")
print(f"  {'Model':15s} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print(f"  {'-'*43}")
print(f"  {'Random Forest':15s} {rf_tm['rmse']:>8.4f} {rf_tm['mae']:>8.4f} {rf_tm['r2']:>8.4f}")
print(f"  {'XGBoost':15s} {xgb_tm['rmse']:>8.4f} {xgb_tm['mae']:>8.4f} {xgb_tm['r2']:>8.4f}")

print(f"\n  Rainfall Regression:")
print(f"  {'Model':15s} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print(f"  {'-'*43}")
print(f"  {'Random Forest':15s} {rf_rm['rmse']:>8.4f} {rf_rm['mae']:>8.4f} {rf_rm['r2']:>8.4f}")
print(f"  {'XGBoost':15s} {xgb_rm['rmse']:>8.4f} {xgb_rm['mae']:>8.4f} {xgb_rm['r2']:>8.4f}")

print(f"\n  Wet Day Classification:")
print(f"  {'Model':15s} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print(f"  {'-'*55}")
print(f"  {'Random Forest':15s} {rf_wm['accuracy']:>10.4f} {rf_wm['precision']:>10.4f} {rf_wm['recall']:>8.4f} {rf_wm['f1']:>8.4f}")
print(f"  {'XGBoost':15s} {xgb_wm['accuracy']:>10.4f} {xgb_wm['precision']:>10.4f} {xgb_wm['recall']:>8.4f} {xgb_wm['f1']:>8.4f}")


results = {
    "rf":  {"temp_model": rf_temp,  "rain_model": rf_rain,  "wet_model": rf_wet,
            "temp_pred": rf_tp, "rain_pred": rf_rp, "wet_pred": rf_wp,
            "temp_metrics": rf_tm, "rain_metrics": rf_rm, "wet_metrics": rf_wm,
            "importances": rf_imp},
    "xgb": {"temp_model": xgb_temp, "rain_model": xgb_rain, "wet_model": xgb_wet,
            "temp_pred": xgb_tp, "rain_pred": xgb_rp, "wet_pred": xgb_wp,
            "temp_metrics": xgb_tm, "rain_metrics": xgb_rm, "wet_metrics": xgb_wm,
            "importances": xgb_imp},
    "y_test_temp": y_te_temp.values,
    "y_test_rain": y_te_rain.values,
    "y_test_wet":  y_te_wet.values,
    "feature_names": {"temp": TEMP_FEATURES, "rain": RAIN_FEATURES},
}

save_path = os.path.join(OUTPUT_FOLDER, "tree_results.pkl")
with open(save_path, "wb") as f:
    pickle.dump(results, f)