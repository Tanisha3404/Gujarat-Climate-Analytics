"""
STEP 3 — Preprocessing
Prepares master_gujarat_climate.csv for all ML models:
  - LSTM / CNN-LSTM  → scaled sequences (X_train, y_train, X_test, y_test)
  - XGBoost / RF     → scaled tabular data
  - ARIMA            → single city time-series
  - Isolation Forest → anomaly detection features

Folder : C:\\Users\\HP\\Desktop\\DA\\Data
Run as : python preprocessing.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#Configuration
DATA_FOLDER  = r"C:\Users\HP\Desktop\DA\Data"
INPUT_FILE   = os.path.join(DATA_FOLDER, "master_gujarat_climate.csv")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "preprocessed")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# LSTM sequence length — use past 30 days to predict next day
SEQ_LEN = 30

# Train / test split — 80% train (1990–2018), 20% test (2019–2024)
SPLIT_DATE = "2018-12-31"

# City to use for ARIMA (single city time-series model)
ARIMA_CITY = "Ahmedabad"

#Load  dataset 
print("Preprocessing")
print(f"\nLoading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
print(f"  Loaded: {len(df):,} rows, {df.shape[1]} columns")


#Cleaning
print("\nBasic cleaning...")

# Fill lag NaNs at series start with forward fill (only affects first 7 rows per city)
lag_cols = ["lag1_temp", "lag7_temp", "lag1_rain"]
df[lag_cols] = df.groupby("city")[lag_cols].transform(lambda x: x.bfill())

# Confirm no missing values remain in core columns
core_cols = ["max_temp","min_temp","avg_temp","humidity","pressure",
             "wind_speed","dewpoint","rainfall_mm","heat_index"]
missing = df[core_cols].isnull().sum().sum()
print(f"  Missing values in core columns: {missing}")
if missing == 0:
    print("  No missing values in core columns")
else:
    # Fill any remaining with column median per city
    for col in core_cols:
        df[col] = df.groupby("city")[col].transform(lambda x: x.fillna(x.median()))
    print("  Filled remaining missing values with city median")


#Encode categorical columns
print("\n Encoding categorical columns...")

# Season → integer (for tree models)
season_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Post_Monsoon": 3}
df["season_enc"] = df["season"].map(season_map)

# City → integer (for tree models)
city_le = LabelEncoder()
df["city_enc"] = city_le.fit_transform(df["city"])

# Save the city label encoder for dashboard use later
le_path = os.path.join(OUTPUT_FOLDER, "city_label_encoder.pkl")
with open(le_path, "wb") as f:
    pickle.dump(city_le, f)

print(f"  Season encoding: {season_map}")
print(f"  City encoding  : {dict(zip(city_le.classes_, city_le.transform(city_le.classes_)))}")
print(f"  Encoders saved to {le_path}")


#Define feature sets
print("\nDefining feature sets...")

# Features for LSTM (temperature forecasting)
LSTM_FEATURES = [
    "avg_temp", "max_temp", "min_temp",
    "humidity", "pressure", "wind_speed",
    "dewpoint", "rainfall_mm", "heat_index",
    "lag1_temp", "lag7_temp", "roll7_temp", "roll30_temp",
    "month", "day", "season_enc"
]
LSTM_TARGET = "avg_temp"

# Features for XGBoost / Random Forest
TREE_FEATURES = [
    "max_temp", "min_temp", "avg_temp",
    "humidity", "pressure", "wind_speed", "wind_speed_max",
    "dewpoint", "rainfall_mm", "heat_index",
    "lag1_temp", "lag7_temp", "lag1_rain",
    "roll7_temp", "roll30_temp", "roll7_rain",
    "temp_anomaly", "rain_anomaly",
    "month", "day", "year", "season_enc", "city_enc"
]
TREE_TARGET_TEMP = "avg_temp"
TREE_TARGET_RAIN = "rainfall_mm"
TREE_TARGET_WET  = "wet_day"        # classification target

# Features for Isolation Forest (anomaly detection)
IF_FEATURES = [
    "avg_temp", "max_temp", "rainfall_mm",
    "humidity", "heat_index",
    "temp_anomaly", "rain_anomaly",
    "roll7_temp", "roll7_rain"
]

print(f"  LSTM features  : {len(LSTM_FEATURES)} features")
print(f"  Tree features  : {len(TREE_FEATURES)} features")
print(f"  IF features    : {len(IF_FEATURES)} features")


#Train / test split
print("\nTrain / test split (chronological)...")

train_df = df[df["date"] <= SPLIT_DATE].copy()
test_df  = df[df["date"] >  SPLIT_DATE].copy()

print(f"  Train: {len(train_df):,} rows  ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"  Test : {len(test_df):,}  rows  ({test_df['date'].min().date()}  to {test_df['date'].max().date()})")
print(f"  Split: {len(train_df)/len(df)*100:.1f}% train / {len(test_df)/len(df)*100:.1f}% test")


#Prepare LSTM data (per city, sliding window sequences)
print("\n Preparing LSTM sequences...")

def make_sequences(data, seq_len, feature_cols, target_col, scaler):
    """Create sliding window sequences for LSTM."""
    scaled = scaler.transform(data[feature_cols])
    target_idx = feature_cols.index(target_col)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])          # past seq_len days
        y.append(scaled[i][target_idx])           # next day target

    return np.array(X), np.array(y)

lstm_city_data = {}

for city in df["city"].unique():
    city_train = train_df[train_df["city"] == city].sort_values("date").reset_index(drop=True)
    city_test  = test_df[test_df["city"]  == city].sort_values("date").reset_index(drop=True)

    # Fit scaler on training data only — prevent data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(city_train[LSTM_FEATURES])

    # Build sequences
    X_train, y_train = make_sequences(city_train, SEQ_LEN, LSTM_FEATURES, LSTM_TARGET, scaler)

    # For test sequences we append last SEQ_LEN rows of train to start of test
    # so we have enough history for the first test sequence
    city_test_full = pd.concat([
        city_train.tail(SEQ_LEN),
        city_test
    ]).reset_index(drop=True)
    X_test, y_test = make_sequences(city_test_full, SEQ_LEN, LSTM_FEATURES, LSTM_TARGET, scaler)

    lstm_city_data[city] = {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
    }

    print(f"  {city:12s} → X_train: {X_train.shape}  X_test: {X_test.shape}")

# Save LSTM data
lstm_path = os.path.join(OUTPUT_FOLDER, "lstm_data.pkl")
with open(lstm_path, "wb") as f:
    pickle.dump(lstm_city_data, f)
print(f"\n  LSTM sequences saved to lstm_data.pkl")
print(f"  Shape explanation: (samples, {SEQ_LEN} days, {len(LSTM_FEATURES)} features)")


#Prepare XGBoost / Random Forest data (all cities combined)
print("\nPreparing XGBoost / Random Forest data...")

# --- Temperature regression ---
X_train_temp = train_df[TREE_FEATURES].copy()
y_train_temp = train_df[TREE_TARGET_TEMP].copy()
X_test_temp  = test_df[TREE_FEATURES].copy()
y_test_temp  = test_df[TREE_TARGET_TEMP].copy()

# Fit StandardScaler on train only
scaler_tree = StandardScaler()
X_train_temp_scaled = scaler_tree.fit_transform(X_train_temp)
X_test_temp_scaled  = scaler_tree.transform(X_test_temp)

# Save as DataFrames with column names preserved
X_train_temp_df = pd.DataFrame(X_train_temp_scaled, columns=TREE_FEATURES)
X_test_temp_df  = pd.DataFrame(X_test_temp_scaled,  columns=TREE_FEATURES)

# --- Rainfall regression ---
# Use same scaler (features are the same)
X_train_rain_df = X_train_temp_df.copy()
X_test_rain_df  = X_test_temp_df.copy()
y_train_rain    = train_df[TREE_TARGET_RAIN].copy().reset_index(drop=True)
y_test_rain     = test_df[TREE_TARGET_RAIN].copy().reset_index(drop=True)

# --- Wet day classification ---
y_train_wet = train_df[TREE_TARGET_WET].copy().reset_index(drop=True)
y_test_wet  = test_df[TREE_TARGET_WET].copy().reset_index(drop=True)

# Reset indices
y_train_temp = y_train_temp.reset_index(drop=True)
y_test_temp  = y_test_temp.reset_index(drop=True)

# Save tree model data
tree_data = {
    "X_train": X_train_temp_df,
    "X_test":  X_test_temp_df,
    "y_train_temp": y_train_temp,
    "y_test_temp":  y_test_temp,
    "y_train_rain": y_train_rain,
    "y_test_rain":  y_test_rain,
    "y_train_wet":  y_train_wet,
    "y_test_wet":   y_test_wet,
    "scaler":       scaler_tree,
    "feature_names": TREE_FEATURES,
}
tree_path = os.path.join(OUTPUT_FOLDER, "tree_data.pkl")
with open(tree_path, "wb") as f:
    pickle.dump(tree_data, f)

print(f"  X_train shape : {X_train_temp_df.shape}")
print(f"  X_test shape  : {X_test_temp_df.shape}")
print(f"  y_train_temp  : {len(y_train_temp):,} values")
print(f"  y_train_rain  : {len(y_train_rain):,} values  (regression)")
print(f"  y_train_wet   : {y_train_wet.value_counts().to_dict()}  (classification: 0=dry, 1=wet)")
print(f"  Tree data saved to tree_data.pkl")


# Prepare ARIMA data (single city, monthly avg temperature)
print("\nPreparing ARIMA data...")


arima_df = df[df["city"] == ARIMA_CITY].sort_values("date").copy()

# Monthly average temperature time-series
arima_monthly = (
    arima_df.groupby(["year", "month"])["avg_temp"]
    .mean()
    .reset_index()
)
arima_monthly["date"] = pd.to_datetime(
    arima_monthly[["year", "month"]].assign(day=1)
)
arima_monthly = arima_monthly[["date", "avg_temp"]].set_index("date")

# Split: train up to Dec 2018, test from Jan 2019
arima_train = arima_monthly[arima_monthly.index <= "2018-12-31"]
arima_test  = arima_monthly[arima_monthly.index >  "2018-12-31"]

arima_path = os.path.join(OUTPUT_FOLDER, "arima_data.pkl")
with open(arima_path, "wb") as f:
    pickle.dump({"train": arima_train, "test": arima_test, "city": ARIMA_CITY}, f)

print(f"  City           : {ARIMA_CITY}")
print(f"  Monthly series : {len(arima_monthly)} months total")
print(f"  Train          : {len(arima_train)} months (up to Dec 2018)")
print(f"  Test           : {len(arima_test)} months (Jan 2019 onward)")
print(f"  ARIMA data saved to arima_data.pkl")


#Prepare Isolation Forest data
print("\nPreparing Isolation Forest data...")

# Isolation Forest is unsupervised — trained on full dataset (no split needed)
# We use all cities combined
if_data = df[IF_FEATURES].copy()

# Scale with StandardScaler
scaler_if = StandardScaler()
if_scaled = scaler_if.fit_transform(if_data)
if_df = pd.DataFrame(if_scaled, columns=IF_FEATURES)

# Keep metadata for result interpretation
if_meta = df[["date", "city", "avg_temp", "rainfall_mm", "temp_anomaly", "rain_anomaly"]].copy()

if_path = os.path.join(OUTPUT_FOLDER, "isolation_forest_data.pkl")
with open(if_path, "wb") as f:
    pickle.dump({
        "X":       if_df,
        "meta":    if_meta,
        "scaler":  scaler_if,
        "features": IF_FEATURES,
    }, f)

print(f"  Features       : {IF_FEATURES}")
print(f"  Total samples  : {len(if_df):,}")
print(f"  Isolation Forest data saved to isolation_forest_data.pkl")

#SUMMARY
print(f"  Folder: {OUTPUT_FOLDER}")
print(f"\n  Files created:")
for fname in sorted(os.listdir(OUTPUT_FOLDER)):
    fpath = os.path.join(OUTPUT_FOLDER, fname)
    size  = os.path.getsize(fpath) / 1024
    print(f"    {fname:40s} {size:8.1f} KB")