import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

#Configuration
DATA_FOLDER = r"C:\Users\HP\Desktop\DA\Data"

CITIES = {
    "Ahmedabad": "openmeteo_ahmedabad.csv",
    "Vadodara":  "openmeteo_vadodara.csv",
    "Surat":     "openmeteo_surat.csv",
    "Rajkot":    "openmeteo_rajkot.csv",
    "Bhavnagar": "openmeteo_bhavnagar.csv",
}

OUTPUT_FILE = os.path.join(DATA_FOLDER, "master_gujarat_climate.csv")

#Load each city CSV
print("Loading Open-Meteo CSVs...")

def load_city_csv(filepath, city_name):
    """
    Open-Meteo CSV structure:
      Line 0 : metadata keys   (latitude, longitude, ...)
      Line 1 : metadata values (23.02, 72.59, ...)
      Line 2 : column headers  (time, temperature_2m_max, ...)
      Line 3+: daily data
    """
    df = pd.read_csv(
        filepath,
        skiprows=2,
        encoding="latin-1",
    )

    # Standardise column names — strip units like (°C), (mm), (%) etc.
    df.columns = (
        df.columns
        .str.replace(r"\s*\(.*?\)", "", regex=True)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    rename_map = {
        "time":                       "date",
        "temperature_2m_max":         "max_temp",
        "temperature_2m_min":         "min_temp",
        "temperature_2m_mean":        "avg_temp",
        "precipitation_sum":          "rainfall_mm",
        "wind_speed_10m_max":         "wind_speed_max",
        "wind_speed_10m_mean":        "wind_speed",
        "relative_humidity_2m_mean":  "humidity",
        "dew_point_2m_mean":          "dewpoint",
        "pressure_msl_mean":          "pressure",
    }
    df.rename(columns=rename_map, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df["city"] = city_name

    num_cols = ["max_temp","min_temp","avg_temp","rainfall_mm",
                "wind_speed_max","wind_speed","humidity","dewpoint","pressure"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rainfall_mm"] = df["rainfall_mm"].fillna(0)

    keep = ["date", "city"] + [c for c in num_cols if c in df.columns]
    df = df[keep]
    return df


all_frames = []
for city_name, csv_file in CITIES.items():
    filepath = os.path.join(DATA_FOLDER, csv_file)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")
    print(f"  Loading {csv_file} ...")
    df = load_city_csv(filepath, city_name)
    all_frames.append(df)
    print(f"    OK  {city_name}: {len(df):,} rows  |  "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

master = pd.concat(all_frames, ignore_index=True)
master = master.sort_values(["city", "date"]).reset_index(drop=True)
print(f"\n  All cities loaded — {len(master):,} total rows")


#Temporal features
print("\nAdding temporal features...")

master["day"]   = master["date"].dt.day
master["month"] = master["date"].dt.month
master["year"]  = master["date"].dt.year

def get_season(month):
    if month in [3, 4, 5]:      return "Summer"
    elif month in [6, 7, 8, 9]: return "Monsoon"
    elif month in [10, 11]:     return "Post_Monsoon"
    else:                        return "Winter"

master["season"] = master["month"].apply(get_season)
print("  OK  day, month, year, season added")


#Heat index
print("\nCalculating heat index...")

def heat_index(T, RH):
    Tf   = T * 9/5 + 32
    HI_f = (-42.379
             + 2.04901523  * Tf
             + 10.14333127 * RH
             - 0.22475541  * Tf * RH
             - 0.00683783  * Tf**2
             - 0.05481717  * RH**2
             + 0.00122874  * Tf**2 * RH
             + 0.00085282  * Tf   * RH**2
             - 0.00000199  * Tf**2 * RH**2)
    HI_c = (HI_f - 32) * 5/9
    return np.where(T > 27, HI_c, T)

master["heat_index"] = heat_index(
    master["avg_temp"].values,
    master["humidity"].values
).round(3)
print("  OK  heat_index calculated")


#Lag and rolling features (per city)
print("\nAdding lag and rolling features...")

lag_roll_frames = []
for city in CITIES:
    city_df = master[master["city"] == city].copy().reset_index(drop=True)

    city_df["lag1_temp"]   = city_df["avg_temp"].shift(1)
    city_df["lag7_temp"]   = city_df["avg_temp"].shift(7)
    city_df["lag1_rain"]   = city_df["rainfall_mm"].shift(1)

    city_df["roll7_temp"]  = city_df["avg_temp"].rolling(7,  min_periods=1).mean().round(3)
    city_df["roll30_temp"] = city_df["avg_temp"].rolling(30, min_periods=1).mean().round(3)
    city_df["roll7_rain"]  = city_df["rainfall_mm"].rolling(7, min_periods=1).sum().round(3)

    lag_roll_frames.append(city_df)

master = pd.concat(lag_roll_frames, ignore_index=True)
master = master.sort_values(["city", "date"]).reset_index(drop=True)
print("  OK  lag1_temp, lag7_temp, lag1_rain, roll7_temp, roll30_temp, roll7_rain added")


#Anomaly features
print("\nCalculating anomaly features...")

monthly_temp_mean = master.groupby(["city", "month"])["avg_temp"].transform("mean")
master["temp_anomaly"] = (master["avg_temp"] - monthly_temp_mean).round(3)

monthly_rain_mean = master.groupby(["city", "month"])["rainfall_mm"].transform("mean")
master["rain_anomaly"] = (master["rainfall_mm"] - monthly_rain_mean).round(3)

master["wet_day"] = (master["rainfall_mm"] >= 2.5).astype(int)

print("  OK  temp_anomaly, rain_anomaly, wet_day added")


#Final clean and save
print("\nCleaning and saving...")

before = len(master)
master = master.dropna(subset=["avg_temp", "max_temp", "min_temp"])
dropped = before - len(master)
if dropped:
    print(f"  Dropped {dropped} rows with missing temperature values")

float_cols = master.select_dtypes(include=[float]).columns
master[float_cols] = master[float_cols].round(3)

col_order = [
    "date", "city", "year", "month", "day", "season",
    "max_temp", "min_temp", "avg_temp",
    "humidity", "pressure", "wind_speed", "wind_speed_max",
    "dewpoint", "rainfall_mm",
    "heat_index", "temp_anomaly", "rain_anomaly", "wet_day",
    "lag1_temp", "lag7_temp", "lag1_rain",
    "roll7_temp", "roll30_temp", "roll7_rain",
]
col_order = [c for c in col_order if c in master.columns]
master = master[col_order]

master.to_csv(OUTPUT_FILE, index=False)
print(f"  OK  Saved to {OUTPUT_FILE}")

#FINAL REPORT
print("\nMASTER DATASET READY")
print(f"  File          : master_gujarat_climate.csv")
print(f"  Total rows    : {len(master):,}")
print(f"  Total columns : {master.shape[1]}")
print(f"  Date range    : {master['date'].min().date()} to {master['date'].max().date()}")
print(f"  Columns       : {list(master.columns)}")

print(f"\n  Rows per city:")
for city, count in master.groupby("city").size().items():
    print(f"    {city:12s} : {count:,} days")

print(f"\n  Missing values:")
missing = master.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("    None — dataset is complete")
else:
    for col, cnt in missing.items():
        pct = cnt / len(master) * 100
        print(f"    {col:20s}: {cnt:,} ({pct:.1f}%)")

print(f"\n  Sample — Vadodara first 3 rows:")
sample = master[master["city"] == "Vadodara"].head(3)
print(sample.to_string(index=False))

print(f"\n  avg_temp stats by city:")
stats = master.groupby("city")["avg_temp"].agg(["mean","min","max"]).round(2)
print(stats.to_string())
