import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.metrics  import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

DATA_FOLDER   = r"C:\Users\HP\Desktop\DA\Data"
PREP_FOLDER   = os.path.join(DATA_FOLDER, "preprocessed")
OUTPUT_FOLDER = r"C:\Users\HP\Desktop\DA\models\models_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#  Load data 
print("Isolation Forest — Anomaly Detection")

with open(os.path.join(PREP_FOLDER, "isolation_forest_data.pkl"), "rb") as f:
    if_data = pickle.load(f)

X        = if_data["X"]
meta     = if_data["meta"].reset_index(drop=True)
features = if_data["features"]

print(f"\n  Samples  : {len(X):,}")
print(f"  Features : {features}")

#Train Isolation Forest
print("\n  Training Isolation Forest (contamination=0.03)...")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
iso_forest.fit(X)

# Predict: -1 = anomaly, 1 = normal
raw_predictions = iso_forest.predict(X)
anomaly_scores  = iso_forest.decision_function(X)  # lower = more anomalous

# Convert to 0/1 flag (1 = anomaly)
meta["anomaly_flag"]  = (raw_predictions == -1).astype(int)
meta["anomaly_score"] = anomaly_scores.round(4)

total_anomalies = meta["anomaly_flag"].sum()
print(f"  Total anomalies detected : {total_anomalies:,} ({total_anomalies/len(meta)*100:.1f}%)")

# Anomaly breakdown by city
print(f"\n  Anomalies by city:")
city_anom = meta.groupby("city")["anomaly_flag"].agg(["sum","count"])
city_anom["pct"] = (city_anom["sum"] / city_anom["count"] * 100).round(2)
city_anom.columns = ["anomalies", "total", "pct%"]
print(city_anom.to_string())

# Anomaly breakdown by type
# Classify each anomaly as temperature or rainfall based anomaly
anomalies = meta[meta["anomaly_flag"] == 1].copy()

temp_anom = anomalies[anomalies["temp_anomaly"].abs() > anomalies["rain_anomaly"].abs()]
rain_anom = anomalies[anomalies["rain_anomaly"].abs() >= anomalies["temp_anomaly"].abs()]

print(f"\n  Anomaly type breakdown:")
print(f"    Temperature-driven anomalies : {len(temp_anom):,}")
print(f"    Rainfall-driven anomalies    : {len(rain_anom):,}")

# Validate against known Gujarat extreme events 
print(f"\n  Validating against known Gujarat extreme events...")

# Known extreme event periods for Gujarat (year, month_start, month_end, type)
KNOWN_EVENTS = [
    (2010,  4,  6, "Heatwave"),
    (2015,  5,  6, "Heatwave"),
    (2016,  4,  6, "Heatwave"),
    (2017,  7,  9, "Flood"),
    (2019,  4,  6, "Heatwave"),
    (2019,  7,  9, "Flood"),
    (2020,  7,  9, "Flood"),
    (2022,  4,  6, "Heatwave"),
    (2023,  7,  9, "Flood"),
]

validation_results = []
for year, m_start, m_end, event_type in KNOWN_EVENTS:
    # Get anomalies detected in this event window (any city)
    event_mask = (
        (meta["date"].dt.year  == year) &
        (meta["date"].dt.month >= m_start) &
        (meta["date"].dt.month <= m_end)
    )
    event_days     = event_mask.sum()
    detected_days  = meta[event_mask & (meta["anomaly_flag"] == 1)].shape[0]
    detection_rate = detected_days / event_days * 100 if event_days > 0 else 0

    flag = "DETECTED" if detection_rate >= 10 else "MISSED"
    print(f"  {year} {event_type:10s} ({m_start:02d}-{m_end:02d}) : "
          f"{detected_days:3d}/{event_days:3d} days flagged  "
          f"({detection_rate:5.1f}%)  [{flag}]")

    validation_results.append({
        "year": year, "event": event_type,
        "event_days": event_days, "detected_days": detected_days,
        "detection_rate": detection_rate, "detected": detection_rate >= 10,
    })

val_df = pd.DataFrame(validation_results)
detected_count = val_df["detected"].sum()
print(f"\n  Events detected : {detected_count}/{len(KNOWN_EVENTS)}")
print(f"  Detection rate  : {detected_count/len(KNOWN_EVENTS)*100:.1f}%")

# Top 20 most anomalous days (Ahmedabad)
print(f"\n  Top 15 most anomalous days (Ahmedabad):")
ahm_anom = meta[
    (meta["city"] == "Ahmedabad") & (meta["anomaly_flag"] == 1)
].sort_values("anomaly_score").head(15)

print(f"  {'Date':12s} {'Temp°C':>8} {'Rain mm':>9} {'Temp Δ':>8} {'Rain Δ':>8}")
print(f"  {'─'*55}")
for _, row in ahm_anom.iterrows():
    print(f"  {str(row['date'])[:10]:12s} "
          f"{row['avg_temp']:>8.1f} "
          f"{row['rainfall_mm']:>9.1f} "
          f"{row['temp_anomaly']:>+8.2f} "
          f"{row['rain_anomaly']:>+8.2f}")

#Save results
if_results = {
    "model":              iso_forest,
    "predictions":        meta,
    "features":           features,
    "validation":         val_df,
    "total_anomalies":    total_anomalies,
    "city_breakdown":     city_anom,
    "metrics": {
        "total_anomalies":   int(total_anomalies),
        "anomaly_pct":       round(total_anomalies/len(meta)*100, 2),
        "events_detected":   int(detected_count),
        "events_total":      len(KNOWN_EVENTS),
        "event_detection_rate": round(detected_count/len(KNOWN_EVENTS)*100, 1),
    },
}

save_path = os.path.join(OUTPUT_FOLDER, "isolation_forest_results.pkl")
with open(save_path, "wb") as f:
    pickle.dump(if_results, f)

# Also save anomaly predictions as CSV for easy inspection
csv_path = os.path.join(DATA_FOLDER, "anomaly_predictions.csv")
meta.to_csv(csv_path, index=False)

print(f"\n  Saved → {save_path}")
print(f"  Saved → {csv_path}  (open in Excel to inspect)")