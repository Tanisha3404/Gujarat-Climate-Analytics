import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Paths
DATA_FOLDER    = r"C:\Users\HP\Desktop\DA\Data"
MODELS_FOLDER  = r"C:\Users\HP\Desktop\DA\models\models_results"
OUTPUT_FOLDER  = r"C:\Users\HP\Desktop\DA\report_charts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8FBFF",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
})

CITY_COLORS = {
    "Ahmedabad": "#E63946",
    "Vadodara":  "#457B9D",
    "Surat":     "#2A9D8F",
    "Rajkot":    "#E9C46A",
    "Bhavnagar": "#6A0572",
}
CITIES = list(CITY_COLORS.keys())

def save(fig, filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")

# Load data 
print("Loading data and model results...")
df = pd.read_csv(os.path.join(DATA_FOLDER, "master_gujarat_climate.csv"))
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

with open(os.path.join(MODELS_FOLDER, "lstm_results.pkl"),      "rb") as f: lstm_res  = pickle.load(f)
with open(os.path.join(MODELS_FOLDER, "tree_results.pkl"),      "rb") as f: tree_res  = pickle.load(f)
with open(os.path.join(MODELS_FOLDER, "arima_results.pkl"),     "rb") as f: arima_res = pickle.load(f)
with open(os.path.join(MODELS_FOLDER, "isolation_forest_results.pkl"), "rb") as f: if_res = pickle.load(f)
with open(os.path.join(MODELS_FOLDER, "hypothesis_results.pkl"), "rb") as f: hyp_res  = pickle.load(f)

anomaly_df = pd.read_csv(os.path.join(DATA_FOLDER, "anomaly_predictions.csv"))
anomaly_df["date"] = pd.to_datetime(anomaly_df["date"])
print("  All files loaded.\n")


# CHART 1 — Gujarat Annual Temperature Trend (Mann-Kendall)
print("Generating Chart 1: Gujarat temperature trend...")
annual = df.groupby("year")["avg_temp"].mean().reset_index()
years  = annual["year"].values
temps  = annual["avg_temp"].values
sen_slope = hyp_res["H3"]["Gujarat_all"]["sen_slope"]
p_val     = hyp_res["H3"]["Gujarat_all"]["p_value"]

# Sen's slope line
trend_line = temps[0] + sen_slope * (years - years[0])

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(years, temps, color="#457B9D", s=50, zorder=5, label="Annual mean temp")
ax.plot(years, temps, color="#457B9D", alpha=0.4, linewidth=1)
ax.plot(years, trend_line, color="#E63946", linewidth=2.5, linestyle="--",
        label=f"Sen's slope = +{sen_slope:.4f}°C/yr  |  p = {p_val:.4f}")
ax.fill_between(years, trend_line - 0.5, trend_line + 0.5, alpha=0.1, color="#E63946")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Temperature (°C)")
ax.set_title("Gujarat Annual Mean Temperature Trend — Mann-Kendall Analysis (1990–2024)",
             fontweight="bold", pad=12)
ax.legend()
ax.text(0.02, 0.95,
        f"Total warming 1990–2024: +{sen_slope*(2024-1990):.2f}°C\n"
        f"Rate: +{sen_slope*10:.3f}°C/decade\np-value: {p_val:.4f} (Significant ✓)",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3CD", alpha=0.8))
save(fig, "01_gujarat_temperature_trend.png")


# CHART 2 — City-wise Temperature Trends (5 subplots)
print("Generating Chart 2: City-wise temperature trends...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for idx, city in enumerate(CITIES):
    ax = axes[idx]
    city_annual = df[df["city"] == city].groupby("year")["avg_temp"].mean()
    yrs  = city_annual.index.values
    tmps = city_annual.values
    h3   = hyp_res["H3"].get(city, {})
    slope = h3.get("sen_slope", 0)
    pv    = h3.get("p_value", 1)
    trend = tmps[0] + slope * (yrs - yrs[0])
    color = CITY_COLORS[city]
    ax.plot(yrs, tmps, color=color, alpha=0.6, linewidth=1.5, marker="o", markersize=3)
    ax.plot(yrs, trend, color="black", linewidth=2, linestyle="--")
    sig = "✓ Significant" if pv < 0.05 else "✗ Not significant"
    ax.set_title(f"{city}\n{slope:+.4f}°C/yr  |  p={pv:.3f}  {sig}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Avg Temp (°C)", fontsize=8)
axes[-1].set_visible(False)
fig.suptitle("City-wise Annual Temperature Trends (1990–2024)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "02_citywise_temperature_trends.png")


# CHART 3 — LSTM Actual vs Predicted (all 5 cities, subplots)
print("Generating Chart 3: LSTM actual vs predicted...")
fig, axes = plt.subplots(5, 1, figsize=(14, 20))
for idx, city in enumerate(CITIES):
    ax = axes[idx]
    ev = lstm_res[city]["lstm_eval"]
    y_true = ev["y_true"][:365]   # show first year of test period
    y_pred = ev["y_pred"][:365]
    rmse   = lstm_res[city]["lstm_eval"]["rmse"]
    r2     = lstm_res[city]["lstm_eval"]["r2"]
    x      = np.arange(len(y_true))
    ax.plot(x, y_true, color="#2C3E50", linewidth=1.2, label="Actual", alpha=0.9)
    ax.plot(x, y_pred, color=CITY_COLORS[city], linewidth=1.2,
            linestyle="--", label="LSTM Predicted", alpha=0.9)
    ax.fill_between(x, y_true, y_pred, alpha=0.1, color=CITY_COLORS[city])
    ax.set_title(f"{city}  —  RMSE: {rmse:.4f}°C  |  R²: {r2:.4f}",
                 fontweight="bold", fontsize=11)
    ax.set_ylabel("Avg Temp (°C)")
    ax.legend(loc="upper right", fontsize=8)
    if idx == 4:
        ax.set_xlabel("Days (Test Period 2019–2020, first 365 days shown)")
fig.suptitle("LSTM: Actual vs Predicted Temperature — All Gujarat Cities",
             fontsize=14, fontweight="bold", y=1.002)
plt.tight_layout()
save(fig, "03_lstm_actual_vs_predicted.png")


# CHART 4 — CNN-LSTM vs LSTM RMSE Comparison per city
print("Generating Chart 4: CNN-LSTM vs LSTM comparison...")
lstm_rmse    = [lstm_res[c]["lstm_eval"]["rmse"]     for c in CITIES]
cnn_rmse     = [lstm_res[c]["cnn_lstm_eval"]["rmse"] for c in CITIES]
lstm_r2      = [lstm_res[c]["lstm_eval"]["r2"]       for c in CITIES]
cnn_r2       = [lstm_res[c]["cnn_lstm_eval"]["r2"]   for c in CITIES]

x = np.arange(len(CITIES))
w = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

bars1 = ax1.bar(x - w/2, lstm_rmse, w, label="LSTM",     color="#457B9D", alpha=0.85)
bars2 = ax1.bar(x + w/2, cnn_rmse,  w, label="CNN-LSTM", color="#E63946", alpha=0.85)
for bar in bars1: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(CITIES, rotation=15)
ax1.set_ylabel("RMSE (°C)"); ax1.set_title("LSTM vs CNN-LSTM — RMSE (lower is better)", fontweight="bold")
ax1.legend()

bars3 = ax2.bar(x - w/2, lstm_r2, w, label="LSTM",     color="#457B9D", alpha=0.85)
bars4 = ax2.bar(x + w/2, cnn_r2,  w, label="CNN-LSTM", color="#E63946", alpha=0.85)
for bar in bars3: ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars4: ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels(CITIES, rotation=15)
ax2.set_ylabel("R²"); ax2.set_ylim(0.8, 1.0)
ax2.set_title("LSTM vs CNN-LSTM — R² (higher is better)", fontweight="bold")
ax2.legend()
fig.suptitle("Deep Learning Model Comparison: LSTM vs CNN-LSTM", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "04_lstm_vs_cnnlstm_comparison.png")


# CHART 5 — All Models RMSE Comparison
print("Generating Chart 5: All models RMSE comparison...")
models      = ["ARIMA\n(Monthly)", "Random\nForest", "XGBoost", "LSTM\n(avg)", "CNN-LSTM\n(avg)"]
rmse_vals   = [
    arima_res["metrics"]["rmse"],
    tree_res["rf"]["temp_metrics"]["rmse"],
    tree_res["xgb"]["temp_metrics"]["rmse"],
    np.mean([lstm_res[c]["lstm_eval"]["rmse"]     for c in CITIES]),
    np.mean([lstm_res[c]["cnn_lstm_eval"]["rmse"] for c in CITIES]),
]
r2_vals     = [
    arima_res["metrics"]["r2"],
    tree_res["rf"]["temp_metrics"]["r2"],
    tree_res["xgb"]["temp_metrics"]["r2"],
    np.mean([lstm_res[c]["lstm_eval"]["r2"]       for c in CITIES]),
    np.mean([lstm_res[c]["cnn_lstm_eval"]["r2"]   for c in CITIES]),
]
colors_bar = ["#6C757D", "#2A9D8F", "#E9C46A", "#457B9D", "#E63946"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
bars = ax1.bar(models, rmse_vals, color=colors_bar, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, rmse_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_ylabel("RMSE (°C)"); ax1.set_title("RMSE Comparison — All Models (lower is better)",
                                            fontweight="bold")

bars2 = ax2.bar(models, r2_vals, color=colors_bar, alpha=0.85, edgecolor="white")
for bar, val in zip(bars2, r2_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_ylabel("R²"); ax2.set_ylim(0.85, 1.01)
ax2.set_title("R² Comparison — All Models (higher is better)", fontweight="bold")
fig.suptitle("Temperature Forecasting: Complete Model Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "05_all_models_rmse_comparison.png")


# CHART 6 — XGBoost Feature Importance
print("Generating Chart 6: XGBoost feature importance...")
xgb_imp = tree_res["xgb"]["importances"].head(15).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = ["#E63946" if v > 0.1 else "#457B9D" if v > 0.02 else "#ADB5BD"
             for v in xgb_imp.values]
bars = ax.barh(xgb_imp.index, xgb_imp.values, color=colors_fi, edgecolor="white")
for bar, val in zip(bars, xgb_imp.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance Score")
ax.set_title("XGBoost Feature Importance — Temperature Prediction\n"
             "(Top 15 features)", fontweight="bold")
ax.axvline(x=0.01, color="gray", linestyle=":", alpha=0.5)
save(fig, "06_xgboost_feature_importance.png")


# CHART 7 — Random Forest Feature Importance
print("Generating Chart 7: Random Forest feature importance...")
rf_imp = tree_res["rf"]["importances"].head(15).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
colors_rf = ["#2A9D8F" if v > 0.05 else "#457B9D" if v > 0.01 else "#ADB5BD"
             for v in rf_imp.values]
bars = ax.barh(rf_imp.index, rf_imp.values, color=colors_rf, edgecolor="white")
for bar, val in zip(bars, rf_imp.values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance Score")
ax.set_title("Random Forest Feature Importance — Temperature Prediction\n"
             "(Top 15 features)", fontweight="bold")
save(fig, "07_rf_feature_importance.png")


# CHART 8 — Rainfall Prediction: Actual vs Predicted (XGBoost)
print("Generating Chart 8: Rainfall prediction scatter...")
y_true_r = tree_res["y_test_rain"]
y_pred_r = tree_res["xgb"]["rain_pred"]
# Sample 2000 points to avoid overcrowding
idx_sample = np.random.choice(len(y_true_r), size=2000, replace=False)
yt_s = y_true_r[idx_sample]
yp_s = y_pred_r[idx_sample]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.scatter(yt_s, yp_s, alpha=0.3, s=10, color="#E63946")
max_val = max(yt_s.max(), yp_s.max())
ax1.plot([0, max_val], [0, max_val], "k--", linewidth=1.5, label="Perfect prediction")
ax1.set_xlabel("Actual Rainfall (mm)")
ax1.set_ylabel("Predicted Rainfall (mm)")
ax1.set_title("XGBoost: Actual vs Predicted Rainfall\n(2000 random test samples)",
              fontweight="bold")
ax1.legend()
rmse_r = tree_res["xgb"]["rain_metrics"]["rmse"]
r2_r   = tree_res["xgb"]["rain_metrics"]["r2"]
ax1.text(0.05, 0.95, f"RMSE: {rmse_r:.2f} mm\nR²: {r2_r:.4f}",
         transform=ax1.transAxes, fontsize=10, verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="#FFF3CD", alpha=0.8))

# RF vs XGB rainfall metrics bar
metrics_labels = ["RMSE (mm)", "MAE (mm)"]
rf_vals  = [tree_res["rf"]["rain_metrics"]["rmse"],  tree_res["rf"]["rain_metrics"]["mae"]]
xgb_vals = [tree_res["xgb"]["rain_metrics"]["rmse"], tree_res["xgb"]["rain_metrics"]["mae"]]
x2 = np.arange(2)
ax2.bar(x2 - 0.2, rf_vals,  0.35, label="Random Forest", color="#2A9D8F", alpha=0.85)
ax2.bar(x2 + 0.2, xgb_vals, 0.35, label="XGBoost",       color="#E63946", alpha=0.85)
ax2.set_xticks(x2); ax2.set_xticklabels(metrics_labels)
ax2.set_title("RF vs XGBoost: Rainfall Regression Metrics", fontweight="bold")
ax2.set_ylabel("Error (mm)")
ax2.legend()
for i, (rv, xv) in enumerate(zip(rf_vals, xgb_vals)):
    ax2.text(i-0.2, rv+0.05, f"{rv:.2f}", ha="center", fontsize=9)
    ax2.text(i+0.2, xv+0.05, f"{xv:.2f}", ha="center", fontsize=9)
fig.suptitle("Rainfall Prediction Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "08_rainfall_prediction_analysis.png")


# CHART 9 — Wet Day Confusion Matrices (RF and XGBoost side by side)
print("Generating Chart 9: Confusion matrices...")
rf_cm  = tree_res["rf"]["wet_metrics"]["confusion_matrix"]
xgb_cm = tree_res["xgb"]["wet_metrics"]["confusion_matrix"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for ax, cm, title in [(ax1, rf_cm, "Random Forest"), (ax2, xgb_cm, "XGBoost")]:
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Predicted Dry", "Predicted Wet"])
    ax.set_yticklabels(["Actual Dry", "Actual Wet"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm_pct[i,j] > 50 else "black",
                    fontweight="bold")
    m = tree_res["rf" if ax==ax1 else "xgb"]["wet_metrics"]
    ax.set_title(f"{title}\nAcc:{m['accuracy']:.3f} | Prec:{m['precision']:.3f} | "
                 f"Rec:{m['recall']:.3f} | F1:{m['f1']:.3f}", fontweight="bold", fontsize=10)
plt.colorbar(im, ax=ax2, label="% of row")
fig.suptitle("Wet Day Classification — Confusion Matrices", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "09_wet_day_confusion_matrices.png")


# CHART 10 — Isolation Forest Anomaly Score Timeline (Ahmedabad)
print("Generating Chart 10: Isolation Forest anomaly timeline...")
ahm = anomaly_df[anomaly_df["city"] == "Ahmedabad"].copy()
ahm = ahm.sort_values("date")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
# Top: temperature with anomalies highlighted
ax1.plot(ahm["date"], ahm["avg_temp"], color="#457B9D", linewidth=0.6, alpha=0.7)
anom_ahm = ahm[ahm["anomaly_flag"] == 1]
ax1.scatter(anom_ahm["date"], anom_ahm["avg_temp"],
            color="#E63946", s=15, zorder=5, label="Anomaly flagged", alpha=0.8)
ax1.set_ylabel("Avg Temperature (°C)")
ax1.set_title("Ahmedabad: Temperature with Isolation Forest Anomaly Flags (1990–2024)",
              fontweight="bold")
ax1.legend()

# Bottom: rainfall with anomalies highlighted
ax2.fill_between(ahm["date"], ahm["rainfall_mm"], alpha=0.5, color="#2A9D8F")
ax2.scatter(anom_ahm["date"], anom_ahm["rainfall_mm"],
            color="#E63946", s=15, zorder=5, alpha=0.8)
ax2.set_ylabel("Rainfall (mm)")
ax2.set_xlabel("Year")
ax2.set_title("Ahmedabad: Rainfall with Anomaly Flags", fontweight="bold")
plt.tight_layout()
save(fig, "10_isolation_forest_anomaly_timeline.png")


# CHART 11 — Top 15 Anomalous Days (Ahmedabad)
print("Generating Chart 11: Top anomalous days...")
top_anom = ahm[ahm["anomaly_flag"] == 1].sort_values("anomaly_score").head(15)
labels = [str(d)[:10] for d in top_anom["date"]]
rain_v = top_anom["rainfall_mm"].values
temp_v = top_anom["avg_temp"].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
colors_anom = ["#E63946" if r > 100 else "#E9C46A" for r in rain_v]
bars = ax1.barh(labels, rain_v, color=colors_anom, edgecolor="white")
for bar, val in zip(bars, rain_v):
    ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
             f"{val:.1f} mm", va="center", fontsize=8)
ax1.set_xlabel("Rainfall (mm)")
ax1.set_title("Top 15 Most Anomalous Days — Ahmedabad\n(by Isolation Forest score)",
              fontweight="bold")
ax1.invert_yaxis()

ax2.barh(labels, temp_v, color="#457B9D", edgecolor="white")
for i, (bar, val) in enumerate(zip(ax2.patches, temp_v)):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}°C", va="center", fontsize=8)
ax2.set_xlabel("Average Temperature (°C)")
ax2.set_title("Temperature on Top Anomalous Days", fontweight="bold")
ax2.invert_yaxis()
plt.tight_layout()
save(fig, "11_top_anomalous_days.png")


# CHART 12 — Decadal Temperature Comparison
print("Generating Chart 12: Decadal temperature comparison...")
def decade(year):
    if year < 2000: return "1990s"
    elif year < 2010: return "2000s"
    elif year < 2020: return "2010s"
    else: return "2020s"

df["decade"] = df["year"].apply(decade)
decade_city = df.groupby(["decade", "city"])["avg_temp"].mean().reset_index()
decade_order = ["1990s", "2000s", "2010s", "2020s"]

fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
for idx, city in enumerate(CITIES):
    ax = axes[idx]
    city_d = decade_city[decade_city["city"] == city]
    city_d = city_d.set_index("decade").reindex(decade_order).reset_index()
    palette = ["#D4E6F1", "#85C1E9", "#2E86C1", "#1B4F72"]
    bars = ax.bar(city_d["decade"], city_d["avg_temp"],
                  color=palette, edgecolor="white", alpha=0.9)
    for bar, val in zip(bars, city_d["avg_temp"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    diff = city_d["avg_temp"].iloc[-1] - city_d["avg_temp"].iloc[0]
    ax.set_title(f"{city}\n(Δ={diff:+.2f}°C)", fontweight="bold", fontsize=10)
    ax.set_xticklabels(decade_order, rotation=30, fontsize=8)
    if idx == 0: ax.set_ylabel("Avg Temperature (°C)")
fig.suptitle("Decadal Average Temperature — Gujarat Cities (1990s to 2020s)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "12_decadal_temperature_comparison.png")


# CHART 13 — Seasonal Temperature Boxplots per City
print("Generating Chart 13: Seasonal temperature boxplots...")
season_order = ["Winter", "Summer", "Monsoon", "Post_Monsoon"]
season_colors = {"Winter": "#AED6F1", "Summer": "#E74C3C",
                 "Monsoon": "#27AE60", "Post_Monsoon": "#F39C12"}

fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
for idx, city in enumerate(CITIES):
    ax = axes[idx]
    city_df = df[df["city"] == city]
    data_by_season = [city_df[city_df["season"] == s]["avg_temp"].values
                      for s in season_order]
    bp = ax.boxplot(data_by_season, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, season in zip(bp["boxes"], season_order):
        patch.set_facecolor(season_colors[season])
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["Win", "Sum", "Mon", "Post"], fontsize=9)
    ax.set_title(city, fontweight="bold")
    if idx == 0: ax.set_ylabel("Avg Temperature (°C)")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=season_colors[s], alpha=0.8, label=s)
                   for s in season_order]
axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=8)
fig.suptitle("Seasonal Temperature Distribution — Gujarat Cities (1990–2024)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "13_seasonal_temperature_boxplots.png")


# CHART 14 — Monthly Average Temperature Heatmap (all cities)
print("Generating Chart 14: Monthly temperature heatmap...")
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
heatmap_data = df.groupby(["city","month"])["avg_temp"].mean().unstack()
heatmap_data.columns = month_names

fig, ax = plt.subplots(figsize=(13, 4))
im = ax.imshow(heatmap_data.values, cmap="RdYlBu_r", aspect="auto",
               vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
ax.set_xticks(range(12)); ax.set_xticklabels(month_names)
ax.set_yticks(range(5)); ax.set_yticklabels(heatmap_data.index)
for i in range(5):
    for j in range(12):
        val = heatmap_data.values[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=8.5, fontweight="bold",
                color="white" if val > 32 or val < 20 else "black")
plt.colorbar(im, ax=ax, label="Temperature (°C)", shrink=0.8)
ax.set_title("Monthly Average Temperature Heatmap — Gujarat Cities (1990–2024)",
             fontweight="bold", pad=12)
plt.tight_layout()
save(fig, "14_monthly_temperature_heatmap.png")


# CHART 15 — Hypothesis Testing Summary Visual Table
print("Generating Chart 15: Hypothesis testing summary...")
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis("off")

headers = ["Hypothesis", "Test Used", "Statistic", "p-value", "α = 0.05", "Decision"]
rows = [
    ["H1: LSTM vs ARIMA",          "Paired t-test",    "t = 0.323",    "0.7476",   "≥ 0.05", "Fail to Reject H₀"],
    ["H2: XGBoost vs RF (Rain)",   "Wilcoxon",         "W = 29,689,852","0.3000",  "≥ 0.05", "Fail to Reject H₀"],
    ["H3: Gujarat Temp Trend",     "Mann-Kendall",     "z = 2.301",    "0.0214",   "< 0.05", "REJECT H₀  ✓"],
    ["H4: IF Anomaly Detection",   "Chi-Square",       "χ² = 168.47",  "< 0.0001", "< 0.05", "REJECT H₀  ✓"],
]
col_widths  = [2.2, 1.8, 1.8, 1.2, 1.0, 2.0]
col_starts  = [0.1, 2.3, 4.1, 5.9, 7.1, 8.1]
row_height  = 1.4
header_y    = 8.8

# Header row
for col, (hdr, x) in enumerate(zip(headers, col_starts)):
    ax.add_patch(FancyBboxPatch((x, header_y-0.4), col_widths[col]-0.1, 0.8,
                                boxstyle="round,pad=0.05", facecolor="#1C3557",
                                edgecolor="white", linewidth=1))
    ax.text(x + (col_widths[col]-0.1)/2, header_y,
            hdr, ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")

for r_idx, row in enumerate(rows):
    y = header_y - (r_idx + 1) * row_height
    rejected = "REJECT" in row[-1]
    bg_color = "#EAFAF1" if rejected else "#FEF9E7"
    for col, (val, x) in enumerate(zip(row, col_starts)):
        ax.add_patch(FancyBboxPatch((x, y-0.4), col_widths[col]-0.1, 0.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor=bg_color, edgecolor="#CCCCCC", linewidth=0.5))
        color = "#1A7A4A" if (col == 5 and rejected) else (
                "#C0392B" if col == 5 else "#2C3E50")
        weight = "bold" if col == 5 else "normal"
        ax.text(x + (col_widths[col]-0.1)/2, y, val,
                ha="center", va="center", fontsize=8.5,
                color=color, fontweight=weight)

ax.set_title("Statistical Hypothesis Testing Summary — Gujarat Climate Analytics",
             fontsize=13, fontweight="bold", pad=15, y=0.98)
save(fig, "15_hypothesis_testing_summary.png")


# CHART 16 — ARIMA Forecast vs Actual (Monthly Temperature)
print("Generating Chart 16: ARIMA forecast vs actual...")
train_s  = arima_res["train_series"]
test_s   = arima_res["test_series"]
forecast = arima_res["test_forecast"]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train_s.index, train_s.values, color="#ADB5BD",
        linewidth=1, label="Training data (1990–2018)", alpha=0.7)
ax.plot(test_s.index,  test_s.values,  color="#2C3E50",
        linewidth=2, label="Actual (2019–2024)")
ax.plot(forecast.index, forecast.values, color="#E63946",
        linewidth=2, linestyle="--", label="SARIMA Forecast")
ax.axvline(x=test_s.index[0], color="gray", linestyle=":", linewidth=1.5)
ax.text(test_s.index[0], ax.get_ylim()[0]+0.5, " Train | Test",
        fontsize=9, color="gray")
m = arima_res["metrics"]
ax.text(0.02, 0.05,
        f"RMSE: {m['rmse']:.4f}°C  |  MAE: {m['mae']:.4f}°C  |  R²: {m['r2']:.4f}\n"
        f"AIC: {m['aic']:.2f}  |  Model: SARIMA(1,0,1)(1,1,1,12)",
        transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="#FFF3CD", alpha=0.8))
ax.set_xlabel("Date"); ax.set_ylabel("Monthly Avg Temperature (°C)")
ax.set_title("SARIMA Baseline: Actual vs Forecast — Ahmedabad Monthly Temperature",
             fontweight="bold")
ax.legend()
save(fig, "16_arima_forecast_vs_actual.png")


# CHART 17 — Monthly Rainfall Pattern 
print("Generating Chart 17: Monthly rainfall patterns...")
monthly_rain = df.groupby(["city","month"])["rainfall_mm"].mean().unstack()
month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(12)
width = 0.15
for i, city in enumerate(CITIES):
    ax.bar(x + i*width, monthly_rain.loc[city].values,
           width, label=city, color=list(CITY_COLORS.values())[i], alpha=0.85)
ax.set_xticks(x + width*2)
ax.set_xticklabels(month_names)
ax.set_xlabel("Month"); ax.set_ylabel("Average Daily Rainfall (mm)")
ax.set_title("Monthly Average Rainfall Pattern — Gujarat Cities (1990–2024)",
             fontweight="bold")
ax.legend(loc="upper left")
save(fig, "17_monthly_rainfall_patterns.png")


# SUMMARY
charts = sorted(os.listdir(OUTPUT_FOLDER))
print(f"\n{'='*55}")
print(f"  ALL CHARTS GENERATED SUCCESSFULLY")
print(f"{'='*55}")
print(f"  Saved to: {OUTPUT_FOLDER}")
print(f"  Total charts: {len(charts)}")
print(f"\n  Files:")
for c in charts:
    size = os.path.getsize(os.path.join(OUTPUT_FOLDER, c)) // 1024
    print(f"    {c:45s} {size:5d} KB")
print(f"\n  Insert these PNG files into your project report")
print(f"  and dashboard as needed.")
print(f"{'='*55}")