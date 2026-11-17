import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, chi2_contingency

BASE_FOLDER   = r"C:\Users\HP\Desktop\DA"
DATA_FOLDER   = os.path.join(BASE_FOLDER, "Data")
MODELS_FOLDER = os.path.join(BASE_FOLDER, "models", "models_results")

print("MODELS_FOLDER =", MODELS_FOLDER)
print("Files:", os.listdir(MODELS_FOLDER))

print("HYPOTHESIS TESTING")
print("  Significance level: α = 0.05")
print("  Reject H₀ if p-value < 0.05")


# H1 — LSTM vs ARIMA (Paired t-test on prediction errors)
print(f"\nHYPOTHESIS 1 — LSTM outperforms ARIMA for temperature forecasting")
print("  H₀: No significant difference in RMSE between LSTM and ARIMA")
print("  H₁: LSTM achieves significantly lower prediction error than ARIMA")
print("  Test: Paired t-test on absolute prediction errors")
print("  City: Ahmedabad (monthly temperature series)")

# Load ARIMA results
with open(os.path.join(MODELS_FOLDER, "arima_results.pkl"), "rb") as f:
    arima_res = pickle.load(f)

# Load LSTM results for Ahmedabad
with open(os.path.join(MODELS_FOLDER, "lstm_results.pkl"), "rb") as f:
    lstm_res = pickle.load(f)

# ARIMA errors (monthly)
arima_true = arima_res["test_series"].values
arima_pred = arima_res["test_forecast"].values
arima_errors = np.abs(arima_true - arima_pred)

# LSTM errors for Ahmedabad — aggregate to monthly for fair comparison
ahm_lstm = lstm_res["Ahmedabad"]
lstm_true_daily = ahm_lstm["lstm_eval"]["y_true"]
lstm_pred_daily = ahm_lstm["lstm_eval"]["y_pred"]
lstm_errors_daily = np.abs(lstm_true_daily - lstm_pred_daily)

n_months = len(arima_errors)  # 72 months
days_per_month = len(lstm_errors_daily) / n_months
lstm_errors_monthly = np.array([
    lstm_errors_daily[int(i*days_per_month):int((i+1)*days_per_month)].mean()
    for i in range(n_months)
])

# Paired t-test
t_stat, p_value = ttest_rel(arima_errors, lstm_errors_monthly)

print(f"\n  ARIMA  — Mean absolute error : {arima_errors.mean():.4f}°C")
print(f"  LSTM   — Mean absolute error : {lstm_errors_monthly.mean():.4f}°C")
print(f"\n  t-statistic : {t_stat:.4f}")
print(f"  p-value     : {p_value:.6f}")
print(f"  α           : 0.05")

if p_value < 0.05:
    if t_stat > 0:
        print(f"\n  RESULT: p={p_value:.6f} < 0.05 → REJECT H₀")
        print(f"  CONCLUSION: LSTM has significantly LOWER error than ARIMA")
        print(f"  LSTM reduces prediction error by {(arima_errors.mean()-lstm_errors_monthly.mean())/arima_errors.mean()*100:.1f}% over ARIMA")
    else:
        print(f"\n  RESULT: p={p_value:.6f} < 0.05 → REJECT H₀")
        print(f"  CONCLUSION: Significant difference found (ARIMA had lower error)")
        print(f"  NOTE: Both models are close — daily vs monthly scale affects comparison")
else:
    print(f"\n  RESULT: p={p_value:.6f} ≥ 0.05 → FAIL TO REJECT H₀")
    print(f"  CONCLUSION: No statistically significant difference at α=0.05")
    print(f"  NOTE: LSTM RMSE (daily) and ARIMA RMSE (monthly) operate at different scales")

h1_result = {"t_stat": t_stat, "p_value": p_value,
             "arima_mae": arima_errors.mean(), "lstm_mae": lstm_errors_monthly.mean()}


# H2 — XGBoost vs Random Forest for rainfall (Wilcoxon signed-rank test)
print(f"\nHYPOTHESIS 2 — XGBoost outperforms Random Forest for rainfall prediction")
print("  H₀: No significant difference in rainfall prediction accuracy")
print("  H₁: XGBoost achieves significantly different accuracy than RF")
print("  Test: Wilcoxon signed-rank test (non-parametric — rainfall is non-normal)")

with open(os.path.join(MODELS_FOLDER, "tree_results.pkl"), "rb") as f:
    tree_res = pickle.load(f)

y_true_rain = tree_res["y_test_rain"]
rf_rain_pred  = tree_res["rf"]["rain_pred"]
xgb_rain_pred = tree_res["xgb"]["rain_pred"]

rf_errors  = np.abs(y_true_rain - rf_rain_pred)
xgb_errors = np.abs(y_true_rain - xgb_rain_pred)

# Wilcoxon test
w_stat, p_value_h2 = wilcoxon(rf_errors, xgb_errors)

print(f"\n  RF  — MAE : {rf_errors.mean():.4f} mm  |  RMSE : {np.sqrt(np.mean((y_true_rain-rf_rain_pred)**2)):.4f} mm")
print(f"  XGB — MAE : {xgb_errors.mean():.4f} mm  |  RMSE : {np.sqrt(np.mean((y_true_rain-xgb_rain_pred)**2)):.4f} mm")
print(f"\n  Wilcoxon W  : {w_stat:.2f}")
print(f"  p-value     : {p_value_h2:.6f}")

if p_value_h2 < 0.05:
    better = "XGBoost" if xgb_errors.mean() < rf_errors.mean() else "Random Forest"
    print(f"\n  RESULT: p={p_value_h2:.6f} < 0.05 → REJECT H₀")
    print(f"  CONCLUSION: {better} has significantly lower rainfall prediction error ")
else:
    print(f"\n  RESULT: p={p_value_h2:.6f} ≥ 0.05 → FAIL TO REJECT H₀")
    print(f"  CONCLUSION: No statistically significant difference between RF and XGBoost")

h2_result = {"w_stat": w_stat, "p_value": p_value_h2,
             "rf_mae": rf_errors.mean(), "xgb_mae": xgb_errors.mean()}


# H3 — Gujarat temperature rising over 30 years (Mann-Kendall trend test)
print(f"\nHYPOTHESIS 3 — Gujarat mean temperature has significantly increased")
print("  H₀: No significant upward trend in Gujarat temperature (1990–2024)")
print("  H₁: Significant upward trend exists in Gujarat temperature")
print("  Test: Mann-Kendall trend test (designed for climate time-series)")

master_path = os.path.join(DATA_FOLDER, "master_gujarat_climate.csv")
df = pd.read_csv(master_path)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

h3_results = {}
print(f"\n  Results per city:")
print(f"  {'City':12s} {'Slope(°C/yr)':>14} {'p-value':>10} {'Trend':>12} {'Result':>10}")

for city in sorted(df["city"].unique()):
    city_df = df[df["city"] == city].copy()

    # Annual mean temperature
    annual = city_df.groupby("year")["avg_temp"].mean().reset_index()
    years  = annual["year"].values
    temps  = annual["avg_temp"].values

    # Mann-Kendall test 
    n = len(temps)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            diff = temps[j] - temps[i]
            if diff > 0:   s += 1
            elif diff < 0: s -= 1

    # Variance of S
    var_s = n * (n-1) * (2*n+5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    p_mk = 2 * (1 - stats.norm.cdf(abs(z)))  # two-tailed

    # Sen's slope (median of all pairwise slopes)
    slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            if years[j] != years[i]:
                slopes.append((temps[j] - temps[i]) / (years[j] - years[i]))
    sen_slope = np.median(slopes)

    trend = "Increasing" if s > 0 else "Decreasing"
    result = "SIGNIFICANT" if p_mk < 0.05 else "not sig."

    print(f"  {city:12s} {sen_slope:>+14.4f} {p_mk:>10.6f} {trend:>12} {result:>10}")

    h3_results[city] = {"s": s, "z": z, "p_value": p_mk,
                        "sen_slope": sen_slope, "trend": trend}

# Overall Gujarat (all cities combined)
annual_all = df.groupby("year")["avg_temp"].mean().reset_index()
years_all  = annual_all["year"].values
temps_all  = annual_all["avg_temp"].values

n = len(temps_all)
s = sum(1 if temps_all[j] > temps_all[i] else (-1 if temps_all[j] < temps_all[i] else 0)
        for i in range(n-1) for j in range(i+1, n))
var_s = n*(n-1)*(2*n+5)/18
z_all = (s-1)/np.sqrt(var_s) if s > 0 else ((s+1)/np.sqrt(var_s) if s < 0 else 0)
p_all = 2*(1-stats.norm.cdf(abs(z_all)))
slopes_all = [(temps_all[j]-temps_all[i])/(years_all[j]-years_all[i])
              for i in range(n-1) for j in range(i+1, n) if years_all[j]!=years_all[i]]
sen_all = np.median(slopes_all)


print(f"  {'GUJARAT ALL':12s} {sen_all:>+14.4f} {p_all:>10.6f} {'Increasing' if s>0 else 'Decreasing':>12} {'SIGNIFICANT' if p_all<0.05 else 'not sig.':>10}")

total_increase = sen_all * (2024 - 1990)
print(f"\n  Gujarat overall Sen's slope : {sen_all:+.4f}°C/year")
print(f"  Total warming 1990–2024    : {total_increase:+.2f}°C")
if p_all < 0.05:
    print(f"  RESULT: p={p_all:.6f} < 0.05 → REJECT H₀ ")
    print(f"  CONCLUSION: Gujarat temperature has significantly increased at {sen_all*10:.2f}°C/decade")
else:
    print(f"  RESULT: p={p_all:.6f} ≥ 0.05 → FAIL TO REJECT H₀")

h3_results["Gujarat_all"] = {"sen_slope": sen_all, "p_value": p_all, "total_increase": total_increase}


# H4 — Isolation Forest detects known extreme events (Chi-square)
print(f"\nHYPOTHESIS 4 — Isolation Forest detects real extreme climate events")
print("  H₀: IF anomaly flags are independent of known extreme events")
print("  H₁: IF anomaly flags are significantly associated with extreme events")
print("  Test: Chi-square test of independence")

anomaly_df = pd.read_csv(os.path.join(DATA_FOLDER, "anomaly_predictions.csv"))
anomaly_df["date"] = pd.to_datetime(anomaly_df["date"])

# Known extreme event dates
EXTREME_EVENTS = {
    # Gujarat heatwaves — peak days typically in May
    (2010, 5): "heatwave", (2015, 5): "heatwave", (2016, 5): "heatwave",
    (2019, 5): "heatwave", (2022, 5): "heatwave",
    # Gujarat floods — peak months July-August
    (2017, 7): "flood", (2017, 8): "flood",
    (2019, 7): "flood", (2019, 8): "flood",
    (2020, 7): "flood", (2020, 8): "flood",
    (2023, 7): "flood", (2023, 8): "flood",
    # Cyclone Tauktae May 2021
    (2021, 5): "cyclone",
}

anomaly_df["yr_mo"] = list(zip(anomaly_df["date"].dt.year,
                                anomaly_df["date"].dt.month))
anomaly_df["is_extreme_event"] = anomaly_df["yr_mo"].apply(
    lambda x: 1 if x in EXTREME_EVENTS else 0
)

A = ((anomaly_df["anomaly_flag"]==1) & (anomaly_df["is_extreme_event"]==1)).sum()
B = ((anomaly_df["anomaly_flag"]==0) & (anomaly_df["is_extreme_event"]==1)).sum()
C = ((anomaly_df["anomaly_flag"]==1) & (anomaly_df["is_extreme_event"]==0)).sum()
D = ((anomaly_df["anomaly_flag"]==0) & (anomaly_df["is_extreme_event"]==0)).sum()

contingency = np.array([[A, B], [C, D]])
chi2, p_chi2, dof, expected = chi2_contingency(contingency)

extreme_days     = A + B
nonextreme_days  = C + D
extreme_flag_rate    = A / extreme_days * 100 if extreme_days > 0 else 0
nonextreme_flag_rate = C / nonextreme_days * 100 if nonextreme_days > 0 else 0

print(f"\n  Contingency table:")
print(f"  {'':20s} {'Flagged':>10} {'Not flagged':>12} {'Total':>8}")
print(f"  {'Extreme event days':20s} {A:>10,} {B:>12,} {A+B:>8,}")
print(f"  {'Normal days':20s} {C:>10,} {D:>12,} {C+D:>8,}")
print(f"  {'Total':20s} {A+C:>10,} {B+D:>12,} {A+B+C+D:>8,}")

print(f"\n  Flag rate on extreme event days : {extreme_flag_rate:.1f}%")
print(f"  Flag rate on normal days        : {nonextreme_flag_rate:.1f}%")
print(f"\n  Chi-square statistic : {chi2:.4f}")
print(f"  Degrees of freedom   : {dof}")
print(f"  p-value              : {p_chi2:.6f}")

if p_chi2 < 0.05:
    print(f"\n  RESULT: p={p_chi2:.6f} < 0.05 → REJECT H₀ ")
    print(f"  CONCLUSION: Isolation Forest flags are significantly associated")
    print(f"  with known extreme climate events in Gujarat")
    print(f"  Anomaly rate is {extreme_flag_rate/nonextreme_flag_rate:.1f}x higher during extreme events")
else:
    print(f"\n  RESULT: p={p_chi2:.6f} ≥ 0.05 → FAIL TO REJECT H₀")
    print(f"  CONCLUSION: No significant association detected at α=0.05")

h4_result = {"chi2": chi2, "p_value": p_chi2, "dof": dof,
             "extreme_flag_rate": extreme_flag_rate,
             "normal_flag_rate": nonextreme_flag_rate}

#summary table
print(f"\nHYPOTHESIS TESTING SUMMARY")
print(f"\n  {'H':>3} {'Test':>20} {'Statistic':>12} {'p-value':>10} {'α=0.05':>8} {'Decision':>12}")


rows = [
    ("H1", "Paired t-test",   f"t={h1_result['t_stat']:.3f}",  h1_result['p_value']),
    ("H2", "Wilcoxon",        f"W={h2_result['w_stat']:.0f}",  h2_result['p_value']),
    ("H3", "Mann-Kendall",    f"z={z_all:.3f}",                p_all),
    ("H4", "Chi-square",      f"χ²={chi2:.3f}",               p_chi2),
]
for h, test, stat, p in rows:
    dec = "Reject H₀ " if p < 0.05 else "Fail to reject"
    print(f"  {h:>3} {test:>20} {stat:>12} {p:>10.6f} {'<0.05' if p<0.05 else '≥0.05':>8} {dec:>12}")


hyp_results = {
    "H1": h1_result,
    "H2": h2_result,
    "H3": h3_results,
    "H4": h4_result,
}
save_path = os.path.join(MODELS_FOLDER, "hypothesis_results.pkl")
with open(save_path, "wb") as f:
    pickle.dump(hyp_results, f)

print(f"\n  Saved → {save_path}")
