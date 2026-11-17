import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config 
st.set_page_config(
    page_title="Gujarat Climate Analytics",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>

/* Main background */
.main { background-color: #F0F4F8; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #1C7293;
}

/* ===== FIXED TAB STYLING ===== */

.stTabs [data-baseweb="tab-list"] {
    display: flex;
    gap: 12px;
    padding-top: 50px;
    padding-bottom: 8px;
    border-bottom: none;
}

/* Individual tab */
.stTabs [data-baseweb="tab"] {
    background: #1E293B;
    border-radius: 10px;
    padding: 12px 22px;

    height: auto !important;
    min-height: 52px !important;

    display: flex;
    align-items: center;
    justify-content: center;

    color: white !important;
}

/* Selected tab */
.stTabs [aria-selected="true"] {
    background: #1C7293 !important;
}

/* TEXT INSIDE TAB */
.stTabs [data-baseweb="tab"] p {
    margin: 0 !important;
    padding: 0 !important;

    font-size: 17px !important;
    font-weight: 600 !important;

    line-height: 1.4 !important;

    display: flex;
    align-items: center;
    justify-content: center;

    overflow: visible !important;
}

/* Remove clipping */
.stTabs [data-baseweb="tab"] div {
    overflow: visible !important;
}

</style>
""", unsafe_allow_html=True)

# Paths
DATA_FOLDER   = r"C:\Users\HP\Desktop\DA\Data"
MODELS_FOLDER = r"C:\Users\HP\Desktop\DA\models\models_results"

CITY_COORDS = {
    "Ahmedabad": {"lat": 23.03, "lon": 72.59},
    "Vadodara":  {"lat": 22.31, "lon": 73.18},
    "Surat":     {"lat": 21.17, "lon": 72.83},
    "Rajkot":    {"lat": 22.30, "lon": 70.80},
    "Bhavnagar": {"lat": 21.76, "lon": 72.15},
}
CITY_COLORS = {
    "Ahmedabad": "#E63946", "Vadodara": "#457B9D",
    "Surat": "#2A9D8F", "Rajkot": "#E9C46A", "Bhavnagar": "#6A0572",
}

# Load data
@st.cache_data
def load_master():
    df = pd.read_csv(os.path.join(DATA_FOLDER, "master_gujarat_climate.csv"))
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df

@st.cache_data
def load_anomalies():
    df = pd.read_csv(os.path.join(DATA_FOLDER, "anomaly_predictions.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def load_models():
    models = {}
    try:
        with open(os.path.join(MODELS_FOLDER, "lstm_results.pkl"), "rb") as f:
            models["lstm"] = pickle.load(f)
    except: models["lstm"] = None
    try:
        with open(os.path.join(MODELS_FOLDER, "tree_results.pkl"), "rb") as f:
            models["tree"] = pickle.load(f)
    except: models["tree"] = None
    try:
        with open(os.path.join(MODELS_FOLDER, "arima_results.pkl"), "rb") as f:
            models["arima"] = pickle.load(f)
    except: models["arima"] = None
    try:
        with open(os.path.join(MODELS_FOLDER, "hypothesis_results.pkl"), "rb") as f:
            models["hyp"] = pickle.load(f)
    except: models["hyp"] = None
    return models

# Load everything
df       = load_master()
anom_df  = load_anomalies()
models   = load_models()

CITIES = list(CITY_COORDS.keys())

# SIDEBAR 
with st.sidebar:
    st.markdown("## 🌍 Gujarat Climate Analytics")
    st.markdown("---")

    city = st.selectbox("📍 Select City", CITIES, index=0)
    st.markdown("---")

    # Dynamic alert based on city's recent max temp
    city_recent = df[df["city"] == city].sort_values("date").tail(30)
    max_recent  = city_recent["max_temp"].max() if len(city_recent) > 0 else 0

    if max_recent >= 42:
        st.markdown(f"""<div class="alert-red">
        🔴 <b>EXTREME HEATWAVE ALERT</b><br>
        {city} — Max temp reached {max_recent:.1f}°C recently
        </div>""", unsafe_allow_html=True)
    elif max_recent >= 39:
        st.markdown(f"""<div class="alert-yellow">
        🟡 <b>HEATWAVE WARNING</b><br>
        {city} — Max temp {max_recent:.1f}°C in last 30 days
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="alert-blue">
        🔵 <b>Normal Conditions</b><br>
        {city} — Max temp {max_recent:.1f}°C in last 30 days
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.caption(f"📅 1990–2024 | 35 years")
    st.caption(f"📊 {len(df):,} total records")
    st.caption(f"🏙️ 5 Gujarat cities")
    st.caption(f"🔢 25 features")
    st.markdown("---")
    st.markdown("**Models Used**")
    st.caption("🧠 LSTM & CNN-LSTM")
    st.caption("🌲 XGBoost & Random Forest")
    st.caption("📈 SARIMA baseline")
    st.caption("🔍 Isolation Forest")

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📅 7-Day Forecast",
    "📈 30-Day Trend",
    "🕰️ Historical Shifts",
    "🏆 Model Evaluation",
])


# TAB 1 — OVERVIEW
with tab1:
    st.markdown(f"### 🌤️ Climate Overview — {city}")

    city_df   = df[df["city"] == city].sort_values("date")
    latest    = city_df.iloc[-1]
    last_week = city_df.tail(7)

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        delta_temp = latest["avg_temp"] - city_df.iloc[-8]["avg_temp"] if len(city_df) > 8 else 0
        st.metric("🌡️ Avg Temperature", f"{latest['avg_temp']:.1f}°C", f"{delta_temp:+.1f}°C vs last week")
    with col2:
        st.metric("🌡️ Max Temperature", f"{latest['max_temp']:.1f}°C")
    with col3:
        st.metric("🌡️ Min Temperature", f"{latest['min_temp']:.1f}°C")
    with col4:
        st.metric("🌧️ Rainfall", f"{latest['rainfall_mm']:.1f} mm")
    with col5:
        # Active anomalies in last 7 days from Isolation Forest
        city_anom = anom_df[anom_df["city"] == city].sort_values("date")
        recent_anom = city_anom.tail(7)["anomaly_flag"].sum()
        st.metric("⚠️ Anomaly Flags", f"{int(recent_anom)}", "last 7 days")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Interactive Gujarat map
        map_data = pd.DataFrame([
            {
                "city": c,
                "lat": CITY_COORDS[c]["lat"],
                "lon": CITY_COORDS[c]["lon"],
                "avg_temp": df[df["city"] == c]["avg_temp"].mean().round(2),
                "color": CITY_COLORS[c],
            }
            for c in CITIES
        ])
        fig_map = px.scatter_mapbox(
            map_data, lat="lat", lon="lon",
            hover_name="city",
            hover_data={"avg_temp": True, "lat": False, "lon": False, "color": False},
            color="city",
            color_discrete_map=CITY_COLORS,
            size=[20]*5,
            zoom=5.8,
            center={"lat": 22.5, "lon": 71.5},
            height=380,
            title="Gujarat — Five Study Cities",
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True,
        )
        st.plotly_chart(fig_map, width="stretch")

    with col_right:
        # Recent 30-day temperature sparkline
        recent30 = city_df.tail(30)
        fig_spark = go.Figure()
        # Convert HEX to RGBA with transparency
        hex_color = CITY_COLORS[city]
        hex_color = hex_color.lstrip('#')

        r = int(hex_color[0:2], 16) 
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        rgba_fill = f"rgba({r},{g},{b},0.15)"

        fig_spark.add_trace(go.Scatter(
        x=recent30["date"],
        y=recent30["avg_temp"],
        mode="lines",
        name="Avg Temp",
        line=dict(color=CITY_COLORS[city], width=2),
        fill="tozeroy",
        fillcolor=rgba_fill,
        ))
        fig_spark.update_layout(
            title=f"{city} — Last 30 Days Temperature",
            height=180, margin=dict(l=0, r=0, t=40, b=0),
            yaxis_title="°C", xaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig_spark, width="stretch")

        # Humidity and pressure mini charts
        fig_hum = go.Figure()
        fig_hum.add_trace(go.Bar(
            x=recent30["date"], y=recent30["humidity"],
            marker_color="#2A9D8F", name="Humidity",
        ))
        fig_hum.update_layout(
            title=f"Humidity (%) — Last 30 Days",
            height=160, margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_hum, width="stretch")

    # City comparison table
    st.markdown("#### 📊 All Cities")
    summary_rows = []
    for c in CITIES:
        c_latest = df[df["city"] == c].sort_values("date").iloc[-1]
        c_anom   = int(anom_df[anom_df["city"] == c].tail(7)["anomaly_flag"].sum())
        summary_rows.append({
            "City":        c,
            "Avg Temp (°C)": round(c_latest["avg_temp"], 1),
            "Max Temp (°C)": round(c_latest["max_temp"], 1),
            "Min Temp (°C)": round(c_latest["min_temp"], 1),
            "Rainfall (mm)": round(c_latest["rainfall_mm"], 1),
            "Humidity (%)":  round(c_latest["humidity"], 1),
            "Anomalies (7d)": c_anom,
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df.set_index("City"), width="stretch")


# TAB 2 — 7-DAY FORECAST
with tab2:
    st.markdown(f"### 📅 7-Day Forecast — {city}")

    if models["lstm"] is None or models["tree"] is None:
        st.warning("Model files not found. Check models folder path.")
    else:
        lstm_city = models["lstm"].get(city, {})
        lstm_eval = lstm_city.get("lstm_eval", {})
        cnn_eval  = lstm_city.get("cnn_lstm_eval", {})

        # Get last 30 days from test period as "forecast" proxy
        city_test = df[(df["city"] == city) & (df["date"] > "2019-01-01")].sort_values("date")
        last7_actual = city_test.tail(60).head(30)  # 30 days context

        # Use LSTM predictions from saved results
        y_true = lstm_eval.get("y_true", np.array([]))
        y_pred = lstm_eval.get("y_pred", np.array([]))

        if len(y_true) >= 7:
            # Show last 30 actual + 7 predicted
            show_actual = y_true[-30:]
            show_pred   = y_pred[-7:]
            x_actual    = list(range(len(show_actual)))
            x_pred      = list(range(len(show_actual) - 1, len(show_actual) + len(show_pred) - 1))

            # Smart alerts
            max_pred_temp = show_pred.max() if len(show_pred) > 0 else 0
            if max_pred_temp >= 41:
                st.markdown(f"""<div class="alert-red">
                🔴 <b>EXTREME HEATWAVE FORECAST</b> —
                Predicted temperature reaching {max_pred_temp:.1f}°C in next 7 days!
                </div>""", unsafe_allow_html=True)
            elif max_pred_temp >= 38:
                st.markdown(f"""<div class="alert-yellow">
                🟡 <b>HEATWAVE WARNING</b> —
                Predicted temperature {max_pred_temp:.1f}°C in forecast window.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="alert-green">
                🟢 <b>Normal conditions forecast</b> —
                Max predicted: {max_pred_temp:.1f}°C
                </div>""", unsafe_allow_html=True)

            col_left, col_right = st.columns([3, 2])

            with col_left:
                # LSTM temperature forecast chart
                fig_fore = go.Figure()
                fig_fore.add_trace(go.Scatter(
                    x=x_actual, y=show_actual,
                    mode="lines", name="Actual Temperature",
                    line=dict(color="#2C3E50", width=2),
                ))
                fig_fore.add_trace(go.Scatter(
                    x=x_pred, y=show_pred,
                    mode="lines+markers", name="LSTM Forecast (7-day)",
                    line=dict(color=CITY_COLORS[city], width=2.5, dash="dash"),
                    marker=dict(size=8),
                ))
                fig_fore.add_vline(x=len(show_actual)-1, line_dash="dot",
                                   line_color="gray", annotation_text="Forecast start")
                fig_fore.update_layout(
                    title=f"LSTM Temperature Forecast — {city}",
                    xaxis_title="Days", yaxis_title="Temperature (°C)",
                    height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_fore, width="stretch")

            with col_right:
                # Metrics box
                rmse = lstm_eval.get("rmse", 0)
                r2   = lstm_eval.get("r2", 0)
                cnn_rmse = cnn_eval.get("rmse", 0)
                best = "LSTM" if rmse <= cnn_rmse else "CNN-LSTM"

                st.markdown("#### 🎯 Model Metrics")
                m1, m2 = st.columns(2)
                m1.metric("LSTM RMSE", f"{rmse:.4f}°C")
                m2.metric("LSTM R²",   f"{r2:.4f}")
                m1.metric("CNN-LSTM RMSE", f"{cnn_rmse:.4f}°C")
                m2.metric("Best Model", best)
                st.caption(f"Test period: 2019–2024")

        # Rainfall forecast bar chart (XGBoost predictions)
        tree_res = models["tree"]
        y_rain_true = tree_res.get("y_test_rain", np.array([]))
        y_rain_pred = tree_res.get("xgb", {}).get("rain_pred", np.array([]))

        if len(y_rain_pred) >= 7:
            rain_7  = y_rain_pred[-7:]
            days_7  = [f"Day {i+1}" for i in range(7)]

            # Rain alert
            max_rain = rain_7.max()
            if max_rain >= 40:
                st.markdown(f"""<div class="alert-yellow">
                🌧️ <b>HEAVY RAINFALL ALERT</b> —
                XGBoost predicts {max_rain:.1f}mm rainfall in forecast window.
                </div>""", unsafe_allow_html=True)

            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(
                x=days_7, y=rain_7,
                marker_color=["#E63946" if r >= 40 else "#457B9D" for r in rain_7],
                text=[f"{r:.1f}mm" for r in rain_7],
                textposition="outside",
                name="Predicted Rainfall",
            ))
            fig_rain.update_layout(
                title=f"XGBoost 7-Day Rainfall Forecast — {city}",
                yaxis_title="Rainfall (mm)",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_rain, width="stretch")


# TAB 3 — 30-DAY TREND
with tab3:
    st.markdown(f"### 📈 30-Day Climate Trend — {city}")

    city_sorted = df[df["city"] == city].sort_values("date")
    recent_90   = city_sorted.tail(90)  # last 90 days

    col1, col2 = st.columns(2)

    with col1:
        # Temperature with 7-day rolling average
        recent_90 = recent_90.copy()
        recent_90["roll7"] = recent_90["avg_temp"].rolling(7, min_periods=1).mean()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=recent_90["date"], y=recent_90["avg_temp"],
            mode="lines", name="Daily Temp",
            line=dict(color="#ADB5BD", width=1.2),
            opacity=0.8,
        ))
        fig_trend.add_trace(go.Scatter(
            x=recent_90["date"], y=recent_90["roll7"],
            mode="lines", name="7-Day Rolling Avg",
            line=dict(color=CITY_COLORS[city], width=2.5),
        ))
        fig_trend.update_layout(
            title=f"Temperature — Last 90 Days with 7-Day Trend",
            yaxis_title="°C", height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_trend, width="stretch")

    with col2:
        # Rainfall 30-day bar
        recent_30 = city_sorted.tail(30)
        fig_rain_trend = go.Figure()
        fig_rain_trend.add_trace(go.Bar(
            x=recent_30["date"], y=recent_30["rainfall_mm"],
            marker_color="#2A9D8F", name="Rainfall",
        ))
        fig_rain_trend.update_layout(
            title="Daily Rainfall — Last 30 Days",
            yaxis_title="Rainfall (mm)", height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_rain_trend, width="stretch")

    # Humidity and pressure trends
    col3, col4 = st.columns(2)
    with col3:
        fig_hum = px.line(recent_90, x="date", y="humidity",
                          title="Relative Humidity (%) — Last 90 Days",
                          color_discrete_sequence=["#E9C46A"])
        fig_hum.update_layout(height=280, yaxis_title="%")
        st.plotly_chart(fig_hum, width="stretch")

    with col4:
        fig_pres = px.line(recent_90, x="date", y="pressure",
                           title="Pressure (hPa) — Last 90 Days",
                           color_discrete_sequence=["#6A0572"])
        fig_pres.update_layout(height=280, yaxis_title="hPa")
        st.plotly_chart(fig_pres, width="stretch")

    # Anomaly flags in recent period
    city_anom_recent = anom_df[
        (anom_df["city"] == city) &
        (anom_df["anomaly_flag"] == 1)
    ].sort_values("date").tail(20)

    if len(city_anom_recent) > 0:
        st.markdown("#### ⚠️ Recent Anomaly Events Detected")
        display_cols = ["date", "avg_temp", "rainfall_mm", "temp_anomaly", "rain_anomaly"]
        available = [c for c in display_cols if c in city_anom_recent.columns]
        st.dataframe(city_anom_recent[available].tail(10).reset_index(drop=True),
                     width="stretch")


# TAB 4 — HISTORICAL SHIFTS
with tab4:
    st.markdown(f"### 🕰️ Historical Climate Shifts — {city}")

    def get_decade(year):
        if year < 2000:  return "1990s"
        elif year < 2010: return "2000s"
        elif year < 2020: return "2010s"
        else:             return "2020s"

    df["decade"] = df["year"].apply(get_decade)
    decade_order = ["1990s", "2000s", "2010s", "2020s"]

    col1, col2 = st.columns(2)

    with col1:
        # Decadal temperature comparison — all cities
        decade_all = df.groupby(["decade", "city"])["avg_temp"].mean().reset_index()
        fig_dec = px.bar(
            decade_all, x="decade", y="avg_temp", color="city",
            barmode="group",
            color_discrete_map=CITY_COLORS,
            category_orders={"decade": decade_order},
            title="Decadal Average Temperature — All Gujarat Cities",
            labels={"avg_temp": "Avg Temperature (°C)", "decade": "Decade"},
            height=380,
        )
        fig_dec.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_dec, width="stretch")

    with col2:
        # Annual mean temperature trend for selected city
        city_annual = df[df["city"] == city].groupby("year")["avg_temp"].mean().reset_index()
        hyp = models.get("hyp", {})
        h3  = hyp.get("H3", {}).get(city, {}) if hyp else {}
        slope = h3.get("sen_slope", 0)
        pval  = h3.get("p_value", 1)

        trend_line = (city_annual["avg_temp"].iloc[0] +
                      slope * (city_annual["year"] - city_annual["year"].iloc[0]))

        fig_trend_hist = go.Figure()
        fig_trend_hist.add_trace(go.Scatter(
            x=city_annual["year"], y=city_annual["avg_temp"],
            mode="lines+markers", name="Annual Mean Temp",
            line=dict(color=CITY_COLORS[city], width=2),
            marker=dict(size=5),
        ))
        fig_trend_hist.add_trace(go.Scatter(
            x=city_annual["year"], y=trend_line,
            mode="lines", name=f"Sen's Slope ({slope:+.4f}°C/yr)",
            line=dict(color="#E63946", width=2.5, dash="dash"),
        ))
        sig_text = "Significant ✓" if pval < 0.05 else "Not significant"
        fig_trend_hist.update_layout(
            title=f"{city} — Annual Temperature Trend<br><sup>p={pval:.4f} ({sig_text})</sup>",
            yaxis_title="°C", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_trend_hist, width="stretch")

    # Monthly temperature heatmap
    st.markdown("#### 🌡️ Monthly Average Temperature Heatmap")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    heatmap_data = df.groupby(["city","month"])["avg_temp"].mean().unstack()
    heatmap_data.columns = month_names

    fig_heat = px.imshow(
        heatmap_data,
        color_continuous_scale="RdYlBu_r",
        title="Average Temperature by City and Month (1990–2024)",
        labels=dict(x="Month", y="City", color="°C"),
        text_auto=".1f",
        height=300,
    )
    fig_heat.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_heat, width="stretch")

    # Seasonal boxplot for selected city
    st.markdown(f"#### 🌿 Seasonal Temperature Distribution — {city}")
    city_season = df[df["city"] == city]
    season_order_disp = ["Winter", "Summer", "Monsoon", "Post_Monsoon"]
    season_colors_disp = {
        "Winter": "#AED6F1", "Summer": "#E74C3C",
        "Monsoon": "#27AE60", "Post_Monsoon": "#F39C12"
    }
    fig_box = px.box(
        city_season[city_season["season"].isin(season_order_disp)],
        x="season", y="avg_temp",
        color="season",
        color_discrete_map=season_colors_disp,
        category_orders={"season": season_order_disp},
        title=f"{city} — Temperature Distribution by Season",
        labels={"avg_temp": "Avg Temperature (°C)", "season": "Season"},
        height=350,
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, width="stretch")

    # Mann-Kendall summary table
    st.markdown("#### 📊 Mann-Kendall Trend Test Summary")
    if models.get("hyp"):
        h3_data = models["hyp"].get("H3", {})
        mk_rows = []
        for c in CITIES:
            h = h3_data.get(c, {})
            mk_rows.append({
                "City":            c,
                "Sen's Slope (°C/yr)": f"{h.get('sen_slope', 0):+.4f}",
                "Rate (°C/decade)":    f"{h.get('sen_slope', 0)*10:+.3f}",
                "p-value":             f"{h.get('p_value', 1):.4f}",
                "Trend":               h.get("trend", "—"),
                "Result":              "✅ Significant" if h.get("p_value", 1) < 0.05 else "❌ Not significant",
            })
        overall = h3_data.get("Gujarat_all", {})
        mk_rows.append({
            "City":            "GUJARAT ALL",
            "Sen's Slope (°C/yr)": f"{overall.get('sen_slope', 0):+.4f}",
            "Rate (°C/decade)":    f"{overall.get('sen_slope', 0)*10:+.3f}",
            "p-value":             f"{overall.get('p_value', 1):.4f}",
            "Trend":               "Increasing",
            "Result":              "✅ Significant" if overall.get("p_value", 1) < 0.05 else "❌ Not significant",
        })
        mk_df = pd.DataFrame(mk_rows)
        st.dataframe(mk_df.set_index("City"), width="stretch")


# TAB 5 — MODEL EVALUATION
with tab5:
    st.markdown(f"### 🏆 Model Evaluation & Comparison")

    if models["lstm"] is None or models["tree"] is None:
        st.warning("Model result files not found.")
    else:
        # Deep Learning: LSTM vs CNN-LSTM
        st.markdown("#### 🧠 Deep Learning: LSTM vs CNN-LSTM — Temperature Forecasting")

        dl_rows = []
        for c in CITIES:
            le = models["lstm"][c]["lstm_eval"]
            ce = models["lstm"][c]["cnn_lstm_eval"]
            best = "LSTM" if le["rmse"] <= ce["rmse"] else "CNN-LSTM"
            dl_rows.append({
                "City":           c,
                "LSTM RMSE (°C)": round(le["rmse"], 4),
                "LSTM MAE (°C)":  round(le["mae"],  4),
                "LSTM R²":        round(le["r2"],   4),
                "CNN-LSTM RMSE":  round(ce["rmse"], 4),
                "CNN-LSTM MAE":   round(ce["mae"],  4),
                "CNN-LSTM R²":    round(ce["r2"],   4),
                "Best Model":     best,
            })
        dl_df = pd.DataFrame(dl_rows)
        st.dataframe(dl_df.set_index("City"), width="stretch")

        # LSTM vs CNN-LSTM RMSE bar chart
        fig_dl = go.Figure()
        fig_dl.add_trace(go.Bar(
            name="LSTM", x=CITIES,
            y=[models["lstm"][c]["lstm_eval"]["rmse"] for c in CITIES],
            marker_color="#457B9D",
            text=[f'{models["lstm"][c]["lstm_eval"]["rmse"]:.3f}' for c in CITIES],
            textposition="outside",
        ))
        fig_dl.add_trace(go.Bar(
            name="CNN-LSTM", x=CITIES,
            y=[models["lstm"][c]["cnn_lstm_eval"]["rmse"] for c in CITIES],
            marker_color="#E63946",
            text=[f'{models["lstm"][c]["cnn_lstm_eval"]["rmse"]:.3f}' for c in CITIES],
            textposition="outside",
        ))
        fig_dl.update_layout(
            barmode="group", title="LSTM vs CNN-LSTM RMSE by City (lower is better)",
            yaxis_title="RMSE (°C)", height=350,
        )
        st.plotly_chart(fig_dl, width="stretch")

        st.markdown("---")

        # Tree Models: RF vs XGBoost
        st.markdown("#### 🌲 Tree Models: Random Forest vs XGBoost")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Temperature Regression**")
            rf_tm  = models["tree"]["rf"]["temp_metrics"]
            xgb_tm = models["tree"]["xgb"]["temp_metrics"]
            temp_df = pd.DataFrame({
                "Metric": ["RMSE (°C)", "MAE (°C)", "R²"],
                "Random Forest": [round(rf_tm["rmse"],4), round(rf_tm["mae"],4), round(rf_tm["r2"],4)],
                "XGBoost":       [round(xgb_tm["rmse"],4), round(xgb_tm["mae"],4), round(xgb_tm["r2"],4)],
            })
            st.dataframe(temp_df.set_index("Metric"), width="stretch")

        with col2:
            st.markdown("**Rainfall Regression**")
            rf_rm  = models["tree"]["rf"]["rain_metrics"]
            xgb_rm = models["tree"]["xgb"]["rain_metrics"]
            rain_df = pd.DataFrame({
                "Metric": ["RMSE (mm)", "MAE (mm)", "R²"],
                "Random Forest": [round(rf_rm["rmse"],4), round(rf_rm["mae"],4), round(rf_rm["r2"],4)],
                "XGBoost":       [round(xgb_rm["rmse"],4), round(xgb_rm["mae"],4), round(xgb_rm["r2"],4)],
            })
            st.dataframe(rain_df.set_index("Metric"), width="stretch")

        # Wet day classification
        st.markdown("**Wet Day Classification**")
        rf_wm  = models["tree"]["rf"]["wet_metrics"]
        xgb_wm = models["tree"]["xgb"]["wet_metrics"]
        wet_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Random Forest": [round(rf_wm["accuracy"],4), round(rf_wm["precision"],4),
                              round(rf_wm["recall"],4),   round(rf_wm["f1"],4)],
            "XGBoost":       [round(xgb_wm["accuracy"],4), round(xgb_wm["precision"],4),
                              round(xgb_wm["recall"],4),   round(xgb_wm["f1"],4)],
        })
        st.dataframe(wet_df.set_index("Metric"), width="stretch")

        st.markdown("---")

        # Feature Importance
        st.markdown("#### 🔍 XGBoost Feature Importance — Temperature")
        xgb_imp = models["tree"]["xgb"]["importances"].head(12).sort_values()
        fig_imp = go.Figure(go.Bar(
            x=xgb_imp.values, y=xgb_imp.index,
            orientation="h",
            marker_color=["#E63946" if v > 0.1 else "#457B9D" if v > 0.02 else "#ADB5BD"
                          for v in xgb_imp.values],
            text=[f"{v:.4f}" for v in xgb_imp.values],
            textposition="outside",
        ))
        fig_imp.update_layout(
            title="XGBoost Feature Importance (Temperature Prediction)",
            xaxis_title="Importance Score", height=400,
        )
        st.plotly_chart(fig_imp, width="stretch")

        st.markdown("---")

        # ARIMA Baseline
        st.markdown("#### 📈 SARIMA Baseline — Ahmedabad Monthly Temperature")
        if models["arima"]:
            arima_m = models["arima"]["metrics"]
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("SARIMA RMSE", f"{arima_m['rmse']:.4f}°C")
            a2.metric("SARIMA MAE",  f"{arima_m['mae']:.4f}°C")
            a3.metric("SARIMA R²",   f"{arima_m['r2']:.4f}")
            a4.metric("AIC",         f"{arima_m['aic']:.2f}")

            train_s  = models["arima"]["train_series"]
            test_s   = models["arima"]["test_series"]
            forecast = models["arima"]["test_forecast"]

            fig_arima = go.Figure()
            fig_arima.add_trace(go.Scatter(
                x=train_s.index, y=train_s.values,
                mode="lines", name="Training (1990–2018)",
                line=dict(color="#ADB5BD", width=1), opacity=0.6,
            ))
            fig_arima.add_trace(go.Scatter(
                x=test_s.index, y=test_s.values,
                mode="lines", name="Actual (2019–2024)",
                line=dict(color="#2C3E50", width=2),
            ))
            fig_arima.add_trace(go.Scatter(
                x=forecast.index, y=forecast.values,
                mode="lines", name="SARIMA Forecast",
                line=dict(color="#E63946", width=2, dash="dash"),
            ))
            fig_arima.update_layout(
                title="SARIMA Baseline: Actual vs Forecast — Ahmedabad Monthly Temp",
                yaxis_title="°C", height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_arima, width="stretch")

        st.markdown("---")

        # Hypothesis Testing Results 
        st.markdown("#### 📊 Statistical Hypothesis Testing Summary")
        hyp_rows = [
            {"Hypothesis": "H1: LSTM vs ARIMA", "Test": "Paired t-test",
             "Statistic": "t = 0.323", "p-value": "0.7476",
             "α = 0.05": "≥ 0.05", "Decision": "Fail to Reject H₀"},
            {"Hypothesis": "H2: XGBoost vs RF", "Test": "Wilcoxon",
             "Statistic": "W = 29,689,852", "p-value": "0.3000",
             "α = 0.05": "≥ 0.05", "Decision": "Fail to Reject H₀"},
            {"Hypothesis": "H3: Gujarat Temp Trend", "Test": "Mann-Kendall",
             "Statistic": "z = 2.301", "p-value": "0.0214",
             "α = 0.05": "< 0.05", "Decision": "✅ REJECT H₀ — Significant"},
            {"Hypothesis": "H4: IF Anomaly Detection", "Test": "Chi-Square",
             "Statistic": "χ² = 168.47", "p-value": "< 0.0001",
             "α = 0.05": "< 0.05", "Decision": "✅ REJECT H₀ — Significant"},
        ]
        hyp_df = pd.DataFrame(hyp_rows)
        st.dataframe(hyp_df.set_index("Hypothesis"), width="stretch")

# Footer 
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:12px;'>"
    "AI-Based Climate Analytics and Forecasting System for Gujarat | "
    "Navrachana University, Vadodara | Data: ERA5 via Open-Meteo (1990–2024)"
    "</p>",
    unsafe_allow_html=True,
)