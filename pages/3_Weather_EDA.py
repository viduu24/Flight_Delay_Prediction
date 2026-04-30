import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Weather EDA", page_icon="🌦️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #0f1629; border-right: 1px solid #1e2d4a; }
    h1,h2,h3 { color: #e2e8f0 !important; }
    .section-header {
        font-size: 1.5rem; font-weight: 600;
        background: linear-gradient(135deg, #60a5fa, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .insight-box {
        background: #111827; border-left: 3px solid #818cf8;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.8rem 0;
        color: #94a3b8; font-size: 0.88rem;
    }
    .stat-card {
        background: linear-gradient(135deg,#111827,#1a2438);
        border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center;
    }
    .stat-card .val { font-size: 1.6rem; font-weight: 700; color: #60a5fa; }
    .stat-card .lbl { font-size: 0.76rem; color: #64748b; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#94a3b8", title_font_color="#e2e8f0",
    xaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
)

@st.cache_data(show_spinner="Loading merged dataset…")
def load_data():
    for p in [Path("../../Merged_Data/merged_flights.csv"),
              Path("../Merged_Data/merged_flights.csv"),
              Path("Merged_Data/merged_flights.csv"),
              Path("merged_flights.csv")]:
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    return None

df = load_data()

st.markdown('<div class="section-header">🌦️ Weather EDA</div>', unsafe_allow_html=True)
st.markdown("How temperature, precipitation, wind speed, and humidity relate to flight delay rates and weather-caused delay minutes.")

if df is None:
    st.warning("⚠️ **merged_flights.csv not found.** Place it at `../../Merged_Data/merged_flights.csv` and reload.")
    st.stop()

# Ensure numeric
for col in ["IS_Delay","weather_delay","Temperature_C","Humidity_pct","Precipitation_mm","Wind_Speed_kmh"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

has_weather_cols = all(c in df.columns for c in ["Temperature_C","Humidity_pct","Precipitation_mm","Wind_Speed_kmh"])

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
if "weather_delay" in df.columns and "IS_Delay" in df.columns:
    total = len(df)
    delayed = int(df["IS_Delay"].fillna(0).sum())
    wx_delayed = int((df["weather_delay"].fillna(0) > 0).sum())
    coverage = (df["Weather_Data_Present"] == "Yes").sum() if "Weather_Data_Present" in df.columns else None
    avg_wx_min = df[df["weather_delay"] > 0]["weather_delay"].mean()

    c1,c2,c3,c4 = st.columns(4)
    cards = [
        (f"{wx_delayed:,}", "Weather-Caused Delays"),
        (f"{wx_delayed/delayed:.1%}" if delayed else "N/A", "% of Delayed Flights"),
        (f"{avg_wx_min:.1f} min", "Avg Weather Delay (when > 0)"),
        (f"{coverage/total:.1%}" if coverage else "N/A", "Flights with Sensor Data"),
    ]
    for col_ui, (val, lbl) in zip([c1,c2,c3,c4], cards):
        with col_ui:
            st.markdown(f'<div class="stat-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Section 1: Weather vs Non-weather delay breakdown ─────────────────────────
st.markdown("### 1 · How Many Delayed Flights Were Weather-Caused?")

if "weather_delay" in df.columns and "IS_Delay" in df.columns:
    not_delayed = total - delayed
    wx_d = int((df["weather_delay"].fillna(0) > 0).sum())
    delayed_no_wx = delayed - wx_d

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            values=[not_delayed, delayed_no_wx, wx_d],
            names=["Not Delayed","Delayed (No Weather)","Weather-Caused Delay"],
            color_discrete_sequence=["#0F4C81","#E9C46A","#FF6B6B"],
            title="All Flights — Delay Breakdown",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Season" in df.columns:
            season_order = ["Winter","Spring","Summer","Fall"]
            seas = (
                df.groupby("Season")
                .agg(total_flights=("IS_Delay","count"),
                     total_delayed=("IS_Delay","sum"),
                     wx_delayed=("weather_delay", lambda x: (x>0).sum()))
                .reindex(season_order)
                .reset_index()
            )
            seas["wx_pct_of_delayed"] = seas["wx_delayed"] / seas["total_delayed"] * 100
            fig = px.bar(seas, x="Season", y="wx_pct_of_delayed",
                         color="Season", color_discrete_sequence=["#60a5fa","#4ade80","#fbbf24","#f97316"],
                         title="Weather Delay % of All Delays by Season",
                         labels={"wx_pct_of_delayed":"% of Delayed Flights"})
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ── Section 2: Sensor conditions vs delay ────────────────────────────────────
st.markdown("### 2 · Sensor Conditions: Weather-Caused vs Non-Weather Delays")

if has_weather_cols and "weather_delay" in df.columns and "IS_Delay" in df.columns:
    delayed_df = df[df["IS_Delay"] == 1].copy()
    wx_group = delayed_df[delayed_df["weather_delay"] > 0]
    no_wx_group = delayed_df[delayed_df["weather_delay"] == 0]

    weather_vars = ["Temperature_C","Humidity_pct","Precipitation_mm","Wind_Speed_kmh"]
    var_labels = {"Temperature_C":"Temperature (°C)", "Humidity_pct":"Humidity (%)",
                  "Precipitation_mm":"Precipitation (mm)", "Wind_Speed_kmh":"Wind Speed (km/h)"}

    tab_names = [var_labels[v] for v in weather_vars]
    tabs = st.tabs(tab_names)

    for tab, var in zip(tabs, weather_vars):
        with tab:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=no_wx_group[var].dropna(), name="No Weather Delay",
                                        marker_color="#0F4C81", opacity=0.6, nbinsx=40))
            fig.add_trace(go.Histogram(x=wx_group[var].dropna(), name="Weather Delay > 0",
                                        marker_color="#FF6B6B", opacity=0.6, nbinsx=40))
            fig.update_layout(barmode="overlay", title=f"{var_labels[var]}: Weather-Caused vs Other Delays",
                               **PLOTLY_LAYOUT, xaxis_title=var_labels[var], yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">Higher precipitation and wind speed are associated with weather-caused delays. Temperature alone is a weaker predictor — very cold and very warm temperatures both correlate with higher delay rates through different mechanisms (ice/snow vs. thunderstorms).</div>', unsafe_allow_html=True)

# ── Section 3: Average conditions comparison ─────────────────────────────────
st.markdown("### 3 · Average Weather Conditions — Delayed vs Not Delayed")

if has_weather_cols and "IS_Delay" in df.columns:
    avg_cond = df.groupby("IS_Delay")[["Temperature_C","Humidity_pct","Precipitation_mm","Wind_Speed_kmh"]].mean()
    avg_cond.index = avg_cond.index.map({0:"Not Delayed",1:"Delayed"})
    avg_melt = avg_cond.reset_index().melt(id_vars="IS_Delay", var_name="Metric", value_name="Average")

    fig = px.bar(avg_melt, x="Metric", y="Average", color="IS_Delay", barmode="group",
                 color_discrete_map={"Not Delayed":"#0F4C81","Delayed":"#FF6B6B"},
                 title="Avg Weather Conditions: Delayed vs Not Delayed")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ── Section 4: Correlation heatmap ───────────────────────────────────────────
st.markdown("### 4 · Correlation: Weather Variables vs IS_Delay")

if has_weather_cols and "IS_Delay" in df.columns:
    corr_cols = ["Temperature_C","Humidity_pct","Precipitation_mm","Wind_Speed_kmh","IS_Delay"]
    corr = df[corr_cols].corr()

    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    text_auto=".2f", title="Correlation Matrix — Weather + IS_Delay",
                    aspect="auto")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">Precipitation_mm shows the strongest positive correlation with IS_Delay among the weather sensor variables. The correlations are modest overall — weather is one contributor among several, which is why the model combines weather with flight-operations features.</div>', unsafe_allow_html=True)

# ── Section 5: Monthly weather delay trend ────────────────────────────────────
st.markdown("### 5 · Monthly Weather Delay Trend")

if "month" in df.columns and "weather_delay" in df.columns:
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = (
        df.groupby("month")
        .agg(total=("IS_Delay","count"),
             wx_delayed=("weather_delay", lambda x: (x>0).sum()))
        .reset_index()
    )
    monthly["wx_pct"] = monthly["wx_delayed"] / monthly["total"] * 100
    monthly["month_name"] = monthly["month"].map(month_map)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["wx_pct"], mode="lines+markers",
                              fill="tozeroy", line=dict(color="#60a5fa", width=2.5),
                              fillcolor="rgba(96,165,250,0.12)", name="Weather Delay %"))
    fig.update_layout(title="Monthly % of Flights with Weather-Caused Delay",
                       xaxis_title="Month", yaxis_title="% of Flights", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
