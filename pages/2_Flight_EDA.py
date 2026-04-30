import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Flight EDA", page_icon="✈️", layout="wide")

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
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background: #111827; border-left: 3px solid #818cf8;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.8rem 0;
        color: #94a3b8; font-size: 0.88rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #111827, #1a2438);
        border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center;
    }
    .kpi-card .val { font-size: 1.8rem; font-weight: 700; color: #60a5fa; }
    .kpi-card .lbl { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#0F4C81","#FF6B6B","#2A9D8F","#E9C46A","#9B5DE5","#F4A261","#457B9D","#8D99AE"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#94a3b8", title_font_color="#e2e8f0",
    xaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading merged dataset…")
def load_data():
    # __file__ is .../pages/2_...py  →  parent = pages/  →  parent.parent = repo root
    here = Path(__file__).resolve().parent.parent  # repo root
    candidates = [
        here / "Merged_Data" / "merged_flights.csv",          # root/Merged_Data/
        here.parent / "Merged_Data" / "merged_flights.csv",   # one level above root
        Path("Merged_Data/merged_flights.csv"),                # cwd fallback
        Path("merged_flights.csv"),                            # cwd fallback
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    return None

df = load_data()

st.markdown('<div class="section-header">✈️ Flight EDA</div>', unsafe_allow_html=True)
st.markdown("Exploratory analysis of flight operations — delay rates, severity, airline rankings, time patterns, and route concentration.")

if df is None:
    st.warning("⚠️ **merged_flights.csv not found.** Charts will populate once the file is available.")
    here = Path(__file__).resolve().parent.parent
    with st.expander("🔍 Debug — paths checked"):
        st.code(f"""
Script location : {Path(__file__).resolve()}
Repo root guess : {here}

Paths checked:
  1. {here / 'Merged_Data' / 'merged_flights.csv'}
  2. {here.parent / 'Merged_Data' / 'merged_flights.csv'}
  3. {Path('Merged_Data/merged_flights.csv').resolve()}
  4. {Path('merged_flights.csv').resolve()}

Current working directory: {Path.cwd()}
        """)
        st.info("Place **merged_flights.csv** in the `Merged_Data/` folder at the repo root, then reload the page.")
    st.stop()

# ── Prep ──────────────────────────────────────────────────────────────────────
for col in ["IS_Delay","delay_in_minutes","Departure_Hour","month","day_of_week",
            "carrier_delay","weather_delay","nas_delay","security_delay","late_aircraft_delay"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
weekday_map = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}

if "month_name" not in df.columns and "month" in df.columns:
    df["month_name"] = df["month"].map(month_map)
if "weekday_name" not in df.columns and "day_of_week" in df.columns:
    df["weekday_name"] = df["day_of_week"].map(weekday_map)

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total = len(df)
delayed = int(df["IS_Delay"].fillna(0).sum())
delay_rate = delayed / total if total > 0 else 0
avg_min = df["delay_in_minutes"].mean()
med_min = df["delay_in_minutes"].median()

c1,c2,c3,c4,c5 = st.columns(5)
for col_ui, val, lbl in zip(
    [c1,c2,c3,c4,c5],
    [f"{total:,}", f"{delayed:,}", f"{delay_rate:.1%}", f"{avg_min:.1f} min", f"{med_min:.1f} min"],
    ["Total Flights","Delayed Flights","Delay Rate","Avg Delay","Median Delay"]
):
    with col_ui:
        st.markdown(f'<div class="kpi-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 1: Target Balance ─────────────────────────────────────────────────
st.markdown("### 1 · Target Balance & Delay Distribution")
col1, col2 = st.columns(2)

with col1:
    bal = df["IS_Delay"].fillna(0).astype(int).value_counts().reset_index()
    bal.columns = ["IS_Delay","count"]
    bal["label"] = bal["IS_Delay"].map({0:"Not Delayed",1:"Delayed (>15 min)"})
    fig = px.pie(bal, values="count", names="label", color_discrete_sequence=["#0F4C81","#FF6B6B"],
                 title="Target Balance")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    sample = df["delay_in_minutes"].dropna().clip(-60,300)
    fig = px.histogram(sample, nbins=80, color_discrete_sequence=["#60a5fa"],
                       title="Delay in Minutes Distribution (clipped −60 to 300)")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="insight-box">About 29% of flights are delayed by more than 15 minutes. The distribution is right-skewed — most delays are mild, but a long tail of severe delays (60–300+ min) drives the average up.</div>', unsafe_allow_html=True)

# ── Section 2: Time Patterns ──────────────────────────────────────────────────
st.markdown("### 2 · Time Patterns")
tab1, tab2, tab3, tab4 = st.tabs(["By Month","By Season","By Weekday","By Departure Hour"])

with tab1:
    if "month" in df.columns:
        monthly = df.groupby("month").agg(delay_rate=("IS_Delay","mean"), avg_min=("delay_in_minutes","mean")).reset_index()
        monthly["month_name"] = monthly["month"].map(month_map)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["delay_rate"], mode="lines+markers",
                                  name="Delay Rate", line=dict(color="#60a5fa",width=2.5)))
        fig.update_layout(title="Monthly Delay Rate", **PLOTLY_LAYOUT, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if "Season" in df.columns:
        seas = df.groupby("Season").agg(delay_rate=("IS_Delay","mean"), avg_min=("delay_in_minutes","mean")).reset_index()
        fig = px.bar(seas.sort_values("delay_rate",ascending=False), x="Season", y="delay_rate",
                     color="Season", color_discrete_sequence=PALETTE,
                     title="Delay Rate by Season", labels={"delay_rate":"Delay Rate"})
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if "weekday_name" in df.columns:
        wday_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        wday = df.groupby("weekday_name").agg(delay_rate=("IS_Delay","mean")).reset_index()
        wday["weekday_name"] = pd.Categorical(wday["weekday_name"], categories=wday_order, ordered=True)
        wday = wday.sort_values("weekday_name")
        fig = px.bar(wday, x="weekday_name", y="delay_rate", color_discrete_sequence=["#818cf8"],
                     title="Delay Rate by Day of Week", labels={"weekday_name":"","delay_rate":"Delay Rate"})
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    if "Departure_Hour" in df.columns:
        hourly = df.groupby("Departure_Hour").agg(delay_rate=("IS_Delay","mean")).reset_index().dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly["Departure_Hour"], y=hourly["delay_rate"], mode="lines+markers",
                                  fill="tozeroy", line=dict(color="#a78bfa",width=2),
                                  fillcolor="rgba(167,139,250,0.15)"))
        fig.update_layout(title="Delay Rate by Departure Hour", **PLOTLY_LAYOUT, yaxis_tickformat=".0%",
                          xaxis_title="Hour of Day", yaxis_title="Delay Rate")
        st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="insight-box">Delays peak in summer (June–August) and around major holidays. Within a day, early-morning flights (5–7 AM) have the lowest delay rates. Risk climbs steadily through the afternoon and peaks in the evening (6–9 PM) due to cascading schedule disruptions.</div>', unsafe_allow_html=True)

# ── Section 3: Airline Rankings ───────────────────────────────────────────────
st.markdown("### 3 · Airline Performance")

if "op_unique_carrier" in df.columns:
    airline_perf = (
        df.groupby("op_unique_carrier")
        .agg(flights=("IS_Delay","size"), delay_rate=("IS_Delay","mean"), avg_min=("delay_in_minutes","mean"))
        .reset_index()
        .sort_values("delay_rate")
    )
    # Filter to airlines with reasonable volume
    airline_perf = airline_perf[airline_perf["flights"] >= 1000]

    fig = px.bar(airline_perf.sort_values("delay_rate", ascending=True),
                 x="delay_rate", y="op_unique_carrier", orientation="h",
                 color="delay_rate", color_continuous_scale="RdYlGn_r",
                 title="Delay Rate by Airline (carriers with ≥1,000 flights)",
                 labels={"delay_rate":"Delay Rate","op_unique_carrier":"Carrier"})
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Section 4: Origin Analysis ────────────────────────────────────────────────
st.markdown("### 4 · Top Origin Cities")

if "origin_city" in df.columns:
    col1, col2 = st.columns(2)
    origin = df.groupby("origin_city").agg(flights=("IS_Delay","size"), delay_rate=("IS_Delay","mean")).reset_index()

    with col1:
        top_vol = origin.nlargest(12,"flights")
        fig = px.bar(top_vol.sort_values("flights"), x="flights", y="origin_city", orientation="h",
                     color_discrete_sequence=["#0F4C81"], title="Top 12 Origins by Volume",
                     labels={"flights":"Flights","origin_city":""})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_rate = origin[origin["flights"] >= 3000].nlargest(12,"delay_rate")
        fig = px.bar(top_rate.sort_values("delay_rate"), x="delay_rate", y="origin_city", orientation="h",
                     color_discrete_sequence=["#FF6B6B"], title="Highest Delay Rate Origins (≥3,000 flights)",
                     labels={"delay_rate":"Delay Rate","origin_city":""})
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

# ── Section 5: Delay Reasons ──────────────────────────────────────────────────
st.markdown("### 5 · Delay Reason Breakdown")

delay_cols = ["carrier_delay","weather_delay","nas_delay","security_delay","late_aircraft_delay"]
reason_cols = [c for c in delay_cols if c in df.columns]

if reason_cols:
    reason_total = df[reason_cols].sum().reset_index()
    reason_total.columns = ["reason","total_minutes"]
    reason_total["reason"] = reason_total["reason"].str.replace("_delay","").str.replace("_"," ").str.title()
    reason_total = reason_total.sort_values("total_minutes", ascending=True)

    fig = px.bar(reason_total, x="total_minutes", y="reason", orientation="h",
                 color="total_minutes", color_continuous_scale="Blues",
                 title="Total Delay Minutes by Cause",
                 labels={"total_minutes":"Total Minutes","reason":""})
    fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">Late aircraft delays and carrier delays account for the largest share of total delay minutes, followed by NAS (National Airspace System) congestion. Weather delay is a smaller but significant contributor — and is directly actionable with sensor data.</div>', unsafe_allow_html=True)
