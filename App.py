import streamlit as st
from styles import apply_theme, PLOTLY_LAYOUT, PURPLE_SEQ, ACCENT_COLOR
from utils import load_merged
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# Pre-load data into cache so all pages share it
df = load_merged()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 6px 0;'>
      <div style='font-family:Space Mono,monospace; font-size:1.1rem;
                  font-weight:700; color:#2D2B6B; line-height:1.3;'>
        ✈️ Flight Delay<br>Prediction
      </div>
      <div style='font-size:0.75rem; color:#9B89C4; margin-top:4px;'>
        AhmadJabbar2502
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:0.8rem; color:#6B6A9B; line-height:1.8;'>
    <b style='color:#2D2B6B;'>Navigate:</b><br>
    • <b>App.py</b> — Overview<br>
    • <b>1_Dataset</b> — Data description<br>
    • <b>2_Flight_EDA</b> — Flight patterns<br>
    • <b>3_Weather_EDA</b> — Weather impact<br>
    • <b>4_Models</b> — ML results<br>
    • <b>5_Predict</b> — Live predictor
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:0.76rem; color:#9B89C4;'>
    <b>Data:</b> BTS Flights + Weather (2024)<br>
    <b>Models:</b> Bagged Trees · KNN<br>
    <b>Target:</b> IS_Delay (≥ 15 min)
    </div>
    """, unsafe_allow_html=True)

# ── Page ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">✈️ Flight Delay Prediction</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#6B6A9B; font-size:1rem;'>End-to-end ML system predicting U.S. domestic flight delays using combined flight and weather data.</p>", unsafe_allow_html=True)
st.divider()

# ── KPIs ───────────────────────────────────────────────────────────────────────
total      = len(df)
delayed    = int(df["is_delay"].sum())
on_time    = total - delayed
delay_rate = delayed / total
avg_delay  = df[df["is_delay"] == 1]["delay_in_minutes"].mean()
wx_delays  = int((df["weather_delay"] > 0).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Flights",    f"{total:,}")
c2.metric("Delayed",          f"{delayed:,}",       f"{delay_rate:.1%} of all")
c3.metric("On Time",          f"{on_time:,}",        f"{1-delay_rate:.1%} of all")
c4.metric("Avg Delay",        f"{avg_delay:.0f} min","when delayed")
c5.metric("Weather-Caused",   f"{wx_delays:,}",      f"{wx_delays/delayed:.1%} of delayed")

st.divider()

# ── Two charts side by side ────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Delay Rate by Airline")
    airline = (
        df.groupby("op_unique_carrier")["is_delay"]
        .mean().reset_index()
        .rename(columns={"op_unique_carrier": "Airline", "is_delay": "Delay Rate"})
        .sort_values("Delay Rate")
    )
    fig = px.bar(
        airline, x="Delay Rate", y="Airline", orientation="h",
        color="Delay Rate",
        color_continuous_scale=["#C4B5E8", "#9B89C4", "#7B6DC4", "#4A3F9F", "#2D2B6B"],
        template="plotly_white",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                      xaxis_tickformat=".0%", coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Delay Distribution")
    fig2 = px.pie(
        names=["On Time", "Delayed"],
        values=[on_time, delayed],
        color_discrete_sequence=["#C4B5E8", "#2D2B6B"],
        hole=0.42,
        template="plotly_white",
    )
    fig2.update_layout(**PLOTLY_LAYOUT, height=340,
                       legend=dict(orientation="h", y=-0.1))
    fig2.update_traces(textfont_color="#FFFFFF",
                       marker=dict(line=dict(color="#EEF0F8", width=2)))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Project goals ──────────────────────────────────────────────────────────────
st.markdown("### 🎯 Project Goals")
col1, col2, col3 = st.columns(3)
goals = [
    ("📊", "Binary Classification",
     "Predict <b>IS_Delay = 1</b> for flights arriving ≥ 15 minutes late, "
     "using only features known before departure."),
    ("🌦️", "Weather Integration",
     "Merge hourly weather sensor data (temperature, precipitation, wind, humidity) "
     "matched to each flight's origin city and departure time."),
    ("🤖", "Model Comparison",
     "Compare <b>Bagged Decision Trees</b> (AUC 0.747) vs <b>KNN</b> (AUC 0.712) "
     "against a majority-class baseline (AUC 0.500)."),
]
for col, (icon, title, body) in zip([col1, col2, col3], goals):
    with col:
        st.markdown(f"""
        <div class="card">
          <div style='font-size:1.8rem; margin-bottom:6px;'>{icon}</div>
          <h4 style='margin:0 0 6px 0; color:#2D2B6B;'>{title}</h4>
          <p style='color:#6B6A9B; font-size:0.88rem; margin:0;'>{body}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Quick insight strip ────────────────────────────────────────────────────────
st.markdown("### 💡 Key Findings")
insights = [
    "🕔 Afternoon departures (3–8 PM) have the highest delay rates — cascading schedule effects.",
    "🌧️ Thunderstorm and Fog conditions drive the sharpest weather-related delay spikes.",
    "📅 June–August and December see elevated delays due to peak travel and winter weather.",
    "🏆 Bagged Decision Trees outperform KNN on all metrics — F1 0.522 vs 0.480.",
]
for txt in insights:
    st.markdown(f'<div class="insight-box">{txt}</div>', unsafe_allow_html=True)
