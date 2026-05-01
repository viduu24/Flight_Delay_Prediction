import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from styles import apply_theme, PLOTLY_LAYOUT, PURPLE_SEQ, ACCENT_COLOR, LEGEND_H, LEGEND_DEFAULT
from utils import load_merged

st.set_page_config(page_title="Weather EDA", page_icon="🌦️", layout="wide")
apply_theme()

df = load_merged()

if df is None:
    st.error("Dataset could not be loaded.")
    st.stop()

for col in ["is_delay","weather_delay","temperature_c","humidity_pct","precipitation_mm","wind_speed_kmh"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

has_wx = all(c in df.columns for c in ["temperature_c","humidity_pct","precipitation_mm","wind_speed_kmh"])

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌦️ Weather EDA</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#6B6A9B;'>How temperature, precipitation, wind, and humidity relate to flight delays.</p>", unsafe_allow_html=True)
st.divider()

# ── KPIs ───────────────────────────────────────────────────────────────────────
total      = len(df)
delayed    = int(df["is_delay"].fillna(0).sum())
wx_delayed = int((df["weather_delay"].fillna(0) > 0).sum())
avg_wx_min = df[df["weather_delay"] > 0]["weather_delay"].mean()
coverage   = (df["weather_data_present"] == "Yes").sum() if "weather_data_present" in df.columns else total

c1, c2, c3, c4 = st.columns(4)
c1.metric("Weather-Caused Delays", f"{wx_delayed:,}")
c2.metric("% of Delayed Flights",  f"{wx_delayed/delayed:.1%}" if delayed else "N/A")
c3.metric("Avg Weather Delay",     f"{avg_wx_min:.1f} min")
c4.metric("Sensor Coverage",       f"{coverage/total:.1%}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Cause Breakdown", "Sensor Distributions",
    "Avg Conditions", "Correlations", "Monthly Trend"
])

# ────────────── TAB 1 ──────────────
with tab1:
    st.markdown("### How Many Delayed Flights Were Weather-Caused?")
    col1, col2 = st.columns(2)

    not_delayed   = total - delayed
    delayed_no_wx = delayed - wx_delayed

    with col1:
        fig = px.pie(
            values=[not_delayed, delayed_no_wx, wx_delayed],
            names=["Not Delayed", "Delayed (Other)", "Weather-Caused"],
            color_discrete_sequence=["#C4B5E8", "#7B6DC4", "#2D2B6B"],
            hole=0.4,
            title="All Flights — Delay Breakdown",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=340)
        fig.update_layout(**LEGEND_H)
        fig.update_traces(marker=dict(line=dict(color="#EEF0F8", width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "season" in df.columns:
            season_order = ["Winter","Spring","Summer","Fall"]
            seas = (
                df.groupby("season")
                .agg(total_delayed=("is_delay","sum"),
                     wx_delayed=("weather_delay", lambda x: (x > 0).sum()))
                .reindex(season_order).reset_index()
            )
            seas["wx_pct"] = seas["wx_delayed"] / seas["total_delayed"] * 100
            fig = px.bar(seas, x="season", y="wx_pct", color="season",
                         color_discrete_sequence=["#2D2B6B","#7B6DC4","#9B89C4","#C4B5E8"],
                         title="Weather Delay % of All Delays by Season",
                         labels={"wx_pct":"% of Delayed Flights","season":"Season"})
            fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Weather Condition vs Delay Rate")
    if "weather_condition" in df.columns:
        wc = df.groupby("weather_condition")["is_delay"].mean().reset_index()
        wc.columns = ["Condition","Delay Rate"]
        fig = px.bar(wc.sort_values("Delay Rate", ascending=False),
                     x="Condition", y="Delay Rate",
                     color="Delay Rate",
                     color_continuous_scale=["#C4B5E8","#9B89C4","#4A3F9F","#2D2B6B"])
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          yaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">☁️ Thunderstorm and Fog conditions drive the highest delay rates. Clear skies show the lowest.</div>', unsafe_allow_html=True)

# ────────────── TAB 2 ──────────────
with tab2:
    st.markdown("### Sensor Conditions: Weather-Caused vs Other Delays")

    if has_wx:
        delayed_df  = df[df["is_delay"] == 1].copy()
        wx_group    = delayed_df[delayed_df["weather_delay"] > 0]
        no_wx_group = delayed_df[delayed_df["weather_delay"] == 0]

        wx_vars   = ["temperature_c","humidity_pct","precipitation_mm","wind_speed_kmh"]
        wx_labels = {"temperature_c":"Temperature (°C)","humidity_pct":"Humidity (%)",
                     "precipitation_mm":"Precipitation (mm)","wind_speed_kmh":"Wind Speed (km/h)"}

        tabs_inner = st.tabs([wx_labels[v] for v in wx_vars])
        for t, var in zip(tabs_inner, wx_vars):
            with t:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=no_wx_group[var].dropna(), name="Other Delays",
                    marker_color="#C4B5E8", opacity=0.65, nbinsx=40,
                ))
                fig.add_trace(go.Histogram(
                    x=wx_group[var].dropna(), name="Weather Delay > 0",
                    marker_color="#2D2B6B", opacity=0.70, nbinsx=40,
                ))
                fig.update_layout(
                    barmode="overlay",
                    title=f"{wx_labels[var]}: Weather-Caused vs Other Delays",
                    **PLOTLY_LAYOUT,
                    xaxis_title=wx_labels[var], yaxis_title="Count", height=340,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="insight-box">🌧️ Higher precipitation and wind speed clearly separate weather-caused delays. Temperature alone is a weaker predictor — extreme cold or heat both contribute.</div>', unsafe_allow_html=True)
    else:
        st.info("Weather sensor columns not available in dataset.")

# ────────────── TAB 3 ──────────────
with tab3:
    st.markdown("### Average Weather Conditions — Delayed vs Not Delayed")

    if has_wx:
        avg_cond = df.groupby("is_delay")[
            ["temperature_c","humidity_pct","precipitation_mm","wind_speed_kmh"]
        ].mean()
        avg_cond.index = avg_cond.index.map({0:"Not Delayed",1:"Delayed"})
        avg_melt = avg_cond.reset_index().melt(id_vars="is_delay", var_name="Metric", value_name="Average")
        avg_melt["Metric"] = avg_melt["Metric"].map({
            "temperature_c":"Temp (°C)","humidity_pct":"Humidity (%)",
            "precipitation_mm":"Precip (mm)","wind_speed_kmh":"Wind (km/h)"
        })

        fig = px.bar(avg_melt, x="Metric", y="Average", color="is_delay",
                     barmode="group",
                     color_discrete_map={"Not Delayed":"#C4B5E8","Delayed":"#2D2B6B"},
                     labels={"is_delay":"Status"})
        fig.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">Delayed flights on average experience slightly higher precipitation and wind. The differences are modest — weather is one factor among several.</div>', unsafe_allow_html=True)

# ────────────── TAB 4 ──────────────
with tab4:
    st.markdown("### Correlation: Weather Variables vs IS_Delay")

    if has_wx:
        corr_cols = ["temperature_c","humidity_pct","precipitation_mm","wind_speed_kmh","is_delay"]
        corr = df[corr_cols].corr()
        pretty = {
            "temperature_c":"Temp (°C)","humidity_pct":"Humidity (%)",
            "precipitation_mm":"Precip (mm)","wind_speed_kmh":"Wind (km/h)",
            "is_delay":"IS_Delay"
        }
        corr.index = [pretty.get(c,c) for c in corr.index]
        corr.columns = [pretty.get(c,c) for c in corr.columns]

        fig = px.imshow(
            corr, color_continuous_scale=["#2D2B6B","#FFFFFF","#9B89C4"],
            zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
            title="Correlation Matrix — Weather + IS_Delay",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">Precipitation_mm shows the strongest positive correlation with IS_Delay. All correlations are modest — consistent with flight delay being a multi-causal phenomenon.</div>', unsafe_allow_html=True)

# ────────────── TAB 5 ──────────────
with tab5:
    st.markdown("### Monthly Weather Delay Trend")

    if "month" in df.columns and "weather_delay" in df.columns:
        month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly = (
            df.groupby("month")
            .agg(total=("is_delay","count"),
                 wx_delayed=("weather_delay", lambda x: (x > 0).sum()))
            .reset_index()
        )
        monthly["wx_pct"]      = monthly["wx_delayed"] / monthly["total"] * 100
        monthly["month_name"]  = monthly["month"].map(month_map)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["month_name"], y=monthly["wx_pct"],
            mode="lines+markers", fill="tozeroy",
            line=dict(color="#2D2B6B", width=2.5),
            fillcolor="rgba(155,137,196,0.15)",
            marker=dict(color=ACCENT_COLOR, size=8),
            name="Weather Delay %",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Monthly % of Flights with Weather-Caused Delay",
            xaxis_title="Month", yaxis_title="% of Flights",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">📅 Winter months (Dec–Feb) and peak summer (Jul–Aug) show elevated weather delay rates — driven by snowstorms and severe convective activity respectively.</div>', unsafe_allow_html=True)
