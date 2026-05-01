import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from styles import apply_theme, PLOTLY_LAYOUT, PURPLE_SEQ, ACCENT_COLOR
from utils import load_merged

st.set_page_config(page_title="Flight EDA", page_icon="✈️", layout="wide")
apply_theme()

df = load_merged()

if df is None:
    st.error("❌ Dataset could not be loaded.")
    st.stop()

for col in ["is_delay", "delay_in_minutes", "departure_hour"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">✈️ Flight Delay EDA</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#6B6A9B;'>Patterns in delay rates across time, airline, route, and delay cause.</p>", unsafe_allow_html=True)
st.divider()

# ── KPIs ───────────────────────────────────────────────────────────────────────
total      = len(df)
delayed    = int(df["is_delay"].fillna(0).sum())
delay_rate = delayed / total
avg_delay  = df["delay_in_minutes"].mean()
med_delay  = df["delay_in_minutes"].median()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Flights",   f"{total:,}")
c2.metric("Delayed Flights", f"{delayed:,}")
c3.metric("Delay Rate",      f"{delay_rate:.1%}")
c4.metric("Avg Delay",       f"{avg_delay:.1f} min")
c5.metric("Median Delay",    f"{med_delay:.1f} min")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📅 Time Patterns", "✈️ Airlines", "🗺️ Routes & Distance", "⏱️ Delay Causes"])

# ────────────── TAB 1: Time ──────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Delay Distribution (On Time vs Delayed)")
        fig = px.pie(
            names=["Not Delayed", "Delayed"],
            values=[total - delayed, delayed],
            color_discrete_sequence=["#C4B5E8", "#2D2B6B"],
            hole=0.42,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          legend=dict(orientation="h", y=-0.15))
        fig.update_traces(marker=dict(line=dict(color="#EEF0F8", width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Delay Minutes Distribution (delayed only)")
        sample = df[df["is_delay"] == 1]["delay_in_minutes"].dropna().clip(0, 240)
        fig = px.histogram(sample, nbins=40,
                           color_discrete_sequence=[ACCENT_COLOR])
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          xaxis_title="Delay (min)", yaxis_title="Count")
        fig.update_traces(marker_line_width=0.5, marker_line_color="#EEF0F8")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Delay Rate by Hour of Day")
    hourly = df.groupby("departure_hour")["is_delay"].mean().reset_index()
    hourly.columns = ["Hour", "Delay Rate"]
    fig = px.area(hourly, x="Hour", y="Delay Rate",
                  color_discrete_sequence=[ACCENT_COLOR])
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      yaxis_tickformat=".0%", xaxis_title="Departure Hour")
    fig.update_traces(fillcolor="rgba(123,109,196,0.18)", line_color=ACCENT_COLOR)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box">🕔 Afternoon departures (15:00–20:00) show the highest delay rates due to cascading schedule disruptions built up throughout the day. Early morning (05:00–09:00) is the safest window.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### By Day of Week")
        dow_labels = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
        dow = (df.groupby("day_of_week")["is_delay"].mean()
               .reindex(range(1,8)).reset_index())
        dow.columns = ["DOW", "Delay Rate"]
        dow["Day"] = dow["DOW"].map(dow_labels)
        fig = px.bar(dow, x="Day", y="Delay Rate",
                     color="Delay Rate",
                     color_continuous_scale=["#C4B5E8","#9B89C4","#4A3F9F","#2D2B6B"])
        fig.update_layout(**PLOTLY_LAYOUT, height=290,
                          yaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### By Month")
        month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly = df.groupby("month")["is_delay"].mean().reset_index()
        monthly.columns = ["Month","Delay Rate"]
        monthly["Month Name"] = monthly["Month"].map(month_map)
        fig = px.line(monthly, x="Month Name", y="Delay Rate", markers=True,
                      color_discrete_sequence=["#2D2B6B"])
        fig.update_layout(**PLOTLY_LAYOUT, height=290, yaxis_tickformat=".0%")
        fig.update_traces(marker=dict(color=ACCENT_COLOR, size=8), line_width=2.5)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### By Season")
    season_order = ["Winter","Spring","Summer","Fall"]
    seas = (df.groupby("season")["is_delay"]
            .agg(["mean","count"]).reindex(season_order).reset_index())
    seas.columns = ["Season","Delay Rate","Flights"]
    fig = px.bar(seas, x="Season", y="Delay Rate",
                 color="Season",
                 color_discrete_sequence=["#2D2B6B","#7B6DC4","#9B89C4","#C4B5E8"])
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ────────────── TAB 2: Airlines ──────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Delay Rate by Airline")
        al = (df.groupby("op_unique_carrier")["is_delay"]
              .agg(["mean","count"]).reset_index())
        al.columns = ["Airline","Delay Rate","Flights"]
        fig = px.bar(al.sort_values("Delay Rate"), x="Delay Rate", y="Airline",
                     orientation="h",
                     color="Delay Rate",
                     color_continuous_scale=["#C4B5E8","#9B89C4","#7B6DC4","#2D2B6B"])
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          xaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Flight Volume by Airline")
        fig2 = px.pie(al, names="Airline", values="Flights", hole=0.35,
                      color_discrete_sequence=PURPLE_SEQ + ["#E0D9F5","#F0ECFC"])
        fig2.update_layout(**PLOTLY_LAYOUT, height=380,
                           legend=dict(orientation="h", y=-0.15))
        fig2.update_traces(marker=dict(line=dict(color="#EEF0F8", width=2)))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Average Delay Minutes by Airline (delayed flights only)")
    al_min = (df[df["is_delay"]==1]
              .groupby("op_unique_carrier")["delay_in_minutes"]
              .mean().reset_index())
    al_min.columns = ["Airline","Avg Delay (min)"]
    fig3 = px.bar(al_min.sort_values("Avg Delay (min)"), x="Avg Delay (min)", y="Airline",
                  orientation="h", color="Avg Delay (min)",
                  color_continuous_scale=["#C4B5E8","#7B6DC4","#2D2B6B"])
    fig3.update_layout(**PLOTLY_LAYOUT, height=320, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

# ────────────── TAB 3: Routes ──────────────
with tab3:
    st.markdown("#### Delay Rate by Origin City")
    city = (df.groupby("origin_city")["is_delay"]
            .agg(["mean","count"]).reset_index())
    city.columns = ["City","Delay Rate","Flights"]
    city = city[city["Flights"] > 50]
    fig = px.bar(city.sort_values("Delay Rate", ascending=False),
                 x="City", y="Delay Rate",
                 color="Delay Rate",
                 color_continuous_scale=["#C4B5E8","#9B89C4","#4A3F9F","#2D2B6B"])
    fig.update_layout(**PLOTLY_LAYOUT, height=340,
                      yaxis_tickformat=".0%", coloraxis_showscale=False,
                      xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Delay Rate by State")
        state = (df.groupby("origin_state")["is_delay"]
                 .agg(["mean","count"]).reset_index())
        state.columns = ["State","Delay Rate","Flights"]
        fig2 = px.bar(state.sort_values("Delay Rate", ascending=False).head(12),
                      x="State", y="Delay Rate",
                      color="Delay Rate",
                      color_continuous_scale=["#C4B5E8","#9B89C4","#2D2B6B"])
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                           yaxis_tickformat=".0%", coloraxis_showscale=False,
                           xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("#### Distance vs Delay Rate")
        df["dist_bin"] = pd.cut(df["distance"], bins=[0,500,1000,1500,2000,3000],
                                labels=["<500","500-1k","1k-1.5k","1.5k-2k",">2k"])
        dist = df.groupby("dist_bin", observed=True)["is_delay"].mean().reset_index()
        dist.columns = ["Distance (miles)","Delay Rate"]
        fig3 = px.bar(dist, x="Distance (miles)", y="Delay Rate",
                      color_discrete_sequence=["#7B6DC4"])
        fig3.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

# ────────────── TAB 4: Delay Causes ──────────────
with tab4:
    delay_cols = ["carrier_delay","weather_delay","nas_delay",
                  "security_delay","late_aircraft_delay"]
    existing = [c for c in delay_cols if c in df.columns]

    if existing:
        st.markdown("#### Total Delay Minutes by Cause")
        reason = df[existing].sum().reset_index()
        reason.columns = ["Cause","Minutes"]
        reason["Cause"] = reason["Cause"].str.replace("_delay","").str.replace("_"," ").str.title()
        fig = px.bar(reason.sort_values("Minutes"), x="Minutes", y="Cause",
                     orientation="h",
                     color="Minutes",
                     color_continuous_scale=["#C4B5E8","#9B89C4","#7B6DC4","#4A3F9F","#2D2B6B"])
        fig.update_layout(**PLOTLY_LAYOUT, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Proportion of Delay Causes")
        fig2 = px.pie(reason, names="Cause", values="Minutes", hole=0.38,
                      color_discrete_sequence=PURPLE_SEQ)
        fig2.update_layout(**PLOTLY_LAYOUT, height=340,
                           legend=dict(orientation="h", y=-0.15))
        fig2.update_traces(marker=dict(line=dict(color="#EEF0F8", width=2)))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="insight-box">Late aircraft (previous-leg cascades) and carrier-internal issues account for the largest share of delay minutes. Weather and NAS contribute meaningfully, while security delays are rare.</div>', unsafe_allow_html=True)
