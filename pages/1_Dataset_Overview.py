import streamlit as st
import pandas as pd
from styles import apply_theme
from utils import load_merged

st.set_page_config(page_title="Dataset Overview", page_icon="📦", layout="wide")
apply_theme()

df = load_merged()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📦 Dataset Overview</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#6B6A9B;'>Source data, engineered features, merge pipeline, and final schema.</p>", unsafe_allow_html=True)
st.divider()

# ── Dataset cards ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h4 style='color:#2D2B6B;'>✈️ Flight Dataset</h4>
        <p style='color:#6B6A9B; font-size:0.88rem;'>
            <b>Source:</b> U.S. Bureau of Transportation Statistics (BTS)<br>
            <b>Period:</b> Full-year 2024 &nbsp;·&nbsp; <b>Rows:</b> ~5.7 M domestic flights<br>
            <b>File:</b> <span class="badge">flight_data_sample.csv</span>
        </p>
        <p style='color:#2D2B6B; font-weight:600; margin:10px 0 4px;'>Key columns</p>
        <div>
            <span class="badge">fl_date</span>
            <span class="badge">op_unique_carrier</span>
            <span class="badge">origin_city</span>
            <span class="badge">dest_city</span>
            <span class="badge">dep_time</span>
            <span class="badge">arr_delay</span>
            <span class="badge">carrier_delay</span>
            <span class="badge">weather_delay</span>
            <span class="badge">nas_delay</span>
            <span class="badge">late_aircraft_delay</span>
            <span class="badge">security_delay</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h4 style='color:#2D2B6B;'>🌦️ Weather Dataset</h4>
        <p style='color:#6B6A9B; font-size:0.88rem;'>
            <b>Source:</b> Hourly weather sensor readings by city<br>
            <b>Period:</b> Full-year 2024 &nbsp;·&nbsp; <b>Granularity:</b> Hourly, per city<br>
            <b>File:</b> <span class="badge">weather_data.csv</span>
        </p>
        <p style='color:#2D2B6B; font-weight:600; margin:10px 0 4px;'>Key columns</p>
        <div>
            <span class="badge">Date_Time</span>
            <span class="badge">Location</span>
            <span class="badge">Temperature_C</span>
            <span class="badge">Humidity_pct</span>
            <span class="badge">Precipitation_mm</span>
            <span class="badge">Wind_Speed_kmh</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Engineered features ────────────────────────────────────────────────────────
st.markdown("### 🔧 Engineered Features")

feat_df = pd.DataFrame({
    "Feature":     ["IS_Delay", "delay_in_minutes", "Season", "Departure_Hour",
                    "Month_Name", "Weekday", "origin_city (cleaned)", "Weather_Data_Present"],
    "Source":      ["Flights", "Flights", "Flights", "Flights",
                    "Weather", "Weather", "Flights", "Merged"],
    "Type":        ["Binary target", "Continuous", "Categorical", "Integer",
                    "Categorical", "Integer (1–7)", "String", "Boolean flag"],
    "Description": [
        "1 if arrival delay ≥ 15 min, else 0",
        "Total delay in minutes (arrival + departure combined)",
        "Derived from month: Winter / Spring / Summer / Fall",
        "Hour of scheduled departure (0–23)",
        "Readable month name for display",
        "Day of week (1 = Mon … 7 = Sun)",
        "City name cleaned and matched to weather lookup",
        "Whether a weather sensor match was found within ±3 h",
    ],
})

st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.divider()

# ── Pipeline ───────────────────────────────────────────────────────────────────
st.markdown("### 🔀 Data Pipeline")

steps = [
    ("1 · Load Raw Data",   "Read flight CSV from Flight_Data/ and weather CSV from Weather_Data/"),
    ("2 · Clean Flights",   "Parse dates, extract departure hour, compute IS_Delay & delay_in_minutes, clean city names"),
    ("3 · Clean Weather",   "Parse Date_Time, extract hour/month/season, standardise city names to match flights"),
    ("4 · Merge",           "Left-join flights to weather on (origin_city, date, hour) — nearest match within ±3 hours"),
    ("5 · Feature Engineer","Encode categoricals (label encoding + cyclical sin/cos for KNN), drop leakage columns"),
    ("6 · Final Output",    "Saved to Merged_Data/ as a single parquet/CSV; PKL models saved to Code/Models/"),
]

for title, desc in steps:
    st.markdown(f"""
    <div class="pipeline-step">
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Schema ─────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Final Dataset Schema")

schema = pd.DataFrame({
    "Column":      ["fl_date", "op_unique_carrier", "origin_city", "origin_state",
                    "departure_hour", "season", "IS_Delay", "delay_in_minutes",
                    "carrier_delay", "weather_delay", "nas_delay",
                    "temperature_c", "humidity_pct", "precipitation_mm", "wind_speed_kmh"],
    "Type":        ["date", "str", "str", "str",
                    "int", "str", "int (0/1)", "float",
                    "float", "float", "float",
                    "float", "float", "float", "float"],
    "Used in model": ["No", "Yes (encoded)", "Yes (encoded)", "Yes (encoded)",
                      "Yes", "Yes (encoded)", "Target", "No (target variant)",
                      "No — post-departure", "No — post-departure", "No — post-departure",
                      "No", "No", "Yes", "No"],
})

st.dataframe(schema, use_container_width=True, hide_index=True)

st.markdown("""
<div class="insight-box">
<b>No target leakage:</b> all post-departure fields (actual delay minutes, delay sub-reason columns)
are excluded from model inputs. Only features known <em>before</em> the flight departs are used.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sample rows ────────────────────────────────────────────────────────────────
st.markdown("### 🔎 Sample Records")
display_cols = ["fl_date", "op_unique_carrier", "origin_city", "departure_hour",
                "season", "precipitation_mm", "temperature_c", "is_delay", "delay_in_minutes"]
st.dataframe(df[display_cols].head(12), use_container_width=True, hide_index=True)
