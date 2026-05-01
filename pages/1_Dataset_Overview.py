import streamlit as st
import pandas as pd
from styles import apply_theme

# MUST be first
st.set_page_config(page_title="Dataset Overview", page_icon="📦", layout="wide")

# Apply global theme
apply_theme()

# ── Extra Page-Specific Styling ─────────────────────────────
st.markdown("""
<style>
.info-card {
    background: #FFFFFF;
    border: 1.5px solid #C4B5E8;
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

.info-card h4 {
    margin-bottom: 0.5rem;
    color: #2D2B6B;
}

.badge {
    background: #EEE8FA;
    color: #2D2B6B;
    border: 1px solid #C4B5E8;
    border-radius: 6px;
    padding: 2px 6px;
    font-size: 0.75rem;
    font-family: monospace;
}

.pipeline-step {
    background: #FFFFFF;
    border-left: 4px solid #9B89C4;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    border-radius: 8px;
}

.step-title {
    font-weight: 600;
    color: #2D2B6B;
}

.step-desc {
    font-size: 0.9rem;
    color: #6B6A9B;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────
st.markdown('<div class="section-header">📦 Dataset Overview</div>', unsafe_allow_html=True)
st.markdown("This page describes the datasets, engineered features, and final merged dataset used for modelling.")

st.divider()

# ── Dataset Cards ──────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>✈️ Flight Dataset</h4>
        <p><strong>Source:</strong> U.S. Bureau of Transportation Statistics (BTS)</p>
        <p><strong>Period:</strong> Full-year 2024</p>
        <p><strong>Rows:</strong> ~5.7 million domestic flights</p>
        <p><strong>File:</strong> <span class="badge">flight_data_sample.csv</span></p>
        <br/>
        <p><strong>Key columns:</strong></p>
        <ul>
            <li><span class="badge">fl_date</span> Flight date</li>
            <li><span class="badge">op_unique_carrier</span> Airline code</li>
            <li><span class="badge">origin_city</span> / <span class="badge">dest_city</span></li>
            <li><span class="badge">dep_time</span> / <span class="badge">arr_delay</span></li>
            <li><span class="badge">carrier_delay</span> <span class="badge">weather_delay</span> <span class="badge">nas_delay</span></li>
            <li><span class="badge">late_aircraft_delay</span> <span class="badge">security_delay</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🌦️ Weather Dataset</h4>
        <p><strong>Source:</strong> Hourly weather sensor readings by city</p>
        <p><strong>Period:</strong> Full-year 2024</p>
        <p><strong>Granularity:</strong> Hourly, per city</p>
        <p><strong>File:</strong> <span class="badge">weather_data.csv</span></p>
        <br/>
        <p><strong>Key columns:</strong></p>
        <ul>
            <li><span class="badge">Date_Time</span> Timestamp</li>
            <li><span class="badge">Location</span> City name</li>
            <li><span class="badge">Temperature_C</span></li>
            <li><span class="badge">Humidity_pct</span></li>
            <li><span class="badge">Precipitation_mm</span></li>
            <li><span class="badge">Wind_Speed_kmh</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Engineered Features ────────────────────────────────
st.markdown("### 🔧 Engineered Features")

feat_df = pd.DataFrame({
    "Feature": ["IS_Delay", "delay_in_minutes", "Season", "Departure_Hour", "Month_Name", "Weekday",
                "origin_city (cleaned)", "Weather_Data_Present"],
    "Source": ["Flights", "Flights", "Flights", "Flights", "Weather", "Weather", "Flights", "Merged"],
    "Description": [
        "Binary target — delay ≥ 15 min",
        "Total delay (arrival + departure)",
        "Season derived from month",
        "Hour extracted from departure time",
        "Readable month name",
        "Day of week",
        "Cleaned city name",
        "Whether weather data matched",
    ],
})

st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.divider()

# ── Pipeline ───────────────────────────────────────────
st.markdown("### 🔀 Data Pipeline")

steps = [
    ("1. Load Raw Data", "Flights + Weather datasets loaded"),
    ("2. Clean Flights", "Created delay features, cleaned city names"),
    ("3. Clean Weather", "Extracted time features"),
    ("4. Merge", "Joined using nearest timestamp (±3 hours)"),
    ("5. Final Output", "Saved merged dataset"),
]

for title, desc in steps:
    st.markdown(f"""
    <div class="pipeline-step">
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Schema ─────────────────────────────────────────────
st.markdown("### 📊 Final Dataset Schema")

schema = pd.DataFrame({
    "Column": ["fl_date","op_unique_carrier","Departure_Hour","IS_Delay","delay_in_minutes",
               "carrier_delay","weather_delay","Temperature_C","Humidity_pct"],
    "Type": ["date","str","int","int","float","float","float","float","float"],
})

st.dataframe(schema, use_container_width=True, hide_index=True)
