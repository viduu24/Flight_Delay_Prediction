import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset Overview", page_icon="📦", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #0f1629; border-right: 1px solid #1e2d4a; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .section-header {
        font-size: 1.5rem; font-weight: 600;
        background: linear-gradient(135deg, #60a5fa, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .info-card {
        background: linear-gradient(135deg, #111827, #1a2438);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .info-card h4 { color: #60a5fa; margin: 0 0 0.5rem 0; }
    .info-card p, .info-card li { color: #94a3b8; font-size: 0.9rem; margin: 0.25rem 0; }
    .badge {
        display: inline-block; background: #1e3a5f; color: #60a5fa;
        border-radius: 6px; padding: 0.15rem 0.5rem;
        font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;
        margin: 0.1rem;
    }
    .badge-green { background: #052e16; color: #4ade80; }
    .badge-yellow { background: #1c1917; color: #fbbf24; }
    .pipeline-step {
        background: #111827; border-left: 3px solid #60a5fa;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
    }
    .pipeline-step .step-title { color: #60a5fa; font-weight: 600; font-size: 0.95rem; }
    .pipeline-step .step-desc { color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">📦 Dataset Overview</div>', unsafe_allow_html=True)
st.markdown("This page describes the two source datasets, the engineered features, and how they were joined into the final modelling dataset.", unsafe_allow_html=False)

st.divider()

# --- Dataset Cards ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>✈️ Flight Dataset</h4>
        <p><strong style="color:#e2e8f0;">Source:</strong> U.S. Bureau of Transportation Statistics (BTS)</p>
        <p><strong style="color:#e2e8f0;">Period:</strong> Full-year 2024</p>
        <p><strong style="color:#e2e8f0;">Rows:</strong> ~5.7 million domestic flights</p>
        <p><strong style="color:#e2e8f0;">File:</strong> <span class="badge">flight_data_sample.csv</span></p>
        <br/>
        <p><strong style="color:#e2e8f0;">Key columns:</strong></p>
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
        <p><strong style="color:#e2e8f0;">Source:</strong> Hourly weather sensor readings by city</p>
        <p><strong style="color:#e2e8f0;">Period:</strong> Full-year 2024</p>
        <p><strong style="color:#e2e8f0;">Granularity:</strong> Hourly, per city</p>
        <p><strong style="color:#e2e8f0;">File:</strong> <span class="badge">weather_data.csv</span></p>
        <br/>
        <p><strong style="color:#e2e8f0;">Key columns:</strong></p>
        <ul>
            <li><span class="badge">Date_Time</span> Hourly timestamp</li>
            <li><span class="badge">Location</span> City name (matched to origin)</li>
            <li><span class="badge">Temperature_C</span></li>
            <li><span class="badge">Humidity_pct</span></li>
            <li><span class="badge">Precipitation_mm</span></li>
            <li><span class="badge">Wind_Speed_kmh</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Engineered Features ---
st.markdown("### 🔧 Engineered Features")
st.markdown("The following features were derived during the cleaning phase and used in modelling.")

feat_df = pd.DataFrame({
    "Feature": ["IS_Delay", "delay_in_minutes", "Season", "Departure_Hour", "Month_Name", "Weekday",
                 "origin_city (cleaned)", "Weather_Data_Present"],
    "Source": ["Flights", "Flights", "Flights", "Flights", "Weather", "Weather", "Flights", "Merged"],
    "Description": [
        "Binary target — 1 if arr_delay + dep_delay ≥ 15 min",
        "Total delay = arr_delay + dep_delay",
        "Winter / Spring / Summer / Fall from month number",
        "Hour extracted from crs_dep_time (0–23)",
        "Human-readable month (January … December)",
        "Day-of-week name (Monday … Sunday)",
        "City name trimmed before the comma",
        "Yes/No — whether hourly weather was matched within ±3 hours",
    ],
})

st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.divider()

# --- Data Pipeline ---
st.markdown("### 🔀 Data Pipeline")

steps = [
    ("1. Load Raw Data", "flight_data_sample.csv loaded into SQLite via SQLAlchemy. weather_data.csv loaded into pandas."),
    ("2. Clean Flights", "Added delay_in_minutes, IS_Delay flag, Season, Departure_Hour. Cleaned city names (strip state suffix). Zeroed delay-reason columns for non-delayed flights."),
    ("3. Clean Weather", "Extracted Hour, Month_Number, Month_Name, Day_of_Month, Weekday from Date_Time using SQLite strftime."),
    ("4. Merge (nearest-time join)", "For each origin city, merge_asof joins flights to the nearest hourly weather reading within a ±3-hour tolerance. Unmatched flights get Weather_Data_Present = 'No'."),
    ("5. Final Output", "Deduplicated on (fl_date, carrier, flight_num). Saved to merged_flights.csv."),
]

for title, desc in steps:
    st.markdown(f"""
    <div class="pipeline-step">
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Schema Summary ---
st.markdown("### 📊 Final Merged Dataset — Column Summary")

schema = pd.DataFrame({
    "Column Group": [
        "Flight identifiers", "Flight identifiers", "Flight identifiers",
        "Time features", "Time features", "Time features", "Time features",
        "Delay targets", "Delay targets",
        "Delay reasons", "Delay reasons", "Delay reasons", "Delay reasons", "Delay reasons",
        "Weather sensors", "Weather sensors", "Weather sensors", "Weather sensors",
        "Metadata",
    ],
    "Column": [
        "fl_date", "op_unique_carrier", "op_carrier_fl_num",
        "month", "day_of_week", "Departure_Hour", "Season",
        "IS_Delay", "delay_in_minutes",
        "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
        "Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh",
        "Weather_Data_Present",
    ],
    "Type": [
        "date", "str", "int",
        "int", "int", "int", "str",
        "int (0/1)", "float",
        "float", "float", "float", "float", "float",
        "float", "float", "float", "float",
        "str (Yes/No)",
    ],
})

st.dataframe(schema, use_container_width=True, hide_index=True)
