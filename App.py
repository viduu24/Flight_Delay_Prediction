import streamlit as st
import pandas as pd
import requests
from io import StringIO

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GOOGLE DRIVE LINKS (FINAL)
# ─────────────────────────────────────────────
FLIGHT_URL  = "https://drive.google.com/uc?export=download&id=1x80CYMrQ_B1XjY_TvQgAweBUea2gcw6b"
WEATHER_URL = "https://drive.google.com/uc?export=download&id=1TKlnXdIsgCj5o3x7ISh6uYTp_wLFytp7"
MERGED_URL  = "https://drive.google.com/uc?export=download&id=1hgMTsjDw8uyi3MZQkrQ6TI11j3YIEPzA"

# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
def load_csv(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data(show_spinner="Loading flight dataset...")
def load_flights():
    return load_csv(FLIGHT_URL)

@st.cache_data(show_spinner="Loading weather dataset...")
def load_weather():
    return load_csv(WEATHER_URL)

@st.cache_data(show_spinner="Loading merged dataset...")
def load_merged():
    return load_csv(MERGED_URL)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
flights = load_flights()
weather = load_weather()
merged = load_merged()

# ─────────────────────────────────────────────
# UI DESIGN
# ─────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}
.hero {
    font-size: 2.8rem;
    font-weight: 700;
    color: #60a5fa;
}
.sub {
    color: #94a3b8;
}
.box {
    padding: 1rem;
    border-radius: 10px;
    background: #111827;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="hero">✈️ Flight Delay Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Predict flight delays using flight + weather data</div>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# DATA STATUS
# ─────────────────────────────────────────────
st.subheader("📊 Data Status")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Flights", "Loaded ✅" if flights is not None else "Failed ❌")

with c2:
    st.metric("Weather", "Loaded ✅" if weather is not None else "Failed ❌")

with c3:
    st.metric("Merged", "Loaded ✅" if merged is not None else "Failed ❌")

st.divider()

# ─────────────────────────────────────────────
# PREVIEW DATA
# ─────────────────────────────────────────────
if flights is not None:
    st.subheader("✈️ Flights Sample")
    st.dataframe(flights.head())

if weather is not None:
    st.subheader("🌦️ Weather Sample")
    st.dataframe(weather.head())

if merged is not None:
    st.subheader("🔗 Merged Sample")
    st.dataframe(merged.head())

# ─────────────────────────────────────────────
# NAVIGATION HELP
# ─────────────────────────────────────────────
st.divider()
st.info("""
Use the sidebar to navigate:

• ✈️ Flight EDA  
• 🌦️ Weather EDA  
• 🤖 Models  
• 🔮 Prediction  

All data is loaded from Google Drive.
""")
