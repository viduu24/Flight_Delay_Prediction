import streamlit as st
import pandas as pd
import requests
from io import BytesIO

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
# GOOGLE DRIVE LINKS
# ─────────────────────────────────────────────
FLIGHT_ID   = "1x80CYMrQ_B1XjY_TvQgAweBUea2gcw6b"
WEATHER_ID  = "1TKlnXdIsgCj5o3x7ISh6uYTp_wLFytp7"
MERGED_ID   = "1hgMTsjDw8uyi3MZQkrQ6TI11j3YIEPzA"

# ─────────────────────────────────────────────
# DATA LOADER  (handles large-file warning cookie)
# ─────────────────────────────────────────────
def download_gdrive(file_id: str) -> pd.DataFrame | None:
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    # Handle Google's large-file confirmation page
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={"id": file_id, "confirm": value}, stream=True)
            break
    try:
        response.raise_for_status()
        return pd.read_csv(BytesIO(response.content), low_memory=False)
    except Exception as e:
        st.error(f"Error loading file {file_id}: {e}")
        return None

@st.cache_data(show_spinner="Loading flight dataset...")
def load_flights():
    return download_gdrive(FLIGHT_ID)

@st.cache_data(show_spinner="Loading weather dataset...")
def load_weather():
    return download_gdrive(WEATHER_ID)

@st.cache_data(show_spinner="Loading merged dataset...")
def load_merged():
    return download_gdrive(MERGED_ID)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
flights = load_flights()
weather = load_weather()
merged  = load_merged()

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0a0e1a; color: #e2e8f0; }
.hero  { font-size: 2.8rem; font-weight: 700; color: #60a5fa; }
.sub   { color: #94a3b8; }
.box   { padding: 1rem; border-radius: 10px; background: #111827; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown('<div class="hero">✈️ Flight Delay Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Predict flight delays using flight + weather data</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# DATA STATUS
# ─────────────────────────────────────────────
st.subheader("📊 Data Status")
c1, c2, c3 = st.columns(3)
c1.metric("Flights", "Loaded ✅" if flights is not None else "Failed ❌")
c2.metric("Weather", "Loaded ✅" if weather is not None else "Failed ❌")
c3.metric("Merged",  "Loaded ✅" if merged  is not None else "Failed ❌")
st.divider()

# ─────────────────────────────────────────────
# PREVIEWS
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
# NAV HELP
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
