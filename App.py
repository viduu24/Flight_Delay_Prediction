import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FLIGHT_ID  = "1x80CYMrQ_B1XjY_TvQgAweBUea2gcw6b"
WEATHER_ID = "1TKlnXdIsgCj5o3x7ISh6uYTp_wLFytp7"
MERGED_ID  = "1hgMTsjDw8uyi3MZQkrQ6TI11j3YIEPzA"

# ── Robust Google Drive downloader ───────────────────────────────────────────
def download_gdrive_csv(file_id: str) -> pd.DataFrame | None:
    session  = requests.Session()
    url      = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(url, stream=True)

    # If Google shows the virus-scan HTML warning, parse the real download action
    if "text/html" in response.headers.get("Content-Type", ""):
        soup = BeautifulSoup(response.text, "html.parser")
        form = soup.find("form", {"id": "download-form"})
        if form:
            action = form.get("action")
            params = {inp.get("name"): inp.get("value")
                      for inp in form.find_all("input") if inp.get("name")}
            response = session.get(action, params=params, stream=True)
        else:
            response = session.get(
                "https://drive.usercontent.google.com/download",
                params={"id": file_id, "export": "download", "confirm": "t"},
                stream=True,
            )

    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content), low_memory=False)

@st.cache_data(show_spinner="Loading flight dataset...")
def load_flights():
    try: return download_gdrive_csv(FLIGHT_ID)
    except Exception as e: st.error(f"Flights: {e}"); return None

@st.cache_data(show_spinner="Loading weather dataset...")
def load_weather():
    try: return download_gdrive_csv(WEATHER_ID)
    except Exception as e: st.error(f"Weather: {e}"); return None

@st.cache_data(show_spinner="Loading merged dataset...")
def load_merged():
    try: return download_gdrive_csv(MERGED_ID)
    except Exception as e: st.error(f"Merged: {e}"); return None

flights = load_flights()
weather = load_weather()
merged  = load_merged()

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0a0e1a; color: #e2e8f0; }
.hero  { font-size: 2.8rem; font-weight: 700; color: #60a5fa; }
.sub   { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero">✈️ Flight Delay Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Predict flight delays using flight + weather data</div>', unsafe_allow_html=True)
st.divider()

st.subheader("📊 Data Status")
c1, c2, c3 = st.columns(3)
c1.metric("Flights", "Loaded ✅" if flights is not None else "Failed ❌")
c2.metric("Weather", "Loaded ✅" if weather is not None else "Failed ❌")
c3.metric("Merged",  "Loaded ✅" if merged  is not None else "Failed ❌")
st.divider()

if flights is not None:
    st.subheader("✈️ Flights Sample")
    st.dataframe(flights.head())

if weather is not None:
    st.subheader("🌦️ Weather Sample")
    st.dataframe(weather.head())

if merged is not None:
    st.subheader("🔗 Merged Sample")
    st.dataframe(merged.head())

st.divider()
st.info("""
Use the sidebar to navigate:

• ✈️ Flight EDA  
• 🌦️ Weather EDA  
• 🤖 Models  
• 🔮 Prediction  

All data is loaded from Google Drive.
""")
