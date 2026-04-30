import streamlit as st
from utils import load_flights, load_weather, load_merged

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Load all datasets once here — cached results are reused by every other page
flights = load_flights()
weather = load_weather()
merged  = load_merged()

st.subheader("📊 Data Status")
c1, c2, c3 = st.columns(3)
c1.metric("Flights", "Loaded ✅" if flights is not None else "Failed ❌")
c2.metric("Weather", "Loaded ✅" if weather is not None else "Failed ❌")
c3.metric("Merged",  "Loaded ✅" if merged  is not None else "Failed ❌")
st.divider()

if flights is not None:
    st.subheader("✈️ Flights Sample")
    st.dataframe(flights.head(), width="stretch")

if weather is not None:
    st.subheader("🌦️ Weather Sample")
    st.dataframe(weather.head(), width="stretch")

if merged is not None:
    st.subheader("🔗 Merged Sample")
    st.dataframe(merged.head(), width="stretch")

st.divider()
st.info("""
Use the sidebar to navigate:

• 📦 Dataset Overview  
• ✈️ Flight EDA  
• 🌦️ Weather EDA  
• 🤖 Models  
• 🔮 Prediction  

All data is loaded from Google Drive.
""")
