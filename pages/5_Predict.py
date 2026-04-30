import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

st.set_page_config(page_title="Predict a Flight", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #0f1629; border-right: 1px solid #1e2d4a; }
    h1,h2,h3 { color: #e2e8f0 !important; }
    label { color: #94a3b8 !important; }
    .section-header {
        font-size: 1.5rem; font-weight: 600;
        background: linear-gradient(135deg, #60a5fa, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .result-card-delay {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 2px solid #ef4444; border-radius: 20px;
        padding: 2rem; text-align: center; margin-top: 1.5rem;
    }
    .result-card-ok {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 2px solid #22c55e; border-radius: 20px;
        padding: 2rem; text-align: center; margin-top: 1.5rem;
    }
    .result-icon  { font-size: 4rem; margin-bottom: 0.5rem; }
    .result-label { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.4rem; }
    .result-prob  { font-size: 1.1rem; opacity: 0.85; font-family: 'JetBrains Mono', monospace; }
    .form-section {
        background: linear-gradient(135deg, #111827, #1a2438);
        border: 1px solid #1e3a5f; border-radius: 14px;
        padding: 1.5rem; margin-bottom: 1rem;
    }
    .form-section h4 { color: #60a5fa; margin: 0 0 1rem 0; font-size: 1rem; }
    .info-note {
        background: #111827; border-left: 3px solid #60a5fa;
        border-radius: 0 8px 8px 0; padding: 0.7rem 1rem;
        color: #94a3b8; font-size: 0.85rem; margin-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #4338ca);
        color: white; border: none; border-radius: 12px;
        padding: 0.7rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%; margin-top: 0.5rem; transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }
</style>
""", unsafe_allow_html=True)

# ── GitHub raw URLs for models ────────────────────────────────────────────────
BAGGING_URL = "https://raw.githubusercontent.com/viduu24/Flight_Delay_Prediction/main/Code/Models/bagging_model.pkl"
KNN_URL     = "https://raw.githubusercontent.com/viduu24/Flight_Delay_Prediction/main/Code/Models/knn_model.pkl"

@st.cache_resource(show_spinner="Loading Bagging model…")
def load_bagging():
    try:
        r = requests.get(BAGGING_URL, timeout=30)
        r.raise_for_status()
        return pickle.loads(r.content)
    except Exception as e:
        st.warning(f"Could not load Bagging model: {e}")
        return None

@st.cache_resource(show_spinner="Loading KNN model…")
def load_knn():
    try:
        r = requests.get(KNN_URL, timeout=30)
        r.raise_for_status()
        return pickle.loads(r.content)
    except Exception as e:
        st.warning(f"Could not load KNN model: {e}")
        return None

bagging_model = load_bagging()
knn_model     = load_knn()

# ── Constants ─────────────────────────────────────────────────────────────────
CARRIERS = ["AA","DL","UA","WN","B6","AS","NK","F9","HA","G4","9E","OH","YX","MQ"]
CARRIER_NAMES = {
    "AA":"American Airlines","DL":"Delta Air Lines","UA":"United Airlines",
    "WN":"Southwest Airlines","B6":"JetBlue","AS":"Alaska Airlines",
    "NK":"Spirit Airlines","F9":"Frontier Airlines","HA":"Hawaiian Airlines",
    "G4":"Allegiant Air","9E":"Endeavor Air","OH":"PSA Airlines",
    "YX":"Republic Airways","MQ":"Envoy Air",
}
STATES = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
          "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
          "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
          "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
          "New Hampshire","New Jersey","New Mexico","New York","North Carolina",
          "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
          "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
          "Virginia","Washington","West Virginia","Wisconsin","Wyoming"]
MAJOR_CITIES = [
    "New York","Los Angeles","Chicago","Dallas","Houston","Atlanta","Miami",
    "Phoenix","Philadelphia","Denver","Boston","Seattle","Minneapolis","Detroit",
    "Las Vegas","San Francisco","Baltimore","Charlotte","Portland","Orlando",
    "Sacramento","San Diego","Tampa","Cleveland","Pittsburgh","Salt Lake City",
    "Raleigh","Indianapolis","Nashville","Austin","Kansas City","Memphis",
    "Milwaukee","Albuquerque","Tucson","Fresno","Mesa","Omaha","Honolulu",
]
SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}

def get_le_encode(val, vocab):
    vocab_sorted = sorted(set(vocab))
    return vocab_sorted.index(val) if val in vocab_sorted else -1

def build_bagging_features(month, day_of_month, day_of_week, carrier, origin_city,
                            origin_state, dep_time_int, departure_hour, precipitation):
    season = SEASON_MAP.get(month, "Spring")
    return pd.DataFrame([{
        "month": month, "day_of_month": day_of_month, "day_of_week": day_of_week,
        "op_unique_carrier": get_le_encode(carrier, CARRIERS),
        "origin_city":       get_le_encode(origin_city, MAJOR_CITIES),
        "origin_state":      get_le_encode(origin_state, STATES),
        "Precipitation_mm":  precipitation,
        "dep_time":          dep_time_int,
        "Season":            get_le_encode(season, ["Fall","Spring","Summer","Winter"]),
        "Departure_Hour":    departure_hour,
    }])

def build_knn_features(month, day_of_month, day_of_week, carrier, origin_city,
                        origin_state, dep_time_int, departure_hour, precipitation):
    season = SEASON_MAP.get(month, "Spring")
    return pd.DataFrame([{
        "month": month, "day_of_month": day_of_month, "day_of_week": day_of_week,
        "op_unique_carrier": get_le_encode(carrier, CARRIERS),
        "origin_city":       get_le_encode(origin_city, MAJOR_CITIES),
        "origin_state":      get_le_encode(origin_state, STATES),
        "Precipitation_mm":  precipitation,
        "dep_time":          dep_time_int,
        "Season":            get_le_encode(season, ["Fall","Spring","Summer","Winter"]),
        "Departure_Hour":    departure_hour,
        "hour_sin":  np.sin(2 * np.pi * departure_hour / 24),
        "hour_cos":  np.cos(2 * np.pi * departure_hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "dow_sin":   np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos":   np.cos(2 * np.pi * day_of_week / 7),
    }])

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔮 Predict a Flight</div>', unsafe_allow_html=True)
st.markdown("Enter your flight details below to get a delay probability prediction.")

if bagging_model is None and knn_model is None:
    st.warning("⚠️ Models could not be loaded from GitHub. Running in Demo Mode.")

st.divider()

col_form, col_result = st.columns([1.2, 1])

with col_form:
    model_options = []
    if bagging_model: model_options.append("Bagged Decision Trees")
    if knn_model:     model_options.append("KNN (k=20, Manhattan)")
    if not model_options: model_options = ["Demo Mode (Heuristic)"]

    selected_model = st.selectbox("🤖 Select Model", model_options)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="form-section"><h4>🗓️ Date & Time</h4>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        month = st.selectbox("Month", list(range(1, 13)),
                             format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                    "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
    with col2:
        day_of_month = st.number_input("Day", min_value=1, max_value=31, value=15)
    with col3:
        day_of_week = st.selectbox("Day of Week", list(range(1, 8)),
                                   format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d-1])
    departure_hour = st.slider("Departure Hour (0–23)", 0, 23, 9)
    dep_time_int = departure_hour * 100
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="form-section"><h4>✈️ Flight Details</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        carrier = st.selectbox("Airline", CARRIERS,
                               format_func=lambda c: f"{c} — {CARRIER_NAMES.get(c, c)}")
    with col2:
        origin_city = st.selectbox("Origin City", sorted(MAJOR_CITIES))
    origin_state = st.selectbox("Origin State", STATES)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="form-section"><h4>🌦️ Weather at Departure</h4>', unsafe_allow_html=True)
    precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🔮 Predict Delay Probability")

with col_result:
    season = SEASON_MAP.get(month, "Spring")
    st.markdown("### Flight Summary")
    st.markdown(f"""
| Field | Value |
|-------|-------|
| Airline | {carrier} — {CARRIER_NAMES.get(carrier, carrier)} |
| Origin | {origin_city}, {origin_state} |
| Date | Month {month}, Day {day_of_month} |
| Departure | {departure_hour:02d}:00 ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week-1]}) |
| Season | {season} |
| Precipitation | {precipitation} mm |
    """)

    if predict_btn:
        prob_delay = None
        model_used = ""
        try:
            if selected_model == "Bagged Decision Trees" and bagging_model:
                X = build_bagging_features(month, day_of_month, day_of_week, carrier,
                                           origin_city, origin_state, dep_time_int,
                                           departure_hour, precipitation)
                prob_delay = bagging_model.predict_proba(X)[0][1]
                model_used = "Bagged Decision Trees"

            elif selected_model == "KNN (k=20, Manhattan)" and knn_model:
                X = build_knn_features(month, day_of_month, day_of_week, carrier,
                                       origin_city, origin_state, dep_time_int,
                                       departure_hour, precipitation)
                prob_delay = knn_model.predict_proba(X)[0][1]
                model_used = "KNN (k=20, Manhattan)"

            else:
                base = 0.29
                if departure_hour >= 17:   base += 0.10
                elif departure_hour <= 6:  base -= 0.08
                if month in [6, 7, 8]:    base += 0.06
                if month in [12, 1]:      base += 0.04
                if day_of_week in [5, 7]: base += 0.04
                if precipitation > 5:     base += 0.08
                if precipitation > 15:    base += 0.07
                if carrier in ["NK","F9","G4"]: base += 0.06
                if carrier in ["AS","HA"]:      base -= 0.05
                prob_delay = float(np.clip(base + np.random.normal(0, 0.02), 0.05, 0.95))
                model_used = "Demo Heuristic"

        except Exception as e:
            st.error(f"Prediction error: {e}")
            prob_delay = None

        if prob_delay is not None:
            if prob_delay >= 0.5:
                st.markdown(f"""
                <div class="result-card-delay">
                    <div class="result-icon">🚨</div>
                    <div class="result-label" style="color:#fca5a5;">Likely Delayed</div>
                    <div class="result-prob">Delay Probability: {prob_delay:.1%}</div>
                    <div style="margin-top:0.8rem;color:#fca5a5;font-size:0.85rem;">Model: {model_used}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-ok">
                    <div class="result-icon">✅</div>
                    <div class="result-label" style="color:#86efac;">Likely On Time</div>
                    <div class="result-prob">Delay Probability: {prob_delay:.1%}</div>
                    <div style="margin-top:0.8rem;color:#86efac;font-size:0.85rem;">Model: {model_used}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(prob_delay), text=f"Delay probability: {prob_delay:.1%}")

    st.markdown("""
    <div class="info-note">
    <strong>Note:</strong> Uses features available before departure only — airline, origin, time, season, and precipitation.
    Overall accuracy ~72%, ROC-AUC ~0.747.
    </div>
    """, unsafe_allow_html=True)
