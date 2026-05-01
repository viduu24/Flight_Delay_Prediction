import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import base64
import requests
from pathlib import Path
from styles import apply_theme, PLOTLY_LAYOUT, ACCENT_COLOR
import plotly.graph_objects as go

st.set_page_config(page_title="Predict a Flight", page_icon="🔮", layout="wide")
apply_theme()

# ── Constants ──────────────────────────────────────────────────────────────────
REPO      = "AhmadJabbar2502/Flight_Delay_Prediction"
BRANCH    = "main"
MODEL_REL = "Code"
LFS_MAGIC = b"version https://git-lfs"

CARRIERS = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "HA", "G4", "9E", "OH", "YX", "MQ"]
CARRIER_NAMES = {
    "AA":"American Airlines","DL":"Delta Air Lines","UA":"United Airlines",
    "WN":"Southwest Airlines","B6":"JetBlue","AS":"Alaska Airlines",
    "NK":"Spirit Airlines","F9":"Frontier Airlines","HA":"Hawaiian Airlines",
    "G4":"Allegiant Air","9E":"Endeavor Air","OH":"PSA Airlines",
    "YX":"Republic Airways","MQ":"Envoy Air",
}
STATES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina",
    "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming",
]
MAJOR_CITIES = sorted([
    "New York","Los Angeles","Chicago","Dallas","Houston","Atlanta","Miami",
    "Phoenix","Philadelphia","Denver","Boston","Seattle","Minneapolis","Detroit",
    "Las Vegas","San Francisco","Baltimore","Charlotte","Portland","Orlando",
    "Sacramento","San Diego","Tampa","Cleveland","Pittsburgh","Salt Lake City",
    "Raleigh","Indianapolis","Nashville","Austin","Kansas City","Memphis",
    "Milwaukee","Albuquerque","Tucson","Fresno","Mesa","Omaha","Honolulu",
])
SEASON_MAP = {
    12:"Winter",1:"Winter",2:"Winter",
    3:"Spring",4:"Spring",5:"Spring",
    6:"Summer",7:"Summer",8:"Summer",
    9:"Fall",10:"Fall",11:"Fall",
}

# ── Model loading ──────────────────────────────────────────────────────────────
def is_lfs_pointer(data):
    return data[:50].lstrip().startswith(LFS_MAGIC)

def fetch_bytes(url, use_api=False):
    try:
        r = requests.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        if use_api:
            payload = json.loads(r.content)
            if payload.get("encoding") == "base64":
                return base64.b64decode(payload["content"]), None
            return None, "unexpected API encoding"
        return r.content, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner="Loading model…")
def load_model(name):
    filename = name + "_model.pkl"
    log = []

    for url, use_api in [
        (f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/{MODEL_REL}/{filename}", False),
        (f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{MODEL_REL}/{filename}", False),
        (f"https://api.github.com/repos/{REPO}/contents/{MODEL_REL}/{filename}", True),
    ]:
        data, err = fetch_bytes(url, use_api)
        if err:
            log.append(f"URL {url[:40]}…: {err}"); continue
        if data and is_lfs_pointer(data):
            log.append("LFS pointer detected"); continue
        if data:
            try:
                return pickle.loads(data), url[:40] + "…"
            except Exception as e:
                log.append(f"unpickle failed: {e}")

    # Local fallback
    for candidate in [
        Path(__file__).parent.parent / "Code" / "Models" / filename,
        Path(__file__).parent.parent / "Code" / filename,
    ]:
        if candidate.exists():
            data = candidate.read_bytes()
            if not is_lfs_pointer(data):
                try:
                    return pickle.loads(data), f"local:{candidate}"
                except Exception as e:
                    log.append(f"local unpickle failed: {e}")

    return None, " | ".join(log)

bagging_model, bagging_src = load_model("bagging")
knn_model,     knn_src     = load_model("knn")

# ── Feature helpers ────────────────────────────────────────────────────────────
def le(val, vocab):
    s = sorted(set(vocab))
    return s.index(val) if val in s else 0

def make_bagging_features(month, dom, dow, carrier, city, state, dep_time, hour, precip):
    season = SEASON_MAP.get(month, "Spring")
    return pd.DataFrame([{
        "month":             month,
        "day_of_month":      dom,
        "day_of_week":       dow,
        "op_unique_carrier": le(carrier, CARRIERS),
        "origin_city":       le(city, MAJOR_CITIES),
        "origin_state":      le(state, STATES),
        "Precipitation_mm":  precip,
        "dep_time":          dep_time,
        "Season":            le(season, ["Fall","Spring","Summer","Winter"]),
        "Departure_Hour":    hour,
    }])

def make_knn_features(month, dom, dow, carrier, city, state, dep_time, hour, precip):
    season = SEASON_MAP.get(month, "Spring")
    return pd.DataFrame([{
        "month":             month,
        "day_of_month":      dom,
        "day_of_week":       dow,
        "op_unique_carrier": le(carrier, CARRIERS),
        "origin_city":       le(city, MAJOR_CITIES),
        "origin_state":      le(state, STATES),
        "Precipitation_mm":  precip,
        "dep_time":          dep_time,
        "Season":            le(season, ["Fall","Spring","Summer","Winter"]),
        "Departure_Hour":    hour,
        "hour_sin":          np.sin(2 * np.pi * hour / 24),
        "hour_cos":          np.cos(2 * np.pi * hour / 24),
        "month_sin":         np.sin(2 * np.pi * month / 12),
        "month_cos":         np.cos(2 * np.pi * month / 12),
        "dow_sin":           np.sin(2 * np.pi * dow / 7),
        "dow_cos":           np.cos(2 * np.pi * dow / 7),
    }])

def heuristic_prob(month, dow, hour, carrier, precip):
    base = 0.29
    if hour >= 17:               base += 0.10
    elif hour <= 6:              base -= 0.08
    if month in [6,7,8]:         base += 0.06
    if month in [12,1]:          base += 0.04
    if dow in [5,7]:             base += 0.04
    if precip > 5:               base += 0.08
    if precip > 15:              base += 0.07
    if carrier in ["NK","F9","G4"]: base += 0.06
    if carrier in ["AS","HA"]:   base -= 0.05
    return float(np.clip(base + np.random.normal(0, 0.015), 0.05, 0.95))

# ── Page ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔮 Predict a Flight</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#6B6A9B;'>Enter flight details to get a delay probability prediction.</p>", unsafe_allow_html=True)
st.divider()

# Model status pills
c1, c2 = st.columns(2)
with c1:
    if bagging_model:
        st.success(f"✅ Bagging model loaded")
    else:
        st.warning("⚠️ Bagging model not found — Demo mode will be used")
        with st.expander("Debug"):
            st.code(bagging_src)
with c2:
    if knn_model:
        st.success(f"✅ KNN model loaded")
    else:
        st.warning("⚠️ KNN model not found — Demo mode will be used")
        with st.expander("Debug"):
            st.code(knn_src)

st.divider()

# ── Form + Result ──────────────────────────────────────────────────────────────
col_form, col_result = st.columns([1.15, 1])

with col_form:
    model_options = []
    if bagging_model: model_options.append("Bagged Decision Trees")
    if knn_model:     model_options.append("KNN (k=20, Manhattan)")
    model_options.append("Demo Mode (Heuristic)")

    selected_model = st.selectbox("🤖 Select Model", model_options)
    st.markdown("<br>", unsafe_allow_html=True)

    # Date & Time
    st.markdown("""<div class="card"><p style='font-weight:600;color:#2D2B6B;margin:0 0 12px 0;'>🗓️ Date & Time</p>""", unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    with ca:
        month = st.selectbox("Month", list(range(1,13)),
                             format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                    "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
    with cb:
        day_of_month = st.number_input("Day", min_value=1, max_value=31, value=15)
    with cc:
        day_of_week = st.selectbox("Day of Week", list(range(1,8)),
                                   format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d-1])
    departure_hour = st.slider("Departure Hour (0–23)", 0, 23, 9)
    dep_time_int   = departure_hour * 100
    st.markdown("</div>", unsafe_allow_html=True)

    # Flight details
    st.markdown("""<div class="card"><p style='font-weight:600;color:#2D2B6B;margin:0 0 12px 0;'>✈️ Flight Details</p>""", unsafe_allow_html=True)
    cd, ce = st.columns(2)
    with cd:
        carrier = st.selectbox("Airline", CARRIERS,
                               format_func=lambda c: c + " — " + CARRIER_NAMES.get(c, c))
    with ce:
        origin_city = st.selectbox("Origin City", MAJOR_CITIES)
    origin_state = st.selectbox("Origin State", STATES)
    st.markdown("</div>", unsafe_allow_html=True)

    # Weather
    st.markdown("""<div class="card"><p style='font-weight:600;color:#2D2B6B;margin:0 0 12px 0;'>🌦️ Weather at Departure</p>""", unsafe_allow_html=True)
    precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🔮 Predict Delay Probability", use_container_width=True)

with col_result:
    season = SEASON_MAP.get(month, "Spring")
    st.markdown("#### ✈️ Flight Summary")
    st.markdown(f"""
    | Field | Value |
    |-------|-------|
    | Airline | {carrier} — {CARRIER_NAMES.get(carrier, carrier)} |
    | Origin | {origin_city}, {origin_state} |
    | Date | Month {month}, Day {day_of_month} |
    | Departure | {str(departure_hour).zfill(2)}:00 — {["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day_of_week-1]} |
    | Season | {season} |
    | Precipitation | {precipitation:.1f} mm |
    """)

    if predict_btn:
        prob_delay = None
        model_used = ""

        try:
            if selected_model == "Bagged Decision Trees" and bagging_model:
                X = make_bagging_features(month, day_of_month, day_of_week,
                                          carrier, origin_city, origin_state,
                                          dep_time_int, departure_hour, precipitation)
                prob_delay = float(bagging_model.predict_proba(X)[0][1])
                model_used = "Bagged Decision Trees"

            elif selected_model == "KNN (k=20, Manhattan)" and knn_model:
                X = make_knn_features(month, day_of_month, day_of_week,
                                      carrier, origin_city, origin_state,
                                      dep_time_int, departure_hour, precipitation)
                prob_delay = float(knn_model.predict_proba(X)[0][1])
                model_used = "KNN (k=20, Manhattan)"

            else:
                prob_delay = heuristic_prob(month, day_of_week, departure_hour, carrier, precipitation)
                model_used = "Demo Heuristic"

        except Exception as e:
            st.error(f"Prediction error: {e}")

        if prob_delay is not None:
            is_delayed = prob_delay >= 0.5

            if is_delayed:
                st.markdown(f"""
                <div class="result-card-delay">
                  <div class="result-icon">🚨</div>
                  <div class="result-label" style="color:#C62828;">Likely Delayed</div>
                  <div class="result-prob">Delay probability: {prob_delay:.1%}</div>
                  <div style="margin-top:10px; font-size:0.8rem; color:#9B89C4;">
                    Model: {model_used}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-ok">
                  <div class="result-icon">✅</div>
                  <div class="result-label" style="color:#1A237E;">Likely On Time</div>
                  <div class="result-prob">Delay probability: {prob_delay:.1%}</div>
                  <div style="margin-top:10px; font-size:0.8rem; color:#9B89C4;">
                    Model: {model_used}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Progress bar styled with probability
            st.markdown(f"**Delay Probability: {prob_delay:.1%}**")
            st.progress(prob_delay)

            # Risk gauge
            st.markdown("#### Risk Breakdown")
            risk_items = {
                "Time of Day":   min(1.0, 0.1 + (0.35 if 15 <= departure_hour <= 20 else 0.0) - (0.15 if 5 <= departure_hour <= 9 else 0.0)),
                "Season":        min(1.0, 0.1 + (0.25 if month in [6,7,8,12,1] else 0.0)),
                "Precipitation": min(1.0, precipitation / 40),
                "Airline":       min(1.0, 0.1 + (0.2 if carrier in ["NK","F9","G4"] else 0.0) - (0.1 if carrier in ["AS","HA"] else 0.0)),
                "Day of Week":   min(1.0, 0.1 + (0.15 if day_of_week in [5,7] else 0.0)),
            }
            risk_df = pd.DataFrame(list(risk_items.items()), columns=["Factor","Risk"])
            fig = go.Figure(go.Bar(
                x=risk_df["Risk"], y=risk_df["Factor"],
                orientation="h",
                marker=dict(
                    color=risk_df["Risk"],
                    colorscale=[
                        [0.0, "#C4B5E8"],
                        [0.5, "#7B6DC4"],
                        [1.0, "#2D2B6B"],
                    ],
                    cmin=0, cmax=1,
                ),
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT, height=220, showlegend=False,
                xaxis=dict(range=[0, 1], tickformat=".0%",
                           gridcolor="#C4B5E8", linecolor="#C4B5E8"),
                yaxis=dict(gridcolor="#C4B5E8", linecolor="#C4B5E8"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Note:</b> Uses only features available before departure —
    airline, origin, time, season, and precipitation.
    Overall accuracy ~72%, ROC-AUC ~0.747 (Bagged Trees).
    </div>
    """, unsafe_allow_html=True)
