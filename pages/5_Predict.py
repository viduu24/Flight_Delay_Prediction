import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
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
    .lfs-note {
        background: #1c1917; border: 1px solid #78350f;
        border-radius: 8px; padding: 0.7rem 1rem;
        color: #fbbf24; font-size: 0.82rem; margin-bottom: 1rem;
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

# ── Model loading — handles both plain GitHub and Git LFS ─────────────────────
#
# Git LFS pointer problem:
#   raw.githubusercontent.com  →  returns the tiny LFS pointer text file (~130 bytes)
#                                  which starts with \x0e (version https://git-lfs...)
#   media.githubusercontent.com →  serves the ACTUAL binary via LFS CDN  ✅
#
# We try three strategies in order:
#   1. media.githubusercontent.com  (LFS CDN — works when files are stored in LFS)
#   2. raw.githubusercontent.com    (works when files are committed normally, <100 MB)
#   3. Local file path              (works when running locally from the cloned repo)

REPO      = "viduu24/Flight_Delay_Prediction"
BRANCH    = "main"
MODEL_REL = "Code/Models"

# Every URL pattern that could serve the actual binary
def _model_urls(filename: str) -> list[tuple[str, str]]:
    """Returns list of (label, url) to try in order."""
    return [
        # 1. GitHub LFS media CDN — serves real binary for LFS-tracked files
        ("LFS CDN",
         f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/{MODEL_REL}/{filename}"),
        # 2. Raw URL — works if committed normally (not via LFS)
        ("Raw URL",
         f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{MODEL_REL}/{filename}"),
        # 3. GitHub API contents endpoint — returns base64, works for files <100MB
        ("API base64",
         f"https://api.github.com/repos/{REPO}/contents/{MODEL_REL}/{filename}"),
    ]

LFS_POINTER_MAGIC = b"version https://git-lfs"

def _is_lfs_pointer(data: bytes) -> bool:
    return data[:50].lstrip().startswith(LFS_POINTER_MAGIC)

def _pkl_from_bytes(data: bytes):
    return pickle.loads(data)


@st.cache_resource(show_spinner="Loading model…")
def _fetch_model(name: str):
    """
    Try every URL strategy in order. Never silently swallow errors —
    collect them all and surface in the status message so it is debuggable.
    Returns (model_object, source_label) or (None, error_summary).
    """
    from pathlib import Path
    import base64

    filename = f"{name}_model.pkl"
    attempts = []   # list of (label, outcome_string)

    for label, url in _model_urls(filename):
        try:
            headers = {"Accept": "application/vnd.github.v3.raw"} if "api.github" in url else {}
            r = requests.get(url, timeout=60, allow_redirects=True, headers=headers)
            r.raise_for_status()

            # GitHub API returns JSON with base64-encoded content
            if "api.github" in url:
                import json as _json
                payload = _json.loads(r.content)
                if payload.get("encoding") == "base64":
                    data = base64.b64decode(payload["content"])
                else:
                    attempts.append((label, "unexpected API encoding"))
                    continue
            else:
                data = r.content

            if _is_lfs_pointer(data):
                attempts.append((label, f"got LFS pointer ({len(data)} bytes) — not the binary"))
                continue

            model = _pkl_from_bytes(data)
            return model, f"{label} ✅"

        except requests.exceptions.HTTPError as e:
            attempts.append((label, f"HTTP {e.response.status_code}"))
        except requests.exceptions.RequestException as e:
            attempts.append((label, f"network error: {e}"))
        except Exception as e:
            attempts.append((label, f"failed: {e}"))

    # ── Local file fallback ───────────────────────────────────────────────────
    here  = Path(__file__).resolve().parent.parent
    local = here / "Code" / "Models" / filename
    if local.exists():
        data = local.read_bytes()
        if _is_lfs_pointer(data):
            attempts.append(("local file", "is a Git LFS pointer (not the binary)"))
        else:
            try:
                model = _pkl_from_bytes(data)
                return model, "local file ✅"
            except Exception as e:
                attempts.append(("local file", f"failed: {e}"))
    else:
        attempts.append(("local file", "not found on disk"))

    summary = " | ".join(f"{lbl}: {outcome}" for lbl, outcome in attempts)
    return None, summary
        try:
            model = pickle.loads(raw)
            return model, f"Local file: `{local.relative_to(here)}`"
        except Exception as e:
            return None, f"Local file found but failed to load: {e}"

    return None, "Not found via GitHub (LFS or raw) or local path"


# Load both models — errors shown inline, not as exceptions
bagging_model, bagging_src = _fetch_model("bagging")
knn_model,     knn_src     = _fetch_model("knn")

# ── Constants ─────────────────────────────────────────────────────────────────
CARRIERS = ["AA","DL","UA","WN","B6","AS","NK","F9","HA","G4","9E","OH","YX","MQ"]
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

def _le(val, vocab):
    s = sorted(set(vocab))
    return s.index(val) if val in s else -1

def build_features_bagging(month, dom, dow, carrier, city, state, dep_time, hour, precip):
    season = SEASON_MAP.get(month, "Spring")
    return pd.DataFrame([{
        "month":              month,
        "day_of_month":       dom,
        "day_of_week":        dow,
        "op_unique_carrier":  _le(carrier, CARRIERS),
        "origin_city":        _le(city, MAJOR_CITIES),
        "origin_state":       _le(state, STATES),
        "Precipitation_mm":   precip,
        "dep_time":           dep_time,
        "Season":             _le(season, ["Fall","Spring","Summer","Winter"]),
        "Departure_Hour":     hour,
    }])

def build_features_knn(month, dom, dow, carrier, city, state, dep_time, hour, precip):
    season = SEASON_MAP.get(month, "Spring")
    return pd.DataFrame([{
        "month":              month,
        "day_of_month":       dom,
        "day_of_week":        dow,
        "op_unique_carrier":  _le(carrier, CARRIERS),
        "origin_city":        _le(city, MAJOR_CITIES),
        "origin_state":       _le(state, STATES),
        "Precipitation_mm":   precip,
        "dep_time":           dep_time,
        "Season":             _le(season, ["Fall","Spring","Summer","Winter"]),
        "Departure_Hour":     hour,
        "hour_sin":           np.sin(2 * np.pi * hour / 24),
        "hour_cos":           np.cos(2 * np.pi * hour / 24),
        "month_sin":          np.sin(2 * np.pi * month / 12),
        "month_cos":          np.cos(2 * np.pi * month / 12),
        "dow_sin":            np.sin(2 * np.pi * dow / 7),
        "dow_cos":            np.cos(2 * np.pi * dow / 7),
    }])

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔮 Predict a Flight</div>', unsafe_allow_html=True)
st.markdown("Enter your flight details below to get a delay probability prediction.")

# Model status badges
col_s1, col_s2 = st.columns(2)
with col_s1:
    if bagging_model is not None:
        st.success(f"✅ Bagging model loaded — {bagging_src}")
    else:
        st.error("❌ Bagging model failed to load")
        with st.expander("🔍 Debug — Bagging load attempts"):
            st.code(bagging_src)
with col_s2:
    if knn_model is not None:
        st.success(f"✅ KNN model loaded — {knn_src}")
    else:
        st.error("❌ KNN model failed to load")
        with st.expander("🔍 Debug — KNN load attempts"):
            st.code(knn_src)

if bagging_model is None and knn_model is None:
    st.markdown("""
    <div class="lfs-note">
    ⚠️ <strong>Git LFS note:</strong> Your <code>.pkl</code> files are stored in Git LFS.
    Streamlit Cloud does not download LFS files automatically, so the app fetches them from the
    GitHub LFS CDN (<code>media.githubusercontent.com</code>). If this fails and the repo is
    <strong>private</strong>, the easiest fix is to recommit the <code>.pkl</code> files without
    LFS tracking — remove them from <code>.gitattributes</code>, then <code>git add</code> and
    recommit. They are small enough to store normally.
    </div>
    """, unsafe_allow_html=True)

st.divider()

col_form, col_result = st.columns([1.2, 1])

with col_form:
    model_options = []
    if bagging_model is not None: model_options.append("Bagged Decision Trees")
    if knn_model is not None:     model_options.append("KNN (k=20, Manhattan)")
    if not model_options: model_options = ["Demo Mode (Heuristic)"]

    selected_model = st.selectbox("🤖 Select Model", model_options)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="form-section"><h4>🗓️ Date & Time</h4>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        month = st.selectbox("Month", list(range(1,13)),
            format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
    with c2:
        day_of_month = st.number_input("Day", min_value=1, max_value=31, value=15)
    with c3:
        day_of_week = st.selectbox("Day of Week", list(range(1,8)),
            format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d-1])
    departure_hour = st.slider("Departure Hour (0–23)", 0, 23, 9)
    dep_time_int   = departure_hour * 100
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="form-section"><h4>✈️ Flight Details</h4>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        carrier = st.selectbox("Airline", CARRIERS,
            format_func=lambda c: f"{c} — {CARRIER_NAMES.get(c, c)}")
    with c2:
        origin_city = st.selectbox("Origin City", MAJOR_CITIES)
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
| Departure | {departure_hour:02d}:00 ({["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day_of_week-1]}) |
| Season | {season} |
| Precipitation | {precipitation} mm |
""")

    if predict_btn:
        prob_delay = None
        model_used = ""
        try:
            if selected_model == "Bagged Decision Trees" and bagging_model is not None:
                X = build_features_bagging(month, day_of_month, day_of_week, carrier,
                                           origin_city, origin_state, dep_time_int,
                                           departure_hour, precipitation)
                prob_delay = float(bagging_model.predict_proba(X)[0][1])
                model_used = "Bagged Decision Trees"

            elif selected_model == "KNN (k=20, Manhattan)" and knn_model is not None:
                X = build_features_knn(month, day_of_month, day_of_week, carrier,
                                       origin_city, origin_state, dep_time_int,
                                       departure_hour, precipitation)
                prob_delay = float(knn_model.predict_proba(X)[0][1])
                model_used = "KNN (k=20, Manhattan)"

            else:  # Demo heuristic
                base = 0.29
                if departure_hour >= 17:    base += 0.10
                elif departure_hour <= 6:   base -= 0.08
                if month in [6,7,8]:        base += 0.06
                if month in [12,1]:         base += 0.04
                if day_of_week in [5,7]:    base += 0.04
                if precipitation > 5:       base += 0.08
                if precipitation > 15:      base += 0.07
                if carrier in ["NK","F9","G4"]: base += 0.06
                if carrier in ["AS","HA"]:      base -= 0.05
                prob_delay = float(np.clip(base + np.random.normal(0, 0.02), 0.05, 0.95))
                model_used = "Demo Heuristic"

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.caption("This usually means the model's training feature schema differs from what's being passed. "
                       "Check that the same label encoders and feature columns were used at training time.")
            prob_delay = None

        if prob_delay is not None:
            is_delayed = prob_delay >= 0.5
            card_cls   = "result-card-delay" if is_delayed else "result-card-ok"
            icon       = "🚨" if is_delayed else "✅"
            label      = "Likely Delayed" if is_delayed else "Likely On Time"
            txt_color  = "#fca5a5" if is_delayed else "#86efac"

            st.markdown(f"""
            <div class="{card_cls}">
                <div class="result-icon">{icon}</div>
                <div class="result-label" style="color:{txt_color};">{label}</div>
                <div class="result-prob">Delay Probability: {prob_delay:.1%}</div>
                <div style="margin-top:0.8rem;color:{txt_color};font-size:0.85rem;">Model: {model_used}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(prob_delay, text=f"Delay probability: {prob_delay:.1%}")

    st.markdown("""
    <div class="info-note">
    <strong>Note:</strong> Uses only features available before departure — airline, origin, time, season, and precipitation.
    Overall accuracy ~72%, ROC-AUC ~0.747.
    </div>
    """, unsafe_allow_html=True)
