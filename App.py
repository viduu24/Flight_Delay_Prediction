import streamlit as st

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f1629;
        border-right: 1px solid #1e2d4a;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {
        color: #94a3b8 !important;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #818cf8 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .nav-card {
        background: linear-gradient(135deg, #111827 0%, #1e2d4a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s;
        cursor: pointer;
    }
    .nav-card:hover {
        border-color: #60a5fa;
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(96,165,250,0.15);
    }
    .nav-card h3 { color: #60a5fa; margin: 0 0 0.4rem 0; font-size: 1.1rem; }
    .nav-card p  { color: #94a3b8; margin: 0; font-size: 0.88rem; }
    .stat-box {
        background: linear-gradient(135deg, #111827, #1a2438);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .stat-box .val { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .stat-box .lbl { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }
    .pill {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        border-radius: 999px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        margin: 0.2rem;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero ---
st.markdown('<div class="hero-title">✈️ Flight Delay Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Predicting U.S. domestic flight delays using historical flight operations and weather sensor data (2024).</div>', unsafe_allow_html=True)

st.divider()

# --- Quick Stats ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><div class="val">~5.7M</div><div class="lbl">Total Flights</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="val">~29%</div><div class="lbl">Delayed (>15 min)</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="val">2</div><div class="lbl">ML Models</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-box"><div class="val">~71%</div><div class="lbl">Best Model Accuracy</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Navigation Cards ---
st.markdown("### 🗺️ Navigate the App")
st.markdown("Use the **sidebar** to switch pages, or explore the sections below.")

cols = st.columns(2)
pages = [
    ("📦", "Dataset Overview", "Explore the flight and weather datasets — sources, schema, key columns, and how they were joined."),
    ("✈️", "Flight EDA", "Flight-operations analysis: delay rates by airline, origin, route, hour, weekday, and season."),
    ("🌦️", "Weather EDA", "Impact of temperature, precipitation, wind, and humidity on delay probability."),
    ("🤖", "Models", "Baseline, Bagged Decision Trees, and KNN — metrics, confusion matrices, and ROC curves."),
    ("🔮", "Predict a Flight", "Enter your flight details and get a real-time delay prediction from the saved models."),
]

for i, (icon, title, desc) in enumerate(pages):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="nav-card">
            <h3>{icon} {title}</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🏗️ Project Structure")
st.markdown("""
<span class="pill">Flight_Data/</span>
<span class="pill">Weather_Data/</span>
<span class="pill">Merged_Data/</span>
<span class="pill">Code/Models/bagging_model.pkl</span>
<span class="pill">Code/Models/knn_model.pkl</span>
""", unsafe_allow_html=True)
