import streamlit as st

# ── Palette ────────────────────────────────────────────────────────────────────
# #EEF0F8  light lavender background
# #2D2B6B  deep navy-indigo (headings, dark text)
# #9B89C4  medium purple (accents, borders, icons)
# #C4B5E8  light lilac (subtle borders, pills)
# #FFFFFF  white (card surfaces)
# #6B6A9B  muted purple (body text, descriptions)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(238,240,248,0)",
    plot_bgcolor="rgba(238,240,248,0)",
    font=dict(color="#2D2B6B", family="Inter, sans-serif"),
    title_font=dict(color="#2D2B6B", size=15),
    xaxis=dict(gridcolor="#C4B5E8", linecolor="#C4B5E8", tickfont=dict(color="#6B6A9B")),
    yaxis=dict(gridcolor="#C4B5E8", linecolor="#C4B5E8", tickfont=dict(color="#6B6A9B")),
    # legend intentionally omitted — set per-chart to avoid duplicate kwarg errors
)

LEGEND_H = dict(legend=dict(orientation="h", y=-0.15, font=dict(color="#2D2B6B")))
LEGEND_DEFAULT = dict(legend=dict(font=dict(color="#2D2B6B")))

PURPLE_SEQ   = ["#2D2B6B", "#4A3F9F", "#7B6DC4", "#9B89C4", "#C4B5E8", "#E0D9F5"]
PURPLE_DIVG  = ["#2D2B6B", "#6B6A9B", "#C4B5E8", "#FFFFFF", "#E8D5B7", "#9B6A4B", "#5C3317"]
ACCENT_COLOR = "#7B6DC4"

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background-color: #EEF0F8;
    color: #2D2B6B;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 2px solid #C4B5E8 !important;
}
[data-testid="stSidebar"] * {
    color: #2D2B6B !important;
}
[data-testid="stSidebarNav"] a {
    border-radius: 8px;
    padding: 6px 12px;
    transition: background 0.15s;
}
[data-testid="stSidebarNav"] a:hover {
    background: #EEF0F8 !important;
}

/* ── Headers ── */
h1, h2, h3, h4, h5 {
    color: #2D2B6B !important;
    font-family: 'Space Mono', monospace !important;
}

/* ── Section header ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #2D2B6B;
    border-bottom: 3px solid #9B89C4;
    padding-bottom: 10px;
    margin-bottom: 6px;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1.5px solid #C4B5E8;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(155,137,196,0.12);
}
[data-testid="metric-container"] label {
    color: #6B6A9B !important;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #2D2B6B !important;
    font-family: 'Space Mono', monospace;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1.5px solid #C4B5E8;
    margin: 20px 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 10px;
    border: 1.5px solid #C4B5E8;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #6B6A9B;
    border-radius: 8px;
    font-weight: 500;
    padding: 6px 18px;
}
.stTabs [aria-selected="true"] {
    background: #9B89C4 !important;
    color: #FFFFFF !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2D2B6B, #7B6DC4);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.6rem;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(123,109,196,0.35);
    color: #FFFFFF;
}

/* ── Selectbox / Slider labels ── */
label, .stSlider label {
    color: #2D2B6B !important;
    font-weight: 500;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #7B6DC4 !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #FFFFFF;
    border: 1.5px solid #C4B5E8;
    border-radius: 8px;
    color: #2D2B6B;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stDataFrame th { background: #9B89C4 !important; color: #FFFFFF !important; }

/* ── Shared card ── */
.card {
    background: #FFFFFF;
    border: 1.5px solid #C4B5E8;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(155,137,196,0.1);
}

/* ── Left-accent card ── */
.accent-card {
    background: #FFFFFF;
    border-left: 4px solid #9B89C4;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 6px rgba(155,137,196,0.1);
}

/* ── Insight box ── */
.insight-box {
    background: #F4F1FB;
    border-left: 3px solid #7B6DC4;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.8rem 0;
    color: #6B6A9B;
    font-size: 0.88rem;
}

/* ── Badge / pill ── */
.badge {
    display: inline-block;
    background: #EEE8FA;
    color: #2D2B6B;
    border: 1px solid #C4B5E8;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin: 2px;
}
.badge-dark {
    background: #2D2B6B;
    color: #FFFFFF;
    border-color: #2D2B6B;
}
.badge-green { background: #E6F4EA; color: #1B5E20; border-color: #A5D6A7; }
.badge-warn  { background: #FFF8E1; color: #6D4C41; border-color: #FFCC80; }

/* ── Pipeline step ── */
.pipeline-step {
    background: #FFFFFF;
    border-left: 4px solid #9B89C4;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    border-radius: 0 10px 10px 0;
    box-shadow: 0 1px 4px rgba(155,137,196,0.1);
}
.step-title { font-weight: 600; color: #2D2B6B; font-family: 'Space Mono', monospace; }
.step-desc  { font-size: 0.88rem; color: #6B6A9B; margin-top: 2px; }

/* ── Model card ── */
.model-card {
    background: #FFFFFF;
    border: 1.5px solid #C4B5E8;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 8px rgba(155,137,196,0.1);
}
.model-card:hover {
    border-color: #9B89C4;
    box-shadow: 0 6px 20px rgba(155,137,196,0.2);
}
.model-card h3 { color: #2D2B6B !important; margin: 0 0 0.5rem 0; }
.model-card p  { color: #6B6A9B; font-size: 0.88rem; margin: 0.2rem 0; }

/* ── Winner badge ── */
.winner-badge {
    display: inline-block;
    background: linear-gradient(135deg, #2D2B6B, #7B6DC4);
    color: #FFFFFF;
    border-radius: 999px;
    padding: 0.15rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 0.5rem;
    vertical-align: middle;
    letter-spacing: 0.05em;
}

/* ── Prediction result cards ── */
.result-card-delay {
    background: linear-gradient(135deg, #FFF0F0, #FFE0E0);
    border: 2px solid #E57373;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(229,115,115,0.2);
}
.result-card-ok {
    background: linear-gradient(135deg, #F0F7FF, #E3F2FD);
    border: 2px solid #7B6DC4;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(123,109,196,0.2);
}
.result-icon  { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label { font-size: 1.5rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.result-prob  { font-size: 1rem; color: #6B6A9B; font-family: 'Space Mono', monospace; margin-top: 0.4rem; }

/* ── Stat card ── */
.stat-card {
    background: #FFFFFF;
    border: 1.5px solid #C4B5E8;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(155,137,196,0.1);
}
.stat-card .val { font-size: 1.5rem; font-weight: 700; color: #2D2B6B; font-family: 'Space Mono', monospace; }
.stat-card .lbl { font-size: 0.73rem; color: #9B89C4; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }

footer { visibility: hidden; }
</style>
"""


def apply_theme():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
