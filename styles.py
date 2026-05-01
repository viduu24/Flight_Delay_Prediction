import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        .stApp {
            background-color: #EEF0F8;
            color: #2D2B6B;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #2D2B6B !important;
        }

        section[data-testid="stSidebar"] * {
            color: #EEF0F8 !important;
        }

        /* Buttons */
        .stButton > button {
            background: #2D2B6B !important;
            color: #EEF0F8 !important;
            border-radius: 10px !important;
            border: none !important;
        }

        .stButton > button:hover {
            background: #9B89C4 !important;
        }

        /* Cards */
        .kpi-card {
            background: #FFFFFF;
            border: 1.5px solid #C4B5E8;
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
        }

        /* Pills */
        .pill {
            background: #EEE8FA;
            color: #2D2B6B;
            border: 1px solid #C4B5E8;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            font-size: 0.75rem;
        }

    </style>
    """, unsafe_allow_html=True)
