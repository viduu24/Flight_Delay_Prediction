import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import BytesIO
from bs4 import BeautifulSoup

st.set_page_config(page_title="Flight EDA", page_icon="✈️", layout="wide")

# ── Google Drive downloader (handles virus-scan warning) ─────────────────────
def download_gdrive_csv(file_id: str) -> pd.DataFrame | None:
    session = requests.Session()

    # Step 1: initial request
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(url, stream=True)

    # Step 2: if Google returns the virus-scan HTML warning, parse the real download URL
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        soup = BeautifulSoup(response.text, "html.parser")
        form = soup.find("form", {"id": "download-form"})
        if form:
            action = form.get("action")
            params = {inp.get("name"): inp.get("value")
                      for inp in form.find_all("input")
                      if inp.get("name")}
            response = session.get(action, params=params, stream=True)
        else:
            # Fallback: try usercontent URL directly
            response = session.get(
                "https://drive.usercontent.google.com/download",
                params={"id": file_id, "export": "download", "confirm": "t"},
                stream=True,
            )

    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content), low_memory=False)

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    FILE_ID = "1hgMTsjDw8uyi3MZQkrQ6TI11j3YIEPzA"
    try:
        return download_gdrive_csv(FILE_ID)
    except Exception as e:
        st.error(f"❌ Google Drive download failed: {e}")
        return None

df = load_data()

# ── Validate ──────────────────────────────────────────────────────────────────
if df is None:
    st.error("❌ Dataset could not be loaded.")
    st.stop()

df.columns = df.columns.str.strip().str.lower()

required = ["is_delay", "delay_in_minutes"]
missing  = [c for c in required if c not in df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.write("Available:", df.columns.tolist())
    st.stop()

# ── Safe column references ────────────────────────────────────────────────────
col_delay   = "is_delay"
col_minutes = "delay_in_minutes"
col_hour    = "departure_hour" if "departure_hour" in df.columns else None

for col in [col_delay, col_minutes, col_hour]:
    if col and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("✈️ Flight Delay EDA")

total      = len(df)
delayed    = int(df[col_delay].fillna(0).sum())
delay_rate = delayed / total if total else 0
avg_delay  = df[col_minutes].mean()
med_delay  = df[col_minutes].median()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Flights",   f"{total:,}")
c2.metric("Delayed Flights", f"{delayed:,}")
c3.metric("Delay Rate",      f"{delay_rate:.1%}")
c4.metric("Avg Delay",       f"{avg_delay:.1f} min")
c5.metric("Median Delay",    f"{med_delay:.1f} min")

st.divider()

# ── Distribution ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    fig = px.pie(names=["Not Delayed", "Delayed"],
                 values=[total - delayed, delayed],
                 title="Delay Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    sample = df[col_minutes].dropna().clip(-60, 300)
    fig = px.histogram(sample, nbins=60, title="Delay Minutes Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ── Hourly pattern ────────────────────────────────────────────────────────────
if col_hour:
    hourly = df.groupby(col_hour)[col_delay].mean().reset_index()
    fig = px.line(hourly, x=col_hour, y=col_delay, title="Delay Rate by Hour")
    st.plotly_chart(fig, use_container_width=True)

# ── Airline performance ───────────────────────────────────────────────────────
if "op_unique_carrier" in df.columns:
    airline = (
        df.groupby("op_unique_carrier")
        .agg(flights=(col_delay, "size"), delay_rate=(col_delay, "mean"))
        .reset_index()
    )
    airline = airline[airline["flights"] > 1000]
    fig = px.bar(airline.sort_values("delay_rate"),
                 x="delay_rate", y="op_unique_carrier",
                 orientation="h", title="Airline Delay Rates")
    st.plotly_chart(fig, use_container_width=True)

# ── Delay reasons ─────────────────────────────────────────────────────────────
delay_cols = ["carrier_delay","weather_delay","nas_delay","security_delay","late_aircraft_delay"]
existing   = [c for c in delay_cols if c in df.columns]
if existing:
    reason = df[existing].sum().reset_index()
    reason.columns = ["reason", "minutes"]
    fig = px.bar(reason, x="minutes", y="reason", orientation="h", title="Delay Causes")
    st.plotly_chart(fig, use_container_width=True)
