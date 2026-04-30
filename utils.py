import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from bs4 import BeautifulSoup

FLIGHT_ID  = "1x80CYMrQ_B1XjY_TvQgAweBUea2gcw6b"
WEATHER_ID = "1TKlnXdIsgCj5o3x7ISh6uYTp_wLFytp7"
MERGED_ID  = "1hgMTsjDw8uyi3MZQkrQ6TI11j3YIEPzA"


def _gdrive_bytes(file_id: str) -> bytes:
    """Download raw bytes from a Google Drive file, handling the virus-scan page."""
    session  = requests.Session()
    url      = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(url, stream=True)

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
    return response.content


def _load_csv(file_id: str, keep_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Download a CSV and return a memory-efficient DataFrame.
    keep_cols should be lowercase — filtering happens AFTER lowercasing columns.
    """
    df = pd.read_csv(BytesIO(_gdrive_bytes(file_id)), low_memory=False)

    # Lowercase all column names first
    df.columns = df.columns.str.strip().str.lower()

    # Now drop columns we don't need (safe because names are already lowercase)
    if keep_cols:
        existing = [c for c in keep_cols if c in df.columns]
        df = df[existing]

    # Downcast to save ~50% memory
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")

    return df


# ── Cached loaders — one per dataset.
# All pages import from here so the cache is shared and data loads only once.

@st.cache_data(show_spinner="Loading flight dataset...")
def load_flights():
    return _load_csv(FLIGHT_ID)


@st.cache_data(show_spinner="Loading weather dataset...")
def load_weather():
    return _load_csv(WEATHER_ID)


@st.cache_data(show_spinner="Loading merged dataset...")
def load_merged():
    keep = [
        "is_delay", "delay_in_minutes", "departure_hour",
        "op_unique_carrier", "season", "month", "day_of_week",
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
        "temperature_c", "humidity_pct", "precipitation_mm",
        "wind_speed_kmh", "weather_data_present",
    ]
    return _load_csv(MERGED_ID, keep_cols=keep)
