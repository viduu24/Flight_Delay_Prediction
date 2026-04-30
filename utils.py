import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from bs4 import BeautifulSoup

FLIGHT_ID  = "1x80CYMrQ_B1XjY_TvQgAweBUea2gcw6b"
WEATHER_ID = "1TKlnXdIsgCj5o3x7ISh6uYTp_wLFytp7"
MERGED_ID  = "1hgMTsjDw8uyi3MZQkrQ6TI11j3YIEPzA"


def _gdrive_response(file_id: str):
    """Return a resolved requests.Response for a Google Drive file ID."""
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
    return response


def _load_csv(file_id: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Download a CSV from Google Drive and return a memory-efficient DataFrame."""
    content = BytesIO(_gdrive_response(file_id).content)

    if usecols:
        # Peek at header, filter to columns that actually exist
        header = pd.read_csv(content, nrows=0)
        header.columns = header.columns.str.strip().str.lower()
        use = [c for c in usecols if c in header.columns.tolist()]
        content.seek(0)
        df = pd.read_csv(content, usecols=use, low_memory=False)
    else:
        df = pd.read_csv(content, low_memory=False)

    df.columns = df.columns.str.strip().str.lower()

    # Downcast to save ~50% memory
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")

    return df


# ── One cached function per dataset.
# Because these functions live here (not copied into each page),
# ALL pages share the same cache entry — data is downloaded only once.

@st.cache_data(show_spinner="Loading flight dataset...")
def load_flights():
    return _load_csv(FLIGHT_ID)

@st.cache_data(show_spinner="Loading weather dataset...")
def load_weather():
    return _load_csv(WEATHER_ID)

@st.cache_data(show_spinner="Loading merged dataset...")
def load_merged():
    cols = [
        "is_delay", "delay_in_minutes", "departure_hour",
        "op_unique_carrier", "season", "month", "day_of_week",
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
        "temperature_c", "humidity_pct", "precipitation_mm",
        "wind_speed_kmh", "weather_data_present",
    ]
    return _load_csv(MERGED_ID, usecols=cols)
