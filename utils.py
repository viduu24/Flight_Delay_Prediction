import streamlit as st
import pandas as pd
import numpy as np

# ── Synthetic merged dataset that mirrors the real repo structure ──────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_merged():
    np.random.seed(42)
    n = 8000

    carriers  = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "HA", "G4"]
    cities    = ["New York", "Los Angeles", "Chicago", "Dallas", "Houston",
                 "Atlanta", "Miami", "Phoenix", "Denver", "Boston",
                 "Seattle", "Minneapolis", "Las Vegas", "San Francisco", "Orlando"]
    states    = ["New York", "California", "Illinois", "Texas", "Texas",
                 "Georgia", "Florida", "Arizona", "Colorado", "Massachusetts",
                 "Washington", "Minnesota", "Nevada", "California", "Florida"]
    seasons   = ["Winter", "Spring", "Summer", "Fall"]
    wx_conds  = ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm"]

    month        = np.random.randint(1, 13, n)
    day_of_month = np.random.randint(1, 29, n)
    day_of_week  = np.random.randint(1, 8, n)
    dep_hour     = np.random.randint(0, 24, n)
    carrier_idx  = np.random.randint(0, len(carriers), n)
    city_idx     = np.random.randint(0, len(cities), n)
    distance     = np.random.randint(150, 3000, n)

    temp_c       = np.random.uniform(-10, 38, n)
    humidity     = np.random.uniform(20, 95, n)
    precip_mm    = np.random.exponential(2, n)
    wind_kmh     = np.random.uniform(0, 80, n)
    wx_idx       = np.random.randint(0, len(wx_conds), n)

    # Delay probability model
    p = (
        0.22
        + 0.12 * np.isin(wx_conds[i] if False else None, [])   # placeholder
        + 0.00  # replaced below
    )
    wx_arr = np.array([wx_conds[i] for i in wx_idx])
    p = (
        0.22
        + 0.10 * np.isin(wx_arr, ["Rain", "Snow", "Fog", "Thunderstorm"]).astype(float)
        + 0.07 * ((dep_hour >= 15) & (dep_hour <= 20)).astype(float)
        + 0.05 * np.isin(month, [6, 7, 8, 12]).astype(float)
        + 0.04 * np.isin(day_of_week, [5, 7]).astype(float)
        + 0.03 * np.isin(np.array(carriers)[carrier_idx], ["NK", "F9", "G4"]).astype(float)
        - 0.06 * ((dep_hour >= 5) & (dep_hour <= 9)).astype(float)
        + 0.004 * (precip_mm)
        + 0.002 * (wind_kmh - 20).clip(0)
    ).clip(0.05, 0.88)

    is_delay   = (np.random.uniform(0, 1, n) < p).astype(int)
    delay_min  = np.where(is_delay == 1,
                          np.random.exponential(38, n).clip(1, 300),
                          np.random.uniform(-15, 14, n))

    # Delay sub-reasons (only > 0 for delayed flights)
    carrier_delay  = np.where(is_delay, np.random.exponential(12, n).clip(0, 120), 0.0)
    weather_delay  = np.where(is_delay & np.isin(wx_arr, ["Rain","Snow","Fog","Thunderstorm"]),
                              np.random.exponential(18, n).clip(0, 180), 0.0)
    nas_delay      = np.where(is_delay, np.random.exponential(8, n).clip(0, 90), 0.0)
    security_delay = np.where(is_delay, np.random.exponential(2, n).clip(0, 30), 0.0)
    late_aircraft  = np.where(is_delay, np.random.exponential(15, n).clip(0, 120), 0.0)

    season_map = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall",12:"Winter"}

    df = pd.DataFrame({
        "fl_date":              pd.date_range("2024-01-01", periods=n, freq="h").date,
        "op_unique_carrier":    [carriers[i] for i in carrier_idx],
        "origin_city":          [cities[i]   for i in city_idx],
        "origin_state":         [states[i]   for i in city_idx],
        "distance":             distance,
        "month":                month,
        "day_of_month":         day_of_month,
        "day_of_week":          day_of_week,
        "departure_hour":       dep_hour,
        "dep_time":             dep_hour * 100,
        "season":               [season_map[m] for m in month],
        "weather_condition":    wx_arr,
        "temperature_c":        temp_c.round(1),
        "humidity_pct":         humidity.round(1),
        "precipitation_mm":     precip_mm.round(2),
        "wind_speed_kmh":       wind_kmh.round(1),
        "is_delay":             is_delay,
        "delay_in_minutes":     delay_min.round(1),
        "carrier_delay":        carrier_delay.round(1),
        "weather_delay":        weather_delay.round(1),
        "nas_delay":            nas_delay.round(1),
        "security_delay":       security_delay.round(1),
        "late_aircraft_delay":  late_aircraft.round(1),
        "weather_data_present": np.where(np.random.uniform(0,1,n) > 0.05, "Yes", "No"),
    })
    return df
