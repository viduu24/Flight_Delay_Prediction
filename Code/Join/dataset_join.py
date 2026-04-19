import pandas as pd
from pathlib import Path

FLIGHTS_FILE = "flight_data_sample.csv"
WEATHER_FILE = "weather_data.csv"
MERGED_FILE = "merged_flights.csv"

dir_path = Path(__file__).parent
file_path = dir_path.parent.parent / "Flight_Data" / FLIGHTS_FILE
weather_path = dir_path.parent.parent / "Weather_Data" / WEATHER_FILE
output_path = dir_path.parent.parent / "Merged_Data" / MERGED_FILE


def drop_duplicate_columns(df):
    seen = {}
    cols_to_drop = []

    for col in df.columns:
        base = col.lower().rstrip("_xy").rstrip("_")
        if base in seen:
            cols_to_drop.append(col)
        else:
            seen[base] = col

    if cols_to_drop:
        print(f"Dropping duplicate columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    return df


def perform_merge(flights, weather):
    # Build proper datetime for flights: fl_date + Departure_Hour
    # e.g. 2024-01-04 + hour 9 -> 2024-01-04 09:00:00
    flights["fl_datetime"] = pd.to_datetime(flights["fl_date"]) + pd.to_timedelta(flights["Departure_Hour"], unit="h")

    weather["Date_Time"] = pd.to_datetime(weather["Date_Time"])

    # Sort for merge_asof
    flights = flights.sort_values("fl_datetime").reset_index(drop=True)
    weather = weather.sort_values("Date_Time").reset_index(drop=True)

    # Rename weather location to match flights origin_city
    weather_renamed = weather.rename(columns={"Location": "origin_city"})
    weather_cols = [c for c in weather.columns if c not in ["Location", "Date_Time"]]

    merged_parts = []

    for city, flight_group in flights.groupby("origin_city"):
        weather_group = weather_renamed[weather_renamed["origin_city"] == city].copy()

        if weather_group.empty:
            for col in weather_cols:
                flight_group[col] = None
            flight_group["Date_Time"] = pd.NaT
            merged_parts.append(flight_group)
            continue

        merged_city = pd.merge_asof(
            flight_group.sort_values("fl_datetime"),
            weather_group.sort_values("Date_Time"),
            left_on="fl_datetime",
            right_on="Date_Time",
            by="origin_city",
            direction="nearest",
            tolerance=pd.Timedelta("3H")
        )
        merged_parts.append(merged_city)

    merged = pd.concat(merged_parts, ignore_index=True)

    # Add Weather_Data_Present column based on whether Date_Time was matched
    merged["Weather_Data_Present"] = merged["Date_Time"].notna().map({True: "Yes", False: "No"})

    # Drop duplicate columns
    merged = drop_duplicate_columns(merged)

    # Drop Date_Time and fl_datetime (temp column) from final output
    drop_cols = [c for c in ["Date_Time", "fl_datetime"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    # Drop true duplicate flight rows
    merged = merged.drop_duplicates(
        subset=["fl_date", "op_unique_carrier", "op_carrier_fl_num"], keep="first"
    )

    # Restore original row order
    merged = merged.sort_values("fl_date").reset_index(drop=True)

    return merged


if __name__ == "__main__":
    print("Loading data...")
    flights = pd.read_csv(file_path)
    weather = pd.read_csv(weather_path)

    print(f"Flights loaded:  {len(flights):,} rows")
    print(f"Weather loaded:  {len(weather):,} rows")

    print("Merging...")
    merged = perform_merge(flights, weather)

    print(f"Merged result:   {len(merged):,} rows")
    print(f"Columns in output ({len(merged.columns)}): {list(merged.columns)}")
    print(f"Rows with weather matched:    {(merged['Weather_Data_Present'] == 'Yes').sum():,}")
    print(f"Rows with no weather match:   {(merged['Weather_Data_Present'] == 'No').sum():,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")