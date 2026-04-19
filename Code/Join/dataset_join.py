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
        # Normalize name: lowercase and strip _x/_y suffixes from merges
        base = col.lower().rstrip("_xy").rstrip("_")
        if base in seen:
            # Keep the first occurrence, drop subsequent ones
            cols_to_drop.append(col)
        else:
            seen[base] = col

    if cols_to_drop:
        print(f"Dropping duplicate columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    return df


def perform_merge(flights, weather):
    # Datetime conversion
    weather["Date_Time"] = pd.to_datetime(weather["Date_Time"])
    flights["fl_date"] = pd.to_datetime(flights["fl_date"])

    # Sort for merge_asof
    flights = flights.sort_values("fl_date").reset_index(drop=True)
    weather = weather.sort_values("Date_Time").reset_index(drop=True)

    # Rename weather location to match flights origin_city
    weather_renamed = weather.rename(columns={"Location": "origin_city"})
    weather_cols = [c for c in weather.columns if c not in ["Location", "Date_Time"]]

    merged_parts = []

    # Group by origin_city and merge_asof within each group
    for city, flight_group in flights.groupby("origin_city"):
        weather_group = weather_renamed[weather_renamed["origin_city"] == city].copy()

        if weather_group.empty:
            # No weather data for this city — keep flights with null weather cols
            for col in weather_cols:
                flight_group[col] = None
            flight_group["Date_Time"] = pd.NaT
            merged_parts.append(flight_group)
            continue

        # merge_asof within this city group
        merged_city = pd.merge_asof(
            flight_group.sort_values("fl_date"),
            weather_group.sort_values("Date_Time"),
            left_on="fl_date",
            right_on="Date_Time",
            by="origin_city",
            direction="nearest",
            tolerance=pd.Timedelta("3H")
        )
        merged_parts.append(merged_city)

    merged = pd.concat(merged_parts, ignore_index=True)

    # Drop duplicate columns (same name or _x/_y variants)
    merged = drop_duplicate_columns(merged)

    # Drop true duplicate flight rows only (same flight date + carrier + flight number)
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
    print(f"Rows with weather matched:    {merged['Date_Time'].notna().sum():,}")
    print(f"Rows with no weather match:   {merged['Date_Time'].isna().sum():,}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")