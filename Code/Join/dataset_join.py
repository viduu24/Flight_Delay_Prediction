import pandas as pd
from pathlib import Path

FLIGHTS_FILE = "flight_data_2024_sample.csv"
WEATHER_FILE = "weather_data_sample.csv"
MERGED_FILE = "merged_flights.csv"

dir_path = Path(__file__).parent # current
file_path = dir_path.parent / "Flight_Data" / FLIGHTS_FILE
weather_path = dir_path.parent / "Weather_data" / WEATHER_FILE
output_path = dir_path.parent / "Merged_Data" / MERGED_FILE


    
def standardize_date_city(flights, weather):
    weather["Date_Time"] = pd.to_datetime(weather["Date_Time"])
    flights["fl_date"]   = pd.to_datetime(flights["fl_date"])

    #regex below is used to remove the State from city name
    flights["origin_city"] = flights["origin_city_name"].str.replace(r",.*", "", regex=True)

    return flights, weather
    
# Current implementation doesn't take time into account!!!
def perform_merge(flights, weather):
    weather["date"] = weather["Date_Time"].dt.date
    flights["date"] = flights["fl_date"].dt.date
    merged = flights.merge(
        weather,
        left_on = ["origin_city", "date",],
        right_on = ["Location",    "date",],
        how= "inner"
    )

    return merged

def main():
    flights = pd.read_csv(file_path)
    weather = pd.read_csv(weather_path)
    
    flights, weather = standardize_date_city(flights, weather)
    merged = perform_merge(flights, weather)
    merged.to_csv(output_path, index=False)
    
    
if __name__ == "__main__":
    main()