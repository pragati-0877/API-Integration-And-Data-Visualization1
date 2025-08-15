import requests
import os
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: API Integration

# Example coordinates (Delhi, India)
lat = 28.7041
lon = 77.1025

# Replace with your OpenWeather API key
api_key = "09747d0b43fcf17a7981be4591146ed1"

# API URL
url = "https://api.openweathermap.org/data/2.5/forecast"

# API Request Parameters
params = {
    "lat": lat,
    "lon": lon,
    "appid": api_key,
    "units": "metric"
}

# Fetch data from OpenWeather API
response = requests.get(url, params=params)

# Step 2: Handle Response
if response.status_code == 200:
    data = response.json()
    
    # Extract relevant weather details
    forecast_list = data["list"]
    
    # Create lists to store extracted data
    dates = []
    temperatures = []
    humidity = []

    for forecast in forecast_list:
        dates.append(forecast["dt_txt"])  # Date & Time
        temperatures.append(forecast["main"]["temp"])  # Temperature in Celsius
        humidity.append(forecast["main"]["humidity"])  # Humidity percentage

    # Convert data into a Pandas DataFrame
    weather_df = pd.DataFrame({"Date": dates, "Temperature (°C)": temperatures, "Humidity (%)": humidity})

    # Step 3: Data Visualization

    # Convert Date column to datetime format
    weather_df["Date"] = pd.to_datetime(weather_df["Date"])

    # Plot the temperature trend
    plt.figure(figsize=(12, 6))
    plt.plot(weather_df["Date"], weather_df["Temperature (°C)"], marker="o", linestyle="-", color="b", label="Temperature (°C)")
    plt.xlabel("Date & Time")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Forecast")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the humidity trend
    plt.figure(figsize=(12, 6))
    plt.plot(weather_df["Date"], weather_df["Humidity (%)"], marker="s", linestyle="--", color="g", label="Humidity (%)")
    plt.xlabel("Date & Time")
    plt.ylabel("Humidity (%)")
    plt.title("Humidity Forecast")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

else:
    print(f"Error {response.status_code}: {response.json().get('message', 'Unknown error')}")
