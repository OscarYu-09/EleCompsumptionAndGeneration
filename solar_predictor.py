# solar_predictor.py
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)  # Suppress SSL warnings

import pandas as pd  # Handle weather data
import numpy as np   # Feature calculations
import joblib  # Load trained models
import requests  # Fetch weather API data
import pytz  # Timezone conversions
from datetime import datetime, timedelta
import matplotlib.pyplot as plt  # Generate prediction charts
import os  # Path handling

class SolarPredictor:
    def __init__(self, api_key, lat, lon):
        # Load models using absolute paths (ensures file location independence)
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script's directory
        self.sunny_model = joblib.load(os.path.join(base_dir, 'sunny_gbr.pkl'))  # From model_trainer.py
        self.cloudy_model = joblib.load(os.path.join(base_dir, 'cloudy_gbr.pkl'))
        
        # API parameters (user-provided values)
        self.api_key = api_key  # Get from OpenWeatherMap.org
        self.lat = lat         # Location latitude (e.g., 35.567)
        self.lon = lon         # Location longitude (e.g., -82.608)
        self.tz = pytz.timezone('America/New_York')  # Local timezone for timestamps

    def _call_weather_api(self):
        """Fetch 48-hour weather forecast from OpenWeather API.
        - Returns JSON with 'hourly' forecast list.
        - API docs: https://openweathermap.org/api/one-call-3
        """
        url = "https://api.openweathermap.org/data/3.0/onecall"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric',  # Metric units (Celsius, m/s)
            'exclude': 'minutely,daily,alerts'  # Request hourly data only
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            return response.json()  # Weather data in JSON format
        except Exception as e:
            print(f"API Call Failed: {str(e)}")
            return None

    def _parse_weather_data(self, api_data):
        """Convert API response to DataFrame with datetime and weather labels.
        - api_data: JSON from OpenWeather.
        - is_sunny: True if weather code 800-802 (clear/partly cloudy).
        """
        forecast = []
        for hour_data in api_data.get('hourly', []):
            dt = datetime.fromtimestamp(hour_data['dt'], tz=self.tz)  # Convert UNIX timestamp
            weather_id = hour_data['weather'][0]['id']  # Weather condition code
            is_sunny = 800 <= weather_id <= 802  # Determine sky condition
            forecast.append({"datetime": dt, "is_sunny": is_sunny})
        return pd.DataFrame(forecast)  # Structured data for prediction

    def _create_features(self, df):
        """Generate model input features identical to preprocessing.
        - df: DataFrame with 'datetime' from API.
        - Values calculated: day_sin, day_cos, etc. (matches training data).
        """
        df['hour'] = df['datetime'].dt.hour        # 0-23 (same as preprocessing)
        df['dayofyear'] = df['datetime'].dt.dayofyear  # 1-365
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)  # Yearly cycle
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)       # Daily cycle
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df[['day_sin', 'day_cos', 'hour_sin', 'hour_cos']]  # Feature order MUST match training

    def predict(self):
        """Main prediction workflow."""
        # Fetch weather data (real-time input)
        api_data = self._call_weather_api()
        if not api_data:
            print("⚠️ Use Backup Data")  # Fallback if API fails
            weather_df = self._load_backup_data()  # Generate mock data
        else:
            weather_df = self._parse_weather_data(api_data)  # Parse real data
        
        # Prepare features for prediction (identical to training format)
        features = self._create_features(weather_df)
        
        # Predict generation using appropriate model per hour
        weather_df['prediction'] = np.where(
            weather_df['is_sunny'],
            self.sunny_model.predict(features),  # Sunny-day predictions
            self.cloudy_model.predict(features)  # Cloudy-day predictions
        )
        
        # Generate visualization (output for users)
        self._generate_plots(weather_df)
        return weather_df

    def _load_backup_data(self):
        """Create mock data if API fails (ensures system reliability)."""
        base_date = datetime.now(self.tz)  # Current time in local timezone
        return pd.DataFrame([{
            "datetime": base_date + timedelta(hours=i),  # 48-hour timeline
            "is_sunny": i % 3 == 0  # Alternate sunny/cloudy
        } for i in range(48)])

    def _generate_plots(self, df):
        """Generate prediction line chart."""
        plt.figure(figsize=(12, 6))  # Figure size
        plt.plot(df['datetime'], df['prediction'], marker='o', linestyle='-')  # Plot data points
        plt.title('Power Generation Forecast for the next 48 hours')  # Chart title
        plt.xlabel('Time')  # X-axis label
        plt.ylabel('Power Generation (kWh)')  # Y-axis label
        plt.grid(True)  # Show grid
        plt.savefig('prediction_chart.png')  # Save as PNG file
        plt.close()  # Free memory

if __name__ == "__main__":
    predictor = SolarPredictor(
        api_key="ee48a471dd4db095b1a35c6dec7afb08",  # Replace with actual OpenWeather API key
        lat=35.567,   # Example latitude
        lon=-82.608   # Example longitude
    )
    try:
        predictor.predict()
        print("✅ Prediction complete, chart saved")
    except Exception as e:
        print(f"❌ Prediction fail: {str(e)}")