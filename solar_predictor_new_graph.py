# solar_predictor.py
import warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import pandas as pd
import numpy as np
import joblib
import requests
import pytz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import os

class SolarPredictor:
    def __init__(self, api_key, lat, lon):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sunny_model = joblib.load(os.path.join(base_dir, 'sunny_gbr.pkl'))
        self.cloudy_model = joblib.load(os.path.join(base_dir, 'cloudy_gbr.pkl'))
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.tz = pytz.timezone('America/New_York')

    def _call_weather_api(self):
        """Fetch weather data with fixed date range"""
        url = "https://api.openweathermap.org/data/3.0/onecall"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric',
            'exclude': 'current,minutely,daily,alerts'
        }
        try:
            response = requests.get(url, params=params, timeout=20, verify=False)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Weather API Error: {str(e)}")
            return None

    def _create_fixed_date_data(self):
        """Generate data from Apr 22 to Apr 28 2024"""
        start_date = datetime(2024, 4, 22, tzinfo=self.tz)
        return pd.DataFrame([{
            "datetime": start_date + timedelta(hours=i),
            "is_sunny": True  # Will be updated with real data
        } for i in range(168)])

    def _generate_3d_chart(self, df):
        """Generate 3D waterfall chart with English labels"""
        try:
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Date formatting
            dates = pd.date_range(start='2024-04-22', periods=7)
            cmap = plt.get_cmap('viridis')
            verts = []
            
            for day_idx in range(7):
                day_data = df[df['datetime'].dt.date == dates[day_idx].date()]
                
                x = day_data['datetime'].dt.hour.values
                y = np.full_like(x, day_idx)
                z = day_data['prediction'].values
                
                # Plot main curve
                ax.plot(x, y, z, 
                       color=cmap(day_idx/7),
                       linewidth=2,
                       marker='o',
                       markersize=5,
                       markerfacecolor=cmap(day_idx/7),
                       label=dates[day_idx].strftime('%b %d'))
                
                # Generate 3D polygons
                if len(x) > 1:
                    base_line = list(zip(x, y, np.zeros_like(z)))
                    top_line = list(zip(x, y, z))
                    verts.append(base_line + top_line[::-1]) 
                    
                    # Add vertical lines
                    for xi, yi, zi in zip(x, y, z):
                        ax.plot([xi, xi], [yi, yi], [0, zi],
                               color=cmap(day_idx/7),
                               alpha=0.2,
                               linestyle=':')

            # Add shaded surfaces
            if verts:
                poly = art3d.Poly3DCollection(verts, 
                                            facecolors=[cmap(i/7, alpha=0.15) for i in range(len(verts))],
                                            edgecolors='none')
                ax.add_collection(poly)

            # Axis settings
            ax.set_xlabel('\nHour of Day (0-23)', fontsize=12, labelpad=15)
            ax.set_ylabel('\nForecast Days', fontsize=12, labelpad=15)
            ax.set_zlabel('\nPower Generation (kWh)', fontsize=12, labelpad=15)
            
            # Tick settings
            ax.set_xticks(np.arange(0, 24, 3))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
            ax.set_yticks(range(7))
            ax.set_yticklabels([dates[i].strftime('%b %d') for i in range(7)])
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            
            # View angle
            ax.view_init(elev=30, azim=-60)
            plt.title("7-Day Solar Power Generation Forecast (Apr 22-28 2024)\n", 
                      fontsize=16, pad=20)
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.tight_layout()
            plt.savefig('fixed_dates_forecast.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Chart generation failed: {str(e)}")
            raise

    def predict(self):
        """Main prediction workflow"""
        try:
            # Try to get real data first
            api_data = self._call_weather_api()
            if api_data and 'hourly' in api_data:
                df = self._process_api_data(api_data)
            else:
                print("Using predefined date range data")
                df = self._create_fixed_date_data()
                features = self._create_features(df)
                df['prediction'] = np.where(
                    df['is_sunny'],
                    self.sunny_model.predict(features),
                    self.cloudy_model.predict(features)
                )
            
            self._generate_3d_chart(df)
            print("Forecast chart generated: fixed_dates_forecast.png")
            return True
            
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return False

    def _process_api_data(self, api_data):
        """Process API data for fixed date range"""
        df = self._create_fixed_date_data()
        
        # Update with real weather data
        for idx, hour_data in enumerate(api_data['hourly'][:168]):
            dt = datetime.fromtimestamp(hour_data['dt'], tz=self.tz)
            if idx < len(df):
                df.at[idx, 'is_sunny'] = 800 <= hour_data['weather'][0]['id'] <= 802
        
        # Generate predictions
        features = self._create_features(df)
        df['prediction'] = np.where(
            df['is_sunny'],
            self.sunny_model.predict(features),
            self.cloudy_model.predict(features)
        )
        return df

    def _create_features(self, df):
        """Create model features"""
        df['hour'] = df['datetime'].dt.hour
        df['dayofyear'] = df['datetime'].dt.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df[['day_sin', 'day_cos', 'hour_sin', 'hour_cos']]

if __name__ == "__main__":
    predictor = SolarPredictor(
        api_key="ee48a471dd4db095b1a35c6dec7afb08",
        lat=35.567,
        lon=-82.608
    )
    predictor.predict()