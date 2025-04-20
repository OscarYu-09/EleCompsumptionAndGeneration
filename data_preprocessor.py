# data_preprocessor.py
import pandas as pd  # Data manipulation library (reads Excel, creates DataFrames)
import numpy as np   # Math operations (sin/cosine for cyclical time encoding)
from datetime import datetime, timedelta  # Date/time conversions

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path  # User-provided path to raw Excel file

    def _excel_date_to_datetime(self, excel_date):
        """Convert Excel serial dates (e.g., 45292) to Python datetime objects.
        - excel_date: Raw number from Excel (1900-based, with leap year bug)
        - Values determined by Excel's date storage format.
        """
        if excel_date < 60:
            # Fix for dates before 1900-03-01 (Excel compatibility issue)
            return datetime(1899, 12, 31) + timedelta(days=int(excel_date))
        return datetime(1899, 12, 30) + timedelta(days=int(excel_date))  # Standard conversion

    def process(self):
        """Main data cleaning and feature engineering pipeline."""
        # Read raw Excel (columns: datetime, time, is_sunny, generation)
        df = pd.read_excel(self.file_path)
        
        # Convert Excel date numbers to real dates using helper function
        df['date'] = df['datetime'].apply(self._excel_date_to_datetime)
        
        # Combine date and time columns into a single datetime column
        df['datetime'] = df['date'] + pd.to_timedelta(df['time'].astype(str))
        df = df.drop(['date', 'time'], axis=1)  # Remove redundant columns
        
        # Extract time features (critical for solar pattern modeling)
        df['hour'] = df['datetime'].dt.hour        # 0-23 (solar intensity varies hourly)
        df['dayofyear'] = df['datetime'].dt.dayofyear  # 1-365 (seasonal changes)
        df['month'] = df['datetime'].dt.month      # 1-12 (monthly trends)
        
        # Cyclical encoding (converts linear time to repeating sine/cosine waves)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)  # Yearly cycle
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)  # Helps model understand seasonality
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)       # Daily cycle (sun position)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)       # 23:59 connects to 00:00
        
        # Save cleaned data for model training
        df.to_csv('processed_data.csv', index=False)  # Output file name fixed
        print("✅ Data pre-processing is complete")
        return df

if __name__ == "__main__":
    processor = DataPreprocessor("/Users/oscaryu/Downloads/School/materials/ScienceFair/generation/solar_generation.xlsx")  # User's Excel path
    processor.process()