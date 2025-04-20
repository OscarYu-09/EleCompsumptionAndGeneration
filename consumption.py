# consumption.py
from prophet import Prophet
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.dates as mdates
import os
import sys
import traceback
import platform
from itertools import product
from prophet.diagnostics import cross_validation, performance_metrics

# ======================
# 0. Environment Configuration
# ======================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 1. Data Loading & Preprocessing
# ======================
def load_and_preprocess_data():
    """Load and preprocess electricity consumption data"""
    try:
        file_path = "/Users/oscaryu/Downloads/School/materials/ScienceFair/Consumption/data.csv"
        print(f"\n[Step 1] Loading data file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found")
        
        df = pd.read_csv(file_path)
        print(" - Raw data loaded successfully")
        
        # Column renaming
        df = df.rename(columns={'date': 'ds', 'electricity': 'y'})
        print(" - Columns renamed to Prophet format")
        
        # Handle missing values
        initial_count = len(df)
        df = df.dropna(subset=['ds', 'y'])
        print(f" - Removed {initial_count - len(df)} records with missing values")
        
        # Set capacity
        max_consumption = df['y'].max()
        df['cap'] = max_consumption * 1.2
        print(f" - Capacity set to {max_consumption*1.2:.2f} kWh")
        
        # Datetime conversion
        df['ds'] = pd.to_datetime(df['ds'], format="%d/%m/%Y %H:%M", errors='coerce')
        invalid_dates = df['ds'].isna().sum()
        if invalid_dates > 0:
            print(f" - Warning: Removed {invalid_dates} invalid datetime records")
            df = df.dropna(subset=['ds'])
        
        # Filter to 2024 data
        df = df[(df['ds'] >= '2024-01-01') & (df['ds'] <= '2024-12-31 23:00:00')]
        print(f" - Final dataset range: {df['ds'].min()} to {df['ds'].max()}")
        
        return df
    
    except Exception as e:
        print("\n[Step 1 Error] Data loading failed:")
        traceback.print_exc()
        sys.exit(1)

# ======================
# Holiday Definitions (Adjusted for 2024)
# ======================
def get_holidays():
    """Define school holidays for 2024 based on academic calendar patterns"""
    holidays = pd.DataFrame({
        'holiday': 'school_holiday',
        'ds': pd.to_datetime([
            # Major breaks and events in 2024
            '2024-08-12',  # Prefect Arrival
            '2024-08-21',  # Opening Day
            '2024-09-26',  # Homecoming Start
            '2024-10-17',  # Fall Break Start
            '2024-11-28',  # Thanksgiving Break (4th Thursday)
            '2024-12-18',  # Winter Break Start
            '2025-01-05',  # Winter Break Return (adjusted for 2024-2025跨年)
            '2024-02-28',  # Spring Break Start
            '2024-04-03',  # Spring Weekend
            '2024-05-18'   # Graduation
        ]),
        'lower_window': -2,  # 2 days before event
        'upper_window': 3    # 3 days after event
    })
    return holidays

# ======================
# 2. Enhanced Grid Search with Extended Parameters
# ======================
def grid_search_parameters(df):
    """Perform hyperparameter tuning with expanded parameter grid"""
    print("\n[Grid Search] Starting parameter optimization...")
    
    # Extended parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.005, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [5.0, 10.0, 15.0, 20.0, 25.0],
        'holidays_prior_scale': [5.0, 10.0, 15.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'n_changepoints': [20, 25, 30]
    }
    
    best_score = float('inf')
    best_params = None
    best_model = None
    
    # Generate all parameter combinations
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    print(f"Total parameter combinations: {len(all_params)}")
    
    for idx, params in enumerate(all_params):
        print(f"\nTesting combination {idx+1}/{len(all_params)}: {params}")
        
        try:
            # Initialize model with current parameters
            model = Prophet(
                holidays=get_holidays(),
                growth='logistic',
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                n_changepoints=params['n_changepoints']
            )
            
            # Add custom seasonality
            model.add_seasonality(name='monthly', period=30.5, fourier_order=6)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
            
            model.fit(df)
            
            # Cross-validation
            df_cv = cross_validation(
                model,
                initial='8000 hours',
                period='720 hours',
                horizon='168 hours'
            )
            df_p = performance_metrics(df_cv)
            
            # Calculate weighted score
            current_rmse = df_p['rmse'].mean()
            current_mae = df_p['mae'].mean()
            weighted_score = 0.7 * current_rmse + 0.3 * current_mae
            
            if weighted_score < best_score:
                best_score = weighted_score
                best_params = params
                best_model = model
                print(f"New best score: {best_score:.2f} (RMSE: {current_rmse:.2f}, MAE: {current_mae:.2f})")
                
        except Exception as e:
            print(f"Error with params {params}: {str(e)}")
            continue
    
    print("\n[Grid Search Complete]")
    print(f"Best parameters: {best_params}")
    print(f"Best weighted score: {best_score:.2f}")
    
    return best_model, best_params

# ======================
# 3. Model Training with Best Parameters
# ======================
"""
def train_prophet_model(df, params):
    #Train final model with optimized parameters
    try:
        print("\n[Step 2] Training final model")
        
        model = Prophet(
            holidays=get_holidays(),
            growth='logistic',
            **params
        )
        
        # Add custom seasonality
        model.add_seasonality(name='monthly', period=30.5, fourier_order=6)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        
        model.fit(df)
        print(" - Model training completed")
        return model
    
    except Exception as e:
        print("\n[Step 2 Error] Model training failed:")
        traceback.print_exc()
        sys.exit(1)
"""
def train_prophet_model(df): 
    """Train model with predefined best parameters"""
    try:
        print("\n[Step 2] Training model with predefined best parameters")
        
    
        best_params = {
            'changepoint_prior_scale': 0.01,
            'seasonality_prior_scale': 25.0,
            'holidays_prior_scale': 5.0,
            'seasonality_mode': 'multiplicative',
            'n_changepoints': 25,
            'yearly_seasonality': True
        }
        
        model = Prophet(
            holidays=get_holidays(),
            growth='logistic',
            **best_params  
        )
        
    
        model.add_seasonality(name='monthly', period=30.5, fourier_order=6)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        
        model.fit(df)
        print(" - Model training completed")
        return model
    
    except Exception as e:
        print("\n[Step 2 Error] Model training failed:")
        traceback.print_exc()
        sys.exit(1)

# ======================
# 4. Model Validation
# ======================
def validate_model(model, df):
    """Perform cross-validation and print metrics"""
    try:
        print("\n[Step 4] Running cross-validation")
        from prophet.diagnostics import cross_validation, performance_metrics
        
        df_cv = cross_validation(
            model,
            initial='8000 hours',
            period='720 hours',
            horizon='168 hours'
        )
        
        df_p = performance_metrics(df_cv)
        print("\n[Validation Metrics]")
        print(df_p[['horizon', 'mae', 'rmse']].tail().to_string(index=False))
    
    except Exception as e:
        print("\n[Step 4 Warning] Cross-validation failed:")
        traceback.print_exc()

# ======================
# 5. Visualization Functions
# ======================
def generate_forecasts(model, df):
    try:
        print("\n[Step 3] Generating predictions")
        
        # Historical Fit
        history_forecast = model.predict(df)
        print(" - Historical fit generated")
        
        # Future Prediction（Future 87600 hours=1 year）
        future = model.make_future_dataframe(periods=8760, freq='h')  # Frequency unit = hour
        future['cap'] = model.history['cap'].max()  # application limit
        future_forecast = model.predict(future)
        
        # Generate graph
        plt.figure(figsize=(25,8))
        model.plot(future_forecast)
        model.plot_components(future_forecast)
        print(" - Future forecast generated")
        
        return history_forecast, future_forecast
    
    except Exception as e:
        print("\n[Step 3 Error] Prediction generation failed:")
        traceback.print_exc()
        sys.exit(1)


def plot_validation(df, history_forecast):
    """Plot actual vs fitted values with error metrics"""
    try:
        print("\n[Step 5] Generating validation plot")
        
        # Merge data
        comparison = df.merge(history_forecast[['ds', 'yhat']], on='ds', how='inner')
        if comparison.empty:
            raise ValueError("No overlapping timestamps found")
        
        # Calculate metrics
        mae = (comparison['y'] - comparison['yhat']).abs().mean()
        rmse = ((comparison['y'] - comparison['yhat']) ** 2).mean() ** 0.5
        
        # Main plot
        plt.figure(figsize=(18, 8))
        plt.plot(comparison['ds'], comparison['y'], label='Actual', linewidth=1.5, color='#1f77b4')
        plt.plot(comparison['ds'], comparison['yhat'], label='Fitted', linewidth=1.5, linestyle='--', color='#ff7f0e')
        
        # Formatting
        plt.title(f"Actual vs Fitted Values (2024)\nMAE: {mae:.2f} kWh, RMSE: {rmse:.2f} kWh", pad=20)
        plt.xlabel("Date", labelpad=15)
        plt.ylabel("Consumption (kWh)", labelpad=15)
        plt.legend()
        
        # Date formatting
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add zoomed subplot
        axins = ax.inset_axes([0.3, 0.5, 0.45, 0.45])
        axins.plot(comparison['ds'].iloc[:720], comparison['y'].iloc[:720], color='#1f77b4')
        axins.plot(comparison['ds'].iloc[:720], comparison['yhat'].iloc[:720], linestyle='--', color='#ff7f0e')
        axins.set_title('First 30 Days Zoom')
        axins.grid(True, alpha=0.3)
        ax.indicate_inset_zoom(axins, edgecolor="gray")
        
        # Save plot
        output_path = os.path.join(os.getcwd(), 'validation_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" - Validation plot saved to {output_path}")
    
    except Exception as e:
        print("\n[Step 5 Error] Validation plot failed:")
        traceback.print_exc()

def plot_seasonal_components(forecast):
    """Visualize daily/weekly/monthly seasonality"""
    try:
        print("\n[Step 6] Generating seasonal plots")
        
        # Daily seasonality
        plt.figure(figsize=(12, 6))
        forecast['hour'] = forecast['ds'].dt.hour
        forecast.groupby('hour')['daily'].mean().plot(kind='line', marker='o', color='#2ca02c')
        plt.title("Daily Consumption Pattern")
        plt.xlabel("Hour of Day")
        plt.ylabel("Seasonal Impact")
        plt.xticks(range(0, 24, 2))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        daily_path = os.path.join(os.getcwd(), 'daily_seasonality.png')
        plt.savefig(daily_path, dpi=300)
        plt.close()
        print(f" - Daily seasonality plot saved to {daily_path}")
        
        # Weekly seasonality
        plt.figure(figsize=(12, 6))
        forecast['weekday'] = forecast['ds'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly = forecast.groupby('weekday')['weekly'].mean().loc[weekday_order]
        weekly.plot(kind='line', marker='o', color='#d62728')
        plt.title("Weekly Consumption Pattern")
        plt.xlabel("Day of Week")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        weekly_path = os.path.join(os.getcwd(), 'weekly_seasonality.png')
        plt.savefig(weekly_path, dpi=300)
        plt.close()
        print(f" - Weekly seasonality plot saved to {weekly_path}")
        
        # Monthly seasonality
        plt.figure(figsize=(12, 6))
        forecast['month'] = forecast['ds'].dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly = forecast.groupby('month')['yearly'].mean().reindex(month_order)
        monthly.plot(kind='line', marker='D', color='#9467bd', linewidth=2)
        plt.title("Monthly Consumption Pattern")
        plt.xlabel("Month")
        plt.ylabel("Seasonal Impact")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        monthly_path = os.path.join(os.getcwd(), 'monthly_seasonality.png')
        plt.savefig(monthly_path, dpi=300)
        plt.close()
        print(f" - Monthly seasonality plot saved to {monthly_path}")
    
    except Exception as e:
        print("\n[Step 6 Error] Seasonal plots failed:")
        traceback.print_exc()

def plot_historical_forecast(hist_fcst, future_fcst):
    """Plot comparison between historical and forecast data"""
    try:
        plt.figure(figsize=(18, 8))
        
        # Historical data
        plt.plot(hist_fcst['ds'], hist_fcst['yhat'], 
                label='2024 Fitted', color='#1f77b4', linewidth=1.5)
        
        # Future forecast
        plt.plot(future_fcst['ds'], future_fcst['yhat'], 
                label='2025 Forecast', color='#ff7f0e', linewidth=1.5, linestyle='--')
        
        # Formatting
        plt.title("2024 Historical vs 2025 Forecast", pad=20)
        plt.xlabel("Date", labelpad=15)
        plt.ylabel("Consumption (kWh)", labelpad=15)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = os.path.join(os.getcwd(), 'historical_forecast_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" - Historical vs Forecast plot saved to {output_path}")
    
    except Exception as e:
        print("\n[Error] Historical forecast plot failed:")
        traceback.print_exc()

# ======================
# Main Execution (Remaining functions unchanged)
# ======================
if __name__ == "__main__":
    print("\n=== Electricity Consumption Forecasting System ===")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Data pipeline
        df = load_and_preprocess_data()
        
        # Step 2a: Parameter tuning(Time conusming)
        # best_model, best_params = grid_search_parameters(df)
        
        # Step 2b: Train final model
        # final_model = train_prophet_model(df, best_params)
        final_model = train_prophet_model(df)
        # Validate and generate forecasts
        validate_model(final_model, df)
        hist_fcst, future_fcst = generate_forecasts(final_model, df)  # 现在函数已定义
        
        # Save results and plots
        hist_path = os.path.join(os.getcwd(), '2024_fitted.csv')
        future_path = os.path.join(os.getcwd(), '2025_forecast.csv')
        hist_fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(hist_path, index=False)
        future_fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(future_path, index=False)
        print("\n[Output Files]")
        print(f" - Historical fit: {hist_path}")
        print(f" - Future forecast: {future_path}")
        
        # Generate plots
        plot_validation(df, hist_fcst)
        plot_seasonal_components(hist_fcst)
        plot_historical_forecast(hist_fcst, future_fcst)
        
        # Final report
        print("\n=== Operation Successful ===")
        print("Generated Files:")
        print("1. validation_comparison.png - Actual vs Fitted with error metrics")
        print("2. daily_seasonality.png    - Hourly consumption patterns")
        print("3. weekly_seasonality.png   - Weekly patterns")
        print("4. monthly_seasonality.png - Monthly patterns")
        print("5. historical_forecast_comparison.png - 2024 vs 2025 comparison")
        print("6. 2024_fitted.csv          - Historical predictions")
        print("7. 2025_forecast.csv        - Future predictions")
    
    except Exception as e:
        print("\n!!! SYSTEM ERROR !!!")
        traceback.print_exc()
        print("\nTroubleshooting Guide:")
        print("1. Verify input file format and path")
        print("2. Check library versions (Prophet 1.1+, Pandas 1.0+)")
        print("3. Validate datetime format: 'DD/MM/YYYY HH:MM'")
        sys.exit(1)
