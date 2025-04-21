# consumption_graphs.py
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import traceback
from itertools import product
from prophet.diagnostics import cross_validation, performance_metrics
import joypy  # 需安装：pip install joypy

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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found")
        df = pd.read_csv(file_path)
        df = df.rename(columns={'date': 'ds', 'electricity': 'y'})
        df = df.dropna(subset=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'], format="%d/%m/%Y %H:%M", errors='coerce')
        df = df.dropna(subset=['ds'])
        df = df[(df['ds'] >= '2024-01-01') & (df['ds'] <= '2024-12-31 23:00:00')]
        max_consumption = df['y'].max()
        df['cap'] = max_consumption * 1.2
        return df
    except Exception:
        traceback.print_exc()
        sys.exit(1)

# ======================
# 2. Holiday Definitions
# ======================

def get_holidays():
    """Define school holidays for Prophet"""
    holidays = pd.DataFrame({
        'holiday': 'school_holiday',
        'ds': pd.to_datetime([
            '2024-08-12','2024-08-21','2024-09-26','2024-10-17',
            '2024-11-28','2024-12-18','2025-01-05','2024-02-28',
            '2024-04-03','2024-05-18'
        ]),
        'lower_window': -2,
        'upper_window': 3
    })
    return holidays

# ======================
# 3. Model Training
# ======================

def train_prophet_model(df):
    """Train Prophet model with predefined parameters"""
    holidays_df = get_holidays()
    model = Prophet(
        holidays=holidays_df,
        growth='logistic',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=25.0,
        holidays_prior_scale=5.0,
        seasonality_mode='multiplicative',
        n_changepoints=25,
        yearly_seasonality=True
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=6)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
    model.fit(df)
    return model

# ======================
# 4. Forecast and Validation
# ======================

def generate_and_validate(model, df):
    hist = model.predict(df)
    future = model.make_future_dataframe(periods=8760, freq='h')
    future['cap'] = model.history['cap'].max()
    fut = model.predict(future)
    return hist, fut

# ======================
# 5. Visualization Functions
# ======================

def plot_contour_plots(forecast):
    """Generate contour plots for daily, weekly, and monthly components"""
    df_fc = forecast.copy()
    df_fc['dayofyear'] = df_fc['ds'].dt.dayofyear
    for comp in ['daily', 'weekly', 'monthly']:
        pivot = df_fc.pivot_table(values='yhat', index='dayofyear', columns=comp, aggfunc='mean')
        pivot = pivot.sort_index(axis=1).ffill(axis=1).ffill(axis=0)
        X, Y = np.meshgrid(pivot.columns, pivot.index)
        Z = pivot.values
        plt.figure(figsize=(10,6))
        plt.contourf(X, Y, Z, cmap='viridis', levels=20)
        plt.colorbar(label='Predicted Consumption')
        plt.xlabel(f'{comp.capitalize()} Seasonality Strength')
        plt.ylabel('Day of Year')
        plt.title(f'Contour: {comp.capitalize()} vs Day of Year')
        plt.tight_layout()
        plt.savefig(f'contour_{comp}.png', dpi=300)
        plt.close()


def plot_ridge_daily(df):
    """Ridge plot: consumption distribution by hour of day"""
    df['hour'] = df['ds'].dt.hour
    data = [df[df['hour']==h]['y'].values for h in range(24)]
    labels = [f'{h}:00' for h in range(24)]
    joypy.joyplot(data=data, labels=labels, figsize=(10,8), colormap=plt.cm.viridis,
                  title='Daily Ridge: Hourly Consumption Distribution')
    plt.xlabel('Consumption (kWh)')
    plt.savefig('ridge_daily.png', dpi=300)
    plt.close()


def plot_ridge_weekly(df):
    """Ridge plot: consumption distribution by weekday"""
    df['weekday'] = df['ds'].dt.day_name()
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    data = [df[df['weekday']==day]['y'].values for day in order]
    joypy.joyplot(data=data, labels=order, figsize=(10,8), colormap=plt.cm.viridis,
                  title='Weekly Ridge: Consumption by Weekday')
    plt.xlabel('Consumption (kWh)')
    plt.savefig('ridge_weekly.png', dpi=300)
    plt.close()


def plot_ridge_monthly(df):
    """Ridge plot: consumption distribution by month"""
    df['month'] = df['ds'].dt.month
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    data = [df[df['month']==i]['y'].values for i in range(1,13)]
    joypy.joyplot(data=data, labels=labels, figsize=(10,8), colormap=plt.cm.viridis,
                  title='Monthly Ridge: Consumption by Month')
    plt.xlabel('Consumption (kWh)')
    plt.savefig('ridge_monthly.png', dpi=300)
    plt.close()


def plot_mirror_bar(df, hist):
    """Mirror bar chart with annotations"""
    df['month'] = df['ds'].dt.month
    actual = df.groupby('month')['y'].mean().reindex(range(1,13), fill_value=0)
    hist['month'] = hist['ds'].dt.month
    predicted = hist.groupby('month')['yhat'].mean().reindex(range(1,13), fill_value=0)
    actual_arr = actual.values
    pred_arr = predicted.values
    months = [f'{m:02d}' for m in range(1,13)]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.barh(x - 0.2, actual_arr, height=0.4, label='Actual', align='center')
    ax.barh(x + 0.2, pred_arr, height=0.4, label='Predicted', align='center')
    for i, (act, pred) in enumerate(zip(actual_arr, pred_arr)):
        ax.text(act + 1, i - 0.2, f'{act:.1f}', va='center')
        ax.text(pred + 1, i + 0.2, f'{pred:.1f}', va='center')
    ax.set_yticks(x)
    ax.set_yticklabels(months)
    ax.set_xlabel('Avg Consumption (kWh)')
    ax.set_title('Mirror Bar Chart: Actual vs Predicted (2024)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('mirror_bar_annotated.png', dpi=300)
    plt.close()

# ======================
# 6. Main Execution
# ======================
if __name__ == "__main__":
    df = load_and_preprocess_data()
    model = train_prophet_model(df)
    hist_fcst, fut_fcst = generate_and_validate(model, df)
    plot_contour_plots(hist_fcst)
    plot_ridge_daily(df)
    plot_ridge_weekly(df)
    plot_ridge_monthly(df)
    plot_mirror_bar(df, hist_fcst)
    print("Plots generated:")
    print(" - contour_daily.png, contour_weekly.png, contour_monthly.png")
    print(" - ridge_daily.png, ridge_weekly.png, ridge_monthly.png")
    print(" - mirror_bar_annotated.png")
