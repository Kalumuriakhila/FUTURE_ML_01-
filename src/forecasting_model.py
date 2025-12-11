import pandas as pd
import numpy as np
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

def train_prophet_model(df, yearly_seasonality=True, weekly_seasonality=True):
    """
    Train a Prophet forecasting model.
    
    Args:
        df: DataFrame with 'ds' (datetime) and 'y' (sales) columns
        yearly_seasonality: Include yearly seasonality
        weekly_seasonality: Include weekly seasonality
    
    Returns:
        Trained Prophet model
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df)
    
    return model

def forecast_sales(model, periods=30):
    """
    Generate forecast for specified number of periods.
    
    Args:
        model: Trained Prophet model
        periods: Number of days to forecast (default: 30)
    
    Returns:
        DataFrame with forecast data
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast

def get_forecast_summary(forecast, periods=30):
    """
    Get summary of the forecast.
    
    Args:
        forecast: Forecast DataFrame from Prophet
        periods: Number of forecast periods to summarize
    
    Returns:
        Dictionary with forecast summary statistics
    """
    future_forecast = forecast.tail(periods)
    
    summary = {
        'avg_forecasted_sales': round(future_forecast['yhat'].mean(), 2),
        'min_forecasted_sales': round(future_forecast['yhat'].min(), 2),
        'max_forecasted_sales': round(future_forecast['yhat'].max(), 2),
        'forecast_trend': 'Increasing' if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0] else 'Decreasing'
    }
    
    return summary

def get_model_components(model):
    """
    Get trend, seasonality, and other components info.
    
    Args:
        model: Trained Prophet model
    
    Returns:
        Dictionary with component info
    """
    return {
        'trend': 'Linear with changepoints',
        'yearly_seasonality': model.yearly_seasonality,
        'weekly_seasonality': model.weekly_seasonality,
        'daily_seasonality': model.daily_seasonality
    }
