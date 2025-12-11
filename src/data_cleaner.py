import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    """Load CSV data from file."""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Clean the data:
    - Convert date column to datetime
    - Handle missing values
    - Sort by date
    """
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Handle missing values by forward fill then backward fill
    df['sales'] = df['sales'].fillna(method='ffill').fillna(method='bfill')
    
    # Remove any remaining NaN values
    df = df.dropna()
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def prepare_for_prophet(df):
    """
    Prepare data for Prophet model.
    Prophet requires columns: ds (datetime) and y (target)
    """
    prophet_df = df[['date', 'sales']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def get_data_summary(df):
    """Get summary statistics of the data."""
    summary = {
        'total_records': len(df),
        'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}",
        'avg_sales': round(df['sales'].mean(), 2),
        'min_sales': df['sales'].min(),
        'max_sales': df['sales'].max(),
        'std_sales': round(df['sales'].std(), 2),
        'missing_values': df.isnull().sum().sum()
    }
    return summary
