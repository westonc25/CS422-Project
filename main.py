import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from datetime import datetime as dt


def load_data(file_path):
    """Load dataset from NYC Traffic volume data and preprocess it."""
    
    "Read the CSV file into the DataFrame"
    df = pd.read_csv(file_path)
    
    "Convert to a readable datetime format"
    df['date_time'] = pd.to_datetime(
        df[['Yr', 'M', 'D', 'HH', 'MM']].rename(
            columns={'Yr': 'year', 'M': 'month', 'D': 'day', 'HH': 'hour', 'MM': 'minute'})
    )

    "Keep only relevant columns"
    df = df[['date_time', 'Vol', 'SegmentID', 'Boro', 'Direction', 'street', 'fromSt', 'toSt']].copy()

    "Sort chronologically"
    df = df.sort_values(['SegmentID', 'Direction', 'date_time']).reset_index(drop=True)
    
    return df


def split_data_by_date(df):

    "The cutoff date for splitting the dataset"
    cutoff_date = pd.to_datetime('2024-09-01')

    "Logic for splitting the dataset based on the cutoff date"
    train_df = df[df['date_time'] < cutoff_date].copy()
    test_df = df[df['date_time'] >= cutoff_date].copy()

    return train_df, test_df

def create_features(df):
    """Several feature engineering steps to enhance the dataset."""
    
    # Time features
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek      # 0 = Monday
    df['month'] = df['date_time'].dt.month
    df['is_weekend'] = df['dayofweek'] >= 5
    df['is_rush_hour'] = df['hour'].isin([7,8,9,16,17,18]) & (~df['is_weekend'])

    # Rolling statistics
    rolls = [4, 8, 24, 96]  # last 1h, 2h, 6h, 24h
    for window in rolls:
        df[f'vol_roll_mean_{window}'] = df.groupby('SegmentID')['Vol'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'vol_roll_std_{window}']  = df.groupby('SegmentID')['Vol'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std())
        df[f'vol_roll_max_{window}']  = df.groupby('SegmentID')['Vol'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max())

    return df