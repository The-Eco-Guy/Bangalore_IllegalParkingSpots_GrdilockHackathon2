import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os

def load_historical_data():
    """
    Load pre-aggregated historical data from Parquet cache.
    """
    if os.path.exists('historical_aggregated.parquet'):
        df = pd.read_parquet('historical_aggregated.parquet')
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    raise FileNotFoundError("historical_aggregated.parquet not found. Run preprocess_and_train.py first.")

def load_dbscan_violations():
    """
    Load individual violation coordinates for DBSCAN clustering from Parquet cache.
    """
    if os.path.exists('violations_dbscan.parquet'):
        df = pd.read_parquet('violations_dbscan.parquet')
        df['ist_dt'] = pd.to_datetime(df['ist_dt'])
        df['date'] = df['ist_dt'].dt.date
        return df
    raise FileNotFoundError("violations_dbscan.parquet not found. Run preprocess_and_train.py first.")

def load_cell_metadata():
    """
    Load precomputed cell metadata (centers, averages, mixes).
    """
    if os.path.exists('cell_metadata.parquet'):
        return pd.read_parquet('cell_metadata.parquet')
    raise FileNotFoundError("cell_metadata.parquet not found. Run preprocess_and_train.py first.")

def get_hours_for_bucket(time_bucket):
    """
    Get the list of hours represented by each user-friendly time bucket.
    """
    if time_bucket == 'AM Peak (07:00 - 10:59)':
        return list(range(7, 11))
    elif time_bucket == 'Midday (11:00 - 16:59)':
        return list(range(11, 17))
    elif time_bucket == 'PM Peak (17:00 - 21:59)':
        return list(range(17, 22))
    elif time_bucket == 'Off-Peak Night (22:00 - 06:59)':
        return [22, 23, 0, 1, 2, 3, 4, 5, 6]
    return []

def filter_aggregated_data(df, start_date=None, end_date=None, time_bucket=None, hour=None):
    """
    Filter aggregated historical cell-hour data dynamically.
    """
    filtered = df.copy()
    
    # 1. Date filter
    if start_date is not None:
        filtered = filtered[filtered['date'] >= start_date]
    if end_date is not None:
        filtered = filtered[filtered['date'] <= end_date]
        
    # 2. Hour filter
    if hour is not None and hour != "All":
        filtered = filtered[filtered['hour'] == int(hour)]
    elif time_bucket is not None and time_bucket != "All":
        hours_list = get_hours_for_bucket(time_bucket)
        filtered = filtered[filtered['hour'].isin(hours_list)]
        
    # Aggregate back to H3 cells for the filtered period
    agg_df = filtered.groupby('h3_8').agg(
        violations=('violation_count', 'sum'),
        weighted_pce=('weighted_pce', 'sum'),
        lat=('lat_mean', 'mean'),
        lon=('lon_mean', 'mean')
    ).reset_index()
    
    return agg_df

def filter_dbscan_violations(df, start_date=None, end_date=None, time_bucket=None, hour=None):
    """
    Filter individual violations for clustering.
    """
    filtered = df.copy()
    
    # Date filter
    if start_date is not None:
        filtered = filtered[filtered['date'] >= start_date]
    if end_date is not None:
        filtered = filtered[filtered['date'] <= end_date]
        
    # Hour filter
    if hour is not None and hour != "All":
        filtered = filtered[filtered['hour'] == int(hour)]
    elif time_bucket is not None and time_bucket != "All":
        hours_list = get_hours_for_bucket(time_bucket)
        filtered = filtered[filtered['hour'].isin(hours_list)]
        
    return filtered

def run_dbscan_clustering(violations_df, eps_meters=200, min_samples=30):
    """
    Run DBSCAN clustering on latitude/longitude in radians using haversine metric.
    """
    if len(violations_df) < min_samples:
        violations_df = violations_df.copy()
        violations_df['cluster'] = -1
        return violations_df
        
    coords_rad = np.radians(violations_df[['latitude', 'longitude']].values)
    eps_rad = eps_meters / 6371000.0  # earth radius in meters
    
    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric='haversine',
        algorithm='ball_tree'
    )
    
    violations_df = violations_df.copy()
    violations_df['cluster'] = db.fit_predict(coords_rad)
    return violations_df
