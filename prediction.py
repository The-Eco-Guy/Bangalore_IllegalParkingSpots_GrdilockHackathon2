import pandas as pd
import pickle
import os

def _apply_temporal_calibration(df):
    """
    Re-introduce realistic hour/day variation into cached forecasts using
    historical per-cell temporal patterns. The original forecast cache was
    generated with flat lag proxies, which makes adjacent hours and dates look
    almost identical on the map.
    """
    hist_path = 'historical_aggregated.parquet'
    if not os.path.exists(hist_path) or df.empty:
        return df

    hist = pd.read_parquet(hist_path)
    hist['date'] = pd.to_datetime(hist['date'])
    hist['day_of_week'] = hist['date'].dt.dayofweek

    cell_base = (
        hist.groupby('h3_8', as_index=False)['violation_count']
        .mean()
        .rename(columns={'violation_count': 'cell_hist_mean'})
    )
    cell_hour = (
        hist.groupby(['h3_8', 'hour'], as_index=False)['violation_count']
        .mean()
        .rename(columns={'violation_count': 'cell_hour_mean'})
    )
    cell_dow = (
        hist.groupby(['h3_8', 'day_of_week'], as_index=False)['violation_count']
        .mean()
        .rename(columns={'violation_count': 'cell_dow_mean'})
    )
    global_hour = (
        hist.groupby('hour', as_index=False)['violation_count']
        .mean()
        .rename(columns={'violation_count': 'global_hour_mean'})
    )
    global_dow = (
        hist.groupby('day_of_week', as_index=False)['violation_count']
        .mean()
        .rename(columns={'violation_count': 'global_dow_mean'})
    )

    global_base = max(hist['violation_count'].mean(), 1e-6)

    calibrated = df.merge(cell_base, on='h3_8', how='left')
    calibrated = calibrated.merge(cell_hour, on=['h3_8', 'hour'], how='left')
    calibrated = calibrated.merge(cell_dow, on=['h3_8', 'day_of_week'], how='left')
    calibrated = calibrated.merge(global_hour, on='hour', how='left')
    calibrated = calibrated.merge(global_dow, on='day_of_week', how='left')

    calibrated['cell_hist_mean'] = calibrated['cell_hist_mean'].fillna(global_base)
    calibrated['cell_hour_mean'] = calibrated['cell_hour_mean'].fillna(calibrated['global_hour_mean']).fillna(calibrated['cell_hist_mean'])
    calibrated['cell_dow_mean'] = calibrated['cell_dow_mean'].fillna(calibrated['global_dow_mean']).fillna(calibrated['cell_hist_mean'])

    hour_factor = calibrated['cell_hour_mean'] / calibrated['cell_hist_mean'].clip(lower=1e-6)
    dow_factor = calibrated['cell_dow_mean'] / calibrated['cell_hist_mean'].clip(lower=1e-6)

    # Blend local and global temporal effects and keep multipliers bounded.
    combined_factor = (0.7 * hour_factor) + (0.3 * dow_factor)
    calibrated['temporal_factor'] = combined_factor.clip(lower=0.6, upper=1.8)

    calibrated['pred_count'] = calibrated['pred_count'] * calibrated['temporal_factor']
    calibrated['pred_weighted_pce'] = calibrated['pred_weighted_pce'] * calibrated['temporal_factor']

    drop_cols = [
        'cell_hist_mean', 'cell_hour_mean', 'cell_dow_mean',
        'global_hour_mean', 'global_dow_mean', 'temporal_factor'
    ]
    return calibrated.drop(columns=drop_cols, errors='ignore')

def load_predictions_7d():
    """
    Load precomputed 7-day hourly forecast from Parquet.
    """
    if os.path.exists('predictions_7d.parquet'):
        df = pd.read_parquet('predictions_7d.parquet')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['datetime'].dt.dayofweek
        df = _apply_temporal_calibration(df)
        return df
    raise FileNotFoundError("predictions_7d.parquet not found. Run preprocess_and_train.py first.")

def load_meta():
    """
    Load trained model metadata (MAE, RMSE, R2, AUC, features, label encoder).
    """
    if os.path.exists('model_meta.pkl'):
        with open('model_meta.pkl', 'rb') as f:
            return pickle.load(f)
    raise FileNotFoundError("model_meta.pkl not found. Run preprocess_and_train.py first.")

def filter_predictions(df, date_str=None, hour=None, time_bucket=None):
    """
    Filter predictions by date, hour, or time bucket.
    When multiple forecast hours are selected, return average per-hour values
    so map intensity reflects the chosen forecast window rather than just the
    number of hours included in that window.
    """
    filtered = df.copy()
    
    # 1. Date filter
    if date_str is not None:
        target_date = pd.to_datetime(date_str).date()
        filtered = filtered[filtered['date'] == target_date]
        
    # 2. Hour / Time Bucket filter
    if hour is not None and hour != "All":
        filtered = filtered[filtered['hour'] == int(hour)]
    elif time_bucket is not None and time_bucket != "All":
        if 'AM Peak' in time_bucket or time_bucket == 'AM_PEAK':
            filtered = filtered[filtered['hour'].between(7, 10)]
        elif 'Midday' in time_bucket or time_bucket == 'MIDDAY':
            filtered = filtered[filtered['hour'].between(11, 16)]
        elif 'PM Peak' in time_bucket or time_bucket == 'PM_PEAK':
            filtered = filtered[filtered['hour'].between(17, 21)]
        elif 'Off-Peak' in time_bucket or time_bucket == 'OFF_PEAK_NIGHT':
            filtered = filtered[(filtered['hour'] >= 22) | (filtered['hour'] <= 6)]
            
    # Aggregate by H3 cell for the selected timeframe.
    # Use mean values so predictive maps remain comparable across hour buckets.
    agg_df = filtered.groupby('h3_8').agg(
        pred_count=('pred_count', 'mean'),
        pred_weighted_pce=('pred_weighted_pce', 'mean'),
        hotspot_prob=('hotspot_prob', 'mean'),
        lat=('lat_mean', 'mean'),
        lon=('lon_mean', 'mean')
    ).reset_index()
    
    # Compute hotspot flag based on averaged probability
    agg_df['hotspot_flag'] = (agg_df['hotspot_prob'] >= 0.5).astype(int)
    
    return agg_df
