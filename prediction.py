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

    # Calculate sums for true hourly/weekly rate ratio calculations
    cell_sums = hist.groupby('h3_8')['violation_count'].sum().rename('cell_sum')
    cell_hour_sums = hist.groupby(['h3_8', 'hour'])['violation_count'].sum().rename('cell_hour_sum')
    cell_dow_sums = hist.groupby(['h3_8', 'day_of_week'])['violation_count'].sum().rename('cell_dow_sum')

    # Global sums for fallback if a cell lacks data for a particular hour/day
    global_sum = hist['violation_count'].sum()
    global_hour_sums = hist.groupby('hour')['violation_count'].sum().rename('global_hour_sum')
    global_dow_sums = hist.groupby('day_of_week')['violation_count'].sum().rename('global_dow_sum')

    # Merge sum columns into predictions
    calibrated = df.merge(cell_sums, on='h3_8', how='left')
    calibrated = calibrated.merge(cell_hour_sums, on=['h3_8', 'hour'], how='left')
    calibrated = calibrated.merge(cell_dow_sums, on=['h3_8', 'day_of_week'], how='left')
    calibrated = calibrated.merge(global_hour_sums, on='hour', how='left')
    calibrated = calibrated.merge(global_dow_sums, on='day_of_week', how='left')

    # Fill NaNs with 0
    calibrated['cell_sum'] = calibrated['cell_sum'].fillna(0.0)
    calibrated['cell_hour_sum'] = calibrated['cell_hour_sum'].fillna(0.0)
    calibrated['cell_dow_sum'] = calibrated['cell_dow_sum'].fillna(0.0)

    # Calculate factors
    # For cells with > 0 total violations:
    #   hour_factor = (cell_hour_sum / num_days) / (cell_sum / (num_days * 24)) = cell_hour_sum / (cell_sum / 24)
    #   dow_factor = (cell_dow_sum / num_weeks) / (cell_sum / (num_weeks * 7)) = cell_dow_sum / (cell_sum / 7)
    cell_has_violations = calibrated['cell_sum'] > 0

    # Local factors
    local_hour_factor = calibrated['cell_hour_sum'] / (calibrated['cell_sum'] / 24.0 + 1e-6)
    local_dow_factor = calibrated['cell_dow_sum'] / (calibrated['cell_sum'] / 7.0 + 1e-6)

    # Global fallback factors
    global_hour_factor = calibrated['global_hour_sum'] / (global_sum / 24.0 + 1e-6)
    global_dow_factor = calibrated['global_dow_sum'] / (global_sum / 7.0 + 1e-6)

    # Combine local and global (using global as fallback or smoothing for small cell volumes)
    # If a cell has very few violations, smooth towards the global temporal pattern
    smoothing_threshold = 20.0
    alpha = (calibrated['cell_sum'] / smoothing_threshold).clip(lower=0.0, upper=1.0)

    hour_factor = alpha * local_hour_factor + (1 - alpha) * global_hour_factor
    dow_factor = alpha * local_dow_factor + (1 - alpha) * global_dow_factor

    # Multiplicative combination of hour and day of week to respect off-peak hours
    combined_factor = hour_factor * dow_factor

    # Clip to a realistic dynamic range [0.01, 10.0] to capture high peaks and low nights
    calibrated['temporal_factor'] = combined_factor.clip(lower=0.01, upper=10.0)

    calibrated['pred_count'] = calibrated['pred_count'] * calibrated['temporal_factor']
    calibrated['pred_weighted_pce'] = calibrated['pred_weighted_pce'] * calibrated['temporal_factor']

    drop_cols = [
        'cell_sum', 'cell_hour_sum', 'cell_dow_sum',
        'global_hour_sum', 'global_dow_sum', 'temporal_factor'
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
