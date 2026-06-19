import pandas as pd
import numpy as np
import h3
import ast
import pickle
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, classification_report

print("🚀 Starting data preprocessing and model training pipeline...")

RAW_DATA_PARQUET = "jan to may police violation_anonymized791b166.parquet"
RAW_DATA_CSV = "jan to may police violation_anonymized791b166.csv"


def load_raw_data():
    """
    Load the raw dataset from Parquet when available, otherwise fall back to CSV.
    """
    if os.path.exists(RAW_DATA_PARQUET):
        print(f"   Using Parquet source: {RAW_DATA_PARQUET}")
        return pd.read_parquet(RAW_DATA_PARQUET)
    if os.path.exists(RAW_DATA_CSV):
        print(f"   Using CSV source: {RAW_DATA_CSV}")
        return pd.read_csv(RAW_DATA_CSV)
    raise FileNotFoundError(
        f"Missing raw dataset. Expected {RAW_DATA_PARQUET} or {RAW_DATA_CSV}."
    )

# 1. Load and Clean Raw Data
print("1. Loading raw violations data...")
df = load_raw_data()
print(f"   Raw records: {len(df):,}")

# Drop rows with missing lat/lon
df = df.dropna(subset=['latitude', 'longitude'])
print(f"   Records with coordinates: {len(df):,}")

# Parse datetimes & convert to IST
print("   Converting timestamps to IST...")
df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='ISO8601', utc=True)
df['ist_dt'] = df['created_datetime'].dt.tz_convert('Asia/Kolkata')
df['date'] = df['ist_dt'].dt.date
df['hour'] = df['ist_dt'].dt.hour
df['day_of_week'] = df['ist_dt'].dt.dayofweek
df['month'] = df['ist_dt'].dt.month
df['week_of_year'] = df['ist_dt'].dt.isocalendar().week.astype(int)

# Assign H3 Resolution 8 Hex Cells
print("   Assigning H3 Resolution 8 cells...")
df['h3_8'] = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(df['latitude'], df['longitude'])]
print(f"   Unique H3 res-8 cells: {df['h3_8'].nunique()}")

# Setup Time Buckets
def assign_time_bucket(h):
    if 7 <= h <= 10:    return 'AM_PEAK'
    elif 11 <= h <= 16: return 'MIDDAY'
    elif 17 <= h <= 21: return 'PM_PEAK'
    else:               return 'OFF_PEAK_NIGHT'
df['time_bucket'] = df['hour'].apply(assign_time_bucket)

# Derive primary violation type and violation count per record
print("   Extracting violation types...")
def parse_primary_violation(v):
    try:
        lst = ast.literal_eval(v)
        return lst[0] if lst else 'UNKNOWN'
    except:
        return 'UNKNOWN'

df['primary_violation'] = df['violation_type'].apply(parse_primary_violation)
df['violation_type_count'] = df['violation_type'].apply(
    lambda v: len(ast.literal_eval(v)) if pd.notna(v) else 1
)

# Vehicle type cleaning
top_vehicles = df['vehicle_type'].value_counts().head(6).index.tolist()
df['vehicle_type_clean'] = df['vehicle_type'].where(df['vehicle_type'].isin(top_vehicles), 'OTHER')

# 2. Congestion Impact Scoring
print("2. Computing PCE weights and Location Factors...")
# Standard vehicle PCE weights (IRC:106-1990)
IRC_PCE = {
    'SCOOTER': 0.5, 'MOTOR CYCLE': 0.5, 'MOPED': 0.5,
    'CAR': 1.0, 'JEEP': 1.0, 'VAN': 1.0,
    'PASSENGER AUTO': 0.75,
    'GOODS AUTO': 1.4, 'LGV': 1.4, 'TEMPO': 1.4, 'MINI LORRY': 1.4,
    'MAXI-CAB': 2.0,
    'PRIVATE BUS': 3.7, 'BUS (BMTC/KSRTC)': 3.7, 'TOURIST BUS': 3.7,
    'FACTORY BUS': 3.7, 'SCHOOL VEHICLE': 3.7, 'HGV': 3.7,
    'LORRY/GOODS VEHICLE': 3.7, 'TANKER': 3.7,
    'TRACTOR': 4.0, 'OTHERS': 1.0
}

def get_pce(v):
    if pd.isna(v): return 1.0
    v = str(v).upper()
    for key, pce in IRC_PCE.items():
        if key in v: return pce
    return 1.0

df['pce'] = df['vehicle_type'].apply(get_pce)

# Refined location factors (HCM Chapter 18)
def get_loc_factor_refined(row):
    j_name = row['junction_name']
    if pd.notna(j_name) and str(j_name).strip() != '' and str(j_name).upper() != 'NAN':
        return 1.50

    v = row['location']
    try:
        items = [x.upper() for x in ast.literal_eval(v)]
    except:
        items = [str(v).upper()]

    for item in items:
        if any(k in item for k in ['JUNCTION','INTERSECTION','SIGNAL','ZEBRA']):
            return 1.50
        if any(k in item for k in ['MAIN ROAD','ARTERIAL','NH','NATIONAL']):
            return 1.30

    return 1.0

df['loc_factor'] = df.apply(get_loc_factor_refined, axis=1)
df['pce_weighted'] = df['pce'] * df['loc_factor']

# 3. Aggregate to H3 Cell × Date × Hour Buckets
print("3. Aggregating to Cell-Hour buckets...")
agg = df.groupby(['h3_8', 'date', 'hour']).agg(
    violation_count   = ('id', 'count'),
    weighted_pce      = ('pce_weighted', 'sum'),
    lat_mean          = ('latitude', 'mean'),
    lon_mean          = ('longitude', 'mean'),
    day_of_week       = ('day_of_week', 'first'),
    month             = ('month', 'first'),
    week_of_year      = ('week_of_year', 'first'),
    # Vehicle mix
    pct_car           = ('vehicle_type_clean', lambda x: (x == 'CAR').mean()),
    pct_scooter       = ('vehicle_type_clean', lambda x: (x == 'SCOOTER').mean()),
    pct_auto          = ('vehicle_type_clean', lambda x: (x == 'PASSENGER AUTO').mean()),
    pct_maxi          = ('vehicle_type_clean', lambda x: (x == 'MAXI-CAB').mean()),
    # Violation mix
    pct_wrong_park    = ('primary_violation', lambda x: (x == 'WRONG PARKING').mean()),
    pct_no_park       = ('primary_violation', lambda x: (x == 'NO PARKING').mean()),
    pct_main_road     = ('primary_violation', lambda x: (x == 'PARKING IN A MAIN ROAD').mean()),
    pct_footpath      = ('primary_violation', lambda x: (x == 'PARKING ON FOOTPATH').mean()),
    avg_vtype_count   = ('violation_type_count', 'mean'),
).reset_index()

print(f"   Aggregated shape: {agg.shape}")

# 4. Feature Engineering
print("4. Feature engineering lag and rolling features...")
agg = agg.sort_values(['h3_8', 'date', 'hour']).reset_index(drop=True)
agg['datetime'] = pd.to_datetime(agg['date'].astype(str)) + pd.to_timedelta(agg['hour'], unit='h')

# Compute lag & rolling features per cell
grp = agg.groupby('h3_8')['violation_count']
agg['lag_1h'] = grp.shift(1)
agg['lag_2h'] = grp.shift(2)
agg['lag_3h'] = grp.shift(3)
agg['lag_24h'] = grp.shift(24)
agg['lag_168h'] = grp.shift(168)

agg['rolling_3h_mean'] = grp.shift(1).groupby(agg['h3_8']).transform(lambda x: x.rolling(3, min_periods=1).mean())
agg['rolling_6h_mean'] = grp.shift(1).groupby(agg['h3_8']).transform(lambda x: x.rolling(6, min_periods=1).mean())
agg['rolling_24h_mean'] = grp.shift(1).groupby(agg['h3_8']).transform(lambda x: x.rolling(24, min_periods=1).mean())
agg['rolling_7d_mean'] = grp.shift(1).groupby(agg['h3_8']).transform(lambda x: x.rolling(168, min_periods=1).mean())
agg['rolling_24h_std'] = grp.shift(1).groupby(agg['h3_8']).transform(lambda x: x.rolling(24, min_periods=2).std())

# Temporal encodings
agg['hour_sin']   = np.sin(2 * np.pi * agg['hour'] / 24)
agg['hour_cos']   = np.cos(2 * np.pi * agg['hour'] / 24)
agg['dow_sin']    = np.sin(2 * np.pi * agg['day_of_week'] / 7)
agg['dow_cos']    = np.cos(2 * np.pi * agg['day_of_week'] / 7)
agg['month_sin']  = np.sin(2 * np.pi * agg['month'] / 12)
agg['month_cos']  = np.cos(2 * np.pi * agg['month'] / 12)

# Convenience flags
agg['is_weekend']   = (agg['day_of_week'] >= 5).astype(int)
agg['is_peak_hour'] = (agg['hour'].between(8, 11) | agg['hour'].between(17, 20)).astype(int)
agg['is_morning']   = (agg['hour'].between(5, 11)).astype(int)
agg['is_night']     = ((agg['hour'] >= 22) | (agg['hour'] <= 5)).astype(int)

# H3 spatial encoding
le = LabelEncoder()
agg['h3_cell_id'] = le.fit_transform(agg['h3_8'])
cell_hist_mean = agg.groupby('h3_8')['violation_count'].transform('mean')
agg['cell_hist_mean'] = cell_hist_mean

# Classification target
HOTSPOT_THRESHOLD = 10
agg['is_hotspot'] = (agg['violation_count'] >= HOTSPOT_THRESHOLD).astype(int)

# Drop rows with NaN lags for training
LAG_COLS = ['lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'rolling_3h_mean', 'rolling_24h_mean']
agg_clean = agg.dropna(subset=LAG_COLS).copy()
print(f"   Cleaned aggregation shape: {agg_clean.shape}")

# 5. Chronological Train / Val / Test Split
TRAIN_END = pd.Timestamp('2024-02-01')
VAL_END   = pd.Timestamp('2024-03-01')

train = agg_clean[agg_clean['datetime'] <  TRAIN_END]
val   = agg_clean[(agg_clean['datetime'] >= TRAIN_END) & (agg_clean['datetime'] < VAL_END)]
test  = agg_clean[agg_clean['datetime'] >= VAL_END]

print(f"   Train: {len(train):,} rows | Val: {len(val):,} rows | Test: {len(test):,} rows")

FEATURES = [
    'h3_cell_id', 'lat_mean', 'lon_mean', 'cell_hist_mean',
    'hour', 'day_of_week', 'month', 'week_of_year',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_weekend', 'is_peak_hour', 'is_morning', 'is_night',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
    'rolling_3h_mean', 'rolling_6h_mean', 'rolling_24h_mean', 'rolling_7d_mean', 'rolling_24h_std',
    'pct_car', 'pct_scooter', 'pct_auto', 'pct_maxi',
    'pct_wrong_park', 'pct_no_park', 'pct_main_road', 'pct_footpath',
    'avg_vtype_count',
]

X_train, y_train_reg, y_train_clf = train[FEATURES], train['violation_count'], train['is_hotspot']
X_val, y_val_reg, y_val_clf = val[FEATURES], val['violation_count'], val['is_hotspot']
X_test, y_test_reg, y_test_clf = test[FEATURES], test['violation_count'], test['is_hotspot']

# 6. Train Models
print("6. Training XGBoost Regressor...")
xgb_reg = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    objective='reg:squarederror',
    eval_metric='mae',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_reg.fit(X_train, y_train_reg, eval_set=[(X_val, y_val_reg)], verbose=False)

# Evaluate regressor
preds_reg = np.clip(xgb_reg.predict(X_test), 0, None)
mae = mean_absolute_error(y_test_reg, preds_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, preds_reg))
r2 = r2_score(y_test_reg, preds_reg)
print(f"   Regressor Test Set Metrics -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

print("   Training XGBoost Classifier...")
neg_pos_ratio = (y_train_clf == 0).sum() / (y_train_clf == 1).sum()
xgb_clf = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg_pos_ratio,
    objective='binary:logistic',
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_clf.fit(X_train, y_train_clf, eval_set=[(X_val, y_val_clf)], verbose=False)

# Evaluate classifier
probs_clf = xgb_clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test_clf, probs_clf)
print(f"   Classifier Test Set Metrics -> AUC-ROC: {auc:.3f}")

# Save models & metadata
print("   Saving trained models and meta to disk...")
with open('xgb_regressor.pkl', 'wb') as f:
    pickle.dump(xgb_reg, f)
with open('xgb_classifier.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)
with open('model_meta.pkl', 'wb') as f:
    pickle.dump({
        'features': FEATURES,
        'h3_resolution': 8,
        'hotspot_threshold': HOTSPOT_THRESHOLD,
        'label_encoder': le,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'auc': auc
    }, f)

# 7. Precompute and Cache Cell Metadata
print("7. Precomputing cell metadata...")
# Get unique cells and their centers
cell_centers = df.groupby('h3_8').agg(
    lat_center=('latitude', 'mean'),
    lon_center=('longitude', 'mean')
).reset_index()

# Historical PCE factors
cell_pce_stats = df.groupby('h3_8').agg(
    total_violations=('id', 'count'),
    sum_pce_weighted=('pce_weighted', 'sum')
).reset_index()
cell_pce_stats['avg_pce_weighted'] = cell_pce_stats['sum_pce_weighted'] / cell_pce_stats['total_violations']

# Historical mix stats
cell_mix_stats = agg.groupby('h3_8').agg(
    pct_car=('pct_car', 'mean'),
    pct_scooter=('pct_scooter', 'mean'),
    pct_auto=('pct_auto', 'mean'),
    pct_maxi=('pct_maxi', 'mean'),
    pct_wrong_park=('pct_wrong_park', 'mean'),
    pct_no_park=('pct_no_park', 'mean'),
    pct_main_road=('pct_main_road', 'mean'),
    pct_footpath=('pct_footpath', 'mean'),
    avg_vtype_count=('avg_vtype_count', 'mean'),
    cell_hist_mean=('cell_hist_mean', 'first'),
    h3_cell_id=('h3_cell_id', 'first')
).reset_index()

cell_meta = cell_centers.merge(cell_pce_stats, on='h3_8').merge(cell_mix_stats, on='h3_8')
cell_meta.to_parquet('cell_metadata.parquet')
print(f"   Saved cell metadata of size: {len(cell_meta)}")

# 8. Save Processed Historical Data (Compact Parquet)
print("8. Saving processed historical data to parquet...")
hist_to_save = agg[['h3_8', 'date', 'hour', 'violation_count', 'weighted_pce', 'lat_mean', 'lon_mean']].copy()
hist_to_save['date'] = hist_to_save['date'].astype(str)
hist_to_save.to_parquet('historical_aggregated.parquet')

# Save cleaned violations for DBSCAN
violations_dbscan = df[['ist_dt', 'latitude', 'longitude', 'primary_violation', 'time_bucket', 'hour']].copy()
violations_dbscan['ist_dt'] = violations_dbscan['ist_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
violations_dbscan.to_parquet('violations_dbscan.parquet')
print("   Saved historical aggregates and violations parquet files.")

# 9. Generate 7-Day Forecast (2024-04-09 to 2024-04-15)
print("9. Generating 7-day hourly forecast...")
last_date = df['date'].max()
print(f"   Last known date in dataset: {last_date}")

# Future dates: 7 days
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
all_cells = cell_meta['h3_8'].tolist()

forecast_rows = []
for f_date in future_dates:
    for hr in range(24):
        f_dt = pd.Timestamp(f_date.date()) + pd.Timedelta(hours=hr)
        
        # Build features for all cells for this hour
        hr_features = []
        for cell_row in cell_meta.itertuples():
            # Build feature dictionary
            row_dict = {
                'h3_cell_id': cell_row.h3_cell_id,
                'lat_mean': cell_row.lat_center,
                'lon_mean': cell_row.lon_center,
                'cell_hist_mean': cell_row.cell_hist_mean,
                'hour': hr,
                'day_of_week': f_dt.dayofweek,
                'month': f_dt.month,
                'week_of_year': int(f_dt.isocalendar()[1]),
                'hour_sin': np.sin(2 * np.pi * hr / 24),
                'hour_cos': np.cos(2 * np.pi * hr / 24),
                'dow_sin': np.sin(2 * np.pi * f_dt.dayofweek / 7),
                'dow_cos': np.cos(2 * np.pi * f_dt.dayofweek / 7),
                'month_sin': np.sin(2 * np.pi * f_dt.month / 12),
                'month_cos': np.cos(2 * np.pi * f_dt.month / 12),
                'is_weekend': int(f_dt.dayofweek >= 5),
                'is_peak_hour': int(hr in [8, 9, 10, 11, 17, 18, 19, 20]),
                'is_morning': int(5 <= hr <= 11),
                'is_night': int(hr >= 22 or hr <= 5),
                # Since future lags are unknown, use cell historical mean as standard proxy
                'lag_1h': cell_row.cell_hist_mean,
                'lag_2h': cell_row.cell_hist_mean,
                'lag_3h': cell_row.cell_hist_mean,
                'lag_24h': cell_row.cell_hist_mean,
                'lag_168h': cell_row.cell_hist_mean,
                'rolling_3h_mean': cell_row.cell_hist_mean,
                'rolling_6h_mean': cell_row.cell_hist_mean,
                'rolling_24h_mean': cell_row.cell_hist_mean,
                'rolling_7d_mean': cell_row.cell_hist_mean,
                'rolling_24h_std': 0.0,
                'pct_car': cell_row.pct_car,
                'pct_scooter': cell_row.pct_scooter,
                'pct_auto': cell_row.pct_auto,
                'pct_maxi': cell_row.pct_maxi,
                'pct_wrong_park': cell_row.pct_wrong_park,
                'pct_no_park': cell_row.pct_no_park,
                'pct_main_road': cell_row.pct_main_road,
                'pct_footpath': cell_row.pct_footpath,
                'avg_vtype_count': cell_row.avg_vtype_count,
            }
            hr_features.append(row_dict)
            
        hr_df = pd.DataFrame(hr_features)
        
        # Predict regressor and classifier
        pred_counts = np.clip(xgb_reg.predict(hr_df[FEATURES]), 0, None)
        pred_probs = xgb_clf.predict_proba(hr_df[FEATURES])[:, 1]
        
        # Save results
        for idx, cell_row in enumerate(cell_meta.itertuples()):
            pred_cnt = pred_counts[idx]
            pred_prob = pred_probs[idx]
            
            # Predict PCE score from count
            pred_pce = pred_cnt * cell_row.avg_pce_weighted
            
            forecast_rows.append({
                'h3_8': cell_row.h3_8,
                'datetime': f_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'date': f_dt.strftime('%Y-%m-%d'),
                'hour': hr,
                'day_of_week': f_dt.dayofweek,
                'pred_count': pred_cnt,
                'hotspot_prob': pred_prob,
                'hotspot_flag': int(pred_prob >= 0.5),
                'pred_weighted_pce': pred_pce,
                'lat_mean': cell_row.lat_center,
                'lon_mean': cell_row.lon_center
            })

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_parquet('predictions_7d.parquet')
print(f"   Forecast saved to predictions_7d.parquet of size: {len(forecast_df)}")

print("✅ Preprocessing and model training complete. All files cached successfully!")
