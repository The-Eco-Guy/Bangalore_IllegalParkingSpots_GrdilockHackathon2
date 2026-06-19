# Generated from: parking_hotspot_prediction (2) (1).ipynb
# Converted at: 2026-06-19T09:28:33.725Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # 🚗 Illegal Parking Hotspot Prediction Model
# **Bangalore Traffic Police Violation Data — Nov 2023 to Apr 2024**
# 
# This notebook builds an end-to-end ML pipeline to:
# 1. Aggregate raw violation records into **H3 Resolution-7 hex cells × hourly time buckets**
# 2. Engineer **temporal, spatial, and rolling lag features**
# 3. Train two models per H3 cell:
#    - **Regression**: predict violation count in the next hour
#    - **Binary Classification**: flag whether a cell will be a hotspot (≥10 violations/hr)
# 4. Evaluate with **time-based cross-validation** (no data leakage)
# 5. Interpret results with **SHAP feature importance**


# ## 1. Install Dependencies & Imports


# Install H3 library (run once)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "h3", "xgboost", "shap", "lightgbm", "--quiet"])

import pandas as pd
import numpy as np
import ast
import warnings
import os
warnings.filterwarnings('ignore')

import h3
import xgboost as xgb
import lightgbm as lgb
import shap

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

print("✅ All libraries loaded")
print(f"   XGBoost  : {xgb.__version__}")
print(f"   LightGBM : {lgb.__version__}")
print(f"   H3       : {h3.__version__}")

# ## 2. Load & Parse Raw Data

RAW_DATA_PARQUET = "jan to may police violation_anonymized791b166.parquet"
RAW_DATA_CSV = "jan to may police violation_anonymized791b166.csv"

if os.path.exists(RAW_DATA_PARQUET):
    df = pd.read_parquet(RAW_DATA_PARQUET)
else:
    df = pd.read_csv(RAW_DATA_CSV)
print(f"Raw records : {len(df):,}")
print(f"Columns     : {df.shape[1]}")
df.head(3)

# ── Parse datetimes & convert to IST ──────────────────────────────────────────
df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='ISO8601', utc=True)
df['created_ist']      = df['created_datetime'].dt.tz_convert('Asia/Kolkata')

# ── NOTE: Do NOT filter to 'approved' only ────────────────────────────────────
# Approval is a lagging process — Feb/Mar/Apr have <5% approval rate vs 60%+ for Nov/Dec.
# Filtering to approved would collapse recent months and empty the test set.
# We use ALL records and rely on the time-based split for honest evaluation.
print("Records by month (all):")
print(df['created_ist'].dt.to_period('M').value_counts().sort_index().to_string())

# ── Drop rows with missing lat/lon ────────────────────────────────────────────
df = df.dropna(subset=['latitude', 'longitude'])
print(f"\nAfter dropping null coordinates: {len(df):,}")

# ── Derive primary violation type (first in list) ─────────────────────────────
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

# ── Vehicle type: collapse rare types ─────────────────────────────────────────
top_vehicles = df['vehicle_type'].value_counts().head(6).index.tolist()
df['vehicle_type_clean'] = df['vehicle_type'].where(df['vehicle_type'].isin(top_vehicles), 'OTHER')

print("\n✅ Data parsed")
print(df[['created_ist','latitude','longitude','primary_violation','vehicle_type_clean']].head(3).to_string())

# ## 3. Assign H3 Resolution-7 Hex Cells


H3_RESOLUTION = 7   # ~1.2 km cell diameter

df['h3_cell'] = df.apply(
    lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], H3_RESOLUTION),
    axis=1
)

print(f"Unique H3 cells (res-7) : {df['h3_cell'].nunique()}")
print(f"Top 10 busiest cells:")
print(df['h3_cell'].value_counts().head(10).to_string())

# ## 4. Aggregate to (H3 Cell × Date × Hour) Buckets


# ── Extract temporal components ───────────────────────────────────────────────
df['date']         = df['created_ist'].dt.date
df['hour']         = df['created_ist'].dt.hour
df['day_of_week']  = df['created_ist'].dt.dayofweek   # 0=Mon, 6=Sun
df['month']        = df['created_ist'].dt.month
df['week_of_year'] = df['created_ist'].dt.isocalendar().week.astype(int)

# ── Aggregate ─────────────────────────────────────────────────────────────────
agg = df.groupby(['h3_cell', 'date', 'hour']).agg(
    violation_count   = ('id', 'count'),
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
    # Violation type mix
    pct_wrong_park    = ('primary_violation', lambda x: (x == 'WRONG PARKING').mean()),
    pct_no_park       = ('primary_violation', lambda x: (x == 'NO PARKING').mean()),
    pct_main_road     = ('primary_violation', lambda x: (x == 'PARKING IN A MAIN ROAD').mean()),
    pct_footpath      = ('primary_violation', lambda x: (x == 'PARKING ON FOOTPATH').mean()),
    avg_vtype_count   = ('violation_type_count', 'mean'),    # avg violations per record
).reset_index()

print(f"Aggregated rows : {len(agg):,}")
print(f"Unique H3 cells : {agg['h3_cell'].nunique()}")
print(f"Date range      : {agg['date'].min()} → {agg['date'].max()}")
print()
print(agg.describe())

# ## 5. Feature Engineering


print(agg.columns.tolist())
print(agg.index.name)

# ── Sort for lag computation ──────────────────────────────────────────────────
agg = agg.sort_values(['h3_cell', 'date', 'hour']).reset_index(drop=True)
agg['datetime'] = pd.to_datetime(agg['date'].astype(str)) + pd.to_timedelta(agg['hour'], unit='h')

# ── Rolling & lag features — using transform to keep h3_cell safe ─────────────
# Group by h3_cell and compute shifts directly on the sorted dataframe
# This avoids groupby.apply() which drops the groupby key in some pandas versions

print("Computing lag features per H3 cell (may take ~30 seconds)...")

grp = agg.groupby('h3_cell')['violation_count']

agg['lag_1h']           = grp.shift(1)
agg['lag_2h']           = grp.shift(2)
agg['lag_3h']           = grp.shift(3)
agg['lag_24h']          = grp.shift(24)
agg['lag_168h']         = grp.shift(168)

agg['rolling_3h_mean']  = grp.shift(1).groupby(agg['h3_cell']).transform(lambda x: x.rolling(3,   min_periods=1).mean())
agg['rolling_6h_mean']  = grp.shift(1).groupby(agg['h3_cell']).transform(lambda x: x.rolling(6,   min_periods=1).mean())
agg['rolling_24h_mean'] = grp.shift(1).groupby(agg['h3_cell']).transform(lambda x: x.rolling(24,  min_periods=1).mean())
agg['rolling_7d_mean']  = grp.shift(1).groupby(agg['h3_cell']).transform(lambda x: x.rolling(168, min_periods=1).mean())
agg['rolling_24h_std']  = grp.shift(1).groupby(agg['h3_cell']).transform(lambda x: x.rolling(24,  min_periods=2).std())

print(f"Done. Shape: {agg.shape}")
print(f"h3_cell present: {'h3_cell' in agg.columns}")
print(agg[['h3_cell','datetime','violation_count','lag_1h','rolling_24h_mean']].head(4).to_string())

# ── Temporal encoding ─────────────────────────────────────────────────────────
# Cyclical encoding: hour and day-of-week wrap around
agg['hour_sin']   = np.sin(2 * np.pi * agg['hour'] / 24)
agg['hour_cos']   = np.cos(2 * np.pi * agg['hour'] / 24)
agg['dow_sin']    = np.sin(2 * np.pi * agg['day_of_week'] / 7)
agg['dow_cos']    = np.cos(2 * np.pi * agg['day_of_week'] / 7)
agg['month_sin']  = np.sin(2 * np.pi * agg['month'] / 12)
agg['month_cos']  = np.cos(2 * np.pi * agg['month'] / 12)

# Binary convenience flags
agg['is_weekend']   = (agg['day_of_week'] >= 5).astype(int)
agg['is_peak_hour'] = (agg['hour'].between(8, 11) | agg['hour'].between(17, 20)).astype(int)
agg['is_morning']   = (agg['hour'].between(5, 11)).astype(int)
agg['is_night']     = ((agg['hour'] >= 22) | (agg['hour'] <= 5)).astype(int)

# ── H3 spatial features ───────────────────────────────────────────────────────
# Encode H3 cell as integer ID (LightGBM/XGBoost can handle this)
le = LabelEncoder()
agg['h3_cell_id'] = le.fit_transform(agg['h3_cell'])

# Cell-level historical mean (global average for each H3 cell across all time)
cell_hist_mean = agg.groupby('h3_cell')['violation_count'].transform('mean')
agg['cell_hist_mean'] = cell_hist_mean

# ── Classification target ─────────────────────────────────────────────────────
HOTSPOT_THRESHOLD = 10   # ≥10 violations/hr in a cell = hotspot
agg['is_hotspot'] = (agg['violation_count'] >= HOTSPOT_THRESHOLD).astype(int)

print(f"Hotspot rate: {agg['is_hotspot'].mean()*100:.1f}% of cell-hour buckets")
print(f"\nFinal feature set shape: {agg.shape}")
print()
print(agg[['hour','violation_count','lag_1h','rolling_24h_mean','is_hotspot','is_peak_hour']].head(8).to_string())

# ## 6. Time-Based Train / Validation / Test Split
# > ⚠️ Never shuffle randomly for time-series — always split chronologically to avoid data leakage.


# datetime is already a plain column (no set_index used above)

# ── Drop rows where lag features are NaN (first few hours per cell) ───────────
LAG_COLS = ['lag_1h','lag_2h','lag_3h','lag_24h','rolling_3h_mean','rolling_24h_mean']
agg_clean = agg.dropna(subset=LAG_COLS).copy()
print(f"Rows after dropping NaN lags: {len(agg_clean):,}")

# ── Chronological split (NO timezone on datetime — it's already naive IST) ────
# Data range: Nov 2023 → Apr 2024
# Train : Nov 2023 – Jan 2024  (high-density approved period)
# Val   : Feb 2024             (transition month)
# Test  : Mar – Apr 2024       (most recent, unseen)
TRAIN_END = pd.Timestamp('2024-02-01')   # naive datetime, matches agg['datetime']
VAL_END   = pd.Timestamp('2024-03-01')

train = agg_clean[agg_clean['datetime'] <  TRAIN_END]
val   = agg_clean[(agg_clean['datetime'] >= TRAIN_END) & (agg_clean['datetime'] < VAL_END)]
test  = agg_clean[agg_clean['datetime'] >= VAL_END]

print(f"\nTrain  : {len(train):>6,} rows  ({train['datetime'].min().date()} → {train['datetime'].max().date()})")
print(f"Val    : {len(val):>6,} rows  ({val['datetime'].min().date()} → {val['datetime'].max().date()})")
print(f"Test   : {len(test):>6,} rows  ({test['datetime'].min().date()} → {test['datetime'].max().date()})")
print()

for name, split in [('Train', train), ('Val', val), ('Test', test)]:
    if len(split) == 0:
        print(f"⚠️  {name} split is EMPTY — check date range above")
    else:
        print(f"Hotspot rate — {name}: {split['is_hotspot'].mean()*100:.1f}%")

# ── Define feature columns ────────────────────────────────────────────────────
FEATURES = [
    # Spatial
    'h3_cell_id', 'lat_mean', 'lon_mean', 'cell_hist_mean',
    # Temporal (raw)
    'hour', 'day_of_week', 'month', 'week_of_year',
    # Temporal (cyclical)
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    # Temporal flags
    'is_weekend', 'is_peak_hour', 'is_morning', 'is_night',
    # Lag features
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
    # Rolling features
    'rolling_3h_mean', 'rolling_6h_mean', 'rolling_24h_mean', 'rolling_7d_mean', 'rolling_24h_std',
    # Vehicle / violation mix
    'pct_car', 'pct_scooter', 'pct_auto', 'pct_maxi',
    'pct_wrong_park', 'pct_no_park', 'pct_main_road', 'pct_footpath',
    'avg_vtype_count',
]

TARGET_REG = 'violation_count'
TARGET_CLF = 'is_hotspot'

X_train = train[FEATURES];  y_train_reg = train[TARGET_REG];  y_train_clf = train[TARGET_CLF]
X_val   = val[FEATURES];    y_val_reg   = val[TARGET_REG];    y_val_clf   = val[TARGET_CLF]
X_test  = test[FEATURES];   y_test_reg  = test[TARGET_REG];   y_test_clf  = test[TARGET_CLF]

print(f"Feature matrix shape — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"\nFeatures ({len(FEATURES)}):")
for f in FEATURES:
    print(f"  {f}")

# ## 7. Model A — Regression (Predict Violation Count)


# ── Train XGBoost Regressor ───────────────────────────────────────────────────
xgb_reg = xgb.XGBRegressor(
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 5,
    reg_alpha         = 0.1,    # L1
    reg_lambda        = 1.0,    # L2
    objective         = 'reg:squarederror',
    eval_metric       = 'mae',
    early_stopping_rounds = 30,
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)

xgb_reg.fit(
    X_train, y_train_reg,
    eval_set=[(X_val, y_val_reg)],
    verbose=False,
)
print(f"Best iteration: {xgb_reg.best_iteration}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
for split_name, X_s, y_s in [('Val', X_val, y_val_reg), ('Test', X_test, y_test_reg)]:
    preds = xgb_reg.predict(X_s)
    preds = np.clip(preds, 0, None)   # violations can't be negative
    mae   = mean_absolute_error(y_s, preds)
    rmse  = np.sqrt(mean_squared_error(y_s, preds))
    r2    = r2_score(y_s, preds)
    print(f"\n{split_name} — MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.3f}")

# ── Plot: Actual vs Predicted (Test set, scatter) ─────────────────────────────
test_preds_reg = np.clip(xgb_reg.predict(X_test), 0, None)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter
ax = axes[0]
lim = max(y_test_reg.max(), test_preds_reg.max()) + 2
ax.scatter(y_test_reg, test_preds_reg, alpha=0.3, s=8, color='steelblue')
ax.plot([0, lim], [0, lim], 'r--', linewidth=1.2, label='Perfect prediction')
ax.set_xlabel('Actual violation count'); ax.set_ylabel('Predicted violation count')
ax.set_title('Regression — Actual vs Predicted (Test set)')
ax.legend(); ax.set_xlim(0, lim); ax.set_ylim(0, lim)

# Residuals
ax = axes[1]
residuals = y_test_reg.values - test_preds_reg
ax.hist(residuals, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
ax.set_xlabel('Residual (Actual − Predicted)'); ax.set_ylabel('Count')
ax.set_title('Regression — Residual Distribution')

plt.tight_layout()
plt.savefig('regression_eval.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → regression_eval.png")

# ## 8. Model B — Classification (Hotspot Flag)


# ── Class weight to handle imbalance (~13% positive) ─────────────────────────
neg_pos_ratio = (y_train_clf == 0).sum() / (y_train_clf == 1).sum()
print(f"Negative:Positive ratio = {neg_pos_ratio:.1f}:1  → scale_pos_weight = {neg_pos_ratio:.1f}")

xgb_clf = xgb.XGBClassifier(
    n_estimators        = 500,
    learning_rate       = 0.05,
    max_depth           = 6,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    min_child_weight    = 5,
    scale_pos_weight    = neg_pos_ratio,    # handles class imbalance
    objective           = 'binary:logistic',
    eval_metric         = 'auc',
    early_stopping_rounds = 30,
    random_state        = 42,
    n_jobs              = -1,
    verbosity           = 0,
)

xgb_clf.fit(
    X_train, y_train_clf,
    eval_set=[(X_val, y_val_clf)],
    verbose=False,
)
print(f"Best iteration: {xgb_clf.best_iteration}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
for split_name, X_s, y_s in [('Val', X_val, y_val_clf), ('Test', X_test, y_test_clf)]:
    probs = xgb_clf.predict_proba(X_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(y_s, probs)
    print(f"\n{'='*50}")
    print(f"{split_name} — AUC-ROC: {auc:.3f}")
    print(classification_report(y_s, preds, target_names=['Normal', 'Hotspot']))

# ── Confusion matrix + ROC curve ──────────────────────────────────────────────
from sklearn.metrics import roc_curve

test_probs_clf = xgb_clf.predict_proba(X_test)[:, 1]
test_preds_clf = (test_probs_clf >= 0.5).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
cm = confusion_matrix(y_test_clf, test_preds_clf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Hotspot'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Classification — Confusion Matrix (Test set)')

# ROC curve
fpr, tpr, _ = roc_curve(y_test_clf, test_probs_clf)
auc_score = roc_auc_score(y_test_clf, test_probs_clf)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc_score:.3f}')
axes[1].plot([0,1], [0,1], 'r--', lw=1.2)
axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve (Test set)')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('classification_eval.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → classification_eval.png")

# ## 9. Feature Importance — SHAP Analysis
# > SHAP explains *why* each prediction was made — critical for communicating to enforcement officers.


# ── SHAP for Regressor ────────────────────────────────────────────────────────
print("Computing SHAP values for regressor (sample of 2000 rows)...")
sample_idx  = X_test.sample(2000, random_state=42).index
X_shap      = X_test.loc[sample_idx]

explainer_reg  = shap.TreeExplainer(xgb_reg)
shap_vals_reg  = explainer_reg.shap_values(X_shap)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plt.sca(axes[0])
shap.summary_plot(shap_vals_reg, X_shap, feature_names=FEATURES,
                  max_display=15, show=False, plot_type='bar')
axes[0].set_title('Regression — Mean |SHAP| Feature Importance')

plt.sca(axes[1])
shap.summary_plot(shap_vals_reg, X_shap, feature_names=FEATURES,
                  max_display=15, show=False)
axes[1].set_title('Regression — SHAP Beeswarm (impact direction)')

plt.tight_layout()
plt.savefig('shap_regression.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → shap_regression.png")

# ── SHAP for Classifier ───────────────────────────────────────────────────────
print("Computing SHAP values for classifier...")
explainer_clf  = shap.TreeExplainer(xgb_clf)
shap_vals_clf  = explainer_clf.shap_values(X_shap)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.sca(axes[0])
shap.summary_plot(shap_vals_clf, X_shap, feature_names=FEATURES,
                  max_display=15, show=False, plot_type='bar')
axes[0].set_title('Classifier — Mean |SHAP| Feature Importance')

plt.sca(axes[1])
shap.summary_plot(shap_vals_clf, X_shap, feature_names=FEATURES,
                  max_display=15, show=False)
axes[1].set_title('Classifier — SHAP Beeswarm')

plt.tight_layout()
plt.savefig('shap_classifier.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → shap_classifier.png")

# ## 10. Generate Hotspot Intelligence — Next-Hour Predictions
# Score every active H3 cell for the next hour and rank by risk.


# ── Build a 'next hour' feature snapshot for each H3 cell ────────────────────
# Use the LAST known state of each cell as the input to predict the next slot

latest = agg_clean.sort_values('datetime').groupby('h3_cell').last().reset_index()

# Simulate 'next hour' temporal features
import datetime as dt
next_hour_dt = pd.Timestamp.now(tz='Asia/Kolkata').replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)

latest['hour']        = next_hour_dt.hour
latest['day_of_week'] = next_hour_dt.dayofweek
latest['month']       = next_hour_dt.month
latest['week_of_year']= next_hour_dt.isocalendar()[1]
latest['is_weekend']  = int(next_hour_dt.dayofweek >= 5)
latest['is_peak_hour']= int(next_hour_dt.hour in range(8, 12) or next_hour_dt.hour in range(17, 21))
latest['is_morning']  = int(5 <= next_hour_dt.hour <= 11)
latest['is_night']    = int(next_hour_dt.hour >= 22 or next_hour_dt.hour <= 5)
latest['hour_sin']    = np.sin(2 * np.pi * latest['hour'] / 24)
latest['hour_cos']    = np.cos(2 * np.pi * latest['hour'] / 24)
latest['dow_sin']     = np.sin(2 * np.pi * latest['day_of_week'] / 7)
latest['dow_cos']     = np.cos(2 * np.pi * latest['day_of_week'] / 7)
latest['month_sin']   = np.sin(2 * np.pi * latest['month'] / 12)
latest['month_cos']   = np.cos(2 * np.pi * latest['month'] / 12)

# Shift lags: last known count becomes lag_1h for next slot
latest['lag_1h']  = latest['violation_count']
latest['lag_2h']  = latest['lag_1h']
latest['lag_3h']  = latest['lag_2h']

X_score = latest[FEATURES].fillna(0)

# Score
latest['pred_count']   = np.clip(xgb_reg.predict(X_score), 0, None).round(1)
latest['hotspot_prob'] = xgb_clf.predict_proba(X_score)[:, 1]
latest['hotspot_flag'] = (latest['hotspot_prob'] >= 0.5).astype(int)

# ── Top 20 predicted hotspots ─────────────────────────────────────────────────
top_hotspots = (
    latest[latest['hotspot_flag'] == 1]
    .sort_values('hotspot_prob', ascending=False)
    [['h3_cell', 'lat_mean', 'lon_mean', 'pred_count', 'hotspot_prob', 'cell_hist_mean']]
    .head(20)
    .reset_index(drop=True)
)
top_hotspots.index += 1
top_hotspots.columns = ['H3 Cell', 'Lat', 'Lon', 'Predicted Violations', 'Hotspot Probability', 'Historical Avg']
top_hotspots['Hotspot Probability'] = top_hotspots['Hotspot Probability'].map('{:.1%}'.format)
top_hotspots['Historical Avg']      = top_hotspots['Historical Avg'].round(1)

print(f"Predicted hotspots for {next_hour_dt.strftime('%Y-%m-%d %H:00 IST')}:")
print(top_hotspots.to_string())

# ## 11. Save Models & Scored Output


import pickle, os

# Save models
with open('xgb_regressor.pkl', 'wb') as f:
    pickle.dump(xgb_reg, f)
with open('xgb_classifier.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)

# Save feature list and H3 encoder
with open('model_meta.pkl', 'wb') as f:
    pickle.dump({'features': FEATURES, 'h3_resolution': H3_RESOLUTION,
                 'hotspot_threshold': HOTSPOT_THRESHOLD, 'label_encoder': le}, f)

# Save hotspot scores to CSV
score_output = latest[['h3_cell', 'lat_mean', 'lon_mean',
                         'pred_count', 'hotspot_prob', 'hotspot_flag', 'cell_hist_mean']]
score_output.to_csv('hotspot_scores.csv', index=False)

print("✅ Saved:")
print("   xgb_regressor.pkl    — violation count regression model")
print("   xgb_classifier.pkl   — hotspot binary classifier")
print("   model_meta.pkl       — features + encoders")
print("   hotspot_scores.csv   — scored H3 cells with predictions")
print()
print(f"Total H3 cells scored  : {len(score_output):,}")
print(f"Predicted hotspots     : {score_output['hotspot_flag'].sum():,}")
print(f"Hotspot rate           : {score_output['hotspot_flag'].mean()*100:.1f}%")

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'folium', 'branca', '--quiet'])
print('folium ready')

import folium
from folium.plugins import HeatMapWithTime
import h3
import numpy as np
import pandas as pd

# ── Reconstruct hourly scores for every active H3 cell ───────────────────
# Use the TEST set so predictions are on unseen data
test_scored = test.copy()
test_scored['pred_count']   = np.clip(xgb_reg.predict(X_test), 0, None)
test_scored['hotspot_prob'] = xgb_clf.predict_proba(X_test)[:, 1]
test_scored['hotspot_flag'] = (test_scored['hotspot_prob'] >= 0.5).astype(int)

# ── Average predictions per (H3 cell × hour) across all test days ────────
hourly = (
    test_scored
    .groupby(['h3_cell', 'hour'])
    .agg(
        lat_mean      = ('lat_mean', 'mean'),
        lon_mean      = ('lon_mean', 'mean'),
        pred_count    = ('pred_count', 'mean'),
        hotspot_prob  = ('hotspot_prob', 'mean'),
        actual_count  = ('violation_count', 'mean'),
    )
    .reset_index()
)

print(f'Unique H3 cells in test set : {hourly["h3_cell"].nunique()}')
print(f'Hours covered               : {sorted(hourly["hour"].unique())}')
print(f'Max predicted count/hr      : {hourly["pred_count"].max():.1f}')
print(hourly.sort_values('pred_count', ascending=False).head(5).to_string())


# ── Get accurate centre lat/lon from H3 for each cell ───────────────────
def h3_center(cell):
    lat, lon = h3.cell_to_latlng(cell)
    return pd.Series({'h3_lat': lat, 'h3_lon': lon})

centers = hourly[['h3_cell']].drop_duplicates()
centers[['h3_lat','h3_lon']] = centers['h3_cell'].apply(h3_center)
hourly = hourly.merge(centers, on='h3_cell')

# ── Normalise intensity 0-1 for heatmap weight ───────────────────────────
max_pred = hourly['pred_count'].max()
hourly['intensity'] = (hourly['pred_count'] / max_pred).clip(0, 1)

# ── Build list-of-lists: one list per hour, each item [lat, lon, weight] ─
all_hours = list(range(24))
heat_data = []
for hr in all_hours:
    hr_df = hourly[hourly['hour'] == hr]
    if len(hr_df) == 0:
        heat_data.append([[12.97, 77.59, 0]])  # dummy Bangalore point
    else:
        heat_data.append(
            hr_df[['h3_lat','h3_lon','intensity']].values.tolist()
        )

time_labels = [f'{hr:02d}:00 IST' for hr in all_hours]
print(f'Heat data frames : {len(heat_data)}')
print(f'Points in 09:00  : {len(heat_data[9])}')


# ── Base map centred on Bangalore ────────────────────────────────────────
m = folium.Map(
    location=[12.97, 77.59],
    zoom_start=12,
    tiles='CartoDB dark_matter',   # dark base looks great for heatmaps
)

# ── Animated heatmap layer ────────────────────────────────────────────────
HeatMapWithTime(
    heat_data,
    index=time_labels,
    name='Predicted Violations',
    radius=35,
    max_opacity=0.85,
    min_opacity=0.05,
    gradient={0.2: '#2c7bb6', 0.4: '#abd9e9', 0.6: '#ffffbf',
              0.8: '#fdae61', 1.0: '#d7191c'},
    auto_play=False,
    display_index=True,
    speed_step=0.5,
    position='bottomleft',
).add_to(m)

# ── Top 10 predicted hotspot markers ─────────────────────────────────────
top10 = (
    hourly.groupby('h3_cell')
    .agg(avg_prob=('hotspot_prob','mean'), avg_count=('pred_count','mean'),
         h3_lat=('h3_lat','first'), h3_lon=('h3_lon','first'))
    .sort_values('avg_prob', ascending=False)
    .head(10)
    .reset_index()
)

for _, row in top10.iterrows():
    folium.CircleMarker(
        location=[row['h3_lat'], row['h3_lon']],
        radius=10,
        color='#ff4444',
        fill=True,
        fill_color='#ff4444',
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>H3:</b> {row['h3_cell']}<br>"
            f"<b>Avg hotspot prob:</b> {row['avg_prob']:.1%}<br>"
            f"<b>Avg predicted violations:</b> {row['avg_count']:.1f}/hr",
            max_width=250
        ),
        tooltip=f"Hotspot prob: {row['avg_prob']:.1%}"
    ).add_to(m)

# ── Legend ────────────────────────────────────────────────────────────────
legend_html = '''
<div style="position:fixed;bottom:30px;right:10px;z-index:1000;
            background:rgba(0,0,0,0.75);padding:12px 16px;
            border-radius:8px;color:white;font-family:Arial;font-size:13px">
  <b>Violation Intensity</b><br>
  <span style='color:#2c7bb6'>■</span> Low &nbsp;
  <span style='color:#ffffbf'>■</span> Medium &nbsp;
  <span style='color:#d7191c'>■</span> High<br><br>
  <span style='color:#ff4444'>●</span> Top 10 Hotspot Zones
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)

# ── Save & display ───────────────────────────────────────────────────────
MAP_PATH = 'hotspot_heatmap_animated.html'
m.save(MAP_PATH)
print(f'Map saved → {MAP_PATH}')
print('Open this HTML file in your browser for the full interactive experience')

# Display inline in notebook
from IPython.display import IFrame
IFrame(MAP_PATH, width='100%', height=550)


# ── Peak hour summary — when and where is enforcement most needed? ───────
peak_summary = (
    hourly.groupby('hour')
    .agg(
        active_cells    = ('h3_cell', 'count'),
        avg_pred_count  = ('pred_count', 'mean'),
        hotspot_cells   = ('hotspot_prob', lambda x: (x >= 0.5).sum()),
        max_pred_count  = ('pred_count', 'max'),
    )
    .reset_index()
)
peak_summary.columns = ['Hour (IST)', 'Active Cells', 'Avg Predicted/hr',
                         'Hotspot Cells', 'Max Predicted/hr']
peak_summary['Avg Predicted/hr'] = peak_summary['Avg Predicted/hr'].round(1)
peak_summary['Max Predicted/hr'] = peak_summary['Max Predicted/hr'].round(1)
peak_summary['Hour (IST)'] = peak_summary['Hour (IST)'].apply(lambda h: f'{h:02d}:00')

print('Enforcement Priority by Hour:')
print(peak_summary.sort_values('Hotspot Cells', ascending=False).to_string(index=False))


import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install',
                'holidays', 'fastapi', 'uvicorn[standard]',
                'pydantic', 'fpdf2', '--quiet'])
print('All packages ready')


import pandas as pd
import numpy as np
import holidays
from datetime import datetime, date

# ── Indian public holidays (Karnataka state) ──────────────────────────────
india_holidays = holidays.India(state='KA', years=[2023, 2024])

def add_holiday_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Adds is_public_holiday, days_to_next_holiday, days_since_last_holiday
    to any dataframe with a datetime column.
    """
    dt = pd.to_datetime(df[datetime_col])
    df['is_public_holiday'] = dt.dt.date.apply(
        lambda d: int(d in india_holidays)
    )
    df['holiday_name'] = dt.dt.date.apply(
        lambda d: india_holidays.get(d, 'None')
    )
    # Days to next / since last holiday (useful signal for pre/post holiday spikes)
    all_holiday_dates = sorted(india_holidays.keys())
    all_holiday_ts    = pd.to_datetime(all_holiday_dates)

    def days_to_next(d):
        future = [h for h in all_holiday_dates if h > d]
        return (pd.Timestamp(future[0]) - pd.Timestamp(d)).days if future else 30

    def days_since_last(d):
        past = [h for h in all_holiday_dates if h < d]
        return (pd.Timestamp(d) - pd.Timestamp(past[-1])).days if past else 30

    df['days_to_next_holiday']   = dt.dt.date.apply(days_to_next)
    df['days_since_last_holiday']= dt.dt.date.apply(days_since_last)
    return df

# ── Demo on a small date range ─────────────────────────────────────────────
demo = pd.DataFrame({'dt': pd.date_range('2024-01-13', periods=10, freq='D')})
demo = add_holiday_features(demo, 'dt')
print(demo[['dt','is_public_holiday','holiday_name',
            'days_to_next_holiday','days_since_last_holiday']].to_string())


import pandas as pd
import numpy as np
import holidays
from datetime import datetime, date

# ── Indian public holidays (Karnataka state) ──────────────────────────────
india_holidays = holidays.India(state='KA', years=[2023, 2024])

def add_holiday_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Adds is_public_holiday, days_to_next_holiday, days_since_last_holiday
    to any dataframe with a datetime column.
    """
    dt = pd.to_datetime(df[datetime_col])
    df['is_public_holiday'] = dt.dt.date.apply(
        lambda d: int(d in india_holidays)
    )
    df['holiday_name'] = dt.dt.date.apply(
        lambda d: india_holidays.get(d, 'None')
    )
    # Days to next / since last holiday (useful signal for pre/post holiday spikes)
    all_holiday_dates = sorted(india_holidays.keys())
    all_holiday_ts    = pd.to_datetime(all_holiday_dates)

    def days_to_next(d):
        future = [h for h in all_holiday_dates if h > d]
        return (pd.Timestamp(future[0]) - pd.Timestamp(d)).days if future else 30

    def days_since_last(d):
        past = [h for h in all_holiday_dates if h < d]
        return (pd.Timestamp(d) - pd.Timestamp(past[-1])).days if past else 30

    df['days_to_next_holiday']   = dt.dt.date.apply(days_to_next)
    df['days_since_last_holiday']= dt.dt.date.apply(days_since_last)
    return df

# ── Demo on a small date range ─────────────────────────────────────────────
demo = pd.DataFrame({'dt': pd.date_range('2024-01-13', periods=10, freq='D')})
demo = add_holiday_features(demo, 'dt')
print(demo[['dt','is_public_holiday','holiday_name',
            'days_to_next_holiday','days_since_last_holiday']].to_string())


# ── Weather feature scaffold ──────────────────────────────────────────────
# In production: fetch from Open-Meteo API (free, no key needed)
# Here we show the integration pattern + generate synthetic weather
# for demonstration purposes

def fetch_weather_features(lat: float, lon: float, dt: datetime) -> dict:
    """
    In production, replace this with:
        import requests
        url = f'https://api.open-meteo.com/v1/forecast?'
              f'latitude={lat}&longitude={lon}&hourly=precipitation,
              f'windspeed_10m&timezone=Asia/Kolkata'
        r = requests.get(url).json()
        # extract hour-matched values
    """
    # Synthetic demo values — replace with real API call
    np.random.seed(dt.hour)
    return {
        'precipitation_mm': float(np.random.choice([0, 0, 0, 2.5, 8.0], p=[0.5,0.2,0.15,0.1,0.05])),
        'is_raining':       int(np.random.random() < 0.15),
        'wind_speed_kmh':   float(np.round(np.random.uniform(5, 25), 1)),
    }

# Test
w = fetch_weather_features(12.97, 77.59, datetime(2024, 3, 15, 9, 0))
print('Weather features for 09:00 IST:', w)

print()
print('To use real weather data, replace fetch_weather_features() with:')
print('  pip install openmeteo-requests')
print('  https://open-meteo.com/en/docs — free, no API key, hourly data')


%%writefile challan_engine.py

import uuid
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from fpdf import FPDF

IST = ZoneInfo('Asia/Kolkata')

# ── Fine amounts per violation type (Motor Vehicles Act 2019) ────────────
FINE_SCHEDULE = {
    'WRONG PARKING':             500,
    'NO PARKING':                500,
    'PARKING IN A MAIN ROAD':   1000,
    'PARKING ON FOOTPATH':       500,
    'OBSTRUCTION TO TRAFFIC':   1000,
    'DOUBLE PARKING':            500,
    'DEFAULT':                   500,
}

VIOLATIONS_LOG = 'violations_issued.json'


def load_log():
    if os.path.exists(VIOLATIONS_LOG):
        with open(VIOLATIONS_LOG) as f:
            return json.load(f)
    return []


def save_log(records):
    with open(VIOLATIONS_LOG, 'w') as f:
        json.dump(records, f, indent=2)


def generate_challan(
    vehicle_number:  str,
    vehicle_type:    str,
    violation_type:  str,
    latitude:        float,
    longitude:       float,
    h3_cell:         str,
    officer_id:      str,
    hotspot_prob:    float,
    location_desc:   str = '',
) -> dict:
    """
    Generate a challan record + PDF notice for a confirmed parking violation.
    Returns the challan dict with challan_id and pdf_path.
    """
    now          = datetime.now(IST)
    challan_id   = f'CH-{now.strftime("%Y%m%d")}-{str(uuid.uuid4())[:8].upper()}'
    fine_amount  = FINE_SCHEDULE.get(violation_type.upper(), FINE_SCHEDULE['DEFAULT'])
    due_date     = now.strftime('%Y-%m-%d')  # pay within 60 days in production

    challan = {
        'challan_id':     challan_id,
        'issued_at':      now.strftime('%Y-%m-%d %H:%M:%S IST'),
        'vehicle_number': vehicle_number.upper(),
        'vehicle_type':   vehicle_type,
        'violation_type': violation_type,
        'fine_amount':    fine_amount,
        'latitude':       latitude,
        'longitude':      longitude,
        'h3_cell':        h3_cell,
        'location_desc':  location_desc,
        'officer_id':     officer_id,
        'hotspot_prob':   round(hotspot_prob, 4),
        'status':         'ISSUED',
        'pdf_path':       f'challans/{challan_id}.pdf'
    }

    # ── Generate PDF challan ──────────────────────────────────────────────
    os.makedirs('challans', exist_ok=True)
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(20, 40, 80)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_xy(10, 8)
    pdf.cell(0, 8, 'BANGALORE TRAFFIC POLICE', ln=True)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_xy(10, 18)
    pdf.cell(0, 6, 'AI-ASSISTED PARKING VIOLATION NOTICE (CHALLAN)', ln=True)

    # Challan ID banner
    pdf.set_fill_color(220, 50, 50)
    pdf.rect(0, 30, 210, 12, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_xy(10, 33)
    pdf.cell(0, 6, f'Challan No: {challan_id}', ln=True)

    # Body
    pdf.set_text_color(30, 30, 30)
    pdf.set_xy(10, 50)

    def row(label, value, bold_val=False):
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_x(10)
        pdf.cell(70, 8, label + ':', ln=False)
        pdf.set_font('Helvetica', 'B' if bold_val else '', 11)
        pdf.cell(0, 8, str(value), ln=True)

    row('Date & Time',     challan['issued_at'])
    row('Vehicle Number',  challan['vehicle_number'], bold_val=True)
    row('Vehicle Type',    challan['vehicle_type'])
    row('Violation',       challan['violation_type'])
    row('Location',        location_desc or f'{latitude:.5f}, {longitude:.5f}')
    row('Zone (H3)',        h3_cell)
    row('Officer ID',      challan['officer_id'])

    # Fine box
    pdf.ln(5)
    pdf.set_fill_color(255, 243, 205)
    pdf.set_draw_color(200, 150, 0)
    pdf.rect(10, pdf.get_y(), 190, 18, 'FD')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(150, 80, 0)
    pdf.set_x(10)
    pdf.cell(190, 18, f'FINE AMOUNT:  Rs. {fine_amount}/-', align='C', ln=True)

    # AI note
    pdf.ln(5)
    pdf.set_text_color(80, 80, 80)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_x(10)
    pdf.multi_cell(190, 5,
        f'This location was flagged by the AI Hotspot Prediction System '
        f'(hotspot probability: {hotspot_prob:.1%}). Fine issued under the '
        f'Motor Vehicles Act 2019. Pay within 60 days to avoid penalty.')

    # Footer
    pdf.set_fill_color(20, 40, 80)
    pdf.rect(0, 275, 210, 22, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_xy(10, 278)
    pdf.cell(0, 5, 'Pay online: https://bangaloretrafficpolice.gov.in', ln=True)
    pdf.set_x(10)
    pdf.cell(0, 5, 'Helpline: 080-22868375 | This is a computer-generated document', ln=True)

    pdf.output(challan['pdf_path'])

    # ── Log the challan ───────────────────────────────────────────────────
    records = load_log()
    records.append(challan)
    save_log(records)

    return challan


def get_violation_stats() -> dict:
    records = load_log()
    if not records:
        return {'total': 0, 'total_fines_rs': 0, 'by_type': {}, 'by_hour': {}}
    df = pd.DataFrame(records)
    return {
        'total':           len(df),
        'total_fines_rs':  int(df['fine_amount'].sum()),
        'by_type':         df['violation_type'].value_counts().to_dict(),
        'by_officer':      df['officer_id'].value_counts().to_dict(),
    }


import importlib, sys
# Remove cached module if re-running
if 'challan_engine' in sys.modules:
    del sys.modules['challan_engine']
from challan_engine import generate_challan, get_violation_stats

from challan_engine import generate_challan, get_violation_stats
import pandas as pd

# Simulate 3 officers issuing challans
test_cases = [
    dict(vehicle_number='KA01AB1234', vehicle_type='CAR',
         violation_type='WRONG PARKING', latitude=12.9716, longitude=77.5946,
         h3_cell='876013a5fffffff', officer_id='OFF-042',
         hotspot_prob=0.87, location_desc='MG Road near Trinity Metro'),
    dict(vehicle_number='KA03MN5678', vehicle_type='SCOOTER',
         violation_type='PARKING ON FOOTPATH', latitude=12.9352, longitude=77.6245,
         h3_cell='876013a5fffffff', officer_id='OFF-017',
         hotspot_prob=0.72, location_desc='Jayanagar 4th Block'),
    dict(vehicle_number='KA05XY9999', vehicle_type='MAXI-CAB',
         violation_type='PARKING IN A MAIN ROAD', latitude=12.9784, longitude=77.5408,
         h3_cell='876013b1fffffff', officer_id='OFF-031',
         hotspot_prob=0.91, location_desc='Rajajinagar Main Road'),
]

challans = []
for tc in test_cases:
    c = generate_challan(**tc)
    challans.append(c)
    print(f"✅ {c['challan_id']} | {c['vehicle_number']} | Rs.{c['fine_amount']} | {c['pdf_path']}")

print()
stats = get_violation_stats()
print('Enforcement stats:')
for k, v in stats.items():
    print(f'  {k}: {v}')


# View the generated challan PDF
from IPython.display import IFrame
IFrame(challans[0]['pdf_path'], width='100%', height=500)


%%writefile main.py

import pickle, uuid, json, os
import numpy as np
import pandas as pd
import h3
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from challan_engine import generate_challan, get_violation_stats

with open('xgb_regressor.pkl',  'rb') as f: reg_model = pickle.load(f)
with open('xgb_classifier.pkl', 'rb') as f: clf_model = pickle.load(f)
with open('model_meta.pkl',     'rb') as f: meta      = pickle.load(f)

FEATURES       = meta['features']
H3_RESOLUTION  = meta['h3_resolution']
HOTSPOT_THRESH = meta['hotspot_threshold']
le             = meta['label_encoder']

scores_df  = pd.read_csv('hotspot_scores.csv')
cell_stats = scores_df.set_index('h3_cell')[['cell_hist_mean']].to_dict('index')
IST        = ZoneInfo('Asia/Kolkata')

app = FastAPI(
    title='Parking Hotspot & Fine Issuance API',
    description='Predict hotspots + issue challans in real time',
    version='2.0.0'
)


# ── Schemas ───────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    latitude:     float
    longitude:    float
    datetime_ist: Optional[str] = None
    lag_1h:       Optional[float] = None
    lag_24h:      Optional[float] = None

class FineRequest(BaseModel):
    vehicle_number: str  = Field(..., example='KA01AB1234')
    vehicle_type:   str  = Field(..., example='CAR')
    violation_type: str  = Field(..., example='WRONG PARKING')
    latitude:       float
    longitude:      float
    officer_id:     str  = Field(..., example='OFF-042')
    location_desc:  str  = Field('', example='MG Road near Trinity Metro')

class BatchRequest(BaseModel):
    locations: List[PredictRequest]


# ── Feature builder ───────────────────────────────────────────────────────
def build_features(lat, lon, dt, lag_1h=None, lag_24h=None):
    cell  = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
    hour  = dt.hour; dow = dt.weekday(); month = dt.month
    week  = dt.isocalendar()[1]
    try:    cell_id = int(le.transform([cell])[0])
    except: cell_id = -1
    hist = cell_stats.get(cell, {}).get('cell_hist_mean',
                                         scores_df['cell_hist_mean'].mean())
    lag1  = lag_1h  if lag_1h  is not None else hist
    lag24 = lag_24h if lag_24h is not None else hist
    row = {
        'h3_cell_id': cell_id, 'lat_mean': lat, 'lon_mean': lon,
        'cell_hist_mean': hist, 'hour': hour, 'day_of_week': dow,
        'month': month, 'week_of_year': week,
        'hour_sin': np.sin(2*np.pi*hour/24), 'hour_cos': np.cos(2*np.pi*hour/24),
        'dow_sin':  np.sin(2*np.pi*dow/7),   'dow_cos':  np.cos(2*np.pi*dow/7),
        'month_sin':np.sin(2*np.pi*month/12),'month_cos':np.cos(2*np.pi*month/12),
        'is_weekend': int(dow>=5), 'is_peak_hour': int(hour in range(8,12) or hour in range(17,21)),
        'is_morning': int(5<=hour<=11), 'is_night': int(hour>=22 or hour<=5),
        'lag_1h': lag1, 'lag_2h': lag1, 'lag_3h': lag1,
        'lag_24h': lag24, 'lag_168h': lag24,
        'rolling_3h_mean': lag1, 'rolling_6h_mean': lag1,
        'rolling_24h_mean': hist, 'rolling_7d_mean': hist, 'rolling_24h_std': 0.0,
        'pct_car': 0.4, 'pct_scooter': 0.3, 'pct_auto': 0.1, 'pct_maxi': 0.05,
        'pct_wrong_park': 0.6, 'pct_no_park': 0.25,
        'pct_main_road': 0.1, 'pct_footpath': 0.05, 'avg_vtype_count': 1.2,
    }
    return pd.DataFrame([row])[FEATURES], cell

def risk_label(p):
    return 'CRITICAL' if p>=0.75 else 'HIGH' if p>=0.5 else 'MEDIUM' if p>=0.25 else 'LOW'


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok', 'version': '2.0.0',
            'h3_resolution': H3_RESOLUTION, 'hotspot_threshold': HOTSPOT_THRESH}

@app.post('/predict')
def predict(req: PredictRequest):
    dt = (datetime.strptime(req.datetime_ist, '%Y-%m-%d %H:%M:%S')
          if req.datetime_ist else datetime.now(IST).replace(tzinfo=None))
    X, cell = build_features(req.latitude, req.longitude, dt, req.lag_1h, req.lag_24h)
    count = float(np.clip(reg_model.predict(X)[0], 0, None))
    prob  = float(clf_model.predict_proba(X)[0][1])
    return {'h3_cell': cell, 'latitude': req.latitude, 'longitude': req.longitude,
            'hour_ist': dt.hour, 'predicted_count': round(count, 2),
            'hotspot_prob': round(prob, 4), 'is_hotspot': prob>=0.5,
            'risk_level': risk_label(prob)}

@app.post('/predict/batch')
def predict_batch(req: BatchRequest):
    return [predict(loc) for loc in req.locations]

@app.get('/hotspots/now')
def hotspots_now(top_n: int = 10):
    now = datetime.now(IST).replace(tzinfo=None)
    results = []
    for cell, stats in list(cell_stats.items()):
        try:
            lat, lon = h3.cell_to_latlng(cell)
            X, _ = build_features(lat, lon, now)
            prob  = float(clf_model.predict_proba(X)[0][1])
            count = float(np.clip(reg_model.predict(X)[0], 0, None))
            results.append({'h3_cell': cell, 'lat': lat, 'lon': lon,
                            'predicted_count': round(count,1),
                            'hotspot_prob': round(prob,4),
                            'risk_level': risk_label(prob)})
        except: continue
    results.sort(key=lambda x: x['hotspot_prob'], reverse=True)
    return {'datetime_ist': now.strftime('%Y-%m-%d %H:%M'), 'hotspots': results[:top_n]}

@app.get('/hotspots/hour/{hour}')
def hotspots_by_hour(hour: int, top_n: int = 10):
    if not 0 <= hour <= 23:
        raise HTTPException(400, 'hour must be 0-23')
    base_dt = datetime.now(IST).replace(hour=hour, minute=0, second=0, tzinfo=None)
    results = []
    for cell, stats in list(cell_stats.items()):
        try:
            lat, lon = h3.cell_to_latlng(cell)
            X, _ = build_features(lat, lon, base_dt)
            prob  = float(clf_model.predict_proba(X)[0][1])
            count = float(np.clip(reg_model.predict(X)[0], 0, None))
            results.append({'h3_cell': cell, 'lat': lat, 'lon': lon,
                            'predicted_count': round(count,1),
                            'hotspot_prob': round(prob,4),
                            'risk_level': risk_label(prob)})
        except: continue
    results.sort(key=lambda x: x['hotspot_prob'], reverse=True)
    return {'hour_ist': f'{hour:02d}:00', 'hotspots': results[:top_n]}

@app.post('/fine/issue')
def issue_fine(req: FineRequest):
    """Officer confirms a violation → predict risk for this cell → issue challan"""
    now   = datetime.now(IST).replace(tzinfo=None)
    X, cell = build_features(req.latitude, req.longitude, now)
    prob    = float(clf_model.predict_proba(X)[0][1])
    challan = generate_challan(
        vehicle_number=req.vehicle_number,
        vehicle_type=req.vehicle_type,
        violation_type=req.violation_type,
        latitude=req.latitude,
        longitude=req.longitude,
        h3_cell=cell,
        officer_id=req.officer_id,
        hotspot_prob=prob,
        location_desc=req.location_desc,
    )
    return {'success': True, 'challan': challan}

@app.get('/fine/stats')
def fine_stats():
    return get_violation_stats()

@app.get('/fine/log')
def fine_log(limit: int = 20):
    import json, os
    if not os.path.exists('violations_issued.json'):
        return []
    with open('violations_issued.json') as f:
        records = json.load(f)
    return records[-limit:]


import pandas as pd
import numpy as np
import pickle, h3

# ── Load models ───────────────────────────────────────────────────────────
with open('xgb_regressor.pkl',  'rb') as f: reg_model  = pickle.load(f)
with open('xgb_classifier.pkl', 'rb') as f: clf_model  = pickle.load(f)
with open('model_meta.pkl',     'rb') as f: meta       = pickle.load(f)
scores_df = pd.read_csv('hotspot_scores.csv')

FEATURES      = meta['features']
H3_RESOLUTION = meta['h3_resolution']
le            = meta['label_encoder']
cell_stats    = scores_df.set_index('h3_cell')[['cell_hist_mean']].to_dict('index')

# ── Shifts ────────────────────────────────────────────────────────────────
SHIFTS = {
    'Morning  (06:00–14:00)': list(range(6, 14)),
    'Afternoon(14:00–22:00)': list(range(14, 22)),
    'Night    (22:00–06:00)': list(range(22, 24)) + list(range(0, 6)),
}

# Officers available per shift
OFFICERS_PER_SHIFT = 8

def score_cells_for_shift(hours: list) -> pd.DataFrame:
    """Score all known cells for a given set of hours and return ranked results."""
    rows = []
    for cell, stats in cell_stats.items():
        try:
            lat, lon = h3.cell_to_latlng(cell)
            shift_probs = []
            shift_counts = []
            for hr in hours:
                try:    cell_id = int(le.transform([cell])[0])
                except: cell_id = -1
                hist = stats.get('cell_hist_mean', scores_df['cell_hist_mean'].mean())
                row = {
                    'h3_cell_id': cell_id, 'lat_mean': lat, 'lon_mean': lon,
                    'cell_hist_mean': hist, 'hour': hr,
                    'day_of_week': 1, 'month': 3, 'week_of_year': 12,
                    'hour_sin': np.sin(2*np.pi*hr/24), 'hour_cos': np.cos(2*np.pi*hr/24),
                    'dow_sin': np.sin(2*np.pi/7), 'dow_cos': np.cos(2*np.pi/7),
                    'month_sin': np.sin(2*np.pi*3/12), 'month_cos': np.cos(2*np.pi*3/12),
                    'is_weekend': 0, 'is_peak_hour': int(hr in range(8,12) or hr in range(17,21)),
                    'is_morning': int(5<=hr<=11), 'is_night': int(hr>=22 or hr<=5),
                    'lag_1h': hist, 'lag_2h': hist, 'lag_3h': hist,
                    'lag_24h': hist, 'lag_168h': hist,
                    'rolling_3h_mean': hist, 'rolling_6h_mean': hist,
                    'rolling_24h_mean': hist, 'rolling_7d_mean': hist, 'rolling_24h_std': 0.0,
                    'pct_car': 0.4, 'pct_scooter': 0.3, 'pct_auto': 0.1, 'pct_maxi': 0.05,
                    'pct_wrong_park': 0.6, 'pct_no_park': 0.25,
                    'pct_main_road': 0.1, 'pct_footpath': 0.05, 'avg_vtype_count': 1.2,
                }
                X = pd.DataFrame([row])[FEATURES]
                shift_probs.append(float(clf_model.predict_proba(X)[0][1]))
                shift_counts.append(float(np.clip(reg_model.predict(X)[0], 0, None)))
            rows.append({
                'h3_cell': cell, 'lat': round(lat,5), 'lon': round(lon,5),
                'avg_hotspot_prob':  round(np.mean(shift_probs), 4),
                'max_hotspot_prob':  round(np.max(shift_probs), 4),
                'avg_pred_count':    round(np.mean(shift_counts), 1),
                'peak_hour':         hours[int(np.argmax(shift_probs))],
            })
        except: continue
    return pd.DataFrame(rows).sort_values('avg_hotspot_prob', ascending=False)


print('Generating patrol schedules for all shifts...\n')
all_schedules = {}

for shift_name, hours in SHIFTS.items():
    ranked = score_cells_for_shift(hours)
    top    = ranked.head(OFFICERS_PER_SHIFT).reset_index(drop=True)
    top.index += 1
    top['Officer Assigned'] = [f'OFF-{str(i).zfill(3)}' for i in range(1, len(top)+1)]
    top['Priority']         = ['🔴 CRITICAL' if p>=0.75 else '🟠 HIGH' if p>=0.5
                               else '🟡 MEDIUM' for p in top['avg_hotspot_prob']]
    all_schedules[shift_name] = top
    print(f'Shift: {shift_name}')
    print(top[['Officer Assigned','h3_cell','lat','lon',
               'avg_hotspot_prob','peak_hour','Priority']].to_string())
    print()


# ── Build a standalone HTML enforcement dashboard ────────────────────────
import json
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo('Asia/Kolkata')
now_str = datetime.now(IST).strftime('%d %b %Y, %H:%M IST')

# Load fine log
try:
    with open('violations_issued.json') as f:
        fine_log = json.load(f)
except:
    fine_log = []

total_fines   = len(fine_log)
total_revenue = sum(r['fine_amount'] for r in fine_log)

# Top hotspot zones from scores
top_zones = scores_df.sort_values('hotspot_prob', ascending=False).head(5)

# Build schedule table rows
morning_schedule = all_schedules.get(list(SHIFTS.keys())[0], pd.DataFrame())

def schedule_rows(df):
    rows = ''
    for _, r in df.iterrows():
        rows += (f'<tr><td>{r["Officer Assigned"]}</td>'
                 f'<td>{r["h3_cell"]}</td>'
                 f'<td>{r["lat"]}, {r["lon"]}</td>'
                 f'<td>{r["avg_hotspot_prob"]:.1%}</td>'
                 f'<td>{r["Priority"]}</td></tr>')
    return rows

def zone_rows(df):
    rows = ''
    for _, r in df.iterrows():
        prob = r.get('hotspot_prob', 0)
        color = '#dc3545' if prob>=0.75 else '#fd7e14' if prob>=0.5 else '#ffc107'
        rows += (f'<tr><td>{r["h3_cell"]}</td>'
                 f'<td>{r["lat_mean"]:.5f}, {r["lon_mean"]:.5f}</td>'
                 f'<td style="color:{color};font-weight:600">{prob:.1%}</td>'
                 f'<td>{r["pred_count"]:.1f}</td></tr>')
    return rows

html = f'''
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bangalore Parking Enforcement Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; }}
  .header {{ background: linear-gradient(135deg, #1e3a5f, #0f172a);
             padding: 20px 30px; border-bottom: 1px solid #334155; }}
  .header h1 {{ font-size: 22px; font-weight: 700; color: #f1f5f9; }}
  .header p  {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr);
              gap: 16px; padding: 24px 30px; }}
  .kpi {{ background: #1e293b; border-radius: 12px; padding: 20px;
          border: 1px solid #334155; }}
  .kpi .val {{ font-size: 32px; font-weight: 700; color: #38bdf8; }}
  .kpi .label {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
  .kpi.red .val  {{ color: #f87171; }}
  .kpi.green .val{{ color: #4ade80; }}
  .kpi.amber .val{{ color: #fbbf24; }}
  .section {{ padding: 0 30px 30px; }}
  .section h2 {{ font-size: 16px; font-weight: 600; color: #cbd5e1;
                 margin-bottom: 12px; padding-bottom: 8px;
                 border-bottom: 1px solid #334155; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #1e293b; color: #94a3b8; padding: 10px 12px;
        text-align: left; font-weight: 500; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #1e293b; color: #e2e8f0; }}
  tr:hover td {{ background: #1e293b; }}
  .badge {{ display:inline-block; padding: 2px 8px; border-radius: 20px;
            font-size: 11px; font-weight: 600; }}
  .badge-red   {{ background:#450a0a; color:#f87171; }}
  .badge-amber {{ background:#451a03; color:#fbbf24; }}
  .footer {{ padding: 16px 30px; font-size: 12px; color: #475569;
             border-top: 1px solid #1e293b; text-align: center; }}
</style></head><body>

<div class="header">
  <h1>🚔 Bangalore Traffic Police — AI Enforcement Dashboard</h1>
  <p>Last updated: {now_str} &nbsp;|&nbsp; Powered by XGBoost Hotspot Prediction Model</p>
</div>

<div class="kpi-row">
  <div class="kpi red">
    <div class="val">{top_zones["hotspot_prob"].iloc[0]:.0%}</div>
    <div class="label">Top Zone Hotspot Probability</div>
  </div>
  <div class="kpi amber">
    <div class="val">{len(scores_df[scores_df["hotspot_flag"]==1])}</div>
    <div class="label">Active Predicted Hotspot Cells</div>
  </div>
  <div class="kpi green">
    <div class="val">{total_fines}</div>
    <div class="label">Challans Issued Today</div>
  </div>
  <div class="kpi">
    <div class="val">₹{total_revenue:,}</div>
    <div class="label">Total Fines Collected</div>
  </div>
</div>

<div class="section">
  <h2>🔴 Top 5 Predicted Hotspot Zones</h2>
  <table><thead><tr><th>H3 Cell</th><th>Coordinates</th>
  <th>Hotspot Prob</th><th>Predicted Violations/hr</th></tr></thead>
  <tbody>{zone_rows(top_zones)}</tbody></table>
</div>

<div class="section">
  <h2>👮 Morning Shift Patrol Assignments (06:00–14:00)</h2>
  <table><thead><tr><th>Officer</th><th>Assigned Zone</th><th>Coordinates</th>
  <th>Risk Score</th><th>Priority</th></tr></thead>
  <tbody>{schedule_rows(morning_schedule)}</tbody></table>
</div>

<div class="footer">
  Bangalore Traffic Police AI System &nbsp;|&nbsp;
  Model: XGBoost H3-Res7 &nbsp;|&nbsp; Data: Jan–Apr 2024
</div>
</body></html>
'''

with open('enforcement_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('Dashboard saved → enforcement_dashboard.html')

from IPython.display import IFrame
IFrame('enforcement_dashboard.html', width='100%', height=650)
