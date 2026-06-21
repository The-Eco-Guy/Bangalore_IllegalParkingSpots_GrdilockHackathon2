import data_processing
import prediction
import scoring
import pandas as pd
import numpy as np

hist_df = data_processing.load_historical_data()
cell_meta = data_processing.load_cell_metadata()
pred_df = prediction.load_predictions_7d()
model_meta = prediction.load_meta()

min_date = hist_df['date'].min()
max_date = hist_df['date'].max()
start_date = max_date - pd.Timedelta(days=14)
end_date = max_date
time_bucket = 'All'
hour = 'All'

df_filtered = data_processing.filter_aggregated_data(
    hist_df, start_date, end_date, time_bucket, hour
)
all_cells = cell_meta['h3_8'].tolist()

df_scored = scoring.compute_gi_star(df_filtered, all_cells, 'weighted_pce', 'h3_8')
df_scored, vol_thresh, pce_thresh = scoring.add_roi_classification(df_scored, 'violations', 'weighted_pce')
df_scored = df_scored.merge(cell_meta, on='h3_8', how='left', suffixes=('', '_meta'))

print("HISTORICAL:")
print("df_scored cols:", list(df_scored.columns))
print("lat nulls:", df_scored['lat'].isnull().sum() if 'lat' in df_scored.columns else 'lat NOT in columns')
print("lat_center nulls:", df_scored['lat_center'].isnull().sum() if 'lat_center' in df_scored.columns else 'lat_center NOT in columns')

df_scored['lat'] = df_scored.get('lat', pd.Series(dtype=float)).fillna(df_scored.get('lat_center', 0.0))
df_scored['lon'] = df_scored.get('lon', pd.Series(dtype=float)).fillna(df_scored.get('lon_center', 0.0))
df_scored_clean = df_scored.dropna(subset=['lat_center', 'lon_center'])
print("df_scored_clean length:", len(df_scored_clean))
print("df_scored_clean sample coords:\n", df_scored_clean[['h3_8', 'lat_center', 'lon_center']].head())

print("\nPREDICTIONS:")
selected_f_date = pred_df['date'].min()
df_f_filtered = prediction.filter_predictions(
    pred_df, date_str=str(selected_f_date), hour='All', time_bucket='All'
)
df_f_scored = scoring.compute_gi_star(df_f_filtered, all_cells, 'pred_weighted_pce', 'h3_8')
df_f_scored, f_vol_t, f_pce_t = scoring.add_roi_classification(df_f_scored, 'pred_count', 'pred_weighted_pce')
df_f_scored = df_f_scored.merge(cell_meta, on='h3_8', how='left', suffixes=('', '_meta'))
print("df_f_scored cols:", list(df_f_scored.columns))
print("lat nulls:", df_f_scored['lat'].isnull().sum() if 'lat' in df_f_scored.columns else 'lat NOT in columns')
df_f_scored_clean = df_f_scored.dropna(subset=['lat_center', 'lon_center'])
print("df_f_scored_clean length:", len(df_f_scored_clean))
