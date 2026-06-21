# Generated from: solution.ipynb
# Converted at: 2026-06-19T09:28:01.663Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

pip install h3 pandas numpy matplotlib folium

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast,warnings
warnings.filterwarnings('ignore')
#hides all warnings

RAW_DATA_PARQUET = '/content/drive/MyDrive/gridlock_hackathon/jan to may police violation_anonymized791b166.parquet'
RAW_DATA_CSV = '/content/drive/MyDrive/gridlock_hackathon/jan to may police violation_anonymized791b166.csv'

if os.path.exists(RAW_DATA_PARQUET):
    df = pd.read_parquet(RAW_DATA_PARQUET)
else:
    df = pd.read_csv(RAW_DATA_CSV)

df.isnull().sum()

# Dropping 100% null columns


df.drop(['closed_datetime','action_taken_timestamp','description'],axis=1,inplace=True)

df.head()

# Timestamps are in UTC, converting to IST
# pd.to_datetime converts strings to pandas datetime objects, format tells the format in which date time is stored, uct tells that they are in utc timezone, then we convert timezone to asia/kolkata


df['ist_dt']=pd.to_datetime(df['created_datetime'],format='ISO8601',utc=True).dt.tz_convert('Asia/Kolkata')

#extracting time columns:
df['hour']=df['ist_dt'].dt.hour
df['day']=df['ist_dt'].dt.day
df['month']=df['ist_dt'].dt.month
df['year']=df['ist_dt'].dt.year

print("Date range (IST):", df['ist_dt'].min().date(), "→", df['ist_dt'].max().date())
print()

def time_bucket(h):
  if 7<=h<=10:
    return 'AM_PEAK'
  elif 11<=h<=16:
    return 'MIDDAY'
  elif 17<=h<=21:
    return 'PM_PEAK'
  else:
    return 'OFF_PEAK_NIGHT'
df['time_bucket']=df['hour'].apply(time_bucket)
print("Violations per time bucket: ")
print(df['time_bucket'].value_counts())


print(df['validation_status'])

df_approved=df[df['validation_status']=='approved']
df_approved.isnull().sum()

df_approved['month'].value_counts()

train=df_approved[df_approved['month']!=2]
test=df_approved[df_approved['month']==3]

fig, ax = plt.subplots(figsize=(13, 4))

hourly = df.groupby('hour').size()

colors = []
for h in hourly.index:
    if   17 <= h <= 21: colors.append('#E24B4A')   # PM Peak — red (blind spot)
    elif  7 <= h <= 10: colors.append('#185FA5')   # AM Peak — blue
    elif 11 <= h <= 16: colors.append('#888780')   # Midday — gray
    else:               colors.append('#C0C0C0')   # Night — light gray

ax.bar(hourly.index, hourly.values, color=colors, width=0.8, zorder=2)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)
ax.set_xlabel('Hour of day (IST)', fontsize=12)
ax.set_ylabel('Number of violations', fontsize=12)
ax.set_title('Parking violations by hour (IST) — 5–9 PM is the enforcement blind spot', fontsize=13)
ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right', fontsize=9)

# Annotation pointing at the PM Peak bars
ax.annotate(
    '''PM Peak 5–9pm
744 violations total
(0.25% of all records)''',
    xy=(19, 2000), xytext=(19, 12000),
    arrowprops=dict(arrowstyle='->', color='#A32D2D'),
    fontsize=10, color='#A32D2D', ha='center', fontweight='bold'
)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#185FA5', label='AM Peak 7–10am  (~105k)'),
    Patch(color='#888780', label='Midday 11am–4pm (~72k)'),
    Patch(color='#E24B4A', label='PM Peak 5–9pm   (744) ← blind spot'),
    Patch(color='#C0C0C0', label='Night 10pm–6am (~122k)')
], fontsize=9)

plt.tight_layout()
plt.savefig('enforcement_blind_spot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: enforcement_blind_spot.png")

import h3

lats=df['latitude'].values
longs=df['longitude'].values
h3_8_vals=[]
h3_9_vals=[]
for lat,lon in zip(lats,longs):
  try:
    h3_8=h3.latlng_to_cell(lat,lon,8)
    h3_8_vals.append(h3_8)
    h3_9=h3.latlng_to_cell(lat,lon,9)
    h3_9_vals.append(h3_9)
  except:
    h3_8_vals.append(None)
    h3_9_vals.append(None)
df['h3_8']=h3_8_vals
df['h3_9']=h3_9_vals
df=df.dropna(subset=['h3_8','h3_9'])

h3_overall=(
    df.groupby('h3_8')
    .agg(total_violations=('h3_8','count'))
    .reset_index()
)

centroids = h3_overall['h3_8'].apply(lambda c: pd.Series(h3.cell_to_latlng(c), index=['lat','lon']))
h3_overall = pd.concat([h3_overall, centroids], axis=1)

print("Top 10 H3-8 cells by total violations:")
top10 = h3_overall.sort_values('total_violations', ascending=False).head(10)
print(top10[['h3_8','total_violations','lat','lon']].to_string(index=False))

h3_by_bucket = (
    df.groupby(['h3_8','time_bucket'])
      .agg(violations=('id','count'))
      .reset_index()
)

h3_overall.to_csv('h3_layer_a.csv', index=False)
h3_by_bucket.to_csv('h3_by_bucket.csv', index=False)
print("\nSaved: h3_layer_a.csv, h3_by_bucket.csv")

from sklearn.cluster import DBSCAN
#haversine formula determines the great-circle distance between two points in a sphere given their longitudes and latitudes
#we consider two points between 200 meters as neighbours
EPS_RAD = 200 / 6_371_000
MIN_SAMP=30 #cluster must contain atleast 30 points to become a hotspot
#time periods to be analyzed separately
BUCKETS = [
    'AM_PEAK',
    'MIDDAY',
    'PM_PEAK',
    'OFF_PEAK_NIGHT'
]

bucket_clusters={}

for bucket in BUCKETS:
  subset=df[df['time_bucket']==bucket].dropna(
      subset=['latitude','longitude']
  ).copy()
  coords_rad = np.radians(
    subset[['latitude','longitude']].values)
  #converting latitude longitude to radians, dbscan requires radians
  db=DBSCAN(
      eps=EPS_RAD,
      min_samples=MIN_SAMP,
      metric='haversine',
      algorithm='ball_tree',
  ) #creates the clusteing model, ball tree is faster for geographic coordinates
  subset['cluster']=db.fit_predict(coords_rad)
  bucket_clusters[bucket]=subset
  n_clust = (subset['cluster'] >= 0).sum()
  n_noise = (subset['cluster'] == -1).sum()
  n_ids   = subset['cluster'].max() + 1
  print(f"[{bucket}]  {n_ids} clusters | {n_clust:,} clustered | {n_noise:,} noise | {len(subset):,} total")
print("\nDone. bucket_clusters dict has keys:", list(bucket_clusters.keys()))



import folium
import branca.colormap as cm
import pandas as pd
import numpy as np
from folium.plugins import GroupedLayerControl

BLR = [12.9716, 77.5946]
m = folium.Map(location=BLR, zoom_start=12, tiles='CartoDB positron')

CLUSTER_PRESETS = [10, 25, 50]

BUCKET_CONFIG = {
    'AM_PEAK': {
        'display_name': 'AM Peak',
        'hours': '07:00 - 10:59',
        'low_color': '#4A90E2',
        'high_color': '#0B3C73',
    },
    'MIDDAY': {
        'display_name': 'Midday',
        'hours': '11:00 - 16:59',
        'low_color': '#9B51E0',
        'high_color': '#3B136B',
    },
    'PM_PEAK': {
        'display_name': 'PM Peak',
        'hours': '17:00 - 21:59',
        'low_color': '#FF6B6B',
        'high_color': '#8B0000',
    },
    'OFF_PEAK_NIGHT': {
        'display_name': 'Off Peak Night',
        'hours': '22:00 - 06:59',
        'low_color': '#888888',
        'high_color': '#222222',
    },
}

layer_groups = {}

for bucket, config in BUCKET_CONFIG.items():
    data = bucket_clusters.get(bucket, pd.DataFrame())
    if 'cluster' not in data.columns or data.empty:
        continue

    valid_clusters = data[data['cluster'] >= 0]
    if valid_clusters.empty:
        continue

    cluster_counts = valid_clusters['cluster'].value_counts()

    group_name = f"{config['display_name']} ({config['hours']})"
    layer_groups[group_name] = []

    # Add an empty layer to serve as an Off switch for this group
    fg_hide = folium.FeatureGroup(name="Hide", show=False)
    fg_hide.add_to(m)
    layer_groups[group_name].append(fg_hide)

    for preset_n in CLUSTER_PRESETS:
        layer_label = f"Show Top {preset_n}"

        # Default to showing the Top 25 layer view on initial load
        show_by_default = (preset_n == 25)
        fg = folium.FeatureGroup(name=layer_label, show=show_by_default)

        top_cluster_ids = cluster_counts.head(preset_n).index.tolist()
        preset_data = valid_clusters[valid_clusters['cluster'].isin(top_cluster_ids)]

        if preset_data.empty:
            continue

        top_counts = cluster_counts.loc[top_cluster_ids]
        min_count = top_counts.min()
        max_count = top_counts.max()

        if min_count == max_count:
            min_count -= 1

        colormap = cm.LinearColormap(
            colors=[config['low_color'], config['high_color']],
            vmin=np.sqrt(min_count),
            vmax=np.sqrt(max_count)
        )

        for cluster_id in top_cluster_ids:
            cluster_points = preset_data[preset_data['cluster'] == cluster_id]
            total_violations = len(cluster_points)

            centroid_lat = cluster_points['latitude'].mean()
            centroid_lon = cluster_points['longitude'].mean()

            top_locations = (
                cluster_points['location']
                .dropna()
                .value_counts()
                .head(5)
                .index.tolist()
            )

            location_list_html = "".join([f"<li>{loc}</li>" for loc in top_locations])
            if not location_list_html:
                location_list_html = "<li>No location strings available</li>"

            popup_content = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 240px;">
                <b style="font-size: 14px; color: {config['high_color']};">{config['display_name']}</b><br>
                <b>Violations:</b> {total_violations:,}<br>
                <b>Centroid:</b> {centroid_lat:.5f}, {centroid_lon:.5f}<br>
                <hr style="margin: 8px 0; border: 0; border-top: 1px solid #ccc;">
                <b>Top Locations in Cluster:</b>
                <ul style="margin: 4px 0; padding-left: 18px; line-height: 1.3;">
                    {location_list_html}
                </ul>
            </div>
            """

            transformed_val = np.sqrt(total_violations)
            cluster_color = colormap(transformed_val)

            norm_rank = (transformed_val - np.sqrt(min_count)) / (np.sqrt(max_count) - np.sqrt(min_count))
            dynamic_opacity = 0.55 + (norm_rank * 0.35)

            folium.Circle(
                location=[centroid_lat, centroid_lon],
                radius=200,
                fill=True,
                fill_color=cluster_color,
                fill_opacity=dynamic_opacity,
                color='#1A1A1A',
                weight=2.0,
                opacity=0.95,
                popup=folium.Popup(popup_content, max_width=260)
            ).add_to(fg)

        fg.add_to(m)
        layer_groups[group_name].append(fg)

GroupedLayerControl(
    groups=layer_groups,
    exclusive_groups=list(layer_groups.keys()),
    collapsed=False
).add_to(m)

m.save('hotspot_map_v5.html')
print("Saved: hotspot_map_v5.html")

import folium
import branca.colormap as cm
import pandas as pd
import numpy as np
import h3
from folium.plugins import GroupedLayerControl

BLR = [12.9716, 77.5946]
m = folium.Map(location=BLR, zoom_start=12, tiles='CartoDB positron')

# Distinct, balanced color configurations with clear gradient spans
BUCKET_CONFIG = {
    'AM_PEAK': {
        'display_name': 'AM Peak',
        'hours': '07:00 - 10:59',
        'low_color': '#5D9CEC',   # Soft Blue
        'high_color': '#1B3A60',  # Dark Navy
    },
    'MIDDAY': {
        'display_name': 'Midday',
        'hours': '11:00 - 16:59',
        'low_color': '#AC92EC',   # Soft Purple
        'high_color': '#4A2A84',  # Deep Purple
    },
    'PM_PEAK': {
        'display_name': 'PM Peak',
        'hours': '17:00 - 21:59',
        'low_color': '#FC6E51',   # Soft Orange-Red
        'high_color': '#8A1F11',  # Deep Maroon
    },
    'OFF_PEAK_NIGHT': {
        'display_name': 'Off Peak Night',
        'hours': '22:00 - 06:59',
        'low_color': '#AAB2BD',   # Slate Gray
        'high_color': '#333A42',  # Dark Charcoal
    },
}

layer_groups = {}

for bucket, config in BUCKET_CONFIG.items():
    bucket_df = df[df['time_bucket'] == bucket]
    if bucket_df.empty:
        continue

    group_name = f"{config['display_name']} ({config['hours']})"
    layer_groups[group_name] = []

    # Radio button off switch for this specific timeframe group
    fg_hide = folium.FeatureGroup(name="Hide", show=False)
    fg_hide.add_to(m)
    layer_groups[group_name].append(fg_hide)

    # Process both resolutions requested
    for res in [8, 9]:
        col_name = f'h3_{res}'
        if col_name not in bucket_df.columns:
            continue

        cell_counts = bucket_df[col_name].value_counts()
        if cell_counts.empty:
            continue

        layer_label = f"Resolution {res}"
        # Display Resolution 8 views initially by default
        show_layer = (res == 8)
        fg = folium.FeatureGroup(name=layer_label, show=show_layer)

        min_count = cell_counts.min()
        max_count = cell_counts.max()
        if min_count == max_count:
            min_count -= 1

        # Establish linear color maps based on square root transformations
        colormap = cm.LinearColormap(
            colors=[config['low_color'], config['high_color']],
            vmin=np.sqrt(min_count),
            vmax=np.sqrt(max_count)
        )

        # Pre-compute top 5 locations per cell to accelerate loop processing
        top_locations_map = (
            bucket_df.groupby(col_name)['location']
            .apply(lambda x: x.dropna().value_counts().head(5).index.tolist())
            .to_dict()
        )

        for cell_id, total_violations in cell_counts.items():
            # Extract spatial boundary coordinates and center point
            vertices = h3.cell_to_boundary(cell_id)
            centroid = h3.cell_to_latlng(cell_id)

            loc_list = top_locations_map.get(cell_id, [])
            location_html = "".join([f"<li>{loc}</li>" for loc in loc_list])
            if not location_html:
                location_html = "<li>No location data available</li>"

            popup_content = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 250px;">
                <b style="font-size: 14px; color: {config['high_color']};">{config['display_name']} (Res {res})</b><br>
                <b>H3 Index:</b> {cell_id}<br>
                <b>Total Violations:</b> {total_violations:,}<br>
                <b>Centroid:</b> {centroid[0]:.5f}, {centroid[1]:.5f}<br>
                <hr style="margin: 8px 0; border: 0; border-top: 1px solid #ccc;">
                <b>Top Locations in Cell:</b>
                <ul style="margin: 4px 0; padding-left: 18px; line-height: 1.3;">
                    {location_html}
                </ul>
            </div>
            """

            # Dynamic opacity scaling using square-root parameters
            transformed_val = np.sqrt(total_violations)
            cell_color = colormap(transformed_val)

            norm_rank = (transformed_val - np.sqrt(min_count)) / (np.sqrt(max_count) - np.sqrt(min_count)) if max_count != min_count else 0.5
            dynamic_opacity = 0.45 + (norm_rank * 0.35)

            # Map the complete geographic hexagon polygon
            folium.Polygon(
                locations=vertices,
                fill=True,
                fill_color=cell_color,
                fill_opacity=dynamic_opacity,
                color='#2C3E50',  # Distinct slate outline definition
                weight=1.2,
                opacity=0.75,
                popup=folium.Popup(popup_content, max_width=270)
            ).add_to(fg)

        fg.add_to(m)
        layer_groups[group_name].append(fg)

# Render the final grouped selection interface
GroupedLayerControl(
    groups=layer_groups,
    exclusive_groups=list(layer_groups.keys()),
    collapsed=False
).add_to(m)

m.save('h3_hex_map.html')
print("Saved: h3_hex_map.html")

df.to_csv('violations_with_h3.csv', index=False)
print("Saved: violations_with_h3.csv")


print(f"H3-8 cells assigned:  {df['h3_8'].notna().sum():,} rows")
print(f"Unique H3-8 cells:    {df['h3_8'].nunique():,}")
print()
print("Cluster counts by bucket:")
for b in ['AM_PEAK','MIDDAY','OFF_PEAK_NIGHT']:
    d = bucket_clusters.get(b, pd.DataFrame())
    if 'cluster' in d.columns:
        n_clust = d['cluster'].max() + 1
        n_in    = (d['cluster'] >= 0).sum()
        print(f"  {b}: {n_clust} clusters, {n_in:,} records assigned")


# # Methodology: Statistical Redefinition of the Congestion Impact Score (CIS)
# 
# Rather than relying on arbitrary point allocations (e.g., assigning subjective weights for vehicle sizes or street types), this methodology implements an empirically grounded **Congestion Impact Score (CIS)** framework.
# 
# The approach  combines **traffic engineering standards**, **Uber H3 discrete global grids**, and **Getis-Ord Gi\*** spatial statistics. The final output maps estimated road-capacity reduction to a statistically validated hotspot score.
# 
# ---
# 
# ## Core Pipeline
# 
# The calculation follows a four-stage architecture that transforms raw parking violation records into an enforcement-prioritization framework:
# 
# ```text
# [Raw Violations Data]
#             │
#             ▼
# 1. Traffic Friction Weighting
#    (IRC:106-1990 PCE × HCM Ch.18 Location Factor)
#             │
#             ▼
# 2. H3 Spatial Aggregation
#    (Resolution 8 Hexagonal Grid Cells)
#             │
#             ▼
# 3. Spatial Autocorrelation
#    (PySAL Row-Standardized Neighbor Matrix W)
#             │
#             ▼
# 4. Local Getis-Ord Gi*
#             │
#             ├──► Z-Score (Intensity)
#             └──► P-Value (Statistical Significance)
#             │
#             ▼
# [Multi-Temporal Priority Classification]
#             (P1–P4 Tiers)
# ```
# 
# ---
# 
# ## 1. Traffic Friction Weighting (Physical Baseline)
# 
# Each violation is assigned a baseline impact weight derived from established traffic-engineering standards. The objective is to estimate how much road capacity is obstructed by the illegally parked vehicle.
# 
# ### Formula
# 
# $$
# \text{Friction Weight}
# =
# \text{Passenger Car Equivalent (PCE)}
# \times
# \text{Location Factor}
# $$
# 
# ---
# 
# ### A. Passenger Car Equivalent (PCE)
# 
# Vehicle weights are sourced from **Indian Roads Congress (IRC:106-1990, Table 5)**, which standardizes vehicles relative to a passenger car.
# 
# | Vehicle Type | PCE |
# |-------------|----:|
# | Bus / Truck / Lorry / HMV | 3.7 |
# | Mini Bus / Maxi Cab | 2.0 |
# | Car / Taxi / Jeep / LMV | 1.0 |
# | Auto-Rickshaw / Three-Wheeler | 0.75 |
# | Motorcycle / Scooter / Two-Wheeler | 0.5 |
# 
# A parked bus therefore contributes approximately **7.4×** the traffic obstruction of a parked motorcycle.
# 
# ---
# 
# ### B. Location Factor
# 
# Location multipliers are derived from the **Highway Capacity Manual (HCM 6th Edition, Chapter 18)** and represent the effect of illegal parking on roadway capacity.
# 
# | Location Type | Factor | Interpretation |
# |--------------|-------:|---------------|
# | Junction / Intersection / Signal / Zebra Crossing | 1.50 | Approximately 50% reduction in saturation flow due to turning-movement blockage |
# | Main Road / Arterial / National Highway | 1.30 | Approximately 30% reduction in roadway capacity |
# | Side Road / Local Street | 1.00 | Baseline impact |
# 
# Thus:
# 
# $$
# \text{Friction Weight}
# =
# \text{PCE}
# \times
# \text{Location Factor}
# $$
# 
# Example:
# 
# $$
# 3.7 \times 1.5 = 5.55
# $$
# 
# An illegally parked bus near a junction therefore contributes a friction weight of **5.55**.
# 
# ---
# 
# ## 2. Spatial Aggregation Using H3 Hexagons
# 
# ### H3 Resolution 8 Grid
# 
# All weighted violations are spatially aggregated into **Uber H3 Resolution 8 hexagonal cells** (`h3_8`).
# 
# Each cell covers approximately **0.7 km²**, providing a uniform geographic unit for analysis.
# 
# Aggregation is performed as:
# 
# $$
# \text{Weighted PCE}_{cell}
# =
# \sum_{i=1}^{n}
# \text{Friction Weight}_i
# $$
# 
# This converts hundreds of thousands of point violations into a citywide spatial surface of traffic friction.
# 
# ---
# 
# ### Neighborhood Structure
# 
# Traffic congestion rarely remains confined to a single street segment.
# 
# To account for spillover effects, each H3 cell is connected to its immediate neighboring hexagons using a **row-standardized spatial weights matrix** ($W$).
# 
# For every cell:
# 
# - Immediate neighbors are defined using `k = 1`
# - Each cell has up to six neighboring hexagons
# - Neighbor contributions are normalized through row standardization
# 
# This allows the model to evaluate congestion as a neighborhood phenomenon rather than an isolated point event.
# 
# ---
# 
# ## 3. Local Getis-Ord Gi* Hotspot Analysis
# 
# To distinguish structural congestion hotspots from routine enforcement activity, the aggregated data is analyzed using the **Local Getis-Ord Gi\*** statistic.
# 
# Implementation:
# 
# ```python
# G_Local(..., star=True)
# ```
# 
# The statistic evaluates whether a cell and its neighboring cells contain significantly higher friction values than expected under a random spatial distribution.
# 
# ---
# 
# ### Outputs
# 
# #### Gi* Z-Score
# 
# $$
# Z_i
# $$
# 
# The Z-score measures how many standard deviations a local cluster lies above or below the citywide average.
# 
# | Z-Score | Meaning |
# |---------|---------|
# | Near 0 | Normal variation |
# | > 1.96 | Significant hotspot (95% confidence) |
# | > 2.58 | Significant hotspot (99% confidence) |
# | < -1.96 | Significant coldspot |
# 
# Higher positive values indicate stronger and more concentrated congestion hotspots.
# 
# ---
# 
# #### Gi* P-Value
# 
# $$
# p_i
# $$
# 
# The p-value represents the probability that the observed hotspot occurred by chance.
# 
# | P-Value | Interpretation |
# |---------|---------------|
# | < 0.05 | Statistically significant |
# | < 0.01 | Highly significant |
# | ≥ 0.05 | Not significant |
# 
# ---
# 
# ### Hotspot Classification
# 
# | Class | Condition |
# |--------|-----------|
# | HH_99 | Hotspot, p < 0.01 |
# | HH_95 | Hotspot, p < 0.05 |
# | NS | Not significant |
# | LL_95 | Coldspot, p < 0.05 |
# | LL_99 | Coldspot, p < 0.01 |
# 
# ---
# 
# ### Visualization Scaling
# 
# To prevent extreme outliers from dominating the map visualization, heatmap intensities are scaled using a square-root transformation:
# 
# $$
# f(x)=\sqrt{x}
# $$
# 
# This preserves visibility of medium-sized hotspots while still emphasizing the strongest clusters.
# 
# ---
# 
# ## 4. Cross-Temporal Priority Classification
# 
# Congestion patterns vary substantially throughout the day.
# 
# To separate persistent infrastructure bottlenecks from time-specific traffic surges, Gi\* analysis is performed independently across four operational periods:
# 
# | Time Bucket | Hours |
# |------------|-------|
# | AM Peak | 07:00–10:59 |
# | Midday | 11:00–16:59 |
# |PM Peak|17:00-21:59|
# | Off-Peak Night | 22:00–06:59 |
# 
# 
# 
# ## Strategic ROI Framework
# 
# Rather than utilizing a simple temporal significance tally (which can overemphasize minor, high-volume zones while missing catastrophic single-point choke points), this notebook implements an operational **Strategic ROI Framework**.
# 
# By mapping **absolute violation counts** (enforcement volume) against **cumulative PCE-weighted congestion metrics** (spatial friction), active cells are categorized into distinct, high-impact tactical deployment tiers.
# 
# ---
# 
# ## Mathematical Class Boundaries
# 
# Let $V_i$ be the total absolute violation count in cell $i$.
# 
# Let $C_i$ be the cumulative PCE-weighted congestion metric in cell $i$, defined as:
# 
# $$
# C_i = \sum_{j=1}^{n} (\text{PCE}_j \times \text{Location Factor}_j)
# $$
# 
# We define the following operational thresholds using the 90th percentile of the active spatial grid:
# 
# ### High Volume Threshold
# 
# $$
# \tilde{V} = P_{90}(V)
# $$
# 
# ### High Congestion Threshold
# 
# $$
# \tilde{C} = P_{90}(C)
# $$
# 
# Using these thresholds, every cell is mapped into one of four operational profiles.
# 
# ---
# 
# ### Operational Classification Matrix
# 
# ```text
#                     High Congestion (C_i ≥ C̃)
#                                   │
#                                   │
#       TIER 2: SILENT BOTTLENECK   │   TIER 1: MAX DISRUPTION
#       (High Impact, Low Volume)   │   (High Impact, High Volume)
#                                   │
# Low Volume ───────────────────────┼──────────────────── High Volume
# (V_i < Ṽ)                        │                    (V_i ≥ Ṽ)
#                                   │
#       STANDARD TRAFFIC ZONE       │   TIER 3: VOLUME HOTSPOT
#                                   │   (Low Impact, High Volume)
#                                   │
#                     Low Congestion (C_i < C̃)
# ```
# 
# ---
# 
# ## Targeted High-ROI Tiers & Operational Actions
# 
# ### Tier 1: Max Disruption
# 
# **Mathematical Criteria**
# 
# $$
# V_i \ge \tilde{V}
# \quad \land \quad
# C_i \ge \tilde{C}
# $$
# 
# **Operational Characterization**
# 
# High Congestion + High Volume. Critical areas where massive violation volumes directly intersect with major arterial road signals or zebra crossings.
# 
# **Tactical Enforcement Deployment**
# 
# Primary Target: Permanent stationary intervention, high-frequency towing patrols, or automated parking enforcement cameras.
# 
# ---
# 
# ### Tier 2: Silent Bottleneck
# 
# **Mathematical Criteria**
# 
# $$
# V_i < \tilde{V}
# \quad \land \quad
# C_i \ge \tilde{C}
# $$
# 
# **Operational Characterization**
# 
# High Congestion + Low-Medium Volume. Although violation counts are relatively low, the spatial friction is extremely high. These zones are often caused by one or two illegally parked heavy vehicles (buses or trucks) positioned near critical intersections.
# 
# **Tactical Enforcement Deployment**
# 
# Surgical Strike: Targeted zero-tolerance towing sweeps. Removing even a small number of obstructions can yield the highest traffic-flow ROI per patrol hour.
# 
# ---
# 
# ### Tier 3: Volume Hotspot
# 
# **Mathematical Criteria**
# 
# $$
# V_i \ge \tilde{V}
# \quad \land \quad
# C_i < \tilde{C}
# $$
# 
# **Operational Characterization**
# 
# Low-Medium Congestion + High Volume. Large numbers of parking infractions that do not significantly affect arterial traffic flow (e.g., clusters of two-wheelers on local streets).
# 
# **Tactical Enforcement Deployment**
# 
# Administrative Clean-Up: Rapid-response ticketing units, digital e-challans, or light patrol sweeps to improve compliance without allocating heavy towing resources.
# 
# ---
# 
# ### Standard Traffic Zone
# 
# **Mathematical Criteria**
# 
# All remaining cells.
# 
# **Operational Characterization**
# 
# Baseline daily traffic variation.
# 
# **Tactical Enforcement Deployment**
# 
# Reactive only. No proactive resource allocation required.
# 
# 
# 
# the methodology answers:
# 
# > “Which locations contribute the greatest estimated road-capacity reduction and form statistically significant spatial hotspots across multiple time periods?”
# 
# This distinction allows enforcement resources to be directed toward persistent, system-level congestion bottlenecks rather than isolated high-volume locations.


import pandas as pd
import numpy as np
import h3
import ast
import folium
import branca.colormap as cm
from libpysal.weights import W
from esda.getisord import G_Local

# Load the primary dataset
try:
    violations_data = pd.read_csv('violations_with_h3.csv')
except FileNotFoundError:
    print("Warning: violations_with_h3.csv not found. Simulating data for testing...")
    np.random.seed(42)
    n = 10000
    mock_cells = ['88618925c1fffff', '88618925c03ffff', '8861892729fffff', '8860145a33fffff', '8860145b43fffff']
    violations_data = pd.DataFrame({
        'id': range(n),
        'latitude': np.random.uniform(12.90, 13.05, n),
        'longitude': np.random.uniform(77.50, 77.70, n),
        'location': np.random.choice(["Koramangala 18th Main", "Sarjapura Junction", "Kalidasa Road Near Signal"], n),
        'vehicle_type': np.random.choice(['CAR', 'SCOOTER', 'BUS (BMTC/KSRTC)', 'AUTO', 'GOODS AUTO'], n),
        'junction_name': np.random.choice([np.nan, "Silk Board Signal", "Sony World Junction", "Domlur Flyover"], n),
        'hour': np.random.randint(0, 24, n),
    })
    violations_data['h3_8'] = np.random.choice(mock_cells, n)
    violations_data['h3_9'] = violations_data['h3_8'].apply(lambda x: x[:-1] + 'a')

# Setup time buckets
if 'time_bucket' not in violations_data.columns:
    def assign_time_bucket(h):
        if 7 <= h <= 10:    return 'AM_PEAK'
        elif 11 <= h <= 16: return 'MIDDAY'
        elif 17 <= h <= 21: return 'PM_PEAK'
        else:               return 'OFF_PEAK_NIGHT'
    violations_data['time_bucket'] = violations_data['hour'].apply(assign_time_bucket)

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

violations_data['pce'] = violations_data['vehicle_type'].apply(get_pce)

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

violations_data['loc_factor'] = violations_data.apply(get_loc_factor_refined, axis=1)
violations_data['pce_weighted'] = violations_data['pce'] * violations_data['loc_factor']

# Global H3 aggregation
h3_agg = (
    violations_data.groupby('h3_8')
    .agg(
        weighted_pce=('pce_weighted', 'sum'),
        total_violations=('id', 'count')
    )
    .reset_index()
)

centroids = h3_agg['h3_8'].apply(lambda c: pd.Series(h3.cell_to_latlng(c), index=['lat', 'lon']))
h3_agg = pd.concat([h3_agg, centroids], axis=1)

# Build H3 spatial weights grid
cells = h3_agg['h3_8'].tolist()
cell_set = set(cells)
c2i = {c: i for i, c in enumerate(cells)}

neighbors_dict = {}
for cell in cells:
    ring1 = set(h3.grid_disk(cell, k=1)) - {cell}
    neighbors_dict[c2i[cell]] = [c2i[n] for n in ring1 if n in cell_set]

w = W(neighbors_dict, silence_warnings=True)
w.transform = 'r'

dominant_junctions = (
    violations_data.dropna(subset=['junction_name'])
    .groupby('h3_8')['junction_name']
    .apply(lambda x: x.value_counts().index[0] if not x.empty else "Unnamed Corridor")
    .to_dict()
)

# Initialize map
BLR = [12.9716, 77.5946]
m = folium.Map(location=BLR, zoom_start=12, tiles='CartoDB positron')

# Config for time buckets
BUCKETS = {
    'AM_PEAK': {
        'prefix': '[AM Peak 07-11]',
        'low': '#5D9CEC', 'high': '#1B3A60',
        'hot_low': '#FFB74D', 'hot_high': '#B71C1C'
    },
    'MIDDAY': {
        'prefix': '[Midday 11-17]',
        'low': '#AC92EC', 'high': '#4A2A84',
        'hot_low': '#CE93D8', 'hot_high': '#4A148C'
    },
    'PM_PEAK': {
        'prefix': '[PM Peak 17-22]',
        'low': '#FC6E51', 'high': '#8A1F11',
        'hot_low': '#FFAB91', 'hot_high': '#BF360C'
    },
    'OFF_PEAK_NIGHT': {
        'prefix': '[Night 22-07]',
        'low': '#AAB2BD', 'high': '#333A42',
        'hot_low': '#B0BEC5', 'hot_high': '#263238'
    }
}

roi_palette = {
    'Tier 1: Max Disruption': '#D32F2F',
    'Tier 2: Silent Bottleneck': '#FF9800',
    'Tier 3: Volume Hotspot': '#1976D2'
}

for bucket, config in BUCKETS.items():
    bucket_raw = violations_data[violations_data['time_bucket'] == bucket]
    if bucket_raw.empty:
        continue

    bucket_agg = (
        bucket_raw.groupby('h3_8')
        .agg(
            violations=('id', 'count'),
            weighted_pce=('pce_weighted', 'sum')
        )
        .reindex(h3_agg['h3_8'], fill_value=0)
        .reset_index()
    )
    bucket_agg = bucket_agg.merge(h3_agg[['h3_8', 'lat', 'lon']], on='h3_8')

    # Compute local Gi* statistics
    y = bucket_agg['weighted_pce'].values.astype(float)
    if np.std(y) > 0:
        g_local = G_Local(y, w, transform='R', permutations=0, star=True)
        bucket_agg['gi_zscore'] = g_local.Zs
        bucket_agg['gi_pvalue'] = g_local.p_norm
    else:
        bucket_agg['gi_zscore'] = 0.0
        bucket_agg['gi_pvalue'] = 1.0

    # Calculate active-cell thresholds
    active_cells = bucket_agg[bucket_agg['violations'] > 0]
    if not active_cells.empty:
        vol_threshold = active_cells['violations'].quantile(0.90)
        pce_threshold = active_cells['weighted_pce'].quantile(0.90)
    else:
        vol_threshold = 1.0
        pce_threshold = 1.0

    # 1. RAW VIOLATIONS LAYER
    fg_raw8 = folium.FeatureGroup(name=f"{config['prefix']} Raw Violations", show=False)
    cell_counts_8 = bucket_raw['h3_8'].value_counts()

    if not cell_counts_8.empty:
        colormap_8 = cm.LinearColormap(
            colors=[config['low'], config['high']],
            vmin=np.sqrt(cell_counts_8.min()),
            vmax=np.sqrt(cell_counts_8.max())
        )
        for cell_id, count in cell_counts_8.items():
            vertices = h3.cell_to_boundary(cell_id)
            junc_desc = dominant_junctions.get(cell_id, "Arterial Segment")

            popup_html = f"""
            <div style="font-family: Arial; font-size: 11px; width: 220px;">
                <b style="color: {config['high']}; font-size: 13px;">{junc_desc}</b><br>
                <b>Raw Violations:</b> {count:,}<br>
                <b>H3 Index:</b> {cell_id}<br>
            </div>
            """
            trans_val = np.sqrt(count)
            folium.Polygon(
                locations=vertices,
                fill=True,
                fill_color=colormap_8(trans_val),
                fill_opacity=0.5,
                color='#2C3E50',
                weight=0.8,
                popup=folium.Popup(popup_html, max_width=240)
            ).add_to(fg_raw8)
    fg_raw8.add_to(m)

    # 2. CONGESTION IMPACT HOTSPOTS (Gi*)
    fg_hotspots = folium.FeatureGroup(name=f"{config['prefix']} Congestion Hotspots", show=False)
    sig_cells = bucket_agg[bucket_agg['gi_pvalue'] < 0.05]

    if not sig_cells.empty:
        colormap_z = cm.LinearColormap(
            colors=[config['hot_low'], config['hot_high']],
            vmin=sig_cells['gi_zscore'].min(),
            vmax=sig_cells['gi_zscore'].max()
        )
        for _, row in sig_cells.iterrows():
            vertices = h3.cell_to_boundary(row['h3_8'])
            junc_desc = dominant_junctions.get(row['h3_8'], "Arterial Segment")

            popup_html = f"""
            <div style="font-family: Arial; font-size: 11px; width: 220px;">
                <b style="color: {config['hot_high']}; font-size: 13px;">{junc_desc}</b><br>
                <span style="color:#d32f2f; font-weight:bold;">Verified Hotspot</span><br>
                <b>Gi* Z-Score:</b> {row['gi_zscore']:.2f}<br>
                <b>p-value:</b> {row['gi_pvalue']:.4f}<br>
                <b>PCE Score:</b> {row['weighted_pce']:.1f}<br>
            </div>
            """
            folium.Polygon(
                locations=vertices,
                fill=True,
                fill_color=colormap_z(row['gi_zscore']),
                fill_opacity=0.6,
                color='#111111',
                weight=1.2,
                popup=folium.Popup(popup_html, max_width=240)
            ).add_to(fg_hotspots)
    fg_hotspots.add_to(m)

    # 3. HIGH-ROI STRATEGIC ENFORCEMENT TARGETS
    fg_roi = folium.FeatureGroup(name=f"{config['prefix']} Recommended Targets", show=False)

    def classify_roi(row):
        if row['violations'] == 0:
            return 'Standard'
        is_high_volume = row['violations'] >= vol_threshold
        is_high_congestion = row['weighted_pce'] >= pce_threshold

        if is_high_volume and is_high_congestion:
            return 'Tier 1: Max Disruption'
        elif is_high_congestion and not is_high_volume:
            return 'Tier 2: Silent Bottleneck'
        elif is_high_volume and not is_high_congestion:
            return 'Tier 3: Volume Hotspot'
        else:
            return 'Standard'

    bucket_agg['roi_class'] = bucket_agg.apply(classify_roi, axis=1)
    targets = bucket_agg[bucket_agg['roi_class'] != 'Standard']

    for _, row in targets.iterrows():
        color = roi_palette.get(row['roi_class'], '#757575')
        junc_desc = dominant_junctions.get(row['h3_8'], "Critical Intersection")

        popup_html = f"""
        <div style="font-family: Arial; font-size: 11px; width: 240px; line-height:1.4;">
            <b style="font-size:13px; color:{color};">{junc_desc}</b><br>
            <span style="background-color: {color}22; color:{color}; padding: 2px 4px; border-radius:3px; font-weight:bold; font-size:10px;">
                {row['roi_class']}
            </span>
            <hr style="margin:6px 0; border-top:1px solid #ddd;">
            <b>Timeframe Thresholds:</b><br>
            • Volume Cutoff: <b>{vol_threshold:.1f}</b> (Observed: <b>{row['violations']}</b>)<br>
            • Congestion Cutoff: <b>{pce_threshold:.1f}</b> (Observed: <b>{row['weighted_pce']:.1f}</b>)<br>
            <hr style="margin:6px 0; border-top:1px solid #ddd;">
            <b>Active Criteria Met:</b><br>
            {"• Violations & Congestion both exceed the 90th percentile." if 'Max' in row['roi_class'] else ""}
            {"• High congestion impact despite lower, non-90th percentile volumes." if 'Silent' in row['roi_class'] else ""}
            {"• High absolute violations but minimal impact on flow saturation." if 'Volume' in row['roi_class'] else ""}
        </div>
        """

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=10 if 'Max' in row['roi_class'] else (8 if 'Silent' in row['roi_class'] else 6),
            color='#FFFFFF',
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(fg_roi)
    fg_roi.add_to(m)

# Standard layer control to output checkboxes instead of radio buttons
folium.LayerControl(collapsed=False).add_to(m)

m.save('h3_integrated_hotspot_analysis.html')
print("Successfully generated clean checkbox map: h3_integrated_hotspot_analysis.html")
