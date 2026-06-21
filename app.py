import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import h3
import pickle
import os
import hashlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import DBSCAN

# Import custom modules
import data_processing
import scoring
import prediction

# Set page configuration
st.set_page_config(
    page_title="Bangalore Gridlock Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions and callbacks for dynamic sidebar filters synchronization
def get_hour_options_for_bucket(bucket):
    if bucket == "All":
        return ["All"] + list(range(24))
    elif bucket == "AM Peak (07:00 - 10:59)":
        return ["All"] + list(range(7, 11))
    elif bucket == "Midday (11:00 - 16:59)":
        return ["All"] + list(range(11, 17))
    elif bucket == "PM Peak (17:00 - 21:59)":
        return ["All"] + list(range(17, 22))
    elif bucket == "Off-Peak Night (22:00 - 06:59)":
        return ["All"] + [22, 23, 0, 1, 2, 3, 4, 5, 6]
    return ["All"]

def on_hist_hour_change():
    h = st.session_state.get('hist_hour', "All")
    if h == "All":
        pass
    else:
        h_val = int(h)
        if 7 <= h_val <= 10:
            st.session_state['hist_time_bucket'] = "AM Peak (07:00 - 10:59)"
        elif 11 <= h_val <= 16:
            st.session_state['hist_time_bucket'] = "Midday (11:00 - 16:59)"
        elif 17 <= h_val <= 21:
            st.session_state['hist_time_bucket'] = "PM Peak (17:00 - 21:59)"
        else:
            st.session_state['hist_time_bucket'] = "Off-Peak Night (22:00 - 06:59)"

def on_hist_bucket_change():
    bucket = st.session_state.get('hist_time_bucket', "All")
    h = st.session_state.get('hist_hour', "All")
    if h != "All":
        h_val = int(h)
        valid_hours = data_processing.get_hours_for_bucket(bucket)
        if bucket != "All" and h_val not in valid_hours:
            st.session_state['hist_hour'] = "All"

def on_pred_hour_change():
    h = st.session_state.get('pred_hour', "All")
    if h == "All":
        pass
    else:
        h_val = int(h)
        if 7 <= h_val <= 10:
            st.session_state['pred_time_bucket'] = "AM Peak (07:00 - 10:59)"
        elif 11 <= h_val <= 16:
            st.session_state['pred_time_bucket'] = "Midday (11:00 - 16:59)"
        elif 17 <= h_val <= 21:
            st.session_state['pred_time_bucket'] = "PM Peak (17:00 - 21:59)"
        else:
            st.session_state['pred_time_bucket'] = "Off-Peak Night (22:00 - 06:59)"

def on_pred_bucket_change():
    bucket = st.session_state.get('pred_time_bucket', "All")
    h = st.session_state.get('pred_hour', "All")
    if h != "All":
        h_val = int(h)
        valid_hours = data_processing.get_hours_for_bucket(bucket)
        if bucket != "All" and h_val not in valid_hours:
            st.session_state['pred_hour'] = "All"

# Custom premium CSS styling for dark mode and glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom background gradient */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%) !important;
        color: #f8fafc !important;
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* Premium glassmorphic cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    
    /* KPI block */
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(30, 41, 59, 0.5) !important;
        padding: 8px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #38bdf8 !important;
        color: #020617 !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Metric Card colors */
    .metric-red { color: #f87171 !important; }
    .metric-orange { color: #fb923c !important; }
    .metric-purple { color: #c084fc !important; }
</style>
""", unsafe_allow_html=True)

# Helper function to cache load operations
@st.cache_data
def load_all_data():
    hist_df = data_processing.load_historical_data()
    dbscan_df = data_processing.load_dbscan_violations()
    cell_meta = data_processing.load_cell_metadata()
    pred_df = prediction.load_predictions_7d()
    model_meta = prediction.load_meta()
    return hist_df, dbscan_df, cell_meta, pred_df, model_meta

# Load datasets
try:
    hist_df, dbscan_df, cell_meta, pred_df, model_meta = load_all_data()
    all_cells = cell_meta['h3_8'].tolist()
    global_pred_count_75th = float(max(0.05, pred_df['pred_count'].quantile(0.75)))
    global_pred_pce_75th = float(max(0.05, pred_df['pred_weighted_pce'].quantile(0.75)))
except Exception as e:
    st.error(f"Failed to load cached datasets. Please make sure to run preprocess_and_train.py first. Error: {e}")
    st.stop()

# Helper functions for tooltips and profile menu
def add_tooltip_strings(df, vol_col='violations', cis_col='weighted_pce'):
    df = df.copy()
    # Format coordinates
    df['centroid_str'] = df.apply(
        lambda r: f"({r['lat_center']:.5f}, {r['lon_center']:.5f})" if pd.notna(r.get('lat_center')) 
        else (f"({r['lat']:.5f}, {r['lon']:.5f})" if 'lat' in df.columns else "N/A"), 
        axis=1
    )
    # Format counts and scores
    df['violations_str'] = df[vol_col].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "0.0")
    df['cis_str'] = df[cis_col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "0.00")
    if 'gi_zscore' in df.columns:
        df['spillover_str'] = df['gi_zscore'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "0.00")
    else:
        df['spillover_str'] = "0.00"
    if 'roi_class' in df.columns:
        df['roi_class_str'] = df['roi_class']
    else:
        df['roi_class_str'] = "Standard"
        
    def get_loc_name(row):
        junc = str(row.get('junction_name', '')).strip()
        street = str(row.get('street_location', '')).strip()
        if junc and junc.lower() not in ['nan', 'none', '']:
            return junc
        if street and street.lower() not in ['nan', 'none', '']:
            return street
        return "Unnamed Location"
        
    df['junction_name'] = df.apply(get_loc_name, axis=1)
    return df

def clean_filename_part(value):
    cleaned = str(value).strip().lower()
    replacements = {
        " ": "",
        ":": "",
        "/": "",
        "\\": "",
        "(": "",
        ")": "",
        "🔮": "",
        "🚔": "",
        "📊": "",
        "&": "and",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned

def get_prediction_visibility_mask(df, selected_layer):
    if df.empty:
        return pd.Series(False, index=df.index)
    if selected_layer == "🔮 Predicted Violations":
        return df['pred_density_active'].fillna(False)
    if selected_layer == "🔮 Predicted Congestion Impact":
        return df['pred_congestion_active'].fillna(False)
    if selected_layer == "🔮 Predicted Combined Map":
        return df['pred_density_active'].fillna(False) | df['pred_congestion_active'].fillna(False)
    if selected_layer == "🔮 Predicted Spillover":
        return df['pred_spillover_significant'].fillna(False)
    if selected_layer == "🚔 Predicted Enforcement Plan":
        return df['pred_enforcement_visible'].fillna(False)
    if selected_layer == "🔮 Predicted Clusters (DBSCAN)":
        return df['pred_count'].fillna(0) > 0
    return pd.Series(True, index=df.index)

def build_prediction_download(df, selected_date, selected_hour, selected_bucket, selected_layer):
    export_df = df.copy()
    export_df['selected_forecast_date'] = str(selected_date)
    export_df['selected_forecast_hour'] = str(selected_hour)
    export_df['selected_time_bucket'] = str(selected_bucket)
    export_df['selected_prediction_mode'] = selected_layer
    
    # Calculate mask
    mask = get_prediction_visibility_mask(export_df, selected_layer)
    export_df['visible_in_selected_layer'] = mask
    
    # Filter to visible rows
    export_df = export_df[mask].copy()
    
    # Add predicted_metric and predicted_value columns based on selected mode/layer
    if selected_layer == "🔮 Predicted Violations":
        export_df['predicted_metric'] = "Violations Count"
        export_df['predicted_value'] = export_df['pred_count']
    elif selected_layer == "🔮 Predicted Congestion Impact":
        export_df['predicted_metric'] = "PCE Congestion Score"
        export_df['predicted_value'] = export_df['pred_weighted_pce']
    elif selected_layer == "🔮 Predicted Combined Map":
        export_df['predicted_metric'] = "Combined PCE and Count"
        export_df['predicted_value'] = export_df['pred_count'] + export_df['pred_weighted_pce']
    elif selected_layer == "🔮 Predicted Spillover":
        export_df['predicted_metric'] = "Spillover Z-Score"
        export_df['predicted_value'] = export_df['gi_zscore']
    elif selected_layer == "🚔 Predicted Enforcement Plan":
        export_df['predicted_metric'] = "Enforcement Classification"
        export_df['predicted_value'] = export_df['roi_class']
    elif selected_layer == "🔮 Predicted Clusters (DBSCAN)":
        export_df['predicted_metric'] = "Projected Violations"
        export_df['predicted_value'] = export_df['pred_count']
    else:
        export_df['predicted_metric'] = "Predicted Count"
        export_df['predicted_value'] = export_df['pred_count']

    export_columns = [
        'h3_8',
        'selected_forecast_date',
        'selected_forecast_hour',
        'selected_time_bucket',
        'selected_prediction_mode',
        'predicted_metric',
        'predicted_value',
        'junction_name',
        'lat_center',
        'lon_center',
        'lat',
        'lon',
        'pred_count',
        'pred_weighted_pce',
        'hotspot_prob',
        'hotspot_flag',
        'gi_zscore',
        'gi_pvalue',
        'roi_class',
    ]
    available_columns = [col for col in export_columns if col in export_df.columns]
    return export_df[available_columns].to_csv(index=False).encode('utf-8')

@st.cache_data
def get_hourly_violation_profile(hist_df):
    hourly = (
        hist_df.groupby('hour', as_index=False)['violation_count']
        .sum()
        .sort_values('hour')
    )
    return hourly

def render_enforcement_blind_spot_chart(hist_df):
    hourly = get_hourly_violation_profile(hist_df)
    fig, ax = plt.subplots(figsize=(13, 4))

    colors = []
    for hour in hourly['hour']:
        if 17 <= hour <= 21:
            colors.append('#E24B4A')
        elif 7 <= hour <= 10:
            colors.append('#2B67A6')
        elif 11 <= hour <= 16:
            colors.append('#94928A')
        else:
            colors.append('#B7B7B7')

    ax.bar(hourly['hour'], hourly['violation_count'], color=colors, width=0.8, zorder=2)
    ax.set_title('Parking violations by hour (IST) — 5–9 PM is the enforcement blind spot', fontsize=13)
    ax.set_xlabel('Hour of day (IST)')
    ax.set_ylabel('Number of violations')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=50, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)

    pm_total = int(hourly[hourly['hour'].between(17, 21)]['violation_count'].sum())
    pm_pct = (pm_total / max(1, int(hourly['violation_count'].sum()))) * 100
    ax.annotate(
        f'PM Peak 5–9pm\n{pm_total:,} violations total\n({pm_pct:.2f}% of all records)',
        xy=(19, max(1, pm_total)),
        xytext=(19, hourly['violation_count'].max() * 0.35),
        ha='center',
        color='#A33',
        fontsize=10,
        fontweight='bold',
        arrowprops={'arrowstyle': '->', 'color': '#A33', 'lw': 1.2},
    )

    legend_items = [
        Patch(color='#2B67A6', label='AM Peak 7-10am'),
        Patch(color='#94928A', label='Midday 11am-4pm'),
        Patch(color='#E24B4A', label='PM Peak 5-9pm'),
        Patch(color='#B7B7B7', label='Night 10pm-6am'),
    ]
    ax.legend(handles=legend_items, loc='upper right')
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

def build_map_key(prefix, *parts):
    safe_parts = [str(part).replace(" ", "_").replace(":", "-").replace("/", "-") for part in parts]
    return "_".join([prefix, *safe_parts])

def stable_seed(*parts):
    joined = "::".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(joined).hexdigest()[:8], 16)

def build_deck(layers, tooltip=None):
    return pdk.Deck(
        layers=layers,
        initial_view_state=blr_view,
        tooltip=tooltip,
        map_provider=None,
        map_style=None,
    )

def render_cell_inspector(df_source, cell_meta, metric_col, select_key, prompt):
    options_df = (
        df_source[df_source[metric_col] > 0]
        .sort_values(metric_col, ascending=False)
        [['h3_8', 'junction_name']]
        .drop_duplicates(subset=['h3_8'])
    )
    if options_df.empty:
        st.info("No active H3 cells are available for inspection in this filter.")
        return None

    label_map = {
        row.h3_8: f"{row.junction_name} ({row.h3_8})"
        for row in options_df.itertuples()
    }
    selected_cell = st.selectbox(
        prompt,
        options=options_df['h3_8'].tolist(),
        format_func=lambda cell_id: label_map.get(cell_id, cell_id),
        key=select_key
    )
    if selected_cell:
        display_detailed_profile(selected_cell, df_source, cell_meta)
    return selected_cell

def render_color_key(section_type):
    if section_type == "density":
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px; font-size: 0.85rem; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
            <span style="color:#94a3b8; font-weight:bold; margin-right:5px;">🎨 COLOR KEY:</span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(255, 255, 178); border: 1px solid rgba(255,255,255,0.2); border-radius: 2px;"></span> Low Vol
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(254, 204, 92); border: 1px solid rgba(255,255,255,0.2); border-radius: 2px;"></span> Med Vol
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(189, 0, 38); border: 1px solid rgba(255,255,255,0.2); border-radius: 2px;"></span> High Vol (Hotspots)
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif section_type == "congestion":
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px; font-size: 0.85rem; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
            <span style="color:#94a3b8; font-weight:bold; margin-right:5px;">🎨 COLOR KEY:</span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(26, 54, 93); border: 1px solid rgba(255,255,255,0.2); border-radius: 2px;"></span> Low CIS
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(137, 84, 57); border: 1px solid rgba(255,255,255,0.2); border-radius: 2px;"></span> Med CIS
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(249, 115, 22); border: 1px solid rgba(255,255,255,0.2); border-radius: 2px;"></span> High CIS (Disruptive)
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif section_type == "combined":
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px; font-size: 0.85rem; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); flex-wrap: wrap;">
            <span style="color:#94a3b8; font-weight:bold; margin-right:5px;">🎨 COLOR KEY:</span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(189, 0, 38); border: 1px solid rgba(255,255,255,0.2); border-radius: 2px;"></span> Fill: Violation Volume (Yellow &rarr; Red)
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; border: 2px solid rgb(253, 141, 60); border-radius: 2px; background: transparent;"></span> Border: Congestion Impact (Yellow &rarr; Orange)
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif section_type == "spillover":
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px; font-size: 0.85rem; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
            <span style="color:#94a3b8; font-weight:bold; margin-right:5px;">🎨 COLOR KEY:</span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgba(71, 85, 105, 0.4); border: 1px solid rgba(255,255,255,0.1); border-radius: 2px;"></span> Non-Significant Grid
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(56, 189, 248); border: 1px solid rgba(255,255,255,0.2); border-radius: 2px;"></span> Spillover Hotspot (Z &ge; 1.96)
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(99, 102, 241); border: 1px solid rgba(255,255,255,0.2); border-radius: 2px;"></span> Extreme Spillover Hotspot
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif section_type == "enforcement":
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px; font-size: 0.85rem; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); flex-wrap: wrap;">
            <span style="color:#94a3b8; font-weight:bold; margin-right:5px;">🎨 COLOR KEY:</span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(239, 68, 68); border-radius: 2px;"></span> Tier 1 (Severe Disruptive)
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(249, 115, 22); border-radius: 2px;"></span> Tier 2 (Silent Bottlenecks)
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgb(234, 179, 8); border-radius: 2px;"></span> Tier 3 (Volume Hotspots)
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 12px; background: rgba(148, 163, 184, 0.4); border-radius: 2px;"></span> Standard Grid
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif section_type == "dbscan":
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px; font-size: 0.85rem; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
            <span style="color:#94a3b8; font-weight:bold; margin-right:5px;">🎨 COLOR KEY:</span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 10px; height: 10px; background: rgb(56, 189, 248); border-radius: 50%;"></span> Individual Violation Cluster Points (Distinct Colors)
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 10px; height: 10px; background: rgba(148, 163, 184, 0.4); border-radius: 50%;"></span> Noise / Unclustered
            </span>
            <span style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 16px; height: 16px; background: rgb(239, 68, 68); border: 2px solid white; border-radius: 50%;"></span> Cluster Centroid
            </span>
        </div>
        """, unsafe_allow_html=True)

def get_selected_cell(event):
    if not event:
        return None
    selection = None
    if hasattr(event, 'selection'):
        selection = event.selection
    elif isinstance(event, dict) and 'selection' in event:
        selection = event['selection']
        
    if selection:
        selected_objects = selection.get('objects', {})
        for layer_id, objs in selected_objects.items():
            if objs:
                return objs[0].get('h3_8')
    return None

def render_pydeck_map(layers, tooltip, key, df_source=None, show_selected_profile=True):
    event = st.pydeck_chart(
        build_deck(layers=layers, tooltip=tooltip),
        key=key,
        on_select="rerun",
        selection_mode="single-object",
    )
    selected_cell = get_selected_cell(event)
    if show_selected_profile and selected_cell and df_source is not None:
        display_detailed_profile(selected_cell, df_source, cell_meta)
    return event

def get_selected_dbscan_object(event):
    if not event:
        return None
    selection = None
    if hasattr(event, 'selection'):
        selection = event.selection
    elif isinstance(event, dict) and 'selection' in event:
        selection = event['selection']
        
    if selection:
        selected_objects = selection.get('objects', {})
        for layer_id, objs in selected_objects.items():
            if objs:
                return objs[0]
    return None

def display_selected_dbscan(selected_obj):
    if not selected_obj:
        return
    
    cluster_id = selected_obj.get('cluster')
    lat = selected_obj.get('latitude', 0.0)
    lon = selected_obj.get('longitude', 0.0)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    if 'violations' in selected_obj:
        dom_violation = selected_obj.get('dominant_violation', 'N/A')
        v_count = selected_obj.get('violations', 0)
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color:#f87171; margin-top:0; margin-bottom:12px;">🔴 Selected Cluster Centroid Details</h4>
            <p>• <b>Cluster ID:</b> <code>{cluster_id}</code></p>
            <p>• <b>Location Centroid:</b> <code>({lat:.5f}, {lon:.5f})</code></p>
            <p>• <b>Total Violations in Cluster:</b> <code>{v_count}</code></p>
            <p>• <b>Dominant Violation Type:</b> <span style="color:#fb923c; font-weight:bold;">{dom_violation}</span></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        violation_type = selected_obj.get('primary_violation', 'N/A')
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color:#38bdf8; margin-top:0; margin-bottom:12px;">📍 Selected Violation Point Details</h4>
            <p>• <b>Cluster ID:</b> <code>{cluster_id}</code> (Noise if -1)</p>
            <p>• <b>Coordinates:</b> <code>({lat:.5f}, {lon:.5f})</code></p>
            <p>• <b>Violation Type:</b> <span style="color:#fb923c; font-weight:bold;">{violation_type}</span></p>
        </div>
        """, unsafe_allow_html=True)

def display_detailed_profile(cell_id, df_source, cell_meta):
    row_source = df_source[df_source['h3_8'] == cell_id]
    row_meta = cell_meta[cell_meta['h3_8'] == cell_id]
    
    if row_source.empty:
        st.markdown(f"<div class='glass-card'>No dynamic data available for H3 Cell <code>{cell_id}</code>.</div>", unsafe_allow_html=True)
        return
        
    src = row_source.iloc[0]
    meta = row_meta.iloc[0] if not row_meta.empty else None
    
    v_val = src.get('violations', src.get('pred_count', 0.0))
    cis_val = src.get('weighted_pce', src.get('pred_weighted_pce', 0.0))
    z_val = src.get('gi_zscore', 0.0)
    p_val = src.get('gi_pvalue', 1.0)
    roi_val = src.get('roi_class', 'Standard')
    
    lat = src.get('lat_center', src.get('lat', 0.0))
    lon = src.get('lon_center', src.get('lon', 0.0))
    junc = src.get('junction_name', 'Unnamed Location')
    
    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color:#38bdf8; margin-top:0; margin-bottom: 12px;">📋 H3 Cell Profile - {junc}</h4>
        <div style="display: flex; gap: 40px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 250px;">
                <b style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase;">📍 Location Details</b><br/>
                <div style="margin-top:8px; line-height: 1.6;">
                    • <b>H3 Index:</b> <code>{cell_id}</code><br/>
                    • <b>Centroid Coordinates:</b> <code>{lat:.5f}, {lon:.5f}</code><br/>
                    • <b>Dominant Junction/Area:</b> {junc}
                </div>
            </div>
            <div style="flex: 1; min-width: 250px;">
                <b style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase;">📊 Key Metrics</b><br/>
                <div style="margin-top:8px; line-height: 1.6;">
                    • <b>Violation Count:</b> <code>{v_val:,.2f}</code><br/>
                    • <b>Congestion Impact (CIS):</b> <code>{cis_val:,.2f}</code><br/>
                    • <b>Spillover Z-Score:</b> <code>{z_val:,.2f}</code> (p-val: <code>{p_val:,.4f}</code>)<br/>
                    • <b>Enforcement Tier:</b> <span style="color:#fb923c; font-weight:bold;">{roi_val}</span>
                </div>
            </div>
    """, unsafe_allow_html=True)
    
    if meta is not None:
        st.markdown(f"""
            <div style="flex: 1; min-width: 250px;">
                <b style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase;">🚗 Vehicle Mix</b><br/>
                <div style="margin-top:8px; line-height: 1.5;">
                    • Cars/Jeeps: <code>{meta.get('pct_car', 0.0):.1%}</code><br/>
                    • Scooters/2W: <code>{meta.get('pct_scooter', 0.0):.1%}</code><br/>
                    • Passenger Autos: <code>{meta.get('pct_auto', 0.0):.1%}</code><br/>
                    • Maxi-Cabs: <code>{meta.get('pct_maxi', 0.0):.1%}</code>
                </div>
            </div>
            <div style="flex: 1; min-width: 250px;">
                <b style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase;">⚠️ Violation Type Mix</b><br/>
                <div style="margin-top:8px; line-height: 1.5;">
                    • Wrong Parking: <code>{meta.get('pct_wrong_park', 0.0):.1%}</code><br/>
                    • No Parking: <code>{meta.get('pct_no_park', 0.0):.1%}</code><br/>
                    • Main Road Parking: <code>{meta.get('pct_main_road', 0.0):.1%}</code><br/>
                    • Footpath Parking: <code>{meta.get('pct_footpath', 0.0):.1%}</code>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# Title banner
st.markdown('<div class="header-title">🚨 Parking & Congestion Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">AI-Driven Spatial Analytics & Forecasting for Bangalore Traffic Enforcement</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/color/96/police-car.png", width=70)
st.sidebar.title("Navigation & Filters")
nav = st.sidebar.radio("Go to:", ["🔍 Historical Analysis", "🔮 Predictive Forecasting"])

# Global map settings
blr_view = pdk.ViewState(
    latitude=12.9716,
    longitude=77.5946,
    zoom=11.5,
    pitch=30,
    bearing=0
)

# Standard map tooltip
map_tooltip = {
    "html": """
    <div style="font-family: Arial; font-size: 12px; padding: 10px; border-radius: 8px;">
        <b style="color: #38bdf8; font-size: 14px;">{junction_name}</b><br/>
        <b>H3 Cell ID:</b> {h3_8}<br/>
        <b>Centroid:</b> {centroid_str}<br/>
        <b>Violations Count:</b> {violations_str}<br/>
        <b>CIS Score (Congestion Impact):</b> {cis_str}<br/>
        <b>Spillover Z-Score:</b> {spillover_str}<br/>
        <b>Enforcement Plan Tier:</b> {roi_class_str}
    </div>
    """,
    "style": {"color": "white", "backgroundColor": "#0f172a", "border": "1px solid rgba(255,255,255,0.1)"}
}

# Historical analysis navigation
if nav == "🔍 Historical Analysis":
    st.sidebar.subheader("Historical Filters")
    
    # Date selection
    min_date = hist_df['date'].min()
    max_date = hist_df['date'].max()
    # Initialize session state keys for historical filters
    if 'hist_time_bucket' not in st.session_state:
        st.session_state['hist_time_bucket'] = "All"
    if 'hist_hour' not in st.session_state:
        st.session_state['hist_hour'] = "All"

    selected_date_range = st.sidebar.date_input(
        "Date Range",
        value=(max_date - pd.Timedelta(days=14), max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Safe date range unpacking and validation
    start_date, end_date = None, None
    if isinstance(selected_date_range, tuple):
        if len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
        elif len(selected_date_range) == 1:
            start_date = selected_date_range[0]
            end_date = selected_date_range[0]
    else:
        start_date = selected_date_range
        end_date = selected_date_range

    if start_date is None or end_date is None:
        start_date = min_date
        end_date = max_date

    # Validate/enforce chronological ordering
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    
    # Time bucket
    time_bucket_val = st.sidebar.selectbox(
        "Time Bucket",
        options=["All", "AM Peak (07:00 - 10:59)", "Midday (11:00 - 16:59)", "PM Peak (17:00 - 21:59)", "Off-Peak Night (22:00 - 06:59)"],
        key='hist_time_bucket',
        on_change=on_hist_bucket_change
    )
    
    # Dynamic hour options based on selected bucket
    allowed_hour_options = get_hour_options_for_bucket(st.session_state['hist_time_bucket'])
    
    # Safety check: make sure current hour is in allowed options
    if st.session_state['hist_hour'] not in allowed_hour_options:
        st.session_state['hist_hour'] = "All"
        
    hour_val = st.sidebar.selectbox(
        "Hour of Day",
        options=allowed_hour_options,
        key='hist_hour',
        on_change=on_hist_hour_change
    )
    
    # Expose current selections to rest of script
    time_bucket = st.session_state['hist_time_bucket']
    hour = st.session_state['hist_hour']
    
    # Filter aggregated data dynamically
    df_filtered = data_processing.filter_aggregated_data(
        hist_df, start_date, end_date, time_bucket, hour
    )
    
    # Run dynamic spatial statistics
    df_scored = scoring.compute_gi_star(df_filtered, all_cells, 'weighted_pce', 'h3_8')
    df_scored, vol_thresh, pce_thresh = scoring.add_roi_classification(df_scored, 'violations', 'weighted_pce')
    
    # Merge all cell metadata back
    df_scored = df_scored.merge(cell_meta, on='h3_8', how='left', suffixes=('', '_meta'))
    df_scored['lat'] = df_scored['lat'].fillna(df_scored['lat_center'])
    df_scored['lon'] = df_scored['lon'].fillna(df_scored['lon_center'])
    
    # Filter to cells with coordinates for visualization
    df_scored_clean = df_scored.dropna(subset=['lat_center', 'lon_center']).copy()
    df_scored_clean = add_tooltip_strings(df_scored_clean, 'violations', 'weighted_pce')
    
    # Precompute normalized values and color arrays to ensure tabs/views can load independently
    if not df_scored_clean.empty:
        v_max = df_scored_clean['violations'].max()
        v_min = df_scored_clean['violations'].min()
        if v_max > v_min:
            df_scored_clean['norm_violations'] = np.sqrt(df_scored_clean['violations'] - v_min) / (np.sqrt(v_max - v_min) + 1e-6)
        else:
            df_scored_clean['norm_violations'] = 0.0
            
        colors = []
        for val in df_scored_clean['norm_violations']:
            r = int(255 - (255 - 189) * val)
            g = int(255 - (255 - 0) * val)
            b = int(178 - (178 - 38) * val)
            colors.append([r, g, b, 160])
        df_scored_clean['fill_color'] = colors

        p_max = df_scored_clean['weighted_pce'].max()
        p_min = df_scored_clean['weighted_pce'].min()
        if p_max > p_min:
            df_scored_clean['norm_pce'] = np.sqrt(df_scored_clean['weighted_pce'] - p_min) / (np.sqrt(p_max - p_min) + 1e-6)
        else:
            df_scored_clean['norm_pce'] = 0.0
            
        colors_pce = []
        for val in df_scored_clean['norm_pce']:
            r = int(26 + (249 - 26) * val)
            g = int(54 + (115 - 54) * val)
            b = int(93 + (22 - 93) * val)
            colors_pce.append([r, g, b, 170])
        df_scored_clean['fill_color_pce'] = colors_pce

        colors_outline = []
        for val in df_scored_clean['norm_pce']:
            r = int(253)
            g = int(141 + (254 - 141) * (1-val))
            b = int(60 + (196 - 60) * (1-val))
            colors_outline.append([r, g, b, 255])
        df_scored_clean['line_color_pce'] = colors_outline
    else:
        df_scored_clean['norm_violations'] = []
        df_scored_clean['fill_color'] = []
        df_scored_clean['norm_pce'] = []
        df_scored_clean['fill_color_pce'] = []
        df_scored_clean['line_color_pce'] = []
    
    # KPIs Layout
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-val">{int(df_scored_clean['violations'].sum()):,}</div>
            <div class="kpi-label">Total Violations</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-val">{df_scored_clean['weighted_pce'].sum():.1f}</div>
            <div class="kpi-label">Congestion (PCE-Weighted)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        hotspot_cells = df_scored_clean[(df_scored_clean['gi_zscore'] >= 1.96) & (df_scored_clean['gi_pvalue'] < 0.05)]
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-val metric-red">{len(hotspot_cells)}</div>
            <div class="kpi-label">Active Hotspots (Gi*)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        tier1_cells = df_scored_clean[df_scored_clean['roi_class'] == 'Tier 1: Max Disruption']
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-val metric-orange">{len(tier1_cells)}</div>
            <div class="kpi-label">Strategic Targets (Tier 1)</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Main content tabs
    tab_option = st.segmented_control(
        "Analysis Layer",
        options=[
            "🔴 Violation Density", 
            "🟠 Congestion Impact", 
            "🟣 Combined Map", 
            "🔵 Spillover Analysis", 
            "🚔 Strategic Enforcement Plan",
            "🟢 Violation Clusters (DBSCAN)"
        ],
        default="🔴 Violation Density",
        key="hist_analysis_layer_select"
    )
    if not tab_option:
        tab_option = "🔴 Violation Density"
        
    # Tab 1: Violation Density
    if tab_option == "🔴 Violation Density":
        st.subheader("🔴 Violation Density Heatmap")
        st.markdown("Displays the total volume of parking violations per H3 cell for the selected timeframe. Focuses on enforcement volumes.")
        render_color_key("density")
        
        # Map layer
        
        layer = pdk.Layer(
            "H3HexagonLayer",
            data=df_scored_clean[df_scored_clean['violations'] > 0],
            id="hist_density_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color",
            stroked=True,
            filled=True,
            get_line_color="[44, 62, 80, 100]",
            line_width_min_pixels=0.8,
            pickable=True,
        )

        render_pydeck_map([layer], map_tooltip, "hist_density_map", df_scored_clean)
        
        # Busiest locations table
        st.write("### 🏆 Busiest Enforcement Zones")
        busiest = df_scored_clean[df_scored_clean['violations'] > 0].sort_values('violations', ascending=False).head(10).copy()
        
        tbl_df = busiest[['h3_8', 'junction_name', 'violations', 'weighted_pce']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'violations': 'Violation Count',
            'weighted_pce': 'PCE Congestion Score'
        })
        
        st.dataframe(
            tbl_df,
            width="stretch",
            hide_index=True,
            key="hist_busiest_tbl"
        )
        
        render_cell_inspector(
            df_scored_clean,
            cell_meta,
            'violations',
            'hist_density_inspector',
            "Inspect an H3 cell from the busiest zones:"
        )

    # Tab 2: Congestion Impact
    elif tab_option == "🟠 Congestion Impact":
        st.subheader("🟠 Congestion Impact Heatmap")
        
        # In detailed explanation of congestion impact score
        st.markdown("""
        > [!NOTE]
        > **Reasoning behind the Congestion Impact Score (CIS):**
        > Traditional traffic enforcement prioritizes raw violation count, which treats a scooter parked on a side alley the same as a bus blocking a major intersection.
        > The Congestion Impact Score (CIS) directly quantifies traffic friction by multiplying the count of each violation by its standard **Passenger Car Equivalent (PCE)** weight (from IRC:106-1990) and a **Location Factor** reflecting HCM Chapter 18:
        > 
        > $$\text{CIS} = \sum_{j} (\text{PCE}_j \times \text{Location Factor}_j)$$
        > 
        > - **PCE Weights**: Scooter/2W = 0.5, Car/Jeep = 1.0, Passenger Auto = 0.75, Goods Auto = 1.4, Maxi-Cab = 2.0, Bus/Heavy Lorry = 3.7.
        > - **Location Factors**: 1.50 for junctions, intersections, signals, and zebra crossings; 1.30 for main/arterial roads; 1.00 for minor/local roads.
        > This ensures high-occupancy corridors and key bottlenecks receive prioritized enforcement.
        """)
        render_color_key("congestion")
        
        # Map layer
        
        layer = pdk.Layer(
            "H3HexagonLayer",
            data=df_scored_clean[df_scored_clean['weighted_pce'] > 0],
            id="hist_congestion_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color_pce",
            stroked=True,
            filled=True,
            get_line_color="[17, 17, 17, 100]",
            line_width_min_pixels=0.8,
            pickable=True,
        )

        render_pydeck_map([layer], map_tooltip, "hist_congestion_map", df_scored_clean)
        
        # Top Congestion Impact Bottlenecks table
        st.write("### 🏆 Top Congestion Impact Bottlenecks")
        busiest_congestion = df_scored_clean[df_scored_clean['weighted_pce'] > 0].sort_values('weighted_pce', ascending=False).head(10).copy()
        tbl_congestion = busiest_congestion[['h3_8', 'junction_name', 'weighted_pce', 'violations']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'weighted_pce': 'PCE Congestion Score',
            'violations': 'Violation Count'
        })
        st.dataframe(
            tbl_congestion,
            width="stretch",
            hide_index=True,
            key="hist_congestion_tbl"
        )
        
        # Detailed inspector panel on click/select
        st.write("### 🔍 H3 Cell Profile Inspector")
        render_cell_inspector(
            df_scored_clean,
            cell_meta,
            'violations',
            'hist_inspect_selectbox',
            "Select an H3 cell to inspect:"
        )

    # Tab 3: Combined Map overlay
    elif tab_option == "🟣 Combined Map":
        st.subheader("🟣 Combined Map Overlay")
        st.markdown("Dual-layer map that overlays **H3 violation counts** (filled hexagons) and **CIS congestion impact** (outlined borders). Allows officers to spot Tier 2 'Silent Bottlenecks' (large outlines, light fills) at a glance.")
        render_color_key("combined")
        
        # Map layers
        
        fill_layer = pdk.Layer(
            "H3HexagonLayer",
            data=df_scored_clean[df_scored_clean['violations'] > 0],
            id="hist_combined_fill_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color",
            stroked=False,
            filled=True,
            pickable=True,
        )
        
        outline_layer = pdk.Layer(
            "H3HexagonLayer",
            data=df_scored_clean[df_scored_clean['weighted_pce'] > 0],
            id="hist_combined_outline_layer",
            get_hexagon="h3_8",
            get_line_color="line_color_pce",
            line_width_min_pixels=3,
            stroked=True,
            filled=False,
            pickable=True,
        )

        render_pydeck_map([fill_layer, outline_layer], map_tooltip, "hist_combined_map", df_scored_clean)

        # Top Busiest Combined Risk Areas table
        st.write("### 🏆 Top Busiest Combined Risk Areas")
        combined_busiest = df_scored_clean[(df_scored_clean['violations'] > 0) | (df_scored_clean['weighted_pce'] > 0)].copy()
        combined_busiest['combined_score'] = combined_busiest['violations'] + combined_busiest['weighted_pce']
        combined_busiest = combined_busiest.sort_values('combined_score', ascending=False).head(10).copy()
        
        tbl_combined = combined_busiest[['h3_8', 'junction_name', 'violations', 'weighted_pce']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'violations': 'Violation Count',
            'weighted_pce': 'PCE Congestion Score'
        })
        st.dataframe(
            tbl_combined,
            width="stretch",
            hide_index=True,
            key="hist_combined_tbl"
        )

        st.write("### 🔍 H3 Cell Profile Inspector")
        render_cell_inspector(
            df_scored_clean,
            cell_meta,
            'weighted_pce',
            'hist_combined_inspector',
            "Inspect a combined-map H3 cell:"
        )
        
        st.markdown("""
        **Legend Map Guide:**
        *   **Filled Hexagons**: Violation counts (faint yellow = low count, solid red = high count).
        *   **Outline Borders**: PCE-weighted congestion score (bright orange/red = severe road obstruction).
        *   **Silent Bottleneck (Tier 2)**: Cells with light fills (low volume) but thick orange outlines (high traffic flow degradation). Zero-tolerance towing recommended.
        """)

    # Tab 4: Spatial Spillover Hotspots
    elif tab_option == "🔵 Spillover Analysis":
        st.subheader("🔵 Spatial Spillover Hotspot Map (Local Getis-Ord Gi*)")
        
        # In detailed explanation of spillover reasoning
        st.markdown("""
        > [!NOTE]
        > **Reasoning behind Spatial Spillover (Local Getis-Ord $G_i^*$):**
        > Parking blockages on one street do not stay isolated; they degrade speed on adjacent streets and spill over to surrounding junctions.
        > We model this spatial dependency using a row-standardized weights matrix ($W$) representing **1-ring adjacent H3 neighbors** ($k=1$).
        > The Local Getis-Ord $G_i^*$ statistic compares the local sum of congestion in each cell and its immediate neighbors against the global sum across all city cells:
        > 
        > $$G_i^* = \\frac{\sum_{j} w_{ij} x_j - \\bar{X} \sum_{j} w_{ij}}{S \sqrt{\\frac{n\sum_{j} w_{ij}^2 - (\sum_{j} w_{ij})^2}{n-1}}}$$
        > 
        > Where $Z_i \ge 1.96$ indicates a statistically significant ($p_i < 0.05$) hot zone where high congestion values are clustered together, identifying systemic neighborhood spillovers.
        """)
        render_color_key("spillover")
        
        z_max = df_scored_clean['gi_zscore'].max()
        colors_z = []
        for idx, row in df_scored_clean.iterrows():
            z = row['gi_zscore']
            p = row['gi_pvalue']
            if z >= 1.96 and p < 0.05:
                val = min(1.0, (z - 1.96) / (max(2.58, z_max) - 1.96 + 1e-6))
                r = int(56 + (99 - 56) * val)
                g = int(189 + (102 - 189) * val)
                b = int(248 + (241 - 248) * val)
                colors_z.append([r, g, b, 230])
            else:
                colors_z.append([0, 0, 0, 0])
        df_scored_clean['fill_color_z'] = colors_z
        
        spillover_layer = pdk.Layer(
            "H3HexagonLayer",
            data=df_scored_clean,
            id="hist_spillover_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color_z",
            stroked=True,
            filled=True,
            get_line_color="[17, 17, 17, 80]",
            line_width_min_pixels=0.8,
            pickable=True,
        )
        
        render_pydeck_map([spillover_layer], map_tooltip, "hist_spillover_map", df_scored_clean)
        
        # Display significant cells in a table
        st.write("### 📈 Significant Spillover Hotspots (Confidence ≥ 95%)")
        sig_spill = df_scored_clean[(df_scored_clean['gi_zscore'] >= 1.96) & (df_scored_clean['gi_pvalue'] < 0.05)].sort_values('gi_zscore', ascending=False).copy()
        
        tbl_spill = sig_spill[['h3_8', 'junction_name', 'gi_zscore', 'gi_pvalue', 'weighted_pce']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'gi_zscore': 'Gi* Z-Score',
            'gi_pvalue': 'p-value',
            'weighted_pce': 'PCE Congestion Score'
        })
        
        st.dataframe(
            tbl_spill,
            width="stretch",
            hide_index=True,
            key="hist_spill_tbl"
        )
        
        render_cell_inspector(
            df_scored_clean,
            cell_meta,
            'weighted_pce',
            'hist_spillover_inspector',
            "Inspect a spillover-analysis H3 cell:"
        )

    # Tab 5: Strategic Enforcement Plan
    elif tab_option == "🚔 Strategic Enforcement Plan":
        st.subheader("🚔 Strategic Enforcement Plan (ROI Classification)")
        
        # In detailed explanation of enforcement plan reasoning
        st.markdown("""
        > [!NOTE]
        > **Reasoning behind the Enforcement Plan (Strategic ROI Tiers):**
        > Enforcement resources (towing trucks, field patrols) are finite. We classify active cells into three distinct enforcement tiers relative to the **90th percentile** of violations ($P_{90,\text{vol}}$) and CIS scores ($P_{90,\text{cis}}$) within the filtered timeframe:
        > 
        > 1. **Tier 1: Max Disruption** ($\text{Violations} \ge P_{90,\text{vol}}$ AND $\text{CIS} \ge P_{90,\text{cis}}$):
        >    High violation frequency causing severe flow obstruction. Priority target for intensive enforcement patrols.
        > 2. **Tier 2: Silent Bottleneck** ($\text{Violations} < P_{90,\text{vol}}$ AND $\text{CIS} \ge P_{90,\text{cis}}$):
        >    Fewer tickets but critical capacity obstruction (e.g. heavy vehicles blocking narrow arterial links). Targeted for zero-tolerance towing.
        > 3. **Tier 3: Volume Hotspot** ($\text{Violations} \ge P_{90,\text{vol}}$ AND $\text{CIS} < P_{90,\text{cis}}$):
        >    High violation frequency but minimal flow impact (e.g. two-wheelers parked on wide side alleys). Targeted for routine ticketing.
        > 4. **Standard**: All other active locations requiring standard monitoring.
        """)
        render_enforcement_blind_spot_chart(hist_df)
        render_color_key("enforcement")
        
        # Color coding for tiers
        colors_roi = []
        for idx, row in df_scored_clean.iterrows():
            tier = row['roi_class']
            if tier == 'Tier 1: Max Disruption':
                colors_roi.append([239, 68, 68, 190]) # Red
            elif tier == 'Tier 2: Silent Bottleneck':
                colors_roi.append([249, 115, 22, 190]) # Orange
            elif tier == 'Tier 3: Volume Hotspot':
                colors_roi.append([234, 179, 8, 190]) # Yellow
            elif row['violations'] > 0:
                colors_roi.append([148, 163, 184, 80]) # Faint Grey
            else:
                colors_roi.append([71, 85, 105, 30])
        df_scored_clean['fill_color_roi'] = colors_roi
        
        roi_layer = pdk.Layer(
            "H3HexagonLayer",
            data=df_scored_clean,
            id="hist_roi_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color_roi",
            stroked=True,
            filled=True,
            get_line_color="[17, 17, 17, 80]",
            line_width_min_pixels=0.8,
            pickable=True,
        )
        
        render_pydeck_map([roi_layer], map_tooltip, "hist_enforcement_map", df_scored_clean)
        
        # Strategic Targets Table
        st.write("### 🚨 Strategic Targeted Patrol Locations")
        targets_df = df_scored_clean[df_scored_clean['roi_class'] != 'Standard'].sort_values('weighted_pce', ascending=False).copy()
        
        tbl_targets = targets_df[['h3_8', 'junction_name', 'roi_class', 'violations', 'weighted_pce']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'roi_class': 'Strategic Enforcement Classification',
            'violations': 'Violation Count',
            'weighted_pce': 'PCE Congestion Score'
        })
        
        st.dataframe(
            tbl_targets,
            width="stretch",
            hide_index=True,
            key="hist_targets_tbl"
        )
        
        render_cell_inspector(
            df_scored_clean,
            cell_meta,
            'weighted_pce',
            'hist_enforcement_inspector',
            "Inspect a strategic enforcement H3 cell:"
        )

    # Tab 6: Violation Clusters DBSCAN
    elif tab_option == "🟢 Violation Clusters (DBSCAN)":
        st.subheader("🟢 Spatial Clusters of Violations (DBSCAN)")
        st.markdown("Groups individual violation coordinates into spatial clusters of high density. Spots neighborhood double parking / footpath parking blocks.")
        render_color_key("dbscan")
        
        raw_filtered = data_processing.filter_dbscan_violations(
            dbscan_df, start_date, end_date, time_bucket, hour
        )
        
        v_types = ["All"] + sorted(raw_filtered['primary_violation'].unique().tolist())
        selected_vtype = st.selectbox("Filter by Primary Violation Type", options=v_types, key="hist_dbscan_vtype_select")
        if selected_vtype != "All":
            raw_filtered = raw_filtered[raw_filtered['primary_violation'] == selected_vtype]
            
        # DBSCAN parameters capped 50-500 meters, increments of 50
        dbscan_eps = st.slider("Clustering Radius (meters)", min_value=50, max_value=500, value=200, step=50, key="hist_dbscan_eps_slider")
        dbscan_min_samples = st.slider("Minimum Violation Count per Cluster", min_value=10, max_value=100, value=30, step=5, key="hist_dbscan_min_slider")
        
        if len(raw_filtered) > 15000:
            st.warning(f"Large subset of {len(raw_filtered):,} violations. Sampling down to 10,000 records for fast map interactions.")
            raw_filtered = raw_filtered.sample(10000, random_state=42)
            
        if len(raw_filtered) >= dbscan_min_samples:
            clustered = data_processing.run_dbscan_clustering(raw_filtered, dbscan_eps, dbscan_min_samples)
            
            unique_cl = clustered['cluster'].unique()
            np.random.seed(42)
            cl_colors = {}
            for c in unique_cl:
                if c == -1:
                    cl_colors[c] = [148, 163, 184, 80]
                else:
                    cl_colors[c] = list(np.random.randint(50, 255, size=3).tolist()) + [210]
            clustered['color'] = clustered['cluster'].map(cl_colors)
            
            points_layer = pdk.Layer(
                "ScatterplotLayer",
                data=clustered,
                id="hist_dbscan_points",
                get_position="[longitude, latitude]",
                get_color="color",
                get_radius=12,
                pickable=True,
            )
            
            centroids = clustered[clustered['cluster'] >= 0].groupby('cluster').agg(
                latitude=('latitude', 'mean'),
                longitude=('longitude', 'mean'),
                violations=('cluster', 'count'),
                dominant_violation=('primary_violation', lambda x: x.value_counts().index[0])
            ).reset_index()
            clustered['coordinates_str'] = clustered.apply(
                lambda row: f"{row['latitude']:.5f}, {row['longitude']:.5f}",
                axis=1
            )
            clustered['latitude_str'] = clustered['latitude'].map(lambda x: f"{x:.5f}")
            clustered['longitude_str'] = clustered['longitude'].map(lambda x: f"{x:.5f}")
            centroids['primary_violation'] = centroids['dominant_violation']
            centroids['coordinates_str'] = centroids.apply(
                lambda row: f"{row['latitude']:.5f}, {row['longitude']:.5f}",
                axis=1
            )
            centroids['latitude_str'] = centroids['latitude'].map(lambda x: f"{x:.5f}")
            centroids['longitude_str'] = centroids['longitude'].map(lambda x: f"{x:.5f}")
            
            centroids_layer = pdk.Layer(
                "ScatterplotLayer",
                data=centroids,
                id="hist_dbscan_centroids",
                get_position="[longitude, latitude]",
                get_color="[239, 68, 68, 230]",
                get_line_color="[255, 255, 255, 255]",
                get_radius=60,
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
                pickable=True,
            )
            
            tooltip_db = {
                "html": """
                <div style="font-family: Arial; font-size: 12px; padding: 10px; border-radius: 8px;">
                    <b>Cluster ID:</b> {cluster}<br/>
                    <b>Type:</b> {primary_violation}<br/>
                    <b>Latitude:</b> {latitude_str}<br/>
                    <b>Longitude:</b> {longitude_str}
                </div>
                """,
                "style": {"color": "white", "backgroundColor": "#0f172a", "border": "1px solid rgba(255,255,255,0.1)"}
            }
            
            dbscan_event = render_pydeck_map(
                [points_layer, centroids_layer],
                tooltip_db,
                "hist_dbscan_map",
                show_selected_profile=False,
            )
            display_selected_dbscan(get_selected_dbscan_object(dbscan_event))

            if not centroids.empty:
                centroid_options = centroids.sort_values('violations', ascending=False).copy()
                centroid_labels = {
                    row.cluster: f"Cluster {row.cluster} - {row.dominant_violation} ({row.violations} violations)"
                    for row in centroid_options.itertuples()
                }
                selected_cluster = st.selectbox(
                    "Inspect a DBSCAN cluster centroid:",
                    options=centroid_options['cluster'].tolist(),
                    format_func=lambda cluster_id: centroid_labels.get(cluster_id, str(cluster_id)),
                    key="hist_dbscan_cluster_inspector"
                )
                selected_centroid = centroids[centroids['cluster'] == selected_cluster]
                if not selected_centroid.empty:
                    display_selected_dbscan(selected_centroid.iloc[0].to_dict())
            
            st.write("### 📍 Clustered Hotspot Centroids Summary")
            st.dataframe(
                centroids.rename(columns={
                    'cluster': 'Cluster ID',
                    'latitude': 'Latitude Centroid',
                    'longitude': 'Longitude Centroid',
                    'violations': 'Total Violations in Cluster',
                    'dominant_violation': 'Dominant Violation Type'
                }),
                width="stretch",
                hide_index=True
            )
        else:
            st.info("Insufficient violations in current filter to form clusters. Try widening the Date Range or selecting a peak Time Bucket.")

# Predictive forecasting navigation
else:
    st.sidebar.subheader("Forecast Filters (7-Day Ahead)")
    
    future_dates_list = sorted(pred_df['date'].unique())
    selected_f_date = st.sidebar.selectbox("Forecast Date", options=future_dates_list)
    
    # Initialize session state keys for forecast filters
    if 'pred_time_bucket' not in st.session_state:
        st.session_state['pred_time_bucket'] = "All"
    if 'pred_hour' not in st.session_state:
        st.session_state['pred_hour'] = "All"
        
    # Time bucket on predictions
    f_time_bucket_val = st.sidebar.selectbox(
        "Forecast Time Bucket",
        options=["All", "AM Peak (07:00 - 10:59)", "Midday (11:00 - 16:59)", "PM Peak (17:00 - 21:59)", "Off-Peak Night (22:00 - 06:59)"],
        key='pred_time_bucket',
        on_change=on_pred_bucket_change
    )
    
    # Filter hours options based on selected bucket
    allowed_f_hour_options = get_hour_options_for_bucket(st.session_state['pred_time_bucket'])
    
    # Safety check: make sure current forecast hour is in allowed options
    if st.session_state['pred_hour'] not in allowed_f_hour_options:
        st.session_state['pred_hour'] = "All"
        
    f_hour_val = st.sidebar.selectbox(
        "Forecast Hour of Day",
        options=allowed_f_hour_options,
        key='pred_hour',
        on_change=on_pred_hour_change
    )
    
    # Expose current selections to rest of script
    f_time_bucket = st.session_state['pred_time_bucket']
    f_hour = st.session_state['pred_hour']
    
    # Filter predictions dynamically
    df_f_filtered = prediction.filter_predictions(
        pred_df, date_str=str(selected_f_date), hour=f_hour, time_bucket=f_time_bucket
    )
    
    # Run Gi* on predicted PCE scores
    df_f_scored = scoring.compute_gi_star(df_f_filtered, all_cells, 'pred_weighted_pce', 'h3_8')
    
    # Merge predicted scores with cell metadata
    df_f_scored = df_f_scored.merge(cell_meta, on='h3_8', how='left', suffixes=('', '_meta'))
    df_f_scored['lat'] = df_f_scored['lat'].fillna(df_f_scored['lat_center'])
    df_f_scored['lon'] = df_f_scored['lon'].fillna(df_f_scored['lon_center'])
    
    # Run ROI target classification for forecasts
    df_f_scored, vol_thresh_f, pce_thresh_f = scoring.add_roi_classification(df_f_scored, 'pred_count', 'pred_weighted_pce')
    
    # Clean up coordinates and add tooltip strings
    df_f_scored_clean = df_f_scored.dropna(subset=['lat_center', 'lon_center']).copy()
    df_f_scored_clean = add_tooltip_strings(df_f_scored_clean, 'pred_count', 'pred_weighted_pce')
    
    # Precompute normalized values and color arrays to ensure tabs/views can load independently
    if not df_f_scored_clean.empty:
        p_v_max = df_f_scored_clean['pred_count'].max()
        p_v_min = df_f_scored_clean['pred_count'].min()
        if p_v_max > p_v_min:
            df_f_scored_clean['norm_pred_v'] = np.sqrt(df_f_scored_clean['pred_count'] - p_v_min) / (np.sqrt(p_v_max - p_v_min) + 1e-6)
        else:
            df_f_scored_clean['norm_pred_v'] = 0.0
            
        colors_pv = []
        for val in df_f_scored_clean['norm_pred_v']:
            r = int(255 - (255 - 189) * val)
            g = int(255 - (255 - 0) * val)
            b = int(178 - (178 - 38) * val)
            colors_pv.append([r, g, b, 160])
        df_f_scored_clean['fill_color_pv'] = colors_pv

        p_c_max = df_f_scored_clean['pred_weighted_pce'].max()
        p_c_min = df_f_scored_clean['pred_weighted_pce'].min()
        if p_c_max > p_c_min:
            df_f_scored_clean['norm_pred_c'] = np.sqrt(df_f_scored_clean['pred_weighted_pce'] - p_c_min) / (np.sqrt(p_c_max - p_c_min) + 1e-6)
        else:
            df_f_scored_clean['norm_pred_c'] = 0.0
            
        colors_pc = []
        for val in df_f_scored_clean['norm_pred_c']:
            r = int(26 + (249 - 26) * val)
            g = int(54 + (115 - 54) * val)
            b = int(93 + (22 - 93) * val)
            colors_pc.append([r, g, b, 170])
        df_f_scored_clean['fill_color_pc'] = colors_pc

        colors_outline_f = []
        for val in df_f_scored_clean['norm_pred_c']:
            r = int(253)
            g = int(141 + (254 - 141) * (1-val))
            b = int(60 + (196 - 60) * (1-val))
            colors_outline_f.append([r, g, b, 255])
        df_f_scored_clean['line_color_pred_c'] = colors_outline_f

        pred_count_cutoff = global_pred_count_75th
        pred_pce_cutoff = global_pred_pce_75th
        df_f_scored_clean['pred_density_active'] = df_f_scored_clean['pred_count'] >= pred_count_cutoff
        df_f_scored_clean['pred_congestion_active'] = df_f_scored_clean['pred_weighted_pce'] >= pred_pce_cutoff
    else:
        df_f_scored_clean['norm_pred_v'] = []
        df_f_scored_clean['fill_color_pv'] = []
        df_f_scored_clean['norm_pred_c'] = []
        df_f_scored_clean['fill_color_pc'] = []
        df_f_scored_clean['line_color_pred_c'] = []
        df_f_scored_clean['pred_density_active'] = []
        df_f_scored_clean['pred_congestion_active'] = []
    df_f_scored_clean['pred_spillover_significant'] = (
        (df_f_scored_clean['gi_zscore'] >= 1.96) &
        (df_f_scored_clean['gi_pvalue'] < 0.05)
    )
    df_f_scored_clean['pred_enforcement_visible'] = (
        (df_f_scored_clean['roi_class'] != 'Standard') |
        df_f_scored_clean['pred_congestion_active']
    )
    
    # KPIs Layout (Predictions)
    pk_col1, pk_col2, pk_col3, pk_col4 = st.columns(4)
    with pk_col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-val">{df_f_scored_clean['pred_count'].sum():.1f}</div>
            <div class="kpi-label">Predicted Violations</div>
        </div>
        """, unsafe_allow_html=True)
    with pk_col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-val">{df_f_scored_clean['pred_weighted_pce'].sum():.1f}</div>
            <div class="kpi-label">Predicted Congestion Score</div>
        </div>
        """, unsafe_allow_html=True)
    with pk_col3:
        pred_hotspots_count = len(df_f_scored_clean[(df_f_scored_clean['gi_zscore'] >= 1.96) & (df_f_scored_clean['gi_pvalue'] < 0.05)])
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-val metric-red">{pred_hotspots_count}</div>
            <div class="kpi-label">Predicted Hotspot Cells (Gi*)</div>
        </div>
        """, unsafe_allow_html=True)
    with pk_col4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-val metric-purple">{model_meta['auc']:.1%}</div>
            <div class="kpi-label">Model AUC-ROC Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write(f"Showing forecasts for: **{selected_f_date} (Hour: {f_hour}, Time Bucket: {f_time_bucket})**")
    
    # Future prediction tabs
    pred_tab_option = st.segmented_control(
        "Prediction Layer",
        options=[
            "🔮 Predicted Violations",
            "🔮 Predicted Congestion Impact",
            "🔮 Predicted Combined Map",
            "🔮 Predicted Spillover",
            "🚔 Predicted Enforcement Plan",
            "🔮 Predicted Clusters (DBSCAN)",
            "📊 Model Metrics & Features"
        ],
        default="🔮 Predicted Violations",
        key="pred_analysis_layer_select"
    )
    if not pred_tab_option:
        pred_tab_option = "🔮 Predicted Violations"

    prediction_csv = build_prediction_download(
        df_f_scored_clean,
        selected_f_date,
        f_hour,
        f_time_bucket,
        pred_tab_option,
    )
    prediction_file_name = (
        "prediction_forecast_"
        f"{clean_filename_part(selected_f_date)}_"
        f"hour_{clean_filename_part(f_hour)}_"
        f"bucket_{clean_filename_part(f_time_bucket)}_"
        f"mode_{clean_filename_part(pred_tab_option)}.csv"
    )
    st.download_button(
        "⬇️ Download Current Forecast CSV",
        data=prediction_csv,
        file_name=prediction_file_name,
        mime="text/csv",
        width="stretch",
        key=build_map_key("pred_download_csv", selected_f_date, f_hour, f_time_bucket, pred_tab_option),
    )
        
    # Forecast Tab 1: Predicted Violations
    if pred_tab_option == "🔮 Predicted Violations":
        st.subheader("🔴 Predicted Violation Counts Map")
        st.markdown("XGBoost forecasts of parking violations per H3-8 cell in the next hour.")
        render_color_key("density")
        
        # Map layer
        
        pv_layer = pdk.Layer(
            "H3HexagonLayer",
            data=df_f_scored_clean[df_f_scored_clean['pred_density_active']],
            id="pred_violations_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color_pv",
            stroked=True,
            filled=True,
            get_line_color="[44, 62, 80, 80]",
            line_width_min_pixels=0.8,
            pickable=True,
        )
        
        render_pydeck_map(
            [pv_layer],
            map_tooltip,
            build_map_key("pred_density_map", selected_f_date, f_hour, f_time_bucket),
            df_f_scored_clean[df_f_scored_clean['pred_density_active']],
        )
        
        # Busiest predicted table
        st.write("### 🏆 Busiest Predicted Zones")
        pred_busiest = df_f_scored_clean[df_f_scored_clean['pred_density_active']].sort_values('pred_count', ascending=False).head(10).copy()
        
        tbl_pred_busiest = pred_busiest[['h3_8', 'junction_name', 'pred_count', 'pred_weighted_pce']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'pred_count': 'Predicted Violations / Hr',
            'pred_weighted_pce': 'Predicted PCE Score'
        })
        
        st.dataframe(
            tbl_pred_busiest,
            width="stretch",
            hide_index=True,
            key="pred_busiest_tbl"
        )
        
        render_cell_inspector(
            df_f_scored_clean[df_f_scored_clean['pred_density_active']],
            cell_meta,
            'pred_count',
            build_map_key("pred_density_inspector", selected_f_date, f_hour, f_time_bucket),
            "Inspect a forecasted H3 cell:"
        )

    # Forecast Tab 2: Predicted Congestion Impact
    elif pred_tab_option == "🔮 Predicted Congestion Impact":
        st.subheader("🟠 Predicted Congestion Impact Map")
        
        st.markdown("""
        > [!NOTE]
        > **Reasoning behind the Congestion Impact Score (CIS):**
        > Traditional traffic enforcement prioritizes raw violation count, which treats a scooter parked on a side alley the same as a bus blocking a major intersection.
        > The Congestion Impact Score (CIS) directly quantifies traffic friction by multiplying the count of each violation by its standard **Passenger Car Equivalent (PCE)** weight (from IRC:106-1990) and a **Location Factor** reflecting HCM Chapter 18:
        > 
        > $$\text{CIS} = \sum_{j} (\text{PCE}_j \times \text{Location Factor}_j)$$
        > 
        > - **PCE Weights**: Scooter/2W = 0.5, Car/Jeep = 1.0, Passenger Auto = 0.75, Goods Auto = 1.4, Maxi-Cab = 2.0, Bus/Heavy Lorry = 3.7.
        > - **Location Factors**: 1.50 for junctions, intersections, signals, and zebra crossings; 1.30 for main/arterial roads; 1.00 for minor/local roads.
        > This ensures high-occupancy corridors and key bottlenecks receive prioritized enforcement.
        """)
        render_color_key("congestion")
        
        # Map layer
        
        pc_layer = pdk.Layer(
            "H3HexagonLayer",
            data=df_f_scored_clean[df_f_scored_clean['pred_congestion_active']],
            id="pred_congestion_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color_pc",
            stroked=True,
            filled=True,
            get_line_color="[17, 17, 17, 80]",
            line_width_min_pixels=0.8,
            pickable=True,
        )
        
        render_pydeck_map(
            [pc_layer],
            map_tooltip,
            build_map_key("pred_congestion_map", selected_f_date, f_hour, f_time_bucket),
            df_f_scored_clean[df_f_scored_clean['pred_congestion_active']],
        )
        
        # Top Predicted Congestion Bottlenecks table
        st.write("### 🏆 Top Predicted Congestion Bottlenecks")
        pred_busiest_congestion = df_f_scored_clean[df_f_scored_clean['pred_congestion_active']].sort_values('pred_weighted_pce', ascending=False).head(10).copy()
        tbl_pred_congestion = pred_busiest_congestion[['h3_8', 'junction_name', 'pred_weighted_pce', 'pred_count']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'pred_weighted_pce': 'Predicted PCE Score',
            'pred_count': 'Predicted Violations / Hr'
        })
        st.dataframe(
            tbl_pred_congestion,
            width="stretch",
            hide_index=True,
            key="pred_congestion_tbl"
        )
        
        # Selectbox inspector for predictions
        st.write("### 🔍 H3 Cell Predicted Profile Inspector")
        render_cell_inspector(
            df_f_scored_clean[df_f_scored_clean['pred_congestion_active']],
            cell_meta,
            'pred_count',
            'pred_inspect_selectbox',
            "Select an H3 cell to inspect predicted metrics:"
        )

    # Forecast Tab 3: Predicted Combined Map
    elif pred_tab_option == "🔮 Predicted Combined Map":
        st.subheader("🟣 Predicted Combined Map Overlay")
        st.markdown("Dual-layer map that overlays **predicted violation counts** (filled hexagons) and **predicted CIS congestion impact** (outlined borders). Identifies forecasted Silent Bottlenecks.")
        render_color_key("combined")
        
        # Map layers
        
        fill_layer_f = pdk.Layer(
            "H3HexagonLayer",
            data=df_f_scored_clean[df_f_scored_clean['pred_density_active']],
            id="pred_combined_fill_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color_pv",
            stroked=False,
            filled=True,
            pickable=True,
        )
        
        outline_layer_f = pdk.Layer(
            "H3HexagonLayer",
            data=df_f_scored_clean[df_f_scored_clean['pred_congestion_active']],
            id="pred_combined_outline_layer",
            get_hexagon="h3_8",
            get_line_color="line_color_pred_c",
            line_width_min_pixels=3,
            stroked=True,
            filled=False,
            pickable=True,
        )
        
        render_pydeck_map(
            [fill_layer_f, outline_layer_f],
            map_tooltip,
            build_map_key("pred_combined_map", selected_f_date, f_hour, f_time_bucket),
            df_f_scored_clean[df_f_scored_clean['pred_density_active'] | df_f_scored_clean['pred_congestion_active']],
        )

        # Top Predicted Combined Risk Areas table
        st.write("### 🏆 Top Predicted Combined Risk Areas")
        combined_pred_busiest = df_f_scored_clean[df_f_scored_clean['pred_density_active'] | df_f_scored_clean['pred_congestion_active']].copy()
        combined_pred_busiest['combined_score'] = combined_pred_busiest['pred_count'] + combined_pred_busiest['pred_weighted_pce']
        combined_pred_busiest = combined_pred_busiest.sort_values('combined_score', ascending=False).head(10).copy()
        
        tbl_pred_combined = combined_pred_busiest[['h3_8', 'junction_name', 'pred_count', 'pred_weighted_pce']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'pred_count': 'Predicted Violations / Hr',
            'pred_weighted_pce': 'Predicted PCE Score'
        })
        st.dataframe(
            tbl_pred_combined,
            width="stretch",
            hide_index=True,
            key="pred_combined_tbl"
        )

        st.write("### 🔍 H3 Cell Predicted Profile Inspector")
        render_cell_inspector(
            df_f_scored_clean[df_f_scored_clean['pred_density_active'] | df_f_scored_clean['pred_congestion_active']],
            cell_meta,
            'pred_weighted_pce',
            build_map_key("pred_combined_inspector", selected_f_date, f_hour, f_time_bucket),
            "Inspect a predicted combined-map H3 cell:"
        )
        
        st.markdown("""
        **Legend Map Guide:**
        *   **Filled Hexagons**: Predicted violation counts (faint yellow = low count, solid red = high count).
        *   **Outline Borders**: Predicted PCE-weighted congestion score (bright orange/red = severe road obstruction).
        *   **Silent Bottleneck (Tier 2)**: Cells with light fills (low volume) but thick orange outlines (high traffic flow degradation). Zero-tolerance towing recommended.
        """)

    # Forecast Tab 4: Predicted Spillover
    elif pred_tab_option == "🔮 Predicted Spillover":
        st.subheader("🔵 Predicted Spillover Hotspots Map")
        
        st.markdown("""
        > [!NOTE]
        > **Reasoning behind Spatial Spillover (Local Getis-Ord $G_i^*$):**
        > Parking blockages on one street do not stay isolated; they degrade speed on adjacent streets and spill over to surrounding junctions.
        > We model this spatial dependency using a row-standardized weights matrix ($W$) representing **1-ring adjacent H3 neighbors** ($k=1$).
        > The Local Getis-Ord $G_i^*$ statistic compares the local sum of congestion in each cell and its immediate neighbors against the global sum across all city cells:
        > 
        > $$G_i^* = \\frac{\sum_{j} w_{ij} x_j - \\bar{X} \sum_{j} w_{ij}}{S \sqrt{\\frac{n\sum_{j} w_{ij}^2 - (\sum_{j} w_{ij})^2}{n-1}}}$$
        > 
        > Where $Z_i \ge 1.96$ indicates a statistically significant ($p_i < 0.05$) hot zone where high congestion values are clustered together, identifying systemic neighborhood spillovers.
        """)
        render_color_key("spillover")
        
        p_z_max = df_f_scored_clean['gi_zscore'].max()
        colors_pz = []
        for idx, row in df_f_scored_clean.iterrows():
            z = row['gi_zscore']
            p = row['gi_pvalue']
            if z >= 1.96 and p < 0.05:
                val = min(1.0, (z - 1.96) / (max(2.58, p_z_max) - 1.96 + 1e-6))
                r = int(56 + (99 - 56) * val)
                g = int(189 + (102 - 189) * val)
                b = int(248 + (241 - 248) * val)
                colors_pz.append([r, g, b, 230])
            else:
                colors_pz.append([0, 0, 0, 0])
        df_f_scored_clean['fill_color_pz'] = colors_pz
        
        pred_sig_spill = df_f_scored_clean[
            df_f_scored_clean['pred_spillover_significant']
        ].sort_values('gi_zscore', ascending=False).copy()
        
        pz_layer = pdk.Layer(
            "H3HexagonLayer",
            data=pred_sig_spill,
            id="pred_spillover_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color_pz",
            stroked=True,
            filled=True,
            get_line_color="[17, 17, 17, 80]",
            line_width_min_pixels=0.8,
            pickable=True,
        )
        
        render_pydeck_map(
            [pz_layer],
            map_tooltip,
            build_map_key("pred_spillover_map", selected_f_date, f_hour, f_time_bucket),
            pred_sig_spill,
        )
        
        # Display significant cells in a table
        st.write("### 📈 Predicted Spillover Hotspots (Confidence ≥ 95%)")
        
        tbl_pred_spill = pred_sig_spill[['h3_8', 'junction_name', 'gi_zscore', 'gi_pvalue', 'pred_weighted_pce']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'gi_zscore': 'Gi* Z-Score',
            'gi_pvalue': 'p-value',
            'pred_weighted_pce': 'Predicted PCE Score'
        })
        
        st.dataframe(
            tbl_pred_spill,
            width="stretch",
            hide_index=True,
            key="pred_spill_tbl"
        )
        
        render_cell_inspector(
            pred_sig_spill,
            cell_meta,
            'pred_weighted_pce',
            build_map_key("pred_spillover_inspector", selected_f_date, f_hour, f_time_bucket),
            "Inspect a predicted spillover H3 cell:"
        )

    # Forecast Tab 5: Predicted Enforcement Plan
    elif pred_tab_option == "🚔 Predicted Enforcement Plan":
        st.subheader("🚔 Predicted Enforcement Plan (Strategic Targets)")
        
        st.markdown("""
        > [!NOTE]
        > **Reasoning behind the Enforcement Plan (Strategic ROI Tiers):**
        > Enforcement resources (towing trucks, field patrols) are finite. We classify active cells into three distinct enforcement tiers relative to the **90th percentile** of violations ($P_{90,\text{vol}}$) and CIS scores ($P_{90,\text{cis}}$) within the filtered timeframe:
        > 
        > 1. **Tier 1: Max Disruption** ($\text{Violations} \ge P_{90,\text{vol}}$ AND $\text{CIS} \ge P_{90,\text{cis}}$):
        >    High violation frequency causing severe flow obstruction. Priority target for intensive enforcement patrols.
        > 2. **Tier 2: Silent Bottleneck** ($\text{Violations} < P_{90,\text{vol}}$ AND $\text{CIS} \ge P_{90,\text{cis}}$):
        >    Fewer tickets but critical capacity obstruction (e.g. heavy vehicles blocking narrow arterial links). Targeted for zero-tolerance towing.
        > 3. **Tier 3: Volume Hotspot** ($\text{Violations} \ge P_{90,\text{vol}}$ AND $\text{CIS} < P_{90,\text{cis}}$):
        >    High violation frequency but minimal flow impact (e.g. two-wheelers parked on wide side alleys). Targeted for routine ticketing.
        > 4. **Standard**: All other active locations requiring standard monitoring.
        """)
        render_color_key("enforcement")
        
        colors_pred_roi = []
        for idx, row in df_f_scored_clean.iterrows():
            tier = row['roi_class']
            if tier == 'Tier 1: Max Disruption':
                colors_pred_roi.append([239, 68, 68, 190]) # Red
            elif tier == 'Tier 2: Silent Bottleneck':
                colors_pred_roi.append([249, 115, 22, 190]) # Orange
            elif tier == 'Tier 3: Volume Hotspot':
                colors_pred_roi.append([234, 179, 8, 190]) # Yellow
            elif row['pred_density_active'] or row['pred_congestion_active']:
                colors_pred_roi.append([148, 163, 184, 80]) # Faint Grey
            else:
                colors_pred_roi.append([71, 85, 105, 30])
        df_f_scored_clean['fill_color_pred_roi'] = colors_pred_roi
        pred_enforcement_visible = df_f_scored_clean[
            df_f_scored_clean['pred_enforcement_visible']
        ]
        
        pred_roi_layer = pdk.Layer(
            "H3HexagonLayer",
            data=pred_enforcement_visible,
            id="pred_roi_layer",
            get_hexagon="h3_8",
            get_fill_color="fill_color_pred_roi",
            stroked=True,
            filled=True,
            get_line_color="[17, 17, 17, 80]",
            line_width_min_pixels=0.8,
            pickable=True,
        )
        
        render_pydeck_map(
            [pred_roi_layer],
            map_tooltip,
            build_map_key("pred_enforcement_map", selected_f_date, f_hour, f_time_bucket),
            pred_enforcement_visible,
        )
        
        # Table of predicted strategic targets
        st.write("### 🚔 Predicted Strategic Targets for Patrol Scheduling")
        pred_targets = pred_enforcement_visible[pred_enforcement_visible['roi_class'] != 'Standard'].sort_values('pred_weighted_pce', ascending=False).copy()
        
        tbl_pred_targets = pred_targets[['h3_8', 'junction_name', 'roi_class', 'pred_count', 'pred_weighted_pce']].rename(columns={
            'h3_8': 'H3 Index',
            'junction_name': 'Dominant Junction / Area',
            'roi_class': 'Strategic Enforcement Classification',
            'pred_count': 'Predicted Violations / Hr',
            'pred_weighted_pce': 'Predicted PCE Score'
        })
        
        st.dataframe(
            tbl_pred_targets,
            width="stretch",
            hide_index=True,
            key="pred_targets_tbl"
        )
        
        render_cell_inspector(
            pred_enforcement_visible,
            cell_meta,
            'pred_weighted_pce',
            build_map_key("pred_enforcement_inspector", selected_f_date, f_hour, f_time_bucket),
            "Inspect a predicted strategic-enforcement H3 cell:"
        )

    # Forecast Tab 6: Predicted Clusters
    elif pred_tab_option == "🔮 Predicted Clusters (DBSCAN)":
        st.subheader("🟢 Predicted Spatial Clusters (DBSCAN Simulation)")
        st.markdown("Simulates future individual violation coordinates based on forecast volumes, then clusters them to identify localized micro-congestion points. Colored by violation code.")
        render_color_key("dbscan")
        
        # DBSCAN parameters capped 50-500 meters, increments of 50
        f_dbscan_eps = st.slider("Forecast Clustering Radius (meters)", min_value=50, max_value=500, value=200, step=50, key="pred_dbscan_eps_slider")
        f_dbscan_min_samples = st.slider("Forecast Minimum Violation Count per Cluster", min_value=5, max_value=100, value=15, step=5, key="pred_dbscan_min_slider")
        
        # Sample points inside cells proportional to prediction count
        simulated_points = []
        for row in df_f_scored_clean.itertuples():
            count = int(np.round(row.pred_count))
            if count > 0:
                # Sample violation codes based on historical percentages
                p_wrong = getattr(row, 'pct_wrong_park', 0.4)
                p_no = getattr(row, 'pct_no_park', 0.3)
                p_main = getattr(row, 'pct_main_road', 0.2)
                p_foot = getattr(row, 'pct_footpath', 0.1)
                
                # Normalize probabilities
                p_sum = p_wrong + p_no + p_main + p_foot
                if p_sum > 0:
                    probs = [p_wrong/p_sum, p_no/p_sum, p_main/p_sum, p_foot/p_sum]
                else:
                    probs = [0.4, 0.3, 0.2, 0.1]
                    
                vtypes = ['WRONG PARKING', 'NO PARKING', 'PARKING IN A MAIN ROAD', 'PARKING ON FOOTPATH']
                
                hour_val = int(f_hour) if isinstance(f_hour, int) or (isinstance(f_hour, str) and f_hour.isdigit()) else 9
                np.random.seed(stable_seed(row.h3_8, hour_val))
                # Probabilistic sampling to ensure active micro-congestion cells aren't rounded away
                final_count = count
                if final_count == 0 and row.pred_count > 0:
                    np.random.seed(stable_seed(row.h3_8, hour_val, "fractional"))
                    if np.random.rand() < row.pred_count:
                        final_count = 1
                for _ in range(final_count):
                    lat_noise = np.random.normal(0, 0.0009)
                    lon_noise = np.random.normal(0, 0.0009)
                    sampled_vtype = np.random.choice(vtypes, p=probs)
                    simulated_points.append({
                        'latitude': row.lat_center + lat_noise,
                        'longitude': row.lon_center + lon_noise,
                        'primary_violation': sampled_vtype,
                        'cluster': -1
                    })
                    
        sim_df = pd.DataFrame(simulated_points)
        
        # Filter by simulated violation type
        if not sim_df.empty:
            f_v_types = ["All"] + sorted(sim_df['primary_violation'].unique().tolist())
            selected_f_vtype = st.selectbox("Filter Forecasted Violation Type", options=f_v_types, key="pred_dbscan_vtype_select")
            if selected_f_vtype != "All":
                sim_df = sim_df[sim_df['primary_violation'] == selected_f_vtype].copy()
        
        if not sim_df.empty and len(sim_df) >= f_dbscan_min_samples:
            # Run clustering on simulated points
            sim_clustered = data_processing.run_dbscan_clustering(sim_df, eps_meters=f_dbscan_eps, min_samples=f_dbscan_min_samples)
            
            # Map colors based on cluster IDs
            unique_sim = sim_clustered['cluster'].unique()
            np.random.seed(42)
            sim_colors = {}
            for c in unique_sim:
                if c == -1:
                    sim_colors[c] = [148, 163, 184, 80] # Noise
                else:
                    sim_colors[c] = list(np.random.randint(50, 255, size=3).tolist()) + [210]
            sim_clustered['color'] = sim_clustered['cluster'].map(sim_colors)
            sim_clustered['coordinates_str'] = sim_clustered.apply(
                lambda row: f"{row['latitude']:.5f}, {row['longitude']:.5f}",
                axis=1
            )
            sim_clustered['latitude_str'] = sim_clustered['latitude'].map(lambda x: f"{x:.5f}")
            sim_clustered['longitude_str'] = sim_clustered['longitude'].map(lambda x: f"{x:.5f}")
            
            tooltip_sim = {
                "html": """
                <div style="font-family: Arial; font-size: 12px; padding: 10px; border-radius: 8px;">
                    <b>Cluster ID:</b> {cluster}<br/>
                    <b>Type:</b> {primary_violation}<br/>
                    <b>Latitude:</b> {latitude_str}<br/>
                    <b>Longitude:</b> {longitude_str}
                </div>
                """,
                "style": {"color": "white", "backgroundColor": "#0f172a", "border": "1px solid rgba(255,255,255,0.1)"}
            }
            
            sim_points_layer = pdk.Layer(
                "ScatterplotLayer",
                data=sim_clustered,
                id="pred_dbscan_points",
                get_position="[longitude, latitude]",
                get_color="color",
                get_radius=15,
                pickable=True,
            )
            
            sim_event = render_pydeck_map(
                [sim_points_layer],
                tooltip_sim,
                build_map_key("pred_dbscan_map", selected_f_date, f_hour, f_time_bucket, selected_f_vtype),
                show_selected_profile=False,
            )
            display_selected_dbscan(get_selected_dbscan_object(sim_event))

            sim_centroids = sim_clustered[sim_clustered['cluster'] >= 0].groupby('cluster').agg(
                latitude=('latitude', 'mean'),
                longitude=('longitude', 'mean'),
                violations=('cluster', 'count'),
                dominant_violation=('primary_violation', lambda x: x.value_counts().index[0])
            ).reset_index()
            if not sim_centroids.empty:
                sim_centroids['primary_violation'] = sim_centroids['dominant_violation']
                sim_centroids['coordinates_str'] = sim_centroids.apply(
                    lambda row: f"{row['latitude']:.5f}, {row['longitude']:.5f}",
                    axis=1
                )
                sim_centroids['latitude_str'] = sim_centroids['latitude'].map(lambda x: f"{x:.5f}")
                sim_centroids['longitude_str'] = sim_centroids['longitude'].map(lambda x: f"{x:.5f}")
                sim_labels = {
                    row.cluster: f"Cluster {row.cluster} - {row.dominant_violation} ({row.violations} projected violations)"
                    for row in sim_centroids.itertuples()
                }
                selected_sim_cluster = st.selectbox(
                    "Inspect a predicted DBSCAN cluster centroid:",
                    options=sim_centroids['cluster'].tolist(),
                    format_func=lambda cluster_id: sim_labels.get(cluster_id, str(cluster_id)),
                    key=build_map_key("pred_dbscan_cluster_inspector", selected_f_date, f_hour, f_time_bucket, selected_f_vtype)
                )
                selected_sim_centroid = sim_centroids[sim_centroids['cluster'] == selected_sim_cluster]
                if not selected_sim_centroid.empty:
                    display_selected_dbscan(selected_sim_centroid.iloc[0].to_dict())
            
            # Show number of predicted clusters
            n_sim_clusters = len(unique_sim) - (1 if -1 in unique_sim else 0)
            st.success(f"Forecast model projects **{n_sim_clusters} dense clusters** forming during this hour.")

            # Clustered Hotspot Centroids Summary table
            if not sim_centroids.empty:
                st.write("### 📍 Predicted Clustered Hotspot Centroids Summary")
                st.dataframe(
                    sim_centroids.rename(columns={
                        'cluster': 'Cluster ID',
                        'latitude': 'Latitude Centroid',
                        'longitude': 'Longitude Centroid',
                        'violations': 'Total Projected Violations in Cluster',
                        'dominant_violation': 'Dominant Violation Type'
                    })[['Cluster ID', 'Latitude Centroid', 'Longitude Centroid', 'Total Projected Violations in Cluster', 'Dominant Violation Type']],
                    width="stretch",
                    hide_index=True,
                    key=build_map_key("pred_dbscan_table", selected_f_date, f_hour, f_time_bucket, selected_f_vtype)
                )
        else:
            st.info("Predicted violations are too sparse during this hour/type combination to form dense cluster coordinates.")

    # Forecast Tab 7: Performance and SHAP
    elif pred_tab_option == "📊 Model Metrics & Features":
        st.subheader("📊 XGBoost Prediction Model Specs")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="glass-card">
                <h3>Regression Performance</h3>
                • <b>MAE (Mean Abs Error):</b> {model_meta['mae']:.3f}<br/>
                • <b>RMSE (Root Mean Sq Error):</b> {model_meta['rmse']:.3f}<br/>
                • <b>R² Score (Variance Explained):</b> {model_meta['r2']:.3f}
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="glass-card">
                <h3>Classifier Performance</h3>
                • <b>AUC-ROC Score:</b> {model_meta['auc']:.3f}<br/>
                • <b>Imbalance Handler:</b> scale_pos_weight = 7.7:1<br/>
                • <b>Classification Threshold:</b> count ≥ 10/hr
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="glass-card">
                <h3>Model Features</h3>
                • <b>Spatial:</b> cell coordinates + historical mean<br/>
                • <b>Temporal:</b> cyclical hour/day/month encodings<br/>
                • <b>Lags:</b> shifts 1h, 2h, 3h, 24h, 168h<br/>
                • <b>Rolling:</b> 3h, 6h, 24h, 7d rolling averages
            </div>
            """, unsafe_allow_html=True)
            
        st.write("### 📈 Predicted Violations Time Series (7-Day Trend)")
        timeline_df = pred_df.groupby('datetime').agg({'pred_count': 'sum'}).reset_index()
        st.line_chart(
            data=timeline_df,
            x='datetime',
            y='pred_count',
            width="stretch"
        )
