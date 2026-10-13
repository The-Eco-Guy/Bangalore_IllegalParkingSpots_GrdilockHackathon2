"""
patrol.py — Tactical Patrol Route Optimizer & Field Dispatch Generator
Part of Bangalore Gridlock Intelligence (Phase B1).
Transforms spatial-statistical clusters into actionable, sequential patrol manifests
for traffic police inspectors, field constables, and towing vehicle dispatchers.
"""

import math
import urllib.parse
import pandas as pd
import numpy as np

# Standard coordinates for Bangalore Traffic Police Headquarters (Infantry Road)
BTP_HQ_LAT = 12.9815
BTP_HQ_LON = 77.5925
AVERAGE_URBAN_SPEED_KMH = 16.0  # Realistic congested patrol transit speed in Bangalore

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two GPS coordinates in kilometers.
    """
    R = 6371.0  # Earth's radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def assign_tactical_action(row):
    """Suggest verification tasks; statistical priorities are not enforcement orders."""
    if row.get('is_exploration_target', False):
        return "Verify local conditions and recording coverage"
    tier = str(row.get('roi_class', 'Standard'))
    return {
        'Tier 1: Max Disruption': 'Inspect potential obstruction and verify high-volume parking activity',
        'Tier 2: Silent Bottleneck': 'Verify potential traffic obstruction despite lower recorded volume',
        'Tier 3: Volume Hotspot': 'Review parking activity and verify applicable restrictions',
    }.get(tier, 'Routine observation and verification')


def assign_optimal_window(row):
    """Use an explicit selected window or hour, without inventing a peak shift."""
    if row.get('selected_hours'):
        return str(row['selected_hours'])
    hour = row.get('hour')
    try:
        h = int(hour)
        if 0 <= h <= 23:
            return f"{h:02d}:00–{h + 1:02d}:00 IST"
    except (ValueError, TypeError, OverflowError):
        pass
    return "Selected planning window; confirm deployment time"


def optimize_patrol_route(
    df,
    start_lat=BTP_HQ_LAT,
    start_lon=BTP_HQ_LON,
    start_name="Traffic Police HQ (Infantry Rd)",
    max_stops=8,
    pce_col='weighted_pce',
    vol_col='violations',
    selected_only=False
):
    """
    Nearest-neighbor TSP heuristic to sequence high-priority enforcement cells
    into an optimal multi-stop patrol route.
    """
    if df.empty:
        return pd.DataFrame()

    pce_field = pce_col if pce_col in df.columns else ('pred_weighted_pce' if 'pred_weighted_pce' in df.columns else 'weighted_pce')
    vol_field = vol_col if vol_col in df.columns else ('pred_count' if 'pred_count' in df.columns else 'violations')

    # Filter candidates: prioritize Tier 1, Tier 2, Exploration Targets, and high PCE
    candidates = df.copy()
    if not selected_only and 'roi_class' in candidates.columns:
        priority_mask = (candidates['roi_class'].isin(['Tier 1: Max Disruption', 'Tier 2: Silent Bottleneck']))
        if 'is_exploration_target' in candidates.columns:
            priority_mask = priority_mask | candidates['is_exploration_target']
        if priority_mask.any():
            candidates = candidates[priority_mask]

    # Sort by impact to pick top candidates
    candidates = candidates.sort_values([pce_field, 'h3_8'], ascending=[False, True])
    if not selected_only:
        candidates = candidates.head(max_stops * 2)

    # Clean coordinates
    lat_col = 'lat_center' if 'lat_center' in candidates.columns else 'lat'
    lon_col = 'lon_center' if 'lon_center' in candidates.columns else 'lon'
    candidates[lat_col] = pd.to_numeric(candidates[lat_col], errors='coerce')
    candidates[lon_col] = pd.to_numeric(candidates[lon_col], errors='coerce')
    candidates = candidates[candidates[lat_col].between(-90, 90) & candidates[lon_col].between(-180, 180)].drop_duplicates('h3_8').copy()
    if not math.isfinite(start_lat) or not math.isfinite(start_lon) or not -90 <= start_lat <= 90 or not -180 <= start_lon <= 180:
        raise ValueError('Patrol origin must have valid latitude and longitude.')

    if candidates.empty:
        return pd.DataFrame()

    # Nearest neighbor sequencing
    unvisited = candidates.to_dict('records')
    route_stops = []
    
    current_lat = start_lat
    current_lon = start_lon
    cumulative_dist = 0.0
    cumulative_time_min = 0.0

    while unvisited and len(route_stops) < max_stops:
        # Find closest unvisited target
        best_idx = -1
        min_dist = float('inf')
        for i, item in enumerate(unvisited):
            d = haversine_distance_km(current_lat, current_lon, item[lat_col], item[lon_col])
            if d < min_dist:
                min_dist = d
                best_idx = i

        chosen = unvisited.pop(best_idx)
        leg_dist = round(min_dist, 2)
        leg_time = round((min_dist / AVERAGE_URBAN_SPEED_KMH) * 60.0, 1)
        cumulative_dist += leg_dist
        cumulative_time_min += leg_time

        c_lat = float(chosen[lat_col])
        c_lon = float(chosen[lon_col])
        junc = str(chosen.get('junction_name', 'Unnamed Location')).strip()
        cell_id = str(chosen.get('h3_8', ''))
        tier = str(chosen.get('roi_class', 'Standard'))
        pce_val = round(float(chosen.get(pce_field, 0.0) or 0.0), 2)
        vol_val = round(float(chosen.get(vol_field, 0.0) or 0.0), 4)

        single_nav_url = f"https://www.google.com/maps/dir/?api=1&destination={c_lat:.5f},{c_lon:.5f}"

        route_stops.append({
            'stop_order': len(route_stops) + 1,
            'junction_name': junc,
            'h3_8': cell_id,
            'latitude': round(c_lat, 5),
            'longitude': round(c_lon, 5),
            'roi_class': tier,
            'pce_impact': pce_val,
            'violation_volume': vol_val,
            'tactical_action': assign_tactical_action(chosen),
            'optimal_window': assign_optimal_window(chosen),
            'leg_distance_km': leg_dist,
            'cumulative_distance_km': round(cumulative_dist, 2),
            'est_transit_time_min': leg_time,
            'cumulative_time_min': round(cumulative_time_min, 1),
            'google_maps_nav_link': single_nav_url
        })

        current_lat = c_lat
        current_lon = c_lon

    route_df = pd.DataFrame(route_stops)
    return route_df

def build_google_maps_multiwaypoint_url(route_df, start_lat=BTP_HQ_LAT, start_lon=BTP_HQ_LON):
    """
    Construct an official multi-waypoint Google Maps route link
    that can be tapped on a field officer's mobile device to immediately open navigation.
    """
    if route_df.empty:
        return ""
    
    origin = f"{start_lat:.5f},{start_lon:.5f}"
    dest_row = route_df.iloc[-1]
    destination = f"{dest_row['latitude']:.5f},{dest_row['longitude']:.5f}"
    
    if len(route_df) > 1:
        waypoint_rows = route_df.iloc[:-1]
        waypoints = "|".join([f"{r['latitude']:.5f},{r['longitude']:.5f}" for _, r in waypoint_rows.iterrows()])
        params = {
            'api': '1',
            'origin': origin,
            'destination': destination,
            'waypoints': waypoints,
            'travelmode': 'driving'
        }
    else:
        params = {
            'api': '1',
            'origin': origin,
            'destination': destination,
            'travelmode': 'driving'
        }
    
    return f"https://www.google.com/maps/dir/?{urllib.parse.urlencode(params)}"

def generate_whatsapp_dispatch_memo(route_df, start_name="Traffic Police HQ", shift_name="Selected planning window",
                                    start_lat=BTP_HQ_LAT, start_lon=BTP_HQ_LON, context=None):
    """Build a copyable, reviewable planning brief without sending a message."""
    if route_df.empty:
        return "No active targets in this patrol plan."
    context = context or {}
    lines = [
        "BTP PATROL PLANNING BRIEF",
        f"Source: {context.get('source', 'Planning selection')}",
        f"Dates: {context.get('selected_start_date', 'Unspecified')} to {context.get('selected_end_date', 'Unspecified')}",
        f"Planning window: {context.get('selected_hours', shift_name)}",
        f"Metric basis: {context.get('metric_basis', 'Selected-period estimates')}",
        f"Patrol origin: {start_name} ({start_lat:.5f}, {start_lon:.5f})",
        f"Straight-line distance estimate: {route_df['cumulative_distance_km'].iloc[-1]:.1f} km",
        f"Approximate transit: {route_df['cumulative_time_min'].iloc[-1]:.0f} min at 16 km/h; excludes time at stops",
        "Reviewed route stops:", "",
    ]
    for _, row in route_df.iterrows():
        lines.extend([
            f"{int(row['stop_order'])}. {row['junction_name']}",
            f"   Impact: {row['pce_impact']:.2f} | Volume: {row['violation_volume']:.2f}",
            f"   Suggested verification: {row['tactical_action']}",
            f"   Area pin: {row['google_maps_nav_link']}", "",
        ])
    lines.extend(["Road navigation (Google Maps):", build_google_maps_multiwaypoint_url(route_df, start_lat, start_lon), "",
                  "Verify current conditions, exact street location, and applicable restrictions before deciding enforcement action."])
    return "\n".join(lines)
