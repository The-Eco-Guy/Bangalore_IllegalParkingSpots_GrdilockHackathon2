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
    """
    Map analytical tier and risk profile to concrete field actions.
    """
    roi_tier = str(row.get('roi_class', 'Standard'))
    is_exploration = bool(row.get('is_exploration_target', False))
    
    if is_exploration:
        return "🔍 RECON PATROL (Verify Unpatrolled Chokepoint)"
    elif roi_tier == 'Tier 1: Max Disruption':
        return "🚨 HEAVY TOW SQUAD (Immediate Clearance & Impound)"
    elif roi_tier == 'Tier 2: Silent Bottleneck':
        return "⛔ ZERO-TOLERANCE TOW / CLAMP (Critical Flow Artery)"
    elif roi_tier == 'Tier 3: Volume Hotspot':
        return "📸 CONTACTLESS E-TICKETING SQUAD (High Volume)"
    else:
        return "🚔 ROUTINE VISIBILITY PATROL"

def assign_optimal_window(row):
    """
    Recommend target enforcement time window based on peak profile.
    """
    hour = row.get('hour', None)
    if hour is not None and str(hour) != 'All':
        try:
            h = int(hour)
            if 7 <= h <= 10:
                return "AM Peak (07:00 - 11:00)"
            elif 11 <= h <= 16:
                return "Midday Flow (11:00 - 16:30)"
            elif 17 <= h <= 21:
                return "PM Peak Blindspot (17:00 - 21:30)"
            else:
                return "Off-Peak Night (22:00 - 06:00)"
        except (ValueError, TypeError):
            pass
    return "PM Peak Blindspot (17:00 - 21:00)"

def optimize_patrol_route(
    df,
    start_lat=BTP_HQ_LAT,
    start_lon=BTP_HQ_LON,
    start_name="Traffic Police HQ (Infantry Rd)",
    max_stops=8,
    pce_col='weighted_pce',
    vol_col='violations'
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
    if 'roi_class' in candidates.columns:
        priority_mask = (candidates['roi_class'].isin(['Tier 1: Max Disruption', 'Tier 2: Silent Bottleneck']))
        if 'is_exploration_target' in candidates.columns:
            priority_mask = priority_mask | candidates['is_exploration_target']
        if priority_mask.any():
            candidates = candidates[priority_mask]

    # Sort by impact to pick top candidates
    candidates = candidates.sort_values(pce_field, ascending=False).head(max_stops * 2)

    # Clean coordinates
    lat_col = 'lat_center' if 'lat_center' in candidates.columns else 'lat'
    lon_col = 'lon_center' if 'lon_center' in candidates.columns else 'lon'
    candidates = candidates.dropna(subset=[lat_col, lon_col]).copy()

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
        vol_val = round(float(chosen.get(vol_field, 0.0) or 0.0), 1)

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

def generate_whatsapp_dispatch_memo(route_df, start_name="Traffic Police HQ", shift_name="PM Peak Shift (17:00 - 21:00)"):
    """
    Generate a WhatsApp-ready patrol dispatch memo for immediate broadcast
    to mobile constables and towing units.
    """
    if route_df.empty:
        return "No active targets to dispatch."
        
    total_km = route_df['cumulative_distance_km'].iloc[-1] if not route_df.empty else 0.0
    total_time = route_df['cumulative_time_min'].iloc[-1] if not route_df.empty else 0.0
    multi_url = build_google_maps_multiwaypoint_url(route_df)

    lines = [
        "🚨 *BTP TACTICAL ENFORCEMENT DISPATCH MANIFEST*",
        f"📅 *Deployment Window:* {shift_name}",
        f"📍 *Patrol Origin:* {start_name}",
        f"🛣️ *Total Circuit Distance:* {total_km:.1f} km (est. transit {total_time:.0f} mins)",
        f"🎯 *Sequential Target Waypoints ({len(route_df)} Stops):*",
        ""
    ]

    for _, row in route_df.iterrows():
        order = row['stop_order']
        name = row['junction_name']
        tier = row['roi_class']
        act = row['tactical_action']
        pce = row['pce_impact']
        dist = row['leg_distance_km']
        nav = row['google_maps_nav_link']
        
        lines.append(f"*{order}. {name}* ({dist} km from prev)")
        lines.append(f"   • Tier: {tier} | Impact: PCE {pce}")
        lines.append(f"   • Directive: {act}")
        lines.append(f"   • Pin: {nav}")
        lines.append("")

    lines.append(f"🔗 *Tap for Full Turn-by-Turn Circuit Route:*")
    lines.append(multi_url)
    lines.append("")
    lines.append("⚠️ *Directive:* Maintain zero-tolerance clearance on arterial lanes. Log vehicle numbers into e-challan terminal.")

    return "\n".join(lines)
