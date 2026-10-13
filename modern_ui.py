"""Shared presentation and interaction logic for the policing workspace."""
from dataclasses import dataclass
from html import escape
import math

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import patrol

COMMAND_CSS = """
<style>
.stApp { background: #f5f7fb; color: #172b4d; }
[data-testid="stSidebar"] { background: #edf1f7; border-right: 1px solid #d5dce7; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1600px; }
h1 { font-size: 2rem !important; letter-spacing: -.03em; }
h2 { font-size: 1.4rem !important; }
h3 { font-size: 1.15rem !important; }
[data-testid="stCaptionContainer"] { color: #52627a; font-size: .95rem; }
.st-key-location_details, .st-key-hist_priority_list, .st-key-pred_priority_list { background: #ffffff; border-radius: 12px; }
.cmd-card, .glass-card { background: #fff; border: 1px solid #d5dce7; border-radius: 12px; padding: 20px; margin-bottom: 12px; }
.kpi-val, .kpi-num { font-size: 2rem; font-weight: 700; color: #175cd3; }
.kpi-label { font-size: .95rem; color: #52627a; margin-top: 6px; }
.header-title { font-size: 2rem; font-weight: 700; color: #172b4d; }
.header-subtitle { color: #52627a; margin-bottom: 16px; }
.metric-red, .tag-tow { color: #b42318; }
.metric-orange, .tag-ticket { color: #975300; }
.metric-purple, .tag-watch { color: #175cd3; }
[data-testid="stButton"] button, [data-testid="stLinkButton"] a,
[data-testid="stDownloadButton"] button { min-height: 44px; }
button:focus-visible, a:focus-visible { outline: 3px solid #175cd3 !important; outline-offset: 3px; }
@media (max-width: 700px) {
 .block-container { padding: 1rem; }
 [data-testid="stHorizontalBlock"] { flex-direction: column; gap: 1rem; }
 [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
 h1 { font-size: 1.65rem !important; }
 .deck-tooltip { left: 8px !important; right: 8px !important; bottom: 8px !important; top: auto !important; max-width: none !important; }
}
</style>
"""

PRIORITIES = {
    'Tier 1: Max Disruption': ('High impact / high volume', [180, 35, 24, 155]),
    'Tier 2: Silent Bottleneck': ('High impact / lower volume', [197, 107, 0, 155]),
    'Tier 3: Volume Hotspot': ('High volume / lower impact', [23, 92, 211, 145]),
    'Standard': ('Routine review', [100, 116, 139, 70]),
}

@dataclass(frozen=True)
class ViewContext:
    source: str = 'Historical Records'
    start_date: object = None
    end_date: object = None
    bucket: str = 'All'
    hour: object = 'All'
    available_start: object = None
    available_end: object = None

    @property
    def forecast(self):
        return self.source == 'Forecast'

    @property
    def window(self):
        if self.hour != 'All':
            return f'{int(self.hour):02d}:00–{(int(self.hour) + 1):02d}:00 IST'
        return 'All hours (IST)' if self.bucket == 'All' else f'{self.bucket} IST'

    @property
    def dates(self):
        return str(self.start_date) if self.start_date == self.end_date else f'{self.start_date} to {self.end_date}'

    def export_fields(self):
        return dict(source=self.source, selected_start_date=str(self.start_date),
                    selected_end_date=str(self.end_date), selected_hours=self.window,
                    timezone='Asia/Kolkata', metric_basis='average per hour' if self.forecast else 'selected-period total')


def inject_theme():
    st.markdown(COMMAND_CSS, unsafe_allow_html=True)


def render_header(mode='Historical Records'):
    st.title('Bengaluru traffic workspace')
    st.caption(f'Parking intelligence · {mode} · Review locations and prepare patrol briefs')


def render_context(context):
    st.markdown(f'**{context.source}** · {context.dates} · {context.window}')
    st.caption(f'Available data: {context.available_start} to {context.available_end}. '
               + ('Cached forecast based on historical records; this is not a current incident feed.'
                  if context.forecast else 'Recorded enforcement activity; this is not a current incident feed.'))
    if context.bucket.startswith('Off-Peak') and context.hour == 'All':
        st.caption('Night selection includes 00:00–06:59 and 22:00–23:59 within each selected calendar date.')


def metric_labels(forecast=False):
    return ('Average predicted violations / hour', 'Average impact / hour') if forecast else ('Recorded violations', 'Congestion impact score')


def priority_label(row):
    if row.get('is_exploration_target', False):
        return 'Verification patrol'
    return PRIORITIES.get(row.get('roi_class', 'Standard'), PRIORITIES['Standard'])[0]


def priority_color(row):
    if row.get('is_exploration_target', False):
        return [109, 40, 217, 135]
    return PRIORITIES.get(row.get('roi_class', 'Standard'), PRIORITIES['Standard'])[1]


def valid_locations(df):
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    result = df.copy()
    for target, fallback in [('lat_center', 'lat'), ('lon_center', 'lon')]:
        if target not in result:
            result[target] = result.get(fallback, np.nan)
        result[target] = pd.to_numeric(result[target], errors='coerce')
    mask = result.lat_center.between(-90, 90) & result.lon_center.between(-180, 180)
    return result[mask].drop_duplicates('h3_8').copy()


def selected_cell_from_event(event):
    if event is None:
        return None
    selection = event.get('selection', {}) if isinstance(event, dict) else getattr(event, 'selection', {})
    for objects in selection.get('objects', {}).values():
        for obj in objects:
            if isinstance(obj, dict) and obj.get('h3_8'):
                return obj['h3_8']
    return None


def reconcile_selection(selected, available):
    return selected if selected in set(available) else None


def selection_key(prefix):
    return f'{prefix}_selected_cell'


def select_location(prefix, cell):
    """External selection clears the map widget's previous internal selection."""
    st.session_state[selection_key(prefix)] = cell
    epoch_key = f'{prefix}_map_epoch'
    st.session_state[epoch_key] = st.session_state.get(epoch_key, 0) + 1



def map_selection_callback(chart_key, prefix, available):
    cell = selected_cell_from_event(st.session_state.get(chart_key))
    if cell in available:
        st.session_state[selection_key(prefix)] = cell


def table_selection_callback(table_key, prefix, cells):
    event = st.session_state.get(table_key, {})
    rows = event.get('selection', {}).get('rows', [])
    if rows and 0 <= rows[0] < len(cells):
        select_location(prefix, cells[rows[0]])


def ranked_locations(df, pce_col='weighted_pce', vol_col='violations'):
    result = valid_locations(df)
    if result.empty:
        return result
    active = (result[pce_col].fillna(0) > 0) | (result[vol_col].fillna(0) > 0)
    active |= result.get('is_exploration_target', pd.Series(False, index=result.index)).fillna(False)
    return result[active].sort_values([pce_col, vol_col, 'h3_8'], ascending=[False, False, True])


def render_location_selector(df, cell_meta, prefix, pce_col='weighted_pce', vol_col='violations', forecast=False):
    ranked = ranked_locations(df, pce_col, vol_col)
    cells = ranked.h3_8.tolist() if not ranked.empty else []
    state_key = selection_key(prefix)
    selected = reconcile_selection(st.session_state.get(state_key), cells)
    st.session_state[state_key] = selected
    choice_key = f'{prefix}_location_choice'
    st.session_state[choice_key] = selected
    names = dict(zip(ranked.h3_8, ranked.junction_name)) if not ranked.empty else {}
    def choose():
        select_location(prefix, st.session_state[choice_key])
    st.selectbox('Find or select a location', [None] + cells,
                 format_func=lambda cell: 'Choose a location' if cell is None else f'{names[cell]} · {cell}',
                 key=choice_key, on_change=choose)
    selected = st.session_state[state_key]
    if selected:
        render_location_details(selected, df, cell_meta, pce_col, vol_col, forecast)
    else:
        st.info('Click a map cell, select a priority row, or search for a location to see its details.')
    return selected


def render_location_details(cell, df, cell_meta=None, pce_col='weighted_pce', vol_col='violations', forecast=False):
    rows = df[df.h3_8 == cell]
    if rows.empty:
        st.info('This location is no longer in the selected results.')
        return
    row = rows.iloc[0]
    with st.container(border=True, key="location_details"):
        st.subheader(str(row.get('junction_name', 'Unnamed location')))
        st.write(priority_label(row))
        st.write(patrol.assign_tactical_action(row))
        labels = metric_labels(forecast)
        c1, c2 = st.columns(2)
        c1.metric(labels[0], f'{row.get(vol_col, 0):,.2f}' if forecast else f'{row.get(vol_col, 0):,.0f}')
        c2.metric(labels[1], f'{row.get(pce_col, 0):,.2f}')
        locations = valid_locations(rows)
        if not locations.empty:
            point = locations.iloc[0]
            st.caption(f'Area centroid: {point.lat_center:.5f}, {point.lon_center:.5f}. Verify the precise street location on arrival.')
            st.link_button('Navigate to this area', f'https://www.google.com/maps/dir/?api=1&destination={point.lat_center:.5f},{point.lon_center:.5f}&travelmode=driving')
        if row.get('is_exploration_target', False):
            st.write(str(row.get('exploration_reason', 'Verify local conditions and recording coverage.')))
        with st.expander('Location metadata and technical statistics'):
            st.write(f'H3 cell: {cell}')
            st.write(f"Spatial spillover z-score: {row.get('gi_zscore', 0):.2f}; p-value: {row.get('gi_pvalue', 1):.4f}")
            st.caption('Priorities are relative to this selection. Impact is a modeled score, not measured delay or a verified obstruction.')
            if cell_meta is not None and not cell_meta.empty:
                meta = cell_meta[cell_meta.h3_8 == cell]
                if not meta.empty:
                    m = meta.iloc[0]
                    mixes = [('Cars / jeeps', 'pct_car'), ('Two-wheelers', 'pct_scooter'), ('Autos', 'pct_auto'),
                             ('Maxi-cabs', 'pct_maxi'), ('Wrong parking', 'pct_wrong_park'), ('No parking', 'pct_no_park'),
                             ('Main-road parking', 'pct_main_road'), ('Footpath parking', 'pct_footpath')]
                    st.dataframe(pd.DataFrame([{'Historical vehicle / violation mix': label, 'Share': f'{m.get(col, 0):.1%}'} for label, col in mixes]), hide_index=True, width='stretch')


def render_kpi_banner(total_violations, total_pce, tier1_targets, forecast=False):
    labels = metric_labels(forecast)
    cols = st.columns(3)
    vals = [f'{total_violations:,.2f}' if forecast else f'{total_violations:,.0f}', f'{total_pce:,.1f}', str(tier1_targets)]
    for col, val, label in zip(cols, vals, [*labels, 'High-impact / high-volume areas']):
        col.metric(label, val)


def _build_3d_colors(df, pce_col='weighted_pce'):
    """Legacy numeric helper retained for callers; Operations uses flat priority colors."""
    df = df.copy()
    values = df[pce_col].fillna(0).clip(lower=0)
    norm = np.sqrt((values - values.min()) / max(float(values.max() - values.min()), 1)) if len(values) else values
    df['fill_color_3d'] = [[23, 92, 211, 145] for _ in values]
    df['elevation_3d'] = (norm * 750).astype(int)
    return df


def render_3d_hex_map(df, key='cmd_map', pce_col='weighted_pce', prefix='hist', vol_col='violations', forecast=False):
    """Compatibility entry point for the selectable, flat Operations map."""
    plot_df = ranked_locations(df, pce_col, vol_col)
    if plot_df.empty:
        st.info('No mapped locations for the selected filters.')
        return
    plot_df['priority_color'] = plot_df.apply(priority_color, axis=1)
    plot_df['priority_label'] = plot_df.apply(priority_label, axis=1)
    plot_df['map_name'] = plot_df.junction_name.map(lambda value: escape(str(value)))
    plot_df['map_volume'] = plot_df[vol_col].map(lambda value: f'{value:,.2f}' if forecast else f'{value:,.0f}')
    plot_df['map_impact'] = plot_df[pce_col].map(lambda value: f'{value:,.2f}')
    selected = st.session_state.get(selection_key(prefix))
    layers = [pdk.Layer('H3HexagonLayer', id=f'{prefix}_locations', data=plot_df,
                        get_hexagon='h3_8', get_fill_color='priority_color', extruded=False,
                        stroked=True, get_line_color=[255, 255, 255, 180], line_width_min_pixels=1,
                        pickable=True, auto_highlight=True)]
    highlight = plot_df[plot_df.h3_8 == selected]
    if not highlight.empty:
        layers.append(pdk.Layer('H3HexagonLayer', id=f'{prefix}_selected', data=highlight,
                                get_hexagon='h3_8', filled=False, stroked=True,
                                get_line_color=[23, 43, 77, 255], line_width_min_pixels=4, pickable=False))
    labels = metric_labels(forecast)
    deck = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=12.9716, longitude=77.5946, zoom=11, pitch=0),
                    map_provider='carto', map_style=pdk.map_styles.LIGHT,
                    tooltip={'html': '<b>{map_name}</b><br>{priority_label}<br>' + labels[0] + ': {map_volume}<br>' + labels[1] + ': {map_impact}',
                             'style': {'backgroundColor': '#fff', 'color': '#172b4d'}})
    st.pydeck_chart(deck, key=key, height=470, selection_mode='single-object',
                    on_select=lambda: map_selection_callback(key, prefix, plot_df.h3_8.tolist()))
    st.caption('Map: © CARTO · © OpenStreetMap contributors. Hexagons represent areas, not exact incident positions.')


def render_action_list(df, max_rows=8, key_prefix='hist', pce_col='weighted_pce', vol_col='violations', forecast=False):
    targets = ranked_locations(df, pce_col, vol_col).head(max_rows)
    if targets.empty:
        st.info('No priority locations for this selection.')
        return
    def select(cell):
        select_location(key_prefix, cell)
    st.caption('Ranked by impact, then volume. Select a location to open its details.')
    with st.container(height=470, border=True, key=f'{key_prefix}_priority_list'):
        for rank, (_, row) in enumerate(targets.iterrows(), 1):
            selected = st.session_state.get(selection_key(key_prefix)) == row.h3_8
            st.button(f'{rank}. {row.junction_name}', key=f'{key_prefix}_priority_{row.h3_8}',
                      width='stretch', type='primary' if selected else 'secondary',
                      on_click=select, args=(row.h3_8,))
            volume = f'{row[vol_col]:,.2f} predicted / hour' if forecast else f'{row[vol_col]:,.0f} recorded'
            impact = f'{row[pce_col]:,.1f}' + (' / hour' if forecast else '')
            st.caption(f'{priority_label(row)} · {volume} · Impact {impact}')


def filter_hourly_data(df, context):
    filtered = df.copy()
    if context.start_date is not None:
        filtered = filtered[filtered.date >= context.start_date]
    if context.end_date is not None:
        filtered = filtered[filtered.date <= context.end_date]
    if context.hour != 'All':
        filtered = filtered[filtered.hour == int(context.hour)]
    elif context.bucket != 'All':
        from data_processing import get_hours_for_bucket
        filtered = filtered[filtered.hour.isin(get_hours_for_bucket(context.bucket))]
    return filtered


def hourly_profile(df, forecast=False):
    value_col = 'pred_count' if forecast else 'violation_count'
    if df is None or df.empty:
        return pd.DataFrame(columns=['hour', 'value'])
    # Forecast data is one selected date: sum the hourly estimates across cells.
    return df.groupby('hour')[value_col].sum().sort_index().rename('value').reset_index()


def forecast_sample_count(value, seed):
    """Deterministic stochastic rounding preserves fractional simulation expectations."""
    if not np.isfinite(value) or value <= 0:
        return 0
    base = math.floor(value)
    return base + int(np.random.default_rng(seed).random() < value - base)


def render_hourly_chart(hist_df, key='hourly_chart', forecast=False):
    hourly = hourly_profile(hist_df, forecast)
    if hourly.empty:
        st.info('No hourly records for this selection.')
        return
    fig = go.Figure(go.Bar(x=hourly.hour, y=hourly.value, marker_color='#175cd3',
                           hovertemplate='%{x:02d}:00 IST · %{y:,.2f}<extra></extra>'))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                      font=dict(color='#172b4d'), xaxis=dict(title='Hour (IST)', tickmode='array', tickvals=list(range(24)),
                      ticktext=[f'{h:02d}' for h in range(24)]),
                      yaxis=dict(title='Predicted violations / hour' if forecast else 'Recorded violations', gridcolor='#e4e9f1'))
    st.plotly_chart(fig, width='stretch', key=key, config={'displayModeBar': False})


def render_patrol_plan(df, context, prefix, pce_col='weighted_pce', vol_col='violations'):
    st.subheader('Prepare a patrol brief')
    st.caption('Review stops and origin before generating a plan. Verify conditions on arrival before deciding enforcement action.')
    ranked = ranked_locations(df, pce_col, vol_col)
    if ranked.empty:
        st.info('No eligible patrol locations in this selection.')
        return
    cells = ranked.h3_8.tolist()
    names = dict(zip(ranked.h3_8, ranked.junction_name))
    signature = (context.source, context.dates, context.window, tuple(cells))
    stops_key = f'{prefix}_patrol_stops'
    reviewed_key = f'{prefix}_reviewed_stops'
    if st.session_state.get(f'{prefix}_patrol_context') != signature:
        st.session_state[reviewed_key] = cells[:8]
        st.session_state[f'{prefix}_patrol_context'] = signature
    st.session_state[stops_key] = [cell for cell in st.session_state.get(reviewed_key, cells[:8]) if cell in cells]
    def save_stops():
        st.session_state[reviewed_key] = st.session_state[stops_key]
    stops = st.multiselect('Reviewed patrol stops (up to 8)', cells, key=stops_key,
                            format_func=lambda cell: f'{names[cell]} · {cell}', max_selections=8, on_change=save_stops)
    def origin_input(column, field, label, default, **kwargs):
        persistent_key = f'{prefix}_saved_origin_{field}'
        widget_key = f'{prefix}_origin_{field}'
        st.session_state[widget_key] = st.session_state.get(persistent_key, default)
        def save():
            st.session_state[persistent_key] = st.session_state[widget_key]
        if field == 'name':
            return column.text_input(label, key=widget_key, on_change=save)
        return column.number_input(label, key=widget_key, on_change=save, **kwargs)
    c1, c2, c3 = st.columns([2, 1, 1])
    origin_name = origin_input(c1, 'name', 'Origin name', 'Traffic Police HQ (Infantry Rd)')
    lat = origin_input(c2, 'lat', 'Origin latitude', patrol.BTP_HQ_LAT, min_value=-90.0, max_value=90.0, format='%.5f')
    lon = origin_input(c3, 'lon', 'Origin longitude', patrol.BTP_HQ_LON, min_value=-180.0, max_value=180.0, format='%.5f')
    fingerprint = (signature, tuple(stops), origin_name.strip(), lat, lon)
    if stops:
        reviewed = ranked.set_index('h3_8').loc[stops].reset_index()
        st.dataframe(reviewed[['junction_name', 'h3_8']].rename(columns={'junction_name': 'Reviewed location', 'h3_8': 'Area ID'}), hide_index=True, width='stretch')
    if st.button('Generate reviewed route', key=f'{prefix}_generate_route', type='primary', disabled=not stops or not origin_name.strip()):
        route = patrol.optimize_patrol_route(reviewed, start_lat=lat, start_lon=lon, start_name=origin_name.strip(),
                                             max_stops=len(stops), pce_col=pce_col, vol_col=vol_col, selected_only=True)
        for field, value in context.export_fields().items():
            route[field] = value
        route['optimal_window'] = context.window
        route['origin_name'], route['origin_latitude'], route['origin_longitude'] = origin_name.strip(), lat, lon
        route['distance_basis'] = 'straight-line estimate'
        route['transit_basis'] = 'approximate at 16 km/h; excludes time at stops'
        st.session_state[f'{prefix}_route'] = (fingerprint, route)
    saved = st.session_state.get(f'{prefix}_route')
    if saved is None or saved[0] != fingerprint:
        st.caption('Generate a route after reviewing the stops. Changing filters, stops, or origin requires a new route.')
        return
    route = saved[1]
    if route.empty:
        st.info('No valid route stops available.')
        return
    st.dataframe(route[['stop_order', 'junction_name', 'tactical_action', 'leg_distance_km']].rename(columns={
        'stop_order': 'Stop', 'junction_name': 'Location', 'tactical_action': 'Suggested verification', 'leg_distance_km': 'Straight-line leg (km)'}), hide_index=True, width='stretch')
    st.caption(f"{len(route)} stops · {route.cumulative_distance_km.iloc[-1]:.1f} km straight-line estimate · "
               f"{route.cumulative_time_min.iloc[-1]:.0f} min approximate transit; excludes time at stops. Google Maps supplies road navigation.")
    url = patrol.build_google_maps_multiwaypoint_url(route, lat, lon)
    st.link_button('Open reviewed route in Google Maps', url)
    st.download_button('Download patrol brief (CSV)', route.to_csv(index=False).encode('utf-8'),
                        file_name=f'patrol_brief_{prefix}_{context.start_date}.csv', mime='text/csv', key=f'{prefix}_brief_download', on_click='ignore')
    memo = patrol.generate_whatsapp_dispatch_memo(route, start_name=origin_name.strip(), shift_name=context.window,
                                                 start_lat=lat, start_lon=lon, context=context.export_fields())
    with st.expander('Copy patrol brief for radio or messaging'):
        st.text_area('Patrol brief', value=memo, height=300, key=f'{prefix}_brief_{hash(fingerprint)}')


def render_command_center(df_scored_clean, cell_meta, hist_df, mode='Historical Records', pce_col='weighted_pce',
                          vol_col='violations', map_key_prefix='hist', context=None):
    context = context or ViewContext(source='Forecast' if 'Forecast' in mode else 'Historical Records')
    prefix = 'pred' if context.forecast else 'hist'
    df = ranked_locations(df_scored_clean, pce_col, vol_col)
    st.session_state[selection_key(prefix)] = reconcile_selection(st.session_state.get(selection_key(prefix)), df.h3_8 if not df.empty else [])
    if df.empty:
        st.info('No active locations match these filters. Broaden the date or hour selection.')
        return
    render_kpi_banner(df[vol_col].sum(), df[pce_col].sum(), (df.roi_class == 'Tier 1: Max Disruption').sum(), context.forecast)
    left, right = st.columns([3, 2], gap='large')
    with left:
        st.subheader('Priority area map')
        st.caption('Red: high impact / high volume · Amber: high impact / lower volume · Blue: high volume / lower impact · Purple: verification patrol · Gray: routine review')
        render_3d_hex_map(df, key=f'{prefix}_operations_map_{context.dates}_{context.window}_{st.session_state.get(f"{prefix}_map_epoch", 0)}', pce_col=pce_col, vol_col=vol_col, prefix=prefix, forecast=context.forecast)
    with right:
        st.subheader('Priority locations')
        render_action_list(df, key_prefix=prefix, pce_col=pce_col, vol_col=vol_col, forecast=context.forecast)
    render_location_selector(df, cell_meta, prefix, pce_col, vol_col, context.forecast)
    render_patrol_plan(df, context, prefix, pce_col, vol_col)
    st.subheader('Hourly forecast' if context.forecast else 'Recorded violations by hour')
    st.caption(f'{context.dates} · {context.window}')
    render_hourly_chart(hist_df, key=f'{prefix}_operations_hourly', forecast=context.forecast)


def get_prediction_visibility_mask(df, selected_layer):
    if df.empty:
        return pd.Series(False, index=df.index)
    if selected_layer == "Predicted violations / hour":
        return df['pred_density_active'].fillna(False)
    if selected_layer == "Predicted impact / hour":
        return df['pred_congestion_active'].fillna(False)
    if selected_layer == "Predicted Combined Map":
        return df['pred_density_active'].fillna(False) | df['pred_congestion_active'].fillna(False)
    if selected_layer == "Predicted Spillover":
        return df['pred_spillover_significant'].fillna(False)
    if selected_layer == "Predicted Enforcement Plan":
        return df['pred_enforcement_visible'].fillna(False)
    if selected_layer == "Predicted Clusters (DBSCAN)":
        return df['pred_count'].fillna(0) > 0
    return pd.Series(True, index=df.index)

def build_prediction_download(df, selected_date, selected_hour, selected_bucket, selected_layer):
    export_df = df.copy()
    export_df['selected_forecast_date'] = str(selected_date)
    export_df['selected_forecast_hour'] = str(selected_hour)
    export_df['selected_time_bucket'] = str(selected_bucket)
    export_df['selected_prediction_mode'] = selected_layer
    export_df['source'] = 'Cached forecast'
    export_df['timezone'] = 'Asia/Kolkata'
    export_df['metric_basis'] = 'average per hour'
    export_df['priority_reason'] = export_df.apply(priority_label, axis=1)
    export_df['suggested_verification'] = export_df.apply(patrol.assign_tactical_action, axis=1)

    # Calculate mask
    mask = get_prediction_visibility_mask(export_df, selected_layer)
    export_df['visible_in_selected_layer'] = mask

    # Filter to visible rows
    export_df = export_df[mask].copy()

    # Add predicted_metric and predicted_value columns based on selected mode/layer
    if selected_layer == "Predicted violations / hour":
        export_df['predicted_metric'] = "Average predicted violations / hour"
        export_df['predicted_value'] = export_df['pred_count']
    elif selected_layer == "Predicted impact / hour":
        export_df['predicted_metric'] = "Average impact / hour"
        export_df['predicted_value'] = export_df['pred_weighted_pce']
    elif selected_layer == "Predicted Combined Map":
        export_df['predicted_metric'] = "Average impact / hour (combined map; count exported separately)"
        export_df['predicted_value'] = export_df['pred_weighted_pce']
    elif selected_layer == "Predicted Spillover":
        export_df['predicted_metric'] = "Spillover Z-Score"
        export_df['predicted_value'] = export_df['gi_zscore']
    elif selected_layer == "Predicted Enforcement Plan":
        export_df['predicted_metric'] = "Enforcement Classification"
        export_df['predicted_value'] = export_df['roi_class']
    elif selected_layer == "Predicted Clusters (DBSCAN)":
        export_df['predicted_metric'] = "Average predicted violations / hour"
        export_df['predicted_value'] = export_df['pred_count']
    else:
        export_df['predicted_metric'] = "Average predicted violations / hour"
        export_df['predicted_value'] = export_df['pred_count']

    export_columns = [
        'h3_8',
        'source', 'timezone', 'metric_basis', 'priority_reason', 'suggested_verification', 'visible_in_selected_layer',
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
        'is_exploration_target',
        'exploration_score',
        'exploration_reason',
    ]
    available_columns = [col for col in export_columns if col in export_df.columns]
    return export_df[available_columns].to_csv(index=False).encode('utf-8')
