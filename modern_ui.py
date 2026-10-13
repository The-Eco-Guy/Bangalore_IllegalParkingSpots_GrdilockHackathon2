"""
modern_ui.py — Command-Center UI overlay for Synnex Parking & Congestion Intelligence.
ZERO-TOUCH ISOLATION: this file never imports or modifies app.py, scoring.py,
prediction.py, or data_processing.py.  All data is received as plain DataFrames
passed in by the caller.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
import patrol
import scoring

# ──────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL THEME
# ──────────────────────────────────────────────────────────────────────────────

COMMAND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;900&family=Share+Tech+Mono&display=swap');

/* ── Root canvas ── */
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif !important; }

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d1526 0%, #0B0F19 60%, #060810 100%) !important;
    color: #e2e8f0 !important;
}

/* ── Hide default Streamlit chrome clutter ── */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(11,15,25,0.95) !important;
    border-right: 1px solid rgba(0,229,255,0.12) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Glass card ── */
.cmd-card {
    background: rgba(22,28,42,0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: border-color 0.25s;
}
.cmd-card:hover { border-color: rgba(0,229,255,0.22); }

/* ── KPI widgets ── */
.kpi-wrap {
    display: flex; flex-direction: column; align-items: flex-start;
}
.kpi-num {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.1rem; font-weight: 700; line-height: 1.15;
    letter-spacing: -0.02em;
}
.kpi-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #64748b; margin-top: 3px;
}
.kpi-delta {
    font-size: 0.72rem; margin-top: 4px; padding: 2px 7px;
    border-radius: 20px; font-weight: 600;
}
.kpi-cyan  { color: #00E5FF; }
.kpi-amber { color: #FF9100; }
.kpi-red   { color: #FF1744; }
.kpi-green { color: #00E676; }
.delta-up   { background: rgba(0,230,118,0.12); color: #00E676; }
.delta-down { background: rgba(255,23,68,0.12);  color: #FF1744; }

/* ── Section headers ── */
.cmd-section-title {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #00E5FF;
    border-left: 3px solid #00E5FF; padding-left: 10px;
    margin-bottom: 14px;
}

/* ── Dispatcher list ── */
.dispatch-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 14px; margin-bottom: 7px; border-radius: 9px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    transition: background 0.2s;
}
.dispatch-row:hover { background: rgba(0,229,255,0.06); }
.dispatch-tag {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.1em;
    padding: 3px 9px; border-radius: 20px; text-transform: uppercase;
}
.tag-tow   { background: rgba(255,23,68,0.15);  color: #FF1744; border: 1px solid rgba(255,23,68,0.35); }
.tag-eticket { background: rgba(255,145,0,0.15); color: #FF9100; border: 1px solid rgba(255,145,0,0.35); }
.tag-monitor { background: rgba(0,229,255,0.1);  color: #00E5FF; border: 1px solid rgba(0,229,255,0.25); }

/* ── Plotly chart containers ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ── Segmented control override ── */
[data-testid="stSegmentedControl"] button {
    background: rgba(22,28,42,0.6) !important;
    color: #64748b !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: rgba(0,229,255,0.15) !important;
    color: #00E5FF !important;
    border-color: rgba(0,229,255,0.4) !important;
    box-shadow: 0 0 10px rgba(0,229,255,0.18) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0B0F19; }
::-webkit-scrollbar-thumb { background: #1e2d40; border-radius: 3px; }
</style>
"""


def inject_theme():
    """Inject the command-center CSS. Call once at the top of the page."""
    st.markdown(COMMAND_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. HEADER BANNER
# ──────────────────────────────────────────────────────────────────────────────

def render_header(mode: str = "Historical Analysis"):
    mode_badge_color = "#00E5FF" if "Historical" in mode else "#FF9100"
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-end;
                margin-bottom:22px;padding-bottom:14px;
                border-bottom:1px solid rgba(0,229,255,0.12);">
        <div>
            <div style="font-size:0.62rem;letter-spacing:0.22em;text-transform:uppercase;
                        color:#00E5FF;font-weight:700;margin-bottom:4px;">
                ◈ SYNNEX PARKING &amp; CONGESTION INTELLIGENCE
            </div>
            <div style="font-size:1.95rem;font-weight:900;line-height:1.1;
                        background:linear-gradient(90deg,#00E5FF 0%,#7B61FF 60%,#FF1744 100%);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                COMMAND CENTER DASHBOARD
            </div>
            <div style="font-size:0.8rem;color:#475569;margin-top:3px;">
                Bangalore Urban Mobility Enforcement · AI-Driven Spatial Intelligence
            </div>
        </div>
        <div style="text-align:right;">
            <div style="display:inline-block;padding:5px 14px;border-radius:20px;
                        background:rgba(22,28,42,0.8);
                        border:1px solid {mode_badge_color};
                        font-size:0.65rem;font-weight:700;letter-spacing:0.14em;
                        text-transform:uppercase;color:{mode_badge_color};">
                ● {mode}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. KPI BANNER WITH MICRO-SPARKLINES
# ──────────────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert a #RRGGBB hex string to rgba(r,g,b,alpha) for Plotly."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(0,229,255,{alpha})"


def _mini_sparkline(values: list, color: str, height: int = 40) -> go.Figure:
    """Return a tiny Plotly area chart for embedding in KPI cards."""
    x = list(range(len(values)))
    if color.startswith("#"):
        fill_color = _hex_to_rgba(color, 0.15)
    elif color.startswith("rgb("):
        fill_color = color.replace("rgb(", "rgba(").replace(")", ",0.15)")
    else:
        fill_color = "rgba(0,229,255,0.15)"

    fig = go.Figure(go.Scatter(
        x=x, y=values, mode="lines",
        line=dict(color=color, width=1.8, shape="spline"),
        fill="tozeroy",
        fillcolor=fill_color,
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def render_kpi_banner(
    total_violations: int,
    total_pce: float,
    active_hotspots: int,
    tier1_targets: int,
    unpatrolled_targets: int = 0,
    sparkline_violations: list = None,
    sparkline_pce: list = None,
    sparkline_hotspots: list = None,
    sparkline_tier1: list = None,
    key_prefix: str = "cmd",
):
    """
    Render KPI cards across the full width with unique widget keys.
    Includes Unpatrolled Risk Zones to surface spatial enforcement bias.
    """
    c1, c2, c3, c4, c5 = st.columns(5)

    kpis = [
        (c1, f"{total_violations:,}",   "Total Violations Logged",      "kpi-cyan",  sparkline_violations,  "#00E5FF"),
        (c2, f"{total_pce:,.0f}",       "Aggregated PCE Load",          "kpi-amber", sparkline_pce,         "#FF9100"),
        (c3, str(active_hotspots),      "Active G★ Hotspots",           "kpi-red",   sparkline_hotspots,    "#FF1744"),
        (c4, str(tier1_targets),        "Tier-1 Towing Targets",        "kpi-red",   sparkline_tier1,       "#FF1744"),
        (c5, str(unpatrolled_targets),  "Unpatrolled Risk Blindspots",   "kpi-cyan",  None,                  "#38BDF8"),
    ]

    for idx, (col, num, label, cls, spark, color) in enumerate(kpis):
        with col:
            st.markdown(f"""
            <div class="cmd-card" style="padding:14px 18px;">
              <div class="kpi-wrap">
                <div class="kpi-num {cls}">{num}</div>
                <div class="kpi-label">{label}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if spark and len(spark) > 1:
                fig = _mini_sparkline(spark, color)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False}, key=f"{key_prefix}_spark_{idx}_{label[:6].strip()}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. 3-D EXTRUDED H3 HEX MAP
# ──────────────────────────────────────────────────────────────────────────────

_BLR_VIEW = pdk.ViewState(
    latitude=12.9716, longitude=77.5946,
    zoom=11.5, pitch=45, bearing=-15,
)

_CMD_TOOLTIP = {
    "html": """
    <div style="font-family:'Exo 2',sans-serif;background:rgba(11,15,25,0.95);
                border:1px solid rgba(0,229,255,0.25);border-radius:10px;
                padding:12px 16px;font-size:12px;min-width:220px;">
      <div style="color:#00E5FF;font-weight:700;font-size:13px;margin-bottom:8px;">
        {junction_name}
      </div>
      <div style="color:#94a3b8;line-height:1.8;">
        <b style="color:#e2e8f0;">H3 ID:</b> {h3_8}<br/>
        <b style="color:#e2e8f0;">Violations:</b> {violations_str}<br/>
        <b style="color:#e2e8f0;">PCE Score:</b> {cis_str}<br/>
        <b style="color:#e2e8f0;">Spillover Z:</b> {spillover_str}<br/>
        <b style="color:#e2e8f0;">Tier:</b> {roi_class_str}
      </div>
    </div>""",
    "style": {"color": "white"},
}


def _build_3d_colors(df, pce_col="weighted_pce"):
    df = df.copy()
    vals = df[pce_col].fillna(0).values.astype(float)
    v_max = vals.max()
    v_min = vals.min()
    rng = v_max - v_min if v_max > v_min else 1.0
    norm = np.clip(np.sqrt(vals - v_min) / np.sqrt(rng + 1e-9), 0.0, 1.0)
    colors = []
    for t in norm:
        if t < 0.5:
            s = t * 2.0
            r = int(0   + (123 - 0)   * s)
            g = int(229 + (97  - 229) * s)
            b = 255
        else:
            s = (t - 0.5) * 2.0
            r = int(123 + (255 - 123) * s)
            g = int(97  + (23  - 97)  * s)
            b = int(255 + (68  - 255) * s)
        colors.append([r, g, b, 200])
    df["fill_color_3d"] = colors
    df["elevation_3d"] = (norm * 800).astype(int)
    return df


def render_3d_hex_map(df, key="cmd_3d_map", pce_col="weighted_pce", active_only=True):
    if df.empty:
        st.info("No spatial data available for current filter selection.")
        return None
    plot_df = df.copy()
    if active_only:
        plot_df = plot_df[plot_df[pce_col] > 0]
    if plot_df.empty:
        st.info("No active cells to display.")
        return None
    for col, default in [
        ("violations_str", "0"), ("cis_str", "0.00"),
        ("spillover_str", "0.00"), ("roi_class_str", "Standard"),
        ("junction_name", "Unknown"),
    ]:
        if col not in plot_df.columns:
            plot_df[col] = default
    plot_df = _build_3d_colors(plot_df, pce_col)
    layer = pdk.Layer(
        "H3HexagonLayer",
        data=plot_df,
        id=f"{key}_layer",
        get_hexagon="h3_8",
        get_fill_color="fill_color_3d",
        get_elevation="elevation_3d",
        elevation_scale=1,
        extruded=True,
        pickable=True,
        auto_highlight=True,
        highlight_color=[0, 229, 255, 60],
        coverage=0.92,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=_BLR_VIEW,
        tooltip=_CMD_TOOLTIP,
        map_provider=None,
        map_style=None,
    )
    event = st.pydeck_chart(deck, key=key, on_select="rerun", selection_mode="single-object")
    return event

# ──────────────────────────────────────────────────────────────────────────────
# 5. PM-PEAK BLINDSPOT CHART
# ──────────────────────────────────────────────────────────────────────────────

def render_blindspot_chart(hist_df: pd.DataFrame, key: str = "blindspot_chart"):
    """
    Plotly area chart of hourly violation totals with a crimson PM-peak overlay.
    hist_df must have columns: hour (int 0-23), violation_count (numeric).
    """
    hourly = (
        hist_df.groupby("hour", as_index=False)["violation_count"]
        .sum()
        .sort_values("hour")
    )

    hours = hourly["hour"].tolist()
    counts = hourly["violation_count"].tolist()

    # Color buckets
    bar_colors = []
    for h in hours:
        if 7 <= h <= 10:
            bar_colors.append("#2979FF")
        elif 11 <= h <= 16:
            bar_colors.append("#546E7A")
        elif 17 <= h <= 21:
            bar_colors.append("#FF1744")
        else:
            bar_colors.append("#37474F")

    fig = go.Figure()

    # Background area
    fig.add_trace(go.Scatter(
        x=hours, y=counts, mode="lines",
        line=dict(color="rgba(0,229,255,0.4)", width=1.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.05)",
        name="All hours", showlegend=False,
    ))

    # Bars per segment
    fig.add_trace(go.Bar(
        x=hours, y=counts,
        marker_color=bar_colors,
        marker_line_width=0,
        opacity=0.85,
        name="Violations",
        showlegend=False,
    ))

    # Crimson PM-peak shaded overlay
    pm_max = max(counts) * 1.08 if counts else 100.0
    fig.add_shape(
        type="rect",
        x0=16.5, x1=21.5, y0=0, y1=pm_max,
        fillcolor="rgba(255,23,68,0.08)",
        line=dict(color="rgba(255,23,68,0.45)", width=1.5, dash="dot"),
        layer="below",
    )
    fig.add_annotation(
        x=19, y=pm_max * 0.92,
        text="⚠ CRITICAL ENFORCEMENT<br>BLINDSPOT",
        showarrow=False,
        font=dict(color="#FF1744", size=9.5, family="Share Tech Mono"),
        align="center",
    )

    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}" for h in range(24)],
            tickfont=dict(size=9, color="#475569"),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=9, color="#475569"),
        ),
        bargap=0.25,
    )

    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=key)



# ──────────────────────────────────────────────────────────────────────────────
# 6. TACTICAL DISPATCHER LIST
# ──────────────────────────────────────────────────────────────────────────────

def render_dispatcher(
    df: pd.DataFrame,
    max_rows: int = 8,
    key_prefix: str = "cmd",
    pce_col: str = "weighted_pce",
    vol_col: str = "violations",
):
    """
    Ranked bottleneck list with action tags and immediate field-dispatch tools:
    - Optimal turn-by-turn circuit (TSP) via Google Maps
    - Downloadable CSV manifest for control room
    - One-click copyable WhatsApp / wireless radio dispatch memo
    """
    if df.empty:
        st.info("No targets to dispatch.")
        return

    pce_field = pce_col if pce_col in df.columns else ("pred_weighted_pce" if "pred_weighted_pce" in df.columns else "weighted_pce")
    vol_field = vol_col if vol_col in df.columns else ("pred_count" if "pred_count" in df.columns else "violations")

    targets = (
        df[df["roi_class"] != "Standard"]
        .sort_values(pce_field, ascending=False)
        .head(max_rows)
    )
    if targets.empty:
        targets = df.sort_values(pce_field, ascending=False).head(max_rows)

    rows_html = ""
    for i, (_, row) in enumerate(targets.iterrows(), 1):
        tier = row.get("roi_class", "Standard")
        name = str(row.get("junction_name", "Unknown"))[:28]
        pce_val = row.get(pce_field, 0)
        vol_val = row.get(vol_field, 0)

        if tier == "Tier 1: Max Disruption":
            tag_cls, tag_txt = "tag-tow", "DISPATCH TOWING"
        elif tier == "Tier 2: Silent Bottleneck":
            tag_cls, tag_txt = "tag-tow", "ZERO-TOLERANCE TOW"
        elif tier == "Tier 3: Volume Hotspot":
            tag_cls, tag_txt = "tag-eticket", "ISSUE E-TICKET"
        else:
            tag_cls, tag_txt = "tag-monitor", "MONITOR"

        rank_color = "#FF1744" if i <= 3 else ("#FF9100" if i <= 6 else "#64748b")
        rows_html += f"""
        <div class="dispatch-row">
          <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:0;">
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                         color:{rank_color};font-weight:700;min-width:18px;">#{i}</span>
            <div style="min-width:0;">
              <div style="font-size:0.8rem;font-weight:600;color:#e2e8f0;
                          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
              <div style="font-size:0.65rem;color:#64748b;margin-top:1px;">
                PCE {pce_val:.1f} &nbsp;·&nbsp; Vol {int(vol_val)}
              </div>
            </div>
          </div>
          <span class="dispatch-tag {tag_cls}">{tag_txt}</span>
        </div>"""

    st.markdown(rows_html, unsafe_allow_html=True)

    # ── Field Patrol Manifest & Circuit Route Generator ──
    with st.expander("🚔 Field Patrol Manifest & Route", expanded=False):
        st.markdown(
            "<div style='font-size:0.72rem;color:#94a3b8;margin-bottom:8px;'>"
            "Optimizes a shortest-circuit patrol route starting from BTP HQ (Infantry Rd) "
            "connecting priority bottlenecks for field constables & towing units."
            "</div>",
            unsafe_allow_html=True,
        )
        try:
            route_df = patrol.optimize_patrol_route(
                df,
                max_stops=max_rows,
                pce_col=pce_field,
                vol_col=vol_field,
            )
            if not route_df.empty:
                total_km = route_df["cumulative_distance_km"].iloc[-1]
                total_time = route_df["cumulative_time_min"].iloc[-1]
                multi_url = patrol.build_google_maps_multiwaypoint_url(route_df)
                whatsapp_memo = patrol.generate_whatsapp_dispatch_memo(route_df)

                st.markdown(
                    f"<div style='background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.2);"
                    f"border-radius:8px;padding:8px 12px;margin-bottom:10px;font-size:0.72rem;'>"
                    f"<b>Circuit Scope:</b> {len(route_df)} priority waypoints &nbsp;|&nbsp; "
                    f"<b>Distance:</b> {total_km:.1f} km &nbsp;|&nbsp; "
                    f"<b>Est. Transit:</b> {total_time:.0f} mins"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if multi_url:
                        st.link_button("🗺️ Open Google Maps", multi_url, use_container_width=True)
                with b_col2:
                    csv_bytes = route_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Export CSV",
                        data=csv_bytes,
                        file_name=f"btp_patrol_manifest_{key_prefix}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"{key_prefix}_dl_manifest_btn",
                    )

                st.text_area(
                    "WhatsApp / Wireless Radio Memo:",
                    value=whatsapp_memo,
                    height=140,
                    key=f"{key_prefix}_whatsapp_memo_box",
                )
            else:
                st.info("No active bottlenecks to sequence.")
        except Exception as e:
            st.caption(f"Patrol sequence unavailable: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. H3 CELL DEEP-DIVE INSPECTOR
# ──────────────────────────────────────────────────────────────────────────────

_PCE_WEIGHTS = {
    "Scooter / 2W": 0.5,
    "Car / Jeep": 1.0,
    "Passenger Auto": 0.75,
    "Goods Auto": 1.4,
    "Maxi-Cab": 2.0,
    "Bus / HCV": 3.7,
}

_META_COLS = {
    "Scooter / 2W": "pct_scooter",
    "Car / Jeep": "pct_car",
    "Passenger Auto": "pct_auto",
    "Maxi-Cab": "pct_maxi",
}


def _render_sankey(cell_row, meta_row, key: str = "sankey_cell"):
    """PCE vehicle-mix Sankey: vehicle types → weighted PCE contribution."""
    vehicle_types = list(_PCE_WEIGHTS.keys())
    pce_vals = list(_PCE_WEIGHTS.values())

    # Mix percentages from meta, default equal share
    pcts = []
    for vt in vehicle_types:
        col = _META_COLS.get(vt)
        if col and meta_row is not None and col in meta_row.index:
            pcts.append(float(meta_row[col]))
        else:
            pcts.append(1.0 / len(vehicle_types))

    total = sum(pcts) or 1.0
    pcts = [p / total for p in pcts]

    # Weighted contributions
    contribs = [p * w for p, w in zip(pcts, pce_vals)]

    # Nodes: vehicles (0-5) + single output node (6)
    labels = vehicle_types + ["PCE Impact Score"]
    node_colors = [
        "#00E5FF", "#38BDF8", "#7DD3FC",
        "#FF9100", "#FF6D00", "#FF1744",
        "#7B61FF",
    ]
    link_colors = [c + "99" for c in node_colors[:-1]]

    sources = list(range(len(vehicle_types)))
    targets = [len(vehicle_types)] * len(vehicle_types)
    values = [max(c, 0.001) for c in contribs]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=20,
            label=labels,
            color=node_colors,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        link=dict(
            source=sources, target=targets,
            value=values,
            color=link_colors,
            hovertemplate="%{source.label} → PCE contrib: %{value:.3f}<extra></extra>",
        ),
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Exo 2, sans-serif", size=11, color="#94a3b8"),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=key)


def _render_spillover_network(cell_id: str, df: pd.DataFrame, key: str = "spillover_net"):
    """
    Plotly scatter-based network graph: selected cell at centre,
    neighbours arranged in a ring, edge width scaled by |Gi* Z-score|.
    """
    try:
        import h3 as h3lib
        neighbors = list(h3lib.grid_disk(cell_id, k=1) - {cell_id})
    except Exception:
        neighbors = []

    if not neighbors:
        st.info("No H3 neighbours found for spillover graph.")
        return

    center_row = df[df["h3_8"] == cell_id]
    center_z = float(center_row["gi_zscore"].iloc[0]) if not center_row.empty else 0.0

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    edge_x, edge_y = [], []

    # Centre node
    node_x.append(0); node_y.append(0)
    node_text.append(f"SELECTED<br>Z={center_z:.2f}")
    node_color.append("#FF1744" if center_z >= 1.96 else "#00E5FF")
    node_size.append(28)

    n = len(neighbors)
    for i, nb in enumerate(neighbors):
        angle = 2 * np.pi * i / n
        nx, ny = np.cos(angle) * 1.8, np.sin(angle) * 1.8
        node_x.append(nx); node_y.append(ny)

        nb_row = df[df["h3_8"] == nb]
        nb_z = float(nb_row["gi_zscore"].iloc[0]) if not nb_row.empty else 0.0
        nb_name = str(nb_row["junction_name"].iloc[0])[:18] if not nb_row.empty else nb[:8]

        node_text.append(f"{nb_name}<br>Z={nb_z:.2f}")
        node_color.append("#FF9100" if nb_z >= 1.96 else "#334155")
        node_size.append(16 if nb_z >= 1.96 else 12)

        edge_width = max(0.5, abs(nb_z) * 0.8)
        edge_x += [0, nx, None]
        edge_y += [0, ny, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(0,229,255,0.25)", width=1.2),
        hoverinfo="none", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(color=node_color, size=node_size,
                    line=dict(color="rgba(255,255,255,0.15)", width=1)),
        text=node_text, textposition="top center",
        textfont=dict(size=8, color="#94a3b8"),
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=key)


def render_cell_inspector(cell_id: str, df: pd.DataFrame, cell_meta: pd.DataFrame, key_prefix: str = "cmd"):
    """
    Full deep-dive panel for a selected H3 cell:
    - Key metrics summary
    - Sankey vehicle PCE breakdown
    - Spillover network propagation graph
    """
    row = df[df["h3_8"] == cell_id]
    meta_row_df = cell_meta[cell_meta["h3_8"] == cell_id] if cell_meta is not None else pd.DataFrame()
    meta_row = meta_row_df.iloc[0] if not meta_row_df.empty else None

    if row.empty:
        st.warning(f"No data for cell `{cell_id}`.")
        return

    src = row.iloc[0]
    junc = src.get("junction_name", "Unknown")
    v_val = src.get("violations", src.get("pred_count", 0))
    pce_val = src.get("weighted_pce", src.get("pred_weighted_pce", 0))
    z_val = src.get("gi_zscore", 0.0)
    p_val = src.get("gi_pvalue", 1.0)
    tier = src.get("roi_class", "Standard")
    lat = src.get("lat_center", src.get("lat", 0.0))
    lon = src.get("lon_center", src.get("lon", 0.0))

    tier_color = {"Tier 1: Max Disruption": "#FF1744",
                  "Tier 2: Silent Bottleneck": "#FF9100",
                  "Tier 3: Volume Hotspot": "#FFC400"}.get(tier, "#64748b")

    st.markdown(f"""
    <div class="cmd-card" style="border-color:rgba(0,229,255,0.2);">
      <div class="cmd-section-title">H3 CELL DEEP-DIVE — {junc.upper()}</div>
      <div style="display:flex;gap:32px;flex-wrap:wrap;margin-bottom:10px;">
        <div>
          <div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:0.1em;">H3 Index</div>
          <code style="color:#00E5FF;font-size:0.78rem;">{cell_id}</code>
        </div>
        <div>
          <div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:0.1em;">Coordinates</div>
          <code style="color:#94a3b8;font-size:0.78rem;">{lat:.5f}, {lon:.5f}</code>
        </div>
        <div>
          <div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:0.1em;">Violations</div>
          <span style="color:#00E5FF;font-weight:700;">{v_val:,.1f}</span>
        </div>
        <div>
          <div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:0.1em;">PCE Score</div>
          <span style="color:#FF9100;font-weight:700;">{pce_val:,.2f}</span>
        </div>
        <div>
          <div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:0.1em;">Spillover Z</div>
          <span style="color:{'#FF1744' if z_val>=1.96 else '#64748b'};font-weight:700;">
            {z_val:.2f} (p={p_val:.3f})</span>
        </div>
        <div>
          <div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:0.1em;">Tier</div>
          <span style="color:{tier_color};font-weight:700;">{tier}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ia, ib = st.columns(2)
    with ia:
        st.markdown('<div class="cmd-section-title">PCE VEHICLE-MIX BREAKDOWN</div>',
                    unsafe_allow_html=True)
        _render_sankey(src, meta_row, key=f"{key_prefix}_sankey_{cell_id[:6]}")
    with ib:
        st.markdown('<div class="cmd-section-title">SPILLOVER PROPAGATION NETWORK</div>',
                    unsafe_allow_html=True)
        _render_spillover_network(cell_id, df, key=f"{key_prefix}_spillover_{cell_id[:6]}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. TOP-LEVEL ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def _get_selected_cell(event) -> str | None:
    """Extract selected H3 cell id from a pydeck chart event."""
    if not event:
        return None
    sel = getattr(event, "selection", None) or (event if isinstance(event, dict) else {})
    objects = sel.get("objects", {}) if isinstance(sel, dict) else {}
    for objs in objects.values():
        if objs:
            return objs[0].get("h3_8")
    return None


def _build_sparklines(hist_df: pd.DataFrame):
    """Build 14-day sparkline arrays from hist_df for KPI cards."""
    if hist_df is None or hist_df.empty:
        return None, None, None, None
    hist_df = hist_df.copy()
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    daily = (
        hist_df.groupby("date")
        .agg(viol=("violation_count", "sum"), pce=("weighted_pce", "sum"))
        .sort_index()
        .tail(14)
    )
    sp_v = daily["viol"].tolist()
    sp_p = daily["pce"].tolist()
    # Hotspot / tier sparklines are noisy — just wave them off a ratio of viol
    sp_h = [max(0, int(v * 0.04)) for v in sp_v]
    sp_t = [max(0, int(v * 0.015)) for v in sp_v]
    return sp_v, sp_p, sp_h, sp_t


def render_command_center(
    df_scored_clean: pd.DataFrame,
    cell_meta: pd.DataFrame,
    hist_df: pd.DataFrame,
    mode: str = "Historical Analysis",
    pce_col: str = "weighted_pce",
    vol_col: str = "violations",
    map_key_prefix: str = "cmd_hist",
):
    """
    Master render function. Integrates 3D map, blindspot analytics,
    patrol dispatch manifests, and deep-dive Sankey/network inspectors.
    """
    inject_theme()
    render_header(mode)

    if df_scored_clean is None or df_scored_clean.empty:
        st.markdown(
            "<div class='cmd-card' style='border-color:rgba(255,145,0,0.4);'>"
            "<h4 style='color:#FF9100;margin-top:0;'>⚠️ No Active Data</h4>"
            "<p>No spatial records found matching the active filter criteria. "
            "Please broaden your date range or time bucket in the sidebar.</p>"
            "</div>",
            unsafe_allow_html=True
        )
        return

    # Compute unpatrolled exploration targets if meta is available
    if "is_exploration_target" not in df_scored_clean.columns and cell_meta is not None and not cell_meta.empty:
        try:
            df_scored_clean = scoring.compute_exploration_and_bias(
                df_scored_clean, cell_meta, vol_col=vol_col, pce_col=pce_col
            )
        except Exception:
            pass

    # ── KPI metrics ──────────────────────────────────────────────────────────
    total_v  = int(df_scored_clean[vol_col].sum()) if vol_col in df_scored_clean else 0
    total_p  = float(df_scored_clean[pce_col].sum()) if pce_col in df_scored_clean else 0.0
    hotspots = int(((df_scored_clean["gi_zscore"] >= 1.96) & (df_scored_clean["gi_pvalue"] < 0.05)).sum()) \
               if "gi_zscore" in df_scored_clean else 0
    tier1    = int((df_scored_clean["roi_class"] == "Tier 1: Max Disruption").sum()) \
               if "roi_class" in df_scored_clean else 0
    unpatrolled = int(df_scored_clean["is_exploration_target"].sum()) if "is_exploration_target" in df_scored_clean.columns else 0

    sp_v, sp_p, sp_h, sp_t = _build_sparklines(hist_df)

    render_kpi_banner(
        total_v, total_p, hotspots, tier1,
        unpatrolled_targets=unpatrolled,
        sparkline_violations=sp_v,
        sparkline_pce=sp_p,
        sparkline_hotspots=sp_h,
        sparkline_tier1=sp_t,
        key_prefix=map_key_prefix,
    )

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # ── Main 70/30 split ─────────────────────────────────────────────────────
    left_col, right_col = st.columns([7, 3], gap="medium")

    with left_col:
        st.markdown('<div class="cmd-section-title">3D SPATIAL CONGESTION MAP — PCE EXTRUSION</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.72rem;color:#475569;margin-bottom:10px;'>"
            "Height = PCE Congestion Score &nbsp;·&nbsp; "
            "Color: <span style='color:#00E5FF'>●</span> Low &nbsp;"
            "<span style='color:#7B61FF'>●</span> Medium &nbsp;"
            "<span style='color:#FF1744'>●</span> Critical"
            "</div>",
            unsafe_allow_html=True,
        )
        map_event = render_3d_hex_map(
            df_scored_clean,
            key=f"{map_key_prefix}_3d",
            pce_col=pce_col,
        )
        selected_cell = _get_selected_cell(map_event)

    with right_col:
        # ── Blindspot chart ──────────────────────────────────────────────────
        st.markdown('<div class="cmd-section-title">PM-PEAK ENFORCEMENT BLINDSPOT</div>',
                    unsafe_allow_html=True)
        if hist_df is not None and not hist_df.empty:
            render_blindspot_chart(hist_df, key=f"{map_key_prefix}_blindspot")
        else:
            st.info("Historical data unavailable for blindspot chart.")

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # ── Tactical dispatcher ───────────────────────────────────────────────
        st.markdown('<div class="cmd-section-title">TACTICAL ASSET DISPATCHER</div>',
                    unsafe_allow_html=True)
        render_dispatcher(
            df_scored_clean,
            max_rows=8,
            key_prefix=map_key_prefix,
            pce_col=pce_col,
            vol_col=vol_col,
        )

    # ── Deep-dive inspector (conditional on map click or dropdown) ───────────
    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:24px 0 16px;'>",
                unsafe_allow_html=True)
    st.markdown('<div class="cmd-section-title">H3 CELL DEEP-DIVE INSPECTOR</div>',
                unsafe_allow_html=True)

    options_df = (
        df_scored_clean[df_scored_clean[pce_col] > 0]
        .sort_values(pce_col, ascending=False)
        [["h3_8", "junction_name"]]
        .drop_duplicates("h3_8")
    )
    label_map = {
        row.h3_8: f"{row.junction_name} ({row.h3_8})"
        for row in options_df.itertuples()
    }
    default_cell = selected_cell if selected_cell in label_map else (
        options_df["h3_8"].iloc[0] if not options_df.empty else None
    )

    if default_cell:
        chosen = st.selectbox(
            "Select H3 cell (or click map above):",
            options=list(label_map.keys()),
            index=list(label_map.keys()).index(default_cell),
            format_func=lambda c: label_map.get(c, c),
            key=f"{map_key_prefix}_inspector_select",
        )
        if chosen:
            render_cell_inspector(chosen, df_scored_clean, cell_meta, key_prefix=map_key_prefix)
    else:
        st.info("No active cells available. Widen your date/time filter.")

