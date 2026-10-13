"""
modern_ui.py — Command-Center UI for Bangalore Traffic Police Operations.
Focused on operational clarity: where the problems are, how severe they are, and what action to take.
"""

try:
    import streamlit as st
except ImportError:
    st = None

import pandas as pd
import numpy as np

try:
    import pydeck as pdk
except ImportError:
    pdk = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

import patrol

# ──────────────────────────────────────────────────────────────────────────────
# 1. CLEAN DARK COMMAND THEME
# ──────────────────────────────────────────────────────────────────────────────

COMMAND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: #090d16 !important;
    color: #e2e8f0 !important;
}

/* Chrome clean up */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Clean cards */
.cmd-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
}

/* KPI display */
.kpi-wrap {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}
.kpi-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-top: 6px;
}
.kpi-cyan  { color: #38bdf8; }
.kpi-amber { color: #fbbf24; }
.kpi-red   { color: #f87171; }

/* Section titles */
.cmd-section-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    border-left: 3px solid #38bdf8;
    padding-left: 8px;
    margin-bottom: 10px;
}

/* Action list rows */
.action-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    margin-bottom: 6px;
    border-radius: 8px;
    background: #162032;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.action-row:hover {
    border-color: rgba(56, 189, 248, 0.3);
}

.action-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 4px;
    text-transform: uppercase;
}
.tag-tow {
    background: rgba(239, 68, 68, 0.18);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.tag-ticket {
    background: rgba(245, 158, 11, 0.18);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.tag-watch {
    background: rgba(56, 189, 248, 0.12);
    color: #7dd3fc;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

/* Plotly canvas */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #090d16; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
"""


def inject_theme():
    """Inject clean command-center CSS."""
    if st is None:
        return
    st.markdown(COMMAND_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. HEADER BANNER
# ──────────────────────────────────────────────────────────────────────────────

def render_header(mode: str = "Historical Analysis"):
    if st is None:
        return
    mode_color = "#38bdf8" if "Historical" in mode else "#fbbf24"
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:18px; padding-bottom:12px;
                border-bottom:1px solid rgba(255,255,255,0.08);">
        <div>
            <div style="font-size:1.5rem; font-weight:800; color:#f8fafc; letter-spacing:-0.02em;">
                🚔 BANGALORE TRAFFIC COMMAND
            </div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:2px;">
                Urban Parking &amp; Congestion Enforcement Desk
            </div>
        </div>
        <div>
            <span style="display:inline-block; padding:4px 12px; border-radius:6px;
                         background:#162032; border:1px solid {mode_color};
                         font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                         font-weight:600; color:{mode_color};">
                ● {mode.upper()}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 3-KPI SUMMARY BANNER
# ──────────────────────────────────────────────────────────────────────────────

def render_kpi_banner(
    total_violations: int,
    total_pce: float,
    tier1_targets: int,
):
    """Render 3 essential KPI cards with zero clutter."""
    if st is None:
        return
    c1, c2, c3 = st.columns(3)

    kpis = [
        (c1, f"{total_violations:,}", "Total Violations", "kpi-cyan"),
        (c2, f"{total_pce:,.0f}", "Traffic Impact Score", "kpi-amber"),
        (c3, str(tier1_targets), "Priority Tow Targets", "kpi-red"),
    ]

    for col, val, label, cls in kpis:
        with col:
            st.markdown(f"""
            <div class="cmd-card">
              <div class="kpi-wrap">
                <div class="kpi-num {cls}">{val}</div>
                <div class="kpi-label">{label}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 3D HOTSPOT MAP
# ──────────────────────────────────────────────────────────────────────────────

_BLR_VIEW = pdk.ViewState(
    latitude=12.9716, longitude=77.5946,
    zoom=11.5, pitch=45, bearing=-15,
) if pdk is not None else None

_CMD_TOOLTIP = {
    "html": """
    <div style="font-family:'Inter',sans-serif; background:#0d1322;
                border:1px solid rgba(56,189,248,0.3); border-radius:8px;
                padding:10px 14px; font-size:12px; min-width:200px;">
      <div style="color:#38bdf8; font-weight:700; font-size:13px; margin-bottom:6px;">
        {junction_name}
      </div>
      <div style="color:#94a3b8; line-height:1.7;">
        <b>Violations:</b> <span style="color:#f8fafc;">{violations_str}</span><br/>
        <b>Impact Score:</b> <span style="color:#fbbf24;">{cis_str}</span><br/>
        <b>Action Tier:</b> <span style="color:#f87171;">{roi_class_str}</span>
      </div>
    </div>""",
    "style": {"color": "white"},
}


def _build_3d_colors(df: pd.DataFrame, pce_col: str = "weighted_pce") -> pd.DataFrame:
    df = df.copy()
    vals = df[pce_col].fillna(0).values.astype(float)
    v_max = vals.max()
    v_min = vals.min()
    rng = v_max - v_min if v_max > v_min else 1.0
    norm = np.clip(np.sqrt(np.maximum(vals - v_min, 0)) / np.sqrt(rng + 1e-9), 0.0, 1.0)

    colors = []
    for t in norm:
        if t < 0.5:
            s = t * 2.0
            r = int(56 + (147 - 56) * s)
            g = int(189 + (51 - 189) * s)
            b = int(248 + (234 - 248) * s)
        else:
            s = (t - 0.5) * 2.0
            r = int(147 + (239 - 147) * s)
            g = int(51 + (68 - 51) * s)
            b = int(234 + (68 - 234) * s)
        colors.append([r, g, b, 200])

    df["fill_color_3d"] = colors
    df["elevation_3d"] = (norm * 750).astype(int)
    return df


def render_3d_hex_map(df: pd.DataFrame, key: str = "cmd_3d_map", pce_col: str = "weighted_pce"):
    if st is None or pdk is None:
        return
    if df.empty:
        st.info("No spatial data available for current selection.")
        return
    plot_df = df[df[pce_col] > 0].copy() if pce_col in df.columns else df.copy()
    if plot_df.empty:
        st.info("No active bottleneck locations to display.")
        return

    for col, default in [
        ("violations_str", "0"),
        ("cis_str", "0.0"),
        ("roi_class_str", "Standard"),
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
        highlight_color=[56, 189, 248, 80],
        coverage=0.92,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=_BLR_VIEW,
        tooltip=_CMD_TOOLTIP,
        map_provider=None,
        map_style=None,
    )
    st.pydeck_chart(deck, key=key)


# ──────────────────────────────────────────────────────────────────────────────
# 5. PRIORITY ACTION LIST & PATROL DISPATCH
# ──────────────────────────────────────────────────────────────────────────────

def render_action_list(
    df: pd.DataFrame,
    max_rows: int = 8,
    key_prefix: str = "cmd",
    pce_col: str = "weighted_pce",
    vol_col: str = "violations",
):
    """
    Ranked priority list of problem locations with clear actionable tags (TOW / TICKET / WATCH)
    and one-click route dispatch tools.
    """
    if st is None:
        return
    if df.empty:
        st.info("No priority locations detected.")
        return

    pce_field = pce_col if pce_col in df.columns else ("pred_weighted_pce" if "pred_weighted_pce" in df.columns else "weighted_pce")
    vol_field = vol_col if vol_col in df.columns else ("pred_count" if "pred_count" in df.columns else "violations")

    # Focus on actionable tiers first
    targets = df[df["roi_class"] != "Standard"].sort_values(pce_field, ascending=False).head(max_rows)
    if targets.empty:
        targets = df.sort_values(pce_field, ascending=False).head(max_rows)

    rows_html = ""
    for i, (_, row) in enumerate(targets.iterrows(), 1):
        tier = row.get("roi_class", "Standard")
        name = str(row.get("junction_name", "Unknown"))[:26]
        pce_val = row.get(pce_field, 0)
        vol_val = row.get(vol_field, 0)

        if "Tier 1" in str(tier) or "Tier 2" in str(tier):
            tag_cls, tag_txt = "tag-tow", "TOW"
        elif "Tier 3" in str(tier):
            tag_cls, tag_txt = "tag-ticket", "TICKET"
        else:
            tag_cls, tag_txt = "tag-watch", "WATCH"

        rank_color = "#f87171" if i <= 3 else ("#fbbf24" if i <= 6 else "#94a3b8")
        rows_html += f"""
        <div class="action-row">
          <div style="display:flex; align-items:center; gap:10px; min-width:0;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                         color:{rank_color}; font-weight:700; min-width:20px;">#{i}</span>
            <div style="min-width:0;">
              <div style="font-size:0.82rem; font-weight:600; color:#f1f5f9;
                          white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{name}</div>
              <div style="font-size:0.7rem; color:#94a3b8;">
                Impact {pce_val:.1f} · {int(vol_val)} Violations
              </div>
            </div>
          </div>
          <span class="action-tag {tag_cls}">{tag_txt}</span>
        </div>"""

    st.markdown(rows_html, unsafe_allow_html=True)

    # ── Field Patrol Route & Dispatch Export ──
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
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
                f"<div style='font-size:0.75rem; color:#94a3b8; margin-bottom:8px;'>"
                f"<b>Circuit:</b> {len(route_df)} stops · <b>Distance:</b> {total_km:.1f} km · "
                f"<b>Est. Transit:</b> {total_time:.0f} mins"
                f"</div>",
                unsafe_allow_html=True,
            )

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if multi_url:
                    st.link_button("🗺️ Patrol Route", multi_url, use_container_width=True)
            with b_col2:
                csv_bytes = route_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Manifest",
                    data=csv_bytes,
                    file_name=f"btp_dispatch_manifest_{key_prefix}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"{key_prefix}_dl_manifest_btn",
                )

            with st.expander("📱 Dispatch Memo (WhatsApp / Radio)", expanded=False):
                st.text_area(
                    "Copy memo:",
                    value=whatsapp_memo,
                    height=130,
                    key=f"{key_prefix}_memo_area",
                    label_visibility="collapsed",
                )
    except Exception as e:
        st.caption(f"Patrol route unavailable: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. HOURLY VIOLATION PROFILE
# ──────────────────────────────────────────────────────────────────────────────

def render_hourly_chart(hist_df: pd.DataFrame, key: str = "hourly_chart"):
    """
    Clean 24-hour violation distribution with peak enforcement windows clearly highlighted.
    """
    if st is None or go is None:
        return
    if hist_df is None or hist_df.empty:
        st.info("Historical hourly data unavailable.")
        return

    hourly = (
        hist_df.groupby("hour", as_index=False)["violation_count"]
        .sum()
        .sort_values("hour")
    )
    hours = hourly["hour"].tolist()
    counts = hourly["violation_count"].tolist()

    # Highlight peak hours cleanly
    bar_colors = []
    for h in hours:
        if 17 <= h <= 21:      # PM Peak
            bar_colors.append("#ef4444")
        elif 8 <= h <= 11:     # AM Peak
            bar_colors.append("#38bdf8")
        elif 12 <= h <= 16:    # Midday
            bar_colors.append("#64748b")
        else:                  # Night
            bar_colors.append("#334155")

    fig = go.Figure()

    # Smooth guideline
    fig.add_trace(go.Scatter(
        x=hours, y=counts, mode="lines",
        line=dict(color="rgba(56, 189, 248, 0.4)", width=1.5, shape="spline"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Clean bars
    fig.add_trace(go.Bar(
        x=hours, y=counts,
        marker_color=bar_colors,
        marker_line_width=0,
        opacity=0.9,
        hovertemplate="Hour %{x:02d}:00: %{y:,} violations<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}" for h in range(24)],
            tickfont=dict(size=9, color="#94a3b8"),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=9, color="#94a3b8"),
        ),
        bargap=0.25,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

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
    Main entry point for Command Center view.
    Zero fluff: 3 KPIs, 3D Hotspot Map, Priority Action List, and 24h Hourly Profile.
    """
    if st is None:
        return
    inject_theme()
    render_header(mode)

    if df_scored_clean is None or df_scored_clean.empty:
        st.warning("⚠️ No active records found for current filters. Please broaden date or hour range in the sidebar.")
        return

    # Compute key numbers
    total_v = int(df_scored_clean[vol_col].sum()) if vol_col in df_scored_clean else 0
    total_p = float(df_scored_clean[pce_col].sum()) if pce_col in df_scored_clean else 0.0
    tier1 = int((df_scored_clean["roi_class"] == "Tier 1: Max Disruption").sum()) if "roi_class" in df_scored_clean else 0

    # 3 Clear KPIs
    render_kpi_banner(total_v, total_p, tier1)

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    # 2-Column Split: Map (65%) | Action List (35%)
    map_col, action_col = st.columns([13, 7], gap="medium")

    with map_col:
        st.markdown('<div class="cmd-section-title">VIOLATION HOTSPOT MAP</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.75rem; color:#94a3b8; margin-bottom:8px;'>"
            "Height = Traffic Impact Score · <span style='color:#38bdf8'>●</span> Moderate · "
            "<span style='color:#ef4444'>●</span> Critical Bottleneck"
            "</div>",
            unsafe_allow_html=True,
        )
        render_3d_hex_map(
            df_scored_clean,
            key=f"{map_key_prefix}_3d_map",
            pce_col=pce_col,
        )

    with action_col:
        st.markdown('<div class="cmd-section-title">PRIORITY ACTION LIST</div>', unsafe_allow_html=True)
        render_action_list(
            df_scored_clean,
            max_rows=8,
            key_prefix=map_key_prefix,
            pce_col=pce_col,
            vol_col=vol_col,
        )

    # Full-width 24h Violation Profile
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="cmd-section-title">VIOLATIONS BY HOUR OF DAY</div>', unsafe_allow_html=True)
    render_hourly_chart(hist_df, key=f"{map_key_prefix}_hourly_chart")
