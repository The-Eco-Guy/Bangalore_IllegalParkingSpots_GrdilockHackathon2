import pandas as pd
import numpy as np
import warnings

try:
    import h3
except ImportError:
    h3 = None

try:
    from libpysal.weights import W
    from libpysal.weights.util import fill_diagonal
    from esda.getisord import G_Local
except ImportError:
    W = None
    fill_diagonal = None
    G_Local = None

def build_spatial_weights(cells):
    """
    Build H3 spatial weights grid using 1-ring neighbors and row standardization.
    """
    if h3 is None or W is None:
        raise ImportError("h3 and libpysal are required for spatial weights. Please install requirements.txt.")

    cell_set = set(cells)
    c2i = {c: i for i, c in enumerate(cells)}

    neighbors_dict = {}
    for cell in cells:
        ring1 = set(h3.grid_disk(cell, k=1)) - {cell}
        neighbors_dict[c2i[cell]] = [c2i[n] for n in ring1 if n in cell_set]

    # Handle isolates safely; disconnected components are expected with sparse city cells.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = W(neighbors_dict, silence_warnings=True)
        w = fill_diagonal(w, val=1.0)
    w.transform = 'r'
    return w

def compute_gi_star(df, all_cells, value_col='weighted_pce', cell_col='h3_8', permutations=0):
    """
    Compute Local Getis-Ord Gi* statistics dynamically for a slice of data, 
    reindexing it first to include all city cells for a consistent spatial grid.
    Permutations > 0 triggers conditional permutation-based pseudo-p-values,
    which provides statistical rigor for zero-inflated urban count distributions.
    """
    # Reindex to the complete set of H3 cells so that the statistical background is correct
    df_full = df.set_index(cell_col).reindex(all_cells).copy()
    
    # Fill missing values with 0
    if 'violations' in df_full.columns:
        df_full['violations'] = df_full['violations'].fillna(0)
    if 'pred_count' in df_full.columns:
        df_full['pred_count'] = df_full['pred_count'].fillna(0)
    df_full[value_col] = df_full[value_col].fillna(0)
    df_full = df_full.reset_index()
    
    w = build_spatial_weights(all_cells)
    y = df_full[value_col].values.astype(float)
    
    if np.std(y) > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g_local = G_Local(y, w, transform='R', permutations=permutations, star=None)
            df_full['gi_zscore'] = g_local.Zs
            if permutations > 0 and hasattr(g_local, 'p_sim') and g_local.p_sim is not None:
                df_full['gi_pvalue'] = g_local.p_sim
            else:
                df_full['gi_pvalue'] = g_local.p_norm
        except Exception:
            df_full['gi_zscore'] = 0.0
            df_full['gi_pvalue'] = 1.0
    else:
        df_full['gi_zscore'] = 0.0
        df_full['gi_pvalue'] = 1.0
        
    return df_full

def classify_roi(row, vol_threshold, pce_threshold, vol_col='violations', pce_col='weighted_pce'):
    """
    Classify H3 cells into strategic enforcement tiers.
    """
    violations = row[vol_col]
    pce = row[pce_col]
    
    if violations == 0:
        return 'Standard'
        
    is_high_volume = violations >= vol_threshold
    is_high_congestion = pce >= pce_threshold

    if is_high_volume and is_high_congestion:
        return 'Tier 1: Max Disruption'
    elif is_high_congestion and not is_high_volume:
        return 'Tier 2: Silent Bottleneck'
    elif is_high_volume and not is_high_congestion:
        return 'Tier 3: Volume Hotspot'
    else:
        return 'Standard'

def add_roi_classification(df, vol_col='violations', pce_col='weighted_pce', fixed_vol_thresh=None, fixed_pce_thresh=None):
    """
    Add ROI classifications to H3 cells.
    If fixed_vol_thresh and fixed_pce_thresh are supplied, uses consistent baseline thresholds
    to avoid shifting classification boundaries across different time/date filters.
    Otherwise, adaptively computes 90th percentile from active cells in current slice.
    """
    df = df.copy()
    active_cells = df[df[vol_col] > 0]
    
    if fixed_vol_thresh is not None and fixed_pce_thresh is not None:
        vol_threshold = float(fixed_vol_thresh)
        pce_threshold = float(fixed_pce_thresh)
    elif not active_cells.empty:
        vol_threshold = float(active_cells[vol_col].quantile(0.90))
        pce_threshold = float(active_cells[pce_col].quantile(0.90))
    else:
        vol_threshold = 1.0
        pce_threshold = 1.0
        
    df['roi_class'] = df.apply(
        lambda r: classify_roi(r, vol_threshold, pce_threshold, vol_col, pce_col), 
        axis=1
    )
    return df, vol_threshold, pce_threshold

def compute_exploration_and_bias(df, cell_meta, vol_col='violations', pce_col='weighted_pce'):
    """
    Operational & Geostatistical bias correction:
    Identifies cells with high structural traffic/congestion vulnerability (arterial junctions,
    high neighbor spillover) but low/zero historical enforcement records.
    Directly counteracts the spatial selection bias (where officers only find violations
    where they already patrol) by surfacing high-priority 'Exploration Patrol Targets'.
    """
    df = df.copy()
    meta_dict = cell_meta.set_index('h3_8').to_dict('index') if cell_meta is not None and not cell_meta.empty else {}
    
    exploration_scores = []
    is_target_list = []
    reasons = []
    
    # Calculate median active volume for baseline reference
    active = df[df[vol_col] > 0]
    ref_vol = active[vol_col].quantile(0.50) if not active.empty else 1.0
    ref_pce = active[pce_col].quantile(0.50) if not active.empty else 1.0
    
    for idx, row in df.iterrows():
        cell_id = row['h3_8']
        vol = float(row.get(vol_col, 0.0) or 0.0)
        pce = float(row.get(pce_col, 0.0) or 0.0)
        z = float(row.get('gi_zscore', 0.0) or 0.0)
        
        meta = meta_dict.get(cell_id, {})
        pct_main_road = float(meta.get('pct_main_road', 0.0) or 0.0)
        avg_pce_w = float(meta.get('avg_pce_weighted', 1.0) or 1.0)
        
        # Exposure proxy: high arterial presence or surrounding cluster spillover
        structural_risk = 0.0
        if z > 1.2:
            structural_risk += (z - 1.2) * 1.5
        if pct_main_road > 0.3:
            structural_risk += pct_main_road * 2.0
        if avg_pce_w > 1.3:
            structural_risk += (avg_pce_w - 1.3) * 1.8
            
        # Enforcement coverage penalty: high volume means area is already covered
        coverage_ratio = vol / (ref_vol + 1e-4)
        
        # Exploration score is high when structural risk is elevated but coverage is near zero
        exp_score = max(0.0, structural_risk / (1.0 + coverage_ratio * 2.5))
        exploration_scores.append(round(exp_score, 2))
        
        # Flag as exploration target if low recorded violations (< 25th percentile or <= 2) and elevated risk
        if vol <= max(2.0, ref_vol * 0.25) and exp_score >= 1.0:
            is_target_list.append(True)
            if z > 1.5 and pct_main_road > 0.3:
                reasons.append("High-risk arterial corridor with severe neighboring spillover but zero patrol coverage.")
            elif pct_main_road > 0.4:
                reasons.append("Major transit artery with suspected unrecorded illegal double-parking.")
            elif z > 1.5:
                reasons.append("Spillover vortex adjacent to primary gridlock hotspot.")
            else:
                reasons.append("High vehicle PCE footprint with low patrol enforcement visibility.")
        else:
            is_target_list.append(False)
            reasons.append("Standard patrol coverage zone.")
            
    df['exploration_score'] = exploration_scores
    df['is_exploration_target'] = is_target_list
    df['exploration_reason'] = reasons
    return df

