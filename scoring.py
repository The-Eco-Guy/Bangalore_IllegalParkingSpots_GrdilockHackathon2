import pandas as pd
import numpy as np
import h3
import warnings
from libpysal.weights import W
from libpysal.weights.util import fill_diagonal
from esda.getisord import G_Local

def build_spatial_weights(cells):
    """
    Build H3 spatial weights grid using 1-ring neighbors and row standardization.
    """
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

def compute_gi_star(df, all_cells, value_col='weighted_pce', cell_col='h3_8'):
    """
    Compute Local Getis-Ord Gi* statistics dynamically for a slice of data, 
    reindexing it first to include all city cells for a consistent spatial grid.
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
                g_local = G_Local(y, w, transform='R', permutations=0, star=None)
            df_full['gi_zscore'] = g_local.Zs
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

def add_roi_classification(df, vol_col='violations', pce_col='weighted_pce'):
    """
    Calculate 90th percentile thresholds of active cells and add ROI classifications.
    """
    df = df.copy()
    active_cells = df[df[vol_col] > 0]
    
    if not active_cells.empty:
        vol_threshold = active_cells[vol_col].quantile(0.90)
        pce_threshold = active_cells[pce_col].quantile(0.90)
    else:
        vol_threshold = 1.0
        pce_threshold = 1.0
        
    df['roi_class'] = df.apply(
        lambda r: classify_roi(r, vol_threshold, pce_threshold, vol_col, pce_col), 
        axis=1
    )
    return df, vol_threshold, pce_threshold
