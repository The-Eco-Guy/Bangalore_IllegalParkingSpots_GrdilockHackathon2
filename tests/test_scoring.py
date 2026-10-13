"""
Unit tests for scoring.py (ROI Classification, Fixed Thresholds, Exploration & Bias Scoring)
"""

import unittest
import pandas as pd
import numpy as np
import scoring

class TestScoringModule(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {'h3_8': 'c1', 'violations': 50.0, 'weighted_pce': 120.0, 'gi_zscore': 2.5},
            {'h3_8': 'c2', 'violations': 5.0,  'weighted_pce': 95.0,  'gi_zscore': 1.8},
            {'h3_8': 'c3', 'violations': 45.0, 'weighted_pce': 20.0,  'gi_zscore': 0.4},
            {'h3_8': 'c4', 'violations': 0.0,  'weighted_pce': 0.0,   'gi_zscore': -0.2},
            {'h3_8': 'c5', 'violations': 1.0,  'weighted_pce': 80.0,  'gi_zscore': 2.1},
        ])
        self.cell_meta = pd.DataFrame([
            {'h3_8': 'c1', 'pct_main_road': 0.6, 'avg_pce_weighted': 2.0},
            {'h3_8': 'c2', 'pct_main_road': 0.5, 'avg_pce_weighted': 1.8},
            {'h3_8': 'c3', 'pct_main_road': 0.1, 'avg_pce_weighted': 0.8},
            {'h3_8': 'c4', 'pct_main_road': 0.0, 'avg_pce_weighted': 0.5},
            {'h3_8': 'c5', 'pct_main_road': 0.7, 'avg_pce_weighted': 2.2},
        ])

    def test_classify_roi_tiers(self):
        # Tier 1: High vol, High PCE
        r1 = scoring.classify_roi({'violations': 40, 'weighted_pce': 100}, vol_threshold=30, pce_threshold=80)
        self.assertEqual(r1, 'Tier 1: Max Disruption')

        # Tier 2: Low vol, High PCE
        r2 = scoring.classify_roi({'violations': 10, 'weighted_pce': 100}, vol_threshold=30, pce_threshold=80)
        self.assertEqual(r2, 'Tier 2: Silent Bottleneck')

        # Tier 3: High vol, Low PCE
        r3 = scoring.classify_roi({'violations': 40, 'weighted_pce': 50}, vol_threshold=30, pce_threshold=80)
        self.assertEqual(r3, 'Tier 3: Volume Hotspot')

        # Standard: Low vol, Low PCE
        r4 = scoring.classify_roi({'violations': 10, 'weighted_pce': 50}, vol_threshold=30, pce_threshold=80)
        self.assertEqual(r4, 'Standard')

        # Zero violations
        r0 = scoring.classify_roi({'violations': 0, 'weighted_pce': 0}, vol_threshold=30, pce_threshold=80)
        self.assertEqual(r0, 'Standard')

    def test_add_roi_classification_fixed_thresholds(self):
        # Test consistent operational classification with fixed thresholds
        res_df, v_thresh, p_thresh = scoring.add_roi_classification(
            self.df,
            vol_col='violations',
            pce_col='weighted_pce',
            fixed_vol_thresh=25.0,
            fixed_pce_thresh=70.0
        )
        self.assertEqual(v_thresh, 25.0)
        self.assertEqual(p_thresh, 70.0)
        self.assertEqual(res_df.loc[res_df['h3_8'] == 'c1', 'roi_class'].iloc[0], 'Tier 1: Max Disruption')
        self.assertEqual(res_df.loc[res_df['h3_8'] == 'c2', 'roi_class'].iloc[0], 'Tier 2: Silent Bottleneck')
        self.assertEqual(res_df.loc[res_df['h3_8'] == 'c3', 'roi_class'].iloc[0], 'Tier 3: Volume Hotspot')

    def test_exploration_and_bias_scoring(self):
        exp_df = scoring.compute_exploration_and_bias(
            self.df,
            self.cell_meta,
            vol_col='violations',
            pce_col='weighted_pce'
        )
        self.assertIn('exploration_score', exp_df.columns)
        self.assertIn('is_exploration_target', exp_df.columns)
        self.assertIn('exploration_reason', exp_df.columns)

        # Cell c5 has very low recorded violations (1.0) but high z-score (2.1), high main road (0.7), high PCE (2.2)
        # It must be flagged as an exploration target (unpatrolled vulnerability zone)
        c5_target = exp_df.loc[exp_df['h3_8'] == 'c5', 'is_exploration_target'].iloc[0]
        self.assertTrue(c5_target)

        # Cell c1 has high recorded violations (50.0), so it is already well-covered
        c1_target = exp_df.loc[exp_df['h3_8'] == 'c1', 'is_exploration_target'].iloc[0]
        self.assertFalse(c1_target)

if __name__ == '__main__':
    unittest.main()
