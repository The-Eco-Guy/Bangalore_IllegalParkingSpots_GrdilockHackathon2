"""
Unit tests for modern_ui.py (Command Center UI logic)
"""

import unittest
import pandas as pd
import numpy as np
import modern_ui

class TestModernUI(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame([
            {
                'h3_8': 'cell_silk_board',
                'junction_name': 'Silk Board Junction',
                'lat_center': 12.9175,
                'lon_center': 77.6238,
                'roi_class': 'Tier 1: Max Disruption',
                'weighted_pce': 160.0,
                'violations': 95.0,
            },
            {
                'h3_8': 'cell_sony_world',
                'junction_name': 'Sony World Signal',
                'lat_center': 12.9352,
                'lon_center': 77.6245,
                'roi_class': 'Tier 2: Silent Bottleneck',
                'weighted_pce': 110.0,
                'violations': 18.0,
            },
            {
                'h3_8': 'cell_indiranagar',
                'junction_name': 'Indiranagar 100ft Rd',
                'lat_center': 12.9784,
                'lon_center': 77.6408,
                'roi_class': 'Tier 3: Volume Hotspot',
                'weighted_pce': 80.0,
                'violations': 60.0,
            },
        ])

    def test_build_3d_colors(self):
        res = modern_ui._build_3d_colors(self.sample_df, pce_col='weighted_pce')
        self.assertIn('fill_color_3d', res.columns)
        self.assertIn('elevation_3d', res.columns)
        self.assertEqual(len(res), len(self.sample_df))
        self.assertTrue(all(isinstance(c, list) and len(c) == 4 for c in res['fill_color_3d']))
        # Highest PCE should have highest elevation
        max_idx = res['weighted_pce'].idxmax()
        min_idx = res['weighted_pce'].idxmin()
        self.assertGreater(res.loc[max_idx, 'elevation_3d'], res.loc[min_idx, 'elevation_3d'])

    def test_css_theme_defined(self):
        self.assertTrue(len(modern_ui.COMMAND_CSS) > 0)
        self.assertIn(".cmd-card", modern_ui.COMMAND_CSS)
        self.assertIn(".tag-tow", modern_ui.COMMAND_CSS)
        self.assertIn(".tag-ticket", modern_ui.COMMAND_CSS)
        self.assertIn(".tag-watch", modern_ui.COMMAND_CSS)

    def test_render_functions_do_not_crash_without_streamlit(self):
        # When headless / without streamlit, these should exit cleanly without exceptions
        modern_ui.render_header()
        modern_ui.render_kpi_banner(100, 250.0, 5)
        modern_ui.render_3d_hex_map(self.sample_df)
        modern_ui.render_hourly_chart(pd.DataFrame({'hour': [17, 18], 'violation_count': [10, 20]}))
        modern_ui.render_command_center(self.sample_df, None, None)

if __name__ == '__main__':
    unittest.main()

