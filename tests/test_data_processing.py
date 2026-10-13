"""
Unit tests for data_processing.py
"""

import unittest
import pandas as pd
import datetime
import data_processing

class TestDataProcessing(unittest.TestCase):
    def test_get_hours_for_bucket(self):
        am = data_processing.get_hours_for_bucket('AM Peak (07:00 - 10:59)')
        self.assertEqual(am, [7, 8, 9, 10])

        mid = data_processing.get_hours_for_bucket('Midday (11:00 - 16:59)')
        self.assertEqual(mid, [11, 12, 13, 14, 15, 16])

        pm = data_processing.get_hours_for_bucket('PM Peak (17:00 - 21:59)')
        self.assertEqual(pm, [17, 18, 19, 20, 21])

        night = data_processing.get_hours_for_bucket('Off-Peak Night (22:00 - 06:59)')
        self.assertEqual(night, [22, 23, 0, 1, 2, 3, 4, 5, 6])

    def test_filter_aggregated_data(self):
        d1 = datetime.date(2024, 1, 15)
        d2 = datetime.date(2024, 1, 16)
        df = pd.DataFrame([
            {'h3_8': 'cell_a', 'date': d1, 'hour': 8, 'violation_count': 10, 'weighted_pce': 20.0, 'lat_mean': 12.9, 'lon_mean': 77.6},
            {'h3_8': 'cell_a', 'date': d1, 'hour': 18, 'violation_count': 15, 'weighted_pce': 30.0, 'lat_mean': 12.9, 'lon_mean': 77.6},
            {'h3_8': 'cell_b', 'date': d2, 'hour': 8, 'violation_count': 5, 'weighted_pce': 10.0, 'lat_mean': 12.95, 'lon_mean': 77.65},
        ])

        # Filter by hour=18
        res = data_processing.filter_aggregated_data(df, hour=18)
        self.assertEqual(len(res), 1)
        self.assertEqual(res['violations'].iloc[0], 15)

        # Filter by time_bucket='AM Peak (07:00 - 10:59)'
        res_am = data_processing.filter_aggregated_data(df, time_bucket='AM Peak (07:00 - 10:59)')
        self.assertEqual(len(res_am), 2)
        self.assertEqual(set(res_am['h3_8'].tolist()), {'cell_a', 'cell_b'})

if __name__ == '__main__':
    unittest.main()
