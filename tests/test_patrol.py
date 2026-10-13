"""
Unit tests for patrol.py (Tactical Patrol Route Optimizer & Mobile Dispatch)
"""

import unittest
import pandas as pd
import patrol

class TestPatrolModule(unittest.TestCase):
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
                'hour': 18
            },
            {
                'h3_8': 'cell_sony_world',
                'junction_name': 'Sony World Signal',
                'lat_center': 12.9352,
                'lon_center': 77.6245,
                'roi_class': 'Tier 2: Silent Bottleneck',
                'weighted_pce': 110.0,
                'violations': 18.0,
                'hour': 18
            },
            {
                'h3_8': 'cell_majestic',
                'junction_name': 'Majestic Circle',
                'lat_center': 12.9767,
                'lon_center': 77.5713,
                'roi_class': 'Tier 1: Max Disruption',
                'weighted_pce': 130.0,
                'violations': 70.0,
                'hour': 17
            },
            {
                'h3_8': 'cell_indiranagar',
                'junction_name': 'Indiranagar 100ft Rd',
                'lat_center': 12.9784,
                'lon_center': 77.6408,
                'roi_class': 'Tier 3: Volume Hotspot',
                'weighted_pce': 80.0,
                'violations': 60.0,
                'hour': 19
            },
            {
                'h3_8': 'cell_exploration',
                'junction_name': 'Outer Ring Rd Chokepoint',
                'lat_center': 12.9250,
                'lon_center': 77.6800,
                'roi_class': 'Standard',
                'weighted_pce': 75.0,
                'violations': 1.0,
                'is_exploration_target': True,
                'hour': 18
            }
        ])

    def test_haversine_distance(self):
        # Known approximate distance between Majestic (12.9767, 77.5713) and Silk Board (12.9175, 77.6238) ~ 8.7 km
        dist = patrol.haversine_distance_km(12.9767, 77.5713, 12.9175, 77.6238)
        self.assertGreater(dist, 8.0)
        self.assertLess(dist, 10.0)

    def test_assign_tactical_action(self):
        act_tier1 = patrol.assign_tactical_action({'roi_class': 'Tier 1: Max Disruption'})
        self.assertIn("HEAVY TOW SQUAD", act_tier1)

        act_tier2 = patrol.assign_tactical_action({'roi_class': 'Tier 2: Silent Bottleneck'})
        self.assertIn("ZERO-TOLERANCE TOW", act_tier2)

        act_tier3 = patrol.assign_tactical_action({'roi_class': 'Tier 3: Volume Hotspot'})
        self.assertIn("E-TICKETING", act_tier3)

        act_exp = patrol.assign_tactical_action({'is_exploration_target': True})
        self.assertIn("RECON PATROL", act_exp)

    def test_optimize_patrol_route(self):
        route = patrol.optimize_patrol_route(self.sample_df, max_stops=4)
        self.assertFalse(route.empty)
        self.assertLessEqual(len(route), 4)

        # Check expected schema
        expected_cols = [
            'stop_order', 'junction_name', 'h3_8', 'latitude', 'longitude',
            'roi_class', 'pce_impact', 'tactical_action',
            'leg_distance_km', 'cumulative_distance_km', 'google_maps_nav_link'
        ]
        for col in expected_cols:
            self.assertIn(col, route.columns)

        # Check cumulative distance is non-decreasing
        cum_dists = route['cumulative_distance_km'].tolist()
        self.assertEqual(cum_dists, sorted(cum_dists))

    def test_multiwaypoint_url(self):
        route = patrol.optimize_patrol_route(self.sample_df, max_stops=3)
        url = patrol.build_google_maps_multiwaypoint_url(route)
        self.assertTrue(url.startswith("https://www.google.com/maps/dir/"))
        self.assertIn("origin=", url)
        self.assertIn("destination=", url)

    def test_whatsapp_dispatch_memo(self):
        route = patrol.optimize_patrol_route(self.sample_df, max_stops=3)
        memo = patrol.generate_whatsapp_dispatch_memo(route, shift_name="PM Peak Shift")
        self.assertIn("BTP TACTICAL ENFORCEMENT DISPATCH", memo)
        self.assertIn("PM Peak Shift", memo)
        self.assertIn("Total Circuit Distance:", memo)
        self.assertIn("https://www.google.com/maps/dir/", memo)

if __name__ == '__main__':
    unittest.main()
