"""Regression tests for selection, metric context, and map eligibility."""
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import pandas as pd
import modern_ui


class TestModernUI(unittest.TestCase):
    def test_selection_parses_dict_and_attribute_events(self):
        selection = {'objects': {'outline': [], 'locations': [{'h3_8': 'cell-a'}]}}
        self.assertEqual(modern_ui.selected_cell_from_event({'selection': selection}), 'cell-a')
        self.assertEqual(modern_ui.selected_cell_from_event(SimpleNamespace(selection=selection)), 'cell-a')
        for event in [None, {}, {'selection': {'objects': {'locations': []}}}]:
            self.assertIsNone(modern_ui.selected_cell_from_event(event))

    def test_callbacks_update_shared_selection_only_for_available_cells(self):
        state = {'map': {'selection': {'objects': {'locations': [{'h3_8': 'a'}]}}}}
        with patch.object(modern_ui.st, 'session_state', state):
            modern_ui.map_selection_callback('map', 'hist', ['a', 'b'])
            self.assertEqual(state['hist_selected_cell'], 'a')
            state['map']['selection']['objects']['locations'][0]['h3_8'] = 'gone'
            modern_ui.map_selection_callback('map', 'hist', ['a', 'b'])
            self.assertEqual(state['hist_selected_cell'], 'a')
            state['table'] = {'selection': {'rows': [1]}}
            modern_ui.table_selection_callback('table', 'hist', ['a', 'b'])
            self.assertEqual(state['hist_selected_cell'], 'b')
            self.assertEqual(state['hist_map_epoch'], 1)
            modern_ui.select_location('hist', 'a')
            self.assertEqual(state['hist_map_epoch'], 2)

    def test_selection_is_preserved_or_cleared_after_filtering(self):
        self.assertEqual(modern_ui.reconcile_selection('a', ['a', 'b']), 'a')
        self.assertIsNone(modern_ui.reconcile_selection('a', ['b']))
        self.assertIsNone(modern_ui.reconcile_selection(None, []))

    def test_hourly_data_respects_dates_bucket_and_exact_hour(self):
        df = pd.DataFrame({'date': [date(2024, 4, 8)] * 3 + [date(2024, 4, 7)],
                           'hour': [7, 10, 17, 7], 'violation_count': [2, 3, 9, 100]})
        ctx = modern_ui.ViewContext(start_date=date(2024, 4, 8), end_date=date(2024, 4, 8), bucket='AM Peak (07:00 - 10:59)')
        filtered = modern_ui.filter_hourly_data(df, ctx)
        self.assertEqual(filtered.violation_count.sum(), 5)
        self.assertEqual(modern_ui.hourly_profile(filtered).value.sum(), 5)
        ctx = modern_ui.ViewContext(start_date=date(2024, 4, 8), end_date=date(2024, 4, 8), hour=10)
        self.assertEqual(modern_ui.filter_hourly_data(df, ctx).hour.tolist(), [10])

    def test_forecasts_keep_fractional_values_and_explicit_units(self):
        ctx = modern_ui.ViewContext('Forecast', date(2024, 4, 9), date(2024, 4, 9), hour=7)
        self.assertEqual(ctx.window, '07:00–08:00 IST')
        self.assertEqual(ctx.export_fields()['metric_basis'], 'average per hour')
        self.assertIn('/ hour', modern_ui.metric_labels(True)[0])
        hourly = modern_ui.hourly_profile(pd.DataFrame({'hour': [7, 7, 8], 'pred_count': [.1, .25, .6]}), True)
        self.assertAlmostEqual(hourly.value.iloc[0], .35)

    def test_invalid_coordinates_are_excluded_and_exploration_retained(self):
        df = pd.DataFrame({'h3_8': ['a', 'b', 'c', 'd'], 'lat_center': [12, float('inf'), 100, 13],
                           'lon_center': [77, 77, 77, 78], 'violations': [1, 3, 3, 0],
                           'weighted_pce': [2, 5, 5, 0], 'is_exploration_target': [False, False, False, True]})
        ranked = modern_ui.ranked_locations(df)
        self.assertEqual(ranked.h3_8.tolist(), ['a', 'd'])
        self.assertEqual(modern_ui.priority_label(ranked.iloc[1]), 'Verification patrol')

    def test_fractional_simulation_is_reproducible_and_not_rounded_away(self):
        counts = [modern_ui.forecast_sample_count(.25, seed) for seed in range(1000)]
        self.assertGreater(sum(counts), 200)
        self.assertLess(sum(counts), 300)
        self.assertEqual(modern_ui.forecast_sample_count(2.25, 42), modern_ui.forecast_sample_count(2.25, 42))
        self.assertEqual(modern_ui.forecast_sample_count(float('nan'), 1), 0)


class TestForecastExports(unittest.TestCase):
    def test_export_rows_match_each_visible_layer_and_keep_metric_units(self):
        from io import BytesIO
        df = pd.DataFrame({
            'h3_8': ['density', 'impact', 'both'], 'junction_name': ['A', 'B', 'C'],
            'pred_count': [.25, .1, 2.5], 'pred_weighted_pce': [1., 4., 8.],
            'pred_density_active': [True, False, True], 'pred_congestion_active': [False, True, True],
            'pred_spillover_significant': [False, True, False], 'pred_enforcement_visible': [False, True, True],
            'gi_zscore': [0., 2., 0.], 'roi_class': ['Standard', 'Tier 2: Silent Bottleneck', 'Tier 1: Max Disruption'],
        })
        expected = {
            'Predicted violations / hour': {'density', 'both'},
            'Predicted impact / hour': {'impact', 'both'},
            'Predicted Combined Map': {'density', 'impact', 'both'},
            'Predicted Spillover': {'impact'},
            'Predicted Enforcement Plan': {'impact', 'both'},
            'Predicted Clusters (DBSCAN)': {'density', 'impact', 'both'},
        }
        for layer, cells in expected.items():
            with self.subTest(layer=layer):
                data = modern_ui.build_prediction_download(df, '2024-04-09', 7, 'AM Peak (07:00 - 10:59)', layer)
                exported = pd.read_csv(BytesIO(data))
                self.assertEqual(set(exported.h3_8), cells)
                self.assertTrue((exported.metric_basis == 'average per hour').all())
                self.assertTrue((exported.selected_forecast_hour == 7).all())
                self.assertTrue(exported.visible_in_selected_layer.all())
                if layer == 'Predicted Combined Map':
                    self.assertEqual(exported.predicted_value.tolist(), exported.pred_weighted_pce.tolist())
                if layer == 'Predicted violations / hour':
                    self.assertIn(.25, exported.predicted_value.tolist())
                self.assertTrue(exported.suggested_verification.str.contains('Inspect|Verify|Routine').all())

    def test_empty_forecast_export_retains_schema(self):
        from io import BytesIO
        df = pd.DataFrame(columns=['h3_8', 'pred_count', 'pred_weighted_pce', 'roi_class'])
        data = modern_ui.build_prediction_download(df, '2024-04-09', 'All', 'All', 'Predicted violations / hour')
        exported = pd.read_csv(BytesIO(data))
        self.assertTrue(exported.empty)
        self.assertIn('metric_basis', exported.columns)
