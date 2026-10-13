"""Exercise real Streamlit widgets and each existing analytics workflow."""
from datetime import date
import unittest
from streamlit.testing.v1 import AppTest


class TestWorkspace(unittest.TestCase):
    def assert_clean(self, app):
        self.assertEqual([e.message for e in app.exception], [])

    def test_both_sources_and_every_analytics_layer(self):
        app = AppTest.from_file('app.py', default_timeout=45).run()
        self.assert_clean(app)
        app.radio(key='workspace').set_value('Analytics').run()
        for layer in app.get('button_group')[0].options:
            with self.subTest(source='Historical Records', layer=layer):
                app.session_state['hist_analysis_layer_select'] = layer
                app.run()
                self.assert_clean(app)
        app.radio(key='data_source').set_value('Forecast').run()
        self.assert_clean(app)
        for layer in app.get('button_group')[0].options:
            with self.subTest(source='Forecast', layer=layer):
                app.session_state['pred_analysis_layer_select'] = layer
                app.run()
                self.assert_clean(app)
        app.radio(key='workspace').set_value('Operations').run()
        self.assert_clean(app)
        app.selectbox(key='pred_time_bucket').set_value('AM Peak (07:00 - 10:59)').run()
        app.selectbox(key='pred_hour').set_value(7).run()
        self.assert_clean(app)
        self.assertIn('07:00–08:00 IST', ' '.join(m.value for m in app.markdown))
        self.assertTrue(any('/ hour' in m.label for m in app.metric))
        app.selectbox(key='pred_time_bucket').set_value('PM Peak (17:00 - 21:59)').run()
        self.assertEqual(app.selectbox(key='pred_hour').value, 'All')
        self.assert_clean(app)

    def test_reviewed_stops_origin_and_stale_route_invalidation(self):
        app = AppTest.from_file('app.py', default_timeout=45).run()
        chosen = app.multiselect(key='hist_patrol_stops').value[:2]
        app.multiselect(key='hist_patrol_stops').set_value(chosen).run()
        app.text_input(key='hist_origin_name').set_value('Central station').run()
        app.number_input(key='hist_origin_lat').set_value(12.95).run()
        app.number_input(key='hist_origin_lon').set_value(77.61).run()
        app.selectbox(key='hist_hour').set_value(7).run()
        # Filter changes intentionally reset defaults: reselect the reviewed current stops.
        chosen = app.multiselect(key='hist_patrol_stops').value[:2]
        app.multiselect(key='hist_patrol_stops').set_value(chosen).run()
        app.button(key='hist_generate_route').click().run()
        self.assert_clean(app)
        route = app.session_state['hist_route'][1]
        self.assertEqual(set(route.h3_8), set(chosen))
        self.assertTrue((route.origin_latitude == 12.95).all())
        self.assertTrue((route.origin_longitude == 77.61).all())
        self.assertTrue((route.origin_name == 'Central station').all())
        self.assertTrue((route.selected_hours == '07:00–08:00 IST').all())
        self.assertTrue((route.optimal_window == '07:00–08:00 IST').all())
        self.assertIn('Central station', app.text_area[0].value)
        app.radio(key='data_source').set_value('Forecast').run()
        app.radio(key='data_source').set_value('Historical Records').run()
        self.assertEqual(app.multiselect(key='hist_patrol_stops').value, chosen)
        self.assertEqual(app.text_input(key='hist_origin_name').value, 'Central station')
        app.number_input(key='hist_origin_lat').set_value(12.96).run()
        self.assert_clean(app)
        self.assertEqual(len(app.text_area), 0)

    def test_selection_survives_workspace_switch_and_empty_results_clear_it(self):
        app = AppTest.from_file('app.py', default_timeout=45).run()
        selector = app.selectbox(key='hist_location_choice')
        # AppTest options are display labels; its setter accepts the underlying cell value.
        cell = app.multiselect(key='hist_patrol_stops').value[0]
        selector.set_value(cell).run()
        self.assertEqual(app.session_state['hist_selected_cell'], cell)
        app.radio(key='workspace').set_value('Analytics').run()
        self.assertEqual(app.session_state['hist_selected_cell'], cell)
        app.radio(key='workspace').set_value('Operations').run()
        self.assert_clean(app)
        # This dataset starts with an empty early-hour window on the first recorded day.
        app.date_input(key='hist_dates').set_value((date(2023, 11, 10), date(2023, 11, 10))).run()
        app.selectbox(key='hist_hour').set_value(16).run()
        self.assert_clean(app)
        self.assertEqual(len(app.metric), 0)
        self.assertIsNone(app.session_state['hist_selected_cell'])
        self.assertTrue(any('No active locations' in i.value for i in app.info))
