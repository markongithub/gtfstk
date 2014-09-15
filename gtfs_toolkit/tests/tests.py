import unittest

import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal, assert_series_equal
from shapely.geometry import Point, LineString, mapping

from gtfs_toolkit.feed import *
from gtfs_toolkit.utils import *

# Load test feeds
cairns = Feed('gtfs_toolkit/tests/cairns_gtfs.zip')
cairns_shapeless = Feed('gtfs_toolkit/tests/cairns_gtfs.zip')
cairns_shapeless.shapes = None

class TestFeed(unittest.TestCase):
    """
    def test_seconds_to_timestr(self):
        seconds = 3600 + 60 + 1
        timestr = '01:01:01'
        self.assertEqual(seconds_to_timestr(seconds), timestr)
        self.assertEqual(seconds_to_timestr(timestr, inverse=True), seconds)
        self.assertIsNone(seconds_to_timestr(timestr))
        self.assertIsNone(seconds_to_timestr(seconds, inverse=True))
        self.assertIsNone(seconds_to_timestr('01:01', inverse=True))

    def test_timestr_mod_24(self):
        timestr1 = '01:01:01'
        self.assertEqual(timestr_mod_24(timestr1), timestr1)
        timestr2 = '25:01:01'
        self.assertEqual(timestr_mod_24(timestr2), timestr1)
        
    def test_to_km(self):
        units = 'mi'
        self.assertEqual(to_km(1, units), 1.6093)

    def test_get_segment_length(self):
        s = LineString([(0, 0), (1, 0)])
        p = Point((1/2, 0))
        self.assertEqual(get_segment_length(s, p), 1/2)
        q = Point((1/3, 0))
        self.assertAlmostEqual(get_segment_length(s, p, q), 1/6)
        p = Point((0, 1/2))
        self.assertEqual(get_segment_length(s, p), 0)

    def test_init(self):
        # Test distance units check
        self.assertRaises(AssertionError, Feed, 
          path='gtfs_toolkit/tests/cairns_gtfs.zip', 
          original_units='bingo')
        # Test other stuff
        portland = Feed('gtfs_toolkit/tests/portland_gtfs.zip', 
          original_units='ft')
        for feed in [cairns, portland]:
            self.assertIsInstance(feed.routes, pd.core.frame.DataFrame)
            self.assertIsInstance(feed.stops, pd.core.frame.DataFrame)
            self.assertIsInstance(feed.shapes, pd.core.frame.DataFrame)
            self.assertIsInstance(feed.trips, pd.core.frame.DataFrame)
            if feed.calendar is not None:
                self.assertIsInstance(feed.calendar, pd.core.frame.DataFrame)
            if feed.calendar_dates is not None:
                self.assertIsInstance(feed.calendar_dates, 
                  pd.core.frame.DataFrame)

    def test_get_dates(self):
        feed = cairns
        dates = feed.get_dates()
        d1 = dt.date(2014, 5, 26)
        d2 = dt.date(2014, 12, 28)
        self.assertEqual(dates[0], d1)
        self.assertEqual(dates[-1], d2)
        self.assertEqual(len(dates), (d2 - d1).days + 1)

    def test_get_first_week(self):
        feed = cairns
        dates = feed.get_first_week()
        d1 = dt.date(2014, 5, 26)
        d2 = dt.date(2014, 6, 1)
        self.assertEqual(dates[0], d1)
        self.assertEqual(dates[-1], d2)
        self.assertEqual(len(dates), 7)

    def test_is_active(self):
        feed = cairns
        trip = 'CNS2014-CNS_MUL-Weekday-00-4165878'
        date1 = dt.date(2014, 5, 26)
        date2 = dt.date(2012, 3, 22)
        self.assertTrue(feed.is_active_trip(trip, date1))
        self.assertFalse(feed.is_active_trip(trip, date2))

        trip = 'CNS2014-CNS_MUL-Sunday-00-4165971'
        date1 = dt.date(2014, 6, 1)
        date2 = dt.date(2012, 6, 2)
        self.assertTrue(feed.is_active_trip(trip, date1))
        self.assertFalse(feed.is_active_trip(trip, date2))

        feed = Feed('gtfs_toolkit/tests/portland_gtfs.zip')
        trip = '4526377'
        date1 = dt.date(2014, 5, 18)
        date2 = dt.date(2012, 5, 17)
        self.assertTrue(feed.is_active_trip(trip, date1))
        self.assertFalse(feed.is_active_trip(trip, date2))

    def test_get_active_trips(self):
        feed = cairns
        date = feed.get_first_week()[0]
        f = feed.get_active_trips(date)
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(f.shape[0] <= feed.trips.shape[0])
        self.assertEqual(f.shape[1], feed.trips.shape[1])
        # Should have correct columns
        self.assertEqual(set(f.columns), set(feed.trips.columns))

        g = feed.get_active_trips(date, "07:30:00")
        # Should be a data frame
        self.assertIsInstance(g, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(g.shape[0] <= f.shape[0])
        self.assertEqual(g.shape[1], f.shape[1])
        # Should have correct columns
        self.assertEqual(set(g.columns), set(feed.trips.columns))

    def test_get_vehicles_locations(self):
        feed = cairns
        trips_stats = feed.get_trips_stats()
        feed.add_dist_to_stop_times(trips_stats)
        linestring_by_shape = feed.get_linestring_by_shape(use_utm=False)
        date = feed.get_first_week()[0]
        timestr = '07:35:00'
        f = feed.get_vehicles_locations(linestring_by_shape, date, timestr)
        g = feed.get_active_trips(date, timestr)
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct number of rows
        self.assertEqual(f.shape[0], g.shape[0])
        # Should have the correct columns
        get_cols = set(f.columns)
        expect_cols = set(list(g.columns) + ['rel_dist', 'lon', 'lat'])
        self.assertEqual(get_cols, expect_cols)

    def test_get_trips_activity(self):
        feed = cairns
        dates = feed.get_first_week()
        trips_activity = feed.get_trips_activity(dates)
        # Should be a data frame
        self.assertIsInstance(trips_activity, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertEqual(trips_activity.shape[0], feed.trips.shape[0])
        self.assertEqual(trips_activity.shape[1], len(dates) + 3)
        # Date columns should contain only zeros and ones
        self.assertEqual(set(trips_activity[dates].values.flatten()), {0, 1})

    def test_get_trips_stats(self):
        feed = cairns
        trips_stats = feed.get_trips_stats()
        # Should be a data frame with the correct number of rows
        self.assertIsInstance(trips_stats, pd.core.frame.DataFrame)
        self.assertEqual(trips_stats.shape[0], feed.trips.shape[0])
        # Shapeless feeds should have null entries for distance column
        feed2 = cairns_shapeless
        trips_stats = feed2.get_trips_stats()
        self.assertEqual(len(trips_stats['distance'].unique()), 1)
        self.assertTrue(np.isnan(trips_stats['distance'].unique()[0]))   
        # Should contain the correct trips
        get_trips = set(trips_stats['trip_id'].values)
        expect_trips = set(feed.trips['trip_id'].values)
        self.assertEqual(get_trips, expect_trips)

    def test_get_linestring_by_shape(self):
        feed = cairns
        linestring_by_shape = feed.get_linestring_by_shape()
        # Should be a dictionary
        self.assertIsInstance(linestring_by_shape, dict)
        # The first element should be a Shapely linestring
        self.assertIsInstance(list(linestring_by_shape.values())[0], 
          LineString)
        # Should contain all shapes
        self.assertEqual(len(linestring_by_shape), 
          feed.shapes.groupby('shape_id').first().shape[0])
        # Should be None if feed.shapes is None
        feed2 = cairns_shapeless
        self.assertIsNone(feed2.get_linestring_by_shape())

    def test_get_point_by_stop(self):
        feed = cairns
        point_by_stop = feed.get_point_by_stop()
        # Should be a dictionary
        self.assertIsInstance(point_by_stop, dict)
        # The first element should be a Shapely point
        self.assertIsInstance(list(point_by_stop.values())[0], Point)
        # Should include all stops
        self.assertEqual(len(point_by_stop), feed.stops.shape[0])

    def test_get_stops_activity(self):
        feed = cairns
        dates = feed.get_first_week()
        stops_activity = feed.get_stops_activity(dates)
        # Should be a data frame
        self.assertIsInstance(stops_activity, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertEqual(stops_activity.shape[0], feed.stops.shape[0])
        self.assertEqual(stops_activity.shape[1], len(dates) + 1)
        # Date columns should contain only zeros and ones
        self.assertEqual(set(stops_activity[dates].values.flatten()), {0, 1})

    def test_add_dist_to_stop_times(self):
        feed = cairns
        st1 = feed.stop_times.copy()
        trips_stats = feed.get_trips_stats()
        feed.add_dist_to_stop_times(trips_stats)
        st2 = feed.stop_times

        # Check that colums of st2 equal the columns of st1 plus
        # a shape_dist_traveled column
        cols1 = list(st1.columns.values) + ['shape_dist_traveled']
        cols2 = list(st2.columns.values)
        self.assertEqual(set(cols1), set(cols2))

        # Check that within each trip the shape_dist_traveled column 
        # is monotonically increasing
        for name, group in st2.groupby('trip_id'):
            sdt = list(group['shape_dist_traveled'].values)
            self.assertEqual(sdt, sorted(sdt))

    def test_add_dist_to_shapes(self):
        feed = cairns
        s1 = feed.shapes.copy()
        feed.add_dist_to_shapes()
        s2 = feed.shapes
        # Check that colums of st2 equal the columns of st1 plus
        # a shape_dist_traveled column
        cols1 = list(s1.columns.values) + ['shape_dist_traveled']
        cols2 = list(s2.columns.values)
        self.assertEqual(set(cols1), set(cols2))

        # Check that within each trip the shape_dist_traveled column 
        # is monotonically increasing
        for name, group in s2.groupby('shape_id'):
            sdt = list(group['shape_dist_traveled'].values)
            self.assertEqual(sdt, sorted(sdt))

    def test_get_stops_stats(self):
        feed = cairns
        dates = feed.get_first_week()
        stops_stats = feed.get_stops_stats(dates)
        # Should be a data frame
        self.assertIsInstance(stops_stats, pd.core.frame.DataFrame)
        # Should contain the correct stops
        get_stops = set(stops_stats['stop_id'].values)
        sa = feed.get_stops_activity(dates)
        sa = sa[sa[dates].sum(axis=1) > 0]
        expect_stops = set(sa['stop_id'].values)
        self.assertEqual(get_stops, expect_stops)

    def test_get_stops_time_series(self):
        feed = cairns
        dates = feed.get_first_week()
        sa = feed.get_stops_activity(dates)
        sa = sa[sa[dates].sum(axis=1) > 0]
        for split_directions in [True, False]:
            f = feed.get_stops_stats(dates, 
              split_directions=split_directions)
            stops_ts = feed.get_stops_time_series(dates, freq='1H',
              split_directions=split_directions) 
            # Should be a data frame
            self.assertIsInstance(stops_ts, pd.core.frame.DataFrame)
            # Should have the correct shape
            self.assertEqual(stops_ts.shape[0], 24)
            self.assertEqual(stops_ts.shape[1], f.shape[0])
            # Should have correct column names
            if split_directions:
                expect = ['statistic', 'stop_id', 'direction_id']
            else:
                expect = ['statistic', 'stop_id']
            self.assertEqual(stops_ts.columns.names, expect)
    """
    def test_get_routes_stats(self):
        feed = cairns
        dates = feed.get_first_week()
        trips_stats = feed.get_trips_stats()
        f = pd.merge(trips_stats, feed.get_trips_activity(dates))
        f = f[f[dates].sum(axis=1) > 0]
        for split_directions in [True, False]:
            rs = feed.get_routes_stats(trips_stats, dates, 
              split_directions=split_directions)
            # Should be a data frame of the correct shape
            self.assertIsInstance(rs, pd.core.frame.DataFrame)
            if split_directions:
                f['tmp'] = f['route_id'] + '-' +\
                  f['direction_id'].map(str)
            else:
                f['tmp'] = f['route_id'].copy()
            expect_num_routes = len(f['tmp'].unique())
            self.assertEqual(rs.shape[0], expect_num_routes)

    def test_get_routes_time_series(self):
        feed = cairns 
        dates = feed.get_first_week()
        trips_stats = feed.get_trips_stats()
        for split_directions in [True, False]:
            f = feed.get_routes_stats(trips_stats, dates, 
              split_directions=split_directions)
            rts = feed.get_routes_time_series(trips_stats, dates, 
              split_directions=split_directions, freq='1H')
            # Should be a data frame of the correct shape
            self.assertIsInstance(rts, pd.core.frame.DataFrame)
            self.assertEqual(rts.shape[0], 24)
            self.assertEqual(rts.shape[1], 5*f.shape[0])
            # Should have correct column names
            if split_directions:
                expect = ['statistic', 'route_id', 'direction_id']
            else:
                expect = ['statistic', 'route_id']
            self.assertEqual(rts.columns.names, expect)   
    
    def test_agg_routes_time_series(self):
        feed = cairns 
        dates = feed.get_first_week()
        trips_stats = feed.get_trips_stats()
        for split_directions in [True, False]:
            rts = feed.get_routes_time_series(trips_stats, dates, 
              split_directions=split_directions, freq='1H')
            arts = agg_routes_time_series(rts)
            if split_directions:
                num_cols = 2*len(rts.columns.levels[0])
                col_names = ['statistic', 'direction_id']
            else:
                num_cols = len(rts.columns.levels[0])
                col_names = [None]
            # Should be a data frame of the correct shape
            self.assertIsInstance(arts, pd.core.frame.DataFrame)
            self.assertEqual(arts.shape[0], 24)
            self.assertEqual(arts.shape[1], num_cols)
            # Should have correct column names
            self.assertEqual(arts.columns.names, col_names)   
    
if __name__ == '__main__':
    unittest.main()