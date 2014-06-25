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
portland = Feed('gtfs_toolkit/tests/portland_gtfs.zip')

class TestFeed(unittest.TestCase):

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
        
    def test_get_segment_length(self):
        s = LineString([(0, 0), (1, 0)])
        p = Point((1/2, 0))
        self.assertEqual(get_segment_length(s, p), 1/2)
        q = Point((1/3, 0))
        self.assertAlmostEqual(get_segment_length(s, p, q), 1/6)
        p = Point((0, 1/2))
        self.assertEqual(get_segment_length(s, p), 0)

    def test_init(self):
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
        d1 = dt.date(2013, 12, 2)
        d2 = dt.date(2014, 6, 29)
        self.assertEqual(dates[0], d1)
        self.assertEqual(dates[-1], d2)
        self.assertEqual(len(dates), (d2 - d1).days + 1)

    def test_get_first_week(self):
        feed = cairns
        dates = feed.get_first_week()
        d1 = dt.date(2013, 12, 2)
        d2 = dt.date(2013, 12, 8)
        self.assertEqual(dates[0], d1)
        self.assertEqual(dates[-1], d2)
        self.assertEqual(len(dates), 7)

    def test_is_active(self):
        feed = cairns
        trip = 'CNS2014-CNS_MUL-Weekday-00-4166103'
        date1 = dt.date(2014, 3, 21)
        date2 = dt.date(2012, 3, 22)
        self.assertTrue(feed.is_active_trip(trip, date1))
        self.assertFalse(feed.is_active_trip(trip, date2))

        trip = 'CNS2014-CNS_MUL-Sunday-00-4165971'
        date1 = dt.date(2014, 4, 18)
        date2 = dt.date(2012, 4, 17)
        self.assertTrue(feed.is_active_trip(trip, date1))
        self.assertFalse(feed.is_active_trip(trip, date2))

        feed = portland
        trip = '4526377'
        date1 = dt.date(2014, 5, 18)
        date2 = dt.date(2012, 5, 17)
        self.assertTrue(feed.is_active_trip(trip, date1))
        self.assertFalse(feed.is_active_trip(trip, date2))

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

    def test_get_xy_by_stop(self):
        feed = cairns
        xy_by_stop = feed.get_xy_by_stop()
        # Should be a dictionary
        self.assertIsInstance(xy_by_stop, dict)
        # The first element should be a pair of numbers
        self.assertEqual(len(list(xy_by_stop.values())[0]), 2)
        # Should all stops
        self.assertEqual(len(xy_by_stop), feed.stops.shape[0])

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
        stops_ts = feed.get_stops_time_series(dates)['mean_daily_num_vehicles']
        # Should have the correct shape
        self.assertEqual(stops_ts.shape[0], 24*60)
        self.assertEqual(stops_ts.shape[1], feed.stops.shape[0])


if __name__ == '__main__':
    unittest.main()