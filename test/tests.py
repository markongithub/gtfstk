from __future__ import division
import unittest

import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal
from shapely.geometry import Point, LineString, mapping

from feed import *

# Create your tests here.
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
        darwin_path = 'test/darwin_20130903/'
        darwin = Feed(darwin_path)
        self.assertIsInstance(darwin.routes, pd.core.frame.DataFrame)
        self.assertIsInstance(darwin.stops, pd.core.frame.DataFrame)
        self.assertIsInstance(darwin.shapes, pd.core.frame.DataFrame)
        self.assertIsInstance(darwin.trips, pd.core.frame.DataFrame)
        self.assertIsInstance(darwin.calendar, pd.core.frame.DataFrame)
        self.assertIsInstance(darwin.calendar_m, pd.core.frame.DataFrame)
        self.assertIsInstance(darwin.calendar_dates, pd.core.frame.DataFrame)

    def test_get_dates(self):
        feed = Feed('test/darwin_20130903/')
        dates = feed.get_dates()
        d1 = dt.date(2012, 11, 5)
        d2 = dt.date(2013, 12, 31)
        self.assertEqual(dates[0], d1)
        self.assertEqual(dates[-1], d2)
        self.assertEqual(len(dates), (d2 - d1).days + 1)

    def test_get_first_week(self):
        feed = Feed('test/darwin_20130903/')
        dates = feed.get_first_week()
        d1 = dt.date(2012, 11, 5)
        d2 = dt.date(2012, 11, 11)
        self.assertEqual(dates[0], d1)
        self.assertEqual(dates[-1], d2)
        self.assertEqual(len(dates), 7)

    def test_is_active(self):
        # This feed has calendar_dates data
        darwin = Feed('test/darwin_20130903/')
        trip = 'i1_2'
        date1 = dt.date(2012, 12, 24)
        date2 = dt.date(2012, 12, 25)
        self.assertTrue(darwin.is_active_trip(trip, date1))
        self.assertFalse(darwin.is_active_trip(trip, date2))

        # This feed doesn't have calendar_dates data
        seq = Feed('test/seq_20140128/')
        trip = '3972078-BBL2014-399-Weekday-01'
        date1 = dt.date(2014, 3, 3)
        date2 = dt.date(2014, 3, 8)
        self.assertTrue(seq.is_active_trip(trip, date1))
        self.assertFalse(seq.is_active_trip(trip, date2))

    def test_get_linestring_by_shape(self):
        # This feed has calendar_dates data
        feed = Feed('test/darwin_20130903/')
        shape = 'i1_shp'
        linestring_by_shape = feed.get_linestring_by_shape()
        # Should be a dictionary
        self.assertIsInstance(linestring_by_shape, dict)
        # The first element should be a Shapely linestring
        self.assertIsInstance(linestring_by_shape.values()[0], LineString)
        # Should contain the correct number of shapes
        self.assertEqual(len(linestring_by_shape), 
          feed.shapes.groupby('shape_id').first().shape[0])


if __name__ == '__main__':
    unittest.main()