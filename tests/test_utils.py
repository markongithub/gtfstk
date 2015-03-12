import unittest
from copy import copy

import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal, assert_series_equal
from shapely.geometry import Point, LineString, mapping
from shapely.geometry import shape as sh_shape

from gtfs_tk.feed import *
from gtfs_tk.utils import *

# Load test feeds
cairns = Feed('data/cairns_gtfs.zip')
cairns_shapeless = Feed('data/cairns_gtfs.zip')
cairns_shapeless.shapes = None

class TestFeed(unittest.TestCase):
    # Test utils functions
    def test_timestr_to_seconds(self):
        timestr1 = '01:01:01'
        seconds1 = 3600 + 60 + 1
        timestr2 = '25:01:01'
        seconds2 = 25*3600 + 60 + 1
        self.assertEqual(timestr_to_seconds(timestr1), seconds1)
        self.assertEqual(timestr_to_seconds(seconds1, inverse=True), timestr1)
        self.assertEqual(timestr_to_seconds(seconds2, inverse=True), timestr2)
        self.assertEqual(timestr_to_seconds(timestr2, mod24=True), seconds1)
        self.assertEqual(
          timestr_to_seconds(seconds2, mod24=True, inverse=True), timestr1)
        # Test error handling
        self.assertTrue(np.isnan(timestr_to_seconds(seconds1)))
        self.assertTrue(np.isnan(timestr_to_seconds(timestr1, inverse=True)))

    def test_timestr_mod24(self):
        timestr1 = '01:01:01'
        self.assertEqual(timestr_mod24(timestr1), timestr1)
        timestr2 = '25:01:01'
        self.assertEqual(timestr_mod24(timestr2), timestr1)

    def test_datestr_to_date(self):
        datestr = '20140102'
        date = dt.date(2014, 1, 2)
        self.assertEqual(datestr_to_date(datestr), date)
        self.assertEqual(datestr_to_date(date, inverse=True), datestr)

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

if __name__ == '__main__':
    unittest.main()