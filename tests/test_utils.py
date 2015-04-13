import unittest
from copy import copy

import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
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
    def test_time_to_seconds(self):
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

    def test_time_mod24(self):
        timestr1 = '01:01:01'
        self.assertEqual(timestr_mod24(timestr1), timestr1)
        timestr2 = '25:01:01'
        self.assertEqual(timestr_mod24(timestr2), timestr1)

    def test_datestr_to_date(self):
        datestr = '20140102'
        date = dt.date(2014, 1, 2)
        self.assertEqual(datestr_to_date(datestr), date)
        self.assertEqual(datestr_to_date(date, inverse=True), datestr)

    def test_get_convert_dist(self):
        di = 'mi'
        do = 'km'
        f = get_convert_dist(di, do)
        self.assertEqual(f(1), 1.609344)

    def test_clean_series(self):
        feed = copy(cairns) # Has all non-NaN route short names
        
        # Set some route short names to NaN
        s = feed.routes['route_short_name'].copy()
        s[s.str.endswith('1')] = 'bang'
        s[s.str.endswith('N')] = np.nan
        dups = s[s == 'bang']
        nans = s[s.isnull()]
        
        t = clean_series(s, 'zing')
        # Duplicates replacement should work
        assert_array_equal(dups.index, t[t.str.startswith('bang')].index)
        # NaNs replacement should work
        assert_array_equal(nans.index, t[t.str.startswith('zing')].index)

        # Should contain no NaNs or duplicates
        self.assertTrue(t[t.duplicated()].empty)
        self.assertTrue(t[t.isnull()].empty)

    def test_get_segment_length(self):
        s = LineString([(0, 0), (1, 0)])
        p = Point((1/2, 0))
        self.assertEqual(get_segment_length(s, p), 1/2)
        q = Point((1/3, 0))
        self.assertAlmostEqual(get_segment_length(s, p, q), 1/6)
        p = Point((0, 1/2))
        self.assertEqual(get_segment_length(s, p), 0)

    def test_get_max_runs(self):
        x = [7, 1, 2, 7, 7, 1, 2]
        get = get_max_runs(x)
        expect = np.array([[0, 1], [3, 5]])
        assert_array_equal(get, expect)

    def test_get_peak_indices(self):
        times = [0, 10, 20, 30, 31, 32, 40]
        counts = [7, 1, 2, 7, 7, 1, 2]
        get = get_peak_indices(times, counts)
        expect = [0, 1]
        assert_array_equal(get, expect)

if __name__ == '__main__':
    unittest.main()