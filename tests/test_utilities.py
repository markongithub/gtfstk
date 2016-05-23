import unittest
from copy import copy
from pathlib import Path

import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
from shapely.geometry import Point, LineString, mapping
from shapely.geometry import shape as sh_shape

from gtfstk.feed import read_gtfs
from gtfstk.utilities import *

# Load test feeds
DATA_DIR = Path('data')
cairns = read_gtfs(DATA_DIR/'cairns_gtfs.zip')

class TestUtilities(unittest.TestCase):
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

    def test_almost_equal(self):
        f = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
        self.assertTrue(almost_equal(f, f))
        g = pd.DataFrame([[4, 3], [2, 1]], columns=['b', 'a'])
        self.assertTrue(almost_equal(f, g))
        h = pd.DataFrame([[1, 2], [5, 4]], columns=['a', 'b'])
        self.assertFalse(almost_equal(f, h))
        h = pd.DataFrame()
        self.assertFalse(almost_equal(f, h))

    def test_is_not_null(self):
        f = None
        c = 'foo'
        self.assertFalse(is_not_null(f, c))

        f = pd.DataFrame(columns=['bar', c])
        self.assertFalse(is_not_null(f, c))

        f = pd.DataFrame([[1, np.nan]], columns=['bar', c])
        self.assertFalse(is_not_null(f, c))

        f = pd.DataFrame([[1, np.nan], [2, 2]], columns=['bar', c])
        self.assertTrue(is_not_null(f, c))

if __name__ == '__main__':
    unittest.main()