from __future__ import division
import unittest
import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal

from feed import *

# Create your tests here.
class TestFeed(unittest.TestCase):

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

    def test_get_first_workweek(self):
        feed = Feed('test/darwin_20130903/')
        dates = feed.get_first_workweek()
        d1 = dt.date(2012, 11, 5)
        d2 = dt.date(2012, 11, 9)
        self.assertEqual(dates[0], d1)
        self.assertEqual(dates[-1], d2)
        self.assertEqual(len(dates), 5)

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

if __name__ == '__main__':
    unittest.main()