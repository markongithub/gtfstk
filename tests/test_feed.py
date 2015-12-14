import unittest
from types import FunctionType

import pandas as pd 
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np 

from gtfstk.feed import *


class TestFeed(unittest.TestCase):

    def test_feed(self):
        feed = Feed(agency=pd.DataFrame())
        for key, value in feed.__dict__.items():
            if key in ['dist_units_in', 'dist_units_out']:
                self.assertEqual(value, 'km')
            elif key == 'convert_dist':
                self.assertIsInstance(value, FunctionType)
            elif key == 'agency':
                self.assertIsInstance(value, pd.DataFrame)
            else:
                self.assertIsNone(value)

    # --------------------------------------------
    # Test functions about inputs and outputs
    # --------------------------------------------
    def test_read_gtfs(self):
        # Bad path
        self.assertRaises(ValueError, read_gtfs,
          path='bad_path!')
        
        feed = read_gtfs('data/cairns_gtfs.zip')

        # Bad dist_units_in:
        self.assertRaises(ValueError, read_gtfs, 
          path='data/cairns_gtfs.zip',  
          dist_units_in='bingo')

        # Requires dist_units_in:
        self.assertRaises(ValueError, read_gtfs,
          path='data/portland_gtfs.zip')

    def test_write_gtfs(self):
        feed1 = read_gtfs('data/cairns_gtfs.zip')

        # Export feed1, import it as feed2, and then test that the
        # attributes of the two feeds are equal.
        path = 'data/test_gtfs.zip'
        write_gtfs(feed1, path)
        feed2 = read_gtfs(path)
        names = cs.REQUIRED_GTFS_FILES + cs.OPTIONAL_GTFS_FILES
        for name in names:
            f1 = getattr(feed1, name)
            f2 = getattr(feed2, name)
            if f1 is None:
                self.assertIsNone(f2)
            else:
                assert_frame_equal(f1, f2)

        # Test that integer columns with NaNs get output properly.
        # To this end, put a NaN, 1.0, and 0.0 in the direction_id column 
        # of trips.txt, export it, and import the column as strings.
        # Should only get np.nan, '0', and '1' entries.
        feed3 = read_gtfs('data/cairns_gtfs.zip')
        f = feed3.trips.copy()
        f['direction_id'] = f['direction_id'].astype(object)
        f.loc[0, 'direction_id'] = np.nan
        f.loc[1, 'direction_id'] = 1.0
        f.loc[2, 'direction_id'] = 0.0
        feed3.trips = f
        write_gtfs(feed3, path)
        archive = zipfile.ZipFile(path)
        dir_name = path.rstrip('.zip') + '/'
        archive.extractall(dir_name)
        t = pd.read_csv(dir_name + 'trips.txt', dtype={'direction_id': str})
        self.assertTrue(t[~t['direction_id'].isin([np.nan, '0', '1'])].empty)
        
        # Remove extracted directory
        shutil.rmtree(dir_name)

    # --------------------------------------------
    # Test functions about basics
    # --------------------------------------------
    def test_copy(self):
        feed1 = read_gtfs('data/cairns_gtfs.zip')
        feed2 = copy(feed1)
        for key, value in feed2.__dict__.items():
            expect_value = getattr(feed1, key)            
            if isinstance(value, pd.DataFrame):
                assert_frame_equal(value, expect_value)
            elif isinstance(value, pd.core.groupby.DataFrameGroupBy):
                self.assertEqual(value.groups, expect_value.groups)
            elif isinstance(value, FunctionType):
                # No need to check this
                continue
            else:
                self.assertEqual(value, expect_value)
