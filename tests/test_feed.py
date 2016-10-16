import unittest
from types import FunctionType
from pathlib import Path 

import pandas as pd 
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np 

import gtfstk.constants as cs
from gtfstk.feed import *


DATA_DIR = Path('data')

class TestFeed(unittest.TestCase):

    def test_feed(self):
        feed = Feed(agency=pd.DataFrame(), dist_units='km')
        for key in cs.FEED_ATTRS:
            val = getattr(feed, key)
            if key == 'dist_units':
                self.assertEqual(val, 'km')
            elif key == 'agency':
                self.assertIsInstance(val, pd.DataFrame)
            else:
                self.assertIsNone(val)

    def test_eq(self):  
        self.assertEqual(Feed(dist_units='m'), Feed(dist_units='m'))

        feed1 = Feed(dist_units='m', 
          stops=pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b']))
        self.assertEqual(feed1, feed1)

        feed2 = Feed(dist_units='m', 
          stops=pd.DataFrame([[4, 3], [2, 1]], columns=['b', 'a']))
        self.assertEqual(feed1, feed2)
        
        feed2 = Feed(dist_units='m',
          stops=pd.DataFrame([[3, 4], [2, 1]], columns=['b', 'a']))
        self.assertNotEqual(feed1, feed2)
    
        feed2 = Feed(dist_units='m', 
          stops=pd.DataFrame([[4, 3], [2, 1]], columns=['b', 'a']))
        self.assertEqual(feed1, feed2)

        feed2 = Feed(dist_units='mi', stops=feed1.stops)
        self.assertNotEqual(feed1, feed2)

    def test_copy(self):
        feed1 = read_gtfs(DATA_DIR/'sample_gtfs.zip', dist_units='km')
        feed2 = feed1.copy()

        # Check attributes
        for key in cs.FEED_ATTRS:
            val = getattr(feed2, key)
            expect_val = getattr(feed1, key)            
            if isinstance(val, pd.DataFrame):
                assert_frame_equal(val, expect_val)
            elif isinstance(val, pd.core.groupby.DataFrameGroupBy):
                self.assertEqual(val.groups, expect_val.groups)
            else:
                self.assertEqual(val, expect_val)

    # --------------------------------------------
    # Test functions about inputs and outputs
    # --------------------------------------------
    def test_read_gtfs(self):
        # Bad path
        self.assertRaises(ValueError, read_gtfs,
          path='bad_path!')
        
        # Bad dist_units:
        self.assertRaises(ValueError, read_gtfs, 
          path=DATA_DIR/'sample_gtfs.zip',  dist_units='bingo')

        # Requires dist_units:
        self.assertRaises(ValueError, read_gtfs,
          path=DATA_DIR/'sample_gtfs.zip')

        # Success
        feed = read_gtfs(DATA_DIR/'sample_gtfs.zip',  dist_units='m')

    def test_write_gtfs(self):
        feed1 = read_gtfs(DATA_DIR/'sample_gtfs.zip', dist_units='km')

        # Export feed1, import it as feed2, and then test that the
        # attributes of the two feeds are equal.
        path = DATA_DIR/'test_gtfs.zip'
        write_gtfs(feed1, path)
        feed2 = read_gtfs(path, dist_units='km')
        names = cs.GTFS_TABLES_REQUIRED + cs.GTFS_TABLES_OPTIONAL
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
        feed3 = read_gtfs(DATA_DIR/'sample_gtfs.zip', dist_units='km')
        f = feed3.trips.copy()
        f['direction_id'] = f['direction_id'].astype(object)
        f.loc[0, 'direction_id'] = np.nan
        f.loc[1, 'direction_id'] = 1.0
        f.loc[2, 'direction_id'] = 0.0
        feed3.trips = f
        write_gtfs(feed3, path)
        archive = zipfile.ZipFile(str(path))
        dir_name = Path(path.stem) #rstrip('.zip') + '/'
        archive.extractall(str(dir_name))
        t = pd.read_csv(dir_name/'trips.txt', dtype={'direction_id': str})
        self.assertTrue(t[~t['direction_id'].isin([np.nan, '0', '1'])].empty)
        
        # Clean up
        shutil.rmtree(str(dir_name))
        path.unlink()

