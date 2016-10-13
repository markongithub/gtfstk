import unittest
from copy import copy
from pathlib import Path

import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
from shapely.geometry import Point, LineString, mapping
from shapely.geometry import shape as sh_shape

from gtfstk.constants import *
from gtfstk.utilities import *
from gtfstk.feed import *
from gtfstk.cleaner import *

DATA_DIR = Path('data')
cairns = read_gtfs(DATA_DIR/'cairns_gtfs.zip')


class TestCleaner(unittest.TestCase):

    def test_clean_route_ids(self):
        f1 = cairns.copy()
        f1.routes.ix[0, 'route_id'] = '  ho   ho ho '
        f2 = clean_route_ids(f1)
        expect_rid = 'ho_ho_ho'
        self.assertEqual(f2.routes.ix[0, 'route_id'], expect_rid)

    def test_clean_route_short_names(self):
        f1  = cairns.copy()
        
        # Should have no effect on a fine feed
        f2 = clean_route_short_names(f1)
        assert_series_equal(f2.routes['route_short_name'], 
          f1.routes['route_short_name'])
        
        # Make route short name duplicates
        f1.routes.loc[1:5, 'route_short_name'] = np.nan
        f1.routes.loc[6:, 'route_short_name'] = '  he llo  '
        f2 = clean_route_short_names(f1)
        # Should have unique route short names
        self.assertEqual(f2.routes['route_short_name'].nunique(), 
          f2.routes.shape[0])
        # NaNs should be replaced by n/a and route IDs
        expect_rsns = ('n/a-' + cairns.routes.ix[1:5]['route_id']).tolist()
        self.assertEqual(f2.routes.ix[1:5]['route_short_name'].values.tolist(),
          expect_rsns)
        # Should have names without leading or trailing whitespace
        self.assertFalse(f2.routes['route_short_name'].str.startswith(' ').any())
        self.assertFalse(f2.routes['route_short_name'].str.endswith(' ').any())

    def test_prune_dead_routes(self):
        # Should not change Cairns routes
        f1 = cairns.copy()
        f2 = prune_dead_routes(f1)
        assert_frame_equal(f2.routes, f1.routes)

        # Create a dummy route which should be removed
        g = pd.DataFrame([[0 for c in f1.routes.columns]], 
          columns=f1.routes.columns)
        f3 = f1.copy()
        f3.routes = pd.concat([f3.routes, g])
        f4 = prune_dead_routes(f3)
        assert_frame_equal(f4.routes, f1.routes)

    def test_clean(self):
        f1 = cairns.copy()
        rid = f1.routes.ix[0, 'route_id']
        f1.routes.ix[0, 'route_id'] = ' ' + rid + '   '
        f2 = clean(f1)
        self.assertEqual(f2.routes.ix[0, 'route_id'], rid)
        assert_frame_equal(f2.trips, cairns.trips)

    def test_aggregate_routes(self):
        feed1 = cairns.copy()
        # Equalize all route short names
        feed1.routes['route_short_name'] = 'bingo'
        feed2 = aggregate_routes(feed1)

        # feed2 should have only one route ID
        self.assertEqual(feed2.routes.shape[0], 1)
        
        # Feeds should have same trip data frames excluding
        # route IDs
        feed1.trips['route_id'] = feed2.trips['route_id']
        self.assertTrue(ut.almost_equal(feed1.trips, feed2.trips))

        # Feeds should have equal attributes excluding
        # routes and trips data frames
        feed2.routes = feed1.routes 
        feed2.trips = feed1.trips
        self.assertEqual(feed1, feed2)

    def test_drop_invalid_columns(self):
        f1 = cairns.copy()
        f1.routes['bingo'] = 'bongo'
        f1.trips['wingo'] = 'wongo'
        f2 = drop_invalid_columns(f1)
        self.assertEqual(f2, cairns)


if __name__ == '__main__':
    unittest.main()