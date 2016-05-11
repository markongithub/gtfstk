import unittest
from copy import copy

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

cairns = read_gtfs('data/cairns_gtfs.zip')


class TestCleaner(unittest.TestCase):

    def test_clean_route_short_names(self):
        feed  = copy(cairns)
        feed.routes.loc[1:5, 'route_short_name'] = np.nan
        feed.routes.loc[6:, 'route_short_name'] = 'hello'
        routes = clean_route_short_names(feed)
        # Should have unique route short names
        self.assertEqual(routes['route_short_name'].nunique(), routes.shape[0])
        # NaNs should be replaced by route IDs
        self.assertEqual(routes.ix[1:5]['route_short_name'].values.tolist(),
          cairns.routes.ix[1:5]['route_id'].values.tolist())

    def test_prune_dead_routes(self):
        # Should not change Cairns routes
        old_routes = cairns.routes
        new_routes = prune_dead_routes(cairns)
        assert_frame_equal(new_routes, old_routes)

        # Create a dummy route which should be removed
        f = pd.DataFrame([[0 for c in old_routes.columns]], 
          columns=old_routes.columns)
        feed = copy(cairns)
        feed.routes = pd.concat([old_routes, f])
        new_routes = prune_dead_routes(feed)
        assert_frame_equal(new_routes, old_routes)

    def test_aggregate_routes(self):
        feed1 = copy(cairns)
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


if __name__ == '__main__':
    unittest.main()