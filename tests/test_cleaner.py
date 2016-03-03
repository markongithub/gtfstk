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


if __name__ == '__main__':
    unittest.main()