import unittest
from copy import copy

import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
from shapely.geometry import Point, LineString, mapping
from shapely.geometry import shape as sh_shape

from gtfstk.calculator import *
from gtfstk.utilities import *

# Load test feeds
cairns = read_gtfs('data/cairns_gtfs.zip')

class TestCleaner(unittest.TestCase):

    def test_clean_stop_times(self):
        pass

    def test_clean_routes(self):
        """
        Mostly tested already via ``test_clean_series()``.
        """
        pass


if __name__ == '__main__':
    unittest.main()