import unittest
from types import FunctionType

import pandas as pd 

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

