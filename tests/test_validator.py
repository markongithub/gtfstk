import pytest
import importlib
from pathlib import Path 

import pandas as pd 
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_array_equal
import utm
import shapely.geometry as sg 

from .context import gtfstk, slow, HAS_GEOPANDAS, DATA_DIR, sample, cairns, cairns_date, cairns_trip_stats
from gtfstk import *


def test_valid_int():
    assert valid_int(3, [-1, 3])
    assert not valid_int(3, [2])

def test_valid_color():
    assert valid_color('00FFFF')
    assert not valid_color('0FF')
    assert not valid_color('GGFFFF')

def test_valid_url():
    assert valid_url('http://www.example.com')
    assert not valid_url('www.example.com')

def test_valid_string():
    assert valid_string('hello3')
    assert not valid_string(np.nan)
    assert not valid_string(' ')

def test_check_table():
    feed = sample.copy()
    cond = feed.routes['route_id'].isnull() 
    assert not check_table([], feed.routes, cond, 'route_id', 'Bingo')
    assert check_table([], feed.routes, ~cond, 'route_id', 'Bongo')

def test_check_for_required_tables():
    assert not check_for_required_tables(sample)

    feed = sample.copy()
    feed.routes = None
    assert check_for_required_tables(feed)

def test_check_for_required_fields():
    assert not check_for_required_fields(sample)

    feed = sample.copy()
    del feed.routes['route_type']
    assert check_for_required_fields(feed)

def test_check_routes():
    assert not check_routes(sample)

    feed = sample.copy()
    feed.routes['route_id'].iat[0] = feed.routes['route_id'].iat[1]
    assert check_routes(feed)

    feed = sample.copy()
    feed.routes['agency_id'] = 'Hubba hubba'
    assert check_routes(feed)

    feed = sample.copy()
    feed.routes['route_short_name'].iat[0] = ''
    assert check_routes(feed)

    feed = sample.copy()
    feed.routes['route_short_name'].iat[0] = ''
    feed.routes['route_long_name'].iat[0] = ''
    assert check_routes(feed)

    feed = sample.copy()
    feed.routes['route_type'].iat[0] = 8
    assert check_routes(feed)

    feed = sample.copy()
    feed.routes['route_color'].iat[0] = 'FFF'
    assert check_routes(feed)

    feed = sample.copy()
    feed.routes['route_text_color'].iat[0] = 'FFF'
    assert check_routes(feed)

def test_validate():    
    assert not validate(sample)
