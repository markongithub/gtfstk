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


def test_valid_color():
    assert valid_color('00FFFF')
    assert not valid_color('0FF')
    assert not valid_color('GGFFFF')

def test_valid_url():
    assert valid_url('http://www.example.com')
    assert not valid_url('www.example.com')

def test_valid_str():
    assert valid_str('hello3')
    assert not valid_str(np.nan)
    assert not valid_str(' ')

def test_valid_timezone():
    assert valid_timezone('Africa/Abidjan')
    assert not valid_timezone('zoom')

def test_check_table():
    feed = sample.copy()
    cond = feed.routes['route_id'].isnull() 
    assert not check_table([], 'routes', feed.routes, cond, 'Bingo')
    assert check_table([], 'routes', feed.routes, ~cond, 'Bongo')

def test_check_field_id():
    feed = sample.copy()
    assert not check_field_id([], 'routes', feed.routes, 'route_id')
    feed.routes['route_id'].iat[0] = np.nan
    assert check_field_id([], 'routes', feed.routes, 'route_id')

def test_check_field_str():
    feed = sample.copy()
    assert not check_field_str([], 'routes', feed.routes, 'route_desc', False)
    feed.routes['route_desc'].iat[0] = ''
    assert check_field_str([], 'routes', feed.routes, 'route_desc', False)

def test_check_field_url():
    feed = sample.copy()
    assert not check_field_url([], 'agency', feed.agency, 'agency_url', True)
    feed.agency['agency_url'].iat[0] = 'example.com'
    assert check_field_url([], 'agency', feed.agency, 'agency_url', True)

def test_check_field_range():
    feed = sample.copy()
    assert not check_field_range([], 'routes', feed.routes, 'route_type', True, 
      range(19))
    feed.routes['route_type'].iat[0] = 7
    assert check_field_range([], 'routes', feed.routes, 'route_type', True, 
      range(2))

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

def test_check_stops():
    assert not check_stops(sample)

    feed = sample.copy()
    feed.stops['stop_id'].iat[0] = feed.stops['stop_id'].iat[1]
    assert check_stops(feed)

    for field in ['stop_code', 'stop_desc', 'zone_id', 'parent_station']:
        feed = sample.copy()
        feed.stops[field] = ''
        assert check_stops(feed)    

    for field in ['stop_url', 'stop_timezone']:
        feed = sample.copy()
        feed.stops[field] = 'Wa wa'
        assert check_stops(feed)

    for field in ['stop_lon', 'stop_lat', 'location_type', 'wheelchair_boarding']:
        feed = sample.copy()
        feed.stops[field] = 185
        assert check_stops(feed)

    feed = sample.copy()
    feed.stops['location_type'] = 1
    feed.stops['parent_station'] = 'bingo' 
    assert check_stops(feed)

    feed = sample.copy()
    feed.stops['location_type'] = 0
    feed.stops['parent_station'] = feed.stops['stop_id'].iat[1]
    assert check_stops(feed)


def test_check_trips():
    assert not check_trips(sample)

    feed = sample.copy()
    feed.trips['trip_id'].iat[0] = feed.trips['trip_id'].iat[1]
    assert check_trips(feed)

    feed = sample.copy()
    feed.trips['route_id'] = 'Hubba hubba'
    assert check_trips(feed)

    feed = sample.copy()
    feed.trips['service_id'] = 'Boom boom'
    assert check_trips(feed)

    feed = sample.copy()
    feed.trips['direction_id'].iat[0] = 7
    assert check_trips(feed)

    feed = sample.copy()
    feed.trips['block_id'].iat[0] = ''
    assert check_trips(feed)

    feed = sample.copy()
    feed.trips['block_id'].iat[0] = 'Bam'
    assert check_trips(feed)

    feed = sample.copy()
    feed.trips['shape_id'].iat[0] = 'Hello'
    assert check_trips(feed)

    feed = sample.copy()
    feed.trips['wheelchair_accessible'] = ''
    assert check_trips(feed)

def test_validate():    
    assert not validate(sample)
