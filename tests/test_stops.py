import pytest
import importlib
from pathlib import Path 

import pandas as pd 
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np
import utm
import shapely.geometry as sg 

from .context import gtfstk, slow, HAS_GEOPANDAS, DATA_DIR, sample, cairns, cairns_date, cairns_trip_stats
from gtfstk import *
if HAS_GEOPANDAS:
    from geopandas import GeoDataFrame


def test_get_stops():
    feed = cairns.copy()
    date = cairns_date
    trip_id = feed.trips['trip_id'].iat[0]
    route_id = feed.routes['route_id'].iat[0]
    frames = [
      feed.get_stops(), 
      feed.get_stops(date=date),
      feed.get_stops(trip_id=trip_id),
      feed.get_stops(route_id=route_id),
      feed.get_stops(date=date, trip_id=trip_id),
      feed.get_stops(date=date, route_id=route_id),
      feed.get_stops(date=date, trip_id=trip_id, route_id=route_id),
      ]
    for f in frames:
        # Should be a data frame
        assert isinstance(f, pd.core.frame.DataFrame)
        # Should have the correct shape
        assert f.shape[0] <= feed.stops.shape[0]
        assert f.shape[1] == feed.stops.shape[1]
        # Should have correct columns
        set(f.columns) == set(feed.stops.columns)
    # Number of rows should be reasonable
    assert frames[0].shape[0] <= frames[1].shape[0]
    assert frames[2].shape[0] <= frames[4].shape[0]
    assert frames[4].shape == frames[6].shape

def test_build_geometry_by_stop():
    feed = cairns.copy()
    stop_ids = feed.stops['stop_id'][:2].values
    d0 = feed.build_geometry_by_stop()
    d1 = feed.build_geometry_by_stop(stop_ids=stop_ids)
    for d in [d0, d1]:
        # Should be a dictionary
        assert isinstance(d, dict)
        # The first key should be a valid shape ID
        assert list(d.keys())[0] in feed.stops['stop_id'].values
        # The first value should be a Shapely linestring
        assert isinstance(list(d.values())[0], sg.Point)
    # Lengths should be right
    assert len(d0) == feed.stops['stop_id'].nunique()
    assert len(d1) == len(stop_ids)

def test_compute_stop_activity():
    feed = cairns.copy()
    dates = feed.get_first_week()
    stop_activity = feed.compute_stop_activity(dates)
    # Should be a data frame
    assert isinstance(stop_activity, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert stop_activity.shape[0] == feed.stops.shape[0]
    assert stop_activity.shape[1] == len(dates) + 1
    # Date columns should contain only zeros and ones
    assert set(stop_activity[dates].values.flatten()) == {0, 1}

def test_compute_stop_stats():
    feed = cairns.copy()
    date = cairns_date
    stop_stats = feed.compute_stop_stats(date)
    # Should be a data frame
    assert isinstance(stop_stats, pd.core.frame.DataFrame)
    # Should contain the correct stops
    get = set(stop_stats['stop_id'].values)
    f = feed.get_stops(date)
    expect = set(f['stop_id'].values)
    assert get == expect
    
    # Empty check
    f = feed.compute_stop_stats('20010101')
    assert f.empty

@slow
def test_compute_stop_time_series():
    feed = cairns.copy()
    date = cairns_date
    ast = pd.merge(feed.get_trips(date), feed.stop_times)
    for split_directions in [True, False]:
        f = feed.compute_stop_stats(date, 
          split_directions=split_directions)
        ts = feed.compute_stop_time_series(date, freq='1H',
          split_directions=split_directions) 
        
        # Should be a data frame
        assert isinstance(ts, pd.core.frame.DataFrame)
        
        # Should have the correct shape
        assert ts.shape[0] == 24
        assert ts.shape[1] == f.shape[0]
        
        # Should have correct column names
        if split_directions:
            expect = ['indicator', 'stop_id', 'direction_id']
        else:
            expect = ['indicator', 'stop_id']
        assert ts.columns.names == expect

        # Each stop should have a correct total trip count
        if split_directions == False:
            astg = ast.groupby('stop_id')
            for stop in set(ast['stop_id'].values):
                get = ts['num_trips'][stop].sum() 
                expect = astg.get_group(stop)['departure_time'].count()
                assert get == expect
    
    # Empty check
    date = '19000101'
    ts = feed.compute_stop_time_series(date, freq='1H',
      split_directions=split_directions) 
    assert ts.empty

def test_get_stop_timetable():
    feed = cairns.copy()
    stop = feed.stops['stop_id'].values[0]
    date = cairns_date
    f = feed.get_stop_timetable(stop, date)
    # Should be a data frame 
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have the correct columns
    expect_cols = set(feed.trips.columns) |\
      set(feed.stop_times.columns)
    assert set(f.columns) == expect_cols

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_get_stops_in_polygon():
    feed = cairns.copy()
    with (DATA_DIR/'cairns_square_stop_750070.geojson').open() as src:
        polygon = sg.shape(json.load(src)['features'][0]['geometry'])
    pstops = feed.get_stops_in_polygon(polygon)
    stop_ids = ['750070']
    assert pstops['stop_id'].values == stop_ids
