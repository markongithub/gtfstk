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


def test_get_routes():
    feed = cairns.copy()
    date = cairns_date
    f = feed.get_routes(date)
    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert f.shape[0] <= feed.routes.shape[0]
    assert f.shape[1] == feed.routes.shape[1]
    # Should have correct columns
    assert set(f.columns) == set(feed.routes.columns)

    g = feed.get_routes(date, "07:30:00")
    # Should be a data frame
    assert isinstance(g, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert g.shape[0] <= f.shape[0]
    assert g.shape[1] == f.shape[1]
    # Should have correct columns
    assert set(g.columns) == set(feed.routes.columns)

@slow
def test_compute_route_stats():
    feed = cairns.copy()
    date = cairns_date
    trip_stats = cairns_trip_stats
    f = pd.merge(trip_stats, feed.get_trips(date))
    for split_directions in [True, False]:
        rs = feed.compute_route_stats(trip_stats, date, 
          split_directions=split_directions)

        # Should be a data frame of the correct shape
        assert isinstance(rs, pd.core.frame.DataFrame)
        if split_directions:
            max_num_routes = 2*feed.routes.shape[0]
        else:
            max_num_routes = feed.routes.shape[0]
            
        assert rs.shape[0] <= max_num_routes

        # Should contain the correct columns
        expect_cols = set([
          'route_id',
          'route_short_name',
          'route_type',
          'num_trips',
          'is_bidirectional',
          'is_loop',
          'start_time',
          'end_time',
          'max_headway',
          'min_headway',
          'mean_headway', 
          'peak_num_trips',
          'peak_start_time',
          'peak_end_time',
          'service_duration', 
          'service_distance',
          'service_speed',              
          'mean_trip_distance',
          'mean_trip_duration',
          ])
        if split_directions:
            expect_cols.add('direction_id')
        assert set(rs.columns) == expect_cols

    # Empty check
    f = feed.compute_route_stats(trip_stats, '20010101')
    assert f.empty

@slow
def test_compute_route_time_series():
    feed = cairns.copy()
    date = cairns_date
    trip_stats = cairns_trip_stats
    ats = pd.merge(trip_stats, feed.get_trips(date))
    for split_directions in [True, False]:
        f = feed.compute_route_stats(trip_stats, date, 
          split_directions=split_directions)
        rts = feed.compute_route_time_series(trip_stats, date, 
          split_directions=split_directions, freq='1H')
        
        # Should be a data frame of the correct shape
        assert isinstance(rts, pd.core.frame.DataFrame)
        assert rts.shape[0] == 24
        assert rts.shape[1] == 5*f.shape[0]
        
        # Should have correct column names
        if split_directions:
            expect = ['indicator', 'route_id', 'direction_id']
        else:
            expect = ['indicator', 'route_id']
        assert rts.columns.names, expect
        
        # Each route have a correct service distance total
        if split_directions == False:
            atsg = ats.groupby('route_id')
            for route in ats['route_id'].values:
                get = rts['service_distance'][route].sum() 
                expect = atsg.get_group(route)['distance'].sum()
                assert abs((get - expect)/expect) < 0.001

    # Empty check
    date = '19000101'
    rts = feed.compute_route_time_series(trip_stats, date, 
      split_directions=split_directions, freq='1H')
    assert rts.empty

def test_get_route_timetable():
    feed = cairns.copy()
    route_id = feed.routes['route_id'].values[0]
    date = cairns_date
    f = feed.get_route_timetable(route_id, date)
    # Should be a data frame 
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have the correct columns
    expect_cols = set(feed.trips.columns) |\
      set(feed.stop_times.columns)
    assert set(f.columns) == expect_cols

def test_route_to_geojson():
    feed = cairns.copy()
    route_id = feed.routes['route_id'].values[0]
    g0 = feed.route_to_geojson(route_id)      
    g1 = feed.route_to_geojson(route_id, include_stops=True)
    for g in [g0, g1]:
        # Should be a dictionary
        assert isinstance(g, dict)

    # Should have the correct number of features
    assert len(g0['features']) == 1
    stop_ids = feed.get_stops(route_id=route_id)['stop_id'].values
    assert len(g1['features']) == 1 + len(stop_ids)
