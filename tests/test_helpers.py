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
if HAS_GEOPANDAS:
    from geopandas import GeoDataFrame



def test_time_to_seconds():
    timestr1 = '01:01:01'
    seconds1 = 3600 + 60 + 1
    timestr2 = '25:01:01'
    seconds2 = 25*3600 + 60 + 1
    assert timestr_to_seconds(timestr1) == seconds1
    assert timestr_to_seconds(seconds1, inverse=True) == timestr1
    assert timestr_to_seconds(seconds2, inverse=True) == timestr2
    assert timestr_to_seconds(timestr2, mod24=True) == seconds1
    assert timestr_to_seconds(seconds2, mod24=True, inverse=True) == timestr1
    # Test error handling
    assert np.isnan(timestr_to_seconds(seconds1))
    assert np.isnan(timestr_to_seconds(timestr1, inverse=True))

def test_time_mod24():
    timestr1 = '01:01:01'
    assert timestr_mod24(timestr1) == timestr1
    timestr2 = '25:01:01'
    assert timestr_mod24(timestr2) == timestr1

def test_datestr_to_date():
    datestr = '20140102'
    date = dt.date(2014, 1, 2)
    assert datestr_to_date(datestr) == date
    assert datestr_to_date(date, inverse=True) == datestr

def test_get_convert_dist():
    di = 'mi'
    do = 'km'
    f = get_convert_dist(di, do)
    assert f(1) == 1.609344

def test_get_segment_length():
    s = sg.LineString([(0, 0), (1, 0)])
    p = sg.Point((1/2, 0))
    assert get_segment_length(s, p) == 1/2
    q = sg.Point((1/3, 0))
    assert get_segment_length(s, p, q) == pytest.approx(1/6)
    p = sg.Point((0, 1/2))
    assert get_segment_length(s, p) == 0

def test_get_max_runs():
    x = [7, 1, 2, 7, 7, 1, 2]
    get = get_max_runs(x)
    expect = np.array([[0, 1], [3, 5]])
    assert_array_equal(get, expect)

def test_get_peak_indices():
    times = [0, 10, 20, 30, 31, 32, 40]
    counts = [7, 1, 2, 7, 7, 1, 2]
    get = get_peak_indices(times, counts)
    expect = [0, 1]
    assert_array_equal(get, expect)

def test_almost_equal():
    f = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    assert almost_equal(f, f)
    g = pd.DataFrame([[4, 3], [2, 1]], columns=['b', 'a'])
    assert almost_equal(f, g)
    h = pd.DataFrame([[1, 2], [5, 4]], columns=['a', 'b'])
    assert not almost_equal(f, h)
    h = pd.DataFrame()
    assert not almost_equal(f, h)

def test_is_not_null():
    f = None
    c = 'foo'
    assert not is_not_null(f, c)

    f = pd.DataFrame(columns=['bar', c])
    assert not is_not_null(f, c)

    f = pd.DataFrame([[1, np.nan]], columns=['bar', c])
    assert not is_not_null(f, c)

    f = pd.DataFrame([[1, np.nan], [2, 2]], columns=['bar', c])
    assert is_not_null(f, c)

@slow
def test_compute_route_stats_base():
    feed = cairns.copy()
    trip_stats = cairns_trip_stats
    for split_directions in [True, False]:
        rs = compute_route_stats_base(trip_stats, 
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
    rs = compute_route_stats_base(pd.DataFrame(), 
      split_directions=split_directions)    
    assert rs.empty

@slow
def test_compute_route_time_series_base():
    feed = cairns.copy()
    trip_stats = cairns_trip_stats
    for split_directions in [True, False]:
        rs = compute_route_stats_base(trip_stats, 
          split_directions=split_directions)
        rts = compute_route_time_series_base(trip_stats, 
          split_directions=split_directions, freq='1H')
        
        # Should be a data frame of the correct shape
        assert isinstance(rts, pd.core.frame.DataFrame)
        assert rts.shape[0] == 24
        assert rts.shape[1] == 5*rs.shape[0]
        
        # Should have correct column names
        if split_directions:
            expect = ['indicator', 'route_id', 'direction_id']
        else:
            expect = ['indicator', 'route_id']
        assert rts.columns.names == expect   
        
        # Each route have a correct service distance total
        if split_directions == False:
            g = trip_stats.groupby('route_id')
            for route in trip_stats['route_id'].values:
                get = rts['service_distance'][route].sum() 
                expect = g.get_group(route)['distance'].sum()
                assert abs((get - expect)/expect) < 0.001

    # Empty check
    rts = compute_route_time_series_base(pd.DataFrame(), 
      split_directions=split_directions, 
      freq='1H')    
    assert rts.empty

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_geometrize_stops():
    stops = cairns.stops.copy()
    geo_stops = geometrize_stops(stops)
    # Should be a GeoDataFrame
    assert isinstance(geo_stops, GeoDataFrame)
    # Should have the correct shape
    assert geo_stops.shape[0] == stops.shape[0]
    assert geo_stops.shape[1] == stops.shape[1] - 1
    # Should have the correct columns
    expect_cols = set(list(stops.columns) + ['geometry']) -\
      set(['stop_lon', 'stop_lat'])
    assert set(geo_stops.columns) == expect_cols

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_ungeometrize_stops():
    stops = cairns.stops.copy()
    geo_stops = geometrize_stops(stops)
    stops2 = ungeometrize_stops(geo_stops)
    # Test columns are correct
    assert set(stops2.columns) == set(stops.columns)
    # Data frames should be equal after sorting columns
    cols = sorted(stops.columns)
    assert_frame_equal(stops2[cols], stops[cols])

def test_compute_stop_stats_base():
    feed = cairns.copy()
    for split_directions in [True, False]:
        stops_stats = compute_stop_stats_base(feed.stop_times,
          feed.trips, split_directions=split_directions)
        # Should be a data frame
        assert isinstance(stops_stats, pd.core.frame.DataFrame)
        # Should contain the correct columns
        expect_cols = set([
          'stop_id',
          'num_routes',
          'num_trips',
          'max_headway',
          'min_headway',
          'mean_headway',
          'start_time',
          'end_time',
          ])
        if split_directions:
            expect_cols.add('direction_id')
        assert set(stops_stats.columns) == expect_cols
        # Should contain the correct stops
        expect_stops = set(feed.stops['stop_id'].values)
        get_stops = set(stops_stats['stop_id'].values)
        assert get_stops == expect_stops

    # Empty check
    stats = compute_stop_stats_base(feed.stop_times, pd.DataFrame())    
    assert stats.empty

@slow
def test_compute_stop_time_series_base():
    feed = cairns.copy()
    for split_directions in [True, False]:
        ss = compute_stop_stats_base(feed.stop_times, 
          feed.trips, split_directions=split_directions)
        sts = compute_stop_time_series_base(feed.stop_times, 
          feed.trips, freq='1H',
          split_directions=split_directions) 
        
        # Should be a data frame
        assert isinstance(sts, pd.core.frame.DataFrame)
        
        # Should have the correct shape
        assert sts.shape[0] == 24
        assert sts.shape[1] == ss.shape[0]
        
        # Should have correct column names
        if split_directions:
            expect = ['indicator', 'stop_id', 'direction_id']
        else:
            expect = ['indicator', 'stop_id']
        assert sts.columns.names == expect

        # Each stop should have a correct total trip count
        if split_directions == False:
            stg = feed.stop_times.groupby('stop_id')
            for stop in set(feed.stop_times['stop_id'].values):
                get = sts['num_trips'][stop].sum() 
                expect = stg.get_group(stop)['departure_time'].count()
                assert get == expect
    
    # Empty check
    stops_ts = compute_stop_time_series_base(feed.stop_times,
      pd.DataFrame(), freq='1H',
      split_directions=split_directions) 
    assert stops_ts.empty

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_geometrize_shapes():
    shapes = cairns.shapes.copy()
    geo_shapes = geometrize_shapes(shapes)
    # Should be a GeoDataFrame
    assert isinstance(geo_shapes, GeoDataFrame)
    # Should have the correct shape
    assert geo_shapes.shape[0] == shapes['shape_id'].nunique()
    assert geo_shapes.shape[1] == shapes.shape[1] - 2
    # Should have the correct columns
    expect_cols = set(list(shapes.columns) + ['geometry']) -\
      set(['shape_pt_lon', 'shape_pt_lat', 'shape_pt_sequence',
      'shape_dist_traveled'])
    assert set(geo_shapes.columns) == expect_cols

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_ungeometrize_shapes():
    shapes = cairns.shapes.copy()
    geo_shapes = geometrize_shapes(shapes)
    shapes2 = ungeometrize_shapes(geo_shapes)
    # Test columns are correct
    expect_cols = set(list(shapes.columns)) -\
      set(['shape_dist_traveled'])
    assert set(shapes2.columns) == expect_cols
    # Data frames should agree on certain columns
    cols = ['shape_id', 'shape_pt_lon', 'shape_pt_lat']
    assert_frame_equal(shapes2[cols], shapes[cols])