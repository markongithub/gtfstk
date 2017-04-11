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


def test_feed():
    feed = Feed(agency=pd.DataFrame(), dist_units='km')
    for key in cs.FEED_ATTRS:
        val = getattr(feed, key)
        if key == 'dist_units':
            assert val == 'km'
        elif key == 'agency':
            assert isinstance(val, pd.DataFrame)
        else:
            assert val is None

def test_eq():  
    assert Feed(dist_units='m') == Feed(dist_units='m')

    feed1 = Feed(dist_units='m', 
      stops=pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b']))
    assert feed1 == feed1

    feed2 = Feed(dist_units='m', 
      stops=pd.DataFrame([[4, 3], [2, 1]], columns=['b', 'a']))
    assert feed1 == feed2
    
    feed2 = Feed(dist_units='m',
      stops=pd.DataFrame([[3, 4], [2, 1]], columns=['b', 'a']))
    assert feed1 != feed2

    feed2 = Feed(dist_units='m', 
      stops=pd.DataFrame([[4, 3], [2, 1]], columns=['b', 'a']))
    assert feed1 == feed2

    feed2 = Feed(dist_units='mi', stops=feed1.stops)
    assert feed1 != feed2

def test_copy():
    feed1 = read_gtfs(DATA_DIR/'sample_gtfs.zip', dist_units='km')
    feed2 = feed1.copy()

    # Check attributes
    for key in cs.FEED_ATTRS:
        val = getattr(feed2, key)
        expect_val = getattr(feed1, key)            
        if isinstance(val, pd.DataFrame):
            assert_frame_equal(val, expect_val)
        elif isinstance(val, pd.core.groupby.DataFrameGroupBy):
            assert val.groups == expect_val.groups
        else:
            assert val == expect_val

def test_describe():
    feed = sample.copy() # No distances here
    a = feed.describe()
    assert isinstance(a, pd.DataFrame)
    assert set(a.columns) == set(['indicator', 'value'])

def test_assess():
    feed = sample.copy() # No distances here
    a = feed.assess()
    assert isinstance(a, pd.DataFrame)
    assert set(a.columns) == set(['indicator', 'value'])

def test_convert_dist():
    # Test with no distances
    feed1 = cairns.copy() # No distances here
    feed2 = feed1.convert_dist('mi')
    assert feed2.dist_units == 'mi'

    # Test with distances and identity conversion
    feed1 = feed1.append_dist_to_shapes() 
    feed2 = feed1.convert_dist(feed1.dist_units)
    assert feed1 == feed2

    # Test with proper conversion
    feed2 = feed1.convert_dist('m')
    assert_series_equal(feed2.shapes['shape_dist_traveled']/1000,
      feed1.shapes['shape_dist_traveled'])

def test_compute_feed_stats():
    feed = cairns.copy()
    date = cairns_date
    trip_stats = cairns_trip_stats
    f = feed.compute_feed_stats(trip_stats, date)
    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have the correct number of rows
    assert f.shape[0] == 1
    # Should contain the correct columns
    expect_cols = set([
      'num_trips',
      'num_routes',
      'num_stops',
      'peak_num_trips',
      'peak_start_time',
      'peak_end_time',
      'service_duration', 
      'service_distance',
      'service_speed',              
      ])
    assert set(f.columns) == expect_cols

    # Empty check
    f = feed.compute_feed_stats(trip_stats, '20010101')
    assert f.empty

def test_compute_feed_time_series():
    feed = cairns.copy()
    date = cairns_date
    trip_stats = cairns_trip_stats
    f = feed.compute_feed_time_series(trip_stats, date, freq='1H')
    # Should be a data frame 
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have the correct number of rows
    assert f.shape[0] == 24
    # Should have the correct columns
    expect_cols = set([
      'num_trip_starts',
      'num_trips',
      'service_distance',
      'service_duration',
      'service_speed',
      ])
    assert set(f.columns) == expect_cols

    # Empty check
    f = feed.compute_feed_time_series(trip_stats, '20010101')
    assert f.empty

def test_create_shapes():
    feed1 = cairns.copy()
    # Remove a trip shape
    trip_id = 'CNS2014-CNS_MUL-Weekday-00-4165878'
    feed1.trips.loc[feed1.trips['trip_id'] == trip_id, 'shape_id'] = np.nan
    feed2 = feed1.create_shapes()
    # Should create only 1 new shape
    assert len(set(feed2.shapes['shape_id']) - set(
      feed1.shapes['shape_id'])) == 1

    feed2 = feed1.create_shapes(all_trips=True)
    # Number of shapes should equal number of unique stop sequences
    st = feed1.stop_times.sort_values(['trip_id', 'stop_sequence'])
    stop_seqs = set([tuple(group['stop_id'].values)
      for __, group in st.groupby('trip_id')])
    assert feed2.shapes['shape_id'].nunique() == len(stop_seqs)

def test_compute_bounds():
    feed = cairns.copy() 
    minlon, minlat, maxlon, maxlat = feed.compute_bounds()
    # Bounds should be in the ball park
    assert 145 < minlon < 146
    assert 145 < maxlon < 146
    assert -18 < minlat < -15
    assert -18 < maxlat < -15

def test_compute_center():
    feed = cairns.copy() 
    centers = [feed.compute_center(), feed.compute_center(20)]
    bounds = feed.compute_bounds()
    for lon, lat in centers:
        # Center should be in the ball park
        assert bounds[0] < lon < bounds[2]
        assert bounds[1] < lat < bounds[3]

def test_restrict_by_routes():
    feed1 = cairns.copy() 
    route_ids = feed1.routes['route_id'][:2].tolist()
    feed2 = feed1.restrict_to_routes(route_ids)
    # Should have correct routes
    assert set(feed2.routes['route_id']) == set(route_ids)
    # Should have correct trips
    trip_ids = feed1.trips[feed1.trips['route_id'].isin(
      route_ids)]['trip_id']
    assert set(feed2.trips['trip_id']) == set(trip_ids)
    # Should have correct shapes
    shape_ids = feed1.trips[feed1.trips['trip_id'].isin(
      trip_ids)]['shape_id']
    assert set(feed2.shapes['shape_id']) == set(shape_ids)
    # Should have correct stops
    stop_ids = feed1.stop_times[feed1.stop_times['trip_id'].isin(
      trip_ids)]['stop_id']
    assert set(feed2.stop_times['stop_id']) == set(stop_ids)

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_restrict_to_polygon():
    feed1 = cairns.copy() 
    with (DATA_DIR/'cairns_square_stop_750070.geojson').open() as src:
        polygon = sg.shape(json.load(src)['features'][0]['geometry'])
    feed2 = feed1.restrict_to_polygon(polygon)
    # Should have correct routes
    rsns = ['120', '120N']
    assert set(feed2.routes['route_short_name']) == set(rsns)
    # Should have correct trips
    route_ids = feed1.routes[feed1.routes['route_short_name'].isin(
      rsns)]['route_id']
    trip_ids = feed1.trips[feed1.trips['route_id'].isin(
      route_ids)]['trip_id']
    assert set(feed2.trips['trip_id']) == set(trip_ids)
    # Should have correct shapes
    shape_ids = feed1.trips[feed1.trips['trip_id'].isin(
      trip_ids)]['shape_id']
    assert set(feed2.shapes['shape_id']) == set(shape_ids)
    # Should have correct stops
    stop_ids = feed1.stop_times[feed1.stop_times['trip_id'].isin(
      trip_ids)]['stop_id']
    assert set(feed2.stop_times['stop_id']) == set(stop_ids)

@slow
def test_compute_screen_line_counts():
    feed = cairns.copy() 
    date = cairns_date
    trip_stats = cairns_trip_stats
    feed = feed.append_dist_to_stop_times(trip_stats)
    
    # Load screen line
    with (DATA_DIR/'cairns_screen_line.geojson').open() as src:
        line = json.load(src)
        line = sg.shape(line['features'][0]['geometry'])
    
    f = feed.compute_screen_line_counts(line, date)

    # Should have correct columns
    expect_cols = set([
      'trip_id',
      'route_id',
      'route_short_name',
      'crossing_time',
      'orientation',
      ])
    assert set(f.columns) == expect_cols

    # Should have correct routes
    rsns = ['120', '120N']
    assert set(f['route_short_name']) == set(rsns)

    # Should have correct number of trips
    expect_num_trips = 34
    assert f['trip_id'].nunique() == expect_num_trips

    # Should have correct orientations
    for ori in [-1, 1]:
        assert f[f['orientation'] == ori].shape[0] == expect_num_trips

# --------------------------------------------
# Test functions about inputs and outputs
# --------------------------------------------
def test_read_gtfs():
    # Bad path
    with pytest.raises(ValueError):
        read_gtfs('bad_path!')
    
    # Bad dist_units:
    with pytest.raises(ValueError):
        read_gtfs(DATA_DIR/'sample_gtfs.zip',  dist_units='bingo')

    # Requires dist_units:
    with pytest.raises(ValueError):
        read_gtfs(path=DATA_DIR/'sample_gtfs.zip')

    # Success
    feed = read_gtfs(DATA_DIR/'sample_gtfs.zip',  dist_units='m')

    # Success
    feed = read_gtfs(DATA_DIR/'sample_gtfs',  dist_units='m')

def test_write_gtfs():
    feed1 = read_gtfs(DATA_DIR/'sample_gtfs.zip', dist_units='km')

    # Export feed1, import it as feed2, and then test equality
    for out_path in [DATA_DIR/'bingo.zip', DATA_DIR/'bingo']:
        write_gtfs(feed1, out_path)
        feed2 = read_gtfs(out_path, 'km')
        assert feed1 == feed2
        try:
            out_path.unlink()
        except:
            shutil.rmtree(str(out_path))

    # Test that integer columns with NaNs get output properly.
    # To this end, put a NaN, 1.0, and 0.0 in the direction_id column of trips.txt, export it, and import the column as strings.
    # Should only get np.nan, '0', and '1' entries.
    feed3 = read_gtfs(DATA_DIR/'sample_gtfs.zip', dist_units='km')
    f = feed3.trips.copy()
    f['direction_id'] = f['direction_id'].astype(object)
    f.loc[0, 'direction_id'] = np.nan
    f.loc[1, 'direction_id'] = 1.0
    f.loc[2, 'direction_id'] = 0.0
    feed3.trips = f
    q = DATA_DIR/'bingo.zip'
    write_gtfs(feed3, q)

    tmp_dir = tempfile.TemporaryDirectory()
    shutil.unpack_archive(str(q), tmp_dir.name, 'zip')
    qq = Path(tmp_dir.name)/'trips.txt'
    t = pd.read_csv(qq, dtype={'direction_id': str})
    assert t[~t['direction_id'].isin([np.nan, '0', '1'])].empty
    tmp_dir.cleanup()
    q.unlink()
