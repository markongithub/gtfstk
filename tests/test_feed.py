import pytest
import importlib
from pathlib import Path 

import pandas as pd 
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np
import utm
import shapely.geometry as sg 

from .context import gtfstk, slow
from gtfstk import *


# Check if GeoPandas is installed
loader = importlib.find_loader('geopandas')
if loader is None:
    HAS_GEOPANDAS = False
else:
    HAS_GEOPANDAS = True
    from geopandas import GeoDataFrame

# Load/create test feeds
DATA_DIR = Path('data')
sample = read_gtfs(DATA_DIR/'sample_gtfs.zip', dist_units='km')
cairns = read_gtfs(DATA_DIR/'cairns_gtfs.zip', dist_units='km')
cairns_shapeless = cairns.copy()
cairns_shapeless.shapes = None
t = cairns_shapeless.trips
t['shape_id'] = np.nan
cairns_shapeless.trips = t
cairns_date = cairns.get_first_week()[0]
cairns_trip_stats = pd.read_csv(DATA_DIR/'cairns_trip_stats.csv', dtype=DTYPE)

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

# --------------------------------------------
# Test methods about calendars
# --------------------------------------------
def test_get_dates():
    feed = cairns.copy()
    for as_date_obj in [True, False]:
        dates = feed.get_dates(as_date_obj=as_date_obj)
        d1 = '20140526'
        d2 = '20141228'
        if as_date_obj:
            d1 = hp.datestr_to_date(d1)
            d2 = hp.datestr_to_date(d2)
            assert len(dates) == (d2 - d1).days + 1
        assert dates[0] == d1
        assert dates[-1] == d2

def test_get_first_week():
    feed = cairns.copy()
    dates = feed.get_first_week()
    d1 = '20140526'
    d2 = '20140601'
    assert dates[0] == d1
    assert dates[-1] == d2
    assert len(dates) == 7

# --------------------------------------------
# Test methods about trips
# --------------------------------------------
def test_is_active_trip():
    feed = cairns.copy()
    trip_id = 'CNS2014-CNS_MUL-Weekday-00-4165878'
    date1 = '20140526'
    date2 = '20120322'
    assert feed.is_active_trip(trip_id, date1)
    assert not feed.is_active_trip(trip_id, date2)

    trip_id = 'CNS2014-CNS_MUL-Sunday-00-4165971'
    date1 = '20140601'
    date2 = '20120602'
    assert feed.is_active_trip(trip_id, date1)
    assert not feed.is_active_trip(trip_id, date2)

def test_get_trips():
    feed = cairns.copy()
    date = cairns_date
    trips1 = feed.get_trips(date)
    # Should be a data frame
    assert isinstance(trips1, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips1.shape[0] <= feed.trips.shape[0]
    assert trips1.shape[1] == feed.trips.shape[1]
    # Should have correct columns
    assert set(trips1.columns) == set(feed.trips.columns)

    trips2 = feed.get_trips(date, "07:30:00")
    # Should be a data frame
    assert isinstance(trips2, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips2.shape[0] <= trips2.shape[0]
    assert trips2.shape[1] == trips1.shape[1]
    # Should have correct columns
    assert set(trips2.columns) == set(feed.trips.columns)

def test_compute_trip_activity():
    feed = cairns.copy()
    dates = feed.get_first_week()
    trips_activity = feed.compute_trip_activity(dates)
    # Should be a data frame
    assert isinstance(trips_activity, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips_activity.shape[0] == feed.trips.shape[0]
    assert trips_activity.shape[1] == 1 + len(dates)
    # Date columns should contain only zeros and ones
    assert set(trips_activity[dates].values.flatten()) == {0, 1}

def test_compute_busiest_date():
    feed = cairns.copy()
    dates = feed.get_first_week() + ['19000101']
    date = feed.compute_busiest_date(dates)
    # Busiest day should lie in first week
    assert date in dates

@slow
def test_compute_trip_stats():
    feed = cairns.copy()
    trip_stats = feed.compute_trip_stats()
    
    # Should be a data frame with the correct number of rows
    assert isinstance(trip_stats, pd.core.frame.DataFrame)
    assert trip_stats.shape[0] == feed.trips.shape[0]
    
    # Should contain the correct columns
    expect_cols = set([
      'trip_id',
      'direction_id',
      'route_id',
      'route_short_name',
      'route_type',
      'shape_id',
      'num_stops',
      'start_time', 
      'end_time',
      'start_stop_id',
      'end_stop_id',
      'distance',
      'duration',
      'speed',
      'is_loop',
      ])
    assert set(trip_stats.columns) == expect_cols
    
    # Shapeless feeds should have null entries for distance column
    feed2 = cairns_shapeless.copy()
    trip_stats = feed2.compute_trip_stats()
    assert len(trip_stats['distance'].unique()) == 1
    assert np.isnan(trip_stats['distance'].unique()[0])  
    
    # Should contain the correct trips
    get_trips = set(trip_stats['trip_id'].values)
    expect_trips = set(feed.trips['trip_id'].values)
    assert get_trips == expect_trips

@slow
def test_compute_trip_locations():
    feed = cairns.copy()
    trip_stats = cairns_trip_stats
    feed = feed.append_dist_to_stop_times(trip_stats)
    date = cairns_date
    times = ['08:00:00']
    f = feed.compute_trip_locations(date, times)
    g = feed.get_trips(date, times[0])
    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have the correct number of rows
    assert f.shape[0] == g.shape[0]
    # Should have the correct columns
    expect_cols = set([
      'route_id',
      'trip_id',
      'direction_id',
      'shape_id',
      'time', 
      'rel_dist', 
      'lon', 
      'lat',
      ])
    assert set(f.columns) == expect_cols

def test_trip_to_geojson():
    feed = cairns.copy()
    trip_id = feed.trips['trip_id'].values[0]
    g0 = feed.trip_to_geojson(trip_id)      
    g1 = feed.trip_to_geojson(trip_id, include_stops=True)
    for g in [g0, g1]:
        # Should be a dictionary
        assert isinstance(g, dict)

    # Should have the correct number of features
    assert len(g0['features']) == 1
    stop_ids = feed.get_stops(trip_id=trip_id)['stop_id'].values
    assert len(g1['features']) == 1 + len(stop_ids)

# ----------------------------------
# Test methods about routes
# ----------------------------------
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

# --------------------------------------------
# Test methods about stops
# --------------------------------------------
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

# ----------------------------------
# Test methods about stop times
# ----------------------------------
def test_get_stop_times():
    feed = cairns.copy()
    date = cairns_date
    f = feed.get_stop_times(date)
    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have a reasonable shape
    assert f.shape[0] <= feed.stop_times.shape[0]
    # Should have correct columns
    assert set(f.columns) == set(feed.stop_times.columns)

def test_get_start_and_end_times():
    feed = cairns.copy()
    date = cairns_date
    st = feed.get_stop_times(date)
    times = feed.get_start_and_end_times(date)
    # Should be strings
    for t in times:
        assert isinstance(t, str)
        # Should lie in stop times
        assert t in st[['departure_time', 'arrival_time']].values.flatten()

    # Should get null times in some cases
    times = feed.get_start_and_end_times('19690711')
    for t in times:
        assert pd.isnull(t)
    feed.stop_times['departure_time'] = np.nan
    times = feed.get_start_and_end_times()
    assert pd.isnull(times[0])

@slow
def test_append_dist_to_stop_times():
    feed1 = cairns.copy()
    st1 = feed1.stop_times
    trip_stats = cairns_trip_stats
    feed2 = feed1.append_dist_to_stop_times(trip_stats)
    st2 = feed2.stop_times 

    # Check that colums of st2 equal the columns of st1 plus
    # a shape_dist_traveled column
    cols1 = list(st1.columns.values) + ['shape_dist_traveled']
    cols2 = list(st2.columns.values)
    assert set(cols1) == set(cols2)

    # Check that within each trip the shape_dist_traveled column 
    # is monotonically increasing
    for trip, group in st2.groupby('trip_id'):
        group = group.sort_values('stop_sequence')
        sdt = list(group['shape_dist_traveled'].values)
        assert sdt == sorted(sdt)

# ----------------------------------
# Test methods about shapes
# ----------------------------------
def test_build_geometry_by_shape():
    feed = cairns.copy()
    shape_ids = feed.shapes['shape_id'].unique()[:2]
    d0 = feed.build_geometry_by_shape()
    d1 = feed.build_geometry_by_shape(shape_ids=shape_ids)
    for d in [d0, d1]:
        # Should be a dictionary
        assert isinstance(d, dict)
        # The first key should be a valid shape ID
        assert list(d.keys())[0] in feed.shapes['shape_id'].values
        # The first value should be a Shapely linestring
        assert isinstance(list(d.values())[0], sg.LineString)
    # Lengths should be right
    assert len(d0) == feed.shapes['shape_id'].nunique()
    assert len(d1) == len(shape_ids)
    # Should be empty if feed.shapes is None
    feed2 = cairns_shapeless.copy()
    assert feed2.build_geometry_by_shape(feed2) == {}

def test_shapes_to_geojson():
    feed = cairns.copy()
    collection = feed.shapes_to_geojson()
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=False)
    for f in collection['features']:
        shape = f['properties']['shape_id']
        geom = sg.shape(f['geometry'])
        assert geom.equals(geometry_by_shape[shape])

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_get_shapes_intersecting_geometry():
    feed = cairns.copy()
    path = DATA_DIR/'cairns_square_stop_750070.geojson'
    polygon = sg.shape(json.load(path.open())['features'][0]['geometry'])
    pshapes = feed.get_shapes_intersecting_geometry(polygon)
    shape_ids = ['120N0005', '1200010', '1200001']
    assert set(pshapes['shape_id'].unique()) == set(shape_ids)

def test_append_dist_to_shapes():
    feed1 = cairns.copy()
    s1 = feed1.shapes
    feed2 = feed1.append_dist_to_shapes()
    s2 = feed2.shapes
    # Check that colums of st2 equal the columns of st1 plus
    # a shape_dist_traveled column
    cols1 = list(s1.columns.values) + ['shape_dist_traveled']
    cols2 = list(s2.columns.values)
    assert set(cols1) == set(cols2)

    # Check that within each trip the shape_dist_traveled column 
    # is monotonically increasing
    for name, group in s2.groupby('shape_id'):
        sdt = list(group['shape_dist_traveled'].values)
        assert sdt == sorted(sdt)

# ----------------------------------
# Test methods about cleaning
# ----------------------------------
def test_clean_ids():
    f1 = sample.copy()
    f1.routes.ix[0, 'route_id'] = '  ho   ho ho '
    f2 = f1.clean_ids()
    expect_rid = 'ho_ho_ho'
    f2.routes.ix[0, 'route_id'] == expect_rid

    f3 = f2.clean_ids()
    f3 == f2

def test_clean_route_short_names():
    f1  = sample.copy()
    
    # Should have no effect on a fine feed
    f2 = f1.clean_route_short_names()
    assert_series_equal(f2.routes['route_short_name'], 
      f1.routes['route_short_name'])
    
    # Make route short name duplicates
    f1.routes.loc[1:5, 'route_short_name'] = np.nan
    f1.routes.loc[6:, 'route_short_name'] = '  he llo  '
    f2 = f1.clean_route_short_names()
    # Should have unique route short names
    assert f2.routes['route_short_name'].nunique() == f2.routes.shape[0]
    # NaNs should be replaced by n/a and route IDs
    expect_rsns = ('n/a-' + sample.routes.ix[1:5]['route_id']).tolist()
    assert f2.routes.ix[1:5]['route_short_name'].values.tolist() == expect_rsns
    # Should have names without leading or trailing whitespace
    assert not f2.routes['route_short_name'].str.startswith(' ').any()
    assert not f2.routes['route_short_name'].str.endswith(' ').any()

def test_prune_dead_routes():
    # Should not change Cairns routes
    f1 = sample.copy()
    f2 = f1.prune_dead_routes()
    assert_frame_equal(f2.routes, f1.routes)

    # Create a dummy route which should be removed
    g = pd.DataFrame([[0 for c in f1.routes.columns]], 
      columns=f1.routes.columns)
    f3 = f1.copy()
    f3.routes = pd.concat([f3.routes, g])
    f4 = f3.prune_dead_routes()
    assert_frame_equal(f4.routes, f1.routes)

def test_aggregate_routes():
    feed1 = sample.copy()
    # Equalize all route short names
    feed1.routes['route_short_name'] = 'bingo'
    feed2 = feed1.aggregate_routes()

    # feed2 should have only one route ID
    assert feed2.routes.shape[0] == 1
    
    # Feeds should have same trip data frames excluding
    # route IDs
    feed1.trips['route_id'] = feed2.trips['route_id']
    assert almost_equal(feed1.trips, feed2.trips)

    # Feeds should have equal attributes excluding
    # routes and trips data frames
    feed2.routes = feed1.routes 
    feed2.trips = feed1.trips
    assert feed1 == feed2

def test_clean():
    f1 = sample.copy()
    rid = f1.routes.ix[0, 'route_id']
    f1.routes.ix[0, 'route_id'] = ' ' + rid + '   '
    f2 = f1.clean()
    assert f2.routes.ix[0, 'route_id'] == rid
    assert_frame_equal(f2.trips, sample.trips)

def test_drop_invalid_columns():
    f1 = sample.copy()
    f1.routes['bingo'] = 'bongo'
    f1.trips['wingo'] = 'wongo'
    f2 = f1.drop_invalid_columns()
    assert f2 == sample


# ----------------------------------
# Test methods about miscellany
# ----------------------------------
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
        names = cs.GTFS_TABLES_REQUIRED + cs.GTFS_TABLES_OPTIONAL
        assert feed1 == feed2
        try:
            out_path.unlink()
        except:
            shutil.rmtree(str(out_path))

    # Test that integer columns with NaNs get output properly.
    # To this end, put a NaN, 1.0, and 0.0 in the direction_id column 
    # of trips.txt, export it, and import the column as strings.
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
