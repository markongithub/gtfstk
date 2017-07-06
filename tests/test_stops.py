import pytest

import pandas as pd
from pandas.util.testing import assert_frame_equal
import shapely.geometry as sg

from .context import gtfstk, slow, HAS_GEOPANDAS, DATA_DIR, sample, cairns, cairns_dates, cairns_trip_stats
from gtfstk import *
if HAS_GEOPANDAS:
    from geopandas import GeoDataFrame


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

def test_get_stops():
    feed = cairns.copy()
    date = cairns_dates[0]
    trip_id = feed.trips['trip_id'].iat[0]
    route_id = feed.routes['route_id'].iat[0]
    frames = [
      get_stops(feed),
      get_stops(feed, date=date),
      get_stops(feed, trip_id=trip_id),
      get_stops(feed, route_id=route_id),
      get_stops(feed, date=date, trip_id=trip_id),
      get_stops(feed, date=date, route_id=route_id),
      get_stops(feed, date=date, trip_id=trip_id, route_id=route_id),
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
    d0 = build_geometry_by_stop(feed)
    d1 = build_geometry_by_stop(feed, stop_ids=stop_ids)
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
    dates = get_first_week(feed)
    stop_activity = compute_stop_activity(feed, dates)
    # Should be a data frame
    assert isinstance(stop_activity, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert stop_activity.shape[0] == feed.stops.shape[0]
    assert stop_activity.shape[1] == len(dates) + 1
    # Date columns should contain only zeros and ones
    assert set(stop_activity[dates].values.flatten()) == {0, 1}

def test_compute_stop_stats():
    dates = cairns_dates + ['20010101']
    feed = cairns.copy()
    for split_directions in [True, False]:
        f = compute_stop_stats(feed, dates,
          split_directions=split_directions)

        # Should be a data frame
        assert isinstance(f, pd.core.frame.DataFrame)

        # Should contain the correct stops
        get = set(f['stop_id'].values)
        g = get_stops(feed, dates[0])
        expect = set(g['stop_id'].values)
        assert get == expect

        # Should contain the correct columns
        expect_cols = {
          'date',
          'stop_id',
          'num_routes',
          'num_trips',
          'max_headway',
          'min_headway',
          'mean_headway',
          'start_time',
          'end_time',
          }
        if split_directions:
            expect_cols.add('direction_id')

        assert set(f.columns) == expect_cols

        # Should have correct dates
        f.date.tolist() == cairns_dates

        # Empty dates should yield empty DataFrame
        f = compute_stop_stats(feed, [],
          split_directions=split_directions)
        assert f.empty

        # No services should yield null stats
        feed1 = feed.copy()
        c = feed1.calendar
        c['monday'] = 0
        feed1.calendar = c
        f = compute_stop_stats(feed1, dates[0],
          split_directions=split_directions)
        assert set(f.columns) == expect_cols
        assert f.date.iat[0] == dates[0]
        assert pd.isnull(f.stop_id.iat[0])

@slow
def test_compute_stop_time_series():
    feed = cairns.copy()
    dates = cairns_dates + ['20010101']
    for split_directions in [True, False]:
        s = compute_stop_stats(feed, dates,
          split_directions=split_directions)
        ts = compute_stop_time_series(feed, dates, freq='1H',
          split_directions=split_directions)

        # Should be a data frame
        assert isinstance(ts, pd.core.frame.DataFrame)

        # Should have the correct shape
        assert ts.shape[0] == 24*2
        assert ts.shape[1] == s.shape[0]/2

        # Should have correct column names
        if split_directions:
            expect = ['indicator', 'stop_id', 'direction_id']
        else:
            expect = ['indicator', 'stop_id']
        assert ts.columns.names == expect

        # Each stop should have a correct total trip count
        if split_directions == False:
            sg = s.groupby('stop_id')
            for stop in s.stop_id.values:
                get = ts['num_trips'][stop].sum()
                expect = sg.get_group(stop)['num_trips'].sum()
                # Stop stats could have more num trips in case of
                # trips without departure times
                assert get <= expect

        # Empty dates should yield empty DataFrame
        ts = compute_stop_time_series(feed, [],
          split_directions=split_directions)
        assert ts.empty

def test_build_stop_timetable():
    feed = cairns.copy()
    stop_id = feed.stops['stop_id'].values[0]
    dates = cairns_dates + ['20010101']
    f = build_stop_timetable(feed, stop_id, dates)

    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)

    # Should have the correct columns
    expect_cols = set(feed.trips.columns)\
      | set(feed.stop_times.columns)\
      | set(['date'])
    assert set(f.columns) == expect_cols

    # Should only have feed dates
    assert f.date.unique().tolist() == cairns_dates

    # Empty check
    f = build_stop_timetable(feed, stop_id, [])
    assert f.empty

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_get_stops_in_polygon():
    feed = cairns.copy()
    with (DATA_DIR/'cairns_square_stop_750070.geojson').open() as src:
        polygon = sg.shape(json.load(src)['features'][0]['geometry'])
    pstops = get_stops_in_polygon(feed, polygon)
    stop_ids = ['750070']
    assert pstops['stop_id'].values == stop_ids

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_geometrize_stops():
    stops = cairns.stops.copy()
    geo_stops = geometrize_stops(stops, use_utm=True)
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
